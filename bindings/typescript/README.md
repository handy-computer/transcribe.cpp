# transcribe-cpp

TypeScript/Node.js bindings for [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp),
a C/C++ speech-to-text library built on ggml — Whisper, Parakeet, Moonshine,
Voxtral, and more, on CPU, Metal, CUDA, and Vulkan.

```sh
npm install transcribe-cpp
```

A matching prebuilt native package is selected automatically for your
platform (`@transcribe-cpp/<platform>`); there is nothing to compile and no
environment variables to set.

## Quickstart

```ts
import { TranscribeModel } from "transcribe-cpp";

const model = await TranscribeModel.load("whisper-tiny-Q5_K_M.gguf");

// 16 kHz mono float32 PCM (bring your own decoder).
const result = await model.transcribe(pcm, { timestamps: "segment" });

console.log(result.text);
console.log(result.language); // detected or requested
for (const seg of result.segments) {
  console.log(`[${seg.t0Ms}–${seg.t1Ms}ms] ${seg.text}`);
}

model.dispose();
```

### Streaming

```ts
const session = model.createSession();
const stream = await session.stream({ commitPolicy: "stable_prefix" });

for (const chunk of pcmChunks) {
  await stream.feed(chunk);
  const { committed, tentative } = stream.text; // committed is append-only/stable
  render(committed, tentative);
}
await stream.finalize();
stream.reset();
```

### Batch

```ts
const items = await session.runBatch([pcmA, pcmB]);
for (const item of items) {
  if (item.ok) console.log(item.result.text);
  else console.error(item.error.message); // per-utterance failure
}
```

### Cancellation

```ts
const ac = new AbortController();
setTimeout(() => ac.abort(), 5000);
try {
  await model.transcribe(pcm, { signal: ac.signal });
} catch (e) {
  if (e instanceof Aborted) console.log("partial:", e.partialResult?.text);
}
```

### Family extensions

Per-family options are a typed, discriminated union passed as `family`:

```ts
// Whisper is a run-slot extension:
await model.transcribe(pcm, { family: { kind: "whisper", initialPrompt: "…" } });

// Streaming families (moonshine, parakeet, voxtral) are stream-slot:
const stream = await session.stream({ family: { kind: "moonshine" } });

model.accepts({ kind: "whisper" }); // does this model take that extension?
```

### Resource management

`TranscribeModel`, `Session`, and `Stream` all implement `Symbol.dispose`, so
`using` works (TypeScript 5.2+ / Node 22+):

```ts
using model = await TranscribeModel.load("model.gguf");
using session = model.createSession();
// disposed automatically at block exit
```

Disposing a model disposes its sessions; disposing a stream resets it (releasing
the model lease). Disposal is idempotent and order-independent.

## Backend selection

```ts
import { getAvailableBackends, backendAvailable } from "transcribe-cpp";

getAvailableBackends(); // [{ kind: "metal", name: "MTL0", description: "…" }, …]
backendAvailable("cuda"); // boolean — never throws

const model = await TranscribeModel.load("model.gguf", { backend: "metal" });
```

`backend` defaults to `"auto"` (best accelerator, else CPU). A missing Vulkan
loader/driver degrades silently to CPU; an explicit backend that cannot be
satisfied raises a `BackendError`.

## CUDA (NVIDIA GPUs)

The default package ships CPU + the platform accelerator (Metal on macOS,
Vulkan on Linux/Windows). CUDA is an **opt-in** package for `linux-x64` /
`win32-x64`, installed alongside the default:

```sh
npm install transcribe-cpp @transcribe-cpp/linux-x64-cuda   # or win32-x64-cuda
```

The CUDA package is a strict superset (cuda + vulkan + cpu), and the loader
**prefers it automatically** when installed — no code change. Without an NVIDIA
driver it degrades to its own Vulkan/CPU modules, exactly like the default.

The **CUDA runtime is not bundled** (mirroring the Python `cu12` wheel, which
gets it from the `nvidia-*-cu12` pip wheels — Node has no equivalent). Supply it
one of two ways:

- a **system CUDA 12 toolkit** already on the loader path (`LD_LIBRARY_PATH` /
  the default DLL search path) — nothing else to do; or
- point **`TRANSCRIBE_CUDA_RUNTIME_DIR`** at a directory containing
  `libcudart.so.12` + `libcublas.so.12` + `libcublasLt.so.12` (Windows:
  `cudart64_12.dll` etc.). A `site-packages/nvidia` tree from
  `pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12` works directly.

If neither is present the CUDA module quietly fails to load and Vulkan/CPU keep
working — the same degradation contract as a missing driver.

## Thread-safety

Your JavaScript runs on one thread, but `run`/`feed`/`finalize` dispatch the
native compute to a **libuv worker thread** (via koffi's async calls), so the
event loop stays responsive while inference runs.

The C library allows **one compute in flight per model** — a `run`, a `runBatch`,
or an *active stream* — across all of its sessions. The binding enforces this:
every compute call serializes through an internal model-wide mutex, and an active
stream holds a model-wide lease for its whole lifetime. While a stream is active
(after `stream()`, before `finalize()`/`reset()`), a `run`/`runBatch`/`stream` on
any session of that model is refused with a `Busy` error rather than allowed to
race. So to parallelize, load one model per worker; to share a model, finalize or
reset the stream first. A single `Session` is single-use-at-a-time — don't call
`run`/`feed` on the same session concurrently.

Hand-offs are ordered, not racy: `finalize()`/`reset()`/`dispose()` release the
lease only after the native teardown runs on the shared queue, so the slot is
never freed early. Issue the teardown before the next `stream()`/`run()` and it
is correctly serialized — `stream.reset(); const next = await session.stream()`
works without awaiting the (void) `reset()`. The reverse order — beginning before
the teardown — is refused with `Busy`, by design.

Because the compute is genuinely on another thread, **do not touch a session
while a call against it is in flight** — it is single-threaded in the C library:

- Reading a stream's `text`/`state`/`revision`/`lastStatus`, or a session's
  `limits`/`wasAborted`, during an un-awaited `feed`/`finalize`/`run` **throws**.
- `reset()` and `dispose()` are safe to call any time: the native teardown is
  deferred behind any in-flight call, so it never frees a session mid-compute.
- Disposing a `Session` or `TranscribeModel` while a stream is still active
  releases the lease and invalidates the stream — its later calls throw rather
  than touch the freed handle.
- The **input PCM is borrowed, not copied**: `run`/`runBatch`/`feed` hand the
  buffer to native code that reads it on the worker thread, so do not mutate it
  (e.g. reuse a scratch/capture buffer) until the returned promise resolves.
  Pass a fresh buffer per call, or `await` before overwriting.

The normal pattern is safe — `await` first, then read:

```ts
await stream.feed(chunk);
const { committed, tentative } = stream.text; // ✓ feed has completed

const p = stream.feed(chunk);
stream.text;                                  // ✗ throws: read while feed in flight
await p;
```

Since JavaScript has no destructors, a stream holds the model's lease until you
`finalize()`, `reset()`, or let `using` dispose it — drop the reference without
one and the model stays `Busy` until you dispose the session or model. Always
finalize/reset (or use `using`).

Result strings are copied at the boundary; the objects you get back are fully
owned and outlive the model.

## Runtime support

The binding is one native artifact (koffi, an N-API addon) plus plain ESM, so any
runtime with N-API support runs the same package — there is no per-runtime build.

- **Node ≥ 22** — the primary, fully tested target.
- **Deno ≥ 2** — runs the package unmodified via `deno run` (use `--allow-ffi
  --allow-read --allow-env`, or `-A`). Note: `deno compile` does not bundle native
  addons, so the compiled-binary path is unsupported.
- **Bun** — not yet. Bun loads the addon and runs, but currently crashes in its
  own N-API finalizer (`napi_reference_unref`); this is an upstream Bun bug, fixed
  on Bun's side, not here.

## Building from source

The universal fallback when no prebuilt package exists for your platform (a new
arch, a custom backend, or a single self-contained library):

```sh
# from a transcribe.cpp checkout:
npm run build:native -- --source /path/to/transcribe.cpp
# add --self-contained to link ggml into one libtranscribe

# from a consumer project (the script ships in the package):
node node_modules/transcribe-cpp/scripts/build-from-source.mjs --source /path/to/transcribe.cpp
```

This drives CMake directly (the binding is FFI over a shared library, not an
N-API addon, so `cmake-js` does not apply) and installs into `prebuilds/<tuple>/`,
which the loader finds automatically. Requires `cmake` and a C/C++ toolchain.
You can also point `TRANSCRIBE_LIBRARY` at any `libtranscribe` you built.

## Packaging an app (Electron / Tauri)

The native code ships as a self-contained `@transcribe-cpp/<platform>` package
(the shared library plus its sibling ggml libs / backend modules, in one
directory). Because npm resolution is transitive, your app can locate that
directory at *build* time — there is nothing special to wire through.
`artifactDir()` returns it **without loading the library** (no dlopen), which is
exactly what a bundler/pack step needs:

```js
import { artifactDir } from "transcribe-cpp";
import * as fs from "node:fs";

const dir = artifactDir();
const lib =
  process.platform === "win32" ? "transcribe.dll"
  : process.platform === "darwin" ? "libtranscribe.dylib"
  : "libtranscribe.so";

// HARD-FAIL if the artifact is missing/empty: better to break the build than
// ship a silently broken installer that crashes at first transcribe() call.
if (!fs.existsSync(`${dir}/${lib}`)) {
  throw new Error(`native library not found in ${dir}; the platform package is not bundled`);
}
// copy `dir` into your app resources, or feed it to your bundler config…
```

The native files are real binaries, so they must not be packed into an asar
archive. With **electron-builder**, unpack the platform packages:

```jsonc
// package.json → "build"
{ "asarUnpack": ["node_modules/@transcribe-cpp/**"] }
```

For a **Tauri** app whose native bits come from the Rust crate instead, see that
crate's "Packaging a distributable" section — there the artifact dir is exposed
to your `build.rs` as `DEP_TRANSCRIBE_CPP_RUNTIME_DIR`.

## Waived requirements

- **Per-field ABI layout is not waived** (unlike Rust/Swift): TypeScript has no
  compiler reading the C headers, so — like the Python ctypes binding — the
  generated struct layout is verified at load against both the captured
  C-compiler layout and the live library (`transcribe_abi_struct_size/_align`).
- **Buffer model load** awaits `transcribe_model_load_buffer` in the C API; only
  file load is exposed today.

## License

MIT. Bundled native packages include third-party license texts (ggml, miniz,
and any bundled backend runtimes).
