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

Both `TranscribeModel` and `Session` implement `Symbol.dispose`, so `using`
works (TypeScript 5.2+ / Node 20+):

```ts
using model = await TranscribeModel.load("model.gguf");
using session = model.createSession();
// disposed automatically at block exit
```

Disposing a model disposes its sessions; disposal is idempotent and order-
independent.

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

## Thread-safety

JavaScript is single-threaded, but inference runs on a libuv worker thread (via
koffi's async calls), so the event loop stays responsive. Each `TranscribeModel`
serializes its own compute with an internal mutex — the C library allows only one
compute in flight per model, across all of its sessions — so concurrent `run`s on
sibling sessions are safe and run one after another. A single `Session` is
single-use-at-a-time: don't call `run`/`feed` on the same session concurrently.

Result strings are copied at the boundary; the objects you get back are fully
owned and outlive the model.

## Runtime support

The binding is one native artifact (koffi, an N-API addon) plus plain ESM, so any
runtime with N-API support runs the same package — there is no per-runtime build.

- **Node ≥ 20** — the primary, fully tested target.
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
# from a transcribe.cpp checkout, or pass --source <repo>
npm run build:native -- --source /path/to/transcribe.cpp
# add --self-contained to link ggml into one libtranscribe
```

This drives CMake directly (the binding is FFI over a shared library, not an
N-API addon, so `cmake-js` does not apply) and installs into `prebuilds/<tuple>/`,
which the loader finds automatically. Requires `cmake` and a C/C++ toolchain.
You can also point `TRANSCRIBE_LIBRARY` at any `libtranscribe` you built.

## Waived requirements

- **Per-field ABI layout is not waived** (unlike Rust/Swift): TypeScript has no
  compiler reading the C headers, so — like the Python ctypes binding — the
  generated struct layout is verified at load against both the captured
  C-compiler layout and the live library (`transcribe_abi_struct_size/_align`).
- **Buffer model load** awaits `transcribe_model_load_buffer` in the C API; only
  file load is exposed today.

## License

MIT. Bundled native packages include third-party license texts (ggml and any
bundled backend runtimes).
