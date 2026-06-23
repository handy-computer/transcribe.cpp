/**
 * transcribe.cpp — TypeScript/Node.js bindings.
 *
 * A koffi FFI binding over the shared native library: offline transcription and
 * batch, streaming with committed/tentative text, typed per-family extensions,
 * backend discovery, log routing, and cooperative cancellation.
 */

import { native, setLogHandler, type LogHandler, type Native } from "./native.js";
import { resolveLibrary } from "./loader.js";
import * as g from "./_generated.js";
import {
  Aborted,
  Busy,
  exceptionForStatus,
  InvalidArgument,
  ModelLoadError,
  NotImplementedByModel,
  OutputTruncated,
  TranscribeError,
  UnsupportedRequest,
} from "./errors.js";
import type {
  Backend,
  BackendInfo,
  BatchItem,
  Capabilities,
  CommitPolicy,
  DeviceType,
  ExtSlot,
  FamilyExtension,
  Feature,
  KvType,
  ModelOptions,
  PcmLike,
  Segment,
  SessionLimits,
  SessionOptions,
  StreamOptions,
  StreamState,
  StreamText,
  StreamUpdate,
  Timings,
  TimestampKind,
  Token,
  TranscribeOptions,
  TranscriptionResult,
  Word,
} from "./types.js";

export * from "./types.js";
export * from "./errors.js";
export { setLogHandler, type LogHandler };

// ---- enum maps -------------------------------------------------------------

const BACKENDS: Record<Backend, number> = {
  auto: g.TRANSCRIBE_BACKEND_AUTO,
  cpu: g.TRANSCRIBE_BACKEND_CPU,
  metal: g.TRANSCRIBE_BACKEND_METAL,
  vulkan: g.TRANSCRIBE_BACKEND_VULKAN,
  cpu_accel: g.TRANSCRIBE_BACKEND_CPU_ACCEL,
  cuda: g.TRANSCRIBE_BACKEND_CUDA,
};
const KV_TYPES: Record<KvType, number> = {
  auto: g.TRANSCRIBE_KV_TYPE_AUTO,
  f32: g.TRANSCRIBE_KV_TYPE_F32,
  f16: g.TRANSCRIBE_KV_TYPE_F16,
};
const TASKS = { transcribe: g.TRANSCRIBE_TASK_TRANSCRIBE, translate: g.TRANSCRIBE_TASK_TRANSLATE };
const TIMESTAMPS: Record<TimestampKind, number> = {
  none: g.TRANSCRIBE_TIMESTAMPS_NONE,
  auto: g.TRANSCRIBE_TIMESTAMPS_AUTO,
  segment: g.TRANSCRIBE_TIMESTAMPS_SEGMENT,
  word: g.TRANSCRIBE_TIMESTAMPS_WORD,
  token: g.TRANSCRIBE_TIMESTAMPS_TOKEN,
};
const TIMESTAMP_NAMES: Record<number, TimestampKind> = Object.fromEntries(
  Object.entries(TIMESTAMPS).map(([k, v]) => [v, k as TimestampKind]),
);
const FEATURES: Record<Feature, number> = {
  initial_prompt: g.TRANSCRIBE_FEATURE_INITIAL_PROMPT,
  temperature_fallback: g.TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK,
  long_form: g.TRANSCRIBE_FEATURE_LONG_FORM,
  cancellation: g.TRANSCRIBE_FEATURE_CANCELLATION,
  pnc: g.TRANSCRIBE_FEATURE_PNC,
  itn: g.TRANSCRIBE_FEATURE_ITN,
};

// ---- helpers ---------------------------------------------------------------

function lookup<T extends string>(map: Record<string, number>, key: T, what: string): number {
  const v = map[key];
  if (v === undefined) {
    throw new TranscribeError(
      `invalid ${what} ${JSON.stringify(key)}; expected one of ${Object.keys(map).join(", ")}`,
    );
  }
  return v;
}

function check(n: Native, status: number, context: string): void {
  if (status === g.TRANSCRIBE_OK) return;
  throw exceptionForStatus(status, n.F.statusString(status), context);
}

function busyError(op: string): Busy {
  return new Busy(
    `cannot ${op}: a stream is active on this model. The C library allows one ` +
      `run / batch / active stream in flight per model across all sessions — ` +
      `finalize or reset the stream first, or use a separate model.`,
  );
}

// Native frees are queued behind the model lock (below), which resolves on a
// promise microtask. `process.exit()` terminates WITHOUT draining the microtask
// queue, so those deferred frees would never run — leaving native resources
// (model weights, session compute buffers) alive when the library's C++ static
// destructors run at process exit. On macOS >= 15 that aborts the process:
// ggml-metal's device teardown asserts every Metal buffer was freed first
// (GGML_ASSERT([rsets->data count] == 0), the residency-set collection). A
// process 'exit' handler runs synchronously and BEFORE those static destructors,
// so we flush any still-pending frees there. Every pending free registers itself
// here; it deletes itself once it has run (idempotent via the `done` guard), so
// the flush never double-frees and a clean (natural) exit finds nothing to do.
const PENDING_FREES = new Set<() => void>();
// Models the user loaded but has not disposed. On exit we force-dispose them so
// their native frees get registered in PENDING_FREES and flushed below —
// otherwise an undisposed model leaks its (residency-set-backed) weight buffers
// and aborts at the macOS-15 ggml-metal teardown just like a disposed-but-
// process.exit()'d one.
const LIVE_MODELS = new Set<TranscribeModel>();
let exitHookInstalled = false;

function ensureExitHook(): void {
  if (exitHookInstalled) return;
  exitHookInstalled = true;
  process.once("exit", () => {
    // 1) Force-dispose any still-live model. dispose() runs synchronously and
    //    enqueues its session + model frees into PENDING_FREES (its deferred
    //    microtask won't run at exit, but the registration does). Snapshot
    //    first: dispose() removes the model from LIVE_MODELS as it goes.
    for (const m of [...LIVE_MODELS]) {
      try {
        m.dispose();
      } catch {
        /* best-effort teardown */
      }
    }
    // 2) Flush every pending native free synchronously. Insertion order frees
    //    sessions before their model, honoring the C contract that a model
    //    outlives its sessions. The lease release (`after`) is intentionally
    //    skipped — the process is exiting, so only the native frees matter.
    for (const free of PENDING_FREES) {
      try {
        free();
      } catch {
        /* best-effort teardown; a native free cannot meaningfully fail */
      }
    }
    PENDING_FREES.clear();
  });
}

/**
 * Run a native free/reset behind the model lock, after any in-flight (and
 * queued) worker call drains, so it never overlaps a compute on a libuv worker.
 * `after` (e.g. the compute-lease release) runs in the SAME queued slot, after
 * the native teardown — so an op queued earlier still observes the lease held /
 * the stream un-reset until the native state has actually been torn down.
 * Best-effort: errors are swallowed (a free cannot meaningfully fail) and the
 * floating promise is marked intentional with `void`.
 *
 * The native free is also registered in PENDING_FREES so a `process.exit()` that
 * skips the microtask queue still tears the resource down at exit (see above).
 */
function deferFree(lock: Mutex, fn: () => void, after?: () => void): void {
  let done = false;
  const free = (): void => {
    if (done) return;
    done = true;
    PENDING_FREES.delete(free);
    fn();
  };
  PENDING_FREES.add(free);
  ensureExitHook();
  void lock.run(async () => {
    try {
      free();
    } catch {
      /* native free is infallible in practice; nothing to recover */
    }
    after?.();
  });
}

function callAsync<T = number>(fn: any, ...args: any[]): Promise<T> {
  return new Promise((resolve, reject) =>
    fn.async(...args, (err: Error | null, res: T) => (err ? reject(err) : resolve(res))),
  );
}

// Coerce caller PCM to a Float32Array. A Float32Array is returned AS-IS (no
// copy): the buffer is borrowed across the async native call, which reads it on
// a worker thread, so callers must not mutate it until the promise resolves
// (documented on run/runBatch/feed). Other inputs already produce a fresh array.
function toFloat32(pcm: PcmLike): Float32Array {
  let out: Float32Array;
  if (pcm instanceof Float32Array) out = pcm;
  else if (Array.isArray(pcm)) out = Float32Array.from(pcm);
  else if (pcm instanceof ArrayBuffer) out = new Float32Array(pcm);
  else if (ArrayBuffer.isView(pcm)) {
    const b = pcm as Buffer;
    if (b.byteLength % 4 !== 0) {
      throw new TranscribeError("PCM byte length must be a multiple of 4 (float32)");
    }
    out = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  } else {
    throw new TranscribeError("PCM must be a Float32Array, number[], ArrayBuffer, or Buffer");
  }
  if (out.length === 0) throw new TranscribeError("PCM is empty");
  return out;
}

// koffi decodes int64 struct fields as bigint. We surface them as number for
// ergonomics; the fields this is used on (millisecond timestamps, kv byte caps)
// stay well under Number.MAX_SAFE_INTEGER, so the narrowing is lossless in
// practice. Revisit if a field can exceed 2^53.
function num(v: number | bigint): number {
  return typeof v === "bigint" ? Number(v) : v;
}

/**
 * A FIFO async mutex plus the model-wide compute lease. One per Model.
 *
 * `run()` serializes every native compute call (run, batch, stream
 * feed/finalize) so they never overlap in time, even when koffi `.async()` puts
 * work on libuv workers. But the C contract is stronger: at most one
 * run/batch/*active stream* may be in flight per model across all sessions, and
 * an active stream occupies the model for its whole lifetime — not just during
 * a feed. `streamActive` is that lease: it is claimed at stream begin and held
 * until finalize/reset, and run/batch/stream refuse (Busy) while it is set.
 */
class Mutex {
  #tail: Promise<void> = Promise.resolve();
  /** True while an active stream holds the model's single compute slot. */
  streamActive = false;
  run<T>(fn: () => Promise<T>): Promise<T> {
    const prev = this.#tail;
    let release!: () => void;
    this.#tail = new Promise<void>((r) => (release = r));
    return prev.then(fn).finally(release);
  }
}

// ---- module-level introspection -------------------------------------------
//
// Note: these (like every public entry point) trigger the lazy native bootstrap
// on first call — they dlopen the library, verify the ABI, and init backends.
// There is no way to probe the binding without loading the native library, with
// one exception: `headerHash` below is the compile-time PUBLIC_HEADER_HASH and
// is the value the binding *expects*, not one read from the loaded library.

export function version(): { version: string; commit: string; headerHash: string } {
  const n = native();
  return { version: n.F.version(), commit: n.F.versionCommit(), headerHash: g.PUBLIC_HEADER_HASH };
}

export function libraryPath(): string {
  return native().libraryPath;
}

/**
 * The directory holding the native library and its sibling ggml libs / backend
 * modules — resolved WITHOUT loading the library (no dlopen, no ABI check, no
 * backend init), unlike every other entry point here.
 *
 * This is the build/packaging hook: call it from a bundler config or pack step
 * to copy the native artifacts into your installer (Electron `asarUnpack`, Tauri
 * `resources`, etc.). Resolution follows the same order as the runtime loader
 * (`TRANSCRIBE_LIBRARY` → the `@transcribe-cpp/<platform>` package → a local
 * prebuild → the dev tree) and throws if nothing is found. For runtime use,
 * `libraryPath()` returns the resolved library path of the *loaded* binding.
 */
export function artifactDir(): string {
  return resolveLibrary().artifactDir;
}

const DEVICE_TYPE_NAMES: Record<number, DeviceType> = {
  [g.TRANSCRIBE_DEVICE_TYPE_CPU]: "cpu",
  [g.TRANSCRIBE_DEVICE_TYPE_GPU]: "gpu",
  [g.TRANSCRIBE_DEVICE_TYPE_IGPU]: "igpu",
  [g.TRANSCRIBE_DEVICE_TYPE_ACCEL]: "accel",
};

// Decode a koffi-filled transcribe_backend_device struct into a BackendInfo.
// memory_* are uint64 (bigint from koffi) but stay well under 2^53 for any
// real device, so num() narrows them losslessly.
function deviceFromRaw(dev: any, index: number | null = null): BackendInfo {
  return {
    name: dev.name ?? "",
    description: dev.description ?? "",
    kind: dev.kind ?? "",
    deviceType: DEVICE_TYPE_NAMES[dev.device_type] ?? "unknown",
    deviceId: dev.device_id ?? null,
    memoryTotal: num(dev.memory_total),
    memoryFree: num(dev.memory_free),
    index,
  };
}

export function getAvailableBackends(): BackendInfo[] {
  const n = native();
  const count = n.F.backendDeviceCount();
  const out: BackendInfo[] = [];
  for (let i = 0; i < count; i++) {
    const dev: any = {};
    n.F.backendDeviceInit(dev);
    check(n, n.F.getBackendDevice(i, dev), `reading backend device ${i}`);
    out.push(deviceFromRaw(dev, i));
  }
  return out;
}

export function backendAvailable(backend: Backend): boolean {
  const n = native();
  return n.F.backendAvailable(lookup(BACKENDS, backend, "backend"));
}

// ---- result materialization ------------------------------------------------

/** Result accessors, parameterized so single and batch reads share the code. */
interface Accessors {
  nSegments(): number;
  getSegment(j: number, out: any): number;
  nWords(): number;
  getWord(j: number, out: any): number;
  nTokens(): number;
  getToken(j: number, out: any): number;
  getTimings(out: any): number;
  fullText(): string;
  detectedLanguage(): string;
  returnedTimestampKind(): number;
}

function singleAccessors(n: Native, h: any): Accessors {
  const F = n.F;
  return {
    nSegments: () => F.nSegments(h),
    getSegment: (j, o) => F.getSegment(h, j, o),
    nWords: () => F.nWords(h),
    getWord: (j, o) => F.getWord(h, j, o),
    nTokens: () => F.nTokens(h),
    getToken: (j, o) => F.getToken(h, j, o),
    getTimings: (o) => F.getTimings(h, o),
    fullText: () => F.fullText(h),
    detectedLanguage: () => F.detectedLanguage(h),
    returnedTimestampKind: () => F.returnedTimestampKind(h),
  };
}

function batchAccessors(n: Native, h: any, i: number): Accessors {
  const F = n.F;
  return {
    nSegments: () => F.batchNSegments(h, i),
    getSegment: (j, o) => F.batchGetSegment(h, i, j, o),
    nWords: () => F.batchNWords(h, i),
    getWord: (j, o) => F.batchGetWord(h, i, j, o),
    nTokens: () => F.batchNTokens(h, i),
    getToken: (j, o) => F.batchGetToken(h, i, j, o),
    getTimings: (o) => F.batchGetTimings(h, i, o),
    fullText: () => F.batchFullText(h, i),
    detectedLanguage: () => F.batchDetectedLanguage(h, i),
    returnedTimestampKind: () => F.batchReturnedTimestampKind(h, i),
  };
}

function materialize(n: Native, acc: Accessors): Omit<TranscriptionResult, "aborted" | "truncated"> {
  const F = n.F;

  const segments: Segment[] = [];
  for (let i = 0, c = acc.nSegments(); i < c; i++) {
    const s: any = {};
    F.segmentInit(s);
    check(n, acc.getSegment(i, s), `reading segment ${i}`);
    segments.push({
      text: s.text ?? "",
      t0Ms: num(s.t0_ms),
      t1Ms: num(s.t1_ms),
      firstWord: s.first_word,
      nWords: s.n_words,
      firstToken: s.first_token,
      nTokens: s.n_tokens,
    });
  }

  const words: Word[] = [];
  for (let i = 0, c = acc.nWords(); i < c; i++) {
    const w: any = {};
    F.wordInit(w);
    check(n, acc.getWord(i, w), `reading word ${i}`);
    words.push({
      text: w.text ?? "",
      t0Ms: num(w.t0_ms),
      t1Ms: num(w.t1_ms),
      segIndex: w.seg_index,
      firstToken: w.first_token,
      nTokens: w.n_tokens,
    });
  }

  const tokens: Token[] = [];
  for (let i = 0, c = acc.nTokens(); i < c; i++) {
    const t: any = {};
    F.tokenInit(t);
    check(n, acc.getToken(i, t), `reading token ${i}`);
    tokens.push({
      text: t.text ?? "",
      id: t.id,
      p: t.p,
      t0Ms: num(t.t0_ms),
      t1Ms: num(t.t1_ms),
      segIndex: t.seg_index,
      wordIndex: t.word_index,
    });
  }

  const tm: any = {};
  F.timingsInit(tm);
  check(n, acc.getTimings(tm), "reading timings");
  const timings: Timings = {
    loadMs: tm.load_ms,
    melMs: tm.mel_ms,
    encodeMs: tm.encode_ms,
    decodeMs: tm.decode_ms,
  };

  return {
    text: acc.fullText() ?? "",
    language: acc.detectedLanguage() ?? "",
    timestampKind: TIMESTAMP_NAMES[acc.returnedTimestampKind()] ?? "none",
    segments,
    words,
    tokens,
    timings,
  };
}

// ---- family extensions -----------------------------------------------------

const COMMIT_POLICIES: Record<CommitPolicy, number> = {
  auto: g.TRANSCRIBE_STREAM_COMMIT_AUTO,
  on_finalize: g.TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE,
  stable_prefix: g.TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX,
};
const STREAM_STATES: Record<number, StreamState> = {
  [g.TRANSCRIBE_STREAM_IDLE]: "idle",
  [g.TRANSCRIBE_STREAM_ACTIVE]: "active",
  [g.TRANSCRIBE_STREAM_FINISHED]: "finished",
  [g.TRANSCRIBE_STREAM_FAILED]: "failed",
};
const SLOT: Record<ExtSlot, number> = {
  run: g.TRANSCRIBE_EXT_SLOT_RUN,
  stream: g.TRANSCRIBE_EXT_SLOT_STREAM,
};

interface FamilyReg {
  slot: ExtSlot;
  kind: number;
  type: string;
  init: string;
  map: (o: any) => Record<string, unknown>;
}

const FAMILY: Record<string, FamilyReg> = {
  whisper: {
    slot: "run",
    kind: g.TRANSCRIBE_EXT_KIND_WHISPER_RUN,
    type: "transcribe_whisper_run_ext",
    init: "whisperRunExtInit",
    map: (o) => ({
      initial_prompt: o.initialPrompt,
      condition_on_prev_tokens: o.conditionOnPrevTokens,
      temperature: o.temperature,
      temperature_inc: o.temperatureInc,
      compression_ratio_thold: o.compressionRatioThold,
      logprob_thold: o.logprobThold,
      no_speech_thold: o.noSpeechThold,
      max_prev_context_tokens: o.maxPrevContextTokens,
      seed: o.seed,
      max_initial_timestamp: o.maxInitialTimestamp,
    }),
  },
  moonshine: {
    slot: "stream",
    kind: g.TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM,
    type: "transcribe_moonshine_streaming_stream_ext",
    init: "moonshineStreamingStreamExtInit",
    map: (o) => ({ min_decode_interval_ms: o.minDecodeIntervalMs }),
  },
  parakeet: {
    slot: "stream",
    kind: g.TRANSCRIBE_EXT_KIND_PARAKEET_STREAM,
    type: "transcribe_parakeet_stream_ext",
    init: "parakeetStreamExtInit",
    map: (o) => ({ att_context_right: o.attContextRight }),
  },
  parakeet_buffered: {
    slot: "stream",
    kind: g.TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM,
    type: "transcribe_parakeet_buffered_stream_ext",
    init: "parakeetBufferedStreamExtInit",
    map: (o) => ({ left_ms: o.leftMs, chunk_ms: o.chunkMs, right_ms: o.rightMs }),
  },
  voxtral: {
    slot: "stream",
    kind: g.TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM,
    type: "transcribe_voxtral_realtime_stream_ext",
    init: "voxtralRealtimeStreamExtInit",
    map: (o) => ({
      num_delay_tokens: o.numDelayTokens,
      min_decode_interval_ms: o.minDecodeIntervalMs,
    }),
  },
};

/**
 * Build a native ext-struct buffer for a family extension and return the koffi
 * pointer to assign to `params.family`. Validates the slot and that the model
 * accepts the kind. The returned buffer must be kept alive (held via the params
 * object) until the native call returns.
 */
function buildFamily(n: Native, modelHandle: any, family: FamilyExtension, slot: ExtSlot): any {
  const reg = FAMILY[family.kind];
  if (!reg) throw new InvalidArgument(`unknown family extension kind ${JSON.stringify(family.kind)}`);
  if (reg.slot !== slot) {
    throw new InvalidArgument(
      `family "${family.kind}" is a ${reg.slot} extension, not valid for a ${slot} call`,
    );
  }
  if (!n.F.modelAcceptsExtKind(modelHandle, SLOT[reg.slot], reg.kind)) {
    throw new UnsupportedRequest(`this model does not accept the "${family.kind}" ${reg.slot} extension`);
  }
  const ext: any = {};
  n.F[reg.init](ext); // defaults + struct_size + kind
  for (const [k, v] of Object.entries(reg.map(family))) {
    if (v !== undefined) ext[k] = v;
  }
  const buf = n.koffi.alloc(n.T[reg.type], 1);
  n.koffi.encode(buf, n.T[reg.type], ext);
  return buf;
}

function toStreamUpdate(u: any): StreamUpdate {
  return {
    resultChanged: u.result_changed,
    isFinal: u.is_final,
    revision: u.revision,
    inputReceivedMs: num(u.input_received_ms),
    audioCommittedMs: num(u.audio_committed_ms),
    bufferedMs: num(u.buffered_ms),
    committedChanged: u.committed_changed,
    tentativeChanged: u.tentative_changed,
  };
}

/**
 * Module-private teardown control for each Stream, keyed by the instance. These
 * ops mutate the stream's `#active`/lease state, so they must NOT be reachable
 * from user code: calling the lease release on a live stream would clear the
 * model-wide lease and let a sibling run/stream overlap it — the exact race the
 * lease prevents. Public `@internal` is only a type hint (the method still exists
 * at runtime), so the control surface lives here instead, reached only by the
 * owning Session within this module. The closures capture the Stream; WeakMap
 * ephemeron semantics keep that from pinning it in memory.
 */
const STREAM_TEARDOWN = new WeakMap<Stream, { deactivate(): void; invalidate(): void; releaseLease(): void }>();

interface SessionControl {
  enterCompute(kind: string): void;
  leaveCompute(kind: string): void;
  isCurrentStream(stream: Stream): boolean;
  replaceCurrentStream(stream: Stream): void;
  clearCurrentStream(stream: Stream): void;
}

const SESSION_CONTROL = new WeakMap<Session, SessionControl>();

// ---- Session ---------------------------------------------------------------

export class Session {
  #n: Native;
  #h: any;
  #model: TranscribeModel; // keep the model alive while this session lives
  #lock: Mutex; // shared with the model; serializes compute model-wide
  #untrack: (self: Session) => void; // drop self from the model's session set
  #inFlight: string | null = null; // set while a native call runs on a worker
  #activeStream: Stream | null = null; // current wrapper for the session's native stream slot
  #disposed = false;

  /** @internal */
  constructor(
    n: Native,
    model: TranscribeModel,
    handle: any,
    lock: Mutex,
    untrack: (self: Session) => void,
  ) {
    this.#n = n;
    this.#model = model;
    this.#h = handle;
    this.#lock = lock;
    this.#untrack = untrack;
    SESSION_CONTROL.set(this, {
      enterCompute: (kind) => {
        this.#inFlight = kind;
      },
      leaveCompute: (kind) => {
        if (this.#inFlight === kind) this.#inFlight = null;
      },
      isCurrentStream: (stream) => this.#activeStream === stream,
      replaceCurrentStream: (stream) => {
        if (this.#activeStream && this.#activeStream !== stream) {
          STREAM_TEARDOWN.get(this.#activeStream)?.invalidate();
        }
        this.#activeStream = stream;
      },
      clearCurrentStream: (stream) => {
        if (this.#activeStream === stream) this.#activeStream = null;
      },
    });
  }

  /** @internal */
  get handle(): any {
    if (this.#disposed) throw new TranscribeError("session has been disposed");
    return this.#h;
  }

  /** Reads touch the session; forbidden while a worker call is in flight. */
  #assertNotComputing(what: string): void {
    if (this.#inFlight) {
      throw new TranscribeError(`cannot read session ${what} while ${this.#inFlight} is in flight; await it first`);
    }
  }

  get limits(): SessionLimits {
    this.#assertNotComputing("limits");
    const n = this.#n;
    const l: any = {};
    n.F.sessionLimitsInit(l);
    check(n, n.F.sessionGetLimits(this.handle, l), "reading session limits");
    return {
      effectiveNCtx: l.effective_n_ctx,
      effectiveMaxAudioMs: num(l.effective_max_audio_ms),
      maxKvBytes: num(l.max_kv_bytes),
    };
  }

  /**
   * Transcribe one clip. The input PCM is borrowed, not copied: native code
   * reads it on a worker thread while this runs, so do not mutate the buffer
   * (e.g. reuse a scratch array) until the returned promise resolves.
   */
  async run(pcm: PcmLike, opts: TranscribeOptions = {}): Promise<TranscriptionResult> {
    const n = this.#n;
    const F = n.F;
    const h = this.handle;
    const samples = toFloat32(pcm);

    const p = this.#buildRunParams(opts);

    return this.#lock.run(async () => {
      if (this.#disposed) throw new TranscribeError("session has been disposed");
      if (this.#lock.streamActive) throw busyError("run");
      const cancel = this.#installAbort(opts.signal);
      let status: number;
      this.#inFlight = "run()";
      try {
        status = await callAsync<number>(F.run, h, samples, samples.length, p);
      } finally {
        this.#inFlight = null;
        cancel?.();
      }

      if (status === g.TRANSCRIBE_ERR_ABORTED || status === g.TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
        const partial: TranscriptionResult = {
          ...materialize(n, singleAccessors(n, h)),
          aborted: F.wasAborted(h),
          truncated: F.wasTruncated(h),
        };
        const exc =
          status === g.TRANSCRIBE_ERR_ABORTED
            ? new Aborted(`run aborted`, status)
            : new OutputTruncated(`run output truncated`, status);
        exc.partialResult = partial;
        throw exc;
      }
      check(n, status, "transcribe_run");

      return { ...materialize(n, singleAccessors(n, h)), aborted: F.wasAborted(h), truncated: F.wasTruncated(h) };
    });
  }

  #buildRunParams(opts: TranscribeOptions): any {
    const n = this.#n;
    const p: any = {};
    n.F.runParamsInit(p);
    p.task = lookup(TASKS, opts.task ?? "transcribe", "task");
    p.timestamps = lookup(TIMESTAMPS, opts.timestamps ?? "none", "timestamps");
    if (opts.language !== undefined) p.language = opts.language;
    if (opts.targetLanguage !== undefined) p.target_language = opts.targetLanguage;
    if (opts.keepSpecialTags !== undefined) p.keep_special_tags = opts.keepSpecialTags;
    if (opts.specKDrafts !== undefined) p.spec_k_drafts = opts.specKDrafts;
    if (opts.family) p.family = buildFamily(n, this.#model.handle, opts.family, "run");
    return p;
  }

  /**
   * Offline batch transcription; each item carries its own success/failure.
   * Inputs are borrowed, not copied: native code reads them on a worker thread
   * while this runs, so do not mutate them until the returned promise resolves.
   */
  async runBatch(pcms: PcmLike[], opts: TranscribeOptions = {}): Promise<BatchItem[]> {
    const n = this.#n;
    const F = n.F;
    const h = this.handle;
    if (pcms.length === 0) throw new InvalidArgument("runBatch requires at least one input");
    const arrays = pcms.map(toFloat32);
    const counts = Int32Array.from(arrays, (a) => a.length);
    const p = this.#buildRunParams(opts);

    return this.#lock.run(async () => {
      if (this.#disposed) throw new TranscribeError("session has been disposed");
      if (this.#lock.streamActive) throw busyError("runBatch");
      const cancel = this.#installAbort(opts.signal);
      let status: number;
      this.#inFlight = "runBatch()";
      try {
        status = await callAsync<number>(F.runBatch, h, arrays, counts, arrays.length, p);
      } finally {
        this.#inFlight = null;
        cancel?.();
      }
      // A batch returns OK even with per-utterance failures; only a top-level
      // error (or a whole-batch abort) is fatal here.
      if (status !== g.TRANSCRIBE_OK && status !== g.TRANSCRIBE_ERR_ABORTED) {
        check(n, status, "transcribe_run_batch");
      }
      const out: BatchItem[] = [];
      for (let i = 0, c = F.batchNResults(h); i < c; i++) {
        const st = F.batchStatus(h, i);
        if (st === g.TRANSCRIBE_OK) {
          out.push({
            ok: true,
            result: { ...materialize(n, batchAccessors(n, h, i)), aborted: false, truncated: false },
          });
        } else {
          const error = exceptionForStatus(st, F.statusString(st), `utterance ${i}`);
          error.utteranceIndex = i;
          if (st === g.TRANSCRIBE_ERR_ABORTED || st === g.TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
            error.partialResult = {
              ...materialize(n, batchAccessors(n, h, i)),
              aborted: st === g.TRANSCRIBE_ERR_ABORTED,
              truncated: st === g.TRANSCRIBE_ERR_OUTPUT_TRUNCATED,
            };
          }
          out.push({ ok: false, error });
        }
      }
      return out;
    });
  }

  /** Begin a streaming session. The returned Stream owns the begin params. */
  async stream(opts: StreamOptions = {}): Promise<Stream> {
    const n = this.#n;
    const F = n.F;
    const h = this.handle;
    const rp = this.#buildRunParams({
      task: opts.task,
      language: opts.language,
      targetLanguage: opts.targetLanguage,
      timestamps: opts.timestamps,
      keepSpecialTags: opts.keepSpecialTags,
      specKDrafts: -1,
    });
    const sp: any = {};
    F.streamParamsInit(sp);
    sp.commit_policy = lookup(COMMIT_POLICIES, opts.commitPolicy ?? "auto", "commitPolicy");
    if (opts.stablePrefixAgreementN !== undefined) {
      sp.stable_prefix_agreement_n = opts.stablePrefixAgreementN;
    }
    if (opts.family) sp.family = buildFamily(n, this.#model.handle, opts.family, "stream");

    return this.#lock.run(async () => {
      // Recheck inside the lock: dispose() may have run after we captured `h`
      // but before this queued body — don't begin a stream on a dead session.
      if (this.#disposed) throw new TranscribeError("session has been disposed");
      if (this.#lock.streamActive) throw busyError("begin a stream");
      check(n, F.streamBegin(h, rp, sp), "transcribe_stream_begin");
      this.#lock.streamActive = true; // claim the lease for the whole stream lifetime
      // The Stream holds the Session (not a raw handle) so its calls fail fast
      // once the session is disposed, and so dispose() can find and invalidate it.
      const stream = new Stream(n, this, this.#lock, [rp, sp]); // pin params until reset
      const control = SESSION_CONTROL.get(this);
      if (!control) throw new TranscribeError("session control is missing");
      control.replaceCurrentStream(stream);
      return stream;
    });
  }

  /**
   * Wire an AbortSignal to a native abort callback for one run. The callback is
   * installed on *this session's* handle, but install/run/uninstall is only safe
   * because every caller holds the model-wide #lock for the whole run — the lock,
   * not the per-session handle, is what guarantees no run overlaps the window
   * between setAbortCallback(cb) and setAbortCallback(null). A future change that
   * relaxes the lock must keep this install/uninstall paired within one run.
   */
  #installAbort(signal?: AbortSignal): (() => void) | null {
    if (!signal) return null;
    const n = this.#n;
    const flag = { aborted: signal.aborted };
    const onAbort = () => {
      flag.aborted = true;
    };
    signal.addEventListener("abort", onAbort, { once: true });
    const cbPtr = n.koffi.register(() => flag.aborted, n.koffi.pointer(n.abortProto));
    n.F.setAbortCallback(this.handle, cbPtr, null);
    return () => {
      n.F.setAbortCallback(this.handle, null, null);
      n.koffi.unregister(cbPtr);
      signal.removeEventListener("abort", onAbort);
    };
  }

  get wasAborted(): boolean {
    this.#assertNotComputing("wasAborted");
    return this.#n.F.wasAborted(this.handle);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#untrack(this); // stop the model from holding a dead session
    // Deactivate any live stream NOW (its reset() no-ops, reads throw via the
    // disposed handle), but release its model lease only inside the deferred
    // teardown, after sessionFree — so a stream/run queued ahead of this can't
    // claim the slot before the native session is actually gone.
    const stream = this.#activeStream;
    this.#activeStream = null;
    const teardown = stream ? STREAM_TEARDOWN.get(stream) : undefined;
    teardown?.deactivate();
    // Free behind the model lock: a run/feed worker may still hold this handle,
    // and sessionFree mid-call is a use-after-free. Queuing on the FIFO lock
    // runs the free after any in-flight (and queued) compute drains. The JS-side
    // guard is already synchronous (#disposed/handle), so use-after-dispose
    // still throws immediately; only the native free + lease release are deferred.
    const n = this.#n;
    const h = this.#h;
    this.#h = null;
    deferFree(this.#lock, () => n.F.sessionFree(h), () => teardown?.releaseLease());
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

// ---- Stream ----------------------------------------------------------------

export class Stream {
  #n: Native;
  #session: Session; // owns the native handle; throws once disposed (no stale handle)
  #lock: Mutex;
  #sessionControl: SessionControl;
  #keepalive: unknown[] | null;
  #active = true;
  #stale = false; // true once the session has begun a newer native stream
  #inFlight = false; // true while a feed/finalize native call runs on a worker
  #holdsLease = true; // born holding the model's compute lease (claimed at begin)

  /** @internal */
  constructor(n: Native, session: Session, lock: Mutex, keepalive: unknown[]) {
    this.#n = n;
    this.#session = session;
    this.#lock = lock;
    const sessionControl = SESSION_CONTROL.get(session);
    if (!sessionControl) throw new TranscribeError("session control is missing");
    this.#sessionControl = sessionControl;
    this.#keepalive = keepalive;
    // Expose the teardown surface ONLY to the owning Session, via the
    // module-private map — never as public methods (see STREAM_TEARDOWN).
    STREAM_TEARDOWN.set(this, {
      // Synchronous deactivation on Session dispose: a later reset() no-ops and
      // reads fail fast via the disposed handle, so nothing touches freed memory.
      deactivate: () => {
        this.#active = false;
        this.#keepalive = null;
      },
      invalidate: () => {
        this.#stale = true;
        this.#keepalive = null;
      },
      // Release the lease (guarded); the Session calls this inside its deferred
      // teardown, after sessionFree, so the slot frees in FIFO order.
      releaseLease: () => this.#releaseLease(),
    });
  }

  /**
   * Release the model's compute lease, once, if this stream still holds it. The
   * `#holdsLease` guard ensures a reset() after finalize() (or a double reset)
   * never clears a lease that another session has since claimed.
   */
  #releaseLease(): void {
    if (this.#holdsLease) {
      this.#holdsLease = false;
      this.#lock.streamActive = false;
    }
  }

  #assertCurrent(what: string): void {
    if (this.#stale) {
      throw new TranscribeError(`cannot ${what}: stream is no longer current for this session`);
    }
  }

  /**
   * Feed one chunk of PCM; returns the update snapshot. The chunk is borrowed,
   * not copied: native code reads it on a worker thread while the feed runs, so
   * do not mutate it (e.g. reuse a capture buffer) until the promise resolves.
   */
  async feed(pcm: PcmLike): Promise<StreamUpdate> {
    const n = this.#n;
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("feed");
    if (!this.#active) throw new TranscribeError("stream has been reset");
    const samples = toFloat32(pcm);
    return this.#lock.run(async () => {
      const u: any = {};
      n.F.streamUpdateInit(u);
      // The native feed runs on a libuv worker. While it is in flight the
      // session must not be touched from the main thread — the C session API
      // is single-threaded (transcribe.h), and stream_get_text hands back
      // pointers the feed may free/realloc. Flag it so the read getters fail
      // fast instead of racing into a use-after-free.
      this.#inFlight = true;
      this.#sessionControl.enterCompute("feed()/finalize()");
      try {
        const status = await callAsync<number>(n.F.streamFeed, h, samples, samples.length, u);
        if (status !== g.TRANSCRIBE_OK) {
          // Native feed failures leave the stream in FAILED, which is no longer
          // an active stream in the C API. Keep the wrapper readable for
          // state/lastStatus, but free the model-wide compute slot.
          this.#releaseLease();
        }
        check(n, status, "transcribe_stream_feed");
      } finally {
        this.#inFlight = false;
        this.#sessionControl.leaveCompute("feed()/finalize()");
      }
      return toStreamUpdate(u);
    });
  }

  /** Flush remaining audio and commit the final text. */
  async finalize(): Promise<StreamUpdate> {
    const n = this.#n;
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("finalize stream");
    if (!this.#active) throw new TranscribeError("stream has been reset");
    return this.#lock.run(async () => {
      const u: any = {};
      n.F.streamUpdateInit(u);
      this.#inFlight = true; // see feed(): worker-thread compute, no concurrent reads
      this.#sessionControl.enterCompute("feed()/finalize()");
      try {
        check(
          n,
          await callAsync<number>(n.F.streamFinalize, h, u),
          "transcribe_stream_finalize",
        );
      } finally {
        this.#inFlight = false;
        this.#sessionControl.leaveCompute("feed()/finalize()");
        // Finalize ends the active stream (FINISHED on success, FAILED on
        // error), so the model is free again — release the lease either way.
        this.#releaseLease();
      }
      return toStreamUpdate(u);
    });
  }

  /**
   * Reads borrow session-owned snapshot memory, so they are forbidden while a
   * feed()/finalize() is computing on a worker thread (concurrent use of a
   * single session is undefined per transcribe.h). The natural pattern —
   * `await stream.feed(chunk)` then read — is unaffected; this only rejects a
   * read issued against an un-awaited feed.
   */
  #assertNotFeeding(what: string): void {
    if (this.#inFlight) {
      throw new TranscribeError(
        `cannot read stream ${what} while a feed()/finalize() is in flight; await it first`,
      );
    }
  }

  /** Current text snapshot (copied at the boundary). */
  get text(): StreamText {
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("read stream text");
    this.#assertNotFeeding("text");
    const n = this.#n;
    const t: any = {};
    n.F.streamTextInit(t);
    check(n, n.F.streamGetText(h, t), "transcribe_stream_get_text");
    return {
      full: t.full_text ?? "",
      committed: t.committed_text ?? "",
      tentative: t.tentative_text ?? "",
    };
  }

  get state(): StreamState {
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("read stream state");
    if (!this.#active) return "idle"; // reset() returns to idle; native reset may still be queued
    this.#assertNotFeeding("state");
    return STREAM_STATES[this.#n.F.streamGetState(h)] ?? "idle";
  }

  get revision(): number {
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("read stream revision");
    this.#assertNotFeeding("revision");
    return this.#n.F.streamRevision(h);
  }

  /**
   * The stream's recorded terminal failure, or `null` while it is healthy. Set
   * after a feed()/finalize() transitions the stream to `"failed"`; reset by a
   * new stream. Inspect it when `state === "failed"`.
   */
  get lastStatus(): TranscribeError | null {
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("read stream lastStatus");
    this.#assertNotFeeding("lastStatus");
    const n = this.#n;
    const status = n.F.streamLastStatus(h);
    if (status === g.TRANSCRIBE_OK) return null;
    return exceptionForStatus(status, n.F.statusString(status), "stream");
  }

  /** End the stream and return the session to idle. Idempotent. */
  reset(): void {
    if (this.#stale) return; // a newer stream owns the session slot now
    if (!this.#active) return; // already reset, finalized-and-reset, or invalidated
    this.#active = false;
    this.#keepalive = null;
    // Defer BOTH the native reset and the lease release into one queued slot,
    // behind any in-flight feed/finalize. Releasing the lease only after the
    // native streamReset means a stream/run queued before this reset still sees
    // the lease held — it cannot begin and overlap the not-yet-reset stream.
    // #active was true, so the session is still alive (dispose() deactivates
    // streams first), and the handle is valid here.
    const n = this.#n;
    const h = this.#session.handle;
    deferFree(
      this.#lock,
      () => {
        if (this.#sessionControl.isCurrentStream(this)) {
          n.F.streamReset(h);
          this.#sessionControl.clearCurrentStream(this);
        }
      },
      () => this.#releaseLease(),
    );
  }

  [Symbol.dispose](): void {
    this.reset();
  }
}

// ---- Model -----------------------------------------------------------------

export class TranscribeModel {
  #n: Native;
  #h: any;
  #disposed = false;
  #sessions = new Set<Session>();
  #lock = new Mutex(); // serializes compute across all sessions of this model

  private constructor(n: Native, handle: any) {
    this.#n = n;
    this.#h = handle;
    // Track the model so an undisposed-then-process.exit() still frees its
    // native (Metal) buffers at exit (see ensureExitHook). Loading alone
    // allocates weight buffers, so install the hook here, not just on dispose.
    LIVE_MODELS.add(this);
    ensureExitHook();
  }

  static async load(path: string, opts: ModelOptions = {}): Promise<TranscribeModel> {
    const n = native();
    const p: any = {};
    n.F.modelLoadParamsInit(p);
    if (opts.backend) p.backend = lookup(BACKENDS, opts.backend, "backend");
    if (opts.gpuDevice !== undefined) p.gpu_device = opts.gpuDevice;

    const out: any[] = [null];
    const st = await callAsync<number>(n.F.modelLoadFile, path, p, out);
    check(n, st, `loading model ${path}`);
    if (!out[0]) throw new ModelLoadError(`model load returned a null handle for ${path}`);
    return new TranscribeModel(n, out[0]);
  }

  /** @internal */
  get handle(): any {
    if (this.#disposed) throw new TranscribeError("model has been disposed");
    return this.#h;
  }

  createSession(opts: SessionOptions = {}): Session {
    const n = this.#n;
    const p: any = {};
    n.F.sessionParamsInit(p);
    if (opts.nThreads !== undefined) p.n_threads = opts.nThreads;
    if (opts.kvType) p.kv_type = lookup(KV_TYPES, opts.kvType, "kvType");
    if (opts.nCtx !== undefined) p.n_ctx = opts.nCtx;

    const out: any[] = [null];
    check(n, n.F.sessionInit(this.handle, p, out), "opening session");
    if (!out[0]) throw new TranscribeError("session init returned a null handle");
    const session = new Session(n, this, out[0], this.#lock, (s) => this.#sessions.delete(s));
    this.#sessions.add(session);
    return session;
  }

  /** Convenience: one session, one run, disposed after. */
  async transcribe(pcm: PcmLike, opts: TranscribeOptions = {}): Promise<TranscriptionResult> {
    const session = this.createSession();
    try {
      return await session.run(pcm, opts);
    } finally {
      session.dispose(); // untracks itself from #sessions
    }
  }

  get capabilities(): Capabilities {
    const n = this.#n;
    const c: any = {};
    n.F.capabilitiesInit(c);
    check(n, n.F.modelGetCapabilities(this.handle, c), "reading capabilities");
    let languages: string[] = [];
    try {
      if (c.languages && c.n_languages > 0) {
        languages = n.koffi.decode(c.languages, "char *", c.n_languages);
      }
    } catch {
      languages = [];
    }
    return {
      nativeSampleRate: c.native_sample_rate,
      languages,
      maxTimestampKind: TIMESTAMP_NAMES[c.max_timestamp_kind] ?? "none",
      supportsLanguageDetect: c.supports_language_detect,
      supportsTranslate: c.supports_translate,
      supportsStreaming: c.supports_streaming,
      supportsSpecDecode: c.supports_spec_decode,
      maxAudioMs: num(c.max_audio_ms),
    };
  }

  supports(feature: Feature): boolean {
    return this.#n.F.modelSupports(this.handle, lookup(FEATURES, feature, "feature"));
  }

  /** Whether this model accepts the given family extension on its slot. */
  accepts(family: FamilyExtension): boolean {
    const reg = FAMILY[family.kind];
    if (!reg) return false;
    return this.#n.F.modelAcceptsExtKind(this.handle, SLOT[reg.slot], reg.kind);
  }

  /** Tokenize plain UTF-8 text into the model's vocabulary (no special tokens). */
  tokenize(text: string): Int32Array {
    const F = this.#n.F;
    const INT_MIN = -2147483648;
    let cap = Math.max(16, text.length + 16);
    for (let attempt = 0; attempt < 4; attempt++) {
      const buf = new Int32Array(cap);
      const r = F.tokenize(this.handle, text, buf, cap);
      if (r === INT_MIN) {
        throw new NotImplementedByModel("this model's tokenizer does not support encode");
      }
      if (r >= 0) return buf.subarray(0, r);
      cap = -r; // buffer too small; -r is the count needed
    }
    throw new TranscribeError("tokenize did not converge");
  }

  get arch(): string {
    return this.#n.F.modelArch(this.handle) ?? "";
  }
  get variant(): string {
    return this.#n.F.modelVariant(this.handle) ?? "";
  }
  get backend(): string {
    return this.#n.F.modelBackend(this.handle) ?? "";
  }

  /** The compute device this model is running on. `memoryFree` is a live
   *  snapshot, so read this again to poll how much device memory is left
   *  after the model loaded. */
  get device(): BackendInfo {
    const dev: any = {};
    this.#n.F.backendDeviceInit(dev);
    check(this.#n, this.#n.F.modelGetDevice(this.handle, dev), "reading model device");
    return deviceFromRaw(dev);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    LIVE_MODELS.delete(this); // its frees are now queued in PENDING_FREES
    // Snapshot: each dispose() untracks itself from #sessions as we go and
    // queues its native free on the model lock. Queue modelFree last, so the
    // FIFO lock runs it after every session free (the C contract: a model may
    // only be freed once all derived sessions are). All deferred behind any
    // in-flight worker call, so nothing is freed out from under a compute.
    for (const s of [...this.#sessions]) s.dispose();
    this.#sessions.clear();
    const n = this.#n;
    const h = this.#h;
    this.#h = null;
    deferFree(this.#lock, () => n.F.modelFree(h));
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

/** One-shot: load (or reuse) a model, transcribe, return the result. */
export async function transcribe(
  model: TranscribeModel | string,
  pcm: PcmLike,
  opts: TranscribeOptions & ModelOptions = {},
): Promise<TranscriptionResult> {
  if (model instanceof TranscribeModel) return model.transcribe(pcm, opts);
  const m = await TranscribeModel.load(model, opts);
  try {
    return await m.transcribe(pcm, opts);
  } finally {
    m.dispose();
  }
}
