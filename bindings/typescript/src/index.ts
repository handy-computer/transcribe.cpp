/**
 * transcribe.cpp — TypeScript/Node.js bindings.
 *
 * A koffi FFI binding over the shared native library: offline transcription and
 * batch, streaming with committed/tentative text, typed per-family extensions,
 * backend discovery, log routing, and cooperative cancellation.
 */

import {
  backendState,
  callAsync,
  initialize,
  loadLibrary,
  nativeAsync,
  nativeReady,
  setLogHandler,
  type BackendState,
  type LogHandler,
  type Native,
} from "./native.js";
import { resolveLibrary } from "./loader.js";
import * as g from "./_generated.js";
import {
  Aborted,
  Busy,
  exceptionForStatus,
  InvalidArgument,
  ModelLoadError,
  NotImplementedByModel,
  OutputRepetition,
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
  Diarize,
  DiarizeInfo,
  DiarizeOptions,
  DiarizeSessionOptions,
  ExtSlot,
  FamilyExtension,
  Feature,
  Itn,
  KvType,
  LangIdCandidate,
  LangIdInfo,
  LangIdOptions,
  LangIdResult,
  LangIdSessionOptions,
  ModelOptions,
  PcmLike,
  Pnc,
  Role,
  Segment,
  SpeakerSegment,
  SessionLimits,
  SessionOptions,
  StreamOptions,
  StreamState,
  StreamText,
  StreamUpdate,
  Timings,
  TimestampKind,
  Token,
  Transcript,
  TranscribeOptions,
  TranscriptionResult,
  Word,
} from "./types.js";

export * from "./types.js";
export * from "./errors.js";
export { backendState, initialize, setLogHandler, type BackendState, type LogHandler };

// ---- enum maps -------------------------------------------------------------

const BACKENDS: Record<Backend, number> = {
  auto: g.TRANSCRIBE_BACKEND_AUTO,
  cpu: g.TRANSCRIBE_BACKEND_CPU,
  metal: g.TRANSCRIBE_BACKEND_METAL,
  vulkan: g.TRANSCRIBE_BACKEND_VULKAN,
  cpu_accel: g.TRANSCRIBE_BACKEND_CPU_ACCEL,
  cuda: g.TRANSCRIBE_BACKEND_CUDA,
  rocm: g.TRANSCRIBE_BACKEND_ROCM,
};
const KV_TYPES: Record<KvType, number> = {
  auto: g.TRANSCRIBE_KV_TYPE_AUTO,
  f32: g.TRANSCRIBE_KV_TYPE_F32,
  f16: g.TRANSCRIBE_KV_TYPE_F16,
};
const TASKS = {
  transcribe: g.TRANSCRIBE_TASK_TRANSCRIBE,
  translate: g.TRANSCRIBE_TASK_TRANSLATE,
  instruct: g.TRANSCRIBE_TASK_INSTRUCT,
};
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
const PNC: Record<Pnc, number> = {
  default: g.TRANSCRIBE_PNC_MODE_DEFAULT,
  off: g.TRANSCRIBE_PNC_MODE_OFF,
  on: g.TRANSCRIBE_PNC_MODE_ON,
};
const ITN: Record<Itn, number> = {
  default: g.TRANSCRIBE_ITN_MODE_DEFAULT,
  off: g.TRANSCRIBE_ITN_MODE_OFF,
  on: g.TRANSCRIBE_ITN_MODE_ON,
};
const DIARIZE: Record<Diarize, number> = {
  default: g.TRANSCRIBE_DIARIZE_MODE_DEFAULT,
  off: g.TRANSCRIBE_DIARIZE_MODE_OFF,
  on: g.TRANSCRIBE_DIARIZE_MODE_ON,
};
const FEATURES: Record<Feature, number> = {
  initial_prompt: g.TRANSCRIBE_FEATURE_INITIAL_PROMPT,
  temperature_fallback: g.TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK,
  long_form: g.TRANSCRIBE_FEATURE_LONG_FORM,
  cancellation: g.TRANSCRIBE_FEATURE_CANCELLATION,
  pnc: g.TRANSCRIBE_FEATURE_PNC,
  itn: g.TRANSCRIBE_FEATURE_ITN,
  diarization: g.TRANSCRIBE_FEATURE_DIARIZATION,
  vocabulary: g.TRANSCRIBE_FEATURE_VOCABULARY,
  context_prompt: g.TRANSCRIBE_FEATURE_CONTEXT_PROMPT,
  instruct: g.TRANSCRIBE_FEATURE_INSTRUCT,
  transcript_prefix: g.TRANSCRIBE_FEATURE_TRANSCRIPT_PREFIX,
};
const ROLES: Record<Role, number> = {
  asr: g.TRANSCRIBE_ROLE_ASR,
  diarize: g.TRANSCRIBE_ROLE_DIARIZE,
  langid: g.TRANSCRIBE_ROLE_LANGID,
};

// ---- helpers ---------------------------------------------------------------

function lookup<T extends string>(
  map: Record<string, number>,
  key: T,
  what: string,
): number {
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
      throw new TranscribeError(
        "PCM byte length must be a multiple of 4 (float32)",
      );
    }
    out = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
  } else {
    throw new TranscribeError(
      "PCM must be a Float32Array, number[], ArrayBuffer, or Buffer",
    );
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
// Native bootstrap is two-phase (see native.ts). `version()` and
// `libraryPath()` only load the library and never initialize a backend.
// `headerHash` is the compile-time PUBLIC_HEADER_HASH: the value the binding
// *expects*, not one read from the loaded library.

export function version(): {
  version: string;
  commit: string;
  headerHash: string;
} {
  const n = loadLibrary();
  return {
    version: n.F.version(),
    commit: n.F.versionCommit(),
    headerHash: g.PUBLIC_HEADER_HASH,
  };
}

export function libraryPath(): string {
  return loadLibrary().libraryPath;
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

const DEVICE_HANDLES = new WeakMap<BackendInfo, unknown>();

const DEVICE_TYPE_NAMES: Record<number, DeviceType> = {
  [g.TRANSCRIBE_DEVICE_TYPE_CPU]: "cpu",
  [g.TRANSCRIBE_DEVICE_TYPE_GPU]: "gpu",
  [g.TRANSCRIBE_DEVICE_TYPE_IGPU]: "igpu",
  [g.TRANSCRIBE_DEVICE_TYPE_ACCEL]: "accel",
};

// Decode a koffi-filled transcribe_device_info struct into a BackendInfo.
// memory_* are uint64 (bigint from koffi) but stay well under 2^53 for any
// real device, so num() narrows them losslessly.
function deviceFromRaw(
  dev: any,
  handle: unknown,
  index: number | null = null,
): BackendInfo {
  const info: BackendInfo = {
    name: dev.name ?? "",
    description: dev.description ?? "",
    kind: dev.kind ?? "",
    deviceType: DEVICE_TYPE_NAMES[dev.device_type] ?? "unknown",
    deviceId: dev.device_id ?? null,
    memoryTotal: num(dev.memory_total),
    memoryFree: num(dev.memory_free),
    index,
  };
  DEVICE_HANDLES.set(info, handle);
  return info;
}

/**
 * Every registered compute device. Initializes backends on the calling thread
 * if needed; throws `BackendInitializing` while {@link initialize} runs.
 */
export function getAvailableBackends(): BackendInfo[] {
  const n = nativeReady();
  const count = n.F.deviceCount();
  const out: BackendInfo[] = [];
  for (let i = 0; i < count; i++) {
    const handle = n.F.deviceGet(i);
    if (!handle) continue;
    const dev: any = {};
    n.F.deviceInfoInit(dev);
    check(n, n.F.deviceGetInfo(handle, dev), `reading backend device ${i}`);
    out.push(deviceFromRaw(dev, handle, i));
  }
  return out;
}

/** {@link getAvailableBackends} without blocking the event loop. */
export async function getAvailableBackendsAsync(): Promise<BackendInfo[]> {
  const n = await nativeAsync();
  const count = n.F.deviceCount();
  const out: BackendInfo[] = [];
  for (let i = 0; i < count; i++) {
    const handle = n.F.deviceGet(i);
    if (!handle) continue;
    const dev: any = {};
    n.F.deviceInfoInit(dev);
    const st = await callAsync<number>(n.F.deviceGetInfo, handle, dev);
    check(n, st, `reading backend device ${i}`);
    out.push(deviceFromRaw(dev, handle, i));
  }
  return out;
}

/**
 * Whether some registered device can satisfy `backend`. Same initialization
 * behavior as {@link getAvailableBackends}.
 */
export function backendAvailable(backend: Backend): boolean {
  const raw = lookup(BACKENDS, backend, "backend");
  return nativeReady().F.backendAvailable(raw);
}

/** {@link backendAvailable} without blocking the event loop. */
export async function backendAvailableAsync(backend: Backend): Promise<boolean> {
  const raw = lookup(BACKENDS, backend, "backend");
  const n = await nativeAsync();
  return n.F.backendAvailable(raw);
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
  nSpeakerSegments(): number;
  getSpeakerSegment(j: number, out: any): number;
  getTimings(out: any): number;
  fullText(): string;
  rawText(): string;
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
    nSpeakerSegments: () => F.nSpeakerSegments(h),
    getSpeakerSegment: (j, o) => F.getSpeakerSegment(h, j, o),
    getTimings: (o) => F.getTimings(h, o),
    fullText: () => F.fullText(h),
    rawText: () => F.rawText(h),
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
    nSpeakerSegments: () => F.batchNSpeakerSegments(h, i),
    getSpeakerSegment: (j, o) => F.batchGetSpeakerSegment(h, i, j, o),
    getTimings: (o) => F.batchGetTimings(h, i, o),
    fullText: () => F.batchFullText(h, i),
    rawText: () => F.batchRawText(h, i),
    detectedLanguage: () => F.batchDetectedLanguage(h, i),
    returnedTimestampKind: () => F.batchReturnedTimestampKind(h, i),
  };
}

function materialize(n: Native, acc: Accessors): Transcript {
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
      speakerId: s.speaker_id,
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

  return {
    text: acc.fullText() ?? "",
    rawText: acc.rawText() ?? "",
    language: acc.detectedLanguage() ?? "",
    timestampKind: TIMESTAMP_NAMES[acc.returnedTimestampKind()] ?? "none",
    segments,
    speakerSegments: readSpeakerSegments(n, acc.nSpeakerSegments(), acc.getSpeakerSegment),
    words,
    tokens,
    timings: readTimings(n, acc.getTimings),
  };
}

// Shared by transcripts and the DIARIZE role's results.
function readSpeakerSegments(
  n: Native,
  count: number,
  get: (i: number, out: any) => number,
): SpeakerSegment[] {
  const out: SpeakerSegment[] = [];
  for (let i = 0; i < count; i++) {
    const s: any = {};
    n.F.speakerSegmentInit(s);
    check(n, get(i, s), `reading speaker segment ${i}`);
    out.push({
      t0Ms: num(s.t0_ms),
      t1Ms: num(s.t1_ms),
      speakerId: s.speaker_id,
      p: s.p,
    });
  }
  return out;
}

function readTimings(n: Native, get: (out: any) => number): Timings {
  const tm: any = {};
  n.F.timingsInit(tm);
  check(n, get(tm), "reading timings");
  return {
    loadMs: tm.load_ms,
    melMs: tm.mel_ms,
    encodeMs: tm.encode_ms,
    decodeMs: tm.decode_ms,
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
  diarize_run: g.TRANSCRIBE_EXT_SLOT_DIARIZE_RUN,
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
      initial_prompt:
        o.initialPrompt === undefined ? undefined : cstr(o.initialPrompt, "initialPrompt"),
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
    map: (o) => ({
      left_ms: o.leftMs,
      chunk_ms: o.chunkMs,
      right_ms: o.rightMs,
    }),
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
  sortformer_diarize: {
    slot: "diarize_run",
    kind: g.TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE,
    type: "transcribe_sortformer_diarize_ext",
    init: "sortformerDiarizeExtInit",
    map: (o) => ({
      preset:
        o.preset === undefined ? undefined : lookup(SORTFORMER_PRESET, o.preset, "sortformer preset"),
    }),
  },
};

const SORTFORMER_PRESET: Record<string, number> = {
  default: g.TRANSCRIBE_SORTFORMER_PRESET_DEFAULT,
  very_high_latency: g.TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY,
  high_latency: g.TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY,
  low_latency: g.TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY,
};

/**
 * Build a native ext-struct buffer for a family extension and return the koffi
 * pointer to assign to `params.family`. Validates the slot and that the model
 * accepts the kind. The returned buffer must be kept alive (held via the params
 * object) until the native call returns, then freed by freeRunParams.
 */
function buildFamily(
  n: Native,
  modelHandle: any,
  family: FamilyExtension,
  slot: ExtSlot,
): any {
  const reg = FAMILY[family.kind];
  if (!reg)
    throw new InvalidArgument(
      `unknown family extension kind ${JSON.stringify(family.kind)}`,
    );
  if (reg.slot !== slot) {
    throw new InvalidArgument(
      `family "${family.kind}" is a ${reg.slot} extension, not valid for a ${slot} call`,
    );
  }
  if (!n.F.modelAcceptsExtKind(modelHandle, SLOT[reg.slot], reg.kind)) {
    throw new UnsupportedRequest(
      `this model does not accept the "${family.kind}" ${reg.slot} extension`,
    );
  }
  const ext: any = {};
  n.F[reg.init](ext); // defaults + struct_size + kind
  for (const [k, v] of Object.entries(reg.map(family))) {
    if (v !== undefined) ext[k] = v;
  }
  const buf = n.koffi.alloc(n.T[reg.type], 1);
  try {
    n.koffi.encode(buf, n.T[reg.type], ext);
  } catch (e) {
    n.koffi.free(buf);
    throw e;
  }
  return buf;
}

/** A string option as passed to C, which would silently cut it at a NUL. */
function cstr(value: string, name: string): string {
  if (value.includes("\0"))
    throw new InvalidArgument(`${name} contains a NUL character`);
  return value;
}

/**
 * Free what #buildRunParams / buildFamily allocated into a params struct; call
 * only once the native call has returned (never while a worker reads it).
 */
function freeRunParams(n: Native, p: any): void {
  if (p.vocabulary) {
    n.koffi.free(p.vocabulary);
    p.vocabulary = null;
    p.n_vocabulary = 0;
  }
  if (p.family) {
    n.koffi.free(p.family);
    p.family = null;
  }
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
const STREAM_TEARDOWN = new WeakMap<
  Stream,
  { deactivate(): void; invalidate(): void; releaseLease(): void }
>();

/**
 * Makes one worker call `fn(...args)` with `signal`'s abort callback installed
 * and the session marked in flight as `kind` (result reads fail fast) until it
 * settles. Only valid inside an exclusive() body.
 */
type ComputeCall = (
  kind: string,
  signal: AbortSignal | undefined,
  fn: any,
  ...args: any[]
) => Promise<number>;

interface SessionControl {
  core: SessionCore;
  isCurrentStream(stream: Stream): boolean;
  replaceCurrentStream(stream: Stream): void;
  clearCurrentStream(stream: Stream): void;
}

const SESSION_CONTROL = new WeakMap<Session, SessionControl>();

/**
 * One native session handle under the model's compute rules, shared by Session
 * and DiarizeSession. Each holds it in a private field, so none of this is
 * reachable from user code. `setAbort` is the role's set_abort_callback.
 */
class SessionCore {
  #n: Native;
  #h: any;
  #lock: Mutex; // shared with the model; serializes compute model-wide
  #setAbort: any;
  #inFlight: string | null = null; // set while a native call runs on a worker
  #disposed = false;

  constructor(n: Native, handle: any, lock: Mutex, setAbort: any) {
    this.#n = n;
    this.#h = handle;
    this.#lock = lock;
    this.#setAbort = setAbort;
  }

  get handle(): any {
    if (this.#disposed) throw new TranscribeError("session has been disposed");
    return this.#h;
  }

  get disposed(): boolean {
    return this.#disposed;
  }

  /** Reads touch the session; forbidden while a worker call is in flight. */
  assertNotComputing(what: string): void {
    if (this.#inFlight) {
      throw new TranscribeError(
        `cannot read ${what} while ${this.#inFlight} is in flight; await it first`,
      );
    }
  }

  /**
   * Run `body` as this session's one native compute on the model-wide FIFO
   * lock (it copies results out before release). Refuses a disposed session,
   * then an active stream (Busy naming `busyOp`); null `busyOp` = that stream.
   * Not `async`: refusals and `body`'s promise are returned as-is (no extra ticks).
   */
  exclusive<T>(
    busyOp: string | null,
    body: (call: ComputeCall) => Promise<T>,
  ): Promise<T> {
    return this.#lock.run(() => {
      if (busyOp !== null && this.#disposed)
        return Promise.reject(new TranscribeError("session has been disposed"));
      if (busyOp !== null && this.#lock.streamActive)
        return Promise.reject(busyError(busyOp));
      return body(this.#call);
    });
  }

  #call: ComputeCall = (kind, signal, fn, ...args) => {
    const cancel = this.#installAbort(signal);
    this.#inFlight = kind;
    return callAsync<number>(fn, ...args).finally(() => {
      if (this.#inFlight === kind) this.#inFlight = null;
      cancel?.();
    });
  };

  /**
   * Wire an AbortSignal to a native abort callback for one run. The callback is
   * installed on *this session's* handle, but install/run/uninstall is only safe
   * because every caller holds the model-wide #lock for the whole run — the lock,
   * not the per-session handle, is what guarantees no run overlaps the window
   * between setAbort(cb) and setAbort(null). A future change that relaxes the
   * lock must keep this install/uninstall paired within one run.
   */
  #installAbort(signal?: AbortSignal): (() => void) | null {
    if (!signal) return null;
    const n = this.#n;
    const flag = { aborted: signal.aborted };
    const onAbort = () => {
      flag.aborted = true;
    };
    signal.addEventListener("abort", onAbort, { once: true });
    const cbPtr = n.koffi.register(
      () => flag.aborted,
      n.koffi.pointer(n.abortProto),
    );
    const h = this.#h; // still valid at cleanup: a mid-run dispose frees behind the lock
    this.#setAbort(h, cbPtr, null);
    return () => {
      this.#setAbort(h, null, null);
      n.koffi.unregister(cbPtr);
      signal.removeEventListener("abort", onAbort);
    };
  }

  /**
   * Mark disposed now (use-after-dispose throws immediately) and free the
   * native handle behind the model lock: a worker may still hold it, and a
   * free mid-call is a use-after-free. Queuing on the FIFO lock runs the free
   * (then `after`) once any in-flight and queued compute drains.
   */
  dispose(free: (h: any) => void, after?: () => void): void {
    this.#disposed = true;
    const h = this.#h;
    this.#h = null;
    deferFree(this.#lock, () => free(h), after);
  }
}

// ---- Session ---------------------------------------------------------------

export class Session {
  #n: Native;
  #core: SessionCore;
  #model: TranscribeModel; // keep the model alive while this session lives
  #lock: Mutex; // shared with the model; serializes compute model-wide
  #untrack: (self: Session) => void; // drop self from the model's session set
  #activeStream: Stream | null = null; // current wrapper for the session's native stream slot

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
    this.#core = new SessionCore(n, handle, lock, n.F.setAbortCallback);
    this.#lock = lock;
    this.#untrack = untrack;
    SESSION_CONTROL.set(this, {
      core: this.#core,
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
    return this.#core.handle;
  }

  get limits(): SessionLimits {
    this.#core.assertNotComputing("session limits");
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
  async run(
    pcm: PcmLike,
    opts: TranscribeOptions = {},
  ): Promise<TranscriptionResult> {
    const n = this.#n;
    const F = n.F;
    const h = this.handle;
    const samples = toFloat32(pcm);

    const p = this.#buildRunParams(opts);

    return this.#core.exclusive("run", async (call) => {
      const status = await call("run()", opts.signal, F.run, h, samples, samples.length, p);

      if (
        status === g.TRANSCRIBE_ERR_ABORTED ||
        status === g.TRANSCRIBE_ERR_OUTPUT_TRUNCATED ||
        status === g.TRANSCRIBE_ERR_OUTPUT_REPETITION
      ) {
        const partial: TranscriptionResult = {
          ...materialize(n, singleAccessors(n, h)),
          aborted: F.wasAborted(h),
          truncated: F.wasTruncated(h),
        };
        const exc =
          status === g.TRANSCRIBE_ERR_ABORTED
            ? new Aborted(`run aborted`, status)
            : status === g.TRANSCRIBE_ERR_OUTPUT_REPETITION
              ? new OutputRepetition(`run stopped: output began repeating`, status)
              : new OutputTruncated(`run output truncated`, status);
        exc.partialResult = partial;
        throw exc;
      }
      check(n, status, "transcribe_run");

      return {
        ...materialize(n, singleAccessors(n, h)),
        aborted: F.wasAborted(h),
        truncated: F.wasTruncated(h),
      };
    }).finally(() => freeRunParams(n, p));
  }

  #buildRunParams(opts: TranscribeOptions): any {
    const n = this.#n;
    const p: any = {};
    n.F.runParamsInit(p);
    try {
      this.#fillRunParams(n, p, opts);
    } catch (e) {
      freeRunParams(n, p); // the caller never gets p, so free what was allocated
      throw e;
    }
    return p;
  }

  #fillRunParams(n: Native, p: any, opts: TranscribeOptions): void {
    p.task = lookup(TASKS, opts.task ?? "transcribe", "task");
    // Default "auto" mirrors the C transcribe_run_params_init default:
    // whisper resolves it to "segment" (its robust path), no-timestamp
    // families resolve to "none".
    p.timestamps = lookup(TIMESTAMPS, opts.timestamps ?? "auto", "timestamps");
    p.pnc = lookup(PNC, opts.pnc ?? "default", "pnc");
    p.itn = lookup(ITN, opts.itn ?? "default", "itn");
    p.diarize = lookup(DIARIZE, opts.diarize ?? "default", "diarize");
    if (opts.language !== undefined) p.language = cstr(opts.language, "language");
    if (opts.targetLanguage !== undefined)
      p.target_language = cstr(opts.targetLanguage, "targetLanguage");
    if (opts.keepSpecialTags !== undefined)
      p.keep_special_tags = opts.keepSpecialTags;
    if (opts.specKDrafts !== undefined) p.spec_k_drafts = opts.specKDrafts;
    if (opts.vocabulary !== undefined) {
      const terms = opts.vocabulary;
      if (!Array.isArray(terms) || !terms.every((t) => typeof t === "string"))
        throw new InvalidArgument("vocabulary must be an array of strings");
      if (terms.length > 0) {
        // Freed by freeRunParams after the call; the library copies the terms.
        terms.forEach((t) => cstr(t, "vocabulary"));
        const type = n.koffi.array("char *", terms.length);
        const arr = n.koffi.alloc(type, 1);
        n.koffi.encode(arr, type, terms);
        p.vocabulary = arr;
        p.n_vocabulary = terms.length;
      }
    }
    if (opts.prompt !== undefined) p.prompt = cstr(opts.prompt, "prompt");
    if (opts.prefix !== undefined) p.prefix = cstr(opts.prefix, "prefix");
    if (opts.family)
      p.family = buildFamily(n, this.#model.handle, opts.family, "run");
  }

  /**
   * Offline batch transcription; each item carries its own success/failure.
   * Inputs are borrowed, not copied: native code reads them on a worker thread
   * while this runs, so do not mutate them until the returned promise resolves.
   */
  async runBatch(
    pcms: PcmLike[],
    opts: TranscribeOptions = {},
  ): Promise<BatchItem[]> {
    const n = this.#n;
    const F = n.F;
    const h = this.handle;
    if (pcms.length === 0)
      throw new InvalidArgument("runBatch requires at least one input");
    const arrays = pcms.map(toFloat32);
    const counts = Int32Array.from(arrays, (a) => a.length);
    const p = this.#buildRunParams(opts);

    return this.#core.exclusive("runBatch", async (call) => {
      const status = await call(
        "runBatch()", opts.signal, F.runBatch, h, arrays, counts, arrays.length, p,
      );
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
            result: {
              ...materialize(n, batchAccessors(n, h, i)),
              aborted: false,
              truncated: false,
            },
          });
        } else {
          const error = exceptionForStatus(
            st,
            F.statusString(st),
            `utterance ${i}`,
          );
          error.utteranceIndex = i;
          if (
            st === g.TRANSCRIBE_ERR_ABORTED ||
            st === g.TRANSCRIBE_ERR_OUTPUT_TRUNCATED ||
            st === g.TRANSCRIBE_ERR_OUTPUT_REPETITION
          ) {
            error.partialResult = {
              ...materialize(n, batchAccessors(n, h, i)),
              aborted: st === g.TRANSCRIBE_ERR_ABORTED,
              truncated:
                st === g.TRANSCRIBE_ERR_OUTPUT_TRUNCATED ||
                st === g.TRANSCRIBE_ERR_OUTPUT_REPETITION,
            };
          }
          out.push({ ok: false, error });
        }
      }
      return out;
    }).finally(() => freeRunParams(n, p));
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
      pnc: opts.pnc,
      itn: opts.itn,
      diarize: opts.diarize,
      keepSpecialTags: opts.keepSpecialTags,
      specKDrafts: -1,
      vocabulary: opts.vocabulary,
      prompt: opts.prompt,
    });
    const sp: any = {};
    F.streamParamsInit(sp);
    try {
      sp.commit_policy = lookup(
        COMMIT_POLICIES,
        opts.commitPolicy ?? "auto",
        "commitPolicy",
      );
      if (opts.stablePrefixAgreementN !== undefined) {
        sp.stable_prefix_agreement_n = opts.stablePrefixAgreementN;
      }
      if (opts.family)
        sp.family = buildFamily(n, this.#model.handle, opts.family, "stream");
    } catch (e) {
      freeRunParams(n, rp); // sp holds nothing yet: buildFamily frees on throw
      throw e;
    }

    // Begin is a synchronous native call, so no in-flight window (no call()).
    return this.#core.exclusive("begin a stream", async () => {
      check(n, F.streamBegin(h, rp, sp), "transcribe_stream_begin");
      this.#lock.streamActive = true; // claim the lease for the whole stream lifetime
      // The Stream holds the Session (not a raw handle) so its calls fail fast
      // once the session is disposed, and so dispose() can find and invalidate it.
      const stream = new Stream(n, this, this.#lock, [rp, sp]); // pin params until reset
      const control = SESSION_CONTROL.get(this);
      if (!control) throw new TranscribeError("session control is missing");
      control.replaceCurrentStream(stream);
      return stream;
    }).finally(() => {
      // begin copied the prompting strings and the family extension
      freeRunParams(n, rp);
      freeRunParams(n, sp);
    });
  }

  get wasAborted(): boolean {
    this.#core.assertNotComputing("session wasAborted");
    return this.#n.F.wasAborted(this.handle);
  }

  dispose(): void {
    if (this.#core.disposed) return;
    this.#untrack(this); // stop the model from holding a dead session
    // Deactivate any live stream NOW (its reset() no-ops, reads throw via the
    // disposed handle), but release its model lease only inside the deferred
    // teardown, after sessionFree — so a stream/run queued ahead of this can't
    // claim the slot before the native session is actually gone.
    const stream = this.#activeStream;
    this.#activeStream = null;
    const teardown = stream ? STREAM_TEARDOWN.get(stream) : undefined;
    teardown?.deactivate();
    this.#core.dispose(this.#n.F.sessionFree, () => teardown?.releaseLease());
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
  #holdsLease = true; // born holding the model's compute lease (claimed at begin)

  /** @internal */
  constructor(n: Native, session: Session, lock: Mutex, keepalive: unknown[]) {
    this.#n = n;
    this.#session = session;
    this.#lock = lock;
    const sessionControl = SESSION_CONTROL.get(session);
    if (!sessionControl)
      throw new TranscribeError("session control is missing");
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
      throw new TranscribeError(
        `cannot ${what}: stream is no longer current for this session`,
      );
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
    return this.#sessionControl.core.exclusive(null, async (call) => {
      const u: any = {};
      n.F.streamUpdateInit(u);
      const status = await call(
        "feed()/finalize()", undefined, n.F.streamFeed, h, samples, samples.length, u,
      );
      // The lease follows the native stream: a feed refused before the family
      // hook (e.g. non-finite samples) leaves it ACTIVE, so keep the lease
      // (siblings stay Busy); a hook failure moves it to FAILED, no longer
      // active, so free the model-wide slot (state/lastStatus stay readable).
      if (status !== g.TRANSCRIBE_OK && n.F.streamGetState(h) !== g.TRANSCRIBE_STREAM_ACTIVE)
        this.#releaseLease();
      check(n, status, "transcribe_stream_feed");
      return toStreamUpdate(u);
    });
  }

  /** Flush remaining audio and commit the final text. */
  async finalize(): Promise<StreamUpdate> {
    const n = this.#n;
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("finalize stream");
    if (!this.#active) throw new TranscribeError("stream has been reset");
    return this.#sessionControl.core.exclusive(null, async (call) => {
      const u: any = {};
      n.F.streamUpdateInit(u);
      try {
        const status = await call("feed()/finalize()", undefined, n.F.streamFinalize, h, u);
        check(n, status, "transcribe_stream_finalize");
      } finally {
        // Finalize ends the active stream (FINISHED on success, FAILED on
        // error), so the model is free again — release the lease either way.
        this.#releaseLease();
      }
      return toStreamUpdate(u);
    });
  }

  /** Current text snapshot (copied at the boundary). */
  get text(): StreamText {
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("read stream text");
    this.#sessionControl.core.assertNotComputing("stream text");
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

  /** Full structured snapshot of the current hypothesis (owned copies). */
  get snapshot(): Transcript {
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("read stream snapshot");
    if (!this.#active) throw new TranscribeError("stream has been reset");
    this.#sessionControl.core.assertNotComputing("stream snapshot");
    return materialize(this.#n, singleAccessors(this.#n, h));
  }

  get state(): StreamState {
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("read stream state");
    if (!this.#active) return "idle"; // reset() returns to idle; native reset may still be queued
    this.#sessionControl.core.assertNotComputing("stream state");
    return STREAM_STATES[this.#n.F.streamGetState(h)] ?? "idle";
  }

  get revision(): number {
    const h = this.#session.handle; // throws if the session was disposed
    this.#assertCurrent("read stream revision");
    this.#sessionControl.core.assertNotComputing("stream revision");
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
    this.#sessionControl.core.assertNotComputing("stream lastStatus");
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

// ---- DiarizeSession --------------------------------------------------------

/** A DIARIZE-role session: who spoke when. Same compute rules as Session. */
export class DiarizeSession {
  #n: Native;
  #core: SessionCore;
  #model: TranscribeModel; // keep the model alive while this session lives
  #untrack: (self: DiarizeSession) => void;

  /** @internal */
  constructor(
    n: Native,
    model: TranscribeModel,
    handle: any,
    lock: Mutex,
    untrack: (self: DiarizeSession) => void,
  ) {
    this.#n = n;
    this.#model = model;
    this.#core = new SessionCore(n, handle, lock, n.F.diarizeSetAbortCallback);
    this.#untrack = untrack;
  }

  /**
   * Diarize one recording; returns its speaker turns, grouped by speaker and
   * time-ordered within a speaker (turns of different speakers may overlap).
   * The input PCM is borrowed, not copied (see Session.run).
   */
  async run(pcm: PcmLike, opts: DiarizeOptions = {}): Promise<SpeakerSegment[]> {
    const n = this.#n;
    const F = n.F;
    const h = this.#core.handle;
    const samples = toFloat32(pcm);
    const p: any = {};
    F.diarizeParamsInit(p);
    if (opts.family)
      p.family = buildFamily(n, this.#model.handle, opts.family, "diarize_run");

    return this.#core.exclusive("diarize", async (call) => {
      const status = await call("run()", opts.signal, F.diarizeRun, h, samples, samples.length, p);
      check(n, status, "transcribe_diarize_run");
      return readSpeakerSegments(n, F.diarizeNSegments(h), (i, o) =>
        F.diarizeGetSegment(h, i, o),
      );
    }).finally(() => freeRunParams(n, p));
  }

  /** load_ms plus the last run's mel / encode time. */
  get timings(): Timings {
    this.#core.assertNotComputing("session timings");
    const h = this.#core.handle;
    return readTimings(this.#n, (o) => this.#n.F.diarizeGetTimings(h, o));
  }

  dispose(): void {
    if (this.#core.disposed) return;
    this.#untrack(this);
    this.#core.dispose(this.#n.F.diarizeSessionFree);
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

// ---- LangIdSession ---------------------------------------------------------

/** A LANGID-role session: which language is spoken. Same compute rules as Session. */
export class LangIdSession {
  #n: Native;
  #core: SessionCore;
  #model: TranscribeModel; // keep the model alive while this session lives
  #untrack: (self: LangIdSession) => void;

  /** @internal */
  constructor(
    n: Native,
    model: TranscribeModel,
    handle: any,
    lock: Mutex,
    untrack: (self: LangIdSession) => void,
  ) {
    this.#n = n;
    this.#model = model;
    this.#core = new SessionCore(n, handle, lock, n.F.langidSetAbortCallback);
    this.#untrack = untrack;
  }

  /**
   * Identify the language of one clip; input longer than langidInfo.maxAudioMs
   * is scored on its first maxAudioMs. The input PCM is borrowed, not copied
   * (see Session.run).
   */
  async run(pcm: PcmLike, opts: LangIdOptions = {}): Promise<LangIdResult> {
    const n = this.#n;
    const F = n.F;
    const h = this.#core.handle;
    const samples = toFloat32(pcm);
    const p: any = {};
    F.langidParamsInit(p);
    if (opts.allowed !== undefined && opts.allowed !== null) {
      const codes = opts.allowed;
      if (!Array.isArray(codes) || !codes.every((c) => typeof c === "string"))
        throw new InvalidArgument("allowed must be an array of strings");
      // NULL would mean "every label", the opposite of an empty list.
      if (codes.length === 0)
        throw new InvalidArgument("allowed is empty; omit it for every label");
      codes.forEach((c) => cstr(c, "allowed"));
      // Freed after the call returns (including async worker calls).
      const type = n.koffi.array("char *", codes.length);
      const arr = n.koffi.alloc(type, 1);
      n.koffi.encode(arr, type, codes);
      p.allowed = arr;
      p.n_allowed = codes.length;
    }

    return this.#core.exclusive("langid", async (call) => {
      const status = await call("run()", opts.signal, F.langidRun, h, samples, samples.length, p);
      check(n, status, "transcribe_langid_run");
      const res: any = {};
      F.langidResultInit(res);
      check(n, F.langidGetResult(h, res), "transcribe_langid_get_result");
      const candidates: LangIdCandidate[] = [];
      for (let i = 0; i < res.n_candidates; i++) {
        const c: any = {};
        F.langidCandidateInit(c);
        check(n, F.langidGetCandidate(h, i, c), "transcribe_langid_get_candidate");
        candidates.push({
          index: c.index,
          code: c.code ?? "",
          name: c.name ?? "",
          p: c.p,
          logit: c.logit,
        });
      }
      return {
        candidates,
        code: candidates.length > 0 ? candidates[0].code : null,
        allowedMass: res.allowed_mass,
      };
    }).finally(() => {
      if (p.allowed) {
        n.koffi.free(p.allowed);
        p.allowed = null;
      }
    });
  }

  /** load_ms plus the last run's mel / encode time. */
  get timings(): Timings {
    this.#core.assertNotComputing("session timings");
    const h = this.#core.handle;
    return readTimings(this.#n, (o) => this.#n.F.langidGetTimings(h, o));
  }

  dispose(): void {
    if (this.#core.disposed) return;
    this.#untrack(this);
    this.#core.dispose(this.#n.F.langidSessionFree);
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

// ---- Model -----------------------------------------------------------------

export class TranscribeModel {
  #n: Native;
  #h: any;
  #disposed = false;
  #sessions = new Set<Session | DiarizeSession | LangIdSession>();
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

  /** Load a GGUF model. Awaits {@link initialize}, so it never blocks the event loop. */
  static async load(
    path: string,
    opts: ModelOptions = {},
  ): Promise<TranscribeModel> {
    const n = await nativeAsync();
    const p: any = {};
    n.F.modelLoadParamsInit(p);
    if ("gpuDevice" in opts) {
      throw new TranscribeError(
        "gpuDevice was removed in 0.2; pass a device from getAvailableBackends() instead",
      );
    }
    if (opts.backend) p.backend = lookup(BACKENDS, opts.backend, "backend");
    if (opts.device !== undefined) {
      const handle = DEVICE_HANDLES.get(opts.device);
      if (!handle) {
        throw new TranscribeError(
          "device must be an entry returned by getAvailableBackends() or model.device",
        );
      }
      p.device = handle;
    }

    const out: any[] = [null];
    const st = await callAsync<number>(n.F.modelLoadFile, path, p, out);
    check(n, st, `loading model ${path}`);
    if (!out[0])
      throw new ModelLoadError(`model load returned a null handle for ${path}`);
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
    if (!out[0])
      throw new TranscribeError("session init returned a null handle");
    const session = new Session(n, this, out[0], this.#lock, (s) =>
      this.#sessions.delete(s),
    );
    this.#sessions.add(session);
    return session;
  }

  /** Convenience: one session, one run, disposed after. */
  async transcribe(
    pcm: PcmLike,
    opts: TranscribeOptions = {},
  ): Promise<TranscriptionResult> {
    const session = this.createSession();
    try {
      return await session.run(pcm, opts);
    } finally {
      session.dispose(); // untracks itself from #sessions
    }
  }

  /** The roles this model serves; ASR calls on a model without "asr" throw UnsupportedRole. */
  get roles(): readonly Role[] {
    const mask = this.#n.F.modelRoles(this.handle);
    return (Object.keys(ROLES) as Role[]).filter((r) => mask & ROLES[r]);
  }

  /** Static facts of a "diarize" model; UnsupportedRole otherwise. */
  get diarizeInfo(): DiarizeInfo {
    const n = this.#n;
    const info: any = {};
    n.F.diarizeInfoInit(info);
    check(n, n.F.diarizeGetInfo(this.handle, info), "reading diarize info");
    return { sampleRate: info.sample_rate, maxSpeakers: info.max_speakers };
  }

  /** Open a DIARIZE-role session; UnsupportedRole on a model without "diarize". */
  createDiarizeSession(opts: DiarizeSessionOptions = {}): DiarizeSession {
    const n = this.#n;
    const p: any = {};
    n.F.diarizeSessionParamsInit(p);
    if (opts.nThreads !== undefined) p.n_threads = opts.nThreads;
    const out: any[] = [null];
    check(n, n.F.diarizeSessionInit(this.handle, p, out), "opening diarize session");
    if (!out[0])
      throw new TranscribeError("diarize session init returned a null handle");
    const session = new DiarizeSession(n, this, out[0], this.#lock, (s) =>
      this.#sessions.delete(s),
    );
    this.#sessions.add(session);
    return session;
  }

  /** Static facts of a "langid" model; UnsupportedRole otherwise. */
  get langidInfo(): LangIdInfo {
    const n = this.#n;
    const info: any = {};
    n.F.langidInfoInit(info);
    check(n, n.F.langidGetInfo(this.handle, info), "reading langid info");
    return {
      sampleRate: info.sample_rate,
      nLabels: info.n_labels,
      minAudioMs: info.min_audio_ms,
      maxAudioMs: info.max_audio_ms,
    };
  }

  /** [code, name] per label index of a "langid" model; UnsupportedRole otherwise. */
  get langidLabels(): Array<[string, string]> {
    const n = this.langidInfo.nLabels;
    const F = this.#n.F;
    const out: Array<[string, string]> = [];
    for (let i = 0; i < n; i++)
      out.push([F.langidLabelCode(this.handle, i) ?? "", F.langidLabelName(this.handle, i) ?? ""]);
    return out;
  }

  /** Label index of a code or alias ("he" and "iw" name the same label), or null. */
  langidLabelIndex(code: string): number | null {
    const i = this.#n.F.langidLabelIndex(this.handle, cstr(code, "code"));
    return i >= 0 ? i : null;
  }

  /** Open a LANGID-role session; UnsupportedRole on a model without "langid". */
  createLangIdSession(opts: LangIdSessionOptions = {}): LangIdSession {
    const n = this.#n;
    const p: any = {};
    n.F.langidSessionParamsInit(p);
    if (opts.nThreads !== undefined) p.n_threads = opts.nThreads;
    const out: any[] = [null];
    check(n, n.F.langidSessionInit(this.handle, p, out), "opening langid session");
    if (!out[0])
      throw new TranscribeError("langid session init returned a null handle");
    const session = new LangIdSession(n, this, out[0], this.#lock, (s) =>
      this.#sessions.delete(s),
    );
    this.#sessions.add(session);
    return session;
  }

  get capabilities(): Capabilities {
    const n = this.#n;
    const c: any = {};
    n.F.capabilitiesInit(c);
    check(n, n.F.modelGetCapabilities(this.handle, c), "reading capabilities");
    let languages: string[] = [];
    let translateTargetLanguages: string[] = [];
    try {
      if (c.languages && c.n_languages > 0) {
        languages = n.koffi.decode(c.languages, "char *", c.n_languages);
      }
    } catch {
      languages = [];
    }
    try {
      if (c.translate_target_languages && c.n_translate_target_languages > 0) {
        translateTargetLanguages = n.koffi.decode(
          c.translate_target_languages,
          "char *",
          c.n_translate_target_languages,
        );
      }
    } catch {
      translateTargetLanguages = [];
    }
    return {
      nativeSampleRate: c.native_sample_rate,
      languages,
      translateTargetLanguages,
      maxTimestampKind: TIMESTAMP_NAMES[c.max_timestamp_kind] ?? "none",
      supportsLanguageDetect: c.supports_language_detect,
      supportsTranslate: c.supports_translate,
      supportsStreaming: c.supports_streaming,
      supportsSpecDecode: c.supports_spec_decode,
      maxAudioMs: num(c.max_audio_ms),
    };
  }

  supports(feature: Feature): boolean {
    return this.#n.F.modelSupports(
      this.handle,
      lookup(FEATURES, feature, "feature"),
    );
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
        throw new NotImplementedByModel(
          "this model's tokenizer does not support encode",
        );
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
    const handle = this.#n.F.modelDevice(this.handle);
    if (!handle) throw new TranscribeError("model has no resolved compute device");
    const dev: any = {};
    this.#n.F.deviceInfoInit(dev);
    check(this.#n, this.#n.F.deviceGetInfo(handle, dev), "reading model device");
    return deviceFromRaw(dev, handle);
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
