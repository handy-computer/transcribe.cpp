/**
 * transcribe.cpp — TypeScript/Node.js bindings.
 *
 * A koffi FFI binding over the shared native library. Offline transcription,
 * backend discovery, log routing, and cooperative cancellation. Streaming,
 * batch, and family extensions land in later milestones.
 */

import { native, setLogHandler, type LogHandler, type Native } from "./native.js";
import * as g from "./_generated.js";
import {
  Aborted,
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

function callAsync<T = number>(fn: any, ...args: any[]): Promise<T> {
  return new Promise((resolve, reject) =>
    fn.async(...args, (err: Error | null, res: T) => (err ? reject(err) : resolve(res))),
  );
}

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

function num(v: number | bigint): number {
  return typeof v === "bigint" ? Number(v) : v;
}

/**
 * A FIFO async mutex. One per Model: it serializes every native compute call
 * (run, batch, stream feed/finalize) so the C "one compute in flight per model"
 * contract holds even when koffi `.async()` puts work on libuv workers.
 */
class Mutex {
  #tail: Promise<void> = Promise.resolve();
  run<T>(fn: () => Promise<T>): Promise<T> {
    const prev = this.#tail;
    let release!: () => void;
    this.#tail = new Promise<void>((r) => (release = r));
    return prev.then(fn).finally(release);
  }
}

// ---- module-level introspection -------------------------------------------

export function version(): { version: string; commit: string; headerHash: string } {
  const n = native();
  return { version: n.F.version(), commit: n.F.versionCommit(), headerHash: g.PUBLIC_HEADER_HASH };
}

export function libraryPath(): string {
  return native().libraryPath;
}

export function getAvailableBackends(): BackendInfo[] {
  const n = native();
  const count = n.F.backendDeviceCount();
  const out: BackendInfo[] = [];
  for (let i = 0; i < count; i++) {
    const dev: any = {};
    n.F.backendDeviceInit(dev);
    check(n, n.F.getBackendDevice(i, dev), `reading backend device ${i}`);
    out.push({ name: dev.name ?? "", description: dev.description ?? "", kind: dev.kind ?? "" });
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

// ---- Session ---------------------------------------------------------------

export class Session {
  #n: Native;
  #h: any;
  #model: TranscribeModel; // keep the model alive while this session lives
  #lock: Mutex; // shared with the model; serializes compute model-wide
  #disposed = false;

  /** @internal */
  constructor(n: Native, model: TranscribeModel, handle: any, lock: Mutex) {
    this.#n = n;
    this.#model = model;
    this.#h = handle;
    this.#lock = lock;
  }

  /** @internal */
  get handle(): any {
    if (this.#disposed) throw new TranscribeError("session has been disposed");
    return this.#h;
  }

  get limits(): SessionLimits {
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

  async run(pcm: PcmLike, opts: TranscribeOptions = {}): Promise<TranscriptionResult> {
    const n = this.#n;
    const F = n.F;
    const h = this.handle;
    const samples = toFloat32(pcm);

    const p = this.#buildRunParams(opts);

    return this.#lock.run(async () => {
      const cancel = this.#installAbort(opts.signal);
      let status: number;
      try {
        status = await callAsync<number>(F.run, h, samples, samples.length, p);
      } finally {
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

  /** Offline batch transcription; each item carries its own success/failure. */
  async runBatch(pcms: PcmLike[], opts: TranscribeOptions = {}): Promise<BatchItem[]> {
    const n = this.#n;
    const F = n.F;
    const h = this.handle;
    if (pcms.length === 0) throw new InvalidArgument("runBatch requires at least one input");
    const arrays = pcms.map(toFloat32);
    const counts = Int32Array.from(arrays, (a) => a.length);
    const p = this.#buildRunParams(opts);

    return this.#lock.run(async () => {
      const cancel = this.#installAbort(opts.signal);
      let status: number;
      try {
        status = await callAsync<number>(F.runBatch, h, arrays, counts, arrays.length, p);
      } finally {
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
            (error as Aborted).partialResult = {
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
      check(n, F.streamBegin(h, rp, sp), "transcribe_stream_begin");
      return new Stream(n, h, this.#lock, [rp, sp]); // pin params until reset
    });
  }

  /** Wire an AbortSignal to a native abort callback for one run. */
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
    return this.#n.F.wasAborted(this.handle);
  }

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#n.F.sessionFree(this.#h);
    this.#h = null;
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

// ---- Stream ----------------------------------------------------------------

export class Stream {
  #n: Native;
  #h: any; // session handle
  #lock: Mutex;
  #keepalive: unknown[] | null;
  #active = true;

  /** @internal */
  constructor(n: Native, sessionHandle: any, lock: Mutex, keepalive: unknown[]) {
    this.#n = n;
    this.#h = sessionHandle;
    this.#lock = lock;
    this.#keepalive = keepalive;
  }

  /** Feed one chunk of PCM; returns the update snapshot. */
  async feed(pcm: PcmLike): Promise<StreamUpdate> {
    const n = this.#n;
    const samples = toFloat32(pcm);
    return this.#lock.run(async () => {
      const u: any = {};
      n.F.streamUpdateInit(u);
      check(
        n,
        await callAsync<number>(n.F.streamFeed, this.#h, samples, samples.length, u),
        "transcribe_stream_feed",
      );
      return toStreamUpdate(u);
    });
  }

  /** Flush remaining audio and commit the final text. */
  async finalize(): Promise<StreamUpdate> {
    const n = this.#n;
    return this.#lock.run(async () => {
      const u: any = {};
      n.F.streamUpdateInit(u);
      check(n, await callAsync<number>(n.F.streamFinalize, this.#h, u), "transcribe_stream_finalize");
      return toStreamUpdate(u);
    });
  }

  /** Current text snapshot (copied at the boundary). */
  get text(): StreamText {
    const n = this.#n;
    const t: any = {};
    n.F.streamTextInit(t);
    check(n, n.F.streamGetText(this.#h, t), "transcribe_stream_get_text");
    return {
      full: t.full_text ?? "",
      committed: t.committed_text ?? "",
      tentative: t.tentative_text ?? "",
    };
  }

  get state(): StreamState {
    return STREAM_STATES[this.#n.F.streamGetState(this.#h)] ?? "idle";
  }

  get revision(): number {
    return this.#n.F.streamRevision(this.#h);
  }

  /** End the stream and return the session to idle. Idempotent. */
  reset(): void {
    if (!this.#active) return;
    this.#active = false;
    this.#n.F.streamReset(this.#h);
    this.#keepalive = null;
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
    const session = new Session(n, this, out[0], this.#lock);
    this.#sessions.add(session);
    return session;
  }

  /** Convenience: one session, one run, disposed after. */
  async transcribe(pcm: PcmLike, opts: TranscribeOptions = {}): Promise<TranscriptionResult> {
    const session = this.createSession();
    try {
      return await session.run(pcm, opts);
    } finally {
      session.dispose();
      this.#sessions.delete(session);
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

  dispose(): void {
    if (this.#disposed) return;
    this.#disposed = true;
    for (const s of this.#sessions) s.dispose();
    this.#sessions.clear();
    this.#n.F.modelFree(this.#h);
    this.#h = null;
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
