/** Public value types for the transcribe.cpp TypeScript binding. */

import type { TranscribeError } from "./errors.js";

export type Backend = "auto" | "cpu" | "cpu_accel" | "cuda" | "rocm" | "vulkan" | "metal";
export type KvType = "auto" | "f32" | "f16";
export type Task = "transcribe" | "translate";
export type TimestampKind = "none" | "auto" | "segment" | "word" | "token";
export type Diarize = "default" | "off" | "on";
export type Feature =
  | "initial_prompt"
  | "temperature_fallback"
  | "long_form"
  | "cancellation"
  | "pnc"
  | "itn"
  | "diarization";

/** Mono float32 PCM at the model's native sample rate (16 kHz for v1). */
export type PcmLike = Float32Array | number[] | ArrayBuffer | Buffer;

export interface Segment {
  text: string;
  t0Ms: number;
  t1Ms: number;
  firstWord: number;
  nWords: number;
  firstToken: number;
  nTokens: number;
  /** 1-based speaker id; 0 means no attribution. */
  speakerId: number;
}

export interface SpeakerSegment {
  t0Ms: number;
  t1Ms: number;
  speakerId: number;
  /** Family-specific confidence, or NaN when unavailable. */
  p: number;
}

export interface Word {
  text: string;
  t0Ms: number;
  t1Ms: number;
  segIndex: number;
  firstToken: number;
  nTokens: number;
}

export interface Token {
  text: string;
  id: number;
  p: number;
  t0Ms: number;
  t1Ms: number;
  segIndex: number;
  wordIndex: number;
}

export interface Timings {
  loadMs: number;
  melMs: number;
  encodeMs: number;
  decodeMs: number;
}

export interface Capabilities {
  nativeSampleRate: number;
  languages: string[];
  translateTargetLanguages: string[];
  maxTimestampKind: TimestampKind;
  supportsLanguageDetect: boolean;
  supportsTranslate: boolean;
  supportsStreaming: boolean;
  supportsSpecDecode: boolean;
  maxAudioMs: number;
}

export interface SessionLimits {
  effectiveNCtx: number;
  effectiveMaxAudioMs: number;
  maxKvBytes: number;
}

export interface TranscriptionResult {
  text: string;
  /** The model's decoded output before family post-processing (diarization
   *  markers, timestamp/special tokens, tag filtering, whitespace trims).
   *  Equal to `text` modulo whitespace for families that emit clean text. */
  rawText: string;
  language: string;
  timestampKind: TimestampKind;
  segments: Segment[];
  speakerSegments: SpeakerSegment[];
  words: Word[];
  tokens: Token[];
  timings: Timings;
  aborted: boolean;
  truncated: boolean;
}

/** Vendor-agnostic device class, orthogonal to {@link BackendInfo.kind}.
 *  `"unknown"` is reported for a device-type value newer than this binding —
 *  distinguish such devices by {@link BackendInfo.deviceId} / name, not this axis. */
export type DeviceType = "cpu" | "gpu" | "igpu" | "accel" | "unknown";

export interface BackendInfo {
  name: string;
  description: string;
  kind: string;
  /** The CPU/GPU/IGPU/ACCEL axis, orthogonal to `kind`. */
  deviceType: DeviceType;
  /** Stable hardware id (PCI bus id) when the backend reports one, else null
   *  (e.g. Metal). */
  deviceId: string | null;
  /** Reported device memory capacity in bytes, or 0 if unreported. */
  memoryTotal: number;
  /** Available device memory in bytes — a snapshot at query time, or 0 if
   *  unreported. Re-query (via {@link getAvailableBackends} or `model.device`)
   *  to refresh; backend-defined and not comparable across device kinds. */
  memoryFree: number;
  /** Process-local registry index for display. Pass this object via
   *  {@link ModelOptions.device} for exact selection; persist `deviceId`, not
   *  the index. */
  index: number | null;
}

export interface ModelOptions {
  /** "auto" (default), or an explicit backend. */
  backend?: Backend;
  /** Exact device returned by {@link getAvailableBackends}. Omit for the
   *  backend's automatic policy. Exact selection never falls back. */
  device?: BackendInfo;
}

export interface SessionOptions {
  /** CPU threads for CPU-side ops; 0 = library default. */
  nThreads?: number;
  kvType?: KvType;
  /** Context-token cap; 0 = model default. */
  nCtx?: number;
}

export interface TranscribeOptions {
  task?: Task;
  language?: string;
  targetLanguage?: string;
  /** Default "auto" (richest the model supports, per-family). */
  timestamps?: TimestampKind;
  /** Default "default" (speaker attribution off for every family). */
  diarize?: Diarize;
  keepSpecialTags?: boolean;
  /** Speculative-decode draft count; -1 = family default. */
  specKDrafts?: number;
  /** Cancel the run cooperatively. */
  signal?: AbortSignal;
  /** A run-slot family extension (e.g. whisper). */
  family?: FamilyExtension;
}

/** One result of a batch run: success carries the transcript, failure the error.
 *  On failure, `error.utteranceIndex` is set, and `error.partialResult` carries any
 *  recovered transcript when the failure was an abort/truncation. */
export type BatchItem =
  | { ok: true; result: TranscriptionResult }
  | { ok: false; error: TranscribeError };

// ---- streaming -------------------------------------------------------------

export type CommitPolicy = "auto" | "on_finalize" | "stable_prefix";
export type StreamState = "idle" | "active" | "finished" | "failed";
export type ExtSlot = "run" | "stream";

export interface StreamUpdate {
  resultChanged: boolean;
  isFinal: boolean;
  revision: number;
  inputReceivedMs: number;
  audioCommittedMs: number;
  bufferedMs: number;
  committedChanged: boolean;
  tentativeChanged: boolean;
}

export interface StreamText {
  /** Raw model hypothesis. */
  full: string;
  /** Append-only, stable prefix. */
  committed: string;
  /** Volatile suffix. */
  tentative: string;
}

export interface StreamOptions {
  task?: Task;
  language?: string;
  targetLanguage?: string;
  timestamps?: TimestampKind;
  diarize?: Diarize;
  keepSpecialTags?: boolean;
  commitPolicy?: CommitPolicy;
  stablePrefixAgreementN?: number;
  /** A stream-slot family extension (moonshine, parakeet, voxtral). */
  family?: FamilyExtension;
}

// ---- family extensions (typed, discriminated by `kind`) --------------------

export interface WhisperRunOptions {
  initialPrompt?: string;
  conditionOnPrevTokens?: boolean;
  temperature?: number;
  temperatureInc?: number;
  compressionRatioThold?: number;
  logprobThold?: number;
  noSpeechThold?: number;
  maxPrevContextTokens?: number;
  seed?: number;
  maxInitialTimestamp?: number;
}
export interface MoonshineStreamingOptions {
  minDecodeIntervalMs?: number;
}
export interface ParakeetStreamOptions {
  attContextRight?: number;
}
export interface ParakeetBufferedStreamOptions {
  leftMs?: number;
  chunkMs?: number;
  rightMs?: number;
}
export interface VoxtralRealtimeStreamOptions {
  numDelayTokens?: number;
  minDecodeIntervalMs?: number;
}
/** Sortformer streaming operating point (latency / accuracy trade-off).
 *  "default" keeps the GGUF-shipped checkpoint configuration;
 *  "very_high_latency" (~30 s lookahead) is the offline-file operating
 *  point; "low_latency" (~1 s) is the real-time point. */
export type SortformerPreset =
  | "default"
  | "very_high_latency"
  | "high_latency"
  | "low_latency";
/** Sortformer diarizer options (run slot). A run produces speaker
 *  segments, no text. */
export interface SortformerStreamOptions {
  preset?: SortformerPreset;
}

export type FamilyExtension =
  | ({ kind: "whisper" } & WhisperRunOptions)
  | ({ kind: "moonshine" } & MoonshineStreamingOptions)
  | ({ kind: "parakeet" } & ParakeetStreamOptions)
  | ({ kind: "parakeet_buffered" } & ParakeetBufferedStreamOptions)
  | ({ kind: "voxtral" } & VoxtralRealtimeStreamOptions)
  | ({ kind: "sortformer" } & SortformerStreamOptions);
