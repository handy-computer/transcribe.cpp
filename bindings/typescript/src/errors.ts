/** Error hierarchy mapped from `transcribe_status`. Mirrors the Python binding. */

import type { TranscriptionResult } from "./types.js";
import * as g from "./_generated.js";

export class TranscribeError extends Error {
  /** The `transcribe_status` value (0 for binding-side errors). */
  readonly status: number;
  /** Set on per-utterance failures from a batch run. */
  utteranceIndex?: number;

  constructor(message: string, status: number = g.TRANSCRIBE_OK) {
    super(message);
    this.name = new.target.name;
    this.status = status;
  }
}

export class InvalidArgument extends TranscribeError {}
export class NotImplementedByModel extends TranscribeError {}
export class ModelFileNotFound extends TranscribeError {}
export class ModelLoadError extends TranscribeError {}
export class OutOfMemory extends TranscribeError {}
export class BackendError extends TranscribeError {}
export class UnsupportedRequest extends TranscribeError {}
export class AbiError extends TranscribeError {}
export class InputTooLong extends TranscribeError {}
export class VersionMismatch extends TranscribeError {}

/** Raised when a run is cancelled; carries any partial transcript. */
export class Aborted extends TranscribeError {
  partialResult?: TranscriptionResult;
}

/** Raised when decode hits the context/generation cap; carries the partial. */
export class OutputTruncated extends TranscribeError {
  partialResult?: TranscriptionResult;
}

const STATUS_TO_EXC: Record<number, new (m: string, s?: number) => TranscribeError> = {
  [g.TRANSCRIBE_ERR_INVALID_ARG]: InvalidArgument,
  [g.TRANSCRIBE_ERR_NOT_IMPLEMENTED]: NotImplementedByModel,
  [g.TRANSCRIBE_ERR_FILE_NOT_FOUND]: ModelFileNotFound,
  [g.TRANSCRIBE_ERR_GGUF]: ModelLoadError,
  [g.TRANSCRIBE_ERR_UNSUPPORTED_ARCH]: ModelLoadError,
  [g.TRANSCRIBE_ERR_UNSUPPORTED_VARIANT]: ModelLoadError,
  [g.TRANSCRIBE_ERR_OOM]: OutOfMemory,
  [g.TRANSCRIBE_ERR_BACKEND]: BackendError,
  [g.TRANSCRIBE_ERR_SAMPLE_RATE]: InvalidArgument,
  [g.TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE]: UnsupportedRequest,
  [g.TRANSCRIBE_ERR_UNSUPPORTED_TASK]: UnsupportedRequest,
  [g.TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS]: UnsupportedRequest,
  [g.TRANSCRIBE_ERR_ABORTED]: Aborted,
  [g.TRANSCRIBE_ERR_BAD_STRUCT_SIZE]: AbiError,
  [g.TRANSCRIBE_ERR_UNSUPPORTED_PNC]: UnsupportedRequest,
  [g.TRANSCRIBE_ERR_UNSUPPORTED_ITN]: UnsupportedRequest,
  [g.TRANSCRIBE_ERR_INPUT_TOO_LONG]: InputTooLong,
  [g.TRANSCRIBE_ERR_OUTPUT_TRUNCATED]: OutputTruncated,
};

/** Build (do not throw) the mapped exception for a status. */
export function exceptionForStatus(
  status: number,
  statusString: string,
  context = "",
): TranscribeError {
  const Cls = STATUS_TO_EXC[status] ?? TranscribeError;
  const msg = context
    ? `${context}: ${statusString} (status ${status})`
    : `${statusString} (status ${status})`;
  return new Cls(msg, status);
}

/** Throw the mapped exception unless the status is OK. */
export function raiseForStatus(status: number, statusString: string, context = ""): void {
  if (status === g.TRANSCRIBE_OK) return;
  throw exceptionForStatus(status, statusString, context);
}
