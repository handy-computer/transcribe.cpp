// No-model tier: status -> error-class mapping, pure (no native call, no GGUF).
// Mirrors bindings/python/tests/test_errors.py.
import { test } from "node:test";
import assert from "node:assert/strict";
import * as g from "../dist/_generated.js";
import {
  exceptionForStatus,
  raiseForStatus,
  TranscribeError,
  InvalidArgument,
  NotImplementedByModel,
  ModelFileNotFound,
  ModelLoadError,
  OutOfMemory,
  BackendError,
  UnsupportedRequest,
  UnsupportedRole,
  Aborted,
  AbiError,
  InputTooLong,
  OutputTruncated,
  OutputRepetition,
} from "../dist/index.js";

const EXPECTED = {
  TRANSCRIBE_ERR_INVALID_ARG: InvalidArgument,
  TRANSCRIBE_ERR_NOT_IMPLEMENTED: NotImplementedByModel,
  TRANSCRIBE_ERR_FILE_NOT_FOUND: ModelFileNotFound,
  TRANSCRIBE_ERR_GGUF: ModelLoadError,
  TRANSCRIBE_ERR_UNSUPPORTED_ARCH: ModelLoadError,
  TRANSCRIBE_ERR_UNSUPPORTED_VARIANT: ModelLoadError,
  TRANSCRIBE_ERR_OOM: OutOfMemory,
  TRANSCRIBE_ERR_BACKEND: BackendError,
  TRANSCRIBE_ERR_SAMPLE_RATE: InvalidArgument,
  TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE: UnsupportedRequest,
  TRANSCRIBE_ERR_UNSUPPORTED_TASK: UnsupportedRequest,
  TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS: UnsupportedRequest,
  TRANSCRIBE_ERR_ABORTED: Aborted,
  TRANSCRIBE_ERR_BAD_STRUCT_SIZE: AbiError,
  TRANSCRIBE_ERR_UNSUPPORTED_PNC: UnsupportedRequest,
  TRANSCRIBE_ERR_UNSUPPORTED_ITN: UnsupportedRequest,
  TRANSCRIBE_ERR_INPUT_TOO_LONG: InputTooLong,
  TRANSCRIBE_ERR_OUTPUT_TRUNCATED: OutputTruncated,
  TRANSCRIBE_ERR_OUTPUT_REPETITION: OutputRepetition,
  TRANSCRIBE_ERR_UNSUPPORTED_ROLE: UnsupportedRole,
};

test("every transcribe_status maps to its documented error class", () => {
  // The expectation covers every non-OK status the header defines, and nothing
  // else: a new C status must be mapped (and listed here) deliberately.
  const generated = Object.keys(g)
    .filter((k) => k.startsWith("TRANSCRIBE_ERR_"))
    .sort();
  assert.deepEqual(generated, Object.keys(EXPECTED).sort());

  for (const [name, Cls] of Object.entries(EXPECTED)) {
    const status = g[name];
    assert.throws(
      () => raiseForStatus(status, "synthetic", "ctx"),
      (e) => {
        assert.equal(e.constructor, Cls, name);
        assert.ok(e instanceof TranscribeError);
        assert.equal(e.status, status);
        assert.equal(e.name, Cls.name);
        assert.match(e.message, /ctx: synthetic/);
        return true;
      },
    );
  }
  assert.equal(Object.getPrototypeOf(UnsupportedRole), TranscribeError);
});

test("unknown status degrades to the base class", () => {
  const e = exceptionForStatus(999, "mystery");
  assert.equal(e.constructor, TranscribeError);
  assert.equal(e.status, 999);
});
