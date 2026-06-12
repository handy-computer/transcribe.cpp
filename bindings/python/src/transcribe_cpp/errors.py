"""Exception hierarchy for transcribe.cpp, mapped from ``transcribe_status``.

Status values come from the generated FFI layer (``_generated``), which is
produced from ``include/transcribe.h`` and drift-gated in CI — a single
source instead of a hand-maintained mirror. Skew between this binding and
the *loaded native library* is caught at import time by the version and
header-hash gates in ``_library``; the explicit status→exception mapping
below is the part that stays a deliberate, reviewable design decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ._generated import (
    TRANSCRIBE_ERR_ABORTED as ERR_ABORTED,
    TRANSCRIBE_ERR_BACKEND as ERR_BACKEND,
    TRANSCRIBE_ERR_BAD_STRUCT_SIZE as ERR_BAD_STRUCT_SIZE,
    TRANSCRIBE_ERR_FILE_NOT_FOUND as ERR_FILE_NOT_FOUND,
    TRANSCRIBE_ERR_GGUF as ERR_GGUF,
    TRANSCRIBE_ERR_INPUT_TOO_LONG as ERR_INPUT_TOO_LONG,
    TRANSCRIBE_ERR_INVALID_ARG as ERR_INVALID_ARG,
    TRANSCRIBE_ERR_NOT_IMPLEMENTED as ERR_NOT_IMPLEMENTED,
    TRANSCRIBE_ERR_OOM as ERR_OOM,
    TRANSCRIBE_ERR_OUTPUT_TRUNCATED as ERR_OUTPUT_TRUNCATED,
    TRANSCRIBE_ERR_SAMPLE_RATE as ERR_SAMPLE_RATE,
    TRANSCRIBE_ERR_UNSUPPORTED_ARCH as ERR_UNSUPPORTED_ARCH,
    TRANSCRIBE_ERR_UNSUPPORTED_ITN as ERR_UNSUPPORTED_ITN,
    TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE as ERR_UNSUPPORTED_LANGUAGE,
    TRANSCRIBE_ERR_UNSUPPORTED_PNC as ERR_UNSUPPORTED_PNC,
    TRANSCRIBE_ERR_UNSUPPORTED_TASK as ERR_UNSUPPORTED_TASK,
    TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS as ERR_UNSUPPORTED_TIMESTAMPS,
    TRANSCRIBE_ERR_UNSUPPORTED_VARIANT as ERR_UNSUPPORTED_VARIANT,
    TRANSCRIBE_OK as OK,
)

if TYPE_CHECKING:  # avoid a runtime import cycle; Result lives in __init__
    from . import Result


class TranscribeError(RuntimeError):
    """Base class for every transcribe.cpp error.

    ``status`` is the numeric ``transcribe_status`` (0 for errors raised purely
    on the Python side, e.g. library loading or input validation).
    ``utterance_index`` is set on per-utterance failures raised out of
    ``Session.run_batch`` (None everywhere else).
    """

    utterance_index: Optional[int] = None

    def __init__(self, message: str, status: int = OK):
        super().__init__(message)
        self.status = status


class InvalidArgument(TranscribeError):
    pass


class NotImplementedByModel(TranscribeError):
    pass


class ModelFileNotFound(TranscribeError):
    pass


class ModelLoadError(TranscribeError):
    """GGUF parse / unsupported arch / unsupported variant."""


class OutOfMemory(TranscribeError):
    pass


class BackendError(TranscribeError):
    pass


class UnsupportedRequest(TranscribeError):
    """Task / language / timestamp granularity the model does not support."""


class AbiError(TranscribeError):
    """Caller-owned struct layout did not match the library (struct_size)."""


class InputTooLong(TranscribeError):
    pass


class Aborted(TranscribeError):
    """The run/stream was ended by ``Session.cancel()``.

    ``partial_result`` holds the materialized transcript of the chunks that
    completed before the abort (the C API preserves it on the session), or
    None when the abort surfaced outside a result-bearing call.
    """

    partial_result: "Optional[Result]" = None


class OutputTruncated(TranscribeError):
    """The decode hit the model's generation budget before end-of-stream —
    the transcript is incomplete by contract.

    ``partial_result`` holds the materialized partial transcript (always
    preserved by the C API for this status), or None when the status
    surfaced outside a result-bearing call.
    """

    partial_result: "Optional[Result]" = None


_STATUS_TO_EXC = {
    ERR_INVALID_ARG: InvalidArgument,
    ERR_NOT_IMPLEMENTED: NotImplementedByModel,
    ERR_FILE_NOT_FOUND: ModelFileNotFound,
    ERR_GGUF: ModelLoadError,
    ERR_UNSUPPORTED_ARCH: ModelLoadError,
    ERR_UNSUPPORTED_VARIANT: ModelLoadError,
    ERR_OOM: OutOfMemory,
    ERR_BACKEND: BackendError,
    ERR_SAMPLE_RATE: InvalidArgument,
    ERR_UNSUPPORTED_LANGUAGE: UnsupportedRequest,
    ERR_UNSUPPORTED_TASK: UnsupportedRequest,
    ERR_UNSUPPORTED_TIMESTAMPS: UnsupportedRequest,
    ERR_ABORTED: Aborted,
    ERR_BAD_STRUCT_SIZE: AbiError,
    ERR_UNSUPPORTED_PNC: UnsupportedRequest,
    ERR_UNSUPPORTED_ITN: UnsupportedRequest,
    ERR_INPUT_TOO_LONG: InputTooLong,
    ERR_OUTPUT_TRUNCATED: OutputTruncated,
}


def exception_for_status(
    status: int, status_string: str, context: str = ""
) -> TranscribeError:
    """Build (without raising) the mapped exception for a non-OK status."""
    exc_type = _STATUS_TO_EXC.get(status, TranscribeError)
    prefix = f"{context}: " if context else ""
    return exc_type(f"{prefix}{status_string} (status {status})", status=status)


def raise_for_status(status: int, status_string: str, context: str = "") -> None:
    """Raise the mapped exception for a non-OK status; return on OK."""
    if status == OK:
        return
    raise exception_for_status(status, status_string, context)
