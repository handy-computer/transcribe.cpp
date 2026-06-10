"""Exception hierarchy for transcribe.cpp, mapped from ``transcribe_status``.

Status numbers mirror the ``transcribe_status`` enum in
``include/transcribe.h``. The mapping is intentionally explicit (not derived
from the native enum) so a binding/library version skew surfaces as a clear
Python error rather than a wrong subclass.
"""

from __future__ import annotations

# transcribe_status values (include/transcribe.h).
OK = 0
ERR_INVALID_ARG = 1
ERR_NOT_IMPLEMENTED = 2
ERR_FILE_NOT_FOUND = 3
ERR_GGUF = 4
ERR_UNSUPPORTED_ARCH = 5
ERR_UNSUPPORTED_VARIANT = 6
ERR_OOM = 7
ERR_BACKEND = 8
ERR_SAMPLE_RATE = 9
ERR_UNSUPPORTED_LANGUAGE = 10
ERR_UNSUPPORTED_TASK = 11
ERR_UNSUPPORTED_TIMESTAMPS = 12
ERR_ABORTED = 13
ERR_BAD_STRUCT_SIZE = 14
ERR_UNSUPPORTED_PNC = 15
ERR_UNSUPPORTED_ITN = 16
ERR_INPUT_TOO_LONG = 17
ERR_OUTPUT_TRUNCATED = 18


class TranscribeError(RuntimeError):
    """Base class for every transcribe.cpp error.

    ``status`` is the numeric ``transcribe_status`` (0 for errors raised purely
    on the Python side, e.g. library loading or input validation).
    """

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


class OutputTruncated(TranscribeError):
    pass


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
    ERR_ABORTED: TranscribeError,
    ERR_BAD_STRUCT_SIZE: AbiError,
    ERR_UNSUPPORTED_PNC: UnsupportedRequest,
    ERR_UNSUPPORTED_ITN: UnsupportedRequest,
    ERR_INPUT_TOO_LONG: InputTooLong,
    ERR_OUTPUT_TRUNCATED: OutputTruncated,
}


def raise_for_status(status: int, status_string: str, context: str = "") -> None:
    """Raise the mapped exception for a non-OK status; return on OK."""
    if status == OK:
        return
    exc = _STATUS_TO_EXC.get(status, TranscribeError)
    prefix = f"{context}: " if context else ""
    raise exc(f"{prefix}{status_string} (status {status})", status=status)
