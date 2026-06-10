"""Python bindings for transcribe.cpp — a ggml speech-to-text library.

    import transcribe_cpp

    with transcribe_cpp.Model("model.gguf") as model:
        with model.session() as session:
            result = session.run(pcm_float32_16k_mono)
            print(result.text)

The native library is loaded at import time; its ABI layout and version are
verified against this binding before any model is touched (see
``_abi.verify_layouts`` and the version check below). PCM passed to ``run`` must
be 16 kHz mono float32; resample external audio first, e.g.::

    ffmpeg -i in.wav -ar 16000 -ac 1 -f f32le out.f32

Long-running native calls (model load, run) release the GIL — ctypes does this
for every foreign call — so other Python threads make progress during inference.
"""

from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from typing import Literal, Sequence, Union

from . import _abi, _generated
from ._library import _base_version, load_library, selected_provider
from .errors import (
    AbiError,
    BackendError,
    InputTooLong,
    InvalidArgument,
    ModelFileNotFound,
    ModelLoadError,
    NotImplementedByModel,
    OutOfMemory,
    OutputTruncated,
    TranscribeError,
    UnsupportedRequest,
    raise_for_status,
)

__version__ = "0.0.1"

# String-enum types, exported so callers (and type checkers) can name them.
Backend = Literal["auto", "cpu", "metal", "vulkan", "cpu_accel", "cuda"]
KVType = Literal["auto", "f32", "f16"]
Task = Literal["transcribe", "translate"]
Timestamps = Literal["none", "auto", "segment", "word", "token"]
CommitPolicy = Literal["auto", "on_finalize", "stable_prefix"]
Feature = Literal[
    "initial_prompt", "temperature_fallback", "long_form",
    "cancellation", "pnc", "itn",
]

__all__ = [
    "__version__",
    "transcribe",
    "Model",
    "Session",
    "Result",
    "Segment",
    "Word",
    "Token",
    "Capabilities",
    "Timings",
    "Stream",
    "StreamUpdate",
    "StreamText",
    "FamilyExtension",
    "WhisperRunOptions",
    "MoonshineStreamingOptions",
    "ParakeetStreamOptions",
    "ParakeetBufferedStreamOptions",
    "VoxtralRealtimeStreamOptions",
    "Backend",
    "KVType",
    "Task",
    "Timestamps",
    "CommitPolicy",
    "Feature",
    "TranscribeError",
    "InvalidArgument",
    "ModelFileNotFound",
    "ModelLoadError",
    "NotImplementedByModel",
    "OutOfMemory",
    "BackendError",
    "UnsupportedRequest",
    "AbiError",
    "InputTooLong",
    "OutputTruncated",
    "native_version",
    "native_commit",
    "library_path",
    "native_provider",
    "set_log_callback",
]

# --- native library bootstrap --------------------------------------------

_lib, _lib_path = load_library()
_generated.configure(_lib)
_abi.verify_layouts(_lib)

# _base_version is single-sourced in _library (re-exported here so the import
# gate and tests can reach it as transcribe_cpp._base_version). Pre-1.0 the
# Python package and native library must agree on the base (MAJOR.MINOR.PATCH)
# release segment exactly; a packaging-only fix (0.0.1.post1) keeps the same
# base and so still loads against the 0.0.1 native library.
_native_version = _lib.transcribe_version().decode("ascii")
if _base_version(_native_version) != _base_version(__version__):
    raise TranscribeError(
        f"transcribe_cpp {__version__} cannot use native library "
        f"{_native_version}: pre-1.0 requires a matching base "
        f"(MAJOR.MINOR.PATCH) version. Rebuild the native library at the "
        "matching version or install a matching native provider."
    )

_byref = ctypes.byref

# Callback function types — must match the generated argtypes for
# transcribe_log_set / transcribe_set_abort_callback. ctypes acquires the GIL
# when C invokes these, and a CFUNCTYPE instance must be kept alive for as long
# as C may call it (a module global for logging; on the Session for abort).
_LOG_CFUNC = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_ABORT_CFUNC = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)

# Struct aliases onto the generated ctypes layer (the low-level names mirror the
# C tags; alias them for readability here).
_ModelLoadParams = _generated.transcribe_model_load_params
_SessionParams = _generated.transcribe_session_params
_RunParams = _generated.transcribe_run_params
_Capabilities = _generated.transcribe_capabilities
_Timings = _generated.transcribe_timings
_Segment = _generated.transcribe_segment
_Word = _generated.transcribe_word
_Token = _generated.transcribe_token
_StreamParams = _generated.transcribe_stream_params
_StreamUpdate = _generated.transcribe_stream_update
_StreamText = _generated.transcribe_stream_text

# --- enum maps (values sourced from the generated enum constants) ---------

_BACKENDS = {
    "auto": _generated.TRANSCRIBE_BACKEND_AUTO,
    "cpu": _generated.TRANSCRIBE_BACKEND_CPU,
    "metal": _generated.TRANSCRIBE_BACKEND_METAL,
    "vulkan": _generated.TRANSCRIBE_BACKEND_VULKAN,
    "cpu_accel": _generated.TRANSCRIBE_BACKEND_CPU_ACCEL,
    "cuda": _generated.TRANSCRIBE_BACKEND_CUDA,
}
_KV_TYPES = {
    "auto": _generated.TRANSCRIBE_KV_TYPE_AUTO,
    "f32": _generated.TRANSCRIBE_KV_TYPE_F32,
    "f16": _generated.TRANSCRIBE_KV_TYPE_F16,
}
_TASKS = {
    "transcribe": _generated.TRANSCRIBE_TASK_TRANSCRIBE,
    "translate": _generated.TRANSCRIBE_TASK_TRANSLATE,
}
_TIMESTAMPS = {
    "none": _generated.TRANSCRIBE_TIMESTAMPS_NONE,
    "auto": _generated.TRANSCRIBE_TIMESTAMPS_AUTO,
    "segment": _generated.TRANSCRIBE_TIMESTAMPS_SEGMENT,
    "word": _generated.TRANSCRIBE_TIMESTAMPS_WORD,
    "token": _generated.TRANSCRIBE_TIMESTAMPS_TOKEN,
}
_TIMESTAMP_NAMES = {v: k for k, v in _TIMESTAMPS.items()}
_COMMIT_POLICIES = {
    "auto": _generated.TRANSCRIBE_STREAM_COMMIT_AUTO,
    "on_finalize": _generated.TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE,
    "stable_prefix": _generated.TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX,
}
_STREAM_STATES = {
    _generated.TRANSCRIBE_STREAM_IDLE: "idle",
    _generated.TRANSCRIBE_STREAM_ACTIVE: "active",
    _generated.TRANSCRIBE_STREAM_FINISHED: "finished",
    _generated.TRANSCRIBE_STREAM_FAILED: "failed",
}
_EXT_SLOTS = {
    "run": _generated.TRANSCRIBE_EXT_SLOT_RUN,
    "stream": _generated.TRANSCRIBE_EXT_SLOT_STREAM,
}
_FEATURES = {
    "initial_prompt": _generated.TRANSCRIBE_FEATURE_INITIAL_PROMPT,
    "temperature_fallback": _generated.TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK,
    "long_form": _generated.TRANSCRIBE_FEATURE_LONG_FORM,
    "cancellation": _generated.TRANSCRIBE_FEATURE_CANCELLATION,
    "pnc": _generated.TRANSCRIBE_FEATURE_PNC,
    "itn": _generated.TRANSCRIBE_FEATURE_ITN,
}


def native_version() -> str:
    """Version string of the loaded native library, e.g. ``"0.0.1"``."""
    return _native_version


def native_commit() -> str:
    """Git commit the native library was built from, or ``"unknown"``."""
    return _lib.transcribe_version_commit().decode("ascii")


def library_path() -> str:
    """Filesystem path of the loaded native library."""
    return str(_lib_path)


def native_provider() -> "Union[str, None]":
    """Name of the installed provider package the native library was loaded
    from, or None for a dev-tree / ``TRANSCRIBE_LIBRARY`` load."""
    return selected_provider()


_log_handler = None
_log_trampoline = None


def _log_thunk(level, msg, _userdata):
    handler = _log_handler
    if handler is not None:
        try:
            handler(level, msg.decode("utf-8", "replace") if msg else "")
        except Exception:
            pass  # a handler error must never propagate back into C


def set_log_callback(handler) -> None:
    """Route native log messages to ``handler(level: int, message: str)``; pass
    None to silence.

    Install ONCE at startup, before loading models or creating threads (the 0.x
    contract for transcribe_log_set). The handler may be invoked from ggml worker
    threads, so it must be thread-safe — route to the ``logging`` module or a
    queue rather than doing heavy work inline. Levels mirror
    TRANSCRIBE_LOG_LEVEL_* (1=info, 2=warn, 3=error, 4=debug)."""
    global _log_handler, _log_trampoline
    _log_handler = handler
    if _log_trampoline is None:
        # Install the trampoline once; later calls just swap the Python handler
        # behind it, so transcribe_log_set is never re-invoked after threads run.
        _log_trampoline = _LOG_CFUNC(_log_thunk)
        _lib.transcribe_log_set(_log_trampoline, None)


def _status_string(status: int) -> str:
    return _lib.transcribe_status_string(status).decode("utf-8", "replace")


def _check(status: int, context: str = "") -> None:
    raise_for_status(status, _status_string(status), context)


def _decode(value) -> str:
    return value.decode("utf-8", "replace") if value else ""


def _enum(mapping: dict, key: str, what: str) -> int:
    try:
        return mapping[key]
    except KeyError:
        raise InvalidArgument(
            f"unknown {what} {key!r}; expected one of {sorted(mapping)}"
        ) from None


PCMLike = Union["ctypes.Array", bytes, bytearray, memoryview, Sequence[float]]


def _pcm_to_carray(pcm: PCMLike):
    """Copy *pcm* into a ``c_float`` array, returning ``(array, n_samples)``."""
    if isinstance(pcm, ctypes.Array) and pcm._type_ is ctypes.c_float:
        n = len(pcm)
        if n == 0:
            raise InvalidArgument("empty PCM buffer")
        return pcm, n

    try:
        mv = memoryview(pcm)
    except TypeError:
        seq = [float(x) for x in pcm]
        if not seq:
            raise InvalidArgument("empty PCM buffer")
        return (ctypes.c_float * len(seq))(*seq), len(seq)

    with mv:
        if not mv.contiguous:
            raise InvalidArgument("PCM buffer must be C-contiguous")
        if mv.format == "f":
            floats = mv
        elif mv.itemsize == 1:  # raw bytes: interpret as little-endian float32
            raw = mv.cast("B")
            if raw.nbytes % 4:
                raise InvalidArgument(
                    "raw PCM byte buffer length is not a multiple of 4 (float32)"
                )
            floats = raw.cast("f")
        else:
            raise InvalidArgument(
                f"PCM must be float32 (got buffer format {mv.format!r}); convert "
                "with array('f', ...) or numpy.asarray(x, dtype='float32')"
            )
        n = floats.shape[0]
        if n == 0:
            raise InvalidArgument("empty PCM buffer")
        return (ctypes.c_float * n).from_buffer_copy(floats), n


# --- result value objects -------------------------------------------------


@dataclass(frozen=True)
class Segment:
    text: str
    t0_ms: int
    t1_ms: int
    first_word: int
    n_words: int
    first_token: int
    n_tokens: int


@dataclass(frozen=True)
class Word:
    text: str
    t0_ms: int
    t1_ms: int
    seg_index: int
    first_token: int
    n_tokens: int


@dataclass(frozen=True)
class Token:
    text: str
    id: int
    p: float
    t0_ms: int
    t1_ms: int
    seg_index: int
    word_index: int


@dataclass(frozen=True)
class Timings:
    load_ms: float
    mel_ms: float
    encode_ms: float
    decode_ms: float


@dataclass(frozen=True)
class Capabilities:
    native_sample_rate: int
    languages: tuple
    max_timestamp_kind: str
    supports_language_detect: bool
    supports_translate: bool
    supports_streaming: bool
    supports_spec_decode: bool
    max_audio_ms: int


@dataclass(frozen=True)
class Result:
    """A fully materialized transcription. Holds no native pointers: every
    string and row was copied out of the session before this object was
    returned, so it stays valid after later runs."""

    text: str
    language: str
    timestamp_kind: str
    segments: tuple
    words: tuple
    tokens: tuple
    timings: Timings


@dataclass(frozen=True)
class StreamUpdate:
    """Per-call change metadata returned by Stream.feed()/finalize()."""

    result_changed: bool
    is_final: bool
    revision: int
    input_received_ms: int
    audio_committed_ms: int
    buffered_ms: int
    committed_changed: bool
    tentative_changed: bool


@dataclass(frozen=True)
class StreamText:
    """The UI-facing text views of an active stream (owned copies).

    ``committed`` is append-only and stable; ``tentative`` is the volatile
    suffix; ``display`` is what a UI should show. ``full`` is the raw model
    hypothesis (may not be ``committed + tentative`` after a revision)."""

    full: str
    committed: str
    tentative: str

    @property
    def display(self) -> str:
        return self.committed + self.tentative


# --- materialization helpers ----------------------------------------------
# Build owned value objects by copying out of the ctypes row structs. Shared by
# the single-result and per-utterance (batch) accessor paths.


def _segment_from(s) -> Segment:
    return Segment(
        text=_decode(s.text), t0_ms=s.t0_ms, t1_ms=s.t1_ms,
        first_word=s.first_word, n_words=s.n_words,
        first_token=s.first_token, n_tokens=s.n_tokens,
    )


def _word_from(w) -> Word:
    return Word(
        text=_decode(w.text), t0_ms=w.t0_ms, t1_ms=w.t1_ms,
        seg_index=w.seg_index, first_token=w.first_token, n_tokens=w.n_tokens,
    )


def _token_from(t) -> Token:
    return Token(
        text=_decode(t.text), id=t.id, p=t.p, t0_ms=t.t0_ms, t1_ms=t.t1_ms,
        seg_index=t.seg_index, word_index=t.word_index,
    )


def _timings_from(tm) -> Timings:
    return Timings(
        load_ms=tm.load_ms, mel_ms=tm.mel_ms,
        encode_ms=tm.encode_ms, decode_ms=tm.decode_ms,
    )


def _stream_update_from(u) -> StreamUpdate:
    return StreamUpdate(
        result_changed=bool(u.result_changed), is_final=bool(u.is_final),
        revision=u.revision, input_received_ms=u.input_received_ms,
        audio_committed_ms=u.audio_committed_ms, buffered_ms=u.buffered_ms,
        committed_changed=bool(u.committed_changed),
        tentative_changed=bool(u.tentative_changed),
    )


def _build_run_params(task, language, target_language, timestamps, keep_special_tags):
    params = _RunParams()
    _lib.transcribe_run_params_init(_byref(params))
    params.task = _enum(_TASKS, task, "task")
    params.timestamps = _enum(_TIMESTAMPS, timestamps, "timestamps")
    params.language = language.encode("utf-8") if language else None
    params.target_language = target_language.encode("utf-8") if target_language else None
    params.keep_special_tags = keep_special_tags
    return params


# --- family extensions ----------------------------------------------------


class FamilyExtension:
    """Base for typed family-extension options.

    A family extension carries family-specific knobs alongside a run or stream
    on a model that accepts them. Subclasses set the slot, kind, ctypes struct,
    and init function, and implement ``_apply()`` to copy their overrides onto
    the initialized struct (only fields the caller set override the family
    defaults). Pass an instance as ``family=`` to Session.run()/stream(); the
    model is probed first and a clear error is raised if it does not accept it.

    Adding another family is one subclass: point it at the generated
    ``transcribe_<family>_*_ext`` struct, its init function, and its
    ``TRANSCRIBE_EXT_KIND_*`` constant."""

    _slot: str = ""
    _kind: int = 0
    _struct = None
    _init: str = ""

    def _build(self):
        ext = self._struct()
        getattr(_lib, self._init)(_byref(ext))
        self._apply(ext)
        return ext

    def _apply(self, ext) -> None:
        raise NotImplementedError


class WhisperRunOptions(FamilyExtension):
    """Whisper run-extension options (run slot): initial prompt, temperature
    fallback, and decode thresholds. Only fields you set override the family
    defaults; the rest keep the values transcribe_whisper_run_ext_init stamps."""

    _slot = "run"
    _kind = _generated.TRANSCRIBE_EXT_KIND_WHISPER_RUN
    _struct = _generated.transcribe_whisper_run_ext
    _init = "transcribe_whisper_run_ext_init"

    def __init__(self, *, initial_prompt: "Union[str, None]" = None,
                 condition_on_prev_tokens: "Union[bool, None]" = None,
                 temperature: "Union[float, None]" = None,
                 temperature_inc: "Union[float, None]" = None,
                 compression_ratio_thold: "Union[float, None]" = None,
                 logprob_thold: "Union[float, None]" = None,
                 no_speech_thold: "Union[float, None]" = None,
                 max_prev_context_tokens: "Union[int, None]" = None,
                 seed: "Union[int, None]" = None,
                 max_initial_timestamp: "Union[float, None]" = None):
        self.initial_prompt = initial_prompt
        self.condition_on_prev_tokens = condition_on_prev_tokens
        self.temperature = temperature
        self.temperature_inc = temperature_inc
        self.compression_ratio_thold = compression_ratio_thold
        self.logprob_thold = logprob_thold
        self.no_speech_thold = no_speech_thold
        self.max_prev_context_tokens = max_prev_context_tokens
        self.seed = seed
        self.max_initial_timestamp = max_initial_timestamp

    def _apply(self, ext) -> None:
        if self.initial_prompt is not None:
            ext.initial_prompt = self.initial_prompt.encode("utf-8")
        if self.condition_on_prev_tokens is not None:
            ext.condition_on_prev_tokens = self.condition_on_prev_tokens
        if self.temperature is not None:
            ext.temperature = self.temperature
        if self.temperature_inc is not None:
            ext.temperature_inc = self.temperature_inc
        if self.compression_ratio_thold is not None:
            ext.compression_ratio_thold = self.compression_ratio_thold
        if self.logprob_thold is not None:
            ext.logprob_thold = self.logprob_thold
        if self.no_speech_thold is not None:
            ext.no_speech_thold = self.no_speech_thold
        if self.max_prev_context_tokens is not None:
            ext.max_prev_context_tokens = self.max_prev_context_tokens
        if self.seed is not None:
            ext.seed = self.seed
        if self.max_initial_timestamp is not None:
            ext.max_initial_timestamp = self.max_initial_timestamp


class MoonshineStreamingOptions(FamilyExtension):
    """Moonshine-streaming stream-extension options (stream slot)."""

    _slot = "stream"
    _kind = _generated.TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM
    _struct = _generated.transcribe_moonshine_streaming_stream_ext
    _init = "transcribe_moonshine_streaming_stream_ext_init"

    def __init__(self, *, min_decode_interval_ms: "Union[int, None]" = None):
        self.min_decode_interval_ms = min_decode_interval_ms

    def _apply(self, ext) -> None:
        if self.min_decode_interval_ms is not None:
            ext.min_decode_interval_ms = self.min_decode_interval_ms


class ParakeetStreamOptions(FamilyExtension):
    """Parakeet cache-aware streaming options (stream slot)."""

    _slot = "stream"
    _kind = _generated.TRANSCRIBE_EXT_KIND_PARAKEET_STREAM
    _struct = _generated.transcribe_parakeet_stream_ext
    _init = "transcribe_parakeet_stream_ext_init"

    def __init__(self, *, att_context_right: "Union[int, None]" = None):
        self.att_context_right = att_context_right

    def _apply(self, ext) -> None:
        if self.att_context_right is not None:
            ext.att_context_right = self.att_context_right


class ParakeetBufferedStreamOptions(FamilyExtension):
    """Parakeet chunked-attention buffered streaming options (stream slot).
    A field left at None keeps the family default (the C sentinel -1)."""

    _slot = "stream"
    _kind = _generated.TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM
    _struct = _generated.transcribe_parakeet_buffered_stream_ext
    _init = "transcribe_parakeet_buffered_stream_ext_init"

    def __init__(self, *, left_ms: "Union[int, None]" = None,
                 chunk_ms: "Union[int, None]" = None,
                 right_ms: "Union[int, None]" = None):
        self.left_ms = left_ms
        self.chunk_ms = chunk_ms
        self.right_ms = right_ms

    def _apply(self, ext) -> None:
        if self.left_ms is not None:
            ext.left_ms = self.left_ms
        if self.chunk_ms is not None:
            ext.chunk_ms = self.chunk_ms
        if self.right_ms is not None:
            ext.right_ms = self.right_ms


class VoxtralRealtimeStreamOptions(FamilyExtension):
    """Voxtral-realtime streaming options (stream slot)."""

    _slot = "stream"
    _kind = _generated.TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM
    _struct = _generated.transcribe_voxtral_realtime_stream_ext
    _init = "transcribe_voxtral_realtime_stream_ext_init"

    def __init__(self, *, num_delay_tokens: "Union[int, None]" = None,
                 min_decode_interval_ms: "Union[int, None]" = None):
        self.num_delay_tokens = num_delay_tokens
        self.min_decode_interval_ms = min_decode_interval_ms

    def _apply(self, ext) -> None:
        if self.num_delay_tokens is not None:
            ext.num_delay_tokens = self.num_delay_tokens
        if self.min_decode_interval_ms is not None:
            ext.min_decode_interval_ms = self.min_decode_interval_ms


# --- high-level handles ---------------------------------------------------


class Model:
    """A loaded model. Thread-safe to share across sessions; must outlive them."""

    def __init__(self, path: "Union[str, os.PathLike]", *,
                 backend: Backend = "auto", gpu_device: int = 0):
        params = _ModelLoadParams()
        _lib.transcribe_model_load_params_init(_byref(params))
        params.backend = _enum(_BACKENDS, backend, "backend")
        params.gpu_device = gpu_device

        handle = ctypes.c_void_p()
        status = _lib.transcribe_model_load_file(
            os.fspath(path).encode("utf-8"), _byref(params), _byref(handle)
        )
        _check(status, f"loading model {os.fspath(path)!r}")
        if not handle.value:
            raise ModelLoadError(f"model load returned a null handle for {path!r}")
        self._handle = handle

    @property
    def _h(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise TranscribeError("model is closed")
        return self._handle

    @property
    def arch(self) -> str:
        return _decode(_lib.transcribe_model_arch_string(self._h))

    @property
    def variant(self) -> str:
        return _decode(_lib.transcribe_model_variant_string(self._h))

    @property
    def backend(self) -> str:
        return _decode(_lib.transcribe_model_backend(self._h))

    @property
    def capabilities(self) -> Capabilities:
        caps = _Capabilities()
        _lib.transcribe_capabilities_init(_byref(caps))
        _check(
            _lib.transcribe_model_get_capabilities(self._h, _byref(caps)),
            "reading capabilities",
        )
        languages = []
        if caps.languages and caps.n_languages > 0:
            for i in range(caps.n_languages):
                languages.append(_decode(caps.languages[i]))
        return Capabilities(
            native_sample_rate=caps.native_sample_rate,
            languages=tuple(languages),
            max_timestamp_kind=_TIMESTAMP_NAMES.get(caps.max_timestamp_kind, "unknown"),
            supports_language_detect=bool(caps.supports_language_detect),
            supports_translate=bool(caps.supports_translate),
            supports_streaming=bool(caps.supports_streaming),
            supports_spec_decode=bool(caps.supports_spec_decode),
            max_audio_ms=caps.max_audio_ms,
        )

    def supports(self, feature: Feature) -> bool:
        """Whether the model exposes a behavioral feature (initial prompt,
        temperature fallback, long-form, cancellation, pnc, itn)."""
        return bool(_lib.transcribe_model_supports(
            self._h, _enum(_FEATURES, feature, "feature")))

    def accepts(self, options: "FamilyExtension") -> bool:
        """Whether the model accepts a family extension on its slot."""
        return bool(_lib.transcribe_model_accepts_ext_kind(
            self._h, _EXT_SLOTS[options._slot], options._kind))

    def session(self, *, n_threads: int = 0, kv_type: KVType = "auto",
                n_ctx: int = 0) -> "Session":
        return Session(self, n_threads=n_threads, kv_type=kv_type, n_ctx=n_ctx)

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            _lib.transcribe_model_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Session:
    """A single transcription context bound to one Model. Not thread-safe; use
    one per thread (sharing the Model is fine)."""

    def __init__(self, model: Model, *, n_threads: int = 0, kv_type: KVType = "auto",
                 n_ctx: int = 0):
        self._model = model  # keep the model alive for the session's lifetime
        params = _SessionParams()
        _lib.transcribe_session_params_init(_byref(params))
        params.n_threads = n_threads
        params.kv_type = _enum(_KV_TYPES, kv_type, "kv_type")
        params.n_ctx = n_ctx

        handle = ctypes.c_void_p()
        status = _lib.transcribe_session_init(model._h, _byref(params), _byref(handle))
        _check(status, "opening session")
        if not handle.value:
            raise TranscribeError("session init returned a null handle")
        self._handle = handle

        # Cancellation: an abort callback polled at chunk/decode boundaries
        # returns True once cancel() is requested, so another thread can abort an
        # in-flight run/stream. Bind the Event (not self) into the trampoline to
        # avoid a reference cycle that would defer __del__.
        self._cancel = threading.Event()
        _event = self._cancel
        self._abort_trampoline = _ABORT_CFUNC(lambda _ud: _event.is_set())
        _lib.transcribe_set_abort_callback(self._handle, self._abort_trampoline, None)

    @property
    def _h(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise TranscribeError("session is closed")
        return self._handle

    def _resolve_family(self, family: "FamilyExtension", slot: str):
        """Validate + probe a family extension and return its built ctypes
        struct (which the caller keeps alive and points params.family at)."""
        if not isinstance(family, FamilyExtension):
            raise InvalidArgument("family must be a FamilyExtension instance")
        if family._slot != slot:
            raise InvalidArgument(
                f"{type(family).__name__} is a {family._slot!r}-slot extension and "
                f"cannot be used on the {slot!r} slot")
        if not _lib.transcribe_model_accepts_ext_kind(
                self._model._h, _EXT_SLOTS[slot], family._kind):
            raise UnsupportedRequest(
                f"this model does not accept {type(family).__name__} on the "
                f"{slot!r} slot")
        return family._build()

    def run(self, pcm: PCMLike, *, task: Task = "transcribe",
            language: "Union[str, None]" = None,
            target_language: "Union[str, None]" = None,
            timestamps: Timestamps = "none",
            keep_special_tags: bool = False,
            family: "Union[FamilyExtension, None]" = None) -> Result:
        """Transcribe 16 kHz mono float32 PCM and return a materialized Result.

        ``family`` is an optional family-specific extension (e.g.
        WhisperRunOptions) carrying per-run knobs for models that accept it."""
        self._cancel.clear()
        array, n_samples = _pcm_to_carray(pcm)
        params = _build_run_params(task, language, target_language, timestamps,
                                   keep_special_tags)
        ext = self._resolve_family(family, "run") if family is not None else None
        if ext is not None:
            params.family = ctypes.cast(
                _byref(ext), ctypes.POINTER(_generated.transcribe_ext))
        _check(_lib.transcribe_run(self._h, array, n_samples, _byref(params)),
               "transcribe_run")
        return self._materialize()

    def run_batch(self, pcms: "Sequence[PCMLike]", *, task: Task = "transcribe",
                  language: "Union[str, None]" = None,
                  target_language: "Union[str, None]" = None,
                  timestamps: Timestamps = "none",
                  keep_special_tags: bool = False) -> "list":
        """Transcribe several utterances in one dispatch — one Result each.

        Families with a batched compute path process every utterance in a single
        device dispatch (≈2x throughput on an underused GPU); others fall back to
        running them in turn, so every model accepts this. Raises if any single
        utterance failed (its index is in the message)."""
        self._cancel.clear()
        pcms = list(pcms)
        if not pcms:
            raise InvalidArgument("run_batch requires at least one PCM buffer")

        arrays = []  # keep the per-utterance float arrays alive across the call
        ptrs = (ctypes.POINTER(ctypes.c_float) * len(pcms))()
        counts = (ctypes.c_int * len(pcms))()
        for k, pcm in enumerate(pcms):
            arr, n = _pcm_to_carray(pcm)
            arrays.append(arr)
            ptrs[k] = ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))
            counts[k] = n

        params = _build_run_params(task, language, target_language, timestamps,
                                   keep_special_tags)
        _check(
            _lib.transcribe_run_batch(self._h, ptrs, counts, len(pcms), _byref(params)),
            "transcribe_run_batch",
        )

        results = []
        for i in range(_lib.transcribe_batch_n_results(self._h)):
            _check(_lib.transcribe_batch_status(self._h, i), f"utterance {i} in batch")
            results.append(self._materialize(i))
        return results

    def stream(self, *, task: Task = "transcribe", language: "Union[str, None]" = None,
               target_language: "Union[str, None]" = None, timestamps: Timestamps = "none",
               keep_special_tags: bool = False, commit_policy: CommitPolicy = "auto",
               stable_prefix_agreement_n: int = 0,
               family: "Union[FamilyExtension, None]" = None) -> "Stream":
        """Begin streaming on this session and return a Stream to feed audio to.

        Requires a model whose capabilities advertise ``supports_streaming``;
        otherwise raises NotImplementedByModel. ``family`` is an optional
        family-specific stream extension (e.g. MoonshineStreamingOptions). The
        session is single-threaded and runs at most one stream at a time. Use
        the Stream as a context manager so it is reset when you are done."""
        self._cancel.clear()
        run_params = _build_run_params(task, language, target_language, timestamps,
                                       keep_special_tags)
        sp = _StreamParams()
        _lib.transcribe_stream_params_init(_byref(sp))
        sp.commit_policy = _enum(_COMMIT_POLICIES, commit_policy, "commit_policy")
        sp.stable_prefix_agreement_n = stable_prefix_agreement_n
        ext = self._resolve_family(family, "stream") if family is not None else None
        if ext is not None:
            sp.family = ctypes.cast(
                _byref(ext), ctypes.POINTER(_generated.transcribe_ext))
        _check(
            _lib.transcribe_stream_begin(self._h, _byref(run_params), _byref(sp)),
            "transcribe_stream_begin",
        )
        return Stream(self)

    def _materialize(self, utt: "Union[int, None]" = None) -> Result:
        """Copy out one result. utt is None for the single-result accessors, or
        an utterance index for the batch accessors (index 0 aliases the single
        result after a plain run, so both paths share this code)."""
        h = self._h
        if utt is None:
            n_seg = lambda: _lib.transcribe_n_segments(h)
            get_seg = lambda j, out: _lib.transcribe_get_segment(h, j, out)
            n_word = lambda: _lib.transcribe_n_words(h)
            get_word = lambda j, out: _lib.transcribe_get_word(h, j, out)
            n_tok = lambda: _lib.transcribe_n_tokens(h)
            get_tok = lambda j, out: _lib.transcribe_get_token(h, j, out)
            full_text = _lib.transcribe_full_text(h)
            language = _lib.transcribe_detected_language(h)
            kind = _lib.transcribe_returned_timestamp_kind(h)
            get_tim = lambda out: _lib.transcribe_get_timings(h, out)
        else:
            n_seg = lambda: _lib.transcribe_batch_n_segments(h, utt)
            get_seg = lambda j, out: _lib.transcribe_batch_get_segment(h, utt, j, out)
            n_word = lambda: _lib.transcribe_batch_n_words(h, utt)
            get_word = lambda j, out: _lib.transcribe_batch_get_word(h, utt, j, out)
            n_tok = lambda: _lib.transcribe_batch_n_tokens(h, utt)
            get_tok = lambda j, out: _lib.transcribe_batch_get_token(h, utt, j, out)
            full_text = _lib.transcribe_batch_full_text(h, utt)
            language = _lib.transcribe_batch_detected_language(h, utt)
            kind = _lib.transcribe_batch_returned_timestamp_kind(h, utt)
            get_tim = lambda out: _lib.transcribe_batch_get_timings(h, utt, out)

        segments = []
        for j in range(n_seg()):
            s = _Segment()
            _lib.transcribe_segment_init(_byref(s))
            get_seg(j, _byref(s))
            segments.append(_segment_from(s))

        words = []
        for j in range(n_word()):
            w = _Word()
            _lib.transcribe_word_init(_byref(w))
            get_word(j, _byref(w))
            words.append(_word_from(w))

        tokens = []
        for j in range(n_tok()):
            tok = _Token()
            _lib.transcribe_token_init(_byref(tok))
            get_tok(j, _byref(tok))
            tokens.append(_token_from(tok))

        tm = _Timings()
        _lib.transcribe_timings_init(_byref(tm))
        get_tim(_byref(tm))

        return Result(
            text=_decode(full_text),
            language=_decode(language),
            timestamp_kind=_TIMESTAMP_NAMES.get(kind, "unknown"),
            segments=tuple(segments),
            words=tuple(words),
            tokens=tuple(tokens),
            timings=_timings_from(tm),
        )

    def cancel(self) -> None:
        """Request cancellation of an in-flight run/stream from another thread.
        The active call aborts at the next chunk/decode boundary and raises;
        ``was_aborted`` then reports True. The flag is cleared at the start of
        the next run/stream."""
        self._cancel.set()

    @property
    def was_aborted(self) -> bool:
        """True if the most recent call was ended by cancel()."""
        return bool(_lib.transcribe_was_aborted(self._h))

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            _lib.transcribe_session_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Stream:
    """An active streaming transcription on a Session.

    Feed audio in chunks with ``feed()``, read the committed/tentative text with
    ``text()``, and call ``finalize()`` when the audio ends. The session is
    returned to idle on context-manager exit (or ``reset()``)."""

    def __init__(self, session: Session):
        self._session = session  # keep the session (and its model) alive
        self._active = True

    @property
    def _h(self) -> ctypes.c_void_p:
        if not self._active:
            raise TranscribeError("stream has been reset")
        return self._session._h

    def feed(self, pcm: PCMLike) -> StreamUpdate:
        """Feed a chunk of 16 kHz mono float32 PCM; returns change metadata."""
        array, n_samples = _pcm_to_carray(pcm)
        update = _StreamUpdate()
        _lib.transcribe_stream_update_init(_byref(update))
        _check(_lib.transcribe_stream_feed(self._h, array, n_samples, _byref(update)),
               "transcribe_stream_feed")
        return _stream_update_from(update)

    def finalize(self) -> StreamUpdate:
        """Signal end of audio and flush the final hypothesis."""
        update = _StreamUpdate()
        _lib.transcribe_stream_update_init(_byref(update))
        _check(_lib.transcribe_stream_finalize(self._h, _byref(update)),
               "transcribe_stream_finalize")
        return _stream_update_from(update)

    def text(self) -> StreamText:
        """Current committed / tentative / full text views (owned copies)."""
        txt = _StreamText()
        _lib.transcribe_stream_text_init(_byref(txt))
        _check(_lib.transcribe_stream_get_text(self._h, _byref(txt)),
               "transcribe_stream_get_text")
        return StreamText(
            full=_decode(txt.full_text),
            committed=_decode(txt.committed_text),
            tentative=_decode(txt.tentative_text),
        )

    @property
    def state(self) -> str:
        """``"idle"`` / ``"active"`` / ``"finished"`` / ``"failed"``."""
        return _STREAM_STATES.get(_lib.transcribe_stream_get_state(self._h), "unknown")

    @property
    def revision(self) -> int:
        return _lib.transcribe_stream_revision(self._h)

    def reset(self) -> None:
        """Return the session to idle, discarding stream state. Idempotent."""
        if self._active:
            _lib.transcribe_stream_reset(self._session._h)
            self._active = False

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc) -> None:
        self.reset()


def transcribe(
    model: "Union[Model, str, os.PathLike]",
    pcm: PCMLike,
    *,
    backend: Backend = "auto",
    gpu_device: int = 0,
    n_threads: int = 0,
    kv_type: KVType = "auto",
    n_ctx: int = 0,
    task: Task = "transcribe",
    language: "Union[str, None]" = None,
    target_language: "Union[str, None]" = None,
    timestamps: Timestamps = "none",
    keep_special_tags: bool = False,
) -> Result:
    """Transcribe *pcm* in one call and return a materialized Result.

    *model* may be a path (loaded and freed within this call) or an existing
    Model (reused and left open). Loading a model is not free, so to transcribe
    many clips keep a Model and call ``model.session().run(...)`` yourself; this
    helper is for the one-shot case. ``backend`` / ``gpu_device`` apply only when
    *model* is a path — they are ignored when an already-loaded Model is passed.
    """
    session_opts = dict(n_threads=n_threads, kv_type=kv_type, n_ctx=n_ctx)
    run_opts = dict(task=task, language=language, target_language=target_language,
                    timestamps=timestamps, keep_special_tags=keep_special_tags)

    if isinstance(model, Model):
        with model.session(**session_opts) as session:
            return session.run(pcm, **run_opts)

    with Model(model, backend=backend, gpu_device=gpu_device) as owned:
        with owned.session(**session_opts) as session:
            return session.run(pcm, **run_opts)
