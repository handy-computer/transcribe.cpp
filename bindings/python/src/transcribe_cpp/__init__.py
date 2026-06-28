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
import weakref
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

from . import _abi, _generated
from ._library import _base_version, artifact_dir, load_library, selected_provider
from .errors import (
    AbiError,
    Aborted,
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
    exception_for_status,
    raise_for_status,
)

__version__ = "0.0.11"

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
    "SessionLimits",
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
    "Aborted",
    "InputTooLong",
    "OutputTruncated",
    "native_version",
    "native_commit",
    "library_path",
    "native_provider",
    "BackendDevice",
    "backends",
    "backend_available",
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

# Load ggml backend modules from the artifact directory (package-local; a
# no-op for static/compiled-in builds). In a dynamic-backend build this is
# what registers CPU/Vulkan/... devices, and a Vulkan module on a machine
# without Vulkan simply fails to load while CPU keeps working. Raises only
# when the process ends up with ZERO compute devices.
_artifact = artifact_dir()
if _artifact is not None:
    _status = _lib.transcribe_init_backends(os.fspath(_artifact).encode("utf-8"))
    if _status != 0:
        raise BackendError(
            f"no usable compute backend: transcribe_init_backends({_artifact}) "
            f"reported {_lib.transcribe_status_string(_status).decode('utf-8', 'replace')}. "
            "In a dynamic-backend build the ggml backend modules must sit next "
            "to the native library."
        )

_byref = ctypes.byref

# transcribe_tokenize returns INT32_MIN to signal "this model has no tokenizer
# encode path" (distinct from the negative grow-buffer signal).
_INT32_MIN = -(2 ** 31)

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


def native_provider() -> str | None:
    """Name of the installed provider package the native library was loaded
    from, or None for a dev-tree / ``TRANSCRIBE_LIBRARY`` load."""
    return selected_provider()


_DEVICE_TYPE_NAMES = {
    _generated.TRANSCRIBE_DEVICE_TYPE_CPU: "cpu",
    _generated.TRANSCRIBE_DEVICE_TYPE_GPU: "gpu",
    _generated.TRANSCRIBE_DEVICE_TYPE_IGPU: "igpu",
    _generated.TRANSCRIBE_DEVICE_TYPE_ACCEL: "accel",
}


@dataclass(frozen=True)
class BackendDevice:
    """One registered compute device (owned copies of the C strings)."""

    name: str
    description: str
    kind: str  # "cpu" | "accel" | "metal" | "vulkan" | "cuda" | "sycl" | "gpu" | "unknown"
    # Vendor-agnostic class: "cpu" | "gpu" | "igpu" | "accel", or "unknown" for a
    # value reported by a runtime newer than this binding (tell such devices
    # apart by device_id / name, not by this axis).
    device_type: str
    device_id: Optional[str]  # stable hw id (PCI bus id), or None (e.g. Metal)
    memory_total: int  # reported capacity in bytes, or 0 if unreported
    # Available bytes — a SNAPSHOT at query time, or 0 if unreported. Re-query
    # (via backends() or Model.device) to refresh; backend-defined and not
    # comparable across device kinds.
    memory_free: int
    # Registry index of this device — the value to pass as ``Model(...,
    # gpu_device=index)`` to select it (0 selects the auto / first device).
    # None when the device came from Model.device, since the underlying
    # transcribe_model_get_device() does not expose an index; correlate such a
    # device back to backends() by device_id / name instead. The index is
    # order-dependent and not stable across driver updates or hosts.
    index: Optional[int] = None


def _backend_device_from_raw(dev, index: Optional[int] = None) -> BackendDevice:
    """Build a BackendDevice from a library-filled transcribe_backend_device."""
    return BackendDevice(
        name=_decode(dev.name),
        description=_decode(dev.description),
        kind=_decode(dev.kind),
        device_type=_DEVICE_TYPE_NAMES.get(dev.device_type, "unknown"),
        device_id=_decode(dev.device_id) if dev.device_id else None,
        memory_total=int(dev.memory_total),
        memory_free=int(dev.memory_free),
        index=index,
    )


def backends() -> list[BackendDevice]:
    """The compute devices registered with the native runtime — what the
    process can actually run on, after backend-module loading and graceful
    degradation (e.g. a Vulkan module skipped on a machine without Vulkan).

    Each device's ``memory_free`` is live as of the call; call again to poll
    a device's available memory over time."""
    devices = []
    for i in range(_lib.transcribe_backend_device_count()):
        dev = _generated.transcribe_backend_device()
        _lib.transcribe_backend_device_init(_byref(dev))
        _check(_lib.transcribe_get_backend_device(i, _byref(dev)),
               f"reading backend device {i}")
        devices.append(_backend_device_from_raw(dev, index=i))
    return devices


def backend_available(backend: Backend) -> bool:
    """Whether ``Model(..., backend=...)`` can be satisfied on this machine —
    the probe that turns ``backend="vulkan"`` without Vulkan into a clear
    answer before any model load."""
    return bool(_lib.transcribe_backend_available(
        _enum(_BACKENDS, backend, "backend")))


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
        if mv.ndim != 1:
            # A (frames, channels) stereo array is the classic mistake here.
            # Accepting it and reading shape[0] samples would transcribe
            # silent garbage; reject loudly instead.
            raise InvalidArgument(
                f"PCM buffer must be 1-D (got shape {mv.shape}); downmix "
                "multichannel audio to mono and flatten, e.g. "
                "audio.mean(axis=1).astype('float32') for a (frames, "
                "channels) numpy array"
            )
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
    languages: tuple[str, ...]
    max_timestamp_kind: str
    supports_language_detect: bool
    supports_translate: bool
    supports_streaming: bool
    supports_spec_decode: bool
    max_audio_ms: int
    translate_target_languages: tuple[str, ...]


@dataclass(frozen=True)
class SessionLimits:
    """Effective per-session limits (model bound narrowed by session params).

    ``effective_max_audio_ms`` is the input ceiling THIS session enforces —
    unlike ``Capabilities.max_audio_ms`` it reflects a lowered ``n_ctx``.
    0 means unbounded, matching the capabilities convention. Check it to
    size input before a run instead of discovering ``InputTooLong``."""

    effective_n_ctx: int
    effective_max_audio_ms: int
    max_kv_bytes: int


@dataclass(frozen=True)
class Result:
    """A fully materialized transcription. Holds no native pointers: every
    string and row was copied out of the session before this object was
    returned, so it stays valid after later runs."""

    text: str
    language: str
    timestamp_kind: str
    segments: tuple[Segment, ...]
    words: tuple[Word, ...]
    tokens: tuple[Token, ...]
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


def _build_run_params(task, language, target_language, timestamps,
                      keep_special_tags, spec_k_drafts):
    if not isinstance(spec_k_drafts, int) or spec_k_drafts < -1:
        raise InvalidArgument(
            f"spec_k_drafts must be -1 (family default), 0 (disabled), or a "
            f"positive draft length; got {spec_k_drafts!r}"
        )
    params = _RunParams()
    _lib.transcribe_run_params_init(_byref(params))
    params.task = _enum(_TASKS, task, "task")
    params.timestamps = _enum(_TIMESTAMPS, timestamps, "timestamps")
    params.language = language.encode("utf-8") if language else None
    params.target_language = target_language.encode("utf-8") if target_language else None
    params.keep_special_tags = keep_special_tags
    params.spec_k_drafts = spec_k_drafts
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

    def __init__(self, *, initial_prompt: str | None = None,
                 condition_on_prev_tokens: bool | None = None,
                 temperature: float | None = None,
                 temperature_inc: float | None = None,
                 compression_ratio_thold: float | None = None,
                 logprob_thold: float | None = None,
                 no_speech_thold: float | None = None,
                 max_prev_context_tokens: int | None = None,
                 seed: int | None = None,
                 max_initial_timestamp: float | None = None):
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

    def __init__(self, *, min_decode_interval_ms: int | None = None):
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

    def __init__(self, *, att_context_right: int | None = None):
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

    def __init__(self, *, left_ms: int | None = None,
                 chunk_ms: int | None = None,
                 right_ms: int | None = None):
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

    def __init__(self, *, num_delay_tokens: int | None = None,
                 min_decode_interval_ms: int | None = None):
        self.num_delay_tokens = num_delay_tokens
        self.min_decode_interval_ms = min_decode_interval_ms

    def _apply(self, ext) -> None:
        if self.num_delay_tokens is not None:
            ext.num_delay_tokens = self.num_delay_tokens
        if self.min_decode_interval_ms is not None:
            ext.min_decode_interval_ms = self.min_decode_interval_ms


# --- high-level handles ---------------------------------------------------


class Model:
    """A loaded model. Sharing it across threads for queries and session
    creation is safe; it must outlive its sessions.

    Known 0.x limitation: at most one run/stream may be IN FLIGHT across all
    sessions of a model at a time — sessions share the model's compute
    backend, so overlapping runs race. Run sessions serially (a pool behind
    a lock is fine), or load one Model per worker for true parallelism.

    ``backend="auto"`` (the default) picks the best available device. The
    ``TRANSCRIBE_BACKEND`` environment variable overrides that *default* —
    the escape hatch for machines whose best-ranked device misbehaves (e.g.
    CI's paravirtualized Metal, a broken GPU driver) without touching code.
    An explicit ``backend=`` argument always wins over the environment.
    """

    def __init__(self, path: str | os.PathLike, *,
                 backend: Backend = "auto", gpu_device: int = 0):
        # Live sessions, tracked weakly: close() must free them before the
        # model, because transcribe_model_free is only valid once every
        # derived session is gone (use-after-free otherwise). Created before
        # the load call so the failure path of __del__ finds it.
        self._sessions = weakref.WeakSet()
        backend_source = "backend"
        if backend == "auto" and os.environ.get("TRANSCRIBE_BACKEND"):
            backend = os.environ["TRANSCRIBE_BACKEND"]  # type: ignore[assignment]
            backend_source = "backend (from TRANSCRIBE_BACKEND)"
        params = _ModelLoadParams()
        _lib.transcribe_model_load_params_init(_byref(params))
        params.backend = _enum(_BACKENDS, backend, backend_source)
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
    def device(self) -> BackendDevice:
        """The compute device this model is running on. ``memory_free`` is a
        live snapshot, so read this again to poll how much device memory is
        left after the model loaded. Raises if the model has no resolved
        compute device."""
        dev = _generated.transcribe_backend_device()
        _lib.transcribe_backend_device_init(_byref(dev))
        _check(_lib.transcribe_model_get_device(self._h, _byref(dev)),
               "model_get_device")
        return _backend_device_from_raw(dev)

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
        translate_targets = []
        if caps.translate_target_languages and caps.n_translate_target_languages > 0:
            for i in range(caps.n_translate_target_languages):
                translate_targets.append(_decode(caps.translate_target_languages[i]))
        return Capabilities(
            native_sample_rate=caps.native_sample_rate,
            languages=tuple(languages),
            max_timestamp_kind=_TIMESTAMP_NAMES.get(caps.max_timestamp_kind, "unknown"),
            supports_language_detect=bool(caps.supports_language_detect),
            supports_translate=bool(caps.supports_translate),
            supports_streaming=bool(caps.supports_streaming),
            supports_spec_decode=bool(caps.supports_spec_decode),
            max_audio_ms=caps.max_audio_ms,
            translate_target_languages=tuple(translate_targets),
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

    def tokenize(self, text: str) -> list[int]:
        """Tokenize plain UTF-8 ``text`` into the model's vocabulary ids (no
        BOS/EOS, no special tags). Raises :class:`NotImplementedByModel` for
        families whose tokenizer has no encode path (e.g. SentencePiece today).
        """
        data = text.encode("utf-8")
        cap = max(len(data), 16)
        while True:
            buf = (ctypes.c_int32 * cap)()
            n = _lib.transcribe_tokenize(self._h, data, buf, cap)
            if n == _INT32_MIN:
                raise NotImplementedByModel(
                    "model tokenizer has no encode path")
            if n < 0:  # buffer too small: library asked for -n slots, retry
                cap = -n
                continue
            return [buf[i] for i in range(n)]

    def session(self, *, n_threads: int = 0, kv_type: KVType = "auto",
                n_ctx: int = 0) -> "Session":
        return Session(self, n_threads=n_threads, kv_type=kv_type, n_ctx=n_ctx)

    def close(self) -> None:
        """Free the model. Any session still open on it is closed first —
        the C contract forbids freeing a model before its sessions, so this
        keeps explicit close()/context-manager exit safe in any order."""
        if getattr(self, "_handle", None) is not None:
            for session in list(getattr(self, "_sessions", ()) or ()):
                session.close()
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
    """A single transcription context bound to one Model. Not thread-safe.

    Sessions of one model must also not RUN concurrently with each other in
    0.x (they share the model's compute backend — see the Model docstring);
    interleave or serialize their runs, or use one Model per worker."""

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

        # Registered last, once the session is fully constructed: Model.close()
        # closes any session still tracked here before freeing the model.
        model._sessions.add(self)

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
            language: str | None = None,
            target_language: str | None = None,
            timestamps: Timestamps = "auto",
            keep_special_tags: bool = False,
            spec_k_drafts: int = -1,
            family: FamilyExtension | None = None) -> Result:
        """Transcribe 16 kHz mono float32 PCM and return a materialized Result.

        ``family`` is an optional family-specific extension (e.g.
        WhisperRunOptions) carrying per-run knobs for models that accept it.
        ``spec_k_drafts`` tunes speculative decoding on models whose
        capabilities advertise ``supports_spec_decode`` (-1 = family default,
        0 = disabled, >0 = draft length; silently ignored elsewhere).

        On ``Aborted`` (via :meth:`cancel`) and ``OutputTruncated`` the
        partial transcript is preserved and attached to the exception as
        ``partial_result``."""
        self._cancel.clear()
        array, n_samples = _pcm_to_carray(pcm)
        params = _build_run_params(task, language, target_language, timestamps,
                                   keep_special_tags, spec_k_drafts)
        ext = self._resolve_family(family, "run") if family is not None else None
        if ext is not None:
            params.family = ctypes.cast(
                _byref(ext), ctypes.POINTER(_generated.transcribe_ext))
        try:
            _check(_lib.transcribe_run(self._h, array, n_samples, _byref(params)),
                   "transcribe_run")
        except (Aborted, OutputTruncated) as exc:
            # The C API preserves the partial transcript on the session for
            # exactly these two statuses; surface it rather than discard it.
            exc.partial_result = self._materialize()
            raise
        return self._materialize()

    def run_batch(self, pcms: Sequence[PCMLike], *, task: Task = "transcribe",
                  language: str | None = None,
                  target_language: str | None = None,
                  timestamps: Timestamps = "auto",
                  keep_special_tags: bool = False,
                  spec_k_drafts: int = -1,
                  family: FamilyExtension | None = None,
                  return_exceptions: bool = False) -> list[Result | TranscribeError]:
        """Transcribe several utterances in one dispatch — one Result each.

        Families with a batched compute path process every utterance in a single
        device dispatch (≈2x throughput on an underused GPU); others fall back to
        running them in turn, so every model accepts this.

        Failure is PER-UTTERANCE (the C API reports one status per clip; a
        :meth:`cancel` surfaces at the batch level but the native side pads
        the per-utterance result set, so it is folded into the same view).
        Default: raise the first failing utterance's exception, with
        ``utterance_index`` set, ``partial_result`` attached where the
        native layer preserved a partial transcript for that slot (None
        otherwise), and ``batch_results`` carrying the full per-utterance
        view (``Result`` or ``TranscribeError`` each) so completed work is
        never discarded. With ``return_exceptions=True`` no exception is
        raised for utterance failures and that mixed list is returned
        directly (the ``asyncio.gather`` convention)."""
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
                                   keep_special_tags, spec_k_drafts)
        ext = self._resolve_family(family, "run") if family is not None else None
        if ext is not None:
            params.family = ctypes.cast(
                _byref(ext), ctypes.POINTER(_generated.transcribe_ext))
        batch_abort = None
        try:
            _check(
                _lib.transcribe_run_batch(self._h, ptrs, counts, len(pcms), _byref(params)),
                "transcribe_run_batch",
            )
        except Aborted as exc:
            # Cancellation surfaces at the BATCH level, but the native side
            # pads the per-utterance result set so clips completed before the
            # abort survive. Fall through to the per-utterance loop, which
            # restores utterance context instead of discarding that work.
            batch_abort = exc

        out: list = []
        first_exc = None
        for i in range(_lib.transcribe_batch_n_results(self._h)):
            status = _lib.transcribe_batch_status(self._h, i)
            if status == 0:
                out.append(self._materialize(i))
                continue
            exc = exception_for_status(status, _status_string(status),
                                       f"utterance {i} in batch")
            exc.utterance_index = i
            if isinstance(exc, (Aborted, OutputTruncated)):
                # Attach the partial transcript only when the native layer
                # actually snapshotted one for this slot — some batch paths
                # record failed slots as empty, and None is more honest than
                # a confidently empty Result.
                partial = self._materialize(i)
                if partial.text or partial.tokens or partial.segments:
                    exc.partial_result = partial
            out.append(exc)
            if first_exc is None:
                first_exc = exc
        if first_exc is None and batch_abort is not None:
            # Anomalous: an abort with no per-utterance trace. Re-raise loud
            # (even under return_exceptions) rather than swallow a cancel.
            batch_abort.batch_results = out
            raise batch_abort
        if first_exc is not None and not return_exceptions:
            first_exc.batch_results = out
            raise first_exc
        return out

    def stream(self, *, task: Task = "transcribe", language: str | None = None,
               target_language: str | None = None, timestamps: Timestamps = "none",
               keep_special_tags: bool = False, commit_policy: CommitPolicy = "auto",
               stable_prefix_agreement_n: int = 0,
               family: FamilyExtension | None = None) -> Stream:
        """Begin streaming on this session and return a Stream to feed audio to.

        Requires a model whose capabilities advertise ``supports_streaming``;
        otherwise raises NotImplementedByModel. ``family`` is an optional
        family-specific stream extension (e.g. MoonshineStreamingOptions). The
        session is single-threaded and runs at most one stream at a time. Use
        the Stream as a context manager so it is reset when you are done."""
        self._cancel.clear()
        # spec_k_drafts is an offline-decode knob; streaming always uses the
        # family default (-1).
        run_params = _build_run_params(task, language, target_language, timestamps,
                                       keep_special_tags, -1)
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
        # The C contract says everything passed to begin may be freed once it
        # returns (strings are copied into session-owned storage). The Stream
        # still pins the params structs until reset() as defense in depth —
        # it costs nothing and keeps the binding safe even against an older
        # or out-of-tree native library that predates that contract.
        return Stream(self, _keepalive=(run_params, sp, ext))

    def _materialize(self, utt: int | None = None) -> Result:
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

    @property
    def limits(self) -> SessionLimits:
        """Effective limits for THIS session (model bound narrowed by
        ``n_ctx``). Check ``effective_max_audio_ms`` to size input up front
        instead of discovering ``InputTooLong`` on failure."""
        lm = _generated.transcribe_session_limits()
        _lib.transcribe_session_limits_init(_byref(lm))
        _check(_lib.transcribe_session_get_limits(self._h, _byref(lm)),
               "reading session limits")
        return SessionLimits(
            effective_n_ctx=lm.effective_n_ctx,
            effective_max_audio_ms=lm.effective_max_audio_ms,
            max_kv_bytes=lm.max_kv_bytes,
        )

    def cancel(self) -> None:
        """Request cancellation of an in-flight run/stream from another thread.
        The active call aborts at the next chunk/decode boundary and raises
        ``Aborted`` (with ``partial_result`` attached); ``was_aborted`` then
        reports True. The flag is cleared at the start of the next
        run/stream."""
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

    def __init__(self, session: Session, *, _keepalive=None):
        self._session = session  # keep the session (and its model) alive
        # Pins the ctypes params structs (and any family ext) passed to
        # transcribe_stream_begin until reset(). The native library copies
        # what it needs at begin; this is belt-and-braces for the FFI layer.
        self._keepalive = _keepalive
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

    @property
    def last_status(self) -> TranscribeError | None:
        """The stream's recorded terminal failure, or None while it is healthy.

        Set after a ``feed()`` / ``finalize()`` transitioned the stream to
        ``"failed"``; reset by ``begin`` / ``reset``. Inspect it when
        :attr:`state` is ``"failed"`` to learn why."""
        status = _lib.transcribe_stream_last_status(self._h)
        if status == _generated.TRANSCRIBE_OK:
            return None
        return exception_for_status(status, _status_string(status), "stream")

    def reset(self) -> None:
        """Return the session to idle, discarding stream state. Idempotent."""
        if self._active:
            _lib.transcribe_stream_reset(self._session._h)
            self._active = False
            self._keepalive = None

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *exc) -> None:
        self.reset()


def transcribe(
    model: Model | str | os.PathLike,
    pcm: PCMLike,
    *,
    backend: Backend = "auto",
    gpu_device: int = 0,
    n_threads: int = 0,
    kv_type: KVType = "auto",
    n_ctx: int = 0,
    task: Task = "transcribe",
    language: str | None = None,
    target_language: str | None = None,
    timestamps: Timestamps = "auto",
    keep_special_tags: bool = False,
    spec_k_drafts: int = -1,
    family: FamilyExtension | None = None,
) -> Result:
    """Transcribe *pcm* in one call and return a materialized Result.

    *model* may be a path (loaded and freed within this call) or an existing
    Model (reused and left open). Loading a model is not free, so to transcribe
    many clips keep a Model and call ``model.session().run(...)`` yourself; this
    helper is for the one-shot case. ``backend`` / ``gpu_device`` apply only when
    *model* is a path — they are ignored when an already-loaded Model is passed.
    ``family`` / ``spec_k_drafts`` pass through to :meth:`Session.run`.
    """
    session_opts = dict(n_threads=n_threads, kv_type=kv_type, n_ctx=n_ctx)
    run_opts = dict(task=task, language=language, target_language=target_language,
                    timestamps=timestamps, keep_special_tags=keep_special_tags,
                    spec_k_drafts=spec_k_drafts, family=family)

    if isinstance(model, Model):
        with model.session(**session_opts) as session:
            return session.run(pcm, **run_opts)

    with Model(model, backend=backend, gpu_device=gpu_device) as owned:
        with owned.session(**session_opts) as session:
            return session.run(pcm, **run_opts)
