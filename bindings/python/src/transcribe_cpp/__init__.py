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
Compute locking and ``Busy`` rules: see ``Model``.
"""

from __future__ import annotations

import collections
import ctypes
import enum
import os
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, TypeVar, Union

from . import _abi, _generated
from ._library import _base_version, artifact_dir, load_library, selected_provider
from .errors import (
    AbiError,
    Aborted,
    BackendError,
    Busy,
    InputTooLong,
    InputTooShort,
    InvalidArgument,
    ModelFileNotFound,
    ModelLoadError,
    NotImplementedByModel,
    OutOfMemory,
    OutputRepetition,
    OutputTruncated,
    TranscribeError,
    UnsupportedRequest,
    UnsupportedRole,
    exception_for_status,
    raise_for_status,
)

__version__ = "0.3.1"

# String-enum types, exported so callers (and type checkers) can name them.
Backend = Literal["auto", "cpu", "metal", "vulkan", "cpu_accel", "cuda", "rocm"]
KVType = Literal["auto", "f32", "f16"]
Task = Literal["transcribe", "translate", "instruct"]
Timestamps = Literal["none", "auto", "segment", "word", "token"]
Pnc = Literal["default", "off", "on"]
Itn = Literal["default", "off", "on"]
Diarize = Literal["default", "off", "on"]
SortformerPreset = Literal["default", "very_high_latency", "high_latency", "low_latency"]
CommitPolicy = Literal["auto", "on_finalize", "stable_prefix"]
Feature = Literal[
    "initial_prompt", "temperature_fallback", "long_form",
    "cancellation", "pnc", "itn", "diarization",
    "vocabulary", "context_prompt", "instruct", "transcript_prefix",
]

__all__ = [
    "__version__",
    "transcribe",
    "Model",
    "Session",
    "DiarizeSession",
    "DiarizeInfo",
    "LangIdSession",
    "LangIdInfo",
    "LangIdResult",
    "LangIdCandidate",
    "Role",
    "Result",
    "Segment",
    "SpeakerSegment",
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
    "SortformerDiarizeOptions",
    "VoxtralRealtimeStreamOptions",
    "Backend",
    "SortformerPreset",
    "KVType",
    "Task",
    "Timestamps",
    "Pnc",
    "Itn",
    "Diarize",
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
    "UnsupportedRole",
    "AbiError",
    "Aborted",
    "Busy",
    "InputTooLong",
    "InputTooShort",
    "OutputTruncated",
    "OutputRepetition",
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
_SpeakerSegment = _generated.transcribe_speaker_segment
_LangIdCandidate = _generated.transcribe_langid_candidate
_LangIdResult = _generated.transcribe_langid_result
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
    "rocm": _generated.TRANSCRIBE_BACKEND_ROCM,
}
_KV_TYPES = {
    "auto": _generated.TRANSCRIBE_KV_TYPE_AUTO,
    "f32": _generated.TRANSCRIBE_KV_TYPE_F32,
    "f16": _generated.TRANSCRIBE_KV_TYPE_F16,
}
_TASKS = {
    "transcribe": _generated.TRANSCRIBE_TASK_TRANSCRIBE,
    "translate": _generated.TRANSCRIBE_TASK_TRANSLATE,
    "instruct": _generated.TRANSCRIBE_TASK_INSTRUCT,
}
_TIMESTAMPS = {
    "none": _generated.TRANSCRIBE_TIMESTAMPS_NONE,
    "auto": _generated.TRANSCRIBE_TIMESTAMPS_AUTO,
    "segment": _generated.TRANSCRIBE_TIMESTAMPS_SEGMENT,
    "word": _generated.TRANSCRIBE_TIMESTAMPS_WORD,
    "token": _generated.TRANSCRIBE_TIMESTAMPS_TOKEN,
}
_TIMESTAMP_NAMES = {v: k for k, v in _TIMESTAMPS.items()}
_PNC = {
    "default": _generated.TRANSCRIBE_PNC_MODE_DEFAULT,
    "off": _generated.TRANSCRIBE_PNC_MODE_OFF,
    "on": _generated.TRANSCRIBE_PNC_MODE_ON,
}
_ITN = {
    "default": _generated.TRANSCRIBE_ITN_MODE_DEFAULT,
    "off": _generated.TRANSCRIBE_ITN_MODE_OFF,
    "on": _generated.TRANSCRIBE_ITN_MODE_ON,
}
_DIARIZE = {
    "default": _generated.TRANSCRIBE_DIARIZE_MODE_DEFAULT,
    "off": _generated.TRANSCRIBE_DIARIZE_MODE_OFF,
    "on": _generated.TRANSCRIBE_DIARIZE_MODE_ON,
}
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
    "diarize_run": _generated.TRANSCRIBE_EXT_SLOT_DIARIZE_RUN,
}
_FEATURES = {
    "initial_prompt": _generated.TRANSCRIBE_FEATURE_INITIAL_PROMPT,
    "temperature_fallback": _generated.TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK,
    "long_form": _generated.TRANSCRIBE_FEATURE_LONG_FORM,
    "cancellation": _generated.TRANSCRIBE_FEATURE_CANCELLATION,
    "pnc": _generated.TRANSCRIBE_FEATURE_PNC,
    "itn": _generated.TRANSCRIBE_FEATURE_ITN,
    "diarization": _generated.TRANSCRIBE_FEATURE_DIARIZATION,
    "vocabulary": _generated.TRANSCRIBE_FEATURE_VOCABULARY,
    "context_prompt": _generated.TRANSCRIBE_FEATURE_CONTEXT_PROMPT,
    "instruct": _generated.TRANSCRIBE_FEATURE_INSTRUCT,
    "transcript_prefix": _generated.TRANSCRIBE_FEATURE_TRANSCRIPT_PREFIX,
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


@dataclass(frozen=True, eq=False)
class BackendDevice:
    """One registered compute device (owned copies of the C strings).

    Equality compares opaque native identity; display index and live memory
    snapshots do not affect whether two values name the same device.
    """

    name: str
    description: str
    kind: str  # "cpu" | "accel" | "metal" | "vulkan" | "cuda" | "rocm" | "sycl" | "gpu" | "unknown"
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
    # Registry index for display. Exact model selection uses the BackendDevice
    # itself; indices are process-local and not stable across driver updates.
    index: Optional[int] = None
    # Opaque process-local native device handle. Applications persist device_id,
    # never this value.
    _handle: Optional[int] = field(default=None, repr=False, compare=False)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, BackendDevice):
            return NotImplemented
        return self._handle is not None and self._handle == other._handle

    def __hash__(self) -> int:
        return hash(self._handle) if self._handle is not None else object.__hash__(self)


def _backend_device_from_raw(dev, handle: Optional[int], index: Optional[int] = None) -> BackendDevice:
    """Build a BackendDevice from a library-filled transcribe_device_info."""
    return BackendDevice(
        name=_decode(dev.name),
        description=_decode(dev.description),
        kind=_decode(dev.kind),
        device_type=_DEVICE_TYPE_NAMES.get(dev.device_type, "unknown"),
        device_id=_decode(dev.device_id) if dev.device_id else None,
        memory_total=int(dev.memory_total),
        memory_free=int(dev.memory_free),
        index=index,
        _handle=handle,
    )


def backends() -> list[BackendDevice]:
    """The compute devices registered with the native runtime — what the
    process can actually run on, after backend-module loading and graceful
    degradation (e.g. a Vulkan module skipped on a machine without Vulkan).

    Each device's ``memory_free`` is live as of the call; call again to poll
    a device's available memory over time."""
    devices = []
    for i in range(_lib.transcribe_device_count()):
        handle = _lib.transcribe_device_get(i)
        if not handle:
            continue
        dev = _generated.transcribe_device_info()
        _lib.transcribe_device_info_init(_byref(dev))
        _check(_lib.transcribe_device_get_info(handle, _byref(dev)),
               f"reading backend device {i}")
        devices.append(_backend_device_from_raw(dev, int(handle), index=i))
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
    TRANSCRIBE_LOG_LEVEL_* (1=info, 2=warn, 3=error, 4=debug).

    The handler may fire while its model's compute lock is held, so it must
    not call compute APIs on that model (raises, or deadlocks off-thread)."""
    global _log_handler, _log_trampoline
    _log_handler = handler
    if _log_trampoline is None:
        # Installed once; later calls only swap the Python handler behind it.
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
    speaker_id: int


@dataclass(frozen=True)
class SpeakerSegment:
    """One diarized turn. Times are zero when the model attributes text but
    does not provide speaker timing; ``p`` is NaN when unavailable."""

    t0_ms: int
    t1_ms: int
    speaker_id: int
    p: float


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


class Role(enum.Enum):
    """What a model serves (``Model.roles``): ASR is ``Model.session()``,
    DIARIZE is ``Model.diarize_session()``, LANGID is
    ``Model.langid_session()``."""

    ASR = _generated.TRANSCRIBE_ROLE_ASR
    DIARIZE = _generated.TRANSCRIBE_ROLE_DIARIZE
    LANGID = _generated.TRANSCRIBE_ROLE_LANGID


@dataclass(frozen=True)
class DiarizeInfo:
    sample_rate: int
    max_speakers: int  # speaker_id is in [1, max_speakers]


@dataclass(frozen=True)
class LangIdInfo:
    sample_rate: int
    n_labels: int      # label indices are [0, n_labels)
    min_audio_ms: int  # shorter scored audio raises InputTooShort
    max_audio_ms: int  # longer input is scored on its first max_audio_ms


@dataclass(frozen=True)
class LangIdCandidate:
    """One ranked label. ``code`` is the model's own label ("en", "iw");
    ``p`` is renormalized over the allowed set."""

    index: int
    code: str
    name: str
    p: float
    logit: float


@dataclass(frozen=True)
class LangIdResult:
    """``candidates`` are ranked by ``p`` (ties keep label order).
    ``allowed_mass`` is the unrestricted probability inside the allowed set
    (1.0 when unrestricted); a low value means the speech is probably outside
    it. There is one candidate per allowed label (every label when
    unrestricted)."""

    candidates: tuple[LangIdCandidate, ...]
    allowed_mass: float

    @property
    def code(self) -> str | None:
        """The top candidate's code, or None when there are no candidates."""
        return self.candidates[0].code if self.candidates else None


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
    #: The model's decoded output before family post-processing (diarization
    #: markers, timestamp/special tokens, tag filtering, whitespace trims).
    #: Equal to ``text`` modulo whitespace for families that emit clean text.
    raw_text: str
    language: str
    timestamp_kind: str
    segments: tuple[Segment, ...]
    speaker_segments: tuple[SpeakerSegment, ...]
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
        speaker_id=s.speaker_id,
    )


def _speaker_segment_from(s) -> SpeakerSegment:
    return SpeakerSegment(
        t0_ms=s.t0_ms, t1_ms=s.t1_ms,
        speaker_id=s.speaker_id, p=s.p,
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


def _cstr(value: str, name: str) -> bytes:
    """UTF-8 bytes for a C string option; a NUL would silently cut it in C."""
    if "\x00" in value:
        raise InvalidArgument(f"{name} contains a NUL character")
    return value.encode("utf-8")


def _build_run_params(task, language, target_language, timestamps,
                      keep_special_tags, spec_k_drafts, diarize="default",
                      pnc="default", itn="default", vocabulary=None,
                      prompt=None, prefix=None):
    if not isinstance(spec_k_drafts, int) or spec_k_drafts < -1:
        raise InvalidArgument(
            f"spec_k_drafts must be -1 (family default), 0 (disabled), or a "
            f"positive draft length; got {spec_k_drafts!r}"
        )
    params = _RunParams()
    _lib.transcribe_run_params_init(_byref(params))
    params.task = _enum(_TASKS, task, "task")
    params.timestamps = _enum(_TIMESTAMPS, timestamps, "timestamps")
    params.pnc = _enum(_PNC, pnc, "pnc")
    params.itn = _enum(_ITN, itn, "itn")
    params.diarize = _enum(_DIARIZE, diarize, "diarize")
    params.language = _cstr(language, "language") if language else None
    params.target_language = _cstr(target_language, "target_language") if target_language else None
    params.keep_special_tags = keep_special_tags
    params.spec_k_drafts = spec_k_drafts
    if vocabulary is not None:
        if isinstance(vocabulary, (str, bytes)):
            raise InvalidArgument("vocabulary must be a sequence of terms, not a single string")
        if isinstance(vocabulary, (set, frozenset, dict)):
            raise InvalidArgument("vocabulary is in priority order; pass a list, not a set or dict")
        vocabulary = list(vocabulary)  # an iterator would be used up by the check below
        if not all(isinstance(t, str) for t in vocabulary):
            raise InvalidArgument("vocabulary terms must be strings")
        terms = [_cstr(t, "vocabulary") for t in vocabulary]
        if terms:
            arr = (ctypes.c_char_p * len(terms))(*terms)
            params.vocabulary = ctypes.cast(arr, ctypes.POINTER(ctypes.c_char_p))
            params.n_vocabulary = len(terms)
            # The C struct holds raw pointers; keep the buffers alive with it.
            params._prompting_keepalive = (arr, terms)
    params.prompt = _cstr(prompt, "prompt") if prompt else None
    params.prefix = _cstr(prefix, "prefix") if prefix else None
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
            ext.initial_prompt = _cstr(self.initial_prompt, "initial_prompt")
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


class SortformerDiarizeOptions(FamilyExtension):
    """Sortformer operating-point options for ``DiarizeSession.run()``
    (diarize-run slot).

    ``preset`` selects the latency / accuracy trade-off from the model's
    published menu; ``"default"`` keeps the GGUF-shipped checkpoint
    configuration. ``"very_high_latency"`` (~30 s lookahead) is the
    offline-file operating point; ``"low_latency"`` (~1 s) is the
    real-time point and costs substantially more compute per audio
    second."""

    _slot = "diarize_run"
    _kind = _generated.TRANSCRIBE_EXT_KIND_SORTFORMER_DIARIZE
    _struct = _generated.transcribe_sortformer_diarize_ext
    _init = "transcribe_sortformer_diarize_ext_init"

    _presets = {
        "default": _generated.TRANSCRIBE_SORTFORMER_PRESET_DEFAULT,
        "very_high_latency": _generated.TRANSCRIBE_SORTFORMER_PRESET_VERY_HIGH_LATENCY,
        "high_latency": _generated.TRANSCRIBE_SORTFORMER_PRESET_HIGH_LATENCY,
        "low_latency": _generated.TRANSCRIBE_SORTFORMER_PRESET_LOW_LATENCY,
    }

    def __init__(self, *, preset: SortformerPreset | None = None):
        if preset is not None and preset not in self._presets:
            raise ValueError(f"unknown sortformer preset {preset!r}; "
                             f"expected one of {sorted(self._presets)}")
        self.preset = preset

    def _apply(self, ext) -> None:
        if self.preset is not None:
            ext.preset = self._presets[self.preset]


# --- high-level handles ---------------------------------------------------


class _StreamLease:
    """One stream's claim on its model's stream lease (identity token for
    ``Model._stream_owner``). ``handle`` is the session handle it began on."""

    __slots__ = ("handle",)

    def __init__(self, handle: ctypes.c_void_p):
        self.handle = handle


class Model:
    """A loaded model. Sharing it across threads is safe: queries
    (``arch``, ``capabilities``, ``supports`` ...) and session creation run
    concurrently. It must outlive its sessions (``close()`` enforces that).

    The native library allows one compute call in flight across ALL sessions
    of a model, so ``Session.run()`` / ``run_batch()`` / ``stream()``,
    ``Stream.feed()`` / ``finalize()`` / ``reset()`` and ``DiarizeSession.run()``
    hold a model-wide lock for the native call and its copy-out. Calls from
    other threads wait; a ``cancel()`` on a queued call is kept, and it
    aborts at its first poll after acquiring the lock. Load one Model per
    worker for parallelism. An active stream holds the model's stream
    lease from ``stream()`` until it is finalized, reset, fails, is
    garbage-collected or its session/model is closed. Meanwhile ``run()``,
    ``run_batch()``, ``stream()`` and diarize ``run()`` on any session raise
    :class:`Busy` once they hold the lock; the stream's own calls proceed.
    ``close()`` and GC never wait for the lock: the handle is closed at once
    and the native free runs right after any in-flight call.

    ``backend="auto"`` (the default) picks the best available device. The
    ``TRANSCRIBE_BACKEND`` environment variable overrides that *default* —
    the escape hatch for machines whose best-ranked device misbehaves (e.g.
    CI's paravirtualized Metal, a broken GPU driver) without touching code.
    An explicit ``backend=`` argument always wins over the environment.
    """

    def __init__(self, path: str | os.PathLike, *,
                 backend: Backend = "auto", device: BackendDevice | None = None):
        # Live sessions, tracked weakly: close() must free them before the
        # model, because transcribe_model_free is only valid once every
        # derived session is gone (use-after-free otherwise). Created before
        # the load call so the failure path of __del__ finds it.
        self._sessions = weakref.WeakSet()
        self._init_compute_state()
        backend_source = "backend"
        if backend == "auto" and os.environ.get("TRANSCRIBE_BACKEND"):
            backend = os.environ["TRANSCRIBE_BACKEND"]  # type: ignore[assignment]
            backend_source = "backend (from TRANSCRIBE_BACKEND)"
        params = _ModelLoadParams()
        _lib.transcribe_model_load_params_init(_byref(params))
        params.backend = _enum(_BACKENDS, backend, backend_source)
        if device is not None:
            if not isinstance(device, BackendDevice) or device._handle is None:
                raise TypeError("device must be a BackendDevice returned by backends()")
            params.device = device._handle

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

    # --- compute exclusion -------------------------------------------------
    # One plain Lock per model serializes native compute calls and copy-out.
    # Not reentrant: the log callback can fire inside a native call, so a
    # nested compute is detected via _compute_owner and raises instead.
    # Native frees never wait: they queue FIFO on _deferred and the lock
    # holder runs them before releasing. close() queues every live session's
    # free before the model's, from _live (strong records, not the weak
    # _sessions, which the cycle collector clears before finalizers run).
    # _free_order makes "pop a session's _live record + queue its free" one
    # step against close() queuing the model's free, so a session closed on
    # another thread can never land behind the model free. An RLock: a GC
    # finalizer can close a session on the thread already holding it.
    # _stream_owner (the active stream's _StreamLease) is written only under it.

    def _init_compute_state(self) -> None:
        self._compute_lock = threading.Lock()
        self._compute_owner: Optional[int] = None  # thread ident while held
        self._deferred: collections.deque = collections.deque()
        self._stream_owner: Optional[_StreamLease] = None
        # id(handle) -> (handle, native free name, pin) per open session.
        self._live: dict = {}
        self._free_order = threading.RLock()

    @contextmanager
    def _exclusive(self, kind: str, busy: Optional[str] = None):
        """Hold the compute lock for one call of *kind* and check the model
        is open; run the native call and its copy-out inside the block. With
        *busy* set, raise Busy(*busy*) if a stream holds the lease."""
        me = threading.get_ident()
        if self._compute_owner == me:
            raise TranscribeError(
                f"cannot start {kind}: this thread is already inside a "
                "compute call on this model (a re-entrant call from a "
                "callback?)")
        try:
            # `with`: no gap between acquire and the release guard in which
            # a KeyboardInterrupt could leak the lock.
            with self._compute_lock:
                try:
                    self._compute_owner = me
                    self._h  # raises if the model is closed
                    if busy is not None and self._stream_owner is not None:
                        raise Busy(busy)
                    yield
                finally:
                    try:
                        self._drain_deferred_locked()
                    finally:
                        # Owner stays set while draining: a native free can
                        # log, and the re-entrancy check must still see this
                        # thread.
                        self._compute_owner = None
        finally:
            self._try_drain()

    def _drain_deferred_locked(self) -> None:
        """Run queued native frees. Caller holds the compute lock."""
        while self._deferred:
            try:
                self._deferred.popleft()()
            except Exception:
                pass  # a native free cannot meaningfully fail

    def _try_drain(self) -> None:
        """Run queued frees now if the lock is free, else leave them to the
        holder. Never blocks; loops so a free queued during release is run."""
        while self._deferred:
            got = False
            try:
                # Acquired inside the try, so the release guard covers all but
                # the call's own return (no non-blocking `with` exists).
                got = self._compute_lock.acquire(False)
                if not got:
                    return
                self._compute_owner = threading.get_ident()
                self._drain_deferred_locked()
            finally:
                if got:
                    self._compute_owner = None
                    self._compute_lock.release()

    def _free_or_defer(self, free) -> None:
        """Run *free* behind any in-flight compute call without waiting for
        it. *free* must not reference the Session (it may already be gone)."""
        self._deferred.append(free)
        self._try_drain()

    def _track(self, handle: ctypes.c_void_p, free_fn: str, pin=None) -> None:
        """Record an open session's native handle (and *pin*, kept alive
        until its free) so close() frees it even if its wrapper is gone."""
        self._live[id(handle)] = (handle, free_fn, pin)

    def _free_session(self, handle: ctypes.c_void_p) -> None:
        """Free a tracked session behind any in-flight call. The record is
        popped first, so the session's own close() and the model's close()
        free it exactly once, whichever runs first."""
        with self._free_order:
            self._queue_session_free(handle)
        self._try_drain()

    def _queue_session_free(self, handle: ctypes.c_void_p) -> None:
        """Pop *handle*'s record and queue its free. Caller holds _free_order."""
        record = self._live.pop(id(handle), None)
        if record is None:
            return
        handle, free_fn, pin = record

        def free(_pin=pin):
            getattr(_lib, free_fn)(handle)
            # Release this session's stream lease only after the free, in the
            # same locked step, so a queued run cannot start before it.
            if getattr(self._stream_owner, "handle", None) is handle:
                self._stream_owner = None

        self._deferred.append(free)

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
        handle = _lib.transcribe_model_device(self._h)
        if not handle:
            raise BackendError("model has no resolved compute device")
        dev = _generated.transcribe_device_info()
        _lib.transcribe_device_info_init(_byref(dev))
        _check(_lib.transcribe_device_get_info(handle, _byref(dev)),
               "device_get_info")
        return _backend_device_from_raw(dev, int(handle))

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
        """Whether the model exposes a behavioral feature (see ``Feature``)."""
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

    @property
    def roles(self) -> frozenset[Role]:
        mask = _lib.transcribe_model_roles(self._h)
        return frozenset(r for r in Role if mask & r.value)

    @property
    def diarize_info(self) -> DiarizeInfo:
        """Raises :class:`UnsupportedRole` without the DIARIZE role."""
        info = _generated.transcribe_diarize_info()
        _lib.transcribe_diarize_info_init(_byref(info))
        _check(_lib.transcribe_diarize_get_info(self._h, _byref(info)),
               "reading diarize info")
        return DiarizeInfo(sample_rate=info.sample_rate,
                           max_speakers=info.max_speakers)

    def diarize_session(self, *, n_threads: int = 0) -> "DiarizeSession":
        """Raises :class:`UnsupportedRole` without the DIARIZE role."""
        return DiarizeSession(self, n_threads=n_threads)

    @property
    def langid_info(self) -> LangIdInfo:
        """Raises :class:`UnsupportedRole` without the LANGID role."""
        info = _generated.transcribe_langid_info()
        _lib.transcribe_langid_info_init(_byref(info))
        _check(_lib.transcribe_langid_get_info(self._h, _byref(info)),
               "reading langid info")
        return LangIdInfo(sample_rate=info.sample_rate, n_labels=info.n_labels,
                          min_audio_ms=info.min_audio_ms,
                          max_audio_ms=info.max_audio_ms)

    @property
    def langid_labels(self) -> tuple[tuple[str, str], ...]:
        """``(code, name)`` per label index. Raises :class:`UnsupportedRole`
        without the LANGID role."""
        n = self.langid_info.n_labels
        return tuple((_decode(_lib.transcribe_langid_label_code(self._h, i)),
                      _decode(_lib.transcribe_langid_label_name(self._h, i)))
                     for i in range(n))

    def langid_label_index(self, code: str) -> int | None:
        """Label index of a code or alias ("he" and "iw" name the same
        label), or None. None as well on a model without the LANGID role.
        Raises :class:`InvalidArgument` if ``code`` contains a NUL."""
        i = _lib.transcribe_langid_label_index(self._h, _cstr(code, "code"))
        return i if i >= 0 else None

    def langid_session(self, *, n_threads: int = 0) -> "LangIdSession":
        """Raises :class:`UnsupportedRole` without the LANGID role."""
        return LangIdSession(self, n_threads=n_threads)

    def close(self) -> None:
        """Free the model. Any session still open on it is closed first —
        the C contract forbids freeing a model before its sessions, so this
        keeps explicit close()/context-manager exit safe in any order.
        Never waits for the compute lock (see above)."""
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        # Closed from here on: a call queued on the lock sees it once it gets
        # the lock; an in-flight call already captured its handles.
        self._handle = None
        for session in list(getattr(self, "_sessions", ()) or ()):
            session.close()
        # One step under _free_order: every session free still pending
        # (wrappers already collected, or a close() racing on another thread)
        # is queued ahead of the model's.
        with self._free_order:
            for record in list(self._live.values()):
                self._queue_session_free(record[0])
            self._deferred.append(lambda: _lib.transcribe_model_free(handle))
        self._try_drain()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


_S = TypeVar("_S", bound="_SessionBase")


class _SessionBase:
    """Handle, cancellation and close plumbing shared by Session and
    DiarizeSession. Subclasses name their native free in ``_free_fn``."""

    _free_fn = ""

    def _arm_abort(self, set_cb) -> None:
        """Install the abort callback (polled at chunk/decode boundaries;
        True once cancel() is requested) and register with the model. Binds
        the Event, not self, into the trampoline to avoid a reference cycle."""
        self._cancel = threading.Event()
        event = self._cancel
        self._abort_trampoline = _ABORT_CFUNC(lambda _ud: event.is_set())
        # The record pins the trampoline the native session points at.
        self._model._track(self._handle, self._free_fn, self._abort_trampoline)
        set_cb(self._handle, self._abort_trampoline, None)
        # Registered last, once fully constructed: Model.close() closes any
        # session still tracked here before freeing the model.
        self._model._sessions.add(self)

    @property
    def _h(self) -> ctypes.c_void_p:
        if self._handle is None:
            raise TranscribeError("session is closed")
        return self._handle

    def _attach_family(self, family: "FamilyExtension | None", slot: str, params) -> None:
        """Resolve *family* for *slot* and point ``params.family`` at it (the
        params struct keeps the built extension alive)."""
        if family is not None:
            ext = self._resolve_family(family, slot)
            params.family = ctypes.cast(_byref(ext), ctypes.POINTER(_generated.transcribe_ext))
            params._family_keepalive = ext

    def _resolve_family(self, family: "FamilyExtension", slot: str):
        """Validate + probe a family extension and return its built ctypes
        struct."""
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

    def cancel(self) -> None:
        """Request cancellation of an in-flight call from another thread: it
        aborts at the next chunk/decode boundary and raises ``Aborted`` (with
        ``partial_result`` where available). The flag is cleared when the next
        run / run_batch / stream / diarize run starts, before it waits for the
        lock, so a cancel() while queued still aborts it. Lock-free."""
        self._cancel.set()

    def close(self) -> None:
        """Free the session. Idempotent; never waits for the compute lock
        (see ``Model``)."""
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        self._handle = None
        self._model._free_session(handle)  # a no-op if the model freed it

    def __enter__(self: _S) -> _S:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class Session(_SessionBase):
    """A single transcription context bound to one Model. Compute locking and
    ``Busy`` rules: see ``Model``. Reads (``limits``, ``was_aborted``, Stream
    ``text()`` ...) are unlocked; don't read a session while another thread
    computes on it."""

    _free_fn = "transcribe_session_free"

    def __init__(self, model: Model, *, n_threads: int = 0, kv_type: KVType = "auto",
                 n_ctx: int = 0):
        self._model = model  # keep the model alive for the session's lifetime
        # Lease of the newest Stream begun here; an older Stream is superseded.
        self._stream_lease: Optional[_StreamLease] = None
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
        self._arm_abort(_lib.transcribe_set_abort_callback)

    def run(self, pcm: PCMLike, *, task: Task = "transcribe",
            language: str | None = None,
            target_language: str | None = None,
            timestamps: Timestamps = "auto",
            pnc: Pnc = "default",
            itn: Itn = "default",
            diarize: Diarize = "default",
            keep_special_tags: bool = False,
            spec_k_drafts: int = -1,
            family: FamilyExtension | None = None,
            vocabulary: Sequence[str] | None = None,
            prompt: str | None = None,
            prefix: str | None = None) -> Result:
        """Transcribe 16 kHz mono float32 PCM and return a materialized Result.

        ``pnc`` controls punctuation/capitalization and ``itn`` controls
        inverse text normalization on models advertising those features.
        ``family`` is an optional family-specific extension (e.g.
        WhisperRunOptions) carrying per-run knobs for models that accept it.
        ``spec_k_drafts`` tunes speculative decoding on models whose
        capabilities advertise ``supports_spec_decode`` (-1 = family default,
        0 = disabled, >0 = draft length; silently ignored elsewhere).
        ``vocabulary`` (terms, priority order), ``prompt`` and ``prefix`` are
        the generic prompting inputs; probe ``model.supports()`` for
        ``"vocabulary"``, ``"context_prompt"``, ``"instruct"`` and
        ``"transcript_prefix"``. With ``task="instruct"`` the ``prompt`` is the
        required instruction and the output is free text. Unsupported
        vocabulary/context is ignored with a warning; an unsupported prefix
        or instruct task raises. With ``prefix``, ``text`` holds only the
        continuation and ``raw_text`` leads with the prefix.

        On ``Aborted`` (via :meth:`cancel`) and ``OutputTruncated`` (including
        its ``OutputRepetition`` subclass) the partial transcript is preserved
        and attached to the exception as ``partial_result``.

        Raises :class:`Busy` if a stream is active on this model."""
        # Cleared BEFORE waiting for the compute lock, so a cancel() issued
        # while this call is queued behind another still aborts it.
        self._cancel.clear()
        array, n_samples = _pcm_to_carray(pcm)
        params = _build_run_params(task, language, target_language, timestamps,
                                   keep_special_tags, spec_k_drafts, diarize, pnc, itn,
                                   vocabulary, prompt, prefix)
        with self._model._exclusive(
                "run", busy="a stream is active on this model; "
                            "finish or drop it before run()"):
            self._attach_family(family, "run", params)
            h = self._h  # captured under the lock; close() defers its free
            try:
                _check(_lib.transcribe_run(h, array, n_samples, _byref(params)),
                       "transcribe_run")
            except (Aborted, OutputTruncated) as exc:
                # The C API preserves the partial transcript on the session for
                # these statuses (OutputRepetition included, as an OutputTruncated
                # subclass); surface it rather than discard it.
                exc.partial_result = self._materialize(h)
                raise
            return self._materialize(h)

    def run_batch(self, pcms: Sequence[PCMLike], *, task: Task = "transcribe",
                  language: str | None = None,
                  target_language: str | None = None,
                  timestamps: Timestamps = "auto",
                  pnc: Pnc = "default",
                  itn: Itn = "default",
                  diarize: Diarize = "default",
                  keep_special_tags: bool = False,
                  spec_k_drafts: int = -1,
                  family: FamilyExtension | None = None,
                  return_exceptions: bool = False,
                  vocabulary: Sequence[str] | None = None,
                  prompt: str | None = None) -> list[Result | TranscribeError]:
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
        directly (the ``asyncio.gather`` convention).

        ``vocabulary`` / ``prompt`` apply to every utterance (see :meth:`run`);
        a transcript prefix is per-utterance and so is not accepted here.

        Raises :class:`Busy` if a stream is active on this model."""
        self._cancel.clear()  # before the lock wait, as in run()
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
                                   keep_special_tags, spec_k_drafts, diarize, pnc, itn,
                                   vocabulary, prompt)
        # The native call AND the per-utterance copy-out run under the lock.
        with self._model._exclusive(
                "run_batch", busy="a stream is active on this model; "
                                  "finish or drop it before run_batch()"):
            self._attach_family(family, "run", params)
            h = self._h  # captured under the lock; close() defers its free
            batch_abort = None
            try:
                _check(
                    _lib.transcribe_run_batch(h, ptrs, counts, len(pcms), _byref(params)),
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
            for i in range(_lib.transcribe_batch_n_results(h)):
                status = _lib.transcribe_batch_status(h, i)
                if status == 0:
                    out.append(self._materialize(h, i))
                    continue
                exc = exception_for_status(status, _status_string(status),
                                           f"utterance {i} in batch")
                exc.utterance_index = i
                if isinstance(exc, (Aborted, OutputTruncated)):
                    # Attach the partial transcript only when the native layer
                    # actually snapshotted one for this slot — some batch paths
                    # record failed slots as empty, and None is more honest than
                    # a confidently empty Result.
                    partial = self._materialize(h, i)
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
               pnc: Pnc = "default", itn: Itn = "default",
               diarize: Diarize = "default",
               keep_special_tags: bool = False, commit_policy: CommitPolicy = "auto",
               stable_prefix_agreement_n: int = 0,
               family: FamilyExtension | None = None,
               vocabulary: Sequence[str] | None = None,
               prompt: str | None = None) -> Stream:
        """Begin streaming on this session and return a Stream to feed audio to.

        Requires a model whose capabilities advertise ``supports_streaming``;
        otherwise raises NotImplementedByModel. ``family`` is an optional
        family-specific stream extension (e.g. MoonshineStreamingOptions). The
        session is single-threaded. Use the Stream as a context manager so it
        is reset when you are done. ``vocabulary`` and ``prompt`` are as in
        :meth:`run`; ``task="instruct"`` raises.

        Takes the model's stream lease (see ``Model``); raises :class:`Busy`
        if another stream is active on this model."""
        self._cancel.clear()  # before the lock wait, as in run()
        # spec_k_drafts is an offline-decode knob; streaming always uses the
        # family default (-1).
        run_params = _build_run_params(task, language, target_language, timestamps,
                                       keep_special_tags, -1, diarize, pnc, itn,
                                       vocabulary, prompt)
        sp = _StreamParams()
        _lib.transcribe_stream_params_init(_byref(sp))
        sp.commit_policy = _enum(_COMMIT_POLICIES, commit_policy, "commit_policy")
        sp.stable_prefix_agreement_n = stable_prefix_agreement_n
        with self._model._exclusive(
                "stream_begin", busy="a stream is already active on this model"):
            self._attach_family(family, "stream", sp)
            h = self._h
            _check(
                _lib.transcribe_stream_begin(h, _byref(run_params), _byref(sp)),
                "transcribe_stream_begin",
            )
            # Claim the lease only now that the native begin succeeded.
            lease = _StreamLease(h)
            self._model._stream_owner = lease
            self._stream_lease = lease
            # The C contract says everything passed to begin may be freed once
            # it returns (strings are copied into session-owned storage). The
            # Stream still pins the params structs until reset() as defense in
            # depth — it costs nothing and keeps the binding safe even against
            # an older or out-of-tree native library that predates that
            # contract. Built under the lock, so the lease always has a Stream.
            return Stream(self, _keepalive=(run_params, sp), _lease=lease)

    def _materialize(self, h: ctypes.c_void_p, utt: int | None = None) -> Result:
        """Copy out one result from session handle *h*. The run paths call
        this under the compute lock with *h* captured there, so a concurrent
        close() cannot pull it away mid-copy; ``Stream.snapshot()`` is an
        unlocked read (see ``Session``). utt is None for the single-result
        accessors, or an utterance index for the batch accessors (index 0
        aliases the single result after a plain run, so both paths share
        this code)."""
        if utt is None:
            n_seg = lambda: _lib.transcribe_n_segments(h)
            get_seg = lambda j, out: _lib.transcribe_get_segment(h, j, out)
            n_word = lambda: _lib.transcribe_n_words(h)
            get_word = lambda j, out: _lib.transcribe_get_word(h, j, out)
            n_tok = lambda: _lib.transcribe_n_tokens(h)
            get_tok = lambda j, out: _lib.transcribe_get_token(h, j, out)
            n_speaker = lambda: _lib.transcribe_n_speaker_segments(h)
            get_speaker = lambda j, out: _lib.transcribe_get_speaker_segment(h, j, out)
            full_text = _lib.transcribe_full_text(h)
            raw_text = _lib.transcribe_raw_text(h)
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
            n_speaker = lambda: _lib.transcribe_batch_n_speaker_segments(h, utt)
            get_speaker = lambda j, out: _lib.transcribe_batch_get_speaker_segment(h, utt, j, out)
            full_text = _lib.transcribe_batch_full_text(h, utt)
            raw_text = _lib.transcribe_batch_raw_text(h, utt)
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

        speaker_segments = []
        for j in range(n_speaker()):
            speaker = _SpeakerSegment()
            _lib.transcribe_speaker_segment_init(_byref(speaker))
            get_speaker(j, _byref(speaker))
            speaker_segments.append(_speaker_segment_from(speaker))

        tm = _Timings()
        _lib.transcribe_timings_init(_byref(tm))
        get_tim(_byref(tm))

        return Result(
            text=_decode(full_text),
            raw_text=_decode(raw_text),
            language=_decode(language),
            timestamp_kind=_TIMESTAMP_NAMES.get(kind, "unknown"),
            segments=tuple(segments),
            speaker_segments=tuple(speaker_segments),
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

    @property
    def was_aborted(self) -> bool:
        """True if the most recent call was ended by cancel()."""
        return bool(_lib.transcribe_was_aborted(self._h))


class Stream:
    """An active streaming transcription on a Session.

    Feed audio in chunks with ``feed()``, read the committed/tentative text with
    ``text()``, and call ``finalize()`` when the audio ends. The session is
    returned to idle on context-manager exit (or ``reset()``).

    While active it holds the model's stream lease (see ``Model``); a feed
    rejected before the model sees it (e.g. NaN/Inf samples) keeps it."""

    def __init__(self, session: Session, *, _lease: _StreamLease, _keepalive=None):
        self._session = session  # keep the session (and its model) alive
        # Pins the ctypes params structs (and any family ext) passed to
        # transcribe_stream_begin until reset(). The native library copies
        # what it needs at begin; this is belt-and-braces for the FFI layer.
        self._keepalive = _keepalive
        self._active = True
        self._lease = _lease  # holds the lease while model._stream_owner is this

    def _release_lease_locked(self) -> None:
        """Release the model's stream lease if THIS stream still holds it
        (never a later stream's). Caller holds the compute lock."""
        model = self._session._model
        if model._stream_owner is self._lease:
            model._stream_owner = None

    def _release_lease_if_ended_locked(self, h: ctypes.c_void_p) -> None:
        """After a feed/finalize: release the lease once the native stream
        is no longer ACTIVE (FINISHED or FAILED). A call rejected before the
        model saw it (e.g. NaN/Inf samples) leaves it ACTIVE and keeps it."""
        if _lib.transcribe_stream_get_state(h) != _generated.TRANSCRIBE_STREAM_ACTIVE:
            self._release_lease_locked()

    @property
    def _h(self) -> ctypes.c_void_p:
        if not self._active:
            raise TranscribeError("stream has been reset")
        if self._session._stream_lease is not self._lease:
            raise TranscribeError("stream superseded by a newer stream on this session")
        return self._session._h

    def feed(self, pcm: PCMLike) -> StreamUpdate:
        """Feed a chunk of 16 kHz mono float32 PCM; returns change metadata."""
        array, n_samples = _pcm_to_carray(pcm)
        update = _StreamUpdate()
        _lib.transcribe_stream_update_init(_byref(update))
        with self._session._model._exclusive("stream_feed"):  # no busy check
            h = self._h
            status = _lib.transcribe_stream_feed(h, array, n_samples, _byref(update))
            self._release_lease_if_ended_locked(h)
            _check(status, "transcribe_stream_feed")
            return _stream_update_from(update)

    def finalize(self) -> StreamUpdate:
        """Signal end of audio and flush the final hypothesis."""
        update = _StreamUpdate()
        _lib.transcribe_stream_update_init(_byref(update))
        with self._session._model._exclusive("stream_finalize"):
            h = self._h
            status = _lib.transcribe_stream_finalize(h, _byref(update))
            self._release_lease_if_ended_locked(h)
            _check(status, "transcribe_stream_finalize")
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

    def snapshot(self) -> Result:
        """Full structured snapshot of the current hypothesis (owned copies)."""
        return self._session._materialize(self._h)

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
        """Return the session to idle, discarding stream state, and release
        the model's stream lease. Idempotent; a no-op once a newer stream has
        begun on the session."""
        if self._active:
            with self._session._model._exclusive("stream_reset"):
                if self._session._stream_lease is self._lease:
                    _lib.transcribe_stream_reset(self._session._h)
                    self._release_lease_locked()
            self._active = False
            self._keepalive = None

    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, exc_type, *exc) -> None:
        try:
            self.reset()
        except TranscribeError:
            # e.g. the session or model was closed in the block: never mask
            # the exception that is already propagating.
            if exc_type is None:
                raise

    def __del__(self):
        # Abandoned while holding the lease: reset and release it behind any
        # in-flight call (never waits). "Not the owner" is final; re-checked locked.
        try:
            lease, model = self._lease, self._session._model
            if model._stream_owner is not lease:
                return

            def reset_and_release():
                if model._stream_owner is lease:  # so the session is still open
                    _lib.transcribe_stream_reset(lease.handle)
                    model._stream_owner = None

            model._free_or_defer(reset_and_release)
        except Exception:
            pass


class DiarizeSession(_SessionBase):
    """A diarization (who spoke when) context on a model with the DIARIZE
    role. Compute locking and ``Busy`` rules: see ``Model``."""

    _free_fn = "transcribe_diarize_session_free"

    def __init__(self, model: Model, *, n_threads: int = 0):
        self._model = model  # keep the model alive for the session's lifetime
        params = _generated.transcribe_diarize_session_params()
        _lib.transcribe_diarize_session_params_init(_byref(params))
        params.n_threads = n_threads

        handle = ctypes.c_void_p()
        _check(_lib.transcribe_diarize_session_init(model._h, _byref(params), _byref(handle)),
               "opening diarize session")
        if not handle.value:
            raise TranscribeError("diarize session init returned a null handle")
        self._handle = handle
        self._arm_abort(_lib.transcribe_diarize_set_abort_callback)

    def run(self, pcm: PCMLike, *,
            family: FamilyExtension | None = None) -> list[SpeakerSegment]:
        """Diarize one recording (16 kHz mono float32 PCM) and return its
        speaker turns: grouped by speaker, time-ordered within a speaker,
        possibly overlapping across speakers. ``family`` is an optional
        diarize-run extension (e.g. SortformerDiarizeOptions).

        Raises :class:`Aborted` after :meth:`cancel`, and :class:`Busy` if a
        stream is active on this model."""
        self._cancel.clear()  # before the lock wait, as in Session.run()
        array, n_samples = _pcm_to_carray(pcm)
        params = _generated.transcribe_diarize_params()
        _lib.transcribe_diarize_params_init(_byref(params))
        with self._model._exclusive(
                "diarize_run", busy="a stream is active on this model; "
                                    "finish or drop it before diarize run()"):
            self._attach_family(family, "diarize_run", params)
            h = self._h  # captured under the lock; close() defers its free
            _check(_lib.transcribe_diarize_run(h, array, n_samples, _byref(params)),
                   "transcribe_diarize_run")
            rows = []
            for i in range(_lib.transcribe_diarize_n_segments(h)):
                s = _SpeakerSegment()
                _lib.transcribe_speaker_segment_init(_byref(s))
                _check(_lib.transcribe_diarize_get_segment(h, i, _byref(s)),
                       "transcribe_diarize_get_segment")
                rows.append(_speaker_segment_from(s))
            return rows

    @property
    def timings(self) -> Timings:
        """Load time plus the last run's mel / encode time. Not locked, like
        ``Session.limits``."""
        tm = _Timings()
        _lib.transcribe_timings_init(_byref(tm))
        _check(_lib.transcribe_diarize_get_timings(self._h, _byref(tm)),
               "transcribe_diarize_get_timings")
        return _timings_from(tm)


class LangIdSession(_SessionBase):
    """A language ID context on a model with the LANGID role. Compute
    locking and ``Busy`` rules: see ``Model``."""

    _free_fn = "transcribe_langid_session_free"

    def __init__(self, model: Model, *, n_threads: int = 0):
        self._model = model  # keep the model alive for the session's lifetime
        params = _generated.transcribe_langid_session_params()
        _lib.transcribe_langid_session_params_init(_byref(params))
        params.n_threads = n_threads

        handle = ctypes.c_void_p()
        _check(_lib.transcribe_langid_session_init(model._h, _byref(params), _byref(handle)),
               "opening langid session")
        if not handle.value:
            raise TranscribeError("langid session init returned a null handle")
        self._handle = handle
        self._arm_abort(_lib.transcribe_langid_set_abort_callback)

    def run(self, pcm: PCMLike, *,
            allowed: "Sequence[str] | None" = None) -> LangIdResult:
        """Identify the language of one clip (16 kHz mono float32 PCM).
        Input longer than ``LangIdInfo.max_audio_ms`` is scored on its first
        ``max_audio_ms``. ``allowed`` restricts the decision to those codes
        or aliases; None means every label, and an empty list is rejected.
        Every allowed label is returned, ranked by ``p``.

        Raises :class:`InputTooShort` below ``LangIdInfo.min_audio_ms``,
        :class:`UnsupportedRequest` for an unknown code, :class:`Aborted`
        after :meth:`cancel`, and :class:`Busy` if a stream is active on this
        model."""
        self._cancel.clear()  # before the lock wait, as in Session.run()
        array, n_samples = _pcm_to_carray(pcm)
        params = _generated.transcribe_langid_params()
        _lib.transcribe_langid_params_init(_byref(params))
        if allowed is not None:
            if isinstance(allowed, (str, bytes)):
                raise InvalidArgument("allowed must be a sequence of codes, not a string")
            codes = list(allowed)
            # An empty list must not reach native as NULL, which means "all".
            if not codes:
                raise InvalidArgument("allowed is empty; pass None for every label")
            if not all(isinstance(c, str) for c in codes):
                raise InvalidArgument("allowed entries must be str")
            encoded = [_cstr(c, "allowed") for c in codes]
            arr = (ctypes.c_char_p * len(encoded))(*encoded)
            params.allowed = ctypes.cast(arr, type(params.allowed))
            params.n_allowed = len(encoded)
            params._allowed_keepalive = (encoded, arr)
        with self._model._exclusive(
                "langid_run", busy="a stream is active on this model; "
                                   "finish or drop it before langid run()"):
            h = self._h  # captured under the lock; close() defers its free
            _check(_lib.transcribe_langid_run(h, array, n_samples, _byref(params)),
                   "transcribe_langid_run")
            res = _LangIdResult()
            _lib.transcribe_langid_result_init(_byref(res))
            _check(_lib.transcribe_langid_get_result(h, _byref(res)),
                   "transcribe_langid_get_result")
            rows = []
            for i in range(res.n_candidates):
                c = _LangIdCandidate()
                _lib.transcribe_langid_candidate_init(_byref(c))
                _check(_lib.transcribe_langid_get_candidate(h, i, _byref(c)),
                       "transcribe_langid_get_candidate")
                rows.append(LangIdCandidate(
                    index=c.index, code=_decode(c.code), name=_decode(c.name),
                    p=c.p, logit=c.logit))
            return LangIdResult(candidates=tuple(rows), allowed_mass=res.allowed_mass)

    @property
    def timings(self) -> Timings:
        """Load time plus the last run's mel / encode time. Not locked, like
        ``Session.limits``."""
        tm = _Timings()
        _lib.transcribe_timings_init(_byref(tm))
        _check(_lib.transcribe_langid_get_timings(self._h, _byref(tm)),
               "transcribe_langid_get_timings")
        return _timings_from(tm)


def transcribe(
    model: Model | str | os.PathLike,
    pcm: PCMLike,
    *,
    backend: Backend = "auto",
    device: BackendDevice | None = None,
    n_threads: int = 0,
    kv_type: KVType = "auto",
    n_ctx: int = 0,
    task: Task = "transcribe",
    language: str | None = None,
    target_language: str | None = None,
    timestamps: Timestamps = "auto",
    pnc: Pnc = "default",
    itn: Itn = "default",
    diarize: Diarize = "default",
    keep_special_tags: bool = False,
    spec_k_drafts: int = -1,
    family: FamilyExtension | None = None,
    vocabulary: Sequence[str] | None = None,
    prompt: str | None = None,
    prefix: str | None = None,
) -> Result:
    """Transcribe *pcm* in one call and return a materialized Result.

    *model* may be a path (loaded and freed within this call) or an existing
    Model (reused and left open). Loading a model is not free, so to transcribe
    many clips keep a Model and call ``model.session().run(...)`` yourself; this
    helper is for the one-shot case. ``backend`` / ``device`` apply only when
    *model* is a path — they are ignored when an already-loaded Model is passed.
    ``family`` / ``spec_k_drafts`` and the prompting inputs (``vocabulary``,
    ``prompt``, ``prefix``) pass through to :meth:`Session.run`.
    """
    session_opts = dict(n_threads=n_threads, kv_type=kv_type, n_ctx=n_ctx)
    run_opts = dict(task=task, language=language, target_language=target_language,
                    timestamps=timestamps, pnc=pnc, itn=itn, diarize=diarize,
                    keep_special_tags=keep_special_tags,
                    spec_k_drafts=spec_k_drafts, family=family,
                    vocabulary=vocabulary, prompt=prompt, prefix=prefix)

    if isinstance(model, Model):
        with model.session(**session_opts) as session:
            return session.run(pcm, **run_opts)

    with Model(model, backend=backend, device=device) as owned:
        with owned.session(**session_opts) as session:
            return session.run(pcm, **run_opts)
