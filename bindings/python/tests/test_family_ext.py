"""Family-extension coverage: option building (model-free) and the
slot/kind validation + happy path against real models.

The model-free half drives ``FamilyExtension._build()`` directly: the built
ctypes struct must carry the registered kind, the caller's sizeof, the init
function's defaults for unset fields, and ONLY the explicitly-set overrides.
The model-gated half pins ``Session._resolve_family``'s three rejection modes
and a real whisper ``initial_prompt`` run.
"""

from __future__ import annotations

import ctypes

import pytest

import transcribe_cpp as t
from transcribe_cpp import _generated

ALL_OPTION_TYPES = [
    t.WhisperRunOptions,
    t.MoonshineStreamingOptions,
    t.ParakeetStreamOptions,
    t.ParakeetBufferedStreamOptions,
    t.VoxtralRealtimeStreamOptions,
]


# --- model-free: every subclass builds a correctly-stamped struct ------------


@pytest.mark.parametrize("cls", ALL_OPTION_TYPES)
def test_build_stamps_kind_and_size(cls):
    ext = cls()._build()
    assert ext.ext.kind == cls._kind
    assert ext.ext.size == ctypes.sizeof(cls._struct)
    assert cls._slot in ("run", "stream")


@pytest.mark.parametrize("cls", ALL_OPTION_TYPES)
def test_build_defaults_equal_init_defaults(cls):
    # No overrides set: every byte must equal what the C init function
    # stamps — the "only fields you set override" contract's baseline.
    built = cls()._build()
    fresh = cls._struct()
    getattr(t._lib, cls._init)(ctypes.byref(fresh))
    assert bytes(built) == bytes(fresh)


def test_whisper_overrides_apply_and_unset_fields_keep_defaults():
    built = t.WhisperRunOptions(temperature=0.7, seed=1234)._build()
    fresh = t._generated.transcribe_whisper_run_ext()
    t._lib.transcribe_whisper_run_ext_init(ctypes.byref(fresh))

    assert abs(built.temperature - 0.7) < 1e-6
    assert built.seed == 1234
    # A field left at None keeps the init default.
    assert built.no_speech_thold == fresh.no_speech_thold
    assert built.condition_on_prev_tokens == fresh.condition_on_prev_tokens


def test_whisper_initial_prompt_encodes():
    built = t.WhisperRunOptions(initial_prompt="Glossary: GGUF, ggml")._build()
    assert built.initial_prompt == b"Glossary: GGUF, ggml"


def test_parakeet_buffered_partial_overrides():
    built = t.ParakeetBufferedStreamOptions(chunk_ms=2000)._build()
    fresh = _generated.transcribe_parakeet_buffered_stream_ext()
    t._lib.transcribe_parakeet_buffered_stream_ext_init(ctypes.byref(fresh))
    assert built.chunk_ms == 2000
    assert built.left_ms == fresh.left_ms       # untouched -> family default
    assert built.right_ms == fresh.right_ms


# --- model-gated: resolve_family validation + a real extension run ----------


def test_run_rejects_non_extension_object(model_path):
    with t.Model(model_path) as model, model.session() as session:
        with pytest.raises(t.InvalidArgument, match="FamilyExtension"):
            session.run([0.0] * 160, family=object())


def test_run_rejects_stream_slot_extension(model_path):
    # A stream-slot extension pointed at the run slot is a caller bug, not
    # something to silently ignore.
    with t.Model(model_path) as model, model.session() as session:
        with pytest.raises(t.InvalidArgument, match="slot"):
            session.run([0.0] * 160, family=t.MoonshineStreamingOptions())


def test_run_rejects_extension_model_does_not_accept(streaming_model_path):
    # WhisperRunOptions is a run-slot kind, but moonshine-streaming does not
    # accept it: UnsupportedRequest, probed BEFORE the native run.
    with t.Model(streaming_model_path) as model:
        assert not model.accepts(t.WhisperRunOptions())
        with model.session() as session:
            with pytest.raises(t.UnsupportedRequest):
                session.run([0.0] * 160, family=t.WhisperRunOptions())


def test_model_accepts_probe(model_path):
    with t.Model(model_path) as model:
        if model.arch != "whisper":
            pytest.skip("default canary is expected to be whisper")
        assert model.accepts(t.WhisperRunOptions()) is True
        assert model.accepts(t.MoonshineStreamingOptions()) is False


def test_whisper_initial_prompt_run(model_path, audio_pcm):
    with t.Model(model_path) as model:
        if not model.supports("initial_prompt"):
            pytest.skip("model does not support initial_prompt")
        with model.session() as session:
            opts = t.WhisperRunOptions(initial_prompt="A speech by JFK.")
            result = session.run(audio_pcm, family=opts)
    assert "country" in result.text.lower(), result.text


def test_run_batch_accepts_family(model_path, audio_pcm):
    # B3 plumbed family= through run_batch; pin the happy path.
    with t.Model(model_path) as model, model.session() as session:
        results = session.run_batch(
            [audio_pcm], family=t.WhisperRunOptions(temperature=0.0))
    assert len(results) == 1
    assert "country" in results[0].text.lower()


def test_supports_probe_all_features(model_path):
    with t.Model(model_path) as model:
        for feature in ("initial_prompt", "temperature_fallback", "long_form",
                        "cancellation", "pnc", "itn"):
            assert model.supports(feature) in (True, False)
        with pytest.raises(t.InvalidArgument, match="unknown feature"):
            model.supports("levitation")


# --- model-gated: per-family stream-extension happy path --------------------
#
# These prove the parakeet/voxtral stream extensions end to end: the typed
# options materialize a kind-tagged struct, the model ACCEPTS the kind on its
# stream slot, ``stream_begin`` consumes it, and a short feed + finalize emits
# text. NOT a transcription-accuracy check (that is the family port's C-level /
# WER job) — a short feed keeps these fast even for the 4B voxtral model, so we
# assert the stream ran and produced non-empty text rather than pinning content.
# Mirrors Swift's FamilyStreamTests. Each gates on its own per-family GGUF and
# skips cleanly when absent (parakeet runs in CI once the canary repos exist;
# voxtral is local-only — ~2.5 GB is too heavy for CI).

SHORT_FEED_SAMPLES = 2 * 16000  # ~2 s at 16 kHz mono


def _drive_short(stream, pcm):
    """Feed ~2 s of audio in 100 ms chunks, then finalize and return the update."""
    clip = pcm[:SHORT_FEED_SAMPLES]
    for i in range(0, len(clip), 1600):
        stream.feed(clip[i : i + 1600])
    return stream.finalize()


def test_parakeet_cache_aware_acceptance_discriminates(parakeet_stream_model_path):
    # The header's documented discrimination: the cache-aware variant accepts
    # PARAKEET_STREAM and rejects PARAKEET_BUFFERED_STREAM.
    with t.Model(parakeet_stream_model_path) as model:
        assert model.accepts(t.ParakeetStreamOptions()) is True
        assert model.accepts(t.ParakeetBufferedStreamOptions()) is False


def test_parakeet_cache_aware_streams_with_extension(
        parakeet_stream_model_path, audio_pcm):
    with t.Model(parakeet_stream_model_path) as model, model.session() as session:
        # att_context_right=-1 selects the model's default (max-accuracy) menu entry.
        with session.stream(
                family=t.ParakeetStreamOptions(att_context_right=-1)) as stream:
            update = _drive_short(stream, audio_pcm)
            text = stream.text().full
    assert update.is_final
    assert text.strip(), "cache-aware stream produced no text"


def test_parakeet_buffered_streams_with_extension(
        parakeet_buffered_model_path, audio_pcm):
    with t.Model(parakeet_buffered_model_path) as model:
        assert model.accepts(t.ParakeetBufferedStreamOptions()) is True
        # Defaults (left/chunk/right = -1) resolve to the model's menu default
        # (L=5600/C=1040/R=1040). An explicit override must be an 80 ms multiple
        # AND land on a tuple in the training menu, else stream_begin returns
        # INVALID_ARG — so the path-proving choice is the default.
        with model.session() as session, session.stream(
                family=t.ParakeetBufferedStreamOptions()) as stream:
            update = _drive_short(stream, audio_pcm)
            text = stream.text().full
    assert update.is_final
    assert text.strip(), "buffered stream produced no text"


def test_voxtral_realtime_streams_with_extension(voxtral_model_path, audio_pcm):
    with t.Model(voxtral_model_path) as model:
        assert model.accepts(t.VoxtralRealtimeStreamOptions()) is True
        with model.session() as session, session.stream(
                family=t.VoxtralRealtimeStreamOptions(num_delay_tokens=4)) as stream:
            update = _drive_short(stream, audio_pcm)
            text = stream.text().full
    assert update.is_final
    assert text.strip(), "voxtral produced no text"
