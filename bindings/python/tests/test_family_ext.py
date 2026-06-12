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
