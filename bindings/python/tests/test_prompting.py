"""Generic prompting inputs: run-params marshalling and model-gated behavior."""

import ctypes

import pytest

import transcribe_cpp as t
from transcribe_cpp import _generated


def test_run_params_carry_prompting_fields():
    params = t._build_run_params("instruct", None, None, "none", False, -1,
                                 vocabulary=["GGUF", "ggml"], prompt="Summarize.",
                                 prefix="And so")
    assert params.task == _generated.TRANSCRIBE_TASK_INSTRUCT
    assert params.n_vocabulary == 2
    assert [params.vocabulary[i] for i in range(2)] == [b"GGUF", b"ggml"]
    assert params.prompt == b"Summarize."
    assert params.prefix == b"And so"


def test_run_params_default_to_no_prompting():
    params = t._build_run_params("transcribe", None, None, "auto", False, -1)
    assert params.n_vocabulary == 0
    assert not params.vocabulary
    assert params.prompt is None and params.prefix is None


def test_vocabulary_rejects_single_string():
    with pytest.raises(t.InvalidArgument):
        t._build_run_params("transcribe", None, None, "auto", False, -1, vocabulary="GGUF")


def test_nul_in_string_option_is_rejected():
    for kwargs in ({"prompt": "a\x00b"}, {"prefix": "a\x00b"}, {"vocabulary": ["a\x00b"]}):
        with pytest.raises(t.InvalidArgument):
            t._build_run_params("transcribe", None, None, "auto", False, -1, **kwargs)


def test_vocabulary_accepts_iterators_rejects_sets():
    params = t._build_run_params("transcribe", None, None, "auto", False, -1,
                                 vocabulary=(w for w in ["GGUF", "ggml"]))
    assert params.n_vocabulary == 2
    with pytest.raises(t.InvalidArgument):
        t._build_run_params("transcribe", None, None, "auto", False, -1, vocabulary={"GGUF"})


def test_prompting_features_probe(model_path):
    with t.Model(model_path, backend="cpu") as model:
        for feature in ("vocabulary", "context_prompt", "instruct", "transcript_prefix"):
            assert isinstance(model.supports(feature), bool)


def test_unsupported_prefix_raises(streaming_model_path, audio_pcm):
    with t.Model(streaming_model_path, backend="cpu") as model, model.session() as session:
        if model.supports("transcript_prefix"):
            pytest.skip("model supports a transcript prefix")
        with pytest.raises(t.InvalidArgument):
            session.run(audio_pcm, prefix="And so")


def test_stream_prompting(streaming_model_path):
    with t.Model(streaming_model_path, backend="cpu") as model, model.session() as session:
        with pytest.raises(t.UnsupportedRequest):
            session.stream(task="instruct", prompt="Summarize.")
        with session.stream(vocabulary=["Kennedy", "Americans"], prompt="A speech."):
            pass


def _whisper(model):
    return model.arch == "whisper"


def test_whisper_prefix_contract(model_path, audio_pcm):
    """The prefix is forced decoder text: text holds only the continuation,
    raw_text leads with the prefix, and nothing is duplicated."""
    prefix = "And so my fellow Americans,"
    with t.Model(model_path, backend="cpu") as model, model.session() as session:
        if not _whisper(model):
            pytest.skip("whisper-specific rendering")
        assert model.supports("transcript_prefix")
        res = session.run(audio_pcm, prefix=prefix)
    assert res.raw_text.strip().startswith(prefix)
    assert not res.text.lower().startswith("and so")
    assert "ask not" in res.text.lower()
    # Whisper's timestamp rules do not compose with a prefix.
    with t.Model(model_path, backend="cpu") as model, model.session() as session:
        with pytest.raises(t.InvalidArgument):
            session.run(audio_pcm, prefix=prefix, timestamps="segment")


def test_whisper_vocabulary_and_context(model_path, audio_pcm):
    with t.Model(model_path, backend="cpu") as model, model.session() as session:
        if not _whisper(model):
            pytest.skip("whisper-specific rendering")
        assert model.supports("vocabulary") and model.supports("context_prompt")
        res = session.run(audio_pcm, vocabulary=["Kennedy", "Americans"],
                          prompt="An inaugural address.")
        assert "country" in res.text.lower()
        batch = session.run_batch([audio_pcm, audio_pcm], vocabulary=["Kennedy", "Americans"])
        assert len(batch) == 2 and all("country" in r.text.lower() for r in batch)
        # The whisper extension's prompt and the generic fields share one slot.
        with pytest.raises(t.InvalidArgument):
            session.run(audio_pcm, vocabulary=["Kennedy"],
                        family=t.WhisperRunOptions(initial_prompt="x"))


def test_control_token_literal_rejected(model_path, audio_pcm):
    """A control-token literal is rejected and leaves the previous result."""
    with t.Model(model_path, backend="cpu") as model, model.session() as session:
        if not _whisper(model):
            pytest.skip("whisper-specific rendering")
        first = session.run(audio_pcm)
        with pytest.raises(t.InvalidArgument):
            session.run(audio_pcm, prompt="hello <|endoftext|>")
        assert session._materialize(session._h).text == first.text
