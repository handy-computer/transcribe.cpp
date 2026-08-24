"""Model-gated generic punctuation/capitalization and ITN controls."""

import transcribe_cpp as t


def test_pnc_changes_canary_prompt(pnc_model_path, audio_pcm):
    with t.Model(pnc_model_path, backend="cpu") as model, model.session() as session:
        assert model.supports("pnc")
        default = session.run(audio_pcm, language="en", pnc="default").text
        enabled = session.run(audio_pcm, language="en", pnc="on").text
        disabled = session.run(audio_pcm, language="en", pnc="off").text

    assert default == enabled
    assert disabled != enabled
    assert disabled == disabled.lower()


def test_itn_changes_sensevoice_text_normalization(itn_model_path, audio_pcm):
    with t.Model(itn_model_path, backend="cpu") as model, model.session() as session:
        assert model.supports("itn")
        default = session.run(audio_pcm, language="en", itn="default")
        disabled = session.run(audio_pcm, language="en", itn="off")
        enabled = session.run(audio_pcm, language="en", itn="on")

    assert (default.text, default.raw_text) == (disabled.text, disabled.raw_text)
    assert enabled.text != disabled.text
    assert "<|woitn|>" in disabled.raw_text
    assert "<|withitn|>" in enabled.raw_text
