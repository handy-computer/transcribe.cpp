"""LANGID role: roles, info, label table, LangIdSession.run and its allowed
contract. Contract tests run on the toy ecapa_tdnn fixture (always
available); accuracy tests need the real VoxLingua107 GGUF. The locking /
Busy / close / cancel rules are pinned model-free in test_compute_lock.py."""

from __future__ import annotations

import array
import math

import pytest

import transcribe_cpp as t
from conftest import SAMPLES, load_wav


def _noise(n: int, seed: int = 7) -> array.array:
    out = array.array("f")
    s = seed | 1
    for _ in range(n):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        out.append(((s >> 8) & 0xFFFFFF) / 16777216.0 - 0.5)
    return out


@pytest.fixture(scope="module")
def noise_1s():
    return _noise(16000)


def test_toy_roles_info_labels(langid_toy_model_path):
    with t.Model(langid_toy_model_path, backend="cpu") as model:
        assert model.roles == {t.Role.LANGID}
        assert model.langid_info == t.LangIdInfo(sample_rate=16000, n_labels=5,
                                                 min_audio_ms=500, max_audio_ms=30000)
        assert model.langid_labels[2] == ("cc", "Charlie")
        assert model.langid_label_index("xx") == 0
        assert model.langid_label_index("zz") is None
        with pytest.raises(t.UnsupportedRole):
            model.capabilities
        with pytest.raises(t.UnsupportedRole):
            model.session()
        with pytest.raises(t.UnsupportedRole):
            model.diarize_session()


def test_asr_only_model_rejects_langid(model_path):
    with t.Model(model_path) as model:
        with pytest.raises(t.UnsupportedRole):
            model.langid_info
        with pytest.raises(t.UnsupportedRole):
            model.langid_session()
        assert model.langid_label_index("en") is None


def test_toy_run_contract(langid_toy_model_path, noise_1s):
    with t.Model(langid_toy_model_path, backend="cpu") as model:
        with model.langid_session(n_threads=1) as lid:
            r = lid.run(noise_1s)
            assert len(r.candidates) == 5  # unrestricted: every label
            assert {c.code for c in r.candidates} == {c for c, _ in model.langid_labels}
            assert r.allowed_mass == 1.0
            assert r.code == r.candidates[0].code
            assert math.isclose(sum(c.p for c in r.candidates), 1.0, rel_tol=1e-5)
            ps = [c.p for c in r.candidates]
            assert ps == sorted(ps, reverse=True)

            restricted = lid.run(noise_1s, allowed=["bb", "dd"])
            assert {c.code for c in restricted.candidates} == {"bb", "dd"}
            assert len(restricted.candidates) == 2 and 0.0 < restricted.allowed_mass < 1.0

            alias = lid.run(noise_1s, allowed=("xx",))
            assert [c.code for c in alias.candidates] == ["aa"] and alias.candidates[0].p == 1.0

            assert lid.timings.encode_ms > 0.0


def test_toy_allowed_rejections(langid_toy_model_path, noise_1s):
    with t.Model(langid_toy_model_path, backend="cpu") as model:
        with model.langid_session() as lid:
            # An empty list must not silently mean "all labels".
            with pytest.raises(t.InvalidArgument):
                lid.run(noise_1s, allowed=[])
            with pytest.raises(t.InvalidArgument):
                lid.run(noise_1s, allowed="en")
            with pytest.raises(t.InvalidArgument):
                lid.run(noise_1s, allowed=["aa", None])
            with pytest.raises(t.UnsupportedRequest):
                lid.run(noise_1s, allowed=["zz"])
            lid.run(noise_1s, allowed=None)  # None is every label


def test_toy_interior_nul_is_rejected(langid_toy_model_path, noise_1s):
    # C would cut "bb\0zz" to "bb"; the code must not silently narrow.
    with t.Model(langid_toy_model_path, backend="cpu") as model:
        with pytest.raises(t.InvalidArgument):
            model.langid_label_index("bb\0zz")
        with model.langid_session() as lid:
            with pytest.raises(t.InvalidArgument):
                lid.run(noise_1s, allowed=["bb\0zz"])


def test_toy_input_rules(langid_toy_model_path, noise_1s):
    with t.Model(langid_toy_model_path, backend="cpu") as model:
        with model.langid_session() as lid:
            with pytest.raises(t.InputTooShort):
                lid.run(noise_1s[:6400])  # 400 ms
            assert lid.run(noise_1s[:8000]).candidates  # exactly min_audio_ms
            # Past max_audio_ms the head is scored, not rejected.
            assert len(lid.run(_noise(16000 * 31)).candidates) == 5
            bad = array.array("f", noise_1s)
            bad[10] = float("nan")
            with pytest.raises(t.InvalidArgument):
                lid.run(bad)


def test_toy_cancel(langid_toy_model_path, noise_1s):
    with t.Model(langid_toy_model_path, backend="cpu") as model:
        with model.langid_session() as lid:
            lid.cancel()
            # run() clears a stale cancel before it starts, so it completes.
            assert lid.run(noise_1s).candidates


@pytest.mark.parametrize("code", ["en", "de", "fr", "es", "ja", "zh", "ru", "id"])
def test_real_fleurs_top1(langid_model_path, code):
    pcm = load_wav(SAMPLES / f"fleurs-{code}.wav")
    with t.Model(langid_model_path, backend="cpu") as model:
        assert model.langid_info.n_labels == 107
        assert model.langid_label_index("he") == model.langid_label_index("iw")
        with model.langid_session() as lid:
            r = lid.run(pcm)
    assert len(r.candidates) == 107
    assert r.code == code and r.candidates[0].p >= 0.5


def test_real_allowed_mass_signals_out_of_set(langid_model_path):
    pcm = load_wav(SAMPLES / "fleurs-ja.wav")
    with t.Model(langid_model_path, backend="cpu") as model:
        with model.langid_session() as lid:
            r = lid.run(pcm, allowed=["en", "de"])
    assert len(r.candidates) == 2
    assert r.code in ("en", "de") and r.allowed_mass < 0.01
