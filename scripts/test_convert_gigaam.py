#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gguf>=0.10",
#   "numpy",
# ]
# ///
"""Unit tests for convert-gigaam.py: the VARIANT_PROFILES table, the source
of the general.* identity block each GigaAM GGUF carries, the refusal of a
source other than the pinned repo, and the frontend values read_hparams
takes from the FeatureExtractor.

Run standalone (exit-code driven):   uv run scripts/test_convert_gigaam.py
Or under pytest:                      pytest scripts/test_convert_gigaam.py
"""

from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_SCRIPTS = Path(__file__).resolve().parent
_GOLDEN = _SCRIPTS.parent / "tests" / "golden" / "gigaam"

_spec = importlib.util.spec_from_file_location("convert_gigaam", _SCRIPTS / "convert-gigaam.py")
convert_gigaam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_gigaam)


class VariantProfilesTest(unittest.TestCase):
    def test_each_slug_has_its_general_identity(self):
        identity = {
            slug: (p["basename"], p["version"], p["languages"])
            for slug, p in convert_gigaam.VARIANT_PROFILES.items()
        }
        self.assertEqual(identity, {
            "gigaam-v3-e2e-rnnt": ("gigaam-v3", "v3", ["ru"]),
            "gigaam-v3-e2e-ctc": ("gigaam-v3", "v3", ["ru"]),
            "gigaam-v3-rnnt": ("gigaam-v3", "v3", ["ru"]),
            "gigaam-v3-ctc": ("gigaam-v3", "v3", ["ru"]),
            "gigaam-multilingual-large-ctc": (
                "gigaam-multilingual", None, ["ru", "en", "kk", "ky", "uz"]),
        })

    def test_every_profile_matches_the_variant_key_and_languages_of_its_golden_manifest(self):
        manifests = {}
        for path in sorted(_GOLDEN.glob("*.manifest.json")):
            m = json.loads(path.read_text(encoding="utf-8"))
            manifests[m["variant"]] = (m["reference"]["variant_key"], m["capabilities"]["languages"])
        profiles = {
            slug: (p["variant_key"], p["languages"])
            for slug, p in convert_gigaam.VARIANT_PROFILES.items()
        }
        self.assertEqual(profiles, manifests)


class MainTest(unittest.TestCase):
    def run_main(self, model: str, slug: str) -> tuple[int, str, list]:
        """Run main() with convert() replaced, and return the exit code,
        stderr, and the arguments convert() was called with."""
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d, redirect_stderr(err), \
                mock.patch.object(convert_gigaam, "convert") as convert:
            code = convert_gigaam.main(
                ["convert-gigaam.py", model, str(Path(d) / "out.gguf"), "--repo-id", slug])
        return code, err.getvalue().strip(), [c.args[:2] for c in convert.call_args_list]

    def test_the_pinned_repo_is_converted(self):
        self.assertEqual(
            self.run_main("ai-sage/GigaAM-Multilingual", "gigaam-multilingual-large-ctc"),
            (0, "", [("multilingual_large_ctc", "gigaam-multilingual-large-ctc")]),
        )

    def test_a_v3_variant_takes_any_model_argument(self):
        self.assertEqual(
            self.run_main("any/informational-id", "gigaam-v3-ctc"),
            (0, "", [("v3_ctc", "gigaam-v3-ctc")]),
        )

    def test_another_repo_or_a_local_directory_is_refused_before_converting(self):
        for model in ("ai-sage/GigaAM-v3", tempfile.gettempdir()):
            with self.subTest(model=model):
                self.assertEqual(
                    self.run_main(model, "gigaam-multilingual-large-ctc"),
                    (2, "error: multilingual_large_ctc converts from the Hugging Face repo "
                        "ai-sage/GigaAM-Multilingual at 3905cd51c3ed4e88c8edf33f3302969ba480a327, "
                        f"got {model!r}", []),
                )


class ReadHparamsTest(unittest.TestCase):
    def test_the_frontend_values_come_from_the_feature_extractor(self):
        # A CTC config with the keys read_hparams reads, and the Multilingual
        # preprocessor, which lists no center, mel scale, norm, or range.
        config = {
            "preprocessor": {"sample_rate": 16000, "features": 64, "win_length": 320,
                             "hop_length": 160, "n_fft": 320},
            "encoder": {"n_layers": 24, "d_model": 1024, "n_heads": 16,
                        "ff_expansion_factor": 4, "conv_kernel_size": 5,
                        "subsampling_factor": 4, "subs_kernel_size": 5,
                        "pos_emb_max_len": 5000, "feat_in": 64,
                        "self_attention_model": "rotary", "conv_norm_type": "layer_norm"},
            "head": {"_target_": "gigaam.decoder.CTCHead", "feat_in": 1024, "num_classes": 71},
        }
        # Stand-in for FeatureExtractor: torchaudio's MelSpectrogram keeps
        # f_max=None and its MelScale holds the resolved value.
        mel = SimpleNamespace(
            sample_rate=16000, n_fft=320, win_length=320, hop_length=160, n_mels=64,
            f_min=20.0, f_max=None,
            spectrogram=SimpleNamespace(center=True),
            mel_scale=SimpleNamespace(mel_scale="htk", norm=None, f_max=7600.0),
        )
        model = SimpleNamespace(preprocessor=SimpleNamespace(featurizer=[mel, SimpleNamespace()]))
        hp = convert_gigaam.read_hparams(config, model)
        frontend = {k: hp[k] for k in ("fe_center", "fe_f_min", "fe_f_max", "fe_mel_norm")}
        self.assertEqual(frontend, {
            "fe_center": True, "fe_f_min": 20.0, "fe_f_max": 7600.0, "fe_mel_norm": "htk",
        })


if __name__ == "__main__":
    unittest.main()
