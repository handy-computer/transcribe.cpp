#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gguf>=0.10",
#   "huggingface-hub>=0.30",
#   "numpy",
# ]
# ///
"""Unit tests for lib.gigaam_hf, the pinned source table and loader of the
GigaAM variants.

Covers the parts that need neither torch nor the weights: the source table
against the golden manifests, the config retargeting, the snapshot fetch at
the pinned commit, the frontend read from a stand-in FeatureExtractor, and
the reference dumper's refusal of a revision or repo other than the pin.

Run standalone (exit-code driven):   uv run scripts/lib/test_gigaam_hf.py
Or under pytest:                      pytest scripts/lib/test_gigaam_hf.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# Read by huggingface_hub at import; lib.hf_source imports it lazily.
os.environ["HF_HUB_OFFLINE"] = "1"

from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))
from lib.gigaam_hf import (  # noqa: E402
    VARIANTS,
    VariantSource,
    author_model_config,
    frontend_params,
    snapshot_dir,
    variant_source,
)

_GOLDEN = _SCRIPTS.parent / "tests" / "golden" / "gigaam"
_ML_COMMIT = "3905cd51c3ed4e88c8edf33f3302969ba480a327"


def _hf_config(head_target: str = "modeling_gigaam.CTCHead") -> dict:
    """config.json in the shape of the ai-sage GigaAM repos, trimmed to the
    fields the loader reads or must carry through."""
    return {
        "model_type": "gigaam",
        "cfg": {
            "model": {
                "_target_": "modeling_gigaam.GigaAMASR",
                "cfg": {
                    "model_name": "multilingual_large_ctc",
                    "preprocessor": {"_target_": "modeling_gigaam.FeatureExtractor", "features": 64},
                    "encoder": {"_target_": "modeling_gigaam.ConformerEncoder", "n_layers": 24},
                    "head": {"_target_": head_target, "num_classes": 71},
                    "decoding": {
                        "_target_": "modeling_gigaam.CTCGreedyDecoding",
                        "vocabulary": [" ", "'", "a", "\u0451", "\u049b"],
                    },
                },
            }
        },
    }


class SourceTableTest(unittest.TestCase):
    def test_every_golden_manifest_names_the_pinned_source_of_its_variant(self):
        manifests = sorted(_GOLDEN.glob("*.manifest.json"))
        self.assertTrue(manifests, f"no manifests under {_GOLDEN}")
        named = {}
        for path in manifests:
            m = json.loads(path.read_text(encoding="utf-8"))
            key = m["reference"]["variant_key"]
            src = m["source_model"]
            named[key] = (src["hf_repo"], src["hf_revision"])
        pinned = {k: (v.repo_id, v.revision) for k, v in VARIANTS.items()}
        self.assertEqual(named, pinned)

    def test_only_multilingual_large_ctc_is_built_from_its_snapshot(self):
        from_snapshot = sorted(k for k, v in VARIANTS.items() if v.from_snapshot)
        self.assertEqual(from_snapshot, ["multilingual_large_ctc"])

    def test_the_pinned_revision_is_accepted(self):
        self.assertEqual(
            variant_source("multilingual_large_ctc", _ML_COMMIT),
            VariantSource("ai-sage/GigaAM-Multilingual", _ML_COMMIT, True),
        )

    def test_a_revision_other_than_the_pin_is_refused(self):
        other = "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8"
        with self.assertRaisesRegex(ValueError, f"{other} differs from .* {_ML_COMMIT}"):
            variant_source("multilingual_large_ctc", other)

    def test_an_unknown_variant_is_refused(self):
        with self.assertRaisesRegex(ValueError, "'v3_ssl'"):
            variant_source("v3_ssl")


class AuthorModelConfigTest(unittest.TestCase):
    def test_ctc_config_targets_the_gigaam_package_modules(self):
        cfg = author_model_config(_hf_config())
        targets = {s: cfg[s]["_target_"] for s in ("preprocessor", "encoder", "head", "decoding")}
        self.assertEqual(targets, {
            "preprocessor": "gigaam.preprocess.FeatureExtractor",
            "encoder": "gigaam.encoder.ConformerEncoder",
            "head": "gigaam.decoder.CTCHead",
            "decoding": "gigaam.decoding.CTCGreedyDecoding",
        })
        self.assertEqual(cfg["decoding"]["vocabulary"], [" ", "'", "a", "\u0451", "\u049b"])

    def test_retargeting_leaves_the_input_config_unchanged(self):
        hf = _hf_config()
        author_model_config(hf)
        self.assertEqual(hf["cfg"]["model"]["cfg"]["head"]["_target_"], "modeling_gigaam.CTCHead")

    def test_a_head_class_without_a_mapping_is_refused(self):
        with self.assertRaisesRegex(ValueError, "'modeling_gigaam.RNNTHead'"):
            author_model_config(_hf_config(head_target="modeling_gigaam.RNNTHead"))

    def test_a_target_outside_modeling_gigaam_is_refused(self):
        with self.assertRaisesRegex(ValueError, "'other_module.CTCHead'"):
            author_model_config(_hf_config(head_target="other_module.CTCHead"))


def _models_dir_with(root: Path, commit: str) -> Path:
    """Return a TRANSCRIBE_MODELS_DIR holding config.json of GigaAM-Multilingual
    downloaded at `commit`, in the directory download_snapshot keeps for it."""
    (root / f"GigaAM-Multilingual@{commit}").mkdir(parents=True)
    (root / f"GigaAM-Multilingual@{commit}" / "config.json").write_text("{}", encoding="utf-8")
    return root


class SnapshotDirTest(unittest.TestCase):
    def test_a_local_copy_of_the_pinned_commit_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            models = _models_dir_with(Path(d), _ML_COMMIT)
            with mock.patch.dict(os.environ, {"TRANSCRIBE_MODELS_DIR": str(models)}):
                got = snapshot_dir("multilingual_large_ctc")
            self.assertEqual(got, (models / f"GigaAM-Multilingual@{_ML_COMMIT}").resolve())

    def test_a_local_copy_of_another_commit_is_not_used(self):
        other = "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8"
        with tempfile.TemporaryDirectory() as d:
            models = _models_dir_with(Path(d), other)
            with mock.patch.dict(os.environ, {"TRANSCRIBE_MODELS_DIR": str(models)}):
                with self.assertRaises(LocalEntryNotFoundError):
                    snapshot_dir("multilingual_large_ctc")


def _feature_extractor(*, center: bool = False, mel_scale: str = "htk", norm=None, f_max: float = 8000.0):
    """Stand-in for gigaam's FeatureExtractor: the torchaudio MelSpectrogram
    attributes frontend_params reads, at the values of the GigaAM
    checkpoints."""
    # MelSpectrogram keeps f_max=None; MelScale holds the resolved value.
    mel = SimpleNamespace(
        sample_rate=16000, n_fft=320, win_length=320, hop_length=160, n_mels=64,
        f_min=0.0, f_max=None,
        spectrogram=SimpleNamespace(center=center),
        mel_scale=SimpleNamespace(mel_scale=mel_scale, norm=norm, f_max=f_max),
    )
    return SimpleNamespace(featurizer=[mel, SimpleNamespace()])


# config.json preprocessor of the ai-sage/GigaAM-v3 branches.
_V3_PREPROCESSOR = {
    "sample_rate": 16000, "features": 64, "win_length": 320, "hop_length": 160,
    "mel_scale": "htk", "n_fft": 320, "mel_norm": None, "center": False,
}
# The ai-sage/GigaAM-Multilingual large_ctc preprocessor lists no mel scale or norm.
_ML_PREPROCESSOR = {
    "sample_rate": 16000, "features": 64, "win_length": 320, "hop_length": 160,
    "n_fft": 320, "center": False,
}
_EXPECTED_FRONTEND = {
    "sample_rate": 16000, "n_fft": 320, "win_length": 320, "hop_length": 160,
    "n_mels": 64, "center": False, "mel_scale": "htk", "mel_norm": None,
    "f_min": 0.0, "f_max": 8000.0,
}


class FrontendParamsTest(unittest.TestCase):
    def test_a_config_that_lists_every_value_reads_them_from_the_module(self):
        self.assertEqual(frontend_params(_V3_PREPROCESSOR, _feature_extractor()), _EXPECTED_FRONTEND)

    def test_a_config_without_mel_scale_takes_the_module_scale(self):
        self.assertEqual(frontend_params(_ML_PREPROCESSOR, _feature_extractor()), _EXPECTED_FRONTEND)

    def test_a_config_mel_scale_the_module_does_not_use_is_refused(self):
        cfg = {**_V3_PREPROCESSOR, "mel_scale": "slaney"}
        with self.assertRaisesRegex(ValueError, r"mel_scale='slaney' \(module: 'htk'\)"):
            frontend_params(cfg, _feature_extractor())

    def test_a_config_without_center_takes_the_module_value(self):
        cfg = {k: v for k, v in _ML_PREPROCESSOR.items() if k != "center"}
        got = frontend_params(cfg, _feature_extractor(center=True))
        self.assertEqual(got["center"], True)

    def test_the_upper_mel_frequency_comes_from_the_module(self):
        got = frontend_params(_ML_PREPROCESSOR, _feature_extractor(f_max=7600.0))
        self.assertEqual((got["f_min"], got["f_max"]), (0.0, 7600.0))

    def test_a_config_f_max_the_module_does_not_use_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"f_max=7600.0 \(module: 8000.0\)"):
            frontend_params({**_ML_PREPROCESSOR, "f_max": 7600.0}, _feature_extractor())

    def test_a_config_mel_count_the_module_does_not_use_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"features=80 \(module: 64\)"):
            frontend_params({**_ML_PREPROCESSOR, "features": 80}, _feature_extractor())

    def test_a_config_mel_norm_the_module_does_not_use_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"mel_norm='slaney' \(module: None\)"):
            frontend_params({**_V3_PREPROCESSOR, "mel_norm": "slaney"}, _feature_extractor())

    def test_a_config_center_the_module_does_not_use_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"center=False \(module: True\)"):
            frontend_params(_ML_PREPROCESSOR, _feature_extractor(center=True))


def _dumper():
    """Import dump_reference_gigaam_author.py, which has no package name."""
    spec = importlib.util.spec_from_file_location(
        "dump_reference_gigaam_author", _SCRIPTS / "dump_reference_gigaam_author.py")
    dumper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dumper)
    return dumper


class _Loaded(Exception):
    """Raised in place of loading the model, once every check has passed."""


class DumperSourceTest(unittest.TestCase):
    def decode(self, *extra: str) -> None:
        dumper = _dumper()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(dumper, "load_reference", side_effect=_Loaded), \
                mock.patch.dict(sys.modules, {"torch": mock.MagicMock()}):
            dumper.main(["decode", "--variant", "multilingual_large_ctc",
                         "--audio", str(Path(d) / "a.wav"), "--out", d, *extra])

    def test_the_pinned_repo_and_revision_reach_the_load(self):
        for extra in ((), ("--model", "ai-sage/GigaAM-Multilingual", "--revision", _ML_COMMIT)):
            with self.subTest(extra=extra), self.assertRaises(_Loaded):
                self.decode(*extra)

    def test_a_revision_other_than_the_pin_stops_the_dump_before_loading(self):
        other = "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8"
        with self.assertRaisesRegex(SystemExit, f"error: .*{other} differs from .* {_ML_COMMIT}"):
            self.decode("--revision", other)

    def test_a_repo_other_than_the_pinned_one_stops_the_dump_before_loading(self):
        with self.assertRaisesRegex(
            SystemExit, "ai-sage/GigaAM-Multilingual at 3905cd51.*got --model 'ai-sage/GigaAM-v3'"
        ):
            self.decode("--model", "ai-sage/GigaAM-v3")


if __name__ == "__main__":
    unittest.main()
