"""Load a GigaAM variant from its pinned source, and read its mel frontend.

`VARIANTS` is the one table of where each GigaAM variant comes from. The
converter, the reference dumper and the WER runner read it, and the golden
manifests under tests/golden/gigaam/ must name the same repo and commit.

Two loaders build the model from the pinned `gigaam` package:

- The v3 variants load through `gigaam.load_model`, which downloads the
  checkpoint from the package's own URL list. Their pinned commit is the
  ai-sage/GigaAM-v3 branch that publishes the same weights; it is recorded
  in reference dumps and WER reports, and nothing downloads it.
- GigaAM Multilingual Large CTC has no `gigaam.load_model` key in the
  pinned package (salute-developers/GigaAM 6e4b027c; the key arrived
  upstream in 559d88d). Its Hugging Face repo publishes `config.json`,
  which names `modeling_gigaam.<Class>` targets, and `pytorch_model.bin`,
  which holds the weights under a `model.` prefix. This module reads both
  at the pinned commit and instantiates the same classes from the pinned
  package.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VariantSource:
    """Where a GigaAM variant's weights come from."""

    repo_id: str
    revision: str
    # True: built from the Hugging Face snapshot at `revision`.
    # False: loaded by gigaam.load_model; repo_id and revision are recorded only.
    from_snapshot: bool


# gigaam variant key -> source. A branch name follows each commit.
VARIANTS: dict[str, VariantSource] = {
    "v3_e2e_rnnt": VariantSource(
        "ai-sage/GigaAM-v3", "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e", False),  # main
    "v3_e2e_ctc": VariantSource(
        "ai-sage/GigaAM-v3", "cec030b4c4f35d928e4a9044a3bdb29ebd499fac", False),  # e2e_ctc
    "v3_rnnt": VariantSource(
        "ai-sage/GigaAM-v3", "c7f128b8accdd9624df905e5c2d7b7a48c27c0d8", False),  # rnnt
    "v3_ctc": VariantSource(
        "ai-sage/GigaAM-v3", "15ef3b5a88da78f93134b3cb7f015c70aefa8946", False),  # ctc
    "multilingual_large_ctc": VariantSource(
        "ai-sage/GigaAM-Multilingual", "3905cd51c3ed4e88c8edf33f3302969ba480a327", True),  # large_ctc
}

# Class named in config.json -> module that defines it in the gigaam package.
_AUTHOR_MODULES: dict[str, str] = {
    "FeatureExtractor": "gigaam.preprocess",
    "ConformerEncoder": "gigaam.encoder",
    "CTCHead": "gigaam.decoder",
    "CTCGreedyDecoding": "gigaam.decoding",
}


def variant_source(variant_key: str, revision: str | None = None) -> VariantSource:
    """Return the pinned source of `variant_key`.

    `revision` is the commit a caller expects, such as a golden manifest's
    `source_model.hf_revision`. Raises ValueError when it differs from the
    pin, so a manifest bumped without this table fails instead of dumping
    the old commit.
    """
    try:
        source = VARIANTS[variant_key]
    except KeyError:
        raise ValueError(
            f"unknown GigaAM variant {variant_key!r}; expected one of {sorted(VARIANTS)}"
        ) from None
    if revision is not None and revision != source.revision:
        raise ValueError(
            f"{variant_key}: revision {revision} differs from the commit pinned in "
            f"scripts/lib/gigaam_hf.py VARIANTS, {source.revision}. Change the pin "
            f"and the golden manifest together."
        )
    return source


def author_model_config(hf_config: dict) -> dict:
    """Return the model cfg of a Hugging Face config.json, retargeted at the
    gigaam package.

    Raises ValueError when a section names a class this module does not map,
    so a new head or decoder type fails here instead of loading the wrong
    module.
    """
    cfg = copy.deepcopy(hf_config["cfg"]["model"]["cfg"])
    for section in ("preprocessor", "encoder", "head", "decoding"):
        target = cfg[section]["_target_"]
        module, _, name = target.rpartition(".")
        if module != "modeling_gigaam" or name not in _AUTHOR_MODULES:
            raise ValueError(f"unsupported {section} target {target!r} in config.json")
        cfg[section]["_target_"] = f"{_AUTHOR_MODULES[name]}.{name}"
    return cfg


def snapshot_dir(variant_key: str) -> Path:
    """Return the snapshot directory of a `from_snapshot` variant.

    The snapshot comes from lib.hf_source.download_snapshot at the pinned
    commit, which keeps each commit in its own directory and raises
    LocalEntryNotFoundError offline when the pinned one is absent.
    """
    from lib.hf_source import download_snapshot

    source = variant_source(variant_key)
    return download_snapshot(source.repo_id, source.revision)


def load_snapshot_model(variant_key: str):
    """Load a GigaAM ASR model in F32 on CPU from its Hugging Face snapshot."""
    import torch
    from gigaam import GigaAMASR
    from omegaconf import OmegaConf

    path = snapshot_dir(variant_key)
    print(f"Loading {variant_key} from {path}")
    hf_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    asr = GigaAMASR(OmegaConf.create(author_model_config(hf_config)))
    state_dict = torch.load(path / "pytorch_model.bin", map_location="cpu", weights_only=True)
    # The Hugging Face wrapper prefixes every key with `model.`; the strict
    # load_state_dict refuses any key left unmatched.
    asr.load_state_dict({k.removeprefix("model."): v for k, v in state_dict.items()})
    return asr.float().cpu().eval()


def load_model(variant_key: str, device: str = "cpu"):
    """Load a GigaAM variant in F32 and eval mode on `device`.

    A `from_snapshot` variant comes from its pinned Hugging Face repo and
    commit. The v3 variants load with fp16_encoder=False, since the package
    defaults to an F16 encoder.
    """
    if variant_source(variant_key).from_snapshot:
        return load_snapshot_model(variant_key).to(device)

    import gigaam

    print(f"Loading gigaam.load_model({variant_key!r}, fp16_encoder=False, device={device!r})")
    return gigaam.load_model(variant_key, fp16_encoder=False, device=device).eval()


def frontend_params(preprocessor_cfg: dict, preprocessor) -> dict:
    """Return the log-mel frontend parameters of a gigaam FeatureExtractor.

    The values come from `preprocessor`, the instantiated module, because
    FeatureExtractor defaults some of them and passes no mel scale,
    filterbank norm, or frequency range to torchaudio's MelSpectrogram.
    Raises ValueError when `preprocessor_cfg` sets one of the keys below to
    a value the module does not use (`features` for n_mels), so a GGUF never
    describes a frontend the reference does not run; other config keys are
    not checked.

    Keys: sample_rate, n_fft, win_length, hop_length, n_mels, center,
    mel_scale, mel_norm, f_min, f_max (Hz).
    """
    mel = preprocessor.featurizer[0]
    params = {
        "sample_rate": int(mel.sample_rate),
        "n_fft": int(mel.n_fft),
        "win_length": int(mel.win_length),
        "hop_length": int(mel.hop_length),
        "n_mels": int(mel.n_mels),
        "center": bool(mel.spectrogram.center),
        "mel_scale": str(mel.mel_scale.mel_scale),
        "mel_norm": mel.mel_scale.norm,
        "f_min": float(mel.f_min),
        "f_max": float(mel.mel_scale.f_max),
    }
    # config.json key -> params key, where the two differ.
    cfg_keys = {"features": "n_mels"}
    mismatches = [
        f"{key}={preprocessor_cfg[key]!r} (module: {params[cfg_keys.get(key, key)]!r})"
        for key in ("sample_rate", "n_fft", "win_length", "hop_length", "features",
                    "center", "mel_scale", "mel_norm", "f_min", "f_max")
        if key in preprocessor_cfg and preprocessor_cfg[key] != params[cfg_keys.get(key, key)]
    ]
    if mismatches:
        raise ValueError(
            "preprocessor config disagrees with the FeatureExtractor it builds: "
            + ", ".join(mismatches)
        )
    return params
