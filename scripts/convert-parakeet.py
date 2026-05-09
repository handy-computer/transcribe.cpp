#!/usr/bin/env python3
"""
convert-parakeet.py - convert a NeMo Parakeet TDT model into a GGUF
that transcribe.cpp's loader can ingest end-to-end. Loads directly from
NeMo (HuggingFace repo id or local .nemo archive). The converter
preserves fp32 source weights; use tools/transcribe-quantize for
deployment quantization.

Run through the repo-local Parakeet reference environment so that
nemo_toolkit is available:

    uv run --project scripts/envs/parakeet \
      scripts/convert-parakeet.py nvidia/parakeet-tdt-0.6b-v2

Source format:
    An HF model name (e.g. "nvidia/parakeet-tdt-0.6b-v2") that NeMo's
    ASRModel.from_pretrained() can resolve, or a local path to a .nemo
    archive or extracted directory.

Target format:
    A single .gguf following the canonical names + shapes encoded in
    src/arch/parakeet/weights.cpp. The loader's per-tensor shape
    validation is the only schema cross-check; if the converter and the
    loader drift, the loader logs the offending tensor on first load.

Layout notes:
    NeMo ships pure PyTorch tensors, so Conv2d/Conv1d weights are
    already in the OIHW / [O, I_per_group, k] layouts the loader
    expects — no transposes here.

    PyTorch's LSTM stores per-layer weights as
    weight_ih_l{i} / weight_hh_l{i} and *two* bias vectors
    (bias_ih_l{i}, bias_hh_l{i}) that are both added to the gate
    pre-activation. We collapse the two biases into a single
    `pred.lstm.{i}.bias` = bias_ih + bias_hh, matching the loader's
    single-bias expectation. Gate order (i, f, g, o) is PyTorch's
    native order; the decoder slices them accordingly.

KV emitted (matches transcribe::read_capability_kv,
read_languages_kv, transcribe::Tokenizer::load, and
transcribe::parakeet::read_parakeet_hparams):

  general.architecture = "parakeet"
  general.basename     = "parakeet-tdt"
  general.size_label   = profile["size_label"]   (e.g. "0.6B" or "1.1B")
  general.version      = profile["version"]      (e.g. "v2", "v3", or "v1")
  general.languages    = [...]                 (1 entry for English-only,
                                                 25 for v3 multilingual)
  stt.variant          = profile["variant"]      (e.g. "tdt-0.6b-v2",
                                                 "tdt-0.6b-v3", "tdt-1.1b")
  stt.capability.lang_detect = true            (v3 only; absent otherwise)
  tokenizer.ggml.model = "bpe"
  tokenizer.ggml.tokens / scores / token_type / *_token_id  (the standard set)
  stt.parakeet.encoder.{n_layers,d_model,n_heads,d_ff,conv_kernel,
                        subsampling_factor,subsampling_channels,
                        pos_emb_max_len,use_bias}
  stt.parakeet.predictor.{hidden,n_layers,vocab}
  stt.parakeet.joint.{hidden,num_extra_outputs,activation}
  stt.parakeet.tdt.{durations,max_symbols}
  stt.frontend.{type,num_mels,sample_rate,n_fft,win_length,hop_length,
                window,normalize,dither,pre_emphasis,f_min,f_max}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from gguf import GGUFWriter, LlamaFileType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_BYTE,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN,
    TOKEN_TYPE_UNUSED,
    gguf_name,
    safe_id,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Reference dtype
# ---------------------------------------------------------------------------

REFERENCE_DTYPE_LABEL = "F32"
REFERENCE_FILE_TYPE = LlamaFileType.ALL_F32


# ---------------------------------------------------------------------------
# Variant profiles
# ---------------------------------------------------------------------------
#
# Each variant entry carries the bits the converter cannot derive from
# the state_dict or model.cfg: the variant string, the language list
# (NeMo's config doesn't carry it cleanly), the size label, and the
# capability flags. Profiles are keyed by the model slug (HF repo
# basename or output-directory name); vocab_size alone cannot
# distinguish e.g. tdt-0.6b-v2 from tdt-1.1b — both ship with a
# 1024-token SPM. expected_vocab_size is cross-checked against the
# loaded model so a slug/profile mismatch fails fast.

# v3 multilingual: 25 European languages, BCP-47 short codes. List
# matches the NVIDIA model card for nvidia/parakeet-tdt-0.6b-v3.
V3_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl",
    "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt",
    "mt", "pl", "pt", "ro", "ru",
    "sk", "sl", "es", "sv", "uk",
]

VARIANT_PROFILES: dict[str, dict] = {
    # v2: 0.6B English-only TDT.
    "parakeet-tdt-0.6b-v2": {
        "variant": "tdt-0.6b-v2",
        "version": "v2",
        "size_label": "0.6B",
        "head_kind": "tdt",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
    },
    # v3: 0.6B multilingual TDT.
    "parakeet-tdt-0.6b-v3": {
        "variant": "tdt-0.6b-v3",
        "version": "v3",
        "size_label": "0.6B",
        "head_kind": "tdt",
        "expected_vocab_size": 8192,
        "languages": V3_LANGUAGES,
        "lang_detect": True,
    },
    # 1.1B English-only TDT. Predates the v2/v3 split; the upstream
    # repo carries no version suffix, so general.version is "v1".
    "parakeet-tdt-1.1b": {
        "variant": "tdt-1.1b",
        "version": "v1",
        "size_label": "1.1B",
        "head_kind": "tdt",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
    },
    # 0.6B English RNNT. Pure transducer, no TDT durations head.
    "parakeet-rnnt-0.6b": {
        "variant": "rnnt-0.6b",
        "version": "v1",
        "size_label": "0.6B",
        "basename": "parakeet-rnnt",
        "head_kind": "rnnt",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
    },
    # 1.1B English RNNT.
    "parakeet-rnnt-1.1b": {
        "variant": "rnnt-1.1b",
        "version": "v1",
        "size_label": "1.1B",
        "basename": "parakeet-rnnt",
        "head_kind": "rnnt",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
    },
    # 0.6B English unified offline+streaming RNNT. v1 transcribe.cpp
    # port targets OFFLINE only; streaming uses the same weights and
    # is deferred until streaming infra lands.
    "parakeet-unified-en-0.6b": {
        "variant": "unified-en-0.6b",
        "version": "v1",
        "size_label": "0.6B",
        "basename": "parakeet-rnnt",
        "head_kind": "rnnt",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
    },
    # 0.6B English CTC. No predictor, no joint — encoder feeds a
    # single 1x1 conv (decoder.decoder_layers.0) projecting d_model
    # to vocab+1.
    "parakeet-ctc-0.6b": {
        "variant": "ctc-0.6b",
        "version": "v1",
        "size_label": "0.6B",
        "basename": "parakeet-ctc",
        "head_kind": "ctc",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
    },
    # 1.1B English CTC.
    "parakeet-ctc-1.1b": {
        "variant": "ctc-1.1b",
        "version": "v1",
        "size_label": "1.1B",
        "basename": "parakeet-ctc",
        "head_kind": "ctc",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
    },
    # 110M English hybrid TDT+CTC. Shipped as TDT-only at runtime
    # per the family-level Open-decisions #1 ("the pure ctc-* variants
    # cover the CTC path; drop the hybrid's auxiliary CTC head"). The
    # converter walks the TDT path and silently drops the
    # `ctc_decoder.*` tensors. head_kind="tdt" keeps the loader on
    # the existing (Stage 4) TDT codepath.
    "parakeet-tdt_ctc-110m": {
        "variant": "tdt_ctc-110m",
        "version": "v1",
        "size_label": "110M",
        "head_kind": "tdt",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
    },
    # 1.1B English hybrid TDT+CTC. Same TDT-only shipping decision.
    # Marked `prefer_direct_load=True` so the converter bypasses
    # NeMo's restore_from() extraction (~4.4 GB transient disk) and
    # reads the cached .nemo archive in place.
    "parakeet-tdt_ctc-1.1b": {
        "variant": "tdt_ctc-1.1b",
        "version": "v1",
        "size_label": "1.1B",
        "head_kind": "tdt",
        "expected_vocab_size": 1024,
        "languages": ["en"],
        "lang_detect": False,
        "prefer_direct_load": True,
    },
}


# ---------------------------------------------------------------------------
# NeMo model loading
# ---------------------------------------------------------------------------


class _DirectNemoArchive:
    """Lightweight stand-in for a constructed NeMo ASRModel, used when
    NeMo's installed ConformerEncoder rejects streaming kwargs the
    archive's YAML carries (parakeet-unified-en-0.6b's
    `att_chunk_context_size` is one such case in NeMo 2.7.x). Exposes
    only the surface the converter actually consumes:

        .cfg                     — OmegaConf dict
        .state_dict()            — torch tensor dict
        .tokenizer.tokenizer     — sentencepiece-loaded SPM
        .joint                   — None (resolve_runtime_hparams falls
                                    back to state_dict shapes)
        .eval()                  — no-op
    """

    def __init__(self, cfg, state_dict, sp_proto: bytes):
        from omegaconf import OmegaConf
        import sentencepiece as spm

        self.cfg = OmegaConf.create(cfg)
        self._sd = state_dict
        sp = spm.SentencePieceProcessor()
        sp.LoadFromSerializedProto(sp_proto)

        class _TokWrap:
            def __init__(self, sp):
                self.tokenizer = _TokWrap._SerWrap(sp)

            class _SerWrap:
                def __init__(self, sp):
                    self._sp = sp

                def serialized_model_proto(self):
                    return self._sp.serialized_model_proto()

                # forward the SentencePiece surface used by extract_tokenizer
                def __getattr__(self, name):
                    return getattr(self._sp, name)

        self.tokenizer = _TokWrap(sp)
        self.joint = None  # resolve_runtime_hparams falls back to sd shapes

    def state_dict(self):
        return self._sd

    def eval(self):
        return self


def _load_nemo_archive_directly(nemo_path: Path):
    """Open a .nemo tar archive and pull out (cfg_dict, state_dict, sp_proto)
    without instantiating any NeMo class. The .nemo layout is stable
    across NeMo versions: model_config.yaml, model_weights.ckpt, and a
    SPM model file with a hash prefix and `_tokenizer.model` suffix.
    """
    import tarfile, io
    import yaml
    import torch

    with tarfile.open(nemo_path) as tf:
        names = tf.getnames()

        def _read(suffix: str) -> bytes:
            for n in names:
                if n.endswith(suffix):
                    return tf.extractfile(n).read()
            raise FileNotFoundError(f"{nemo_path}: missing entry ending with {suffix!r}")

        cfg_bytes = _read("model_config.yaml")
        cfg = yaml.safe_load(cfg_bytes)

        sd_bytes = _read("model_weights.ckpt")
        sd = torch.load(io.BytesIO(sd_bytes), map_location="cpu", weights_only=False)

        # SPM model file. Skip vocab.txt / tokenizer.vocab; only the
        # _tokenizer.model entry carries the SentencePiece proto.
        sp_proto = None
        for n in names:
            if n.endswith("_tokenizer.model") or n.endswith("/tokenizer.model"):
                sp_proto = tf.extractfile(n).read()
                break
        if sp_proto is None:
            raise FileNotFoundError(f"{nemo_path}: no SPM tokenizer.model entry")

    return cfg, sd, sp_proto


def _find_cached_nemo(model_spec: str) -> Path | None:
    """Locate a cached .nemo for an HF repo id, if any."""
    if "/" not in model_spec or Path(model_spec).expanduser().exists():
        return None
    org_name = model_spec.replace("/", "--")
    cache_root = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / f"models--{org_name}" / "snapshots"
    )
    if not cache_root.exists():
        return None
    for snap in cache_root.iterdir():
        for cand in snap.glob("*.nemo"):
            return cand
    return None


def load_nemo_model(model_spec: str, prefer_direct: bool = False):
    """Load a Parakeet model via NeMo, falling back to a direct
    .nemo-archive read when NeMo's installed module catalog rejects
    streaming-specific encoder kwargs (unified-en-0.6b) or NeMo's
    extraction blows the disk budget on a large hybrid (tdt_ctc-1.1b).

    Accepts either an HF model name or a local .nemo file / dir.

    `prefer_direct=True` short-circuits NeMo's class instantiation
    when a cached .nemo is already present. NeMo's restore_from
    extracts the full archive to a temp dir before reading, which
    transiently doubles disk usage; the direct path streams entries
    out of the archive at near-zero transient cost.
    """
    from nemo.collections.asr.models import ASRModel

    local = Path(model_spec).expanduser()

    if prefer_direct:
        nemo_direct = local if (local.exists() and local.suffix == ".nemo") else _find_cached_nemo(model_spec)
        if nemo_direct is not None and nemo_direct.exists():
            print(f"Loading Parakeet directly from .nemo: {nemo_direct}")
            cfg, sd, sp_proto = _load_nemo_archive_directly(nemo_direct)
            return _DirectNemoArchive(cfg, sd, sp_proto)
        # else fall through to NeMo path

    nemo_path: Path | None = None
    if local.exists():
        print(f"Loading Parakeet from local path: {local}")
        try:
            model = ASRModel.restore_from(str(local), map_location="cpu")
            model.eval()
            return model
        except Exception as e:
            print(f"  NeMo restore_from failed: {e}\n  falling back to direct archive read",
                  file=sys.stderr)
            nemo_path = local if local.suffix == ".nemo" else None
    else:
        print(f"Loading Parakeet from HuggingFace: {model_spec}")
        try:
            model = ASRModel.from_pretrained(model_spec, map_location="cpu")
            model.eval()
            return model
        except Exception as e:
            print(f"  NeMo from_pretrained failed: {e}\n  falling back to direct archive read",
                  file=sys.stderr)
            nemo_path = _find_cached_nemo(model_spec)

    if nemo_path is None or not nemo_path.exists():
        raise RuntimeError(
            f"could not load {model_spec}: NeMo class instantiation failed "
            f"and no cached .nemo archive was found"
        )

    print(f"  loading directly from .nemo: {nemo_path}")
    cfg, sd, sp_proto = _load_nemo_archive_directly(nemo_path)
    return _DirectNemoArchive(cfg, sd, sp_proto)


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------


def extract_tokenizer(sp, blank_piece: str = "<blank>"):
    """Walk a SentencePieceProcessor and return the GGUF tokenizer payload.

    The +1 row at the end is the <blank> / start-state token that lives
    outside the SentencePiece vocab but inside the predictor's embed
    table (shape [vocab+1, hidden]).
    """
    vocab_size = sp.vocab_size()

    tokens: list[str] = []
    scores: list[float] = []
    types:  list[int]   = []

    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)

        if sp.is_unknown(i):
            ttype = TOKEN_TYPE_UNKNOWN
        elif sp.is_control(i):
            ttype = TOKEN_TYPE_CONTROL
        elif sp.is_unused(i):
            ttype = TOKEN_TYPE_UNUSED
        elif sp.is_byte(i):
            ttype = TOKEN_TYPE_BYTE
        else:
            ttype = TOKEN_TYPE_NORMAL

        tokens.append(piece)
        scores.append(score)
        types.append(ttype)

    tokens.append(blank_piece)
    scores.append(0.0)
    types.append(TOKEN_TYPE_CONTROL)

    blank_id = vocab_size

    return {
        "tokens":   tokens,
        "scores":   scores,
        "types":    types,
        "unk_id":   safe_id(sp.unk_id),
        "bos_id":   safe_id(sp.bos_id),
        "eos_id":   safe_id(sp.eos_id),
        "blank_id": blank_id,
    }


# ---------------------------------------------------------------------------
# Hparams from model.cfg
# ---------------------------------------------------------------------------


def _validate_durations(durations) -> list[int]:
    """Sanity-check the TDT durations array. The C++ loader enforces
    the same invariants but we'd rather catch a misconfigured source
    here with a precise diagnostic than emit a GGUF that the loader
    rejects with a more generic error."""
    if not isinstance(durations, list) or not durations:
        raise ValueError(
            f"decoding.durations must be a non-empty list, got {durations!r}"
        )
    out: list[int] = []
    for v in durations:
        if not isinstance(v, int) or v < 0:
            raise ValueError(
                f"decoding.durations entry {v!r} must be a non-negative int"
            )
        out.append(int(v))
    return out


def read_hparams(config: dict) -> dict:
    """Pull every hparam the loader's read_parakeet_hparams() requires
    out of NeMo's cfg dict (model.cfg serialized via OmegaConf).
    Cross-field invariants (d_model % n_heads == 0, win_length <= n_fft,
    etc.) are validated by the loader, not here — the converter is
    intentionally a thin pass-through.

    Predictor, joint, and TDT-durations resolution lives in convert()
    rather than here: the YAML cfg is sparse for the RNNT variants
    (`prednet={}`, `jointnet={}`, no `decoding.durations`) and the
    truth is on the live module / state_dict. read_hparams stays
    cfg-only and structurally invariant.

    Frontend defaults: NeMo's preprocessor carries most fields
    directly, but a few that PLAN.md declares mandatory in
    stt.frontend.* are not in NeMo's config because NeMo defaults them
    inside its preprocessor module. We hard-code the documented
    defaults for the AudioToMelSpectrogramPreprocessor used by every
    Parakeet variant we know about:

      - pre_emphasis: NeMo's FilterbankFeatures defaults `preemph=0.97`
        (features.py:250). The C++ frontend applies the standard one-tap
        filter `y[n] = x[n] - 0.97*x[n-1]`.
      - f_min: lowfreq defaults to 0 Hz. NeMo's config doesn't carry it.
      - f_max: highfreq defaults to sample_rate / 2 (Nyquist). NeMo's
        config doesn't carry it.
      - type: no config field for this; preprocessor target is
        AudioToMelSpectrogramPreprocessor so we emit "mel".
    """
    enc  = config["encoder"]
    pre  = config["preprocessor"]
    dec  = config["decoder"]

    # CTC's cfg.decoder uses `num_classes` instead of `vocab_size`
    # (ConvASRDecoder vs RNNTDecoder shapes). Both name the same
    # quantity: the SPM vocab size, excluding the blank that the head
    # output dim adds. Fall back rather than branch the read.
    raw_vocab = dec.get("vocab_size")
    if raw_vocab is None:
        raw_vocab = dec.get("num_classes")
    if raw_vocab is None:
        raise ValueError("config.decoder has neither vocab_size nor num_classes")

    sample_rate   = int(pre["sample_rate"])
    window_size   = float(pre["window_size"])
    window_stride = float(pre["window_stride"])
    win_length    = int(round(window_size  * sample_rate))
    hop_length    = int(round(window_stride * sample_rate))

    target = pre.get("_target_", "")
    if "MelSpectrogram" not in target:
        raise ValueError(
            f"unsupported preprocessor _target_: {target!r}; "
            f"converter only handles AudioToMelSpectrogramPreprocessor"
        )

    return {
        "enc_n_layers":             int(enc["n_layers"]),
        "enc_d_model":              int(enc["d_model"]),
        "enc_n_heads":              int(enc["n_heads"]),
        "enc_d_ff":                 int(enc["d_model"]) * int(enc["ff_expansion_factor"]),
        "enc_conv_kernel":          int(enc["conv_kernel_size"]),
        "enc_subsampling_factor":   int(enc["subsampling_factor"]),
        "enc_subsampling_channels": int(enc["subsampling_conv_channels"]),
        "enc_pos_emb_max_len":      int(enc["pos_emb_max_len"]),
        # use_bias is resolved from the state_dict in convert() — NeMo's
        # YAML omits the key on some checkpoints (tdt-1.1b) while the
        # constructor default is True, and on others (tdt-0.6b-v2/v3)
        # it is set explicitly to False. Trusting the YAML alone
        # silently drops 462 bias tensors on tdt-1.1b, so resolution
        # waits until the state_dict is in hand. Set to None here.
        "enc_use_bias":             None,

        # Predictor / joint / tdt-durations are resolved in convert();
        # see resolve_runtime_hparams() below. Set to None here so a
        # caller that forgets the resolve step fails loudly.
        "pred_hidden":              None,
        "pred_n_layers":            None,
        "pred_vocab":               int(raw_vocab) + 1,

        "joint_hidden":             None,
        "joint_num_extra_outputs":  None,
        "joint_activation":         None,

        "tdt_durations":            None,
        "tdt_max_symbols":          None,

        "fe_type":         "mel",
        "fe_num_mels":     int(pre["features"]),
        "fe_sample_rate":  sample_rate,
        "fe_n_fft":        int(pre["n_fft"]),
        "fe_win_length":   win_length,
        "fe_hop_length":   hop_length,
        "fe_window":       str(pre["window"]),
        "fe_normalize":    str(pre["normalize"]),
        "fe_dither":       float(pre["dither"]),
        "fe_pre_emphasis": 0.97,
        "fe_f_min":        0.0,
        "fe_f_max":        float(sample_rate) / 2.0,
    }


def resolve_runtime_hparams(hp: dict, model, config: dict, head_kind: str) -> None:
    """Fill in predictor / joint / TDT fields on `hp` using the live
    model and state_dict. The RNNT YAMLs (rnnt-0.6b/1.1b, unified-en)
    carry empty `prednet={}` / `jointnet={}` and no
    `decoding.durations`; the constructed module and state_dict are
    the source of truth.

    For CTC variants there is no predictor and no joint (the head is
    a single 1x1 conv); the predictor/joint/tdt fields are left as
    None and the corresponding KV writes are skipped in convert().

    Mutates `hp` in place. Asserts cross-field invariants where they
    apply (durations length matches num_extra_outputs for TDT models).
    """
    if head_kind == "ctc":
        return

    sd = model.state_dict()
    dec = config["decoder"]
    pred_cfg = dec.get("prednet") or {}
    joint_cfg = config.get("joint") or {}
    jointnet_cfg = joint_cfg.get("jointnet") or {}

    # Predictor: pred_hidden = embed second dim; n_layers = number of
    # weight_ih_l<i> tensors. Fall through to cfg if it carries values.
    pred_hidden_cfg = pred_cfg.get("pred_hidden")
    if pred_hidden_cfg is not None:
        hp["pred_hidden"] = int(pred_hidden_cfg)
    else:
        embed = sd.get("decoder.prediction.embed.weight")
        if embed is None:
            raise KeyError("state_dict missing decoder.prediction.embed.weight")
        hp["pred_hidden"] = int(embed.shape[1])

    pred_n_layers_cfg = pred_cfg.get("pred_rnn_layers")
    if pred_n_layers_cfg is not None:
        hp["pred_n_layers"] = int(pred_n_layers_cfg)
    else:
        n = 0
        while f"decoder.prediction.dec_rnn.lstm.weight_ih_l{n}" in sd:
            n += 1
        if n == 0:
            raise KeyError("state_dict has no LSTM weight_ih_l<i> tensors")
        hp["pred_n_layers"] = n

    # Joint: prefer the live module's resolved attrs; fall back to cfg.
    joint_module = getattr(model, "joint", None)

    def _from_module_or_cfg(attr: str, cfg_key: str, default=None):
        if joint_module is not None and hasattr(joint_module, attr):
            v = getattr(joint_module, attr)
            if v is not None and not callable(v):
                return v
        if cfg_key in jointnet_cfg and jointnet_cfg[cfg_key] is not None:
            return jointnet_cfg[cfg_key]
        if cfg_key in joint_cfg and joint_cfg[cfg_key] is not None:
            return joint_cfg[cfg_key]
        return default

    hp["joint_hidden"] = int(_from_module_or_cfg("joint_hidden", "joint_hidden"))
    hp["joint_activation"] = str(_from_module_or_cfg("activation", "activation")).lower()

    num_extra = _from_module_or_cfg("num_extra_outputs", "num_extra_outputs", 0)
    hp["joint_num_extra_outputs"] = int(num_extra)

    # TDT durations (optional — pure RNNT has none).
    decoding = config.get("decoding") or {}
    raw_durations = decoding.get("durations")
    if raw_durations is None:
        hp["tdt_durations"] = None
        hp["tdt_max_symbols"] = None
        if hp["joint_num_extra_outputs"] != 0:
            raise ValueError(
                f"joint.num_extra_outputs={hp['joint_num_extra_outputs']} "
                f"but decoding.durations missing — checkpoint looks like "
                f"TDT without a duration list"
            )
    else:
        durations = _validate_durations(list(raw_durations))
        if hp["joint_num_extra_outputs"] != len(durations):
            raise ValueError(
                f"joint.num_extra_outputs ({hp['joint_num_extra_outputs']}) "
                f"!= len(decoding.durations) ({len(durations)})"
            )
        hp["tdt_durations"] = durations
        greedy_cfg = decoding.get("greedy") or {}
        hp["tdt_max_symbols"] = int(greedy_cfg.get("max_symbols") or 10)


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------
#
# NeMo ships tensors in the layout the loader expects, so every entry
# is a plain passthrough. The ordering here is for human readability
# and matches src/arch/parakeet/weights.cpp; gguf-py writes tensors in
# add_tensor() call order, but the loader looks up by name.


ENCODER_BLOCK_TABLE: list[tuple[str, str]] = [
    # Macaron FF1.
    ("norm_feed_forward1.weight",       "norm_ff1.weight"),
    ("norm_feed_forward1.bias",         "norm_ff1.bias"),
    ("feed_forward1.linear1.weight",    "ff1.linear1.weight"),
    ("feed_forward1.linear2.weight",    "ff1.linear2.weight"),

    # Self-attention with relative position.
    ("norm_self_att.weight",            "norm_attn.weight"),
    ("norm_self_att.bias",              "norm_attn.bias"),
    ("self_attn.linear_q.weight",       "attn.linear_q.weight"),
    ("self_attn.linear_k.weight",       "attn.linear_k.weight"),
    ("self_attn.linear_v.weight",       "attn.linear_v.weight"),
    ("self_attn.linear_out.weight",     "attn.linear_out.weight"),
    ("self_attn.linear_pos.weight",     "attn.linear_pos.weight"),
    ("self_attn.pos_bias_u",            "attn.pos_bias_u"),
    ("self_attn.pos_bias_v",            "attn.pos_bias_v"),

    # Convolution module.
    ("norm_conv.weight",                "norm_conv.weight"),
    ("norm_conv.bias",                  "norm_conv.bias"),
    ("conv.pointwise_conv1.weight",     "conv.pointwise1.weight"),
    ("conv.depthwise_conv.weight",      "conv.depthwise.weight"),
    ("conv.pointwise_conv2.weight",     "conv.pointwise2.weight"),
    ("conv.batch_norm.weight",          "conv.bn.weight"),
    ("conv.batch_norm.bias",            "conv.bn.bias"),
    ("conv.batch_norm.running_mean",    "conv.bn.running_mean"),
    ("conv.batch_norm.running_var",     "conv.bn.running_var"),

    # Macaron FF2.
    ("norm_feed_forward2.weight",       "norm_ff2.weight"),
    ("norm_feed_forward2.bias",         "norm_ff2.bias"),
    ("feed_forward2.linear1.weight",    "ff2.linear1.weight"),
    ("feed_forward2.linear2.weight",    "ff2.linear2.weight"),

    # Final per-block layer norm.
    ("norm_out.weight",                 "norm_out.weight"),
    ("norm_out.bias",                   "norm_out.bias"),
]


# Conformer biases that are emitted only when enc_use_bias=true.
# v2/v3 have use_bias=false so this table is skipped; tdt-1.1b carries
# all of these. Layer-norm gammas/biases and the BN affine params live
# in ENCODER_BLOCK_TABLE because they are present regardless of the
# linear/conv use_bias flag.
ENCODER_BLOCK_BIAS_TABLE: list[tuple[str, str]] = [
    ("feed_forward1.linear1.bias",   "ff1.linear1.bias"),
    ("feed_forward1.linear2.bias",   "ff1.linear2.bias"),
    ("self_attn.linear_q.bias",      "attn.linear_q.bias"),
    ("self_attn.linear_k.bias",      "attn.linear_k.bias"),
    ("self_attn.linear_v.bias",      "attn.linear_v.bias"),
    ("self_attn.linear_out.bias",    "attn.linear_out.bias"),
    ("conv.pointwise_conv1.bias",    "conv.pointwise1.bias"),
    ("conv.depthwise_conv.bias",     "conv.depthwise.bias"),
    ("conv.pointwise_conv2.bias",    "conv.pointwise2.bias"),
    ("feed_forward2.linear1.bias",   "ff2.linear1.bias"),
    ("feed_forward2.linear2.bias",   "ff2.linear2.bias"),
]


PRE_ENCODE_TABLE: list[tuple[str, str]] = [
    ("encoder.pre_encode.conv.0.weight", "enc.pre_encode.conv.0.weight"),
    ("encoder.pre_encode.conv.0.bias",   "enc.pre_encode.conv.0.bias"),
    ("encoder.pre_encode.conv.2.weight", "enc.pre_encode.conv.2.weight"),
    ("encoder.pre_encode.conv.2.bias",   "enc.pre_encode.conv.2.bias"),
    ("encoder.pre_encode.conv.3.weight", "enc.pre_encode.conv.3.weight"),
    ("encoder.pre_encode.conv.3.bias",   "enc.pre_encode.conv.3.bias"),
    ("encoder.pre_encode.conv.5.weight", "enc.pre_encode.conv.5.weight"),
    ("encoder.pre_encode.conv.5.bias",   "enc.pre_encode.conv.5.bias"),
    ("encoder.pre_encode.conv.6.weight", "enc.pre_encode.conv.6.weight"),
    ("encoder.pre_encode.conv.6.bias",   "enc.pre_encode.conv.6.bias"),
    ("encoder.pre_encode.out.weight",    "enc.pre_encode.out.weight"),
    ("encoder.pre_encode.out.bias",      "enc.pre_encode.out.bias"),
]


JOINT_TABLE: list[tuple[str, str]] = [
    ("joint.enc.weight",          "joint.enc.weight"),
    ("joint.enc.bias",            "joint.enc.bias"),
    ("joint.pred.weight",         "joint.pred.weight"),
    ("joint.pred.bias",           "joint.pred.bias"),
    ("joint.joint_net.2.weight",  "joint.out.weight"),
    ("joint.joint_net.2.bias",    "joint.out.bias"),
]


# CTC head: a single 1x1 conv (Conv1d, kernel_size=1) projecting
# encoder d_model to vocab+1. NeMo names the module
# `decoder.decoder_layers.0`; we flatten to `head.ctc.*` to match
# the loader's eventual head dispatch (`stt.parakeet.head_kind=ctc`).
CTC_HEAD_TABLE: list[tuple[str, str]] = [
    ("decoder.decoder_layers.0.weight", "head.ctc.weight"),
    ("decoder.decoder_layers.0.bias",   "head.ctc.bias"),
]


# NeMo state_dict contains buffers and preprocessor tables that the
# loader computes itself at runtime. Skipping them silently keeps the
# unused-key check meaningful for genuine misses.
EXPECTED_UNUSED_PREFIXES_BASE = (
    "preprocessor.",  # mel filterbank + Hann window — C++ recomputes
)
EXPECTED_UNUSED_SUFFIXES = (
    ".num_batches_tracked",  # BN bookkeeping counter (scalar int64)
)

# The hybrid tdt_ctc-* checkpoints carry an auxiliary CTC head
# (`ctc_decoder.*`) alongside the full TDT path. Per the family-level
# decision (Open-decisions #1), we ship the hybrids as TDT-only and
# drop the CTC head — the pure ctc-* variants cover the CTC path.
EXPECTED_UNUSED_PREFIXES_TDT_CTC_DROPPED = ("ctc_decoder.",)


def is_expected_unused(key: str, head_kind: str, drop_aux_ctc: bool) -> bool:
    if key.startswith(EXPECTED_UNUSED_PREFIXES_BASE):
        return True
    if key.endswith(EXPECTED_UNUSED_SUFFIXES):
        return True
    if drop_aux_ctc and key.startswith(EXPECTED_UNUSED_PREFIXES_TDT_CTC_DROPPED):
        return True
    return False


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def tensor_to_fp32_numpy(t) -> np.ndarray:
    """Torch Tensor -> contiguous fp32 numpy. Source is expected to be
    fp32 already; any drop in precision should surface here rather than
    silently collapse into the target dtype."""
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t).__name__}")
    if t.dtype != torch.float32:
        raise ValueError(f"expected fp32 tensor, got {t.dtype}")
    arr = t.detach().cpu().numpy()
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert(model_spec: str, out_path: Path) -> None:
    from omegaconf import OmegaConf

    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    # The output slug is the canonical variant key. main() always sets
    # out_path = models/<slug>/<slug>-<REFDTYPE>.gguf, so the slug is
    # the parent directory name. Vocab_size alone is not unique across
    # the parakeet TDT family (0.6b-v2 and 1.1b both ship 1024 SPM
    # tokens), hence the slug-based dispatch.
    slug = out_path.parent.name
    if slug not in VARIANT_PROFILES:
        raise ValueError(
            f"unknown parakeet variant slug: {slug!r}; "
            f"known variants: {sorted(VARIANT_PROFILES)}"
        )
    profile = VARIANT_PROFILES[slug]
    head_kind = profile["head_kind"]
    prefer_direct = bool(profile.get("prefer_direct_load", False))
    drop_aux_ctc = "tdt_ctc" in slug  # hybrid checkpoints carry an aux CTC head we drop
    print(f"Variant: {profile['variant']} (head_kind={head_kind})")

    model = load_nemo_model(model_spec, prefer_direct=prefer_direct)

    config = OmegaConf.to_container(model.cfg, resolve=True)
    hp = read_hparams(config)
    resolve_runtime_hparams(hp, model, config, head_kind)

    raw_vocab_size = hp["pred_vocab"] - 1
    print(f"Detected raw vocab_size = {raw_vocab_size}")

    if raw_vocab_size != profile["expected_vocab_size"]:
        raise ValueError(
            f"vocab_size mismatch for {slug}: model carries "
            f"{raw_vocab_size}, profile expects "
            f"{profile['expected_vocab_size']}. The slug picked the "
            f"wrong profile, or the model is not what its name claims."
        )

    # Re-load the SPM model from its serialized proto so we get a plain
    # sentencepiece.SentencePieceProcessor with the canonical method
    # surface (vocab_size(), get_score(), is_*(), id_to_piece()). The
    # SPM instance NeMo hands out through model.tokenizer.tokenizer has
    # vocab_size monkey-patched to a property-like int on some builds,
    # which breaks the extract_tokenizer() contract.
    import sentencepiece as spm
    proto = model.tokenizer.tokenizer.serialized_model_proto()
    sp = spm.SentencePieceProcessor()
    sp.LoadFromSerializedProto(proto)
    tok = extract_tokenizer(sp)
    if len(tok["tokens"]) != hp["pred_vocab"]:
        raise ValueError(
            f"tokenizer length mismatch: {len(tok['tokens'])} tokens "
            f"(incl. <blank>) vs hp.pred_vocab={hp['pred_vocab']}"
        )

    sd = model.state_dict()
    sd_keys = set(sd.keys())

    # Resolve enc_use_bias from the state_dict. The presence of
    # encoder.layers.0.feed_forward1.linear1.bias is authoritative; the
    # YAML default is unreliable across checkpoints. We also assert
    # the answer is consistent (every layer in the same state) so a
    # mid-stack discrepancy fails fast rather than producing a
    # half-biased GGUF.
    bias_present = [
        f"encoder.layers.{i}.feed_forward1.linear1.bias" in sd_keys
        for i in range(hp["enc_n_layers"])
    ]
    if any(bias_present) and not all(bias_present):
        raise ValueError(
            f"inconsistent encoder use_bias across layers: "
            f"{sum(bias_present)}/{hp['enc_n_layers']} layers carry biases"
        )
    hp["enc_use_bias"] = bool(bias_present[0])
    print(f"Encoder use_bias: {hp['enc_use_bias']}")

    print(f"Writing GGUF to {out_path}")
    writer = GGUFWriter(str(out_path), "parakeet")

    # ----- general.* metadata -----
    writer.add_string("general.basename",   profile.get("basename", "parakeet-tdt"))
    writer.add_string("general.size_label", profile["size_label"])
    writer.add_string("general.version",    profile["version"])
    writer.add_uint32("general.file_type",  int(REFERENCE_FILE_TYPE))
    writer.add_array ("general.languages",  profile["languages"])

    # ----- stt.variant + capability KV -----
    writer.add_string("stt.variant", profile["variant"])
    if profile["lang_detect"]:
        writer.add_bool("stt.capability.lang_detect", True)

    # Head discriminator. The C++ loader currently always reads TDT
    # KV (durations etc.); Stage 4 will gate predictor/joint/tdt
    # reads on this. Emit unconditionally so the GGUF carries the
    # truth even before the loader uses it. New KV; loader treats
    # absent as "tdt" for backwards compat with already-shipped
    # v2/v3 GGUFs.
    writer.add_string("stt.parakeet.head_kind", head_kind)

    # ----- tokenizer.ggml.* -----
    writer.add_string("tokenizer.ggml.model", "bpe")
    writer.add_array("tokenizer.ggml.tokens",     tok["tokens"])
    writer.add_array("tokenizer.ggml.scores",     tok["scores"])
    writer.add_array("tokenizer.ggml.token_type", tok["types"])
    if tok["unk_id"] is not None:
        writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
    if tok["bos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
    if tok["eos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
    writer.add_uint32("tokenizer.ggml.blank_token_id", tok["blank_id"])

    # ----- stt.parakeet.* hparams -----
    writer.add_uint32("stt.parakeet.encoder.n_layers",             hp["enc_n_layers"])
    writer.add_uint32("stt.parakeet.encoder.d_model",              hp["enc_d_model"])
    writer.add_uint32("stt.parakeet.encoder.n_heads",              hp["enc_n_heads"])
    writer.add_uint32("stt.parakeet.encoder.d_ff",                 hp["enc_d_ff"])
    writer.add_uint32("stt.parakeet.encoder.conv_kernel",          hp["enc_conv_kernel"])
    writer.add_uint32("stt.parakeet.encoder.subsampling_factor",   hp["enc_subsampling_factor"])
    writer.add_uint32("stt.parakeet.encoder.subsampling_channels", hp["enc_subsampling_channels"])
    writer.add_uint32("stt.parakeet.encoder.pos_emb_max_len",      hp["enc_pos_emb_max_len"])
    writer.add_bool  ("stt.parakeet.encoder.use_bias",             hp["enc_use_bias"])

    if head_kind != "ctc":
        writer.add_uint32("stt.parakeet.predictor.hidden",   hp["pred_hidden"])
        writer.add_uint32("stt.parakeet.predictor.n_layers", hp["pred_n_layers"])
        writer.add_uint32("stt.parakeet.predictor.vocab",    hp["pred_vocab"])

        writer.add_uint32("stt.parakeet.joint.hidden",            hp["joint_hidden"])
        writer.add_uint32("stt.parakeet.joint.num_extra_outputs", hp["joint_num_extra_outputs"])
        writer.add_string("stt.parakeet.joint.activation",        hp["joint_activation"])

        if hp["tdt_durations"] is not None:
            writer.add_array(
                "stt.parakeet.tdt.durations",
                [int(d) for d in hp["tdt_durations"]],
            )
            writer.add_uint32("stt.parakeet.tdt.max_symbols", hp["tdt_max_symbols"])

    writer.add_string("stt.frontend.type",         hp["fe_type"])
    writer.add_uint32("stt.frontend.num_mels",     hp["fe_num_mels"])
    writer.add_uint32("stt.frontend.sample_rate",  hp["fe_sample_rate"])
    writer.add_uint32("stt.frontend.n_fft",        hp["fe_n_fft"])
    writer.add_uint32("stt.frontend.win_length",   hp["fe_win_length"])
    writer.add_uint32("stt.frontend.hop_length",   hp["fe_hop_length"])
    writer.add_string("stt.frontend.window",       hp["fe_window"])
    writer.add_string("stt.frontend.normalize",    hp["fe_normalize"])
    writer.add_float32("stt.frontend.dither",       hp["fe_dither"])
    writer.add_float32("stt.frontend.pre_emphasis", hp["fe_pre_emphasis"])
    writer.add_float32("stt.frontend.f_min",        hp["fe_f_min"])
    writer.add_float32("stt.frontend.f_max",        hp["fe_f_max"])

    # ----- tensors -----
    consumed: set[str] = set()
    n_added = 0
    bytes_out = 0

    def add(nemo_name: str, gguf_name: str) -> None:
        nonlocal n_added, bytes_out
        if nemo_name not in sd_keys:
            raise KeyError(f"state_dict missing tensor: {nemo_name!r}")
        arr = tensor_to_fp32_numpy(sd[nemo_name])
        writer.add_tensor(gguf_name, arr)
        consumed.add(nemo_name)
        bytes_out += int(arr.nbytes)
        n_added += 1

    def add_combined(nemo_a: str, nemo_b: str, gguf_name: str) -> None:
        """Sum two source tensors and emit under gguf_name. Used to
        collapse PyTorch LSTM's bias_ih + bias_hh into the single
        pred.lstm.{i}.bias the loader expects."""
        nonlocal n_added, bytes_out
        for k in (nemo_a, nemo_b):
            if k not in sd_keys:
                raise KeyError(f"state_dict missing tensor: {k!r}")
        a = tensor_to_fp32_numpy(sd[nemo_a])
        b = tensor_to_fp32_numpy(sd[nemo_b])
        if a.shape != b.shape:
            raise ValueError(
                f"shape mismatch for {nemo_a} ({a.shape}) vs "
                f"{nemo_b} ({b.shape})"
            )
        arr = np.ascontiguousarray(a + b)
        writer.add_tensor(gguf_name, arr)
        consumed.add(nemo_a)
        consumed.add(nemo_b)
        bytes_out += int(arr.nbytes)
        n_added += 1

    # pre_encode
    for nemo_name, gguf_name in PRE_ENCODE_TABLE:
        add(nemo_name, gguf_name)

    # encoder layers
    for i in range(hp["enc_n_layers"]):
        for suffix_nemo, suffix_gguf in ENCODER_BLOCK_TABLE:
            add(
                f"encoder.layers.{i}.{suffix_nemo}",
                f"enc.blocks.{i}.{suffix_gguf}",
            )
        if hp["enc_use_bias"]:
            for suffix_nemo, suffix_gguf in ENCODER_BLOCK_BIAS_TABLE:
                add(
                    f"encoder.layers.{i}.{suffix_nemo}",
                    f"enc.blocks.{i}.{suffix_gguf}",
                )

    if head_kind == "ctc":
        # CTC: encoder feeds a single 1x1 conv directly. No predictor,
        # no joint.
        for nemo_name, gguf_name in CTC_HEAD_TABLE:
            add(nemo_name, gguf_name)
    else:
        # predictor: embed + n_lstm_layers * (Wx, Wh, bias).
        add("decoder.prediction.embed.weight", "pred.embed.weight")
        for i in range(hp["pred_n_layers"]):
            add(
                f"decoder.prediction.dec_rnn.lstm.weight_ih_l{i}",
                f"pred.lstm.{i}.Wx",
            )
            add(
                f"decoder.prediction.dec_rnn.lstm.weight_hh_l{i}",
                f"pred.lstm.{i}.Wh",
            )
            add_combined(
                f"decoder.prediction.dec_rnn.lstm.bias_ih_l{i}",
                f"decoder.prediction.dec_rnn.lstm.bias_hh_l{i}",
                f"pred.lstm.{i}.bias",
            )

        # joint
        for nemo_name, gguf_name in JOINT_TABLE:
            add(nemo_name, gguf_name)

    per_layer_tensors = len(ENCODER_BLOCK_TABLE) + (
        len(ENCODER_BLOCK_BIAS_TABLE) if hp["enc_use_bias"] else 0
    )
    if head_kind == "ctc":
        head_tensors = len(CTC_HEAD_TABLE)
    else:
        head_tensors = (
            1                                       # pred.embed
            + hp["pred_n_layers"] * 3               # pred.lstm.{Wx,Wh,bias}
            + len(JOINT_TABLE)
        )
    expected = (
        len(PRE_ENCODE_TABLE)
        + hp["enc_n_layers"] * per_layer_tensors
        + head_tensors
    )
    if n_added != expected:
        raise RuntimeError(
            f"tensor count mismatch: added {n_added}, expected {expected}"
        )
    print(f"Added {n_added} tensors ({bytes_out / (1024 * 1024):.1f} MB)")

    unused = sorted(
        k for k in (sd_keys - consumed) if not is_expected_unused(k, head_kind, drop_aux_ctc)
    )
    if unused:
        print(
            f"WARNING: {len(unused)} state_dict keys were not consumed:",
            file=sys.stderr,
        )
        for k in unused[:10]:
            print(f"  {k}", file=sys.stderr)
        if len(unused) > 10:
            print(f"  ... and {len(unused) - 10} more", file=sys.stderr)

    print("Writing header + KV + tensor info...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    print("Writing tensor data (this takes a while for ~2.4 GB)...")
    writer.write_tensors_to_file()
    writer.close()

    print(f"Done. Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a NeMo Parakeet TDT model to a GGUF.",
    )
    p.add_argument(
        "model",
        type=str,
        help="HF repo id (e.g. nvidia/parakeet-tdt-0.6b-v2) or local .nemo path",
    )
    p.add_argument(
        "out_path",
        type=Path,
        nargs="?",
        help="Output .gguf path. If omitted, derived from --repo-id or "
             "the model argument when it looks like an HF repo id.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HF repo id used to derive the output slug when converting "
             "from a local path. Ignored if out_path is given.",
    )
    args = p.parse_args(argv[1:])

    out_path = args.out_path
    if out_path is None:
        repo_id = args.repo_id
        if repo_id is None and "/" in args.model and not Path(args.model).exists():
            repo_id = args.model
        if not repo_id:
            print(
                "error: provide out_path, --repo-id, or pass an HF repo id as model",
                file=sys.stderr,
            )
            return 2
        slug = slug_from_repo_id(repo_id)
        out_path = REPO_ROOT / "models" / slug / gguf_name(slug, REFERENCE_DTYPE_LABEL)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    convert(args.model, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
