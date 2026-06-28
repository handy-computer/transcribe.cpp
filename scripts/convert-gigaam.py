#!/usr/bin/env python3
"""
convert-gigaam.py - convert a salute-developers/GigaAM-v3 checkpoint into
a GGUF that transcribe.cpp's loader can ingest end-to-end. Loads via the
upstream `gigaam` package (gigaam.load_model). Source dtype is F32; use
tools/transcribe-quantize for deployment quantization.

Run through the per-family reference environment:

    uv run --project scripts/envs/gigaam \\
      scripts/convert-gigaam.py ai-sage/GigaAM-v3 \\
      --repo-id gigaam-v3-e2e-rnnt \\
      --variant-key v3_e2e_rnnt

The HF repo argument is informational only — the `gigaam` package
downloads from its own URL list keyed by --variant-key. The --repo-id
controls the output slug; it must be one of the five GigaAM-v3 family
slugs in `VARIANT_PROFILES`.

The four ported variants share a 16-layer Conformer encoder with
rotary attention. They differ in head + tokenizer:

  v3_e2e_rnnt -> RNN-T head, SentencePiece 1024 + punctuation/casing
  v3_e2e_ctc  -> CTC   head, SentencePiece 256  + punctuation/casing
  v3_rnnt     -> RNN-T head, charwise 33 entries (lowercased Russian)
  v3_ctc      -> CTC   head, charwise 33 entries (lowercased Russian)

The upstream `v3_ssl` (HuBERT-CTC pretraining checkpoint, encoder-only)
is out of scope for this runtime — it has no head, no tokenizer, no
decoding path, and transcribe.cpp has no encoder-output emission CLI.

Frontend gotchas baked into the GGUF: center=false, mel_scale=htk,
mel_norm=null. We ship the model's own mel filterbank tensor +
hann window so the C++ frontend uses bit-identical buffers; the
intake-declared mel_norm/window stay as informational KV.

KV emitted (matches the per-family read_gigaam_hparams the Stage 4
loader will define):

  general.architecture       = "gigaam"
  general.basename           = "gigaam-v3"
  general.size_label         = "242M"
  general.version            = "v3"
  general.languages          = ["ru"]
  stt.variant                = profile["variant"]
  stt.gigaam.head_kind       = "rnnt" | "ctc"

  tokenizer.ggml.model       = "bpe" (SP variants) | "char" (charwise variants)
  tokenizer.ggml.tokens / scores / token_type / *_token_id

  stt.gigaam.encoder.{n_layers, d_model, n_heads, d_ff, conv_kernel,
                       subsampling_factor, subs_kernel_size, pos_emb_max_len,
                       feat_in, self_attention_model, conv_norm_type}

  RNN-T variants additionally emit:
    stt.gigaam.predictor.{hidden, n_layers, vocab}
    stt.gigaam.joint.{hidden, activation}

  CTC variants additionally emit:
    stt.gigaam.head.num_classes

  stt.frontend.{type, num_mels, sample_rate, n_fft, win_length, hop_length,
                window, normalize, dither, pre_emphasis, f_min, f_max,
                center, mel_norm}

Layout conversions: NONE. PyTorch weights are already in OIHW / [O, I, k]
order for convs and (out, in) for linears.

LSTM bias collapse: PyTorch stores bias_ih_l{i} + bias_hh_l{i} as two
separate vectors that both get added to the gate pre-activation. We
collapse to a single `pred.lstm.{i}.bias` = bias_ih + bias_hh, matching
the loader's single-bias expectation. Gate order is PyTorch native
(i, f, g, o).

The "batch_norm" attribute in GigaAM's Conformer conv module is actually
a `torch.nn.LayerNorm` (no running_mean/running_var in state_dict).
Source naming preserved for fidelity to the upstream cfg
(conv_norm_type=layer_norm), but the runtime op is LayerNorm.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, LlamaFileType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    gguf_writer,
    TOKEN_TYPE_BYTE,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN,
    TOKEN_TYPE_UNUSED,
    add_general_identity,
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
# All five GigaAM-v3 variants share the same Conformer encoder shape
# (16 layers, d=768, 16 heads). They differ only in head + tokenizer.
# The slug (HF-style basename used as the output directory + GGUF base
# name) is the key the converter dispatches on; --variant-key picks the
# actual `gigaam.load_model(...)` argument.

VARIANT_PROFILES: dict[str, dict] = {
    "gigaam-v3-e2e-rnnt": {
        "variant": "gigaam-v3-e2e-rnnt",
        "variant_key": "v3_e2e_rnnt",
        "head_kind": "rnnt",
        "tokenizer": "sentencepiece",
        "expected_num_classes": 1025,  # vocab(1024) + 1 blank
    },
    "gigaam-v3-e2e-ctc": {
        "variant": "gigaam-v3-e2e-ctc",
        "variant_key": "v3_e2e_ctc",
        "head_kind": "ctc",
        "tokenizer": "sentencepiece",
        "expected_num_classes": 257,   # vocab(256) + 1 blank
    },
    "gigaam-v3-rnnt": {
        "variant": "gigaam-v3-rnnt",
        "variant_key": "v3_rnnt",
        "head_kind": "rnnt",
        "tokenizer": "charwise",
        "expected_num_classes": 34,    # 33 chars + 1 blank
    },
    "gigaam-v3-ctc": {
        "variant": "gigaam-v3-ctc",
        "variant_key": "v3_ctc",
        "head_kind": "ctc",
        "tokenizer": "charwise",
        "expected_num_classes": 34,
    },
}


GENERAL_BASENAME = "gigaam-v3"
GENERAL_VERSION  = "v3"
GENERAL_LANGUAGES = ["ru"]


# Friendly general.name per variant slug (== profile["variant"]).
VARIANT_DISPLAY_NAMES: dict[str, str] = {
    "gigaam-v3-ctc":      "GigaAM v3 CTC",
    "gigaam-v3-e2e-ctc":  "GigaAM v3 E2E-CTC",
    "gigaam-v3-rnnt":     "GigaAM v3 RNN-T",
    "gigaam-v3-e2e-rnnt": "GigaAM v3 E2E-RNN-T",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_gigaam(variant_key: str):
    """Load a GigaAM-v3 variant via the upstream `gigaam` package.

    Forces fp16_encoder=False (F32) and device='cpu' to match the
    Stage 2 reference dumps.
    """
    import gigaam

    print(f"Loading gigaam.load_model({variant_key!r}, fp16_encoder=False, device='cpu')")
    model = gigaam.load_model(variant_key, fp16_encoder=False, device="cpu")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------


def extract_sp_tokenizer(sp, blank_piece: str = "<blank>") -> dict:
    """Walk a SentencePieceProcessor and return the GGUF tokenizer payload.

    The trailing <blank> entry mirrors the predictor embed table layout
    used by the RNN-T head (shape [num_classes, pred_hidden]) and the
    CTC head's [num_classes, d_model, 1] kernel — both already carry
    the blank at index = vocab_size.
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

    return {
        "kind":     "sentencepiece",
        "tokens":   tokens,
        "scores":   scores,
        "types":    types,
        "unk_id":   safe_id(sp.unk_id),
        "bos_id":   safe_id(sp.bos_id),
        "eos_id":   safe_id(sp.eos_id),
        "blank_id": vocab_size,
    }


def extract_charwise_tokenizer(vocab: list[str], blank_piece: str = "<blank>") -> dict:
    """Emit a charwise GigaAM vocab (33 entries: space + 32 Cyrillic
    letters) as a tokenizer.ggml payload. Scores are all zero (charwise
    has no learned scores). Blank goes last and is the only CONTROL
    token; the 33 chars are all NORMAL.
    """
    tokens: list[str] = list(vocab)
    scores: list[float] = [0.0] * len(vocab)
    types:  list[int]   = [TOKEN_TYPE_NORMAL] * len(vocab)

    tokens.append(blank_piece)
    scores.append(0.0)
    types.append(TOKEN_TYPE_CONTROL)

    return {
        "kind":     "charwise",
        "tokens":   tokens,
        "scores":   scores,
        "types":    types,
        "unk_id":   None,
        "bos_id":   None,
        "eos_id":   None,
        "blank_id": len(vocab),
    }


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


def read_hparams(config: dict, model) -> dict:
    """Pull hparams from cfg + live module. Predictor / joint / head
    fields are head-kind-dependent; the caller selects which subset to
    emit based on the profile's head_kind.
    """
    enc = config["encoder"]
    pre = config["preprocessor"]

    sample_rate = int(pre["sample_rate"])
    n_fft       = int(pre["n_fft"])
    hop_length  = int(pre["hop_length"])
    win_length  = int(pre["win_length"])
    n_mels      = int(pre["features"])
    mel_scale   = str(pre["mel_scale"])
    mel_norm_raw = pre.get("mel_norm")
    center      = bool(pre.get("center", False))

    # SpecScaler clamp constants used by GigaAM's preprocessor:
    # log(clamp(x, 1e-9, 1e9)). The upper clamp is so large it never
    # binds in practice but we record it for completeness.
    log_clamp_min = 1e-9
    log_clamp_max = 1e9

    hp = {
        "enc_n_layers":             int(enc["n_layers"]),
        "enc_d_model":              int(enc["d_model"]),
        "enc_n_heads":               int(enc["n_heads"]),
        "enc_d_ff":                 int(enc["d_model"]) * int(enc["ff_expansion_factor"]),
        "enc_conv_kernel":          int(enc["conv_kernel_size"]),
        "enc_subsampling_factor":   int(enc["subsampling_factor"]),
        "enc_subs_kernel_size":     int(enc["subs_kernel_size"]),
        "enc_pos_emb_max_len":      int(enc["pos_emb_max_len"]),
        "enc_feat_in":              int(enc["feat_in"]),
        "enc_self_attention_model": str(enc["self_attention_model"]),
        "enc_conv_norm_type":       str(enc["conv_norm_type"]),
    }

    # Head-side hparams resolved from the live module / cfg.
    head_cfg = config.get("head", {}) or {}
    if head_cfg.get("_target_", "").endswith("RNNTHead"):
        dec_cfg = head_cfg["decoder"]
        joint_cfg = head_cfg["joint"]
        hp.update({
            "pred_hidden":     int(dec_cfg["pred_hidden"]),
            "pred_n_layers":   int(dec_cfg["pred_rnn_layers"]),
            "pred_vocab":      int(dec_cfg["num_classes"]),
            "joint_hidden":    int(joint_cfg["joint_hidden"]),
            "joint_enc_in":    int(joint_cfg["enc_hidden"]),
            "joint_pred_in":   int(joint_cfg["pred_hidden"]),
            "joint_num_classes": int(joint_cfg["num_classes"]),
            "joint_activation":  _infer_joint_activation(model),
        })
    elif head_cfg.get("_target_", "").endswith("CTCHead"):
        hp.update({
            "head_feat_in":     int(head_cfg["feat_in"]),
            "head_num_classes": int(head_cfg["num_classes"]),
        })
    # SSL: head_cfg empty; nothing to add.

    hp.update({
        "fe_type":         "mel",
        "fe_num_mels":     n_mels,
        "fe_sample_rate":  sample_rate,
        "fe_n_fft":        n_fft,
        "fe_win_length":   win_length,
        "fe_hop_length":   hop_length,
        "fe_window":       "hann_periodic",  # torch.hann_window(N, periodic=True)
        "fe_normalize":    "none",            # no per-utterance/feature normalize
        "fe_dither":       0.0,
        "fe_pre_emphasis": 0.0,
        "fe_f_min":        0.0,
        "fe_f_max":        float(sample_rate) / 2.0,
        "fe_center":       center,
        # `mel_scale` encodes both the HTK frequency formula and the
        # absence of slaney area normalization (mel_norm=null). The C++
        # loader treats this string as the build-mode for the filterbank
        # when it is not already baked in. We also bake the actual
        # filterbank tensor below so this is informational on the
        # gigaam path.
        "fe_mel_norm":     mel_scale if mel_norm_raw is None else f"{mel_scale}+{mel_norm_raw}",
        "fe_log_clamp_min": log_clamp_min,
        "fe_log_clamp_max": log_clamp_max,
    })
    return hp


def _infer_joint_activation(model) -> str:
    """RNNTHead.joint.joint_net is a Sequential. GigaAM ships
    [ReLU, Linear] (length 2); parakeet's TDT ships [Linear, ReLU, Linear]
    (length 3). We pick the activation by class-name lookup so the
    converter doesn't hard-code one. Falls back to 'relu' since both
    upstream gigaam variants observed at intake time use ReLU.
    """
    try:
        jn = model.head.joint.joint_net
    except AttributeError:
        return "relu"
    for m in jn:
        name = type(m).__name__.lower()
        if name in ("relu", "gelu", "tanh", "silu", "swish"):
            return name
    return "relu"


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------


ENCODER_PRE_ENCODE_TABLE: list[tuple[str, str]] = [
    ("encoder.pre_encode.conv.0.weight", "enc.pre_encode.conv.0.weight"),
    ("encoder.pre_encode.conv.0.bias",   "enc.pre_encode.conv.0.bias"),
    ("encoder.pre_encode.conv.2.weight", "enc.pre_encode.conv.2.weight"),
    ("encoder.pre_encode.conv.2.bias",   "enc.pre_encode.conv.2.bias"),
]


# 34 tensors per Conformer block. The source attribute "batch_norm" is
# a LayerNorm at runtime (no running stats); we rename to ".ln." on the
# GGUF side so the loader's expectation is unambiguous.
ENCODER_BLOCK_TABLE: list[tuple[str, str]] = [
    # Macaron FF1.
    ("norm_feed_forward1.weight",       "norm_ff1.weight"),
    ("norm_feed_forward1.bias",         "norm_ff1.bias"),
    ("feed_forward1.linear1.weight",    "ff1.linear1.weight"),
    ("feed_forward1.linear1.bias",      "ff1.linear1.bias"),
    ("feed_forward1.linear2.weight",    "ff1.linear2.weight"),
    ("feed_forward1.linear2.bias",      "ff1.linear2.bias"),

    # Conv module (LayerNorm on the channel axis, despite source name).
    ("norm_conv.weight",                "norm_conv.weight"),
    ("norm_conv.bias",                  "norm_conv.bias"),
    ("conv.pointwise_conv1.weight",     "conv.pointwise1.weight"),
    ("conv.pointwise_conv1.bias",       "conv.pointwise1.bias"),
    ("conv.depthwise_conv.weight",      "conv.depthwise.weight"),
    ("conv.depthwise_conv.bias",        "conv.depthwise.bias"),
    ("conv.batch_norm.weight",          "conv.ln.weight"),
    ("conv.batch_norm.bias",            "conv.ln.bias"),
    ("conv.pointwise_conv2.weight",     "conv.pointwise2.weight"),
    ("conv.pointwise_conv2.bias",       "conv.pointwise2.bias"),

    # Self-attention with rotary PE (no linear_pos / pos_bias_*).
    ("norm_self_att.weight",            "norm_attn.weight"),
    ("norm_self_att.bias",              "norm_attn.bias"),
    ("self_attn.linear_q.weight",       "attn.linear_q.weight"),
    ("self_attn.linear_q.bias",         "attn.linear_q.bias"),
    ("self_attn.linear_k.weight",       "attn.linear_k.weight"),
    ("self_attn.linear_k.bias",         "attn.linear_k.bias"),
    ("self_attn.linear_v.weight",       "attn.linear_v.weight"),
    ("self_attn.linear_v.bias",         "attn.linear_v.bias"),
    ("self_attn.linear_out.weight",     "attn.linear_out.weight"),
    ("self_attn.linear_out.bias",       "attn.linear_out.bias"),

    # Macaron FF2.
    ("norm_feed_forward2.weight",       "norm_ff2.weight"),
    ("norm_feed_forward2.bias",         "norm_ff2.bias"),
    ("feed_forward2.linear1.weight",    "ff2.linear1.weight"),
    ("feed_forward2.linear1.bias",      "ff2.linear1.bias"),
    ("feed_forward2.linear2.weight",    "ff2.linear2.weight"),
    ("feed_forward2.linear2.bias",      "ff2.linear2.bias"),

    # Block-final layer norm.
    ("norm_out.weight",                 "norm_out.weight"),
    ("norm_out.bias",                   "norm_out.bias"),
]


RNNT_JOINT_TABLE: list[tuple[str, str]] = [
    ("head.joint.enc.weight",            "joint.enc.weight"),
    ("head.joint.enc.bias",              "joint.enc.bias"),
    ("head.joint.pred.weight",           "joint.pred.weight"),
    ("head.joint.pred.bias",             "joint.pred.bias"),
    # joint_net = Sequential(ReLU(0), Linear(1)). The Linear is the
    # output projection to num_classes; rename to joint.out.
    ("head.joint.joint_net.1.weight",    "joint.out.weight"),
    ("head.joint.joint_net.1.bias",      "joint.out.bias"),
]


CTC_HEAD_TABLE: list[tuple[str, str]] = [
    ("head.decoder_layers.0.weight", "head.ctc.weight"),
    ("head.decoder_layers.0.bias",   "head.ctc.bias"),
]


EXPECTED_UNUSED_PREFIXES = (
    "preprocessor.",  # filterbank + window — baked separately as frontend.*
)


def is_expected_unused(key: str) -> bool:
    return key.startswith(EXPECTED_UNUSED_PREFIXES)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def tensor_to_fp32_numpy(t) -> np.ndarray:
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


def convert(variant_key: str, slug: str, out_path: Path, repo_id: str | None = None) -> None:
    from omegaconf import OmegaConf

    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    if slug not in VARIANT_PROFILES:
        raise ValueError(
            f"unknown gigaam variant slug: {slug!r}; "
            f"known: {sorted(VARIANT_PROFILES)}"
        )
    profile = VARIANT_PROFILES[slug]
    if variant_key != profile["variant_key"]:
        raise ValueError(
            f"variant_key {variant_key!r} does not match slug {slug!r} "
            f"(profile expects {profile['variant_key']!r})"
        )
    head_kind = profile["head_kind"]
    print(f"Variant: {profile['variant']} (head_kind={head_kind}, "
          f"tokenizer={profile['tokenizer']})")

    model = load_gigaam(variant_key)
    config = OmegaConf.to_container(model.cfg, resolve=True)
    hp = read_hparams(config, model)

    # Cross-check the expected vocab/num_classes against the live model.
    if profile["expected_num_classes"] is not None:
        if head_kind == "rnnt":
            observed = hp["pred_vocab"]
        else:  # ctc
            observed = hp["head_num_classes"]
        if observed != profile["expected_num_classes"]:
            raise ValueError(
                f"num_classes mismatch for {slug}: model carries {observed}, "
                f"profile expects {profile['expected_num_classes']}"
            )

    # Tokenizer payload.
    tok = None
    if profile["tokenizer"] == "sentencepiece":
        import sentencepiece as spm
        sp_path = model.cfg.decoding.model_path
        sp = spm.SentencePieceProcessor(model_file=str(sp_path))
        tok = extract_sp_tokenizer(sp)
        if head_kind == "rnnt":
            expected_total = hp["pred_vocab"]
        else:
            expected_total = hp["head_num_classes"]
        if len(tok["tokens"]) != expected_total:
            raise ValueError(
                f"SP tokenizer length {len(tok['tokens'])} != "
                f"head num_classes {expected_total}"
            )
        if tok["blank_id"] != expected_total - 1:
            raise ValueError(
                f"SP blank_id {tok['blank_id']} != num_classes-1 "
                f"({expected_total - 1})"
            )
    elif profile["tokenizer"] == "charwise":
        vocab = list(config["decoding"]["vocabulary"])
        tok = extract_charwise_tokenizer(vocab)
        if head_kind == "rnnt":
            expected_total = hp["pred_vocab"]
        else:
            expected_total = hp["head_num_classes"]
        if len(tok["tokens"]) != expected_total:
            raise ValueError(
                f"charwise tokenizer length {len(tok['tokens'])} != "
                f"head num_classes {expected_total}"
            )
    else:
        raise ValueError(
            f"unexpected tokenizer kind {profile['tokenizer']!r} for {slug}"
        )

    sd = model.state_dict()
    sd_keys = set(sd.keys())

    # Frontend buffers (baked into the GGUF for bit-identical reproduction).
    # Source orientation in PyTorch's MelScale is [n_freq_bins, n_mels];
    # transpose to [n_mels, n_freq_bins] to match the convention
    # used by other transcribe.cpp converters (qwen3_asr, etc.).
    fb_src   = tensor_to_fp32_numpy(sd["preprocessor.featurizer.0.mel_scale.fb"])
    fb       = np.ascontiguousarray(fb_src.T)            # [n_mels, n_fft/2+1]
    window   = tensor_to_fp32_numpy(sd["preprocessor.featurizer.0.spectrogram.window"])

    expected_fb_shape = (hp["fe_num_mels"], hp["fe_n_fft"] // 2 + 1)
    if fb.shape != expected_fb_shape:
        raise ValueError(
            f"unexpected filterbank shape {fb.shape}, expected {expected_fb_shape}"
        )
    if window.shape != (hp["fe_win_length"],):
        raise ValueError(
            f"unexpected window shape {window.shape}, "
            f"expected ({hp['fe_win_length']},)"
        )

    # Size label from total params (excluding frontend buffers).
    total = sum(
        int(np.prod(t.shape)) for k, t in sd.items()
        if not k.startswith("preprocessor.")
    )
    if total >= 1_000_000_000:
        size_label = f"{total / 1_000_000_000:.1f}B"
    elif total >= 1_000_000:
        size_label = f"{total / 1_000_000:.0f}M"
    else:
        size_label = f"{total / 1_000:.0f}K"
    print(f"Total params (encoder+head): {total:,} -> size_label={size_label}")

    print(f"Writing GGUF to {out_path}")
    writer = gguf_writer(str(out_path), "gigaam")

    # ----- general.* -----
    if profile["variant"] not in VARIANT_DISPLAY_NAMES:
        raise ValueError(
            f"unknown gigaam variant: {profile['variant']!r}; "
            f"add it to VARIANT_DISPLAY_NAMES"
        )
    add_general_identity(
        writer,
        name=VARIANT_DISPLAY_NAMES[profile["variant"]],
        basename=GENERAL_BASENAME,
        size_label=size_label,
        version=GENERAL_VERSION,
        file_type=REFERENCE_FILE_TYPE,
        languages=GENERAL_LANGUAGES,
        author="Salute Developers",
        organization="ai-sage",
        license="mit",
        license_name="MIT License",
        license_link="https://opensource.org/license/mit",
        repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
    )

    # ----- stt.variant + head_kind -----
    writer.add_string("stt.variant", profile["variant"])
    writer.add_string("stt.gigaam.head_kind", head_kind)

    # ----- tokenizer.ggml.* -----
    if tok is not None:
        writer.add_string("tokenizer.ggml.model",
                          "bpe" if tok["kind"] == "sentencepiece" else "char")
        writer.add_array ("tokenizer.ggml.tokens",     tok["tokens"])
        writer.add_array ("tokenizer.ggml.scores",     tok["scores"])
        writer.add_array ("tokenizer.ggml.token_type", tok["types"])
        if tok["unk_id"] is not None:
            writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
        if tok["bos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
        if tok["eos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
        writer.add_uint32("tokenizer.ggml.blank_token_id", tok["blank_id"])

    # ----- stt.gigaam.encoder.* -----
    writer.add_uint32("stt.gigaam.encoder.n_layers",           hp["enc_n_layers"])
    writer.add_uint32("stt.gigaam.encoder.d_model",            hp["enc_d_model"])
    writer.add_uint32("stt.gigaam.encoder.n_heads",            hp["enc_n_heads"])
    writer.add_uint32("stt.gigaam.encoder.d_ff",               hp["enc_d_ff"])
    writer.add_uint32("stt.gigaam.encoder.conv_kernel",        hp["enc_conv_kernel"])
    writer.add_uint32("stt.gigaam.encoder.subsampling_factor", hp["enc_subsampling_factor"])
    writer.add_uint32("stt.gigaam.encoder.subs_kernel_size",   hp["enc_subs_kernel_size"])
    writer.add_uint32("stt.gigaam.encoder.pos_emb_max_len",    hp["enc_pos_emb_max_len"])
    writer.add_uint32("stt.gigaam.encoder.feat_in",            hp["enc_feat_in"])
    writer.add_string("stt.gigaam.encoder.self_attention_model", hp["enc_self_attention_model"])
    writer.add_string("stt.gigaam.encoder.conv_norm_type",     hp["enc_conv_norm_type"])

    if head_kind == "rnnt":
        writer.add_uint32("stt.gigaam.predictor.hidden",   hp["pred_hidden"])
        writer.add_uint32("stt.gigaam.predictor.n_layers", hp["pred_n_layers"])
        writer.add_uint32("stt.gigaam.predictor.vocab",    hp["pred_vocab"])
        writer.add_uint32("stt.gigaam.joint.hidden",       hp["joint_hidden"])
        writer.add_uint32("stt.gigaam.joint.num_classes",  hp["joint_num_classes"])
        writer.add_string("stt.gigaam.joint.activation",   hp["joint_activation"])
    elif head_kind == "ctc":
        writer.add_uint32("stt.gigaam.head.feat_in",     hp["head_feat_in"])
        writer.add_uint32("stt.gigaam.head.num_classes", hp["head_num_classes"])

    # ----- stt.frontend.* -----
    writer.add_string ("stt.frontend.type",         hp["fe_type"])
    writer.add_uint32 ("stt.frontend.num_mels",     hp["fe_num_mels"])
    writer.add_uint32 ("stt.frontend.sample_rate",  hp["fe_sample_rate"])
    writer.add_uint32 ("stt.frontend.n_fft",        hp["fe_n_fft"])
    writer.add_uint32 ("stt.frontend.win_length",   hp["fe_win_length"])
    writer.add_uint32 ("stt.frontend.hop_length",   hp["fe_hop_length"])
    writer.add_string ("stt.frontend.window",       hp["fe_window"])
    writer.add_string ("stt.frontend.normalize",    hp["fe_normalize"])
    writer.add_float32("stt.frontend.dither",       hp["fe_dither"])
    writer.add_float32("stt.frontend.pre_emphasis", hp["fe_pre_emphasis"])
    writer.add_float32("stt.frontend.f_min",        hp["fe_f_min"])
    writer.add_float32("stt.frontend.f_max",        hp["fe_f_max"])
    writer.add_bool   ("stt.frontend.center",       hp["fe_center"])
    writer.add_string ("stt.frontend.mel_norm",     hp["fe_mel_norm"])
    writer.add_float32("stt.frontend.log_clamp_min", hp["fe_log_clamp_min"])
    writer.add_float32("stt.frontend.log_clamp_max", hp["fe_log_clamp_max"])

    # ----- tensors -----
    consumed: set[str] = set()
    n_added = 0
    bytes_out = 0

    def add(src_name: str, dst_name: str) -> None:
        nonlocal n_added, bytes_out
        if src_name not in sd_keys:
            raise KeyError(f"state_dict missing tensor: {src_name!r}")
        arr = tensor_to_fp32_numpy(sd[src_name])
        writer.add_tensor(dst_name, arr, raw_dtype=GGMLQuantizationType.F32)
        consumed.add(src_name)
        bytes_out += int(arr.nbytes)
        n_added += 1

    def add_combined(src_a: str, src_b: str, dst_name: str) -> None:
        """Sum two source tensors. Used to collapse PyTorch LSTM
        bias_ih + bias_hh into the single pred.lstm.{i}.bias the C++
        loader expects."""
        nonlocal n_added, bytes_out
        for k in (src_a, src_b):
            if k not in sd_keys:
                raise KeyError(f"state_dict missing tensor: {k!r}")
        a = tensor_to_fp32_numpy(sd[src_a])
        b = tensor_to_fp32_numpy(sd[src_b])
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch {src_a} {a.shape} vs {src_b} {b.shape}")
        arr = np.ascontiguousarray(a + b)
        writer.add_tensor(dst_name, arr, raw_dtype=GGMLQuantizationType.F32)
        consumed.add(src_a)
        consumed.add(src_b)
        bytes_out += int(arr.nbytes)
        n_added += 1

    # Frontend buffers (baked).
    writer.add_tensor("frontend.mel_filterbank", fb,
                      raw_dtype=GGMLQuantizationType.F32)
    writer.add_tensor("frontend.window", window,
                      raw_dtype=GGMLQuantizationType.F32)
    consumed.add("preprocessor.featurizer.0.mel_scale.fb")
    consumed.add("preprocessor.featurizer.0.spectrogram.window")
    bytes_out += int(fb.nbytes) + int(window.nbytes)
    n_added += 2

    # pre_encode
    for src, dst in ENCODER_PRE_ENCODE_TABLE:
        add(src, dst)

    # encoder blocks
    for i in range(hp["enc_n_layers"]):
        for suffix_src, suffix_dst in ENCODER_BLOCK_TABLE:
            add(f"encoder.layers.{i}.{suffix_src}",
                f"enc.blocks.{i}.{suffix_dst}")

    if head_kind == "rnnt":
        # Predictor: embed + n_layers * (Wx, Wh, bias).
        add("head.decoder.embed.weight", "pred.embed.weight")
        for i in range(hp["pred_n_layers"]):
            add(f"head.decoder.lstm.weight_ih_l{i}", f"pred.lstm.{i}.Wx")
            add(f"head.decoder.lstm.weight_hh_l{i}", f"pred.lstm.{i}.Wh")
            add_combined(
                f"head.decoder.lstm.bias_ih_l{i}",
                f"head.decoder.lstm.bias_hh_l{i}",
                f"pred.lstm.{i}.bias",
            )
        # Joint.
        for src, dst in RNNT_JOINT_TABLE:
            add(src, dst)
        head_tensors = 1 + hp["pred_n_layers"] * 3 + len(RNNT_JOINT_TABLE)
    elif head_kind == "ctc":
        for src, dst in CTC_HEAD_TABLE:
            add(src, dst)
        head_tensors = len(CTC_HEAD_TABLE)
    else:
        raise ValueError(f"unsupported head_kind {head_kind!r}")
    expected = (
        2  # frontend.mel_filterbank + frontend.window
        + len(ENCODER_PRE_ENCODE_TABLE)
        + hp["enc_n_layers"] * len(ENCODER_BLOCK_TABLE)
        + head_tensors
    )
    if n_added != expected:
        raise RuntimeError(
            f"tensor count mismatch: added {n_added}, expected {expected}"
        )
    print(f"Added {n_added} tensors ({bytes_out / (1024 * 1024):.1f} MB)")

    unused = sorted(k for k in (sd_keys - consumed) if not is_expected_unused(k))
    if unused:
        print(f"WARNING: {len(unused)} state_dict keys were not consumed:",
              file=sys.stderr)
        for k in unused[:10]:
            print(f"  {k}", file=sys.stderr)
        if len(unused) > 10:
            print(f"  ... and {len(unused) - 10} more", file=sys.stderr)

    print("Writing header + KV + tensor info...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    print("Writing tensor data...")
    writer.write_tensors_to_file()
    writer.close()

    print(f"Done. Wrote {out_path} "
          f"({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a GigaAM-v3 author-repo checkpoint to a GGUF.",
    )
    p.add_argument(
        "model",
        type=str,
        help="HF repo id (informational; e.g. ai-sage/GigaAM-v3) — the "
             "gigaam package downloads from its own URL list keyed by "
             "--variant-key.",
    )
    p.add_argument(
        "out_path",
        type=Path,
        nargs="?",
        help="Output .gguf path. If omitted, derived from --repo-id "
             "(the per-variant family slug).",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="One of the five gigaam-v3-* family slugs (drives the output "
             "filename and selects the VARIANT_PROFILES entry).",
    )
    p.add_argument(
        "--variant-key",
        type=str,
        default=None,
        help="gigaam.load_model() key (v3_e2e_rnnt, v3_e2e_ctc, v3_rnnt, "
             "v3_ctc). If omitted, derived from --repo-id.",
    )
    args = p.parse_args(argv[1:])

    # Derive slug.
    slug = None
    if args.repo_id:
        slug = slug_from_repo_id(args.repo_id)
    elif args.out_path is not None:
        slug = args.out_path.parent.name
    if slug is None or slug not in VARIANT_PROFILES:
        print(
            f"error: provide --repo-id (one of {sorted(VARIANT_PROFILES)})",
            file=sys.stderr,
        )
        return 2
    profile = VARIANT_PROFILES[slug]

    variant_key = args.variant_key or profile["variant_key"]

    out_path = args.out_path
    if out_path is None:
        out_path = REPO_ROOT / "models" / slug / gguf_name(slug, REFERENCE_DTYPE_LABEL)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    convert(variant_key, slug, out_path, repo_id=args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
