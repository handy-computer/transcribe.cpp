#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gguf>=0.10.0",
#     "numpy>=1.26",
#     "safetensors>=0.4",
#     "sentencepiece>=0.2",
#     "torch>=2.0",
# ]
# ///
"""
convert-cohere.py - convert a Cohere ASR HuggingFace directory to a GGUF
that transcribe.cpp's loader can ingest end-to-end. Defaults to f16.
Pass --quant <preset> to pick a canned preset (f32, bf16, f16, q8_0,
q4_0/q4_1/q5_0/q5_1, q6_k, q4_k_m, q5_k_m), or pass a raw ggml type
name (e.g. --quant Q3_K) to override the linear bucket with any
GGMLQuantizationType while conv/norm/embed stay at their defaults.

Source format:
    A directory from HuggingFace, e.g. cohere-transcribe-03-2026/, with:

      config.json             Model config (encoder/decoder dims, frontend)
      generation_config.json  Special token IDs (bos, eos, pad, decoder_start)
      tokenizer.model         SentencePiece BPE model (binary)
      model.safetensors       ~5 GB bfloat16 weights

    All weights are bfloat16. The converter reads them via safetensors
    with framework="pt" (torch) since numpy does not support bf16, then
    converts to fp32 via .float().numpy() before quantization.

    Weights are already in PyTorch layout (OIHW for conv2d, [out, in/g, k]
    for conv1d). NO transposition is needed for conv weights (unlike the
    Parakeet converter which handles MLX layout).

Architecture differences from Parakeet:
    - Encoder-decoder (not transducer): NO predictor/LSTM, NO joint network
    - Transformer decoder with cross-attention
    - Tied token embedding: log_softmax.mlp.layer0.weight is bitwise
      identical to transf_decoder._embedding.token_embedding.weight.
      Store only ONE copy as dec.embed.token.weight. The bias is stored
      separately as head.bias.
    - Encoder-decoder projection (linear layer between encoder and decoder)

Layout conversions: NONE. Source is PyTorch layout, which is the target.

Tensors included from preprocessor:
    - preprocessor.featurizer.fb       [1, 128, 257] — mel filterbank (stored as f32)
    - preprocessor.featurizer.window   [400]         — Hann window (stored as f32)

Tensors skipped:
    - encoder.layers.{i}.conv.batch_norm.num_batches_tracked — PyTorch bookkeeping

KV emitted (matches the C++ loader's read_cohere_hparams):

  general.architecture = "cohere_asr"
  general.basename     = "cohere-transcribe"
  general.size_label   = computed from total params
  general.languages    = [14 languages]

  stt.variant          = "cohere-transcribe-03-2026"

  tokenizer.ggml.model = "bpe"
  tokenizer.ggml.tokens / scores / token_type / *_token_id

  stt.cohere.encoder.* / decoder.* / head.*
  stt.cohere.decoder_start_token_id
  stt.frontend.*

CLI:
    uv run scripts/convert-cohere.py <model-dir> <out.gguf> [--quant f16]

The script is intentionally one file with no helpers — linear flow for
easy auditing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType
from safetensors import safe_open
import sentencepiece as spm


# ---------------------------------------------------------------------------
# Quantization presets
# ---------------------------------------------------------------------------
#
# Each preset declares a target ggml type per tensor "bucket" and the
# llama-style file_type tag. Buckets:
#
#   linear: ggml_mul_mat operands. Encoder FF + attention projections,
#           decoder self/cross-attention, FFN, enc-dec proj.
#
#   embed:  the token embedding + head bias. Tied weights, so the
#           embedding matrix is ALSO the output projection — hurts WER
#           noticeably when pushed below the linear bucket. Mixed
#           k-quant presets (q4_k_m / q5_k_m) bump this to Q6_K,
#           matching the llama.cpp convention. Presets that omit
#           `embed` fall back to their `linear` bucket.
#
#   conv:   conv kernels (pre_encode + per-block conv module). Left at
#           F32 since im2col has no quantized path in ggml.
#
#   norm:   biases, LayerNorm scale/bias, BatchNorm stats, positional
#           biases/encodings. Tiny and precision-sensitive — stays fp32.

QUANT_PRESETS: dict[str, dict] = {
    "f32": {
        "linear":    GGMLQuantizationType.F32,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.ALL_F32,
    },
    "bf16": {
        "linear":    GGMLQuantizationType.BF16,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_BF16,
    },
    "f16": {
        "linear":    GGMLQuantizationType.F16,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_F16,
    },
    "q8_0": {
        "linear":    GGMLQuantizationType.Q8_0,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q8_0,
    },
    # Legacy linear quants. Conv/norm/embed stay at F32 — the embed
    # fallback to linear means dec.embed.token.weight also gets this
    # bucket, matching llama.cpp's "mostly X" behaviour.
    "q4_0": {
        "linear":    GGMLQuantizationType.Q4_0,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q4_0,
    },
    "q4_1": {
        "linear":    GGMLQuantizationType.Q4_1,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q4_1,
    },
    "q5_0": {
        "linear":    GGMLQuantizationType.Q5_0,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q5_0,
    },
    "q5_1": {
        "linear":    GGMLQuantizationType.Q5_1,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q5_1,
    },
    # Pure k-quant. Q6_K for all linear weights is the highest-quality
    # k-quant. Embed falls back to linear (Q6_K), which is fine.
    "q6_k": {
        "linear":    GGMLQuantizationType.Q6_K,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q6_K,
    },
    # Mixed k-quants (llama.cpp convention): Q4_K / Q5_K for most
    # linear weights, but bump the tied token embedding to Q6_K for
    # quality. `embed` bucket is explicit for these presets.
    "q4_k_m": {
        "linear":    GGMLQuantizationType.Q4_K,
        "embed":     GGMLQuantizationType.Q6_K,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q4_K_M,
    },
    "q5_k_m": {
        "linear":    GGMLQuantizationType.Q5_K,
        "embed":     GGMLQuantizationType.Q6_K,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q5_K_M,
    },
}


# Set of GGMLQuantizationType names we accept as an ad-hoc --quant value
# (so users can say `--quant Q3_K` without editing the preset table).
# We only include types that `gguf.quants.quantize` can actually produce
# AND that the C++ loader has runtime support for via standard ggml
# dequantization. Tensors outside the `linear` bucket stay at their
# QUANT_PRESETS["q8_0"] conv/norm defaults (F32).
_RAW_LINEAR_TYPES: dict[str, GGMLQuantizationType] = {
    "F32":   GGMLQuantizationType.F32,
    "F16":   GGMLQuantizationType.F16,
    "BF16":  GGMLQuantizationType.BF16,
    "Q4_0":  GGMLQuantizationType.Q4_0,
    "Q4_1":  GGMLQuantizationType.Q4_1,
    "Q5_0":  GGMLQuantizationType.Q5_0,
    "Q5_1":  GGMLQuantizationType.Q5_1,
    "Q8_0":  GGMLQuantizationType.Q8_0,
    "Q2_K":  GGMLQuantizationType.Q2_K,
    "Q3_K":  GGMLQuantizationType.Q3_K,
    "Q4_K":  GGMLQuantizationType.Q4_K,
    "Q5_K":  GGMLQuantizationType.Q5_K,
    "Q6_K":  GGMLQuantizationType.Q6_K,
    "IQ4_NL": GGMLQuantizationType.IQ4_NL,
    "IQ4_XS": GGMLQuantizationType.IQ4_XS,
}


def classify_tensor(gguf_name: str) -> str:
    """Return the bucket name ('linear', 'conv', 'norm', 'embed') for
    a canonical Cohere GGUF tensor name. Drives QUANT_PRESETS lookup
    at write time. `embed` covers the tied token-embedding matrix and
    the head bias row; presets that omit `embed` fall back to `linear`.
    """
    # Tied token embedding (also serves as the output projection).
    # Gets the `embed` bucket so mixed k-quant presets can bump it
    # to Q6_K. Note head.bias stays in the norm bucket — the C++
    # loader requires it as F32 since it's a 1D per-logit offset
    # added after the matmul.
    if gguf_name == "dec.embed.token.weight":
        return "embed"
    # Biases of every kind — short and precision-sensitive.
    if gguf_name.endswith(".bias"):
        return "norm"
    # BatchNorm: bn.{weight,bias,running_mean,running_var}.
    if ".bn." in gguf_name:
        return "norm"
    # LayerNorm scale (norm_ff1.weight, norm_attn.weight, norm_self.weight, etc).
    if "norm_" in gguf_name and gguf_name.endswith(".weight"):
        return "norm"
    # Decoder norms: dec.embed.norm.weight, dec.blocks.{i}.norm_*.weight, dec.final_norm.weight
    if "norm." in gguf_name and gguf_name.endswith(".weight"):
        return "norm"
    if gguf_name.endswith("final_norm.weight"):
        return "norm"
    # Per-head positional biases — added directly to fp32 q.
    if gguf_name.endswith(".pos_bias_u") or gguf_name.endswith(".pos_bias_v"):
        return "norm"
    # Positional encoding — precision-sensitive sinusoidal table.
    if gguf_name.endswith(".pos_enc"):
        return "norm"
    # Conv kernels: enc.pre_encode.conv.*.weight and
    # enc.blocks.{i}.conv.{pointwise1,depthwise,pointwise2}.weight.
    if ".conv." in gguf_name and gguf_name.endswith(".weight"):
        return "conv"
    # Everything else: linear weights (attention, FFN, embeddings, projections).
    return "linear"


def encode_for_gguf(arr: np.ndarray, target: GGMLQuantizationType) -> tuple[np.ndarray, GGMLQuantizationType | None]:
    """Convert an fp32 numpy array to whatever wire format `target`
    requires, and return (encoded, raw_dtype) for `add_tensor`.
    """
    if arr.dtype != np.float32:
        raise TypeError(f"encode_for_gguf expects fp32 input, got {arr.dtype}")

    if target == GGMLQuantizationType.F32:
        return arr, None
    if target == GGMLQuantizationType.F16:
        return arr.astype(np.float16), GGMLQuantizationType.F16
    if target == GGMLQuantizationType.BF16:
        packed = gguf.quants.quantize(arr, GGMLQuantizationType.BF16)
        return packed, GGMLQuantizationType.BF16

    packed = gguf.quants.quantize(arr, target)
    return packed, target


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------

TOKEN_TYPE_NORMAL  = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER    = 4
TOKEN_TYPE_UNUSED  = 5
TOKEN_TYPE_BYTE    = 6


def extract_tokenizer(tokenizer_model_path: Path):
    """Read a SentencePiece model and return the GGUF tokenizer payload.

    Returns a dict ready to feed into GGUFWriter:
      tokens: list[str]   length vocab_size (no extra blank appended)
      scores: list[float] length matches tokens
      types:  list[int]   length matches tokens
      unk_id: int or None (from SentencePiece)

    Unlike Parakeet, Cohere ASR has no CTC blank token and no predictor
    embed table, so we do NOT append a <blank> row. The vocab is exactly
    what SentencePiece provides (16384 tokens).
    """
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer_model_path))

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

    def safe_id(method) -> int | None:
        v = method()
        return v if v >= 0 else None

    return {
        "tokens":   tokens,
        "scores":   scores,
        "types":    types,
        "unk_id":   safe_id(sp.unk_id),
    }


# ---------------------------------------------------------------------------
# Hparams from config.json + generation_config.json
# ---------------------------------------------------------------------------


def read_hparams(config: dict, gen_config: dict) -> dict:
    """Pull every hparam the loader's read_cohere_hparams() requires.

    The config.json carries encoder/decoder/head dims plus preprocessor
    settings. generation_config.json carries special token IDs.
    """
    enc  = config["encoder"]
    pre  = config["preprocessor"]
    dec  = config["transf_decoder"]["config_dict"]

    sample_rate   = int(pre["sample_rate"])
    window_size   = float(pre["window_size"])    # seconds
    window_stride = float(pre["window_stride"])  # seconds
    win_length    = int(round(window_size   * sample_rate))
    hop_length    = int(round(window_stride * sample_rate))

    return {
        "enc_n_layers":             int(enc["n_layers"]),
        "enc_d_model":              int(enc["d_model"]),
        "enc_n_heads":              int(enc["n_heads"]),
        "enc_d_ff":                 int(enc["d_model"]) * int(enc["ff_expansion_factor"]),
        "enc_conv_kernel":          int(enc["conv_kernel_size"]),
        "enc_subsampling_factor":   int(enc["subsampling_factor"]),
        "enc_subsampling_channels": int(enc["subsampling_conv_channels"]),
        "enc_pos_emb_max_len":      int(enc["pos_emb_max_len"]),
        "enc_use_bias":             True,  # Cohere ASR encoder always uses bias

        "dec_n_layers":   int(dec["num_layers"]),
        "dec_hidden":     int(dec["hidden_size"]),
        "dec_n_heads":    int(dec["num_attention_heads"]),
        "dec_inner":      int(dec["inner_size"]),
        "dec_max_seq":    int(dec["max_sequence_length"]),
        "dec_activation": str(dec["hidden_act"]).lower(),

        "vocab_size":             int(config["vocab_size"]),
        "decoder_start_token_id": int(gen_config["decoder_start_token_id"]),
        "bos_token_id":           int(gen_config["bos_token_id"]),
        "eos_token_id":           int(gen_config["eos_token_id"]),
        "pad_token_id":           int(gen_config["pad_token_id"]),

        "head_log_softmax": bool(config["head"]["log_softmax"]),

        "languages": config.get("supported_languages", ["en"]),

        # Frontend
        "fe_type":         "mel",
        "fe_num_mels":     int(pre["features"]),
        "fe_sample_rate":  sample_rate,
        "fe_n_fft":        int(pre["n_fft"]),
        "fe_win_length":   win_length,
        "fe_hop_length":   hop_length,
        "fe_window":       str(pre["window"]),
        "fe_normalize":    str(pre["normalize"]),
        "fe_dither":       0.0,                        # inference dither = 0
        "fe_pre_emphasis": 0.97,                       # NeMo default
        "fe_f_min":        0.0,                        # NeMo default
        "fe_f_max":        float(sample_rate) / 2.0,   # Nyquist
        "fe_pad_mode":     "constant",
    }


# ---------------------------------------------------------------------------
# Tensor name + shape mapping
# ---------------------------------------------------------------------------
#
# Weights are already in PyTorch layout. No transpositions needed.


def passthrough(arr: np.ndarray) -> np.ndarray:
    """No layout change needed — weights are already in PyTorch order."""
    return np.ascontiguousarray(arr)


# Per-encoder-layer block: 39 tensors stored (40 in safetensors minus
# num_batches_tracked).
ENCODER_BLOCK_TABLE: list[tuple[str, str, callable]] = [
    # Macaron FF1.
    ("norm_feed_forward1.weight",       "norm_ff1.weight",        passthrough),
    ("norm_feed_forward1.bias",         "norm_ff1.bias",          passthrough),
    ("feed_forward1.linear1.weight",    "ff1.linear1.weight",     passthrough),
    ("feed_forward1.linear1.bias",      "ff1.linear1.bias",       passthrough),
    ("feed_forward1.linear2.weight",    "ff1.linear2.weight",     passthrough),
    ("feed_forward1.linear2.bias",      "ff1.linear2.bias",       passthrough),

    # Self-attention with relative position.
    ("norm_self_att.weight",            "norm_attn.weight",        passthrough),
    ("norm_self_att.bias",              "norm_attn.bias",          passthrough),
    ("self_attn.linear_q.weight",       "attn.linear_q.weight",   passthrough),
    ("self_attn.linear_q.bias",         "attn.linear_q.bias",     passthrough),
    ("self_attn.linear_k.weight",       "attn.linear_k.weight",   passthrough),
    ("self_attn.linear_k.bias",         "attn.linear_k.bias",     passthrough),
    ("self_attn.linear_v.weight",       "attn.linear_v.weight",   passthrough),
    ("self_attn.linear_v.bias",         "attn.linear_v.bias",     passthrough),
    ("self_attn.linear_out.weight",     "attn.linear_out.weight", passthrough),
    ("self_attn.linear_out.bias",       "attn.linear_out.bias",   passthrough),
    ("self_attn.linear_pos.weight",     "attn.linear_pos.weight", passthrough),
    ("self_attn.pos_bias_u",            "attn.pos_bias_u",        passthrough),
    ("self_attn.pos_bias_v",            "attn.pos_bias_v",        passthrough),

    # Convolution module.
    ("norm_conv.weight",                "norm_conv.weight",            passthrough),
    ("norm_conv.bias",                  "norm_conv.bias",              passthrough),
    ("conv.pointwise_conv1.weight",     "conv.pointwise1.weight",     passthrough),
    ("conv.pointwise_conv1.bias",       "conv.pointwise1.bias",       passthrough),
    ("conv.depthwise_conv.weight",      "conv.depthwise.weight",      passthrough),
    ("conv.depthwise_conv.bias",        "conv.depthwise.bias",        passthrough),
    ("conv.pointwise_conv2.weight",     "conv.pointwise2.weight",     passthrough),
    ("conv.pointwise_conv2.bias",       "conv.pointwise2.bias",       passthrough),
    ("conv.batch_norm.weight",          "conv.bn.weight",             passthrough),
    ("conv.batch_norm.bias",            "conv.bn.bias",               passthrough),
    ("conv.batch_norm.running_mean",    "conv.bn.running_mean",       passthrough),
    ("conv.batch_norm.running_var",     "conv.bn.running_var",        passthrough),

    # Macaron FF2.
    ("norm_feed_forward2.weight",       "norm_ff2.weight",        passthrough),
    ("norm_feed_forward2.bias",         "norm_ff2.bias",          passthrough),
    ("feed_forward2.linear1.weight",    "ff2.linear1.weight",     passthrough),
    ("feed_forward2.linear1.bias",      "ff2.linear1.bias",       passthrough),
    ("feed_forward2.linear2.weight",    "ff2.linear2.weight",     passthrough),
    ("feed_forward2.linear2.bias",      "ff2.linear2.bias",       passthrough),

    # Final per-block layer norm.
    ("norm_out.weight",                 "norm_out.weight",        passthrough),
    ("norm_out.bias",                   "norm_out.bias",          passthrough),
]


PRE_ENCODE_TABLE: list[tuple[str, str, callable]] = [
    ("encoder.pre_encode.conv.0.weight", "enc.pre_encode.conv.0.weight", passthrough),
    ("encoder.pre_encode.conv.0.bias",   "enc.pre_encode.conv.0.bias",   passthrough),
    ("encoder.pre_encode.conv.2.weight", "enc.pre_encode.conv.2.weight", passthrough),
    ("encoder.pre_encode.conv.2.bias",   "enc.pre_encode.conv.2.bias",   passthrough),
    ("encoder.pre_encode.conv.3.weight", "enc.pre_encode.conv.3.weight", passthrough),
    ("encoder.pre_encode.conv.3.bias",   "enc.pre_encode.conv.3.bias",   passthrough),
    ("encoder.pre_encode.conv.5.weight", "enc.pre_encode.conv.5.weight", passthrough),
    ("encoder.pre_encode.conv.5.bias",   "enc.pre_encode.conv.5.bias",   passthrough),
    ("encoder.pre_encode.conv.6.weight", "enc.pre_encode.conv.6.weight", passthrough),
    ("encoder.pre_encode.conv.6.bias",   "enc.pre_encode.conv.6.bias",   passthrough),
    ("encoder.pre_encode.out.weight",    "enc.pre_encode.out.weight",    passthrough),
    ("encoder.pre_encode.out.bias",      "enc.pre_encode.out.bias",      passthrough),
]


ENC_DEC_PROJ_TABLE: list[tuple[str, str, callable]] = [
    ("encoder_decoder_proj.weight", "enc_dec_proj.weight", passthrough),
    ("encoder_decoder_proj.bias",   "enc_dec_proj.bias",   passthrough),
]


DEC_EMBED_TABLE: list[tuple[str, str, callable]] = [
    ("transf_decoder._embedding.token_embedding.weight",       "dec.embed.token.weight",  passthrough),
    ("transf_decoder._embedding.position_embedding.pos_enc",   "dec.embed.pos_enc",       passthrough),
    ("transf_decoder._embedding.layer_norm.weight",            "dec.embed.norm.weight",   passthrough),
    ("transf_decoder._embedding.layer_norm.bias",              "dec.embed.norm.bias",     passthrough),
]


# Per-decoder-layer block: 26 tensors per layer.
DECODER_BLOCK_TABLE: list[tuple[str, str, callable]] = [
    # Self-attention sub-layer
    ("layer_norm_1.weight",                     "norm_self.weight",       passthrough),
    ("layer_norm_1.bias",                       "norm_self.bias",         passthrough),
    ("first_sub_layer.query_net.weight",        "self_attn.q.weight",    passthrough),
    ("first_sub_layer.query_net.bias",          "self_attn.q.bias",      passthrough),
    ("first_sub_layer.key_net.weight",          "self_attn.k.weight",    passthrough),
    ("first_sub_layer.key_net.bias",            "self_attn.k.bias",      passthrough),
    ("first_sub_layer.value_net.weight",        "self_attn.v.weight",    passthrough),
    ("first_sub_layer.value_net.bias",          "self_attn.v.bias",      passthrough),
    ("first_sub_layer.out_projection.weight",   "self_attn.out.weight",  passthrough),
    ("first_sub_layer.out_projection.bias",     "self_attn.out.bias",    passthrough),

    # Cross-attention sub-layer
    ("layer_norm_2.weight",                      "norm_cross.weight",      passthrough),
    ("layer_norm_2.bias",                        "norm_cross.bias",        passthrough),
    ("second_sub_layer.query_net.weight",        "cross_attn.q.weight",   passthrough),
    ("second_sub_layer.query_net.bias",          "cross_attn.q.bias",     passthrough),
    ("second_sub_layer.key_net.weight",          "cross_attn.k.weight",   passthrough),
    ("second_sub_layer.key_net.bias",            "cross_attn.k.bias",     passthrough),
    ("second_sub_layer.value_net.weight",        "cross_attn.v.weight",   passthrough),
    ("second_sub_layer.value_net.bias",          "cross_attn.v.bias",     passthrough),
    ("second_sub_layer.out_projection.weight",   "cross_attn.out.weight", passthrough),
    ("second_sub_layer.out_projection.bias",     "cross_attn.out.bias",   passthrough),

    # Feed-forward sub-layer
    ("layer_norm_3.weight",                     "norm_ff.weight",         passthrough),
    ("layer_norm_3.bias",                       "norm_ff.bias",           passthrough),
    ("third_sub_layer.dense_in.weight",         "ff.dense_in.weight",    passthrough),
    ("third_sub_layer.dense_in.bias",           "ff.dense_in.bias",      passthrough),
    ("third_sub_layer.dense_out.weight",        "ff.dense_out.weight",   passthrough),
    ("third_sub_layer.dense_out.bias",          "ff.dense_out.bias",     passthrough),
]


DEC_FINAL_NORM_TABLE: list[tuple[str, str, callable]] = [
    ("transf_decoder._decoder.final_layer_norm.weight", "dec.final_norm.weight", passthrough),
    ("transf_decoder._decoder.final_layer_norm.bias",   "dec.final_norm.bias",   passthrough),
]


HEAD_TABLE: list[tuple[str, str, callable]] = [
    # log_softmax.mlp.layer0.weight is tied to dec.embed.token.weight — skip it.
    # Only store the bias.
    ("log_softmax.mlp.layer0.bias", "head.bias", passthrough),
]


# Tensors to skip: BN tracking counter.  The preprocessor filterbank and
# window are explicitly converted (not skipped); only other preprocessor
# keys are ignored.
SKIP_PREFIXES = (
    "preprocessor.featurizer.stft.",  # STFT kernel, not needed
)
SKIP_SUFFIXES = (
    ".num_batches_tracked",
)
# The head weight is tied to the embedding; skip it explicitly.
SKIP_EXACT = {
    "log_softmax.mlp.layer0.weight",
}


def _compute_size_label(total_params: int) -> str:
    """Format total parameter count as a human-readable size label."""
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    elif total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.0f}M"
    else:
        return f"{total_params / 1_000:.0f}K"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def _resolve_quant(quant: str) -> tuple[str, dict]:
    """Resolve --quant into (label, preset_dict).

    `quant` may be either a QUANT_PRESETS name (e.g. "q4_k_m") OR a
    raw ggml type name from _RAW_LINEAR_TYPES (e.g. "Q3_K"). For raw
    types we synthesize a preset on the fly with conv/norm=F32 and
    no explicit `embed` entry (so embed falls back to linear).
    """
    if quant in QUANT_PRESETS:
        return quant, QUANT_PRESETS[quant]

    raw = quant.upper()
    if raw in _RAW_LINEAR_TYPES:
        linear_type = _RAW_LINEAR_TYPES[raw]
        # No matching LlamaFileType for every raw override; just tag
        # as GUESSED so the loader doesn't complain.
        return f"custom:{raw}", {
            "linear":    linear_type,
            "conv":      GGMLQuantizationType.F32,
            "norm":      GGMLQuantizationType.F32,
            "file_type": LlamaFileType.GUESSED,
        }

    raise ValueError(
        f"unknown --quant value: {quant!r}; "
        f"known presets: {sorted(QUANT_PRESETS)}, "
        f"or a raw ggml type: {sorted(_RAW_LINEAR_TYPES)}"
    )


def _bucket_type(preset: dict, bucket: str) -> GGMLQuantizationType:
    """Look up the target quant type for a bucket with the embed -> linear
    fallback rule. Presets that omit `embed` reuse `linear`."""
    if bucket in preset:
        return preset[bucket]
    if bucket == "embed":
        return preset["linear"]
    raise KeyError(f"preset missing required bucket: {bucket!r}")


# ---------------------------------------------------------------------------
# Two-step dispatch for types gguf-py can't quantize
# ---------------------------------------------------------------------------
#
# gguf-py 0.18 implements F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
# directly. Every K-quant (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) and every IQ
# quant raises NotImplementedError from gguf.quants.quantize. For those
# targets we fall back to a two-step: first write a bf16 GGUF via the
# pure-Python path, then shell out to the `transcribe-quantize` C tool
# to re-quantize. The tool lives at build/bin/transcribe-quantize and
# is gated behind -DTRANSCRIBE_BUILD_TOOLS=ON.

# Presets that the C-tool handles. Names must match transcribe-quantize
# PRESETS in tools/transcribe-quantize/main.cpp.
_CTOOL_PRESETS = {"q4_k_m", "q5_k_m", "q6_k"}

# Raw ggml types that require the C-tool path (Python can't produce them).
_CTOOL_RAW_TYPES = {
    GGMLQuantizationType.Q2_K,
    GGMLQuantizationType.Q3_K,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
    GGMLQuantizationType.Q6_K,
    GGMLQuantizationType.IQ4_NL,
    GGMLQuantizationType.IQ4_XS,
}


def _needs_ctool(quant: str) -> bool:
    """Return True if `quant` requires the transcribe-quantize C tool
    (i.e. the Python gguf package can't produce it)."""
    if quant in _CTOOL_PRESETS:
        return True
    if quant in QUANT_PRESETS:
        return False
    raw = quant.upper()
    if raw in _RAW_LINEAR_TYPES:
        return _RAW_LINEAR_TYPES[raw] in _CTOOL_RAW_TYPES
    return False


def _find_transcribe_quantize() -> Path:
    """Locate the transcribe-quantize binary. Scripts run from anywhere;
    the tool lives at transcribe.cpp/build/bin/transcribe-quantize
    relative to this script."""
    script_dir = Path(__file__).resolve().parent
    candidate = script_dir.parent / "build" / "bin" / "transcribe-quantize"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        f"transcribe-quantize binary not found at {candidate}. "
        "Rebuild with -DTRANSCRIBE_BUILD_TOOLS=ON: "
        "cmake -B build -DTRANSCRIBE_BUILD_TOOLS=ON && "
        "cmake --build build --target transcribe-quantize"
    )


def _convert_via_ctool(model_dir: Path, out_path: Path, quant: str) -> None:
    """Two-step convert: write a bf16 intermediate via the Python path,
    then run transcribe-quantize to produce `quant`. Deletes the
    intermediate on success."""
    import subprocess
    import tempfile

    tool = _find_transcribe_quantize()

    # Raw ggml types need a preset name the C tool recognizes. For now
    # the tool only ships q4_k_m / q5_k_m / q6_k / q8_0 / f16; reject
    # ad-hoc raw-type overrides that don't have a matching preset.
    if quant not in _CTOOL_PRESETS:
        raise ValueError(
            f"--quant {quant!r} needs the C tool but transcribe-quantize "
            f"has no preset for it. Supported C-tool presets: "
            f"{sorted(_CTOOL_PRESETS)}"
        )

    with tempfile.TemporaryDirectory(prefix="cohere-quant-", dir=str(out_path.parent)) as td:
        intermediate = Path(td) / f"{out_path.stem}.bf16.intermediate.gguf"
        print(
            f"Two-step quant: writing bf16 intermediate to {intermediate.name}, "
            f"then requantizing to {quant} via {tool}"
        )
        # Step 1: Python convert → bf16 intermediate.
        convert(model_dir, intermediate, "bf16")

        # Step 2: C tool re-quantize.
        cmd = [str(tool), str(intermediate), str(out_path), "--quant", quant]
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"transcribe-quantize failed with exit code {result.returncode}"
            )

    print(f"Done. Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


def convert(model_dir: Path, out_path: Path, quant: str) -> None:
    quant_label, preset = _resolve_quant(quant)
    embed_type = _bucket_type(preset, "embed")
    print(
        f"Quant: {quant_label} "
        f"(linear={preset['linear'].name}, "
        f"conv={preset['conv'].name}, "
        f"norm={preset['norm'].name}, "
        f"embed={embed_type.name})"
    )

    config_path      = model_dir / "config.json"
    gen_config_path  = model_dir / "generation_config.json"
    tokenizer_path   = model_dir / "tokenizer.model"
    safetensors_path = model_dir / "model.safetensors"

    for p in (config_path, gen_config_path, tokenizer_path, safetensors_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file: {p}")

    print(f"Reading config from {config_path}")
    with config_path.open() as f:
        config = json.load(f)

    print(f"Reading generation config from {gen_config_path}")
    with gen_config_path.open() as f:
        gen_config = json.load(f)

    hp = read_hparams(config, gen_config)

    print(f"vocab_size = {hp['vocab_size']}")
    print(f"Variant: cohere-transcribe-03-2026")

    print(f"Reading tokenizer from {tokenizer_path}")
    tok = extract_tokenizer(tokenizer_path)
    if len(tok["tokens"]) != hp["vocab_size"]:
        raise ValueError(
            f"tokenizer length mismatch: {len(tok['tokens'])} tokens "
            f"vs config vocab_size={hp['vocab_size']}"
        )

    print(f"Opening safetensors at {safetensors_path}")
    # Use framework="pt" because safetensors are bfloat16 (numpy can't
    # handle bf16). We convert to fp32 numpy via .float().numpy().
    with safe_open(str(safetensors_path), framework="pt") as st:
        st_keys = set(st.keys())

        # Count total params for size label.
        total_params = 0
        for k in st_keys:
            t = st.get_tensor(k)
            total_params += t.numel()
        # Subtract tied weight (counted twice in safetensors).
        tied_key = "log_softmax.mlp.layer0.weight"
        if tied_key in st_keys:
            total_params -= st.get_tensor(tied_key).numel()
        size_label = _compute_size_label(total_params)
        print(f"Total params (deduplicated): {total_params:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = GGUFWriter(str(out_path), "cohere_asr")

        # ----- general.* metadata -----
        writer.add_string("general.basename",   "cohere-transcribe")
        writer.add_string("general.size_label", size_label)
        writer.add_uint32("general.file_type",  int(preset["file_type"]))
        writer.add_array("general.languages",   hp["languages"])

        # ----- stt.variant -----
        writer.add_string("stt.variant", "cohere-transcribe-03-2026")

        # ----- tokenizer.ggml.* -----
        writer.add_string("tokenizer.ggml.model", "bpe")
        writer.add_array("tokenizer.ggml.tokens",     tok["tokens"])
        writer.add_array("tokenizer.ggml.scores",     tok["scores"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        if tok["unk_id"] is not None:
            writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
        # Use HF generation_config values for bos/eos/pad, not SentencePiece's
        # (which are -1 / unset for this model).
        writer.add_uint32("tokenizer.ggml.bos_token_id",     hp["bos_token_id"])
        writer.add_uint32("tokenizer.ggml.eos_token_id",     hp["eos_token_id"])
        writer.add_uint32("tokenizer.ggml.padding_token_id", hp["pad_token_id"])

        # ----- stt.cohere.* hparams -----
        writer.add_uint32("stt.cohere.encoder.n_layers",             hp["enc_n_layers"])
        writer.add_uint32("stt.cohere.encoder.d_model",              hp["enc_d_model"])
        writer.add_uint32("stt.cohere.encoder.n_heads",              hp["enc_n_heads"])
        writer.add_uint32("stt.cohere.encoder.d_ff",                 hp["enc_d_ff"])
        writer.add_uint32("stt.cohere.encoder.conv_kernel",          hp["enc_conv_kernel"])
        writer.add_uint32("stt.cohere.encoder.subsampling_factor",   hp["enc_subsampling_factor"])
        writer.add_uint32("stt.cohere.encoder.subsampling_channels", hp["enc_subsampling_channels"])
        writer.add_uint32("stt.cohere.encoder.pos_emb_max_len",      hp["enc_pos_emb_max_len"])
        writer.add_bool  ("stt.cohere.encoder.use_bias",             hp["enc_use_bias"])

        writer.add_uint32("stt.cohere.decoder.n_layers",    hp["dec_n_layers"])
        writer.add_uint32("stt.cohere.decoder.hidden_size", hp["dec_hidden"])
        writer.add_uint32("stt.cohere.decoder.n_heads",     hp["dec_n_heads"])
        writer.add_uint32("stt.cohere.decoder.inner_size",  hp["dec_inner"])
        writer.add_uint32("stt.cohere.decoder.max_seq_len", hp["dec_max_seq"])
        writer.add_string("stt.cohere.decoder.activation",  hp["dec_activation"])

        writer.add_uint32("stt.cohere.decoder_start_token_id", hp["decoder_start_token_id"])
        writer.add_bool  ("stt.cohere.head.log_softmax",       hp["head_log_softmax"])
        writer.add_bool  ("stt.cohere.head.tied_weights",      True)

        # ----- stt.frontend.* -----
        writer.add_string ("stt.frontend.type",          hp["fe_type"])
        writer.add_uint32 ("stt.frontend.num_mels",      hp["fe_num_mels"])
        writer.add_uint32 ("stt.frontend.sample_rate",   hp["fe_sample_rate"])
        writer.add_uint32 ("stt.frontend.n_fft",         hp["fe_n_fft"])
        writer.add_uint32 ("stt.frontend.win_length",    hp["fe_win_length"])
        writer.add_uint32 ("stt.frontend.hop_length",    hp["fe_hop_length"])
        writer.add_string ("stt.frontend.window",        hp["fe_window"])
        writer.add_string ("stt.frontend.normalize",     hp["fe_normalize"])
        writer.add_float32("stt.frontend.dither",        hp["fe_dither"])
        writer.add_float32("stt.frontend.pre_emphasis",  hp["fe_pre_emphasis"])
        writer.add_float32("stt.frontend.f_min",         hp["fe_f_min"])
        writer.add_float32("stt.frontend.f_max",         hp["fe_f_max"])
        writer.add_string ("stt.frontend.pad_mode",      hp["fe_pad_mode"])

        # ----- tensors -----
        n_added       = 0
        bucket_counts = {"linear": 0, "conv": 0, "norm": 0, "embed": 0}
        bytes_in  = 0
        bytes_out = 0

        def add(src_name: str, gguf_name: str, transform) -> None:
            nonlocal n_added, bytes_in, bytes_out
            if src_name not in st_keys:
                raise KeyError(
                    f"safetensors missing tensor: {src_name!r}"
                )
            # Read as torch bf16 tensor, convert to fp32 numpy.
            t = st.get_tensor(src_name)
            arr = t.float().numpy()
            arr = transform(arr)
            if arr.dtype != np.float32:
                raise ValueError(
                    f"{src_name}: expected float32 after conversion, got {arr.dtype}"
                )
            bucket = classify_tensor(gguf_name)
            target = _bucket_type(preset, bucket)
            encoded, raw_dtype = encode_for_gguf(arr, target)
            writer.add_tensor(gguf_name, encoded, raw_dtype=raw_dtype)
            bucket_counts[bucket] += 1
            bytes_in  += int(arr.nbytes)
            bytes_out += int(encoded.nbytes)
            n_added += 1

        # pre_encode (12 tensors)
        for src_name, gguf_name, transform in PRE_ENCODE_TABLE:
            add(src_name, gguf_name, transform)

        # encoder layers (n_layers * 39 tensors)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, suffix_gguf, transform in ENCODER_BLOCK_TABLE:
                add(
                    f"encoder.layers.{i}.{suffix_src}",
                    f"enc.blocks.{i}.{suffix_gguf}",
                    transform,
                )

        # encoder-decoder projection (2 tensors)
        for src_name, gguf_name, transform in ENC_DEC_PROJ_TABLE:
            add(src_name, gguf_name, transform)

        # decoder embedding (4 tensors)
        for src_name, gguf_name, transform in DEC_EMBED_TABLE:
            add(src_name, gguf_name, transform)

        # decoder layers (n_layers * 26 tensors)
        for i in range(hp["dec_n_layers"]):
            for suffix_src, suffix_gguf, transform in DECODER_BLOCK_TABLE:
                add(
                    f"transf_decoder._decoder.layers.{i}.{suffix_src}",
                    f"dec.blocks.{i}.{suffix_gguf}",
                    transform,
                )

        # decoder final norm (2 tensors)
        for src_name, gguf_name, transform in DEC_FINAL_NORM_TABLE:
            add(src_name, gguf_name, transform)

        # head bias (1 tensor — weight is tied to dec.embed.token.weight)
        for src_name, gguf_name, transform in HEAD_TABLE:
            add(src_name, gguf_name, transform)

        # Mel frontend buffers (filterbank + window) — always stored as f32.
        # These are the exact values the model was trained with; using them
        # instead of recomputing from scratch eliminates mel-level divergence.
        fb_src = "preprocessor.featurizer.fb"
        fb_tensor = st.get_tensor(fb_src).float().numpy()
        if fb_tensor.ndim == 3:
            fb_tensor = fb_tensor.squeeze(0)  # [1, 128, 257] -> [128, 257]
        writer.add_tensor("frontend.mel_filterbank", fb_tensor)
        n_added += 1
        bytes_in += int(fb_tensor.nbytes)
        bytes_out += int(fb_tensor.nbytes)

        win_src = "preprocessor.featurizer.window"
        win_tensor = st.get_tensor(win_src).float().numpy()
        writer.add_tensor("frontend.window", win_tensor)
        n_added += 1
        bytes_in += int(win_tensor.nbytes)
        bytes_out += int(win_tensor.nbytes)

        # Validate tensor count.
        expected = (
            len(PRE_ENCODE_TABLE)                              # 12
            + hp["enc_n_layers"] * len(ENCODER_BLOCK_TABLE)    # 48 * 39 = 1872
            + len(ENC_DEC_PROJ_TABLE)                          # 2
            + len(DEC_EMBED_TABLE)                             # 4
            + hp["dec_n_layers"] * len(DECODER_BLOCK_TABLE)    # 8 * 26 = 208
            + len(DEC_FINAL_NORM_TABLE)                        # 2
            + len(HEAD_TABLE)                                  # 1
            + 2                                                # frontend fb + window
        )
        if n_added != expected:
            raise RuntimeError(
                f"tensor count mismatch: added {n_added}, expected {expected}"
            )
        print(
            f"Added {n_added} tensors "
            f"(linear={bucket_counts['linear']}, "
            f"embed={bucket_counts['embed']}, "
            f"conv={bucket_counts['conv']}, "
            f"norm={bucket_counts['norm']})"
        )
        print(
            f"Tensor data: {bytes_in / (1024 * 1024):.1f} MB in (fp32) -> "
            f"{bytes_out / (1024 * 1024):.1f} MB out "
            f"({100.0 * bytes_out / max(bytes_in, 1):.1f}% of source)"
        )

        # Warn about unconsumed safetensors keys.
        consumed = set()
        for src_name, _, _ in PRE_ENCODE_TABLE:
            consumed.add(src_name)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, _, _ in ENCODER_BLOCK_TABLE:
                consumed.add(f"encoder.layers.{i}.{suffix_src}")
        for src_name, _, _ in ENC_DEC_PROJ_TABLE:
            consumed.add(src_name)
        for src_name, _, _ in DEC_EMBED_TABLE:
            consumed.add(src_name)
        for i in range(hp["dec_n_layers"]):
            for suffix_src, _, _ in DECODER_BLOCK_TABLE:
                consumed.add(f"transf_decoder._decoder.layers.{i}.{suffix_src}")
        for src_name, _, _ in DEC_FINAL_NORM_TABLE:
            consumed.add(src_name)
        for src_name, _, _ in HEAD_TABLE:
            consumed.add(src_name)

        # Mark the frontend buffers as consumed.
        consumed.add(fb_src)
        consumed.add(win_src)

        # Also mark the explicitly skipped tensors as "consumed" so they
        # don't trigger the warning.
        for k in st_keys:
            if k in SKIP_EXACT:
                consumed.add(k)
            elif any(k.startswith(p) for p in SKIP_PREFIXES):
                consumed.add(k)
            elif any(k.endswith(s) for s in SKIP_SUFFIXES):
                consumed.add(k)

        unused = sorted(st_keys - consumed)
        if unused:
            print(
                f"WARNING: {len(unused)} safetensors keys were not consumed:",
                file=sys.stderr,
            )
            for k in unused[:20]:
                print(f"  {k}", file=sys.stderr)
            if len(unused) > 20:
                print(f"  ... and {len(unused) - 20} more", file=sys.stderr)

        print("Writing header + KV + tensor info...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        print("Writing tensor data...")
        writer.write_tensors_to_file()
        writer.close()

    print(f"Done. Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a Cohere ASR HuggingFace directory to a GGUF.",
    )
    p.add_argument("model_dir", type=Path,
                   help="Path to a cohere-transcribe-* directory")
    p.add_argument("out_path", type=Path,
                   help="Output .gguf path")
    p.add_argument(
        "--quant",
        default="f16",
        metavar="PRESET_OR_TYPE",
        help=(
            "Quantization preset OR raw ggml type name. "
            "Presets: " + ", ".join(sorted(QUANT_PRESETS)) + ". "
            "Raw ggml types (applied to the linear bucket only, "
            "conv/norm stay F32): "
            + ", ".join(sorted(_RAW_LINEAR_TYPES))
            + ". Default: f16 (matches source bf16). "
            "Mixed presets (q4_k_m, q5_k_m) bump the tied token "
            "embedding to Q6_K."
        ),
    )
    args = p.parse_args(argv[1:])

    if not args.model_dir.is_dir():
        print(f"error: {args.model_dir} is not a directory", file=sys.stderr)
        return 2

    # Dispatch: pure Python for types gguf-py can quantize directly;
    # two-step via the transcribe-quantize C tool for k-quants.
    if _needs_ctool(args.quant):
        _convert_via_ctool(args.model_dir, args.out_path, args.quant)
    else:
        convert(args.model_dir, args.out_path, args.quant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
