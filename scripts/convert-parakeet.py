#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gguf>=0.10.0",
#     "numpy>=1.26",
#     "safetensors>=0.4",
#     "sentencepiece>=0.2",
# ]
# ///
"""
convert-parakeet.py - convert a NeMo Parakeet MLX directory to a GGUF
that transcribe.cpp's loader can ingest end-to-end. Defaults to fp32;
pass --quant f16 (or future quant presets) to write a smaller file.

Source format:
    A directory laid out the way nemo->mlx exporters produce, e.g.
    parakeet-tdt-0.6b-v2-mlx/, with three files:

      config.json         NeMo training config (encoder/predictor/joint
                          dims, frontend params, tokenizer settings)
      tokenizer.model     SentencePiece BPE model (binary)
      model.safetensors   ~2.4 GB fp32 weights, MLX-style tensor layouts

    The converter reads numeric architecture from config.json and uses
    decoder.vocab_size as the v2 (1024) vs v3 (8192) discriminator. No
    --variant flag — auto-detect is sufficient because every difference
    that matters at the loader level is reflected in either the
    safetensors shapes or the config.

Target format:
    A single .gguf following the canonical names + shapes encoded in
    src/arch/parakeet/weights.cpp. The loader's per-tensor shape
    validation is the only schema cross-check; if the converter and the
    loader drift, the loader logs the offending tensor on first load.

Layout conversions performed here (so the loader doesn't have to):

  - Conv2d: NeMo MLX [out, kh, kw, in] -> numpy [out, in, kh, kw] (OIHW)
            via transpose(0, 3, 1, 2). gguf-py reverses the numpy shape
            when writing tensor info, so this lands as ggml
            ne=[kw, kh, in, out].
  - Conv1d: NeMo MLX [out, k, in_per_group] -> numpy [out, in_per_group,
            k] via transpose(0, 2, 1). ggml ne=[k, in_per_group, out].
  - Linear: NeMo numpy [out, in] passes through unchanged. ggml ne=[in, out].
  - LSTM gate matrices (Wx, Wh): NeMo concatenated [4*hidden, input]
            passes through unchanged. PyTorch i/f/g/o gate order; the
            decoder will know to slice them.
  - BN: weight, bias, running_mean, running_var pass through. The
        loader keeps all four; folding into the surrounding conv
        happens at compute time in phase 4.

KV emitted (matches transcribe::read_capability_kv,
read_languages_kv, transcribe::Tokenizer::load, and
transcribe::parakeet::read_parakeet_hparams):

  general.architecture = "parakeet"
  general.basename     = "parakeet-tdt"
  general.size_label   = "0.6B"
  general.version      = "v2" or "v3"
  general.languages    = [...]                 (1 entry for v2, 25 for v3)
  stt.variant          = "tdt-0.6b-v2" or "tdt-0.6b-v3"
  stt.capability.lang_detect = true            (v3 only; absent for v2)
  tokenizer.ggml.model = "bpe"
  tokenizer.ggml.tokens / scores / token_type / *_token_id  (the standard set)
  stt.parakeet.encoder.{n_layers,d_model,n_heads,d_ff,conv_kernel,
                        subsampling_factor,subsampling_channels,
                        pos_emb_max_len,use_bias}
  stt.parakeet.predictor.{hidden,n_layers,vocab}
  stt.parakeet.joint.{hidden,num_extra_outputs}
  stt.frontend.{type,num_mels,sample_rate,n_fft,win_length,hop_length,
                window,normalize,dither,pre_emphasis,f_min,f_max}

CLI:
    uv run scripts/convert-parakeet.py <model-dir> <out.gguf>

The script is intentionally one file with no helpers split out — it's
load-bearing exactly once per model conversion and the linear flow is
easier to audit than a layered abstraction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType
from safetensors import safe_open
import sentencepiece as spm


# ---------------------------------------------------------------------------
# Quantization presets
# ---------------------------------------------------------------------------
#
# Each preset declares a target ggml type per tensor "bucket" and the
# llama-style file_type tag that goes into general.file_type. Buckets
# match the C++ loader's GET_LIN / GET_CONV / GET_F32 macros in
# src/arch/parakeet/weights.cpp:
#
#   linear: ggml_mul_mat operands. Encoder FF + attention projections,
#           predictor LSTM gate matrices, predictor embedding,
#           joint enc/pred/out projections. ggml_mul_mat handles a
#           quantized W against an fp32 X natively, so this is the
#           bucket that grows as new quant types come online.
#
#   conv:   conv kernels (pre_encode + per-block conv module). The
#           local f32-friendly conv wrappers in encoder.cpp im2col
#           against the kernel's real type and there's no quantized
#           im2col path in ggml; cap this bucket at F16. Cost is small
#           (~5 MB total across all conv kernels) so we just leave it
#           at F32 today and revisit when we want to fight Metal
#           kernel coverage.
#
#   norm:   biases, LayerNorm scale/bias, BatchNorm stats, attn
#           pos_bias_u/v. Tiny and precision-sensitive — stays fp32
#           across every preset.
#
# Adding a new preset = add a row here. Adding a new quant type =
# extend the linear allowlist in weights.cpp::GET_LIN AND make sure
# gguf.quants.quantize() actually implements that type in the gguf
# version pinned at the top of this script (gguf 0.18.0 ships F16,
# BF16, Q8_0, Q4_0, Q5_0 in pure Python; the K-quants raise
# NotImplementedError and need a ctypes bridge to libggml).
QUANT_PRESETS: dict[str, dict] = {
    "f32": {
        "linear":    GGMLQuantizationType.F32,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.ALL_F32,
    },
    "f16": {
        "linear":    GGMLQuantizationType.F16,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_F16,
    },
    # Q8_0 is the largest blockwise quant — 8 bits + per-block fp16
    # scale, no rounding to a smaller bit-width. Block size is 32, and
    # every Parakeet 0.6B linear weight has its inner dim (ne[0]) in
    # {640, 1024, 4096}, all divisible by 32, so no per-tensor
    # fallback is needed. Q8_0 typically lands within 0.05 WER of fp16
    # on encoder-heavy ASR models (we'll measure with the WER harness
    # in a follow-up).
    "q8_0": {
        "linear":    GGMLQuantizationType.Q8_0,
        "conv":      GGMLQuantizationType.F32,
        "norm":      GGMLQuantizationType.F32,
        "file_type": LlamaFileType.MOSTLY_Q8_0,
    },
}


def classify_tensor(gguf_name: str) -> str:
    """Return the bucket name ('linear', 'conv', 'norm') for a canonical
    Parakeet GGUF tensor name. Drives QUANT_PRESETS lookup at write
    time. The classification mirrors the GET_*-macro choice in
    src/arch/parakeet/weights.cpp tensor by tensor: if you change one,
    change the other or the loader will reject the file.
    """
    # Biases of every kind — short and precision-sensitive.
    if gguf_name.endswith(".bias"):
        return "norm"
    # BatchNorm: bn.{weight,bias,running_mean,running_var}.
    if ".bn." in gguf_name:
        return "norm"
    # LayerNorm scale (norm_ff1.weight, norm_attn.weight, etc).
    # The .bias case was already caught above. norm_conv.weight does
    # NOT contain ".conv." as a substring so the conv branch below
    # won't grab it.
    if "norm_" in gguf_name and gguf_name.endswith(".weight"):
        return "norm"
    # Per-head positional biases — added directly to fp32 q via
    # ggml_add inside rel_pos_mhsa.
    if gguf_name.endswith(".pos_bias_u") or gguf_name.endswith(".pos_bias_v"):
        return "norm"
    # Conv kernels: enc.pre_encode.conv.{0,2,3,5,6}.weight and
    # enc.blocks.{i}.conv.{pointwise1,depthwise,pointwise2}.weight.
    # ".conv." (with dots on both sides) distinguishes these from
    # norm_conv.weight which we already routed above.
    if ".conv." in gguf_name and gguf_name.endswith(".weight"):
        return "conv"
    # Everything else: linear weights.
    return "linear"


def encode_for_gguf(arr: np.ndarray, target: GGMLQuantizationType) -> tuple[np.ndarray, GGMLQuantizationType | None]:
    """Convert an fp32 numpy array to whatever wire format `target`
    requires, and return (encoded, raw_dtype) for `add_tensor`.

    For F32, returns the array unchanged with raw_dtype=None (gguf-py
    infers F32 from the float32 ndarray).
    For F16, returns a float16 view that gguf-py also infers natively.
    For other quant types, returns the packed uint8 byte buffer plus
    the explicit raw_dtype tag so gguf-py records the correct type
    rather than inferring uint8.
    """
    if arr.dtype != np.float32:
        raise TypeError(f"encode_for_gguf expects fp32 input, got {arr.dtype}")

    if target == GGMLQuantizationType.F32:
        return arr, None
    if target == GGMLQuantizationType.F16:
        # gguf-py's add_tensor infers F16 from float16 ndarrays. The
        # explicit tag is harmless and makes intent obvious.
        return arr.astype(np.float16), GGMLQuantizationType.F16

    # Quantized path. gguf.quants.quantize raises NotImplementedError
    # for K-quants in this gguf version; that's caught at preset
    # registration time, not here.
    packed = gguf.quants.quantize(arr, target)
    return packed, target


# ---------------------------------------------------------------------------
# Variant profiles
# ---------------------------------------------------------------------------
#
# Each variant entry carries the bits the converter cannot derive from
# the safetensors or config.json: the variant string, the language list
# (NeMo's config.json doesn't carry it cleanly), and the
# capability flags. Auto-detected by decoder.vocab_size.

# v3 multilingual: 25 European languages, BCP-47 short codes. List
# matches the NVIDIA model card for nvidia/parakeet-tdt-0.6b-v3.
V3_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl",
    "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt",
    "mt", "pl", "pt", "ro", "ru",
    "sk", "sl", "es", "sv", "uk",
]

VARIANT_PROFILES: dict[int, dict] = {
    # v2: 0.6B English-only TDT.
    1024: {
        "variant": "tdt-0.6b-v2",
        "version": "v2",
        "languages": ["en"],
        "lang_detect": False,
    },
    # v3: 0.6B multilingual TDT.
    8192: {
        "variant": "tdt-0.6b-v3",
        "version": "v3",
        "languages": V3_LANGUAGES,
        "lang_detect": True,
    },
}


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------

# llama.cpp / whisper.cpp tokenizer.ggml.token_type values. We follow
# the same conventions so an inspector built for either project can
# read our GGUFs without surprises.
TOKEN_TYPE_NORMAL  = 1
TOKEN_TYPE_UNKNOWN = 2
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER    = 4
TOKEN_TYPE_UNUSED  = 5
TOKEN_TYPE_BYTE    = 6


def extract_tokenizer(tokenizer_model_path: Path, blank_piece: str = "<blank>"):
    """Read a SentencePiece model and return the GGUF tokenizer payload.

    Returns a dict ready to feed into GGUFWriter:
      tokens: list[str]   length vocab_size + 1 (last entry is <blank>)
      scores: list[float] length matches tokens
      types:  list[int]   length matches tokens
      unk_id, bos_id, eos_id: int or None
      blank_id: int (always vocab_size, the appended <blank> position)

    The +1 row for the blank token at the end matches the predictor
    embed table layout in NeMo: prediction.embed.weight has shape
    [vocab+1, hidden] with the extra row used as the "no previous
    token" / start state.
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

    # Append the blank/start row that lives outside the SentencePiece
    # vocab but inside the predictor's embed table. The blank id is
    # always vocab_size — the loader stores it as
    # tokenizer.ggml.blank_token_id.
    tokens.append(blank_piece)
    scores.append(0.0)
    types.append(TOKEN_TYPE_CONTROL)

    blank_id = vocab_size

    def safe_id(method) -> int | None:
        v = method()
        return v if v >= 0 else None

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
# Hparams from config.json
# ---------------------------------------------------------------------------


def _validate_durations(durations: list[int]) -> list[int]:
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
    out of NeMo's config.json. Cross-field invariants (d_model %
    n_heads == 0, win_length <= n_fft, etc.) are validated by the
    loader, not here — the converter is intentionally a thin
    pass-through.

    Frontend defaults: NeMo's preprocessor config carries most fields
    directly, but a few that PLAN.md declares mandatory in
    stt.frontend.* are not in NeMo's config because NeMo defaults
    them inside its preprocessor module. We hard-code the documented
    defaults for the AudioToMelSpectrogramPreprocessor used by every
    Parakeet variant we know about:

      - pre_emphasis: NeMo's FilterbankFeatures defaults
        `preemph=0.97` (features.py:250). Verified against the
        `preemph_2_cast` initializer baked into Parakeet's
        nemo128.onnx preprocessor export. The C++ frontend applies
        the standard one-tap filter `y[n] = x[n] - 0.97*x[n-1]`.
      - f_min: lowfreq defaults to 0 Hz. NeMo's config doesn't carry
        it.
      - f_max: highfreq defaults to sample_rate / 2 (Nyquist). NeMo's
        config doesn't carry it.
      - type: there's no config field for this; we know the
        preprocessor target is AudioToMelSpectrogramPreprocessor so
        we emit "mel".
    """
    enc  = config["encoder"]
    pre  = config["preprocessor"]
    dec  = config["decoder"]
    pred = dec["prednet"]
    joint = config["joint"]
    decoding = config.get("decoding", {})

    # TDT decoding metadata. PLAN.md fixes the GGUF KV name as
    # `stt.parakeet.tdt.durations`; the C++ loader requires it and
    # enforces durations.length == joint.num_extra_outputs. NeMo
    # Parakeet 0.6B v2 and v3 both ship [0, 1, 2, 3, 4]; we read it
    # straight from config rather than hard-coding so the converter
    # is honest about future variants that might choose different
    # values.
    raw_durations = decoding.get("durations")
    if raw_durations is None:
        raise ValueError(
            "config.decoding.durations missing — this converter only "
            "handles TDT models that publish a duration list"
        )
    tdt_durations = _validate_durations(raw_durations)
    if int(joint["num_extra_outputs"]) != len(tdt_durations):
        raise ValueError(
            f"joint.num_extra_outputs ({joint['num_extra_outputs']}) "
            f"must equal len(decoding.durations) ({len(tdt_durations)})"
        )

    # Joint activation. The rnnt.py default is "tanh" but every
    # published Parakeet 0.6B variant ships "relu". We read what's
    # actually in config rather than guessing — the C++ loader's
    # joint activation allow-list will reject anything outside
    # {relu, sigmoid, tanh}.
    joint_activation = str(joint["jointnet"]["activation"]).lower()

    # max_symbols caps how many consecutive zero-duration "stuck"
    # emissions the greedy decoder will accept before forcing a
    # +1 frame advance. Optional KV with a documented default of 10
    # in the C++ loader; we still emit the value from config when
    # present so the converter is a faithful pass-through.
    greedy_cfg = decoding.get("greedy") or {}
    tdt_max_symbols = int(greedy_cfg.get("max_symbols") or 10)

    sample_rate  = int(pre["sample_rate"])
    window_size  = float(pre["window_size"])   # seconds
    window_stride = float(pre["window_stride"])  # seconds
    win_length   = int(round(window_size  * sample_rate))
    hop_length   = int(round(window_stride * sample_rate))

    # Sanity: NeMo's AudioToMelSpectrogramPreprocessor is the only
    # frontend we support today. Surface a clear error if a different
    # _target_ shows up so future variants don't silently produce a
    # GGUF that the C++ frontend would misinterpret.
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
        "enc_use_bias":             bool(enc.get("use_bias", False)),

        # pred_vocab is the embed table row count: vocab_size + 1 for
        # the prepended "no previous token" row.
        "pred_hidden":   int(pred["pred_hidden"]),
        "pred_n_layers": int(pred["pred_rnn_layers"]),
        "pred_vocab":    int(dec["vocab_size"]) + 1,

        "joint_hidden":            int(joint["jointnet"]["joint_hidden"]),
        "joint_num_extra_outputs": int(joint["num_extra_outputs"]),
        "joint_activation":        joint_activation,

        "tdt_durations":           tdt_durations,
        "tdt_max_symbols":         tdt_max_symbols,

        # Full stt.frontend.* block. PLAN.md declares this list as the
        # complete contract — every field the loader expects must
        # appear here.
        "fe_type":         "mel",
        "fe_num_mels":     int(pre["features"]),
        "fe_sample_rate":  sample_rate,
        "fe_n_fft":        int(pre["n_fft"]),
        "fe_win_length":   win_length,
        "fe_hop_length":   hop_length,
        "fe_window":       str(pre["window"]),
        "fe_normalize":    str(pre["normalize"]),
        "fe_dither":       float(pre["dither"]),
        "fe_pre_emphasis": 0.97,                   # NeMo FilterbankFeatures default
        "fe_f_min":        0.0,                    # NeMo default lowfreq
        "fe_f_max":        float(sample_rate) / 2.0,  # NeMo default highfreq (Nyquist)
    }


# ---------------------------------------------------------------------------
# Tensor name + shape mapping
# ---------------------------------------------------------------------------
#
# Each entry maps a NeMo safetensors key template to a transcribe.cpp
# canonical name template + a numpy transform. {i} is the encoder layer
# index; {j} is the predictor LSTM layer index. The transform takes a
# numpy array straight from safetensors and returns the array in the
# layout the loader expects (still numpy; gguf-py reverses the shape
# at write time).
#
# Required ordering rationale: gguf-py writes tensors in the order
# add_tensor() is called, and the loader reads them in arbitrary
# order via name lookup. So the order here is purely for human
# readability and matches the order in arch/parakeet/weights.cpp.


def transpose_conv2d(arr: np.ndarray) -> np.ndarray:
    """MLX [O, kh, kw, I] -> PyTorch OIHW [O, I, kh, kw]."""
    assert arr.ndim == 4, f"expected 4D conv2d, got {arr.shape}"
    return np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))


def transpose_conv1d(arr: np.ndarray) -> np.ndarray:
    """MLX [O, k, I_per_group] -> PyTorch [O, I_per_group, k]."""
    assert arr.ndim == 3, f"expected 3D conv1d, got {arr.shape}"
    return np.ascontiguousarray(np.transpose(arr, (0, 2, 1)))


def passthrough(arr: np.ndarray) -> np.ndarray:
    """No layout change. Linear weights, biases, LayerNorm, BN stats,
    LSTM gate matrices — all of these are already in the layout the
    loader expects."""
    # Force C-contiguous so .tobytes() (called by gguf-py) is stable.
    return np.ascontiguousarray(arr)


# Static rename table for the per-encoder-layer block. Each tuple is
# (nemo_suffix, gguf_suffix, transform). The encoder layer index `i`
# is substituted into both the NeMo and GGUF names.
ENCODER_BLOCK_TABLE: list[tuple[str, str, callable]] = [
    # Macaron FF1.
    ("norm_feed_forward1.weight",       "norm_ff1.weight",        passthrough),
    ("norm_feed_forward1.bias",         "norm_ff1.bias",          passthrough),
    ("feed_forward1.linear1.weight",    "ff1.linear1.weight",     passthrough),
    ("feed_forward1.linear2.weight",    "ff1.linear2.weight",     passthrough),

    # Self-attention with relative position.
    ("norm_self_att.weight",            "norm_attn.weight",       passthrough),
    ("norm_self_att.bias",              "norm_attn.bias",         passthrough),
    ("self_attn.linear_q.weight",       "attn.linear_q.weight",   passthrough),
    ("self_attn.linear_k.weight",       "attn.linear_k.weight",   passthrough),
    ("self_attn.linear_v.weight",       "attn.linear_v.weight",   passthrough),
    ("self_attn.linear_out.weight",     "attn.linear_out.weight", passthrough),
    ("self_attn.linear_pos.weight",     "attn.linear_pos.weight", passthrough),
    ("self_attn.pos_bias_u",            "attn.pos_bias_u",        passthrough),
    ("self_attn.pos_bias_v",            "attn.pos_bias_v",        passthrough),

    # Convolution module.
    ("norm_conv.weight",                "norm_conv.weight",            passthrough),
    ("norm_conv.bias",                  "norm_conv.bias",              passthrough),
    ("conv.pointwise_conv1.weight",     "conv.pointwise1.weight",      transpose_conv1d),
    ("conv.depthwise_conv.weight",      "conv.depthwise.weight",       transpose_conv1d),
    ("conv.pointwise_conv2.weight",     "conv.pointwise2.weight",      transpose_conv1d),
    ("conv.batch_norm.weight",          "conv.bn.weight",              passthrough),
    ("conv.batch_norm.bias",            "conv.bn.bias",                passthrough),
    ("conv.batch_norm.running_mean",    "conv.bn.running_mean",        passthrough),
    ("conv.batch_norm.running_var",     "conv.bn.running_var",         passthrough),

    # Macaron FF2.
    ("norm_feed_forward2.weight",       "norm_ff2.weight",        passthrough),
    ("norm_feed_forward2.bias",         "norm_ff2.bias",          passthrough),
    ("feed_forward2.linear1.weight",    "ff2.linear1.weight",     passthrough),
    ("feed_forward2.linear2.weight",    "ff2.linear2.weight",     passthrough),

    # Final per-block layer norm.
    ("norm_out.weight",                 "norm_out.weight",        passthrough),
    ("norm_out.bias",                   "norm_out.bias",          passthrough),
]


PRE_ENCODE_TABLE: list[tuple[str, str, callable]] = [
    ("encoder.pre_encode.conv.0.weight", "enc.pre_encode.conv.0.weight", transpose_conv2d),
    ("encoder.pre_encode.conv.0.bias",   "enc.pre_encode.conv.0.bias",   passthrough),
    ("encoder.pre_encode.conv.2.weight", "enc.pre_encode.conv.2.weight", transpose_conv2d),
    ("encoder.pre_encode.conv.2.bias",   "enc.pre_encode.conv.2.bias",   passthrough),
    ("encoder.pre_encode.conv.3.weight", "enc.pre_encode.conv.3.weight", transpose_conv2d),
    ("encoder.pre_encode.conv.3.bias",   "enc.pre_encode.conv.3.bias",   passthrough),
    ("encoder.pre_encode.conv.5.weight", "enc.pre_encode.conv.5.weight", transpose_conv2d),
    ("encoder.pre_encode.conv.5.bias",   "enc.pre_encode.conv.5.bias",   passthrough),
    ("encoder.pre_encode.conv.6.weight", "enc.pre_encode.conv.6.weight", transpose_conv2d),
    ("encoder.pre_encode.conv.6.bias",   "enc.pre_encode.conv.6.bias",   passthrough),
    ("encoder.pre_encode.out.weight",    "enc.pre_encode.out.weight",    passthrough),
    ("encoder.pre_encode.out.bias",      "enc.pre_encode.out.bias",      passthrough),
]


JOINT_TABLE: list[tuple[str, str, callable]] = [
    ("joint.enc.weight",          "joint.enc.weight",  passthrough),
    ("joint.enc.bias",            "joint.enc.bias",    passthrough),
    ("joint.pred.weight",         "joint.pred.weight", passthrough),
    ("joint.pred.bias",           "joint.pred.bias",   passthrough),
    ("joint.joint_net.2.weight",  "joint.out.weight",  passthrough),
    ("joint.joint_net.2.bias",    "joint.out.bias",    passthrough),
]


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert(model_dir: Path, out_path: Path, quant: str) -> None:
    if quant not in QUANT_PRESETS:
        raise ValueError(
            f"unknown --quant preset: {quant!r}; "
            f"known presets: {sorted(QUANT_PRESETS)}"
        )
    preset = QUANT_PRESETS[quant]
    print(f"Quant preset: {quant} (linear={preset['linear'].name}, "
          f"conv={preset['conv'].name}, norm={preset['norm'].name})")

    config_path     = model_dir / "config.json"
    tokenizer_path  = model_dir / "tokenizer.model"
    safetensors_path = model_dir / "model.safetensors"

    for p in (config_path, tokenizer_path, safetensors_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file: {p}")

    print(f"Reading config from {config_path}")
    with config_path.open() as f:
        config = json.load(f)

    hp = read_hparams(config)

    raw_vocab_size = hp["pred_vocab"] - 1  # back out the +1 start row
    print(f"Detected raw vocab_size = {raw_vocab_size}")

    if raw_vocab_size not in VARIANT_PROFILES:
        raise ValueError(
            f"unknown parakeet variant: vocab_size={raw_vocab_size}; "
            f"known variants: {sorted(VARIANT_PROFILES)}"
        )
    profile = VARIANT_PROFILES[raw_vocab_size]
    print(f"Variant: {profile['variant']}")

    print(f"Reading tokenizer from {tokenizer_path}")
    tok = extract_tokenizer(tokenizer_path)
    if len(tok["tokens"]) != hp["pred_vocab"]:
        raise ValueError(
            f"tokenizer length mismatch: {len(tok['tokens'])} tokens "
            f"(incl. <blank>) vs hp.pred_vocab={hp['pred_vocab']}"
        )

    print(f"Opening safetensors at {safetensors_path}")
    # safe_open with framework="numpy" gives us np.ndarray for each
    # get_tensor() call. The shapes match the safetensors header.
    with safe_open(str(safetensors_path), framework="numpy") as st:
        st_keys = set(st.keys())

        print(f"Writing GGUF to {out_path}")
        # GGUFWriter takes the architecture as a positional argument
        # which it writes as general.architecture. We want exactly
        # "parakeet" so the loader's find_arch() picks us up.
        writer = GGUFWriter(str(out_path), "parakeet")

        # ----- general.* metadata -----
        writer.add_string("general.basename",   "parakeet-tdt")
        writer.add_string("general.size_label", "0.6B")
        writer.add_string("general.version",    profile["version"])
        # general.file_type is the llama-style "what quant did this
        # converter pick" tag. The C++ loader doesn't read it today
        # (it relies on per-tensor type at the gguf level) but it
        # belongs in the metadata so `gguf-dump` and similar tools
        # can identify the variant without inspecting tensors.
        writer.add_uint32("general.file_type", int(preset["file_type"]))
        # general.languages is the descriptive language list. The
        # loader's read_languages_kv pulls this and feeds it through
        # transcribe_model::set_languages().
        writer.add_array("general.languages", profile["languages"])

        # ----- stt.variant + capability KV -----
        writer.add_string("stt.variant", profile["variant"])
        if profile["lang_detect"]:
            writer.add_bool("stt.capability.lang_detect", True)

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

        writer.add_uint32("stt.parakeet.predictor.hidden",   hp["pred_hidden"])
        writer.add_uint32("stt.parakeet.predictor.n_layers", hp["pred_n_layers"])
        writer.add_uint32("stt.parakeet.predictor.vocab",    hp["pred_vocab"])

        writer.add_uint32("stt.parakeet.joint.hidden",            hp["joint_hidden"])
        writer.add_uint32("stt.parakeet.joint.num_extra_outputs", hp["joint_num_extra_outputs"])
        writer.add_string("stt.parakeet.joint.activation",        hp["joint_activation"])

        # TDT decoding metadata. The durations array is required by the
        # C++ loader (the decoder cannot run without it); max_symbols
        # is optional with a default of 10. gguf-py's add_array infers
        # INT32 element type from a Python `list[int]` (see
        # GGUFValueType.get_type: int -> INT32), which is exactly the
        # type the loader's read_int32_array_kv expects.
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
        # Helper that pulls a tensor by NeMo name, runs the layout
        # transform, looks up the target dtype from the active preset,
        # and adds the (possibly cast / quantized) tensor to the
        # writer. Errors loud and early on missing keys or unexpected
        # source dtype — the converter prefers to fail with a precise
        # diagnostic over emitting an incomplete GGUF that the loader
        # will then reject anyway.
        n_added       = 0
        bucket_counts = {"linear": 0, "conv": 0, "norm": 0}
        # Track on-disk byte totals so the closing log can show the
        # quant savings vs the source fp32 size at a glance.
        bytes_in  = 0
        bytes_out = 0

        def add(nemo_name: str, gguf_name: str, transform) -> None:
            nonlocal n_added, bytes_in, bytes_out
            if nemo_name not in st_keys:
                raise KeyError(
                    f"safetensors missing tensor: {nemo_name!r}"
                )
            arr = st.get_tensor(nemo_name)
            arr = transform(arr)
            # Source must be fp32. Any safetensors accidentally stored
            # at lower precision should surface here, not silently
            # collapse into the target dtype.
            if arr.dtype != np.float32:
                raise ValueError(
                    f"{nemo_name}: expected float32, got {arr.dtype}"
                )
            bucket = classify_tensor(gguf_name)
            target = preset[bucket]
            encoded, raw_dtype = encode_for_gguf(arr, target)
            # Do NOT pass raw_shape: for f32/f16 the encoded ndarray's
            # shape already matches the source, and for quantized
            # tensors gguf-py reads the byte-shape off the encoded
            # buffer and converts it back to the logical shape via
            # quant_shape_from_byte_shape(). Passing the logical
            # shape explicitly would make it try to interpret
            # `(out, in)` as a byte-shape and crash on the divisibility
            # check.
            writer.add_tensor(gguf_name, encoded, raw_dtype=raw_dtype)
            bucket_counts[bucket] += 1
            bytes_in  += int(arr.nbytes)
            bytes_out += int(encoded.nbytes)
            n_added += 1

        # pre_encode (12 tensors)
        for nemo_name, gguf_name, transform in PRE_ENCODE_TABLE:
            add(nemo_name, gguf_name, transform)

        # encoder layers (n_layers * 28 tensors)
        for i in range(hp["enc_n_layers"]):
            for suffix_nemo, suffix_gguf, transform in ENCODER_BLOCK_TABLE:
                add(
                    f"encoder.layers.{i}.{suffix_nemo}",
                    f"enc.blocks.{i}.{suffix_gguf}",
                    transform,
                )

        # predictor: embed + n_lstm_layers * (Wx, Wh, bias) tensors.
        add(
            "decoder.prediction.embed.weight",
            "pred.embed.weight",
            passthrough,
        )
        for i in range(hp["pred_n_layers"]):
            add(
                f"decoder.prediction.dec_rnn.lstm.{i}.Wx",
                f"pred.lstm.{i}.Wx",
                passthrough,
            )
            add(
                f"decoder.prediction.dec_rnn.lstm.{i}.Wh",
                f"pred.lstm.{i}.Wh",
                passthrough,
            )
            add(
                f"decoder.prediction.dec_rnn.lstm.{i}.bias",
                f"pred.lstm.{i}.bias",
                passthrough,
            )

        # joint (6 tensors)
        for nemo_name, gguf_name, transform in JOINT_TABLE:
            add(nemo_name, gguf_name, transform)

        expected = (
            len(PRE_ENCODE_TABLE)
            + hp["enc_n_layers"] * len(ENCODER_BLOCK_TABLE)
            + 1                                                # pred.embed
            + hp["pred_n_layers"] * 3                          # pred.lstm.{Wx,Wh,bias}
            + len(JOINT_TABLE)
        )
        if n_added != expected:
            raise RuntimeError(
                f"tensor count mismatch: added {n_added}, expected {expected}"
            )
        print(
            f"Added {n_added} tensors "
            f"(linear={bucket_counts['linear']}, "
            f"conv={bucket_counts['conv']}, "
            f"norm={bucket_counts['norm']})"
        )
        print(
            f"Tensor data: {bytes_in / (1024 * 1024):.1f} MB in (fp32) -> "
            f"{bytes_out / (1024 * 1024):.1f} MB out "
            f"({100.0 * bytes_out / max(bytes_in, 1):.1f}% of source)"
        )

        # Sanity check: warn if there are safetensors keys we didn't
        # consume. Catches a future converter that misses a tensor
        # NeMo added in a new variant. Not fatal — extra tensors are
        # likely safe to ignore — but loud enough to investigate.
        consumed = set()
        for nemo_name, _, _ in PRE_ENCODE_TABLE:
            consumed.add(nemo_name)
        for i in range(hp["enc_n_layers"]):
            for suffix_nemo, _, _ in ENCODER_BLOCK_TABLE:
                consumed.add(f"encoder.layers.{i}.{suffix_nemo}")
        consumed.add("decoder.prediction.embed.weight")
        for i in range(hp["pred_n_layers"]):
            consumed.add(f"decoder.prediction.dec_rnn.lstm.{i}.Wx")
            consumed.add(f"decoder.prediction.dec_rnn.lstm.{i}.Wh")
            consumed.add(f"decoder.prediction.dec_rnn.lstm.{i}.bias")
        for nemo_name, _, _ in JOINT_TABLE:
            consumed.add(nemo_name)

        unused = sorted(st_keys - consumed)
        if unused:
            print(
                f"WARNING: {len(unused)} safetensors keys were not consumed:",
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
        description="Convert a NeMo Parakeet MLX directory to a GGUF.",
    )
    p.add_argument("model_dir", type=Path,
                   help="Path to a parakeet-tdt-*-mlx directory")
    p.add_argument("out_path", type=Path,
                   help="Output .gguf path")
    p.add_argument(
        "--quant",
        choices=sorted(QUANT_PRESETS),
        default="f32",
        help=(
            "Quantization preset. f32 (default) is the lossless reference "
            "build. f16 casts every linear weight to half precision and "
            "leaves conv kernels and norms / biases at fp32. Additional "
            "presets land as the C++ loader and quantizer support grow."
        ),
    )
    args = p.parse_args(argv[1:])

    if not args.model_dir.is_dir():
        print(f"error: {args.model_dir} is not a directory", file=sys.stderr)
        return 2

    convert(args.model_dir, args.out_path, args.quant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
