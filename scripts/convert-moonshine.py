#!/usr/bin/env python3
"""
convert-moonshine.py - convert a Moonshine HuggingFace directory to a
reference-dtype GGUF that transcribe.cpp's loader will ingest. Preserves
the source dtype (F32 for moonshine-tiny); block quantization (Q8_0,
Q5_K_M, ...) goes through tools/transcribe-quantize later.

Source format:
    A HuggingFace directory (or repo id), e.g. UsefulSensors/moonshine-tiny:

      config.json               Model config (hidden_size, layers, heads, ...)
      generation_config.json    Special-token IDs (bos=1, eos=2, decoder_start=1),
                                max_length=194 (= max_position_embeddings)
      preprocessor_config.json  Wav2Vec2FeatureExtractor parameters
                                (sampling_rate=16000, feature_size=1, no mel
                                filterbank — moonshine consumes raw waveform).
      tokenizer.json            HF-fast tokenizer; SentencePiece-style BPE with
                                byte_fallback. 32000 base + 768 added (<<ST_N>>)
                                = 32768 vocab.
      model.safetensors         F32 weights (160 tensors).

Architecture: encoder-decoder transformer over raw 16 kHz audio.
    encoder
      conv1 (k=127, stride=64, no bias) -> tanh
        + GroupNorm(num_groups=1, channels=embed_dim, eps=1e-5)
      conv2 (k=7,  stride=3) + GELU
      conv3 (k=3,  stride=2) + GELU
      total temporal stride 64*3*2 = 384, no padding
      6x encoder block (pre-LN: self_attn -> MLP)
      final layer norm (no bias)
    decoder
      token embedding (tied with output projection)
      6x decoder block (pre-LN: self_attn -> cross_attn -> MLP)
      final layer norm (no bias)
      tied lm_head -> [vocab=32768] logits
    attention
      partial RoPE 0.9 on encoder + decoder self-attn (cross-attn unrotated)
      attention_bias=False (q/k/v/o have no bias)
      pad_head_dim_to_multiple_of=8 (head_dim=288/8=36 → padded to 40 inside
      attention; weights are stored at d_model=288 — no padding in the GGUF)
    MLPs
      encoder MLP: GELU. fc1 [hidden→inter], fc2 [inter→hidden].
      decoder MLP: SwiGLU. fc1 [hidden→2·inter], split into [x, gate];
        out = silu(gate) * x; fc2 [inter→hidden].

Layout conversions: NONE. Conv1d kernels are already [out, in, k]; all
linears are [out, in]. Both match ggml's expected layout.

Tensor naming (whisper-style):
    Encoder top-level
      enc.conv.0.weight                  [288, 1,   127]   (no bias)
      enc.conv.1.weight / .bias          [576, 288, 7]   / [576]
      enc.conv.2.weight / .bias          [288, 576, 3]   / [288]
      enc.conv.norm.weight / .bias       [288] / [288]   (GroupNorm)
      enc.final_norm.weight              [288]            (LayerNorm, no bias)
    Encoder per-layer (i = 0..enc_n_layers-1)
      enc.blocks.{i}.norm_attn.weight              (no bias)
      enc.blocks.{i}.attn.q.weight                 (no bias; attention_bias=False)
      enc.blocks.{i}.attn.k.weight                 (no bias)
      enc.blocks.{i}.attn.v.weight                 (no bias)
      enc.blocks.{i}.attn.out.weight               (no bias)
      enc.blocks.{i}.norm_ffn.weight               (no bias)
      enc.blocks.{i}.ffn.fc1.weight / .bias        (GELU MLP: hidden→inter)
      enc.blocks.{i}.ffn.fc2.weight / .bias        (inter→hidden)
    Decoder top-level
      dec.token_embd.weight              [vocab=32768, hidden=288]  (tied)
      dec.final_norm.weight              [288]                       (no bias)
    Decoder per-layer
      dec.blocks.{i}.norm_self.weight              (no bias)
      dec.blocks.{i}.self_attn.q.weight            (no bias)
      dec.blocks.{i}.self_attn.k.weight            (no bias)
      dec.blocks.{i}.self_attn.v.weight            (no bias)
      dec.blocks.{i}.self_attn.out.weight          (no bias)
      dec.blocks.{i}.norm_cross.weight             (no bias)
      dec.blocks.{i}.cross_attn.q.weight           (no bias)
      dec.blocks.{i}.cross_attn.k.weight           (no bias)
      dec.blocks.{i}.cross_attn.v.weight           (no bias)
      dec.blocks.{i}.cross_attn.out.weight         (no bias)
      dec.blocks.{i}.norm_ffn.weight               (no bias)
      dec.blocks.{i}.ffn.fc1.weight / .bias        (SwiGLU: hidden→2·inter)
      dec.blocks.{i}.ffn.fc2.weight / .bias        (inter→hidden)

KV emitted:
    general.architecture = "moonshine"
    general.basename     = "moonshine"
    general.size_label   = derived from total params
    general.file_type    = ALL_F32
    general.languages    = inferred from slug suffix (see _infer_languages),
                           e.g. moonshine-tiny-vi → ["vi"], moonshine-tiny → ["en"].
                           Override with --language.

    stt.variant = <slug>                    (e.g. "moonshine-tiny")

    stt.capability.lang_detect = false
    stt.capability.translate   = false
    stt.capability.timestamps  = false

    tokenizer.ggml.model = "bpe"            (SentencePiece-style BPE)
    tokenizer.ggml.pre   = "default"
    tokenizer.ggml.tokens / token_type / merges
    tokenizer.ggml.byte_fallback = true
    tokenizer.ggml.bos / eos / padding / unknown_token_id
    tokenizer.ggml.add_bos_token = true     (decoder_start_token_id=1=<s>)

    stt.moonshine.encoder.n_layers / d_model / n_heads / n_kv_heads /
                         ffn_dim / activation
    stt.moonshine.decoder.n_layers / d_model / n_heads / n_kv_heads /
                         ffn_dim / activation / vocab_size /
                         max_position_embeddings / tie_word_embeddings
    stt.moonshine.decoder_start_token_id
    stt.moonshine.bos_token_id / eos_token_id / pad_token_id

    stt.moonshine.partial_rotary_factor = 0.9
    stt.moonshine.rope_theta            = 10000.0
    stt.moonshine.attention_bias        = false
    stt.moonshine.pad_head_dim_to_multiple_of = 8

    stt.moonshine.conv_stem.channels      = [enc1, enc2, enc3] out channels
    stt.moonshine.conv_stem.kernel_sizes  = [127, 7, 3]
    stt.moonshine.conv_stem.strides       = [64, 3, 2]
    stt.moonshine.conv_stem.groupnorm_num_groups = 1
    stt.moonshine.conv_stem.groupnorm_eps = 1e-5

    stt.frontend.type        = "raw"
    stt.frontend.sample_rate = 16000
    stt.frontend.num_mels    = 1            (Wav2Vec2FeatureExtractor.feature_size:
                                             this is the conv input channel
                                             count, not a mel bin count)

CLI:
    # HF repo id (downloads into $TRANSCRIBE_MODELS_DIR or HF cache)
    uv run --project scripts/envs/moonshine \
      scripts/convert-moonshine.py UsefulSensors/moonshine-tiny

    # Local directory
    uv run --project scripts/envs/moonshine \
      scripts/convert-moonshine.py <model-dir> --repo-id UsefulSensors/moonshine-tiny

Single-file, top-to-bottom — no hidden helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, LlamaFileType
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.hf_source import download_snapshot, looks_like_repo_id  # noqa: E402
from lib.gguf_common import (  # noqa: E402
    gguf_writer,
    TOKEN_TYPE_BYTE,
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN,
    add_general_identity,
    encode_for_gguf,
    gguf_name,
    reference_dtype_for,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Reference dtype
# ---------------------------------------------------------------------------
# Moonshine ships F32 across the published variants (tiny, base). Mirror
# whisper's detector for forward compatibility — if a future variant ships
# F16/BF16 we don't have to touch the converter.


def detect_reference_dtype(safetensors_path: Path) -> tuple[str, LlamaFileType, GGMLQuantizationType]:
    with safe_open(str(safetensors_path), framework="pt") as st:
        dtypes = set()
        for k in st.keys():
            t = st.get_slice(k)
            dtypes.add(str(t.get_dtype()))
    if dtypes == {"F32"}:
        return ("F32", LlamaFileType.ALL_F32, GGMLQuantizationType.F32)
    if dtypes == {"F16"}:
        return ("F16", LlamaFileType.MOSTLY_F16, GGMLQuantizationType.F16)
    if dtypes <= {"F32", "F16"}:
        return ("F32", LlamaFileType.ALL_F32, GGMLQuantizationType.F32)
    raise ValueError(
        f"unsupported safetensors dtype mix: {sorted(dtypes)} in {safetensors_path}"
    )


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------
#
# Moonshine ships only tokenizer.json (no tokenizer_config.json, no
# vocab.json/merges.txt). The model is a SentencePiece-style BPE with
# byte_fallback=True, normalizer Prepend('▁') + Replace(' ', '▁'),
# decoder Replace('▁', ' ') + ByteFallback. Vocab layout:
#
#   id 0           = "<unk>"        UNKNOWN
#   id 1           = "<s>"          CONTROL (bos / decoder_start)
#   id 2           = "</s>"         CONTROL (eos / pad)
#   ids 3..258     = "<0xNN>"       BYTE     (256 byte-fallback tokens)
#   ids 259..31999 = SP pieces      NORMAL
#   ids 32000..32767 = "<<ST_N>>"   CONTROL  (768 added special tokens)
#
# 32000 base + 768 unique added = 32768 = config.vocab_size.


def _is_byte_token(piece: str) -> bool:
    """SentencePiece byte-fallback piece, e.g. '<0x00>' .. '<0xFF>'."""
    if len(piece) != 6:
        return False
    if not (piece.startswith("<0x") and piece.endswith(">")):
        return False
    try:
        int(piece[3:5], 16)
    except ValueError:
        return False
    return True


def extract_tokenizer(model_dir: Path, vocab_size: int) -> dict:
    tokenizer_json = model_dir / "tokenizer.json"
    with tokenizer_json.open(encoding="utf-8") as f:
        tj = json.load(f)

    if tj["model"].get("type") != "BPE":
        raise ValueError(
            f"expected tokenizer.json model.type=BPE, got {tj['model'].get('type')!r}"
        )
    if not tj["model"].get("byte_fallback", False):
        raise ValueError(
            "expected tokenizer.json model.byte_fallback=true; "
            "moonshine ships SentencePiece-style BPE with byte fallback"
        )

    base_vocab: dict[str, int] = tj["model"]["vocab"]
    merges_raw = tj["model"].get("merges", [])

    if merges_raw and isinstance(merges_raw[0], list):
        merges = [f"{a} {b}" for a, b in merges_raw]
    else:
        merges = [str(m) for m in merges_raw]

    added_tokens = tj.get("added_tokens", []) or []

    tok_by_id: dict[int, tuple[str, bool]] = {}
    for tok, tid in base_vocab.items():
        tok_by_id[int(tid)] = (tok, False)
    for entry in added_tokens:
        tid = int(entry["id"])
        tok_by_id[tid] = (entry["content"], bool(entry.get("special", False)))

    max_id = max(tok_by_id.keys())
    if max_id + 1 != vocab_size:
        raise ValueError(
            f"tokenizer max id {max_id} (+1={max_id + 1}) does not match "
            f"config vocab_size={vocab_size}"
        )

    tokens: list[str] = []
    types:  list[int] = []
    for i in range(vocab_size):
        if i not in tok_by_id:
            raise ValueError(f"tokenizer missing id {i}")
        tok, is_special = tok_by_id[i]
        tokens.append(tok)
        if tok == "<unk>":
            types.append(TOKEN_TYPE_UNKNOWN)
        elif is_special:
            types.append(TOKEN_TYPE_CONTROL)
        elif _is_byte_token(tok):
            types.append(TOKEN_TYPE_BYTE)
        else:
            types.append(TOKEN_TYPE_NORMAL)

    content_to_id = {tok: tid for tok, tid in base_vocab.items()}
    for entry in added_tokens:
        content_to_id[entry["content"]] = int(entry["id"])

    def tok_id(content: str) -> int | None:
        return content_to_id.get(content)

    return {
        "tokens":  tokens,
        "types":   types,
        "merges":  merges,
        "unk_id":  tok_id("<unk>"),
        "bos_id":  tok_id("<s>"),
        "eos_id":  tok_id("</s>"),
    }


# ---------------------------------------------------------------------------
# Hparams from config.json + generation_config.json + preprocessor_config.json
# ---------------------------------------------------------------------------


def read_hparams(config: dict, gen_config: dict, preproc: dict) -> dict:
    hidden_size           = int(config["hidden_size"])
    intermediate_size     = int(config["intermediate_size"])
    enc_layers            = int(config["encoder_num_hidden_layers"])
    dec_layers            = int(config["decoder_num_hidden_layers"])
    enc_heads             = int(config["encoder_num_attention_heads"])
    dec_heads             = int(config["decoder_num_attention_heads"])
    enc_kv_heads          = int(config.get("encoder_num_key_value_heads", enc_heads))
    dec_kv_heads          = int(config.get("decoder_num_key_value_heads", dec_heads))
    enc_act               = str(config["encoder_hidden_act"]).lower()
    dec_act               = str(config["decoder_hidden_act"]).lower()
    max_position_embeddings = int(config["max_position_embeddings"])
    vocab_size            = int(config["vocab_size"])
    partial_rotary_factor = float(config.get("partial_rotary_factor", 1.0))
    rope_theta            = float(config.get("rope_theta", 10000.0))
    attention_bias        = bool(config.get("attention_bias", False))
    pad_head_dim          = config.get("pad_head_dim_to_multiple_of", None)
    pad_head_dim          = int(pad_head_dim) if pad_head_dim is not None else 0
    tie_word_embeddings   = bool(config.get("tie_word_embeddings", True))

    bos_id           = int(gen_config.get("bos_token_id", 1))
    eos_id           = int(gen_config.get("eos_token_id", 2))
    pad_id           = int(gen_config.get("pad_token_id", eos_id))
    decoder_start_id = int(gen_config.get("decoder_start_token_id", bos_id))

    sample_rate  = int(preproc.get("sampling_rate", 16000))
    feature_size = int(preproc.get("feature_size", 1))

    return {
        "hidden_size":             hidden_size,
        "intermediate_size":       intermediate_size,
        "enc_n_layers":            enc_layers,
        "dec_n_layers":            dec_layers,
        "enc_n_heads":             enc_heads,
        "dec_n_heads":             dec_heads,
        "enc_n_kv_heads":          enc_kv_heads,
        "dec_n_kv_heads":          dec_kv_heads,
        "enc_activation":          enc_act,
        "dec_activation":          dec_act,
        "max_position_embeddings": max_position_embeddings,
        "vocab_size":              vocab_size,
        "partial_rotary_factor":   partial_rotary_factor,
        "rope_theta":              rope_theta,
        "attention_bias":          attention_bias,
        "pad_head_dim":            pad_head_dim,
        "tie_word_embeddings":     tie_word_embeddings,

        "bos_id":                  bos_id,
        "eos_id":                  eos_id,
        "pad_id":                  pad_id,
        "decoder_start_id":        decoder_start_id,

        "fe_type":         "raw",
        "fe_sample_rate":  sample_rate,
        "fe_feature_size": feature_size,
    }


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------


def passthrough(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr)


# Encoder top-level: 3-conv stem + GroupNorm (after conv1 + tanh) + final LN.
ENCODER_TOP_TABLE: list[tuple[str, str]] = [
    ("model.encoder.conv1.weight",     "enc.conv.0.weight"),   # no bias
    ("model.encoder.conv2.weight",     "enc.conv.1.weight"),
    ("model.encoder.conv2.bias",       "enc.conv.1.bias"),
    ("model.encoder.conv3.weight",     "enc.conv.2.weight"),
    ("model.encoder.conv3.bias",       "enc.conv.2.bias"),
    ("model.encoder.groupnorm.weight", "enc.conv.norm.weight"),
    ("model.encoder.groupnorm.bias",   "enc.conv.norm.bias"),
    ("model.encoder.layer_norm.weight", "enc.final_norm.weight"),  # no bias
]


# Encoder block: pre-LN(self_attn) -> pre-LN(MLP). attention_bias=False so
# all four projections have weight only. MLP fc1/fc2 have default Linear
# bias=True.
ENCODER_BLOCK_TABLE: list[tuple[str, str]] = [
    ("input_layernorm.weight",            "norm_attn.weight"),
    ("self_attn.q_proj.weight",           "attn.q.weight"),
    ("self_attn.k_proj.weight",           "attn.k.weight"),
    ("self_attn.v_proj.weight",           "attn.v.weight"),
    ("self_attn.o_proj.weight",           "attn.out.weight"),
    ("post_attention_layernorm.weight",   "norm_ffn.weight"),
    ("mlp.fc1.weight",                    "ffn.fc1.weight"),
    ("mlp.fc1.bias",                      "ffn.fc1.bias"),
    ("mlp.fc2.weight",                    "ffn.fc2.weight"),
    ("mlp.fc2.bias",                      "ffn.fc2.bias"),
]


# Decoder top-level: tied token_embd + final LN. lm_head is tied to
# embed_tokens (no proj_out tensor in the safetensors).
DECODER_TOP_TABLE: list[tuple[str, str]] = [
    ("model.decoder.embed_tokens.weight", "dec.token_embd.weight"),
    ("model.decoder.norm.weight",         "dec.final_norm.weight"),  # no bias
]


# Decoder block: pre-LN(self_attn) -> pre-LN(cross_attn) -> pre-LN(MLP).
# Mapping HF layer-norm names to whisper-style suffixes:
#   input_layernorm           = norm_self   (before self-attn)
#   post_attention_layernorm  = norm_cross  (between self-attn and cross-attn)
#   final_layernorm           = norm_ffn    (between cross-attn and MLP)
DECODER_BLOCK_TABLE: list[tuple[str, str]] = [
    ("input_layernorm.weight",            "norm_self.weight"),
    ("self_attn.q_proj.weight",           "self_attn.q.weight"),
    ("self_attn.k_proj.weight",           "self_attn.k.weight"),
    ("self_attn.v_proj.weight",           "self_attn.v.weight"),
    ("self_attn.o_proj.weight",           "self_attn.out.weight"),
    ("post_attention_layernorm.weight",   "norm_cross.weight"),
    ("encoder_attn.q_proj.weight",        "cross_attn.q.weight"),
    ("encoder_attn.k_proj.weight",        "cross_attn.k.weight"),
    ("encoder_attn.v_proj.weight",        "cross_attn.v.weight"),
    ("encoder_attn.o_proj.weight",        "cross_attn.out.weight"),
    ("final_layernorm.weight",            "norm_ffn.weight"),
    ("mlp.fc1.weight",                    "ffn.fc1.weight"),
    ("mlp.fc1.bias",                      "ffn.fc1.bias"),
    ("mlp.fc2.weight",                    "ffn.fc2.weight"),
    ("mlp.fc2.bias",                      "ffn.fc2.bias"),
]


# ---------------------------------------------------------------------------
# Size label
# ---------------------------------------------------------------------------


def compute_size_label(total_params: int) -> str:
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.0f}M"
    return f"{total_params / 1_000:.0f}K"


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


_KNOWN_SIZE_SUFFIXES = {"tiny", "base", "small", "medium", "large"}


def _infer_languages(variant: str) -> list[str]:
    """Derive general.languages from the variant slug.

    UsefulSensors's published variants follow `moonshine[-streaming]-<size>[-<lang>]`:
        moonshine-tiny             → ["en"]   (no language suffix)
        moonshine-base             → ["en"]
        moonshine-streaming-tiny   → ["en"]
        moonshine-tiny-vi          → ["vi"]
        moonshine-base-zh          → ["zh"]

    A trailing token shorter than four characters that is not a known size
    keyword is treated as a BCP-47 language code. Anything else defaults
    to English. Callers can override via --language for variants whose
    slug doesn't fit this convention.
    """
    parts = variant.lower().split("-")
    if not parts:
        return ["en"]
    tail = parts[-1]
    if tail in _KNOWN_SIZE_SUFFIXES or not tail.isalpha() or len(tail) > 3:
        return ["en"]
    return [tail]


def convert(model_dir: Path, out_path: Path, variant: str,
            languages: list[str] | None = None,
            repo_id: str | None = None) -> None:
    config_path     = model_dir / "config.json"
    gen_config_path = model_dir / "generation_config.json"
    preproc_path    = model_dir / "preprocessor_config.json"
    safetensors_path = model_dir / "model.safetensors"

    for p in (config_path, gen_config_path, preproc_path, safetensors_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file: {p}")

    REFERENCE_DTYPE_LABEL, REFERENCE_FILE_TYPE, REFERENCE_GGML_TYPE = \
        detect_reference_dtype(safetensors_path)
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    if languages is None:
        languages = _infer_languages(variant)
    print(f"Languages: {languages}")

    with config_path.open() as f:
        config = json.load(f)
    with gen_config_path.open() as f:
        gen_config = json.load(f)
    with preproc_path.open() as f:
        preproc = json.load(f)

    hp = read_hparams(config, gen_config, preproc)
    print(f"Encoder: {hp['enc_n_layers']} layers, hidden={hp['hidden_size']}, "
          f"heads={hp['enc_n_heads']}, ffn={hp['intermediate_size']}, "
          f"act={hp['enc_activation']}")
    print(f"Decoder: {hp['dec_n_layers']} layers, hidden={hp['hidden_size']}, "
          f"heads={hp['dec_n_heads']}, ffn={hp['intermediate_size']}, "
          f"act={hp['dec_activation']}")
    print(f"Vocab: {hp['vocab_size']}; max_position_embeddings={hp['max_position_embeddings']}; "
          f"partial_rotary={hp['partial_rotary_factor']}, theta={hp['rope_theta']}")
    print(f"Variant: {variant}")

    print(f"Reading tokenizer from {model_dir}")
    tok = extract_tokenizer(model_dir, hp["vocab_size"])

    print(f"Opening safetensors at {safetensors_path}")
    with safe_open(str(safetensors_path), framework="pt") as st:
        st_keys = set(st.keys())

        total = 0
        for k in st_keys:
            total += st.get_tensor(k).numel()
        size_label = compute_size_label(total)
        print(f"Total params: {total:,} -> size_label={size_label}")

        # Conv-stem channel layout, recovered from the actual conv1/2/3
        # output channels rather than hard-coded so future variants don't
        # require touching the converter.
        c1_out = int(st.get_slice("model.encoder.conv1.weight").get_shape()[0])
        c2_out = int(st.get_slice("model.encoder.conv2.weight").get_shape()[0])
        c3_out = int(st.get_slice("model.encoder.conv3.weight").get_shape()[0])
        conv_channels = [c1_out, c2_out, c3_out]

        print(f"Writing GGUF to {out_path}")
        writer = gguf_writer(str(out_path), "moonshine")

        # ---- general.* ----
        _DISPLAY_NAMES = {
            "moonshine-tiny": "Moonshine Tiny",
            "moonshine-base": "Moonshine Base",
        }
        if variant not in _DISPLAY_NAMES:
            raise ValueError(f"unknown moonshine variant slug: {variant!r}")
        add_general_identity(
            writer,
            name=_DISPLAY_NAMES[variant],
            basename="moonshine",
            size_label=size_label,
            file_type=int(REFERENCE_FILE_TYPE),
            languages=languages,
            author="Useful Sensors",
            organization="UsefulSensors",
            license="mit",
            license_name="MIT License",
            license_link="https://opensource.org/license/mit",
            repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
        )

        # ---- stt.variant ----
        writer.add_string("stt.variant", variant)

        # ---- stt.capability.* ----
        # English-only; no language detection, no translation, no
        # timestamps. Hallucination throttling lives in the runtime, not
        # in capability flags.
        writer.add_bool("stt.capability.lang_detect", False)
        writer.add_bool("stt.capability.translate",   False)
        writer.add_bool("stt.capability.timestamps",  False)

        # ---- tokenizer.ggml.* (SentencePiece-style BPE with byte fallback) ----
        # The tag is "bpe" (matching parakeet/cohere) — transcribe.cpp's
        # tokenizer recognizes "unigram"/"bpe" for the SentencePiece-style
        # decode path and "gpt2" for byte-level BPE. (llama.cpp's
        # equivalent tag is "llama", but our loader has no such alias.)
        # Pre is "default": the normalizer chain (Prepend ▁ + Replace ' ' → ▁)
        # is applied at encode time, not via a regex pretokenizer.
        writer.add_string("tokenizer.ggml.model", "bpe")
        writer.add_string("tokenizer.ggml.pre",   "default")
        writer.add_array("tokenizer.ggml.tokens",     tok["tokens"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        writer.add_array("tokenizer.ggml.merges",     tok["merges"])
        writer.add_bool ("tokenizer.ggml.byte_fallback", True)
        if tok["unk_id"] is not None:
            writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
        if tok["bos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
        if tok["eos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
        # pad_token_id == eos_token_id (= 2) per generation_config.
        writer.add_uint32("tokenizer.ggml.padding_token_id", hp["pad_id"])
        # decoder_start_token_id is <s>=1, so a fresh decode prepends bos.
        writer.add_bool("tokenizer.ggml.add_bos_token", True)

        # ---- stt.moonshine.encoder.* ----
        writer.add_uint32("stt.moonshine.encoder.n_layers",   hp["enc_n_layers"])
        writer.add_uint32("stt.moonshine.encoder.d_model",    hp["hidden_size"])
        writer.add_uint32("stt.moonshine.encoder.n_heads",    hp["enc_n_heads"])
        writer.add_uint32("stt.moonshine.encoder.n_kv_heads", hp["enc_n_kv_heads"])
        writer.add_uint32("stt.moonshine.encoder.ffn_dim",    hp["intermediate_size"])
        writer.add_string("stt.moonshine.encoder.activation", hp["enc_activation"])

        # ---- stt.moonshine.decoder.* ----
        writer.add_uint32("stt.moonshine.decoder.n_layers",                hp["dec_n_layers"])
        writer.add_uint32("stt.moonshine.decoder.d_model",                 hp["hidden_size"])
        writer.add_uint32("stt.moonshine.decoder.n_heads",                 hp["dec_n_heads"])
        writer.add_uint32("stt.moonshine.decoder.n_kv_heads",              hp["dec_n_kv_heads"])
        writer.add_uint32("stt.moonshine.decoder.ffn_dim",                 hp["intermediate_size"])
        writer.add_string("stt.moonshine.decoder.activation",              hp["dec_activation"])
        writer.add_uint32("stt.moonshine.decoder.vocab_size",              hp["vocab_size"])
        writer.add_uint32("stt.moonshine.decoder.max_position_embeddings", hp["max_position_embeddings"])
        writer.add_bool  ("stt.moonshine.decoder.tie_word_embeddings",     hp["tie_word_embeddings"])

        # ---- moonshine attention / RoPE ----
        writer.add_float32("stt.moonshine.partial_rotary_factor", hp["partial_rotary_factor"])
        writer.add_float32("stt.moonshine.rope_theta",            hp["rope_theta"])
        writer.add_bool   ("stt.moonshine.attention_bias",        hp["attention_bias"])
        writer.add_uint32 ("stt.moonshine.pad_head_dim_to_multiple_of", hp["pad_head_dim"])

        # ---- moonshine conv stem ----
        writer.add_array  ("stt.moonshine.conv_stem.channels",     conv_channels)
        writer.add_array  ("stt.moonshine.conv_stem.kernel_sizes", [127, 7, 3])
        writer.add_array  ("stt.moonshine.conv_stem.strides",      [64, 3, 2])
        writer.add_uint32 ("stt.moonshine.conv_stem.groupnorm_num_groups", 1)
        writer.add_float32("stt.moonshine.conv_stem.groupnorm_eps",        1e-5)

        # ---- moonshine prompt / decoding ----
        writer.add_uint32("stt.moonshine.decoder_start_token_id", hp["decoder_start_id"])
        writer.add_uint32("stt.moonshine.bos_token_id",           hp["bos_id"])
        writer.add_uint32("stt.moonshine.eos_token_id",           hp["eos_id"])
        writer.add_uint32("stt.moonshine.pad_token_id",           hp["pad_id"])

        # ---- stt.frontend.* (Wav2Vec2FeatureExtractor — raw waveform) ----
        # No mel / no STFT / no normalization — the conv stem is the
        # frontend. We still emit num_mels=feature_size (=1) so the GGUF
        # carries the conv input channel count, matching how preflight's
        # frontend cross-check reads feature_size.
        writer.add_string ("stt.frontend.type",        hp["fe_type"])
        writer.add_uint32 ("stt.frontend.sample_rate", hp["fe_sample_rate"])
        writer.add_uint32 ("stt.frontend.num_mels",    hp["fe_feature_size"])

        # ---- tensors ----
        n_added   = 0
        bytes_in  = 0
        bytes_out = 0

        def add(src_name: str, dst_name: str, transform=passthrough) -> None:
            nonlocal n_added, bytes_in, bytes_out
            if src_name not in st_keys:
                raise KeyError(f"safetensors missing tensor: {src_name!r}")
            t = st.get_tensor(src_name)
            if t.dtype not in (torch.float32, torch.float16):
                raise ValueError(
                    f"{src_name}: expected float32 or float16, got {t.dtype}"
                )
            arr = transform(t.float().numpy())
            if arr.dtype != np.float32:
                raise ValueError(
                    f"{src_name}: expected float32 after transform, got {arr.dtype}"
                )
            target_type = reference_dtype_for(dst_name, REFERENCE_GGML_TYPE)
            encoded, raw_dtype = encode_for_gguf(arr, target_type)
            writer.add_tensor(dst_name, encoded, raw_dtype=raw_dtype)
            bytes_in  += int(arr.nbytes)
            bytes_out += int(encoded.nbytes)
            n_added += 1

        # Encoder top-level (conv stem + groupnorm + final LN)
        for src, dst in ENCODER_TOP_TABLE:
            add(src, dst)
        # Encoder layers
        for i in range(hp["enc_n_layers"]):
            for suffix_src, suffix_dst in ENCODER_BLOCK_TABLE:
                add(
                    f"model.encoder.layers.{i}.{suffix_src}",
                    f"enc.blocks.{i}.{suffix_dst}",
                )

        # Decoder top-level (tied embed + final LN)
        for src, dst in DECODER_TOP_TABLE:
            add(src, dst)
        # Decoder layers
        for i in range(hp["dec_n_layers"]):
            for suffix_src, suffix_dst in DECODER_BLOCK_TABLE:
                add(
                    f"model.decoder.layers.{i}.{suffix_src}",
                    f"dec.blocks.{i}.{suffix_dst}",
                )

        expected = (
            len(ENCODER_TOP_TABLE)
            + hp["enc_n_layers"] * len(ENCODER_BLOCK_TABLE)
            + len(DECODER_TOP_TABLE)
            + hp["dec_n_layers"] * len(DECODER_BLOCK_TABLE)
        )
        if n_added != expected:
            raise RuntimeError(
                f"tensor count mismatch: added {n_added}, expected {expected}"
            )
        print(
            f"Added {n_added} tensors "
            f"({bytes_in / (1024 * 1024):.1f} MB fp32 -> "
            f"{bytes_out / (1024 * 1024):.1f} MB on disk)"
        )

        consumed: set[str] = set()
        for src, _ in ENCODER_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, _ in ENCODER_BLOCK_TABLE:
                consumed.add(f"model.encoder.layers.{i}.{suffix_src}")
        for src, _ in DECODER_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["dec_n_layers"]):
            for suffix_src, _ in DECODER_BLOCK_TABLE:
                consumed.add(f"model.decoder.layers.{i}.{suffix_src}")
        unused = sorted(st_keys - consumed)
        if unused:
            print(f"WARNING: {len(unused)} safetensors keys not consumed:",
                  file=sys.stderr)
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
        description="Convert a Moonshine checkpoint to a reference-dtype GGUF.",
    )
    p.add_argument("model", type=str,
                   help="HF repo id (e.g. UsefulSensors/moonshine-tiny) or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug when "
                        "converting from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch / tag / commit SHA) to pin the "
                        "download to. Recommended for reproducibility.")
    p.add_argument("--variant", type=str, default=None,
                   help="stt.variant string (default: derived from slug)")
    p.add_argument("--language", action="append", default=None,
                   help="BCP-47 code(s) to write into general.languages. "
                        "Repeat for multiple. Default: inferred from the "
                        "variant slug (e.g. moonshine-tiny-vi → ['vi']; "
                        "moonshine-tiny → ['en']).")
    args = p.parse_args(argv[1:])

    if looks_like_repo_id(args.model):
        repo_id = args.repo_id or args.model
        model_dir = download_snapshot(args.model, args.revision)
    else:
        model_dir = Path(args.model)
        if not model_dir.is_dir():
            print(f"error: {model_dir} is not a directory and not an HF repo id",
                  file=sys.stderr)
            return 2
        repo_id = args.repo_id

    out_path = args.out_path
    if out_path is None:
        if not repo_id:
            print("error: provide out_path, --repo-id, or pass an HF repo id as model",
                  file=sys.stderr)
            return 2
        slug = slug_from_repo_id(repo_id)
        ref_label, _, _ = detect_reference_dtype(model_dir / "model.safetensors")
        out_path = REPO_ROOT / "models" / slug / gguf_name(slug, ref_label)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    variant = args.variant
    if variant is None:
        if repo_id:
            variant = slug_from_repo_id(repo_id).lower()
        else:
            stem = out_path.stem
            known_quants = {"bf16", "f32", "f16", "q8_0", "q5_k_m",
                            "q4_k_m", "q6_k", "q3_k_m", "q2_k"}
            lowered = stem.lower()
            stripped = lowered
            for q in known_quants:
                suffix = "-" + q
                if lowered.endswith(suffix):
                    stripped = lowered[: -len(suffix)]
                    break
            variant = stripped

    convert(model_dir, out_path, variant, languages=args.language, repo_id=repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
