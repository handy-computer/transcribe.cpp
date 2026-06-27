#!/usr/bin/env python3
"""
convert-moonshine_streaming.py - convert a Moonshine Streaming HuggingFace
directory (or HF repo id) to a reference-dtype GGUF that transcribe.cpp's
loader will ingest. Preserves source dtype (F32 across published variants:
tiny, small, medium); block quantization (Q8_0, Q5_K_M, ...) is a Stage 5
step via tools/transcribe-quantize.

Differences vs moonshine (non-streaming):

  - Architecture string in GGUF:  general.architecture = "moonshine_streaming"
  - KV namespace:                 stt.moonshine_streaming.*
  - Frontend: NO 3-conv stem on raw PCM. Instead a streaming embedder:
      cmvn (parameter-free frame-level mean/RMS centering) ->
      asinh-compression with a learnable log_k scalar ->
      Linear(frame_len -> hidden, bias=False) + SiLU ->
      CausalConv1d(hidden, 2*hidden, k=5, s=2, bias=True) + SiLU ->
      CausalConv1d(2*hidden, hidden, k=5, s=2, bias=True)
    Frame size = round(sample_rate * frame_ms / 1000) = 80 at 16 kHz / 5 ms.
  - Encoder layer norms are MoonshineStreamingLayerNorm with unit_offset:
      effective gain = gamma + 1.0
    The safetensors only stores `gamma` (init at 0.0). We FOLD the +1.0
    into the GGUF tensor so the C++ port can apply a vanilla
    "LayerNorm(no affine) * scale" pattern.
  - Encoder is "ergodic" — no positional embeddings on encoder self-attn.
    Sliding-window attention masks are per-layer; we emit
    `stt.moonshine_streaming.encoder.sliding_windows` as a flattened
    [L0, R0, L1, R1, ...] array.
  - Decoder layer norms use nn.LayerNorm(bias=False), parameter is `.weight`.
  - Adapter (between encoder output and decoder cross-attn):
      adapter.pos_emb.weight = decoder.pos_emb.weight  [max_pos, enc_hidden]
      adapter.proj.weight    = decoder.proj.weight     when enc_hidden != dec_hidden
                              (Identity / absent when they match — e.g. tiny)
  - tie_word_embeddings = false: the safetensors carry a separate
    `proj_out.weight`. We map it to `dec.lm_head.weight`.
  - pad_token_id = 0 (vs moonshine's 2).
  - Tokenizer is structurally identical to moonshine's: SentencePiece-style
    BPE with byte_fallback, vocab 32768 (32000 base + 768 <<ST_N>>).

CLI:
    # HF repo id (downloads into $TRANSCRIBE_MODELS_DIR or HF cache)
    uv run --project scripts/envs/moonshine_streaming \
      scripts/convert-moonshine_streaming.py \
      UsefulSensors/moonshine-streaming-tiny

    # Local directory (slug derived from --repo-id)
    uv run --project scripts/envs/moonshine_streaming \
      scripts/convert-moonshine_streaming.py <model-dir> \
      --repo-id UsefulSensors/moonshine-streaming-tiny
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType
from huggingface_hub import snapshot_download
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
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


def detect_reference_dtype(safetensors_path: Path) -> tuple[str, LlamaFileType, GGMLQuantizationType]:
    with safe_open(str(safetensors_path), framework="pt") as st:
        dtypes = set()
        for k in st.keys():
            dtypes.add(str(st.get_slice(k).get_dtype()))
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
# Tokenizer extraction (identical layout to moonshine; vocab content differs
# but the structure is the same — SentencePiece-style BPE with byte fallback,
# vocab 32768).
# ---------------------------------------------------------------------------


def _is_byte_token(piece: str) -> bool:
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
            "moonshine_streaming ships SentencePiece-style BPE with byte fallback"
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
    types: list[int] = []
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
        "tokens": tokens,
        "types": types,
        "merges": merges,
        "unk_id": tok_id("<unk>"),
        "bos_id": tok_id("<s>"),
        "eos_id": tok_id("</s>"),
    }


# ---------------------------------------------------------------------------
# Hparams from config.json + generation_config.json + preprocessor_config.json
# ---------------------------------------------------------------------------


def read_hparams(config: dict, gen_config: dict, preproc: dict) -> dict:
    enc_cfg = config["encoder_config"]

    # Decoder hparams sit at the top of config.json.
    dec_hidden = int(config["hidden_size"])
    dec_intermediate = int(config["intermediate_size"])
    dec_layers = int(config["num_hidden_layers"])
    dec_heads = int(config["num_attention_heads"])
    dec_kv_heads = int(config.get("num_key_value_heads", dec_heads))
    dec_act = str(config["hidden_act"]).lower()
    enc_hidden_top = int(config["encoder_hidden_size"])
    head_dim = int(config["head_dim"])
    max_position_embeddings = int(config["max_position_embeddings"])
    vocab_size = int(config["vocab_size"])

    # rope_parameters is a dict (rope_type, rope_theta, partial_rotary_factor)
    rope = config.get("rope_parameters", {})
    partial_rotary_factor = float(rope.get("partial_rotary_factor", 1.0))
    rope_theta = float(rope.get("rope_theta", 10000.0))

    attention_bias = bool(config.get("attention_bias", False))
    pad_head_dim = config.get("pad_head_dim_to_multiple_of", None)
    pad_head_dim = int(pad_head_dim) if pad_head_dim is not None else 0
    tie_word_embeddings = bool(config.get("tie_word_embeddings", False))

    # Encoder hparams sit in encoder_config.
    enc_hidden = int(enc_cfg["hidden_size"])
    enc_intermediate = int(enc_cfg["intermediate_size"])
    enc_layers = int(enc_cfg["num_hidden_layers"])
    enc_heads = int(enc_cfg["num_attention_heads"])
    enc_kv_heads = int(enc_cfg.get("num_key_value_heads", enc_heads))
    enc_head_dim = int(enc_cfg.get("head_dim", enc_hidden // enc_heads))
    enc_act = str(enc_cfg["hidden_act"]).lower()
    sample_rate_enc = int(enc_cfg["sample_rate"])
    frame_ms = float(enc_cfg["frame_ms"])
    sliding_windows = enc_cfg["sliding_windows"]  # list of [L, R] pairs
    if len(sliding_windows) != enc_layers:
        raise ValueError(
            f"sliding_windows length ({len(sliding_windows)}) does not match "
            f"encoder_num_hidden_layers ({enc_layers})"
        )
    sliding_windows_flat: list[int] = []
    for w in sliding_windows:
        if len(w) != 2:
            raise ValueError(f"sliding_windows entry not (L,R): {w}")
        sliding_windows_flat.extend(int(x) for x in w)
    frame_len = int(round(sample_rate_enc * frame_ms / 1000.0))

    if enc_hidden != enc_hidden_top:
        raise ValueError(
            f"top-level encoder_hidden_size ({enc_hidden_top}) does not match "
            f"encoder_config.hidden_size ({enc_hidden})"
        )

    bos_id = int(config.get("bos_token_id", gen_config.get("bos_token_id", 1)))
    eos_id = int(config.get("eos_token_id", gen_config.get("eos_token_id", 2)))
    pad_id = int(config.get("pad_token_id", gen_config.get("pad_token_id", 0)))
    decoder_start_id = int(
        config.get("decoder_start_token_id", gen_config.get("decoder_start_token_id", bos_id))
    )

    # Preprocessor frontend: Wav2Vec2FeatureExtractor wrapped by
    # MoonshineStreamingProcessor. sample_rate must agree with encoder_config.
    sample_rate = int(preproc.get("sampling_rate", sample_rate_enc))
    feature_size = int(preproc.get("feature_size", 1))
    if sample_rate != sample_rate_enc:
        raise ValueError(
            f"preprocessor sample_rate ({sample_rate}) != encoder_config sample_rate "
            f"({sample_rate_enc})"
        )

    return {
        "enc_hidden":             enc_hidden,
        "enc_intermediate":       enc_intermediate,
        "enc_n_layers":           enc_layers,
        "enc_n_heads":            enc_heads,
        "enc_n_kv_heads":         enc_kv_heads,
        "enc_head_dim":           enc_head_dim,
        "enc_activation":         enc_act,
        "enc_frame_ms":           frame_ms,
        "enc_frame_len":          frame_len,
        "enc_sliding_windows":    sliding_windows_flat,
        "enc_sample_rate":        sample_rate_enc,

        "dec_hidden":             dec_hidden,
        "dec_intermediate":       dec_intermediate,
        "dec_n_layers":           dec_layers,
        "dec_n_heads":            dec_heads,
        "dec_n_kv_heads":         dec_kv_heads,
        "dec_head_dim":           head_dim,
        "dec_activation":         dec_act,

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


def add_unit_offset(arr: np.ndarray) -> np.ndarray:
    """MoonshineStreamingLayerNorm folds a +1.0 offset into the gamma
    parameter at runtime: effective_gain = gamma + 1.0. Pre-fold it here
    so the C++ port treats this like a regular per-channel scale."""
    return np.ascontiguousarray((arr.astype(np.float32) + 1.0))


def reshape_scalar(arr: np.ndarray) -> np.ndarray:
    """0-d safetensors scalars (e.g. comp.log_k) -> 1-element 1-D tensor."""
    a = np.asarray(arr, dtype=np.float32)
    return np.ascontiguousarray(a.reshape(1))


# Encoder embedder + final norm. The SiLU activations between modules are
# implicit in the C++ port — we only ship the parameters.
ENCODER_EMBEDDER_TABLE: list[tuple[str, str, callable]] = [
    ("model.encoder.embedder.comp.log_k",     "enc.embedder.comp.log_k",      reshape_scalar),
    ("model.encoder.embedder.linear.weight",  "enc.embedder.linear.weight",   passthrough),
    ("model.encoder.embedder.conv1.weight",   "enc.embedder.conv1.weight",    passthrough),
    ("model.encoder.embedder.conv1.bias",     "enc.embedder.conv1.bias",      passthrough),
    ("model.encoder.embedder.conv2.weight",   "enc.embedder.conv2.weight",    passthrough),
    ("model.encoder.embedder.conv2.bias",     "enc.embedder.conv2.bias",      passthrough),
    # final_norm uses MoonshineStreamingLayerNorm (gamma + 1.0).
    ("model.encoder.final_norm.gamma",        "enc.final_norm.weight",        add_unit_offset),
]


# Encoder block: pre-norm self-attn (sliding-window, no RoPE) -> pre-norm
# MLP (GELU). Both layer norms use unit_offset; pre-fold +1.0.
ENCODER_BLOCK_TABLE: list[tuple[str, str, callable]] = [
    ("input_layernorm.gamma",          "norm_attn.weight",  add_unit_offset),
    ("self_attn.q_proj.weight",        "attn.q.weight",     passthrough),
    ("self_attn.k_proj.weight",        "attn.k.weight",     passthrough),
    ("self_attn.v_proj.weight",        "attn.v.weight",     passthrough),
    ("self_attn.o_proj.weight",        "attn.out.weight",   passthrough),
    ("post_attention_layernorm.gamma", "norm_ffn.weight",   add_unit_offset),
    ("mlp.fc1.weight",                 "ffn.fc1.weight",    passthrough),
    ("mlp.fc1.bias",                   "ffn.fc1.bias",      passthrough),
    ("mlp.fc2.weight",                 "ffn.fc2.weight",    passthrough),
    ("mlp.fc2.bias",                   "ffn.fc2.bias",      passthrough),
]


# Decoder top-level. pos_emb belongs conceptually to the adapter; we lift
# it out of decoder.* into adapter.pos_emb.weight. proj.weight (when
# enc_hidden != dec_hidden) likewise.
DECODER_TOP_TABLE: list[tuple[str, str, callable]] = [
    ("model.decoder.embed_tokens.weight", "dec.token_embd.weight", passthrough),
    ("model.decoder.norm.weight",         "dec.final_norm.weight", passthrough),
    # NOT tied — proj_out is a distinct weight.
    ("proj_out.weight",                   "dec.lm_head.weight",    passthrough),
]


# Decoder block: pre-norm self-attn (causal, partial RoPE) -> pre-norm
# cross-attn (no RoPE) -> pre-norm SwiGLU MLP. All layer norms use
# nn.LayerNorm(bias=False) — parameter is `.weight`, no offset trick.
DECODER_BLOCK_TABLE: list[tuple[str, str, callable]] = [
    ("input_layernorm.weight",          "norm_self.weight",        passthrough),
    ("self_attn.q_proj.weight",         "self_attn.q.weight",      passthrough),
    ("self_attn.k_proj.weight",         "self_attn.k.weight",      passthrough),
    ("self_attn.v_proj.weight",         "self_attn.v.weight",      passthrough),
    ("self_attn.o_proj.weight",         "self_attn.out.weight",    passthrough),
    ("post_attention_layernorm.weight", "norm_cross.weight",       passthrough),
    ("encoder_attn.q_proj.weight",      "cross_attn.q.weight",     passthrough),
    ("encoder_attn.k_proj.weight",      "cross_attn.k.weight",     passthrough),
    ("encoder_attn.v_proj.weight",      "cross_attn.v.weight",     passthrough),
    ("encoder_attn.o_proj.weight",      "cross_attn.out.weight",   passthrough),
    ("final_layernorm.weight",          "norm_ffn.weight",         passthrough),
    ("mlp.fc1.weight",                  "ffn.fc1.weight",          passthrough),
    ("mlp.fc1.bias",                    "ffn.fc1.bias",            passthrough),
    ("mlp.fc2.weight",                  "ffn.fc2.weight",          passthrough),
    ("mlp.fc2.bias",                    "ffn.fc2.bias",            passthrough),
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


def convert(model_dir: Path, out_path: Path, variant: str, repo_id: str | None = None) -> None:
    config_path = model_dir / "config.json"
    gen_config_path = model_dir / "generation_config.json"
    preproc_path = model_dir / "preprocessor_config.json"
    safetensors_path = model_dir / "model.safetensors"

    for p in (config_path, gen_config_path, preproc_path, safetensors_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file: {p}")

    REFERENCE_DTYPE_LABEL, REFERENCE_FILE_TYPE, REFERENCE_GGML_TYPE = \
        detect_reference_dtype(safetensors_path)
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    with config_path.open() as f:
        config = json.load(f)
    with gen_config_path.open() as f:
        gen_config = json.load(f)
    with preproc_path.open() as f:
        preproc = json.load(f)

    hp = read_hparams(config, gen_config, preproc)
    print(f"Encoder: {hp['enc_n_layers']} layers, hidden={hp['enc_hidden']}, "
          f"heads={hp['enc_n_heads']}, ffn={hp['enc_intermediate']}, "
          f"act={hp['enc_activation']}, frame_ms={hp['enc_frame_ms']}, "
          f"frame_len={hp['enc_frame_len']}")
    print(f"  sliding_windows (flattened L,R per layer): {hp['enc_sliding_windows']}")
    print(f"Decoder: {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, "
          f"heads={hp['dec_n_heads']}, ffn={hp['dec_intermediate']}, "
          f"act={hp['dec_activation']}")
    print(f"Vocab: {hp['vocab_size']}; max_position_embeddings={hp['max_position_embeddings']}; "
          f"partial_rotary={hp['partial_rotary_factor']}, theta={hp['rope_theta']}")
    print(f"tie_word_embeddings={hp['tie_word_embeddings']}; pad_token_id={hp['pad_id']}")
    print(f"Variant: {variant}")

    print(f"Reading tokenizer from {model_dir}")
    tok = extract_tokenizer(model_dir, hp["vocab_size"])

    has_adapter_proj = hp["enc_hidden"] != hp["dec_hidden"]
    print(f"Adapter proj weight expected in safetensors: {has_adapter_proj}")

    print(f"Opening safetensors at {safetensors_path}")
    with safe_open(str(safetensors_path), framework="pt") as st:
        st_keys = set(st.keys())

        total = 0
        for k in st_keys:
            total += st.get_tensor(k).numel()
        size_label = compute_size_label(total)
        print(f"Total params: {total:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = GGUFWriter(str(out_path), "moonshine_streaming")

        # ---- general.* ----
        _DISPLAY_NAMES = {
            "moonshine-streaming-tiny":   "Moonshine Streaming Tiny",
            "moonshine-streaming-small":  "Moonshine Streaming Small",
            "moonshine-streaming-medium": "Moonshine Streaming Medium",
        }
        if variant not in _DISPLAY_NAMES:
            raise ValueError(f"unknown moonshine_streaming variant slug: {variant!r}")
        add_general_identity(
            writer,
            name=_DISPLAY_NAMES[variant],
            basename="moonshine_streaming",
            size_label=size_label,
            file_type=int(REFERENCE_FILE_TYPE),
            languages=["en"],
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
        writer.add_bool("stt.capability.lang_detect", False)
        writer.add_bool("stt.capability.translate",   False)
        writer.add_bool("stt.capability.timestamps",  False)
        writer.add_bool("stt.capability.streaming",   True)

        # ---- tokenizer.ggml.* ----
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
        writer.add_uint32("tokenizer.ggml.padding_token_id", hp["pad_id"])
        writer.add_bool("tokenizer.ggml.add_bos_token", True)

        # ---- stt.moonshine_streaming.encoder.* ----
        writer.add_uint32("stt.moonshine_streaming.encoder.n_layers",   hp["enc_n_layers"])
        writer.add_uint32("stt.moonshine_streaming.encoder.d_model",    hp["enc_hidden"])
        writer.add_uint32("stt.moonshine_streaming.encoder.n_heads",    hp["enc_n_heads"])
        writer.add_uint32("stt.moonshine_streaming.encoder.n_kv_heads", hp["enc_n_kv_heads"])
        writer.add_uint32("stt.moonshine_streaming.encoder.head_dim",   hp["enc_head_dim"])
        writer.add_uint32("stt.moonshine_streaming.encoder.ffn_dim",    hp["enc_intermediate"])
        writer.add_string("stt.moonshine_streaming.encoder.activation", hp["enc_activation"])
        writer.add_float32("stt.moonshine_streaming.encoder.frame_ms",  hp["enc_frame_ms"])
        writer.add_uint32("stt.moonshine_streaming.encoder.frame_len",  hp["enc_frame_len"])
        # Flattened (L0, R0, L1, R1, ...). C++ loader splits 2 ints per layer.
        writer.add_array("stt.moonshine_streaming.encoder.sliding_windows",
                         hp["enc_sliding_windows"])

        # ---- stt.moonshine_streaming.decoder.* ----
        writer.add_uint32("stt.moonshine_streaming.decoder.n_layers",                hp["dec_n_layers"])
        writer.add_uint32("stt.moonshine_streaming.decoder.d_model",                 hp["dec_hidden"])
        writer.add_uint32("stt.moonshine_streaming.decoder.n_heads",                 hp["dec_n_heads"])
        writer.add_uint32("stt.moonshine_streaming.decoder.n_kv_heads",              hp["dec_n_kv_heads"])
        writer.add_uint32("stt.moonshine_streaming.decoder.head_dim",                hp["dec_head_dim"])
        writer.add_uint32("stt.moonshine_streaming.decoder.ffn_dim",                 hp["dec_intermediate"])
        writer.add_string("stt.moonshine_streaming.decoder.activation",              hp["dec_activation"])
        writer.add_uint32("stt.moonshine_streaming.decoder.vocab_size",              hp["vocab_size"])
        writer.add_uint32("stt.moonshine_streaming.decoder.max_position_embeddings", hp["max_position_embeddings"])
        writer.add_bool  ("stt.moonshine_streaming.decoder.tie_word_embeddings",     hp["tie_word_embeddings"])

        # ---- moonshine_streaming attention / RoPE / norms ----
        writer.add_float32("stt.moonshine_streaming.partial_rotary_factor",   hp["partial_rotary_factor"])
        writer.add_float32("stt.moonshine_streaming.rope_theta",              hp["rope_theta"])
        writer.add_bool   ("stt.moonshine_streaming.attention_bias",          hp["attention_bias"])
        writer.add_uint32 ("stt.moonshine_streaming.pad_head_dim_to_multiple_of", hp["pad_head_dim"])
        # Encoder LayerNorms use unit_offset (gain = gamma + 1.0). Converter
        # PRE-FOLDS +1.0; consumer treats GGUF tensor as the effective gain.
        writer.add_bool   ("stt.moonshine_streaming.encoder_layernorm_unit_offset", True)
        # FrameCMVN epsilon (fixed in modeling code at 1e-6).
        writer.add_float32("stt.moonshine_streaming.cmvn_eps",                1e-6)

        # ---- moonshine_streaming adapter ----
        # When encoder_hidden_size differs from decoder hidden, an adapter
        # Linear is applied; otherwise it's an Identity. The C++ loader
        # uses this flag to know whether to expect adapter.proj.weight.
        writer.add_uint32("stt.moonshine_streaming.encoder_hidden_size", hp["enc_hidden"])
        writer.add_bool  ("stt.moonshine_streaming.adapter_has_proj",    has_adapter_proj)

        # ---- moonshine_streaming prompt / decoding ----
        writer.add_uint32("stt.moonshine_streaming.decoder_start_token_id", hp["decoder_start_id"])
        writer.add_uint32("stt.moonshine_streaming.bos_token_id",           hp["bos_id"])
        writer.add_uint32("stt.moonshine_streaming.eos_token_id",           hp["eos_id"])
        writer.add_uint32("stt.moonshine_streaming.pad_token_id",           hp["pad_id"])

        # ---- stt.frontend.* ----
        # Raw waveform input (Wav2Vec2FeatureExtractor wraps PCM, no mel).
        # The model's "frontend ops" (CMVN, asinh, linear, conv stack) live
        # inside the encoder graph, parameterized by tensors above.
        writer.add_string ("stt.frontend.type",        hp["fe_type"])
        writer.add_uint32 ("stt.frontend.sample_rate", hp["fe_sample_rate"])
        writer.add_uint32 ("stt.frontend.num_mels",    hp["fe_feature_size"])

        # ---- tensors ----
        n_added = 0
        bytes_in = 0
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
            bytes_in += int(arr.nbytes)
            bytes_out += int(encoded.nbytes)
            n_added += 1

        # Encoder embedder + final norm
        for src, dst, xform in ENCODER_EMBEDDER_TABLE:
            add(src, dst, xform)
        # Encoder layers
        for i in range(hp["enc_n_layers"]):
            for suffix_src, suffix_dst, xform in ENCODER_BLOCK_TABLE:
                add(
                    f"model.encoder.layers.{i}.{suffix_src}",
                    f"enc.blocks.{i}.{suffix_dst}",
                    xform,
                )

        # Adapter (lifted out of decoder.* in the GGUF naming)
        add("model.decoder.pos_emb.weight", "adapter.pos_emb.weight", passthrough)
        if has_adapter_proj:
            add("model.decoder.proj.weight", "adapter.proj.weight", passthrough)

        # Decoder top-level (NOT tied: proj_out lives at the model root)
        for src, dst, xform in DECODER_TOP_TABLE:
            add(src, dst, xform)
        # Decoder layers
        for i in range(hp["dec_n_layers"]):
            for suffix_src, suffix_dst, xform in DECODER_BLOCK_TABLE:
                add(
                    f"model.decoder.layers.{i}.{suffix_src}",
                    f"dec.blocks.{i}.{suffix_dst}",
                    xform,
                )

        expected = (
            len(ENCODER_EMBEDDER_TABLE)
            + hp["enc_n_layers"] * len(ENCODER_BLOCK_TABLE)
            + 1                                             # adapter.pos_emb
            + (1 if has_adapter_proj else 0)                # adapter.proj
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
        for src, _, _ in ENCODER_EMBEDDER_TABLE:
            consumed.add(src)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, _, _ in ENCODER_BLOCK_TABLE:
                consumed.add(f"model.encoder.layers.{i}.{suffix_src}")
        consumed.add("model.decoder.pos_emb.weight")
        if has_adapter_proj:
            consumed.add("model.decoder.proj.weight")
        for src, _, _ in DECODER_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["dec_n_layers"]):
            for suffix_src, _, _ in DECODER_BLOCK_TABLE:
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


def _looks_like_repo_id(s: str) -> bool:
    return "/" in s and not Path(s).exists()


def _download_snapshot(repo_id: str, revision: str | None) -> Path:
    slug = slug_from_repo_id(repo_id)
    models_root = os.environ.get("TRANSCRIBE_MODELS_DIR")
    local_dir = Path(models_root) / slug if models_root else None
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
    if revision:
        print(f"Downloading {repo_id}@{revision} from Hugging Face...")
    else:
        print(f"Downloading {repo_id} from Hugging Face "
              f"(no revision pin; reproducibility depends on upstream)...")
    resolved = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir) if local_dir is not None else None,
    )
    return Path(resolved)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a Moonshine Streaming checkpoint to a reference-dtype GGUF.",
    )
    p.add_argument("model", type=str,
                   help="HF repo id (e.g. UsefulSensors/moonshine-streaming-tiny) or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug when "
                        "converting from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch / tag / commit SHA) to pin the download to.")
    p.add_argument("--variant", type=str, default=None,
                   help="stt.variant string (default: derived from slug)")
    args = p.parse_args(argv[1:])

    if _looks_like_repo_id(args.model):
        repo_id = args.repo_id or args.model
        model_dir = _download_snapshot(args.model, args.revision)
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

    convert(model_dir, out_path, variant, repo_id=repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
