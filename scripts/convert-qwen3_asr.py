#!/usr/bin/env python3
"""
convert-qwen3_asr.py - convert a Qwen3-ASR HuggingFace directory to a
GGUF that transcribe.cpp's loader can ingest. Preserves the source BF16
dtype; block quantization (Q8_0, Q5_K_M, ...) goes through
tools/transcribe-quantize.

Source format:
    HuggingFace directory, e.g. Qwen3-ASR-0.6B/, with:

      config.json             Model config (thinker_config: audio_config + text_config)
      generation_config.json  Generation defaults
      preprocessor_config.json WhisperFeatureExtractor parameters
      tokenizer_config.json   Qwen2Tokenizer (Qwen3 BPE) + added tokens + chat template
      vocab.json              BPE vocab (151,643 base tokens)
      merges.txt              BPE merges (151,388 merges)
      chat_template.json      Jinja chat template
      model.safetensors       BF16 weights (612 tensors)

Architecture: audio-llm (audio encoder + Qwen3 causal LM with audio-token
injection). The HF state dict organizes weights under three top-level
prefixes:

    thinker.audio_tower.*   -> audio encoder (Whisper-style transformer with
                               3x Conv2d subsampler, sinusoidal PE,
                               chunked bidirectional attention)
    thinker.model.*         -> Qwen3 causal LM (28 layers, GQA 16/8,
                               head_dim=128, q/k RMSNorm, SwiGLU,
                               interleaved MRoPE)
    thinker.lm_head.weight  -> TIED to thinker.model.embed_tokens.weight;
                               verified bitwise identical, so we skip it.

Layout conversions: NONE. All linear weights are in PyTorch `(out, in)`
order, which is also what ggml matmul expects. Conv2d kernels are in
`(out, in, kH, kW)` order, which is what ggml im2col expects.

KV emitted:

    general.architecture   = "qwen3_asr"
    general.basename       = "qwen3-asr"
    general.size_label     = "0.6B" / "1.7B" / ...
    general.languages      = BCP-47 list (from support_languages)

    stt.variant            = "qwen3-asr-0.6b" (or larger)

    tokenizer.ggml.model   = "gpt2"        (BPE with byte-level pre-tokenizer)
    tokenizer.ggml.tokens  = full token list (vocab.json + added tokens)
    tokenizer.ggml.token_type
    tokenizer.ggml.merges  = BPE merges (space-joined pairs)
    tokenizer.ggml.eos_token_id / padding_token_id / ...
    tokenizer.chat_template = rendered Jinja template

    stt.qwen3_asr.encoder.*     Audio encoder hparams
    stt.qwen3_asr.decoder.*     Text LM hparams (Qwen3-flavored)
    stt.qwen3_asr.audio_token_id / audio_start_token_id / audio_end_token_id
    stt.frontend.*              Whisper frontend parameters

CLI:

    # From an HF repo id (downloads into $TRANSCRIBE_MODELS_DIR/<slug>/)
    uv run --project scripts/envs/qwen3_asr \
      scripts/convert-qwen3_asr.py Qwen/Qwen3-ASR-0.6B

    # From a local directory with an explicit output
    uv run --project scripts/envs/qwen3_asr \
      scripts/convert-qwen3_asr.py \
        models/Qwen3-ASR-0.6B \
        --repo-id Qwen/Qwen3-ASR-0.6B

Single-file, top-to-bottom — no hidden helpers.
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
from contextlib import ExitStack

from safetensors import safe_open


class _ShardedSafetensors:
    """Multi-file safe_open shim.

    Reads model.safetensors.index.json, opens each shard lazily, and
    exposes the same keys()/get_tensor() surface as safe_open so the
    converter doesn't care whether weights are single-file or sharded.
    """

    def __init__(self, model_dir: Path) -> None:
        self._stack = ExitStack()
        single = model_dir / "model.safetensors"
        if single.is_file():
            self._shard_for: dict[str, str] = {}
            sf = self._stack.enter_context(safe_open(str(single), framework="pt"))
            self._handles: dict[str, "safe_open"] = {"__single__": sf}
            for k in sf.keys():
                self._shard_for[k] = "__single__"
            return

        index_path = model_dir / "model.safetensors.index.json"
        with index_path.open() as f:
            index = json.load(f)
        weight_map: dict[str, str] = index["weight_map"]
        shards = sorted(set(weight_map.values()))
        self._handles = {
            name: self._stack.enter_context(
                safe_open(str(model_dir / name), framework="pt"))
            for name in shards
        }
        self._shard_for = weight_map

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._stack.close()

    def keys(self):
        return list(self._shard_for.keys())

    def get_tensor(self, name: str):
        return self._handles[self._shard_for[name]].get_tensor(name)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    add_general_identity,
    encode_for_gguf,
    gguf_name,
    reference_dtype_for,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DTYPE_LABEL = "BF16"
REFERENCE_FILE_TYPE = LlamaFileType.MOSTLY_BF16
REFERENCE_GGML_TYPE = GGMLQuantizationType.BF16


# ---------------------------------------------------------------------------
# BCP-47 mapping for the `support_languages` field in config.json
# ---------------------------------------------------------------------------
LANGUAGE_TO_BCP47 = {
    "Chinese":    "zh",
    "English":    "en",
    "Cantonese":  "yue",
    "Arabic":     "ar",
    "German":     "de",
    "French":     "fr",
    "Spanish":    "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian":    "it",
    "Korean":     "ko",
    "Russian":    "ru",
    "Thai":       "th",
    "Vietnamese": "vi",
    "Japanese":   "ja",
    "Turkish":    "tr",
    "Hindi":      "hi",
    "Malay":      "ms",
    "Dutch":      "nl",
    "Swedish":    "sv",
    "Danish":     "da",
    "Finnish":    "fi",
    "Polish":     "pl",
    "Czech":      "cs",
    "Filipino":   "fil",
    "Persian":    "fa",
    "Greek":      "el",
    "Romanian":   "ro",
    "Hungarian":  "hu",
    "Macedonian": "mk",
}


# ---------------------------------------------------------------------------
# Tokenizer extraction (Qwen3 / Qwen2Tokenizer — byte-level BPE)
# ---------------------------------------------------------------------------
#
# Qwen3 ships vocab.json + merges.txt + tokenizer_config.json, not a
# unified tokenizer.json. We:
#   1. Load the base BPE vocab (vocab.json: token_string -> id)
#   2. Overlay added_tokens_decoder (special tokens past vocab_size
#      covering 151643..151935)
#   3. Emit llama.cpp-style `tokenizer.ggml.model="gpt2"`, tokens,
#      token_type, merges.
#
# Per llama.cpp convention, byte-level BPE tokenizers are tagged "gpt2";
# the loader reassembles bytes via the same GPT-2 bytes↔unicode mapping
# used by the HF fast tokenizer.


def extract_tokenizer(model_dir: Path, vocab_size: int) -> dict:
    vocab_path      = model_dir / "vocab.json"
    merges_path     = model_dir / "merges.txt"
    tokcfg_path     = model_dir / "tokenizer_config.json"

    with vocab_path.open(encoding="utf-8") as f:
        base_vocab: dict[str, int] = json.load(f)
    with merges_path.open(encoding="utf-8") as f:
        merges = [line.rstrip("\n") for line in f.readlines()]
        if merges and merges[0].startswith("#"):
            merges = merges[1:]
    with tokcfg_path.open(encoding="utf-8") as f:
        tokcfg = json.load(f)

    added = tokcfg.get("added_tokens_decoder", {}) or {}
    # Build id -> (token_str, is_special)
    tok_by_id: dict[int, tuple[str, bool]] = {}
    for tok, tid in base_vocab.items():
        tok_by_id[int(tid)] = (tok, False)
    for tid_str, info in added.items():
        tid = int(tid_str)
        tok_by_id[tid] = (info["content"], bool(info.get("special", False)))

    max_id = max(tok_by_id.keys())
    if max_id + 1 > vocab_size:
        raise ValueError(
            f"tokenizer has id {max_id} but config vocab_size={vocab_size}"
        )

    tokens: list[str] = []
    types:  list[int] = []
    for i in range(vocab_size):
        if i not in tok_by_id:
            # Unused slot — Qwen reserves <blank{N}> tokens; treat as
            # normal placeholder so downstream code can detokenize
            # uniformly.
            tokens.append(f"<|unused_{i}|>")
            types.append(TOKEN_TYPE_NORMAL)
            continue
        tok, is_special = tok_by_id[i]
        tokens.append(tok)
        types.append(TOKEN_TYPE_CONTROL if is_special else TOKEN_TYPE_NORMAL)

    def _name_to_id(name_field: str) -> int | None:
        tok = tokcfg.get(name_field)
        if tok is None:
            return None
        if isinstance(tok, dict):
            tok = tok.get("content")
        if not tok:
            return None
        if tok in base_vocab:
            return base_vocab[tok]
        return next(
            (int(tid) for tid, info in added.items() if info["content"] == tok),
            None,
        )

    return {
        "tokens":            tokens,
        "types":             types,
        "merges":            merges,
        "eos_id":            _name_to_id("eos_token"),
        "pad_id":            _name_to_id("pad_token"),
        "bos_id":            _name_to_id("bos_token"),
        "audio_start_id":    _name_to_id("audio_bos_token"),
        "audio_end_id":      _name_to_id("audio_eos_token"),
        "audio_pad_id":      _name_to_id("audio_token"),
    }


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


def read_hparams(config: dict, preproc: dict) -> dict:
    thinker = config["thinker_config"]
    aenc    = thinker["audio_config"]
    tdec    = thinker["text_config"]

    # Frontend (WhisperFeatureExtractor). The preprocessor_config drives
    # the numerics; we record it verbatim so the loader can validate.
    sample_rate = int(preproc.get("sampling_rate", 16000))
    hop_length  = int(preproc["hop_length"])
    n_fft       = int(preproc["n_fft"])
    n_mels      = int(preproc["feature_size"])
    chunk_len   = int(preproc.get("chunk_length", 30))
    n_samples   = int(preproc.get("n_samples", chunk_len * sample_rate))
    nb_max_frm  = int(preproc.get("nb_max_frames", n_samples // hop_length))

    rope_scaling = tdec.get("rope_scaling") or {}
    mrope_section = rope_scaling.get("mrope_section") or [24, 20, 20]
    mrope_interleaved = bool(rope_scaling.get("mrope_interleaved", True))

    # Support-languages from the outer config (publisher's names).
    pub_names = config.get("support_languages", [])
    languages = [LANGUAGE_TO_BCP47.get(n, n) for n in pub_names]

    return {
        # Audio encoder (thinker.audio_tower)
        "enc_n_layers":       int(aenc["encoder_layers"]),
        "enc_d_model":        int(aenc["d_model"]),
        "enc_n_heads":        int(aenc["encoder_attention_heads"]),
        "enc_ffn_dim":        int(aenc["encoder_ffn_dim"]),
        "enc_n_mels":         int(aenc["num_mel_bins"]),
        "enc_downsample_h":   int(aenc["downsample_hidden_size"]),
        "enc_output_dim":     int(aenc["output_dim"]),
        "enc_max_src_pos":    int(aenc["max_source_positions"]),
        "enc_n_window":       int(aenc["n_window"]),
        "enc_n_window_infer": int(aenc["n_window_infer"]),
        "enc_conv_chunksize": int(aenc["conv_chunksize"]),
        "enc_activation":     str(aenc["activation_function"]).lower(),

        # Text LM (thinker.model — Qwen3 causal LM)
        "dec_n_layers":       int(tdec["num_hidden_layers"]),
        "dec_hidden":         int(tdec["hidden_size"]),
        "dec_intermediate":   int(tdec["intermediate_size"]),
        "dec_n_heads":        int(tdec["num_attention_heads"]),
        "dec_n_kv_heads":     int(tdec["num_key_value_heads"]),
        "dec_head_dim":       int(tdec["head_dim"]),
        "dec_hidden_act":     str(tdec["hidden_act"]).lower(),
        "dec_rms_norm_eps":   float(tdec["rms_norm_eps"]),
        "dec_rope_theta":     float(tdec["rope_theta"]),
        "dec_rope_mrope_section":    mrope_section,
        "dec_rope_mrope_interleaved": mrope_interleaved,
        "dec_max_pos_emb":    int(tdec["max_position_embeddings"]),
        "dec_tie_embeddings": bool(tdec.get("tie_word_embeddings", True)),
        "dec_vocab_size":     int(tdec["vocab_size"]),

        # Fusion tokens
        "audio_token_id":        int(thinker["audio_token_id"]),
        "audio_start_token_id":  int(thinker["audio_start_token_id"]),
        "audio_end_token_id":    int(thinker["audio_end_token_id"]),

        # Frontend (Whisper-style)
        "fe_type":        "mel",
        "fe_sample_rate": sample_rate,
        "fe_num_mels":    n_mels,
        "fe_n_fft":       n_fft,
        "fe_win_length":  n_fft,               # WhisperFeatureExtractor: win=n_fft
        "fe_hop_length":  hop_length,
        "fe_window":      "hann_periodic",     # numpy.hanning default in Whisper
        "fe_normalize":   "per_utterance",     # Whisper log-mel clamp + scale
        "fe_dither":      float(preproc.get("dither", 0.0)),
        "fe_pre_emphasis": 0.0,
        "fe_f_min":        0.0,
        "fe_f_max":        float(sample_rate) / 2.0,
        "fe_pad_mode":     "reflect",
        "fe_center":       True,
        "fe_mel_norm":     "slaney",
        "fe_chunk_length": chunk_len,
        "fe_n_samples":    n_samples,
        "fe_nb_max_frm":   nb_max_frm,

        # Metadata
        "languages":       languages,
    }


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------


def passthrough(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr)


# Top-level audio encoder tensors (not inside layers.N).
AUDIO_TOP_TABLE: list[tuple[str, str]] = [
    ("thinker.audio_tower.conv2d1.weight", "enc.conv.0.weight"),
    ("thinker.audio_tower.conv2d1.bias",   "enc.conv.0.bias"),
    ("thinker.audio_tower.conv2d2.weight", "enc.conv.1.weight"),
    ("thinker.audio_tower.conv2d2.bias",   "enc.conv.1.bias"),
    ("thinker.audio_tower.conv2d3.weight", "enc.conv.2.weight"),
    ("thinker.audio_tower.conv2d3.bias",   "enc.conv.2.bias"),
    ("thinker.audio_tower.conv_out.weight", "enc.conv_out.weight"),
    # ln_post + proj1/proj2 are the post-encoder projection head.
    ("thinker.audio_tower.ln_post.weight", "enc.ln_post.weight"),
    ("thinker.audio_tower.ln_post.bias",   "enc.ln_post.bias"),
    ("thinker.audio_tower.proj1.weight",   "enc.proj1.weight"),
    ("thinker.audio_tower.proj1.bias",     "enc.proj1.bias"),
    ("thinker.audio_tower.proj2.weight",   "enc.proj2.weight"),
    ("thinker.audio_tower.proj2.bias",     "enc.proj2.bias"),
]


# Per-encoder-layer block (13 tensors per layer).
AUDIO_BLOCK_TABLE: list[tuple[str, str]] = [
    ("self_attn_layer_norm.weight", "norm_attn.weight"),
    ("self_attn_layer_norm.bias",   "norm_attn.bias"),
    ("self_attn.q_proj.weight",     "attn.q.weight"),
    ("self_attn.q_proj.bias",       "attn.q.bias"),
    ("self_attn.k_proj.weight",     "attn.k.weight"),
    ("self_attn.k_proj.bias",       "attn.k.bias"),
    ("self_attn.v_proj.weight",     "attn.v.weight"),
    ("self_attn.v_proj.bias",       "attn.v.bias"),
    ("self_attn.out_proj.weight",   "attn.out.weight"),
    ("self_attn.out_proj.bias",     "attn.out.bias"),
    ("final_layer_norm.weight",     "norm_ffn.weight"),
    ("final_layer_norm.bias",       "norm_ffn.bias"),
    ("fc1.weight",                  "ffn.fc1.weight"),
    ("fc1.bias",                    "ffn.fc1.bias"),
    ("fc2.weight",                  "ffn.fc2.weight"),
    ("fc2.bias",                    "ffn.fc2.bias"),
]


# Text LM top-level tensors.
#
# Tied output: thinker.lm_head.weight is bitwise equal to
# thinker.model.embed_tokens.weight, so we store only one copy under
# token_embd.weight (llama.cpp convention) and the loader reuses it for
# the output projection.
TEXT_TOP_TABLE: list[tuple[str, str]] = [
    ("thinker.model.embed_tokens.weight", "dec.token_embd.weight"),
    ("thinker.model.norm.weight",         "dec.output_norm.weight"),
]


# Per-text-layer block (12 tensors per layer; attention has separate
# q_norm / k_norm but no biases).
TEXT_BLOCK_TABLE: list[tuple[str, str]] = [
    ("input_layernorm.weight",         "norm_attn.weight"),
    ("post_attention_layernorm.weight", "norm_ffn.weight"),
    ("self_attn.q_proj.weight",        "attn.q.weight"),
    ("self_attn.k_proj.weight",        "attn.k.weight"),
    ("self_attn.v_proj.weight",        "attn.v.weight"),
    ("self_attn.o_proj.weight",        "attn.o.weight"),
    ("self_attn.q_norm.weight",        "attn.q_norm.weight"),
    ("self_attn.k_norm.weight",        "attn.k_norm.weight"),
    ("mlp.gate_proj.weight",           "ffn.gate.weight"),
    ("mlp.up_proj.weight",             "ffn.up.weight"),
    ("mlp.down_proj.weight",           "ffn.down.weight"),
]


# Safetensors keys we deliberately skip.
SKIP_EXACT = {
    # tied to thinker.model.embed_tokens.weight (verified bitwise equal
    # in this checkpoint; tie_word_embeddings=true in config)
    "thinker.lm_head.weight",
}


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
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    config_path     = model_dir / "config.json"
    preproc_path    = model_dir / "preprocessor_config.json"
    single_st       = model_dir / "model.safetensors"
    sharded_index   = model_dir / "model.safetensors.index.json"
    chat_template_path = model_dir / "chat_template.json"

    for p in (config_path, preproc_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file: {p}")
    if not single_st.is_file() and not sharded_index.is_file():
        raise FileNotFoundError(
            f"missing weights: neither {single_st.name} nor "
            f"{sharded_index.name} present in {model_dir}")

    with config_path.open() as f:
        config = json.load(f)
    with preproc_path.open() as f:
        preproc = json.load(f)
    chat_template = None
    if chat_template_path.is_file():
        with chat_template_path.open() as f:
            chat_template = json.load(f).get("chat_template")

    hp = read_hparams(config, preproc)
    print(f"Audio encoder: {hp['enc_n_layers']} layers, d_model={hp['enc_d_model']}")
    print(f"Text LM: {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, "
          f"heads={hp['dec_n_heads']}/{hp['dec_n_kv_heads']}")
    print(f"Vocab: {hp['dec_vocab_size']}; tie_word_embeddings={hp['dec_tie_embeddings']}")

    print(f"Reading tokenizer from {model_dir}")
    tok = extract_tokenizer(model_dir, hp["dec_vocab_size"])

    # Sanity: audio-token IDs in config vs tokenizer_config.
    for role, cfg_id, tok_id in (
        ("audio_start", hp["audio_start_token_id"], tok["audio_start_id"]),
        ("audio_end",   hp["audio_end_token_id"],   tok["audio_end_id"]),
        ("audio_pad",   hp["audio_token_id"],       tok["audio_pad_id"]),
    ):
        if tok_id is not None and tok_id != cfg_id:
            raise ValueError(
                f"{role} token id mismatch: config={cfg_id} tokenizer={tok_id}"
            )

    print(f"Opening safetensors (single or sharded) under {model_dir}")
    with _ShardedSafetensors(model_dir) as st:
        st_keys = set(st.keys())

        # Count params for the size label (exclude tied head).
        total = 0
        for k in st_keys:
            if k in SKIP_EXACT:
                continue
            total += st.get_tensor(k).numel()
        size_label = compute_size_label(total)
        print(f"Total params (deduplicated): {total:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = GGUFWriter(str(out_path), "qwen3_asr")

        # ---- general.* ----
        add_general_identity(
            writer,
            name={
                "qwen3-asr-0.6b": "Qwen3-ASR 0.6B",
                "qwen3-asr-1.7b": "Qwen3-ASR 1.7B",
            }[variant],
            basename="qwen3-asr",
            size_label=size_label,
            file_type=REFERENCE_FILE_TYPE,
            languages=hp["languages"],
            author="Alibaba Qwen Team",
            organization="Qwen",
            license="apache-2.0",
            license_name="Apache License 2.0",
            license_link="https://www.apache.org/licenses/LICENSE-2.0",
            repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
        )

        # ---- stt.variant ----
        writer.add_string("stt.variant", variant)

        # ---- stt.capability.* ----
        # Qwen3-ASR auto-detects the audio's language (the LM prefixes
        # its output with "language X"); advertise that in caps so
        # transcribe_model_capabilities().supports_language_detect is
        # true. Translation is out of scope for this family; no
        # capability KV needed (apply_family_invariants defaults it
        # false). Streaming is a future port — not yet wired end-to-end.
        writer.add_bool("stt.capability.lang_detect", True)

        # ---- tokenizer.ggml.* (llama.cpp "gpt2" byte-level BPE) ----
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_string("tokenizer.ggml.pre",   "qwen2")
        writer.add_array("tokenizer.ggml.tokens",     tok["tokens"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        writer.add_array("tokenizer.ggml.merges",     tok["merges"])
        if tok["eos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
        if tok["pad_id"] is not None:
            writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])
        if tok["bos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
        writer.add_bool("tokenizer.ggml.add_bos_token", False)
        if chat_template is not None:
            writer.add_string("tokenizer.chat_template", chat_template)

        # ---- stt.qwen3_asr.encoder.* ----
        writer.add_uint32("stt.qwen3_asr.encoder.n_layers",         hp["enc_n_layers"])
        writer.add_uint32("stt.qwen3_asr.encoder.d_model",          hp["enc_d_model"])
        writer.add_uint32("stt.qwen3_asr.encoder.n_heads",          hp["enc_n_heads"])
        writer.add_uint32("stt.qwen3_asr.encoder.ffn_dim",          hp["enc_ffn_dim"])
        writer.add_uint32("stt.qwen3_asr.encoder.num_mel_bins",     hp["enc_n_mels"])
        writer.add_uint32("stt.qwen3_asr.encoder.downsample_hidden", hp["enc_downsample_h"])
        writer.add_uint32("stt.qwen3_asr.encoder.output_dim",       hp["enc_output_dim"])
        writer.add_uint32("stt.qwen3_asr.encoder.max_source_positions", hp["enc_max_src_pos"])
        writer.add_uint32("stt.qwen3_asr.encoder.n_window",         hp["enc_n_window"])
        writer.add_uint32("stt.qwen3_asr.encoder.n_window_infer",   hp["enc_n_window_infer"])
        writer.add_uint32("stt.qwen3_asr.encoder.conv_chunksize",   hp["enc_conv_chunksize"])
        writer.add_string("stt.qwen3_asr.encoder.activation",       hp["enc_activation"])

        # ---- stt.qwen3_asr.decoder.* (text LM) ----
        writer.add_uint32("stt.qwen3_asr.decoder.n_layers",       hp["dec_n_layers"])
        writer.add_uint32("stt.qwen3_asr.decoder.hidden_size",    hp["dec_hidden"])
        writer.add_uint32("stt.qwen3_asr.decoder.intermediate_size", hp["dec_intermediate"])
        writer.add_uint32("stt.qwen3_asr.decoder.n_heads",        hp["dec_n_heads"])
        writer.add_uint32("stt.qwen3_asr.decoder.n_kv_heads",     hp["dec_n_kv_heads"])
        writer.add_uint32("stt.qwen3_asr.decoder.head_dim",       hp["dec_head_dim"])
        writer.add_string("stt.qwen3_asr.decoder.hidden_act",     hp["dec_hidden_act"])
        writer.add_float32("stt.qwen3_asr.decoder.rms_norm_eps",  hp["dec_rms_norm_eps"])
        writer.add_float32("stt.qwen3_asr.decoder.rope_theta",    hp["dec_rope_theta"])
        # MRoPE section is a fixed 3-tuple (temporal, height, width).
        # Emit as three scalars rather than an array to dodge the
        # GGUF-writer int-type ambiguity (int32 vs int64 per platform).
        mrope_t, mrope_h, mrope_w = hp["dec_rope_mrope_section"]
        writer.add_uint32("stt.qwen3_asr.decoder.rope_mrope_section_t", int(mrope_t))
        writer.add_uint32("stt.qwen3_asr.decoder.rope_mrope_section_h", int(mrope_h))
        writer.add_uint32("stt.qwen3_asr.decoder.rope_mrope_section_w", int(mrope_w))
        writer.add_bool("stt.qwen3_asr.decoder.rope_mrope_interleaved",
                        hp["dec_rope_mrope_interleaved"])
        writer.add_uint32("stt.qwen3_asr.decoder.max_position_embeddings",
                          hp["dec_max_pos_emb"])
        writer.add_bool("stt.qwen3_asr.decoder.tie_word_embeddings",
                        hp["dec_tie_embeddings"])
        writer.add_uint32("stt.qwen3_asr.decoder.vocab_size",    hp["dec_vocab_size"])

        # ---- audio fusion tokens ----
        writer.add_uint32("stt.qwen3_asr.audio_token_id",       hp["audio_token_id"])
        writer.add_uint32("stt.qwen3_asr.audio_start_token_id", hp["audio_start_token_id"])
        writer.add_uint32("stt.qwen3_asr.audio_end_token_id",   hp["audio_end_token_id"])

        # ---- stt.frontend.* (Whisper feature extractor) ----
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
        writer.add_bool   ("stt.frontend.center",        hp["fe_center"])
        writer.add_string ("stt.frontend.mel_norm",      hp["fe_mel_norm"])
        writer.add_uint32 ("stt.frontend.chunk_length",  hp["fe_chunk_length"])
        writer.add_uint32 ("stt.frontend.n_samples",     hp["fe_n_samples"])
        writer.add_uint32 ("stt.frontend.nb_max_frames", hp["fe_nb_max_frm"])

        # ---- tensors ----
        n_added  = 0
        bytes_in  = 0
        bytes_out = 0

        # ---- frontend buffers (filterbank + window) ----
        # Bake librosa's slaney-normalized mel filterbank and Whisper's
        # periodic Hann window directly into the GGUF so the C++
        # MelFrontend uses bit-identical filterbank + window values.
        # Removes filterbank/window reconstruction as a variable during
        # numerical bring-up.
        import librosa as _lb
        mel_fb = _lb.filters.mel(
            sr=hp["fe_sample_rate"],
            n_fft=hp["fe_n_fft"],
            n_mels=hp["fe_num_mels"],
            fmin=hp["fe_f_min"],
            fmax=hp["fe_f_max"],
            norm="slaney",
            htk=False,
        ).astype(np.float32)
        # [num_mels, n_fft/2 + 1] row-major. The loader reshapes into a
        # flat contiguous buffer.
        writer.add_tensor(
            "frontend.mel_filterbank",
            np.ascontiguousarray(mel_fb),
            raw_dtype=GGMLQuantizationType.F32,
        )
        # Whisper Hann window: periodic (denominator = N, not N-1).
        N = int(hp["fe_win_length"])
        hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / N)).astype(np.float32)
        writer.add_tensor(
            "frontend.window",
            np.ascontiguousarray(hann),
            raw_dtype=GGMLQuantizationType.F32,
        )
        n_added += 2
        bytes_in  += mel_fb.nbytes + hann.nbytes
        bytes_out += mel_fb.nbytes + hann.nbytes

        def add(src_name: str, dst_name: str, transform=passthrough) -> None:
            nonlocal n_added, bytes_in, bytes_out
            if src_name not in st_keys:
                raise KeyError(f"safetensors missing tensor: {src_name!r}")
            t = st.get_tensor(src_name)
            if t.dtype != torch.bfloat16:
                raise ValueError(
                    f"{src_name}: expected torch.bfloat16, got {t.dtype}"
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

        # Audio encoder: top-level + per-layer.
        for src, dst in AUDIO_TOP_TABLE:
            add(src, dst)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, suffix_dst in AUDIO_BLOCK_TABLE:
                add(
                    f"thinker.audio_tower.layers.{i}.{suffix_src}",
                    f"enc.blocks.{i}.{suffix_dst}",
                )

        # Text LM: top-level + per-layer.
        for src, dst in TEXT_TOP_TABLE:
            add(src, dst)
        for i in range(hp["dec_n_layers"]):
            for suffix_src, suffix_dst in TEXT_BLOCK_TABLE:
                add(
                    f"thinker.model.layers.{i}.{suffix_src}",
                    f"dec.blocks.{i}.{suffix_dst}",
                )

        expected = (
            len(AUDIO_TOP_TABLE)
            + hp["enc_n_layers"] * len(AUDIO_BLOCK_TABLE)
            + len(TEXT_TOP_TABLE)
            + hp["dec_n_layers"] * len(TEXT_BLOCK_TABLE)
            + 2  # frontend.mel_filterbank + frontend.window
        )
        if n_added != expected:
            raise RuntimeError(
                f"tensor count mismatch: added {n_added}, expected {expected}"
            )
        print(f"Added {n_added} tensors "
              f"({bytes_in / (1024 * 1024):.1f} MB fp32 -> "
              f"{bytes_out / (1024 * 1024):.1f} MB on disk)")

        # Warn about unconsumed safetensors keys.
        consumed = set(SKIP_EXACT)
        for src, _ in AUDIO_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, _ in AUDIO_BLOCK_TABLE:
                consumed.add(f"thinker.audio_tower.layers.{i}.{suffix_src}")
        for src, _ in TEXT_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["dec_n_layers"]):
            for suffix_src, _ in TEXT_BLOCK_TABLE:
                consumed.add(f"thinker.model.layers.{i}.{suffix_src}")
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
        description="Convert a Qwen3-ASR checkpoint to a BF16 accuracy GGUF.",
    )
    p.add_argument("model", type=str,
                   help="HF repo id (e.g. Qwen/Qwen3-ASR-0.6B) or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug "
                        "when converting from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch / tag / commit SHA) to pin the "
                        "download to. Recommended for reproducibility — the "
                        "golden manifest records the canonical pinned "
                        "revision for each variant. Ignored when `model` is "
                        "a local directory.")
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
        out_path = REPO_ROOT / "models" / slug / gguf_name(slug, REFERENCE_DTYPE_LABEL)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    variant = args.variant
    if variant is None:
        if repo_id:
            variant = slug_from_repo_id(repo_id).lower()
        else:
            # Fallback when the converter was invoked with only a local
            # path + out_path. out_path.stem follows the
            # `<slug>-<QUANT>.gguf` convention (e.g. `Qwen3-ASR-0.6B-BF16`),
            # so strip the trailing `-<QUANT>` so stt.variant carries
            # just the architectural slug (`qwen3-asr-0.6b`), not the
            # quant tag of this particular file. Without this the
            # derived variant stutters into `qwen3-asr-0.6b-bf16`.
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
