#!/usr/bin/env python3
"""
convert-whisper.py - convert a Whisper HuggingFace directory to a GGUF
that transcribe.cpp's loader will ingest. Preserves the source/reference
dtype (F32 for whisper-tiny); block quantization (Q8_0, Q5_K_M, ...)
goes through tools/transcribe-quantize later.

Source format:
    A HuggingFace directory (or repo id), e.g. openai/whisper-tiny, with:

      config.json               Model config (d_model, layers, heads, ...)
      generation_config.json    Special-token IDs, suppress_tokens, lang_to_id
      preprocessor_config.json  WhisperFeatureExtractor parameters + the exact
                                slaney mel filterbank the model was trained
                                with (mel_filters: [n_mels][n_fft/2+1])
      tokenizer.json            Full HF-fast tokenizer (BPE + added_tokens)
      vocab.json                GPT-2 BPE base vocab (50258 entries)
      merges.txt                GPT-2 BPE merges (50000 entries)
      model.safetensors         F32 weights (167 tensors for tiny)

Architecture: encoder-decoder transformer with a convolutional stem.
    encoder
      conv.0 (k=3, stride=1) + GELU  -> [B, d_model, 3000]
      conv.1 (k=3, stride=2) + GELU  -> [B, d_model, 1500]
      + learned positional embedding (stored weight [1500, d_model];
        upstream initializes it from sinusoidal, but it is a plain
        nn.Embedding at inference time)
      4x encoder block (pre-LN: self_attn -> FF)
      final layer norm
    decoder
      token embedding (shared with output projection) +
      learned positional embedding [448, d_model]
      4x decoder block (pre-LN: self_attn -> cross_attn -> FF)
      final layer norm
      tied lm_head projects onto the full 51865-way vocab

Whisper attention quirk: q_proj / v_proj / out_proj all carry bias;
k_proj does NOT (both for self- and cross-attention). This matches the
original OpenAI Whisper implementation and Transformers' WhisperAttention
(bias=False for k_proj).

Layout conversions: NONE. Conv1d kernels are already [out, in, k]; all
linears are [out, in]. Both match ggml's expected layout.

Tensor naming:
    Encoder top-level
      enc.conv.0.weight / .bias          [384, 80, 3] / [384]
      enc.conv.1.weight / .bias          [384, 384, 3] / [384]
      enc.pos_emb.weight                 [max_source_positions=1500, d_model]
      enc.final_norm.weight / .bias      [d_model]
    Encoder per-layer (i = 0..enc_n_layers-1)
      enc.blocks.{i}.norm_attn.weight/bias
      enc.blocks.{i}.attn.q.weight/bias
      enc.blocks.{i}.attn.k.weight            (no bias)
      enc.blocks.{i}.attn.v.weight/bias
      enc.blocks.{i}.attn.out.weight/bias
      enc.blocks.{i}.norm_ffn.weight/bias
      enc.blocks.{i}.ffn.fc1.weight/bias
      enc.blocks.{i}.ffn.fc2.weight/bias
    Decoder top-level
      dec.token_embd.weight              [vocab_size, d_model]  (tied to lm_head)
      dec.pos_emb.weight                 [max_target_positions, d_model]
      dec.final_norm.weight / .bias
    Decoder per-layer
      dec.blocks.{i}.norm_self.weight/bias
      dec.blocks.{i}.self_attn.q.weight/bias
      dec.blocks.{i}.self_attn.k.weight       (no bias)
      dec.blocks.{i}.self_attn.v.weight/bias
      dec.blocks.{i}.self_attn.out.weight/bias
      dec.blocks.{i}.norm_cross.weight/bias
      dec.blocks.{i}.cross_attn.q.weight/bias
      dec.blocks.{i}.cross_attn.k.weight      (no bias)
      dec.blocks.{i}.cross_attn.v.weight/bias
      dec.blocks.{i}.cross_attn.out.weight/bias
      dec.blocks.{i}.norm_ffn.weight/bias
      dec.blocks.{i}.ffn.fc1.weight/bias
      dec.blocks.{i}.ffn.fc2.weight/bias
    Frontend
      frontend.mel_filterbank  [n_mels=80, n_fft/2+1=201]  (from preprocessor)
      frontend.window          [win_length=400]           (hann_periodic)

KV emitted:
    general.architecture = "whisper"
    general.basename     = "whisper"
    general.size_label   = derived from total params (e.g. "39M")
    general.file_type    = reference dtype
    general.languages    = BCP-47 language list from generation_config.lang_to_id

    stt.variant = <slug>                           (e.g. "whisper-tiny")

    tokenizer.ggml.model = "gpt2"                  (byte-level BPE)
    tokenizer.ggml.tokens / token_type / merges
    tokenizer.ggml.bos/eos/padding_token_id

    stt.whisper.encoder.n_layers / d_model / n_heads / ffn_dim /
                       num_mel_bins / max_source_positions / activation
    stt.whisper.decoder.n_layers / d_model / n_heads / ffn_dim /
                       max_target_positions / vocab_size / activation
    stt.whisper.decoder_start_token_id
    stt.whisper.no_timestamps_token_id
    stt.whisper.prev_sot_token_id
    stt.whisper.transcribe_token_id
    stt.whisper.translate_token_id
    stt.whisper.tie_word_embeddings = True
    stt.whisper.decoder.scale_embedding = False   (HF config.scale_embedding)
    stt.whisper.suppress_tokens / begin_suppress_tokens

    stt.frontend.type / num_mels / sample_rate / n_fft / win_length /
                 hop_length / window / normalize / pad_mode /
                 center / mel_norm / chunk_length / n_samples / nb_max_frames

CLI:
    # HF repo id (downloads into $TRANSCRIBE_MODELS_DIR or HF cache)
    uv run --project scripts/envs/whisper \
      scripts/convert-whisper.py openai/whisper-tiny

    # Local directory
    uv run --project scripts/envs/whisper \
      scripts/convert-whisper.py <model-dir> --repo-id openai/whisper-tiny

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
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    encode_for_gguf,
    gguf_name,
    reference_dtype_for,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Reference dtype
# ---------------------------------------------------------------------------
# Most Whisper checkpoints on HF ship in float32 (tiny … large-v2). The
# large-v3 family (large-v3, large-v3-turbo) ship float16. The reference
# dtype is whatever the safetensors header says — picked at convert time
# below in `detect_reference_dtype` and threaded through as locals.


def detect_reference_dtype(safetensors_path: Path) -> tuple[str, LlamaFileType, GGMLQuantizationType]:
    """Inspect the safetensors header and pick (label, file_type, ggml_type).
    Returns F32 by default; F16 if every floating tensor is float16."""
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
        # Mixed F32 + F16 — promote to F32 reference. Upcast on load.
        return ("F32", LlamaFileType.ALL_F32, GGMLQuantizationType.F32)
    raise ValueError(
        f"unsupported safetensors dtype mix: {sorted(dtypes)} in {safetensors_path}"
    )


# ---------------------------------------------------------------------------
# Tokenizer extraction (GPT-2 byte-level BPE + whisper's added_tokens)
# ---------------------------------------------------------------------------
#
# Whisper's HF tokenizer is a byte-level BPE built over GPT-2's vocab,
# extended with 1607 added tokens: 99 language tokens, 2 task tokens,
# ~5 auxiliary control tokens, and 1501 timestamp tokens at 20ms
# granularity. Base + added = 50258 + 1607 = 51865, which matches the
# model output dim. The tokenizer.json has the canonical combined view.


def extract_tokenizer(model_dir: Path, vocab_size: int) -> dict:
    tokenizer_json = model_dir / "tokenizer.json"
    with tokenizer_json.open(encoding="utf-8") as f:
        tj = json.load(f)

    if tj["model"].get("type") != "BPE":
        raise ValueError(
            f"expected tokenizer.json model.type=BPE, got {tj['model'].get('type')!r}"
        )
    base_vocab: dict[str, int] = tj["model"]["vocab"]
    merges_raw = tj["model"].get("merges", [])

    # HF v0.15+ encodes merges as [[a, b], ...]; older versions use
    # "a b" joined strings. Emit llama.cpp's space-joined form either way.
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
    if max_id + 1 > vocab_size:
        raise ValueError(
            f"tokenizer has id {max_id} but config vocab_size={vocab_size}"
        )

    tokens: list[str] = []
    types:  list[int] = []
    for i in range(vocab_size):
        if i not in tok_by_id:
            tokens.append(f"<|unused_{i}|>")
            types.append(TOKEN_TYPE_NORMAL)
            continue
        tok, is_special = tok_by_id[i]
        tokens.append(tok)
        types.append(TOKEN_TYPE_CONTROL if is_special else TOKEN_TYPE_NORMAL)

    # Look up canonical Whisper special tokens by their literal content
    # (not by tokenizer_config role names — Whisper overloads bos/eos/pad
    # all to <|endoftext|>=50257, so we also carry the ids that matter
    # for decoding as top-level stt.whisper.* KV below).
    content_to_id = {tok: tid for tok, tid in base_vocab.items()}
    for entry in added_tokens:
        content_to_id[entry["content"]] = int(entry["id"])

    def tok_id(content: str) -> int | None:
        return content_to_id.get(content)

    return {
        "tokens":            tokens,
        "types":             types,
        "merges":            merges,
        "bos_id":            tok_id("<|endoftext|>"),
        "eos_id":            tok_id("<|endoftext|>"),
        "pad_id":            tok_id("<|endoftext|>"),
        "sot_id":            tok_id("<|startoftranscript|>"),
        "transcribe_id":     tok_id("<|transcribe|>"),
        "translate_id":      tok_id("<|translate|>"),
        "no_timestamps_id":  tok_id("<|notimestamps|>"),
        "prev_sot_id":       tok_id("<|startofprev|>"),
    }


# ---------------------------------------------------------------------------
# Hparams from config.json + generation_config.json + preprocessor_config.json
# ---------------------------------------------------------------------------


def read_hparams(config: dict, gen_config: dict, preproc: dict) -> dict:
    d_model             = int(config["d_model"])
    enc_layers          = int(config["encoder_layers"])
    enc_heads           = int(config["encoder_attention_heads"])
    enc_ffn             = int(config["encoder_ffn_dim"])
    dec_layers          = int(config["decoder_layers"])
    dec_heads           = int(config["decoder_attention_heads"])
    dec_ffn             = int(config["decoder_ffn_dim"])
    num_mel_bins        = int(config["num_mel_bins"])
    max_source_positions = int(config["max_source_positions"])
    max_target_positions = int(config["max_target_positions"])
    vocab_size          = int(config["vocab_size"])
    activation          = str(config["activation_function"]).lower()
    scale_embedding     = bool(config.get("scale_embedding", False))

    # generation_config carries special-token ids and suppression lists.
    decoder_start_id = int(gen_config["decoder_start_token_id"])
    no_ts_id         = int(gen_config["no_timestamps_token_id"])
    prev_sot_id      = int(gen_config.get("prev_sot_token_id", 0) or 0)
    suppress_tokens  = [int(x) for x in gen_config.get("suppress_tokens",  []) or []]
    begin_suppress   = [int(x) for x in gen_config.get("begin_suppress_tokens", []) or []]

    # Frontend (WhisperFeatureExtractor fields are canonical; the exact
    # mel_filters array comes out as a tensor, not a KV — see convert()).
    sample_rate  = int(preproc.get("sampling_rate", 16000))
    n_fft        = int(preproc["n_fft"])
    hop_length   = int(preproc["hop_length"])
    feature_size = int(preproc["feature_size"])
    chunk_length = int(preproc.get("chunk_length", 30))
    n_samples    = int(preproc.get("n_samples", chunk_length * sample_rate))
    nb_max_frm   = int(preproc.get("nb_max_frames", n_samples // hop_length))

    # Languages: derive BCP-47 list from generation_config.lang_to_id
    # keys (shaped like "<|en|>", "<|zh|>", ...). Preserves the publisher's
    # declared language set; English-only .en variants omit lang_to_id,
    # so we fall back to ["en"] in that case.
    lang_to_id = gen_config.get("lang_to_id") or {}
    languages = [tok[2:-2] for tok in lang_to_id.keys()] or ["en"]

    return {
        "d_model":              d_model,
        "enc_n_layers":         enc_layers,
        "enc_n_heads":          enc_heads,
        "enc_ffn_dim":          enc_ffn,
        "dec_n_layers":         dec_layers,
        "dec_n_heads":          dec_heads,
        "dec_ffn_dim":          dec_ffn,
        "num_mel_bins":         num_mel_bins,
        "max_source_positions": max_source_positions,
        "max_target_positions": max_target_positions,
        "vocab_size":           vocab_size,
        "activation":           activation,
        "scale_embedding":      scale_embedding,

        "decoder_start_token_id":  decoder_start_id,
        "no_timestamps_token_id":  no_ts_id,
        "prev_sot_token_id":       prev_sot_id,
        "suppress_tokens":         suppress_tokens,
        "begin_suppress_tokens":   begin_suppress,

        "fe_type":        "mel",
        "fe_sample_rate": sample_rate,
        "fe_num_mels":    feature_size,
        "fe_n_fft":       n_fft,
        "fe_win_length":  n_fft,            # WhisperFeatureExtractor: win=n_fft
        "fe_hop_length":  hop_length,
        "fe_window":      "hann_periodic",
        "fe_normalize":   "whisper_logmel", # log10 + dynamic-range clamp + scale
        "fe_dither":      0.0,
        "fe_pre_emphasis": 0.0,
        "fe_f_min":        0.0,
        "fe_f_max":        float(sample_rate) / 2.0,
        "fe_pad_mode":     "reflect",
        "fe_center":       True,
        "fe_mel_norm":     "slaney",
        "fe_chunk_length": chunk_length,
        "fe_n_samples":    n_samples,
        "fe_nb_max_frm":   nb_max_frm,

        "languages": languages,
    }


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------


def passthrough(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr)


ENCODER_TOP_TABLE: list[tuple[str, str]] = [
    ("model.encoder.conv1.weight",            "enc.conv.0.weight"),
    ("model.encoder.conv1.bias",              "enc.conv.0.bias"),
    ("model.encoder.conv2.weight",            "enc.conv.1.weight"),
    ("model.encoder.conv2.bias",              "enc.conv.1.bias"),
    ("model.encoder.embed_positions.weight",  "enc.pos_emb.weight"),
    ("model.encoder.layer_norm.weight",       "enc.final_norm.weight"),
    ("model.encoder.layer_norm.bias",         "enc.final_norm.bias"),
]


# Whisper attention: q / v / out have bias, k does NOT.
ENCODER_BLOCK_TABLE: list[tuple[str, str]] = [
    ("self_attn_layer_norm.weight", "norm_attn.weight"),
    ("self_attn_layer_norm.bias",   "norm_attn.bias"),
    ("self_attn.q_proj.weight",     "attn.q.weight"),
    ("self_attn.q_proj.bias",       "attn.q.bias"),
    ("self_attn.k_proj.weight",     "attn.k.weight"),
    # k has no bias — intentionally omitted.
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


DECODER_TOP_TABLE: list[tuple[str, str]] = [
    ("model.decoder.embed_tokens.weight",     "dec.token_embd.weight"),
    ("model.decoder.embed_positions.weight",  "dec.pos_emb.weight"),
    ("model.decoder.layer_norm.weight",       "dec.final_norm.weight"),
    ("model.decoder.layer_norm.bias",         "dec.final_norm.bias"),
]


DECODER_BLOCK_TABLE: list[tuple[str, str]] = [
    # Self-attention
    ("self_attn_layer_norm.weight",     "norm_self.weight"),
    ("self_attn_layer_norm.bias",       "norm_self.bias"),
    ("self_attn.q_proj.weight",         "self_attn.q.weight"),
    ("self_attn.q_proj.bias",           "self_attn.q.bias"),
    ("self_attn.k_proj.weight",         "self_attn.k.weight"),
    ("self_attn.v_proj.weight",         "self_attn.v.weight"),
    ("self_attn.v_proj.bias",           "self_attn.v.bias"),
    ("self_attn.out_proj.weight",       "self_attn.out.weight"),
    ("self_attn.out_proj.bias",         "self_attn.out.bias"),
    # Cross-attention (encoder_attn)
    ("encoder_attn_layer_norm.weight",  "norm_cross.weight"),
    ("encoder_attn_layer_norm.bias",    "norm_cross.bias"),
    ("encoder_attn.q_proj.weight",      "cross_attn.q.weight"),
    ("encoder_attn.q_proj.bias",        "cross_attn.q.bias"),
    ("encoder_attn.k_proj.weight",      "cross_attn.k.weight"),
    ("encoder_attn.v_proj.weight",      "cross_attn.v.weight"),
    ("encoder_attn.v_proj.bias",        "cross_attn.v.bias"),
    ("encoder_attn.out_proj.weight",    "cross_attn.out.weight"),
    ("encoder_attn.out_proj.bias",      "cross_attn.out.bias"),
    # Feed-forward
    ("final_layer_norm.weight",         "norm_ffn.weight"),
    ("final_layer_norm.bias",           "norm_ffn.bias"),
    ("fc1.weight",                      "ffn.fc1.weight"),
    ("fc1.bias",                        "ffn.fc1.bias"),
    ("fc2.weight",                      "ffn.fc2.weight"),
    ("fc2.bias",                        "ffn.fc2.bias"),
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


def convert(model_dir: Path, out_path: Path, variant: str) -> None:
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

    with config_path.open() as f:
        config = json.load(f)
    with gen_config_path.open() as f:
        gen_config = json.load(f)
    with preproc_path.open() as f:
        preproc = json.load(f)

    hp = read_hparams(config, gen_config, preproc)
    print(f"Encoder: {hp['enc_n_layers']} layers, d_model={hp['d_model']}, "
          f"heads={hp['enc_n_heads']}, ffn={hp['enc_ffn_dim']}")
    print(f"Decoder: {hp['dec_n_layers']} layers, d_model={hp['d_model']}, "
          f"heads={hp['dec_n_heads']}, ffn={hp['dec_ffn_dim']}")
    print(f"Vocab: {hp['vocab_size']}; mel_bins={hp['num_mel_bins']}; "
          f"src_pos={hp['max_source_positions']}; tgt_pos={hp['max_target_positions']}")
    print(f"Languages: {len(hp['languages'])}")
    print(f"Variant: {variant}")

    print(f"Reading tokenizer from {model_dir}")
    tok = extract_tokenizer(model_dir, hp["vocab_size"])

    print(f"Opening safetensors at {safetensors_path}")
    with safe_open(str(safetensors_path), framework="pt") as st:
        st_keys = set(st.keys())

        # Count params for size label. All whisper tensors are already
        # in the safetensors (lm_head is tied and not stored separately).
        total = 0
        for k in st_keys:
            total += st.get_tensor(k).numel()
        size_label = compute_size_label(total)
        print(f"Total params: {total:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = GGUFWriter(str(out_path), "whisper")

        # ---- general.* ----
        writer.add_string("general.basename",   "whisper")
        writer.add_string("general.size_label", size_label)
        writer.add_uint32("general.file_type",  int(REFERENCE_FILE_TYPE))
        writer.add_array("general.languages",   hp["languages"])

        # ---- stt.variant ----
        writer.add_string("stt.variant", variant)

        # ---- stt.capability.* ----
        # Multilingual checkpoints support auto language detection and
        # speech translation via the <|translate|> task token. The .en
        # checkpoints (English-only) drop both: their vocab has no
        # language tokens and no <|translate|> task token.
        is_multilingual = len(hp["languages"]) > 1
        writer.add_bool("stt.capability.lang_detect", is_multilingual)
        writer.add_bool("stt.capability.translate",   is_multilingual)
        writer.add_bool("stt.capability.timestamps",  True)

        # ---- tokenizer.ggml.* (llama.cpp "gpt2" byte-level BPE) ----
        # tokenizer.ggml.pre="gpt2" selects the original GPT-2
        # pretokenizer regex (digit runs merged, lowercase-only
        # contractions). Whisper's HF tokenizer is a ByteLevel
        # use_regex=true pretokenizer, which the tokenizers crate
        # implements with the GPT-2 regex verbatim.
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_string("tokenizer.ggml.pre",   "gpt2")
        writer.add_array("tokenizer.ggml.tokens",     tok["tokens"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        writer.add_array("tokenizer.ggml.merges",     tok["merges"])
        if tok["bos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
        if tok["eos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
        if tok["pad_id"] is not None:
            writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])
        writer.add_bool("tokenizer.ggml.add_bos_token", False)

        # ---- stt.whisper.encoder.* ----
        writer.add_uint32("stt.whisper.encoder.n_layers",            hp["enc_n_layers"])
        writer.add_uint32("stt.whisper.encoder.d_model",             hp["d_model"])
        writer.add_uint32("stt.whisper.encoder.n_heads",             hp["enc_n_heads"])
        writer.add_uint32("stt.whisper.encoder.ffn_dim",             hp["enc_ffn_dim"])
        writer.add_uint32("stt.whisper.encoder.num_mel_bins",        hp["num_mel_bins"])
        writer.add_uint32("stt.whisper.encoder.max_source_positions", hp["max_source_positions"])
        writer.add_string("stt.whisper.encoder.activation",          hp["activation"])

        # ---- stt.whisper.decoder.* ----
        writer.add_uint32("stt.whisper.decoder.n_layers",             hp["dec_n_layers"])
        writer.add_uint32("stt.whisper.decoder.d_model",              hp["d_model"])
        writer.add_uint32("stt.whisper.decoder.n_heads",              hp["dec_n_heads"])
        writer.add_uint32("stt.whisper.decoder.ffn_dim",              hp["dec_ffn_dim"])
        writer.add_uint32("stt.whisper.decoder.max_target_positions", hp["max_target_positions"])
        writer.add_uint32("stt.whisper.decoder.vocab_size",           hp["vocab_size"])
        writer.add_string("stt.whisper.decoder.activation",           hp["activation"])
        writer.add_bool  ("stt.whisper.decoder.tie_word_embeddings",  True)
        writer.add_bool  ("stt.whisper.decoder.scale_embedding",      hp["scale_embedding"])

        # ---- whisper decoder prompt / suppression ----
        writer.add_uint32("stt.whisper.decoder_start_token_id", hp["decoder_start_token_id"])
        writer.add_uint32("stt.whisper.no_timestamps_token_id", hp["no_timestamps_token_id"])
        if tok["sot_id"] is not None:
            writer.add_uint32("stt.whisper.sot_token_id", tok["sot_id"])
        if tok["transcribe_id"] is not None:
            writer.add_uint32("stt.whisper.transcribe_token_id", tok["transcribe_id"])
        if tok["translate_id"] is not None:
            writer.add_uint32("stt.whisper.translate_token_id", tok["translate_id"])
        if tok["prev_sot_id"] is not None:
            writer.add_uint32("stt.whisper.prev_sot_token_id", tok["prev_sot_id"])
        if hp["suppress_tokens"]:
            writer.add_array("stt.whisper.suppress_tokens",       hp["suppress_tokens"])
        if hp["begin_suppress_tokens"]:
            writer.add_array("stt.whisper.begin_suppress_tokens", hp["begin_suppress_tokens"])

        # ---- stt.frontend.* (WhisperFeatureExtractor) ----
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
        n_added   = 0
        bytes_in  = 0
        bytes_out = 0

        # ---- frontend buffers ----
        # The Whisper preprocessor_config.json on tiny … large-v2 ships the
        # model's exact slaney mel filterbank (fixed-precision JSON). Use
        # those raw values when present. The large-v3 family
        # (large-v3, large-v3-turbo) drops `mel_filters` from the JSON and
        # expects WhisperFeatureExtractor to compute them at load time via
        # transformers.audio_utils.mel_filter_bank. Reproduce that exact
        # call here so the GGUF carries the same filters HF would build.
        if "mel_filters" in preproc:
            mel_fb_list = preproc["mel_filters"]
            mel_fb = np.asarray(mel_fb_list, dtype=np.float32)
        else:
            from transformers.audio_utils import mel_filter_bank
            mel_fb = mel_filter_bank(
                num_frequency_bins=1 + hp["fe_n_fft"] // 2,
                num_mel_filters=hp["fe_num_mels"],
                min_frequency=0.0,
                max_frequency=hp["fe_f_max"],
                sampling_rate=hp["fe_sample_rate"],
                norm="slaney",
                mel_scale="slaney",
            ).T.astype(np.float32)
            print(f"computed mel filterbank via transformers.audio_utils "
                  f"(no mel_filters in preprocessor_config.json)")
        if mel_fb.shape != (hp["fe_num_mels"], hp["fe_n_fft"] // 2 + 1):
            raise ValueError(
                f"mel_filters shape {mel_fb.shape} does not match "
                f"[num_mels={hp['fe_num_mels']}, n_fft/2+1="
                f"{hp['fe_n_fft'] // 2 + 1}]"
            )
        writer.add_tensor(
            "frontend.mel_filterbank",
            np.ascontiguousarray(mel_fb),
            raw_dtype=GGMLQuantizationType.F32,
        )
        # WhisperFeatureExtractor uses a periodic Hann window of length
        # n_fft (denominator = N, not N-1). torch.hann_window(N) with the
        # default periodic=True produces the same values.
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
            if t.dtype not in (torch.float32, torch.float16):
                raise ValueError(
                    f"{src_name}: expected float32 or float16, got {t.dtype}"
                )
            # encode_for_gguf wants fp32 input. F16 → F32 is exact, so we
            # upcast on read and let encode_for_gguf re-pack to the
            # per-tensor target dtype (which respects REFERENCE_GGML_TYPE).
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

        # Encoder top-level (conv stem + pos_emb + final norm)
        for src, dst in ENCODER_TOP_TABLE:
            add(src, dst)
        # Encoder layers
        for i in range(hp["enc_n_layers"]):
            for suffix_src, suffix_dst in ENCODER_BLOCK_TABLE:
                add(
                    f"model.encoder.layers.{i}.{suffix_src}",
                    f"enc.blocks.{i}.{suffix_dst}",
                )

        # Decoder top-level (embeds + final norm; lm_head is tied)
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
            + 2  # frontend.mel_filterbank + frontend.window
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

        # Warn about unconsumed safetensors keys (should be empty for whisper).
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
        description="Convert a Whisper checkpoint to an F32 reference GGUF.",
    )
    p.add_argument("model", type=str,
                   help="HF repo id (e.g. openai/whisper-tiny) or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug when "
                        "converting from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch / tag / commit SHA) to pin the "
                        "download to. Recommended for reproducibility; the "
                        "intake records the canonical pinned revision.")
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
        # Derive output filename from the safetensors header dtype so the
        # filename matches the actual GGUF contents (F32 vs F16).
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

    convert(model_dir, out_path, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
