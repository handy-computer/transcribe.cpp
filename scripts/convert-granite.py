#!/usr/bin/env python3
"""
convert-granite.py - convert an IBM Granite Speech HuggingFace directory
to a GGUF that transcribe.cpp's loader can ingest. Preserves the source
BF16 dtype; block quantization (Q8_0, Q5_K_M, ...) goes through
tools/transcribe-quantize.

Source format:
    HuggingFace directory, e.g. granite-4.0-1b-speech/, with:

      config.json             Model config (encoder/projector/text sub-configs)
      preprocessor_config.json torchaudio melspec parameters (melspec_kwargs)
      processor_config.json   GraniteSpeechProcessor + audio_token
      tokenizer.json          BPE vocab + merges (byte-level) + 97 added tokens
      tokenizer_config.json   Special token strings, GPT2Tokenizer class
      chat_template.jinja     Jinja chat template (template differs per variant)
      model.safetensors(.*)   BF16 weights (sharded across 3 files)

Architecture: audio-llm (Conformer encoder + BLIP-2 Q-Former projector
+ Granite-4 causal LM with audio-token injection). HF state dict
organizes weights under three top-level prefixes:

    encoder.*               Conformer encoder (macaroni FFN, GLU conv,
                            Shaw-style relative positional attention,
                            self-conditioned CTC bypass via out_mid)
    projector.*             3 learned queries, 2-layer Q-Former with
                            self+cross attention, linear lift to LM hidden
    language_model.*        Granite-4 causal LM (40 layers, GQA 16/4,
                            head_dim=128, RMSNorm, SwiGLU, multipliers
                            embedding=12, logits_scaling=8,
                            attention=1/128, residual=0.22)

Layout conversions: NONE. Linear weights are PyTorch (out, in) which
matches ggml. Conv1d kernels are (out, in/g, kW) which matches ggml
im2col.

Variant differences handled in one dispatch:

  granite-4.0-1b-speech / granite-speech-4.1-2b
    cat_hidden_layers = []           (projector input = encoder hidden = 1024)
    tie_word_embeddings = False      (separate language_model.lm_head present)

  granite-speech-4.1-2b-plus
    cat_hidden_layers = [3]          (encoder concatenates layer-3 hidden
                                      with final layer hidden along the
                                      channel dim; projector input = 2048)
    tie_word_embeddings = True       (no lm_head in safetensors; head
                                      reuses embed_tokens)

KV emitted (consumed by Stage 4 read_granite_hparams):

  general.architecture   = "granite_speech"
  general.basename       = "granite-speech"
  general.size_label     = "1.0B" / "2.0B" / ...
  general.languages      = BCP-47 ASR source-language list (5 or 6 langs)

  stt.variant            = e.g. "granite-4.0-1b-speech"
  stt.capability.translate   = bool (true for 1b/2b; false for plus)
  stt.translation.target_languages = BCP-47 target list when translation is true

  tokenizer.ggml.model   = "gpt2"        (BPE with byte-level pre-tokenizer)
  tokenizer.ggml.tokens / merges / token_type
  tokenizer.ggml.bos_token_id / eos_token_id / pad_token_id / unknown_token_id
  tokenizer.chat_template = jinja template (variant-specific)

  stt.granite.encoder.*  Conformer hparams + cat_hidden_layers
  stt.granite.projector.* Q-Former hparams + linear-lift dims
  stt.granite.decoder.*  Granite-4 hparams (incl. multipliers)
  stt.granite.audio_token_id / window_size / downsample_rate
  stt.frontend.*         torchaudio melspec parameters

CLI:

  # HF repo id
  uv run --project scripts/envs/granite \
    scripts/convert-granite.py ibm-granite/granite-4.0-1b-speech

  # local directory
  uv run --project scripts/envs/granite \
    scripts/convert-granite.py models/granite-4.0-1b-speech \
      --repo-id ibm-granite/granite-4.0-1b-speech

Single file, top-to-bottom — no hidden helpers.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, LlamaFileType
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.hf_source import download_snapshot, looks_like_repo_id  # noqa: E402
from lib.gguf_common import (  # noqa: E402
    gguf_writer,
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


# Granite Speech shipping languages (from the model cards). The 4.1-2b
# and 4.0-1b variants advertise English plus FR/DE/ES/PT (+ JA on
# 4.0-1b/4.1-2b base). The -plus variant drops JA on its public spec.
LANG_BY_VARIANT = {
    "granite-4.0-1b-speech":      ["en", "fr", "de", "es", "pt", "ja"],
    "granite-speech-4.1-2b":      ["en", "fr", "de", "es", "pt", "ja"],
    "granite-speech-4.1-2b-plus": ["en", "fr", "de", "es", "pt"],
}

# Translation targets are not exactly the ASR language set: the base AR
# variants also advertise English-to-Italian and English-to-Mandarin. Keep this
# separate from general.languages so `language=it/zh` is still rejected as an
# unsupported source language while `target_language=it/zh` is accepted.
TRANSLATION_TARGET_LANG_BY_VARIANT = {
    "granite-4.0-1b-speech": ["en", "fr", "de", "es", "pt", "ja", "it", "zh"],
    "granite-speech-4.1-2b": ["en", "fr", "de", "es", "pt", "ja", "it", "zh"],
}


def granite_translation_pairs(asr_langs: list[str]) -> list[str]:
    """Model-card translation directions for the base AR variants."""
    pairs: list[str] = []
    for lang in asr_langs:
        if lang == "en":
            continue
        pairs.append(f"en>{lang}")
        pairs.append(f"{lang}>en")
    pairs.extend(["en>it", "en>zh"])
    return pairs


# ---------------------------------------------------------------------------
# Sharded safetensors shim
# ---------------------------------------------------------------------------


class _ShardedSafetensors:
    """Multi-file safe_open shim — matches qwen3_asr's shape."""

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


# ---------------------------------------------------------------------------
# Tokenizer extraction (Granite-4 BPE — byte-level, GPT-2 family)
# ---------------------------------------------------------------------------


def extract_tokenizer(model_dir: Path, vocab_size: int) -> dict:
    """Read tokenizer.json (canonical) and overlay added_tokens.

    Granite-4 ships a tokenizer.json with base BPE vocab (100,256 tokens)
    plus 97 added tokens (IDs 100256..100352) covering pad, eos, FIM,
    chat roles, tool/think tags, audio. tokenizer_config.json provides
    the special-token *strings*; we resolve them to IDs via the union of
    base vocab and added_tokens.

    GGUF uses llama.cpp's "gpt2" convention for byte-level BPE — the
    loader reassembles bytes via the GPT-2 bytes<->unicode mapping.
    """
    with (model_dir / "tokenizer.json").open(encoding="utf-8") as f:
        tj = json.load(f)
    with (model_dir / "tokenizer_config.json").open(encoding="utf-8") as f:
        tokcfg = json.load(f)

    model = tj.get("model") or {}
    base_vocab: dict[str, int] = dict(model.get("vocab") or {})
    raw_merges = model.get("merges") or []
    merges: list[str] = []
    for m in raw_merges:
        if isinstance(m, list):
            if len(m) != 2:
                raise ValueError(f"unexpected merge format: {m!r}")
            merges.append(f"{m[0]} {m[1]}")
        elif isinstance(m, str):
            merges.append(m)
        else:
            raise ValueError(f"unexpected merge entry type: {type(m).__name__}")

    added_list = tj.get("added_tokens") or []
    # id -> (content, is_special). Base vocab entries default to non-special.
    tok_by_id: dict[int, tuple[str, bool]] = {}
    for tok, tid in base_vocab.items():
        tok_by_id[int(tid)] = (tok, False)
    for a in added_list:
        tid = int(a["id"])
        tok_by_id[tid] = (a["content"], bool(a.get("special", False)))

    max_id = max(tok_by_id.keys())
    if max_id + 1 > vocab_size:
        raise ValueError(
            f"tokenizer has id {max_id} but text_config.vocab_size={vocab_size}"
        )

    tokens: list[str] = []
    types: list[int] = []
    for i in range(vocab_size):
        if i not in tok_by_id:
            tokens.append(f"<|unused_{i}|>")
            types.append(TOKEN_TYPE_NORMAL)
            continue
        tok, is_special = tok_by_id[i]
        tokens.append(tok)
        types.append(TOKEN_TYPE_CONTROL if is_special else TOKEN_TYPE_NORMAL)

    def _name_to_id(field: str) -> int | None:
        val = tokcfg.get(field)
        if val is None:
            return None
        if isinstance(val, dict):
            val = val.get("content")
        if not val:
            return None
        if val in base_vocab:
            return int(base_vocab[val])
        return next(
            (int(a["id"]) for a in added_list if a.get("content") == val),
            None,
        )

    return {
        "tokens": tokens,
        "types":  types,
        "merges": merges,
        "bos_id": _name_to_id("bos_token"),
        "eos_id": _name_to_id("eos_token"),
        "pad_id": _name_to_id("pad_token"),
        "unk_id": _name_to_id("unk_token"),
    }


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


def read_hparams(config: dict, preproc: dict, processor: dict) -> dict:
    enc = config["encoder_config"]
    prj = config["projector_config"]
    txt = config["text_config"]

    cat_hidden_layers = list(enc.get("cat_hidden_layers") or [])

    # The top-level tie_word_embeddings can be missing/false while the
    # nested text_config.tie_word_embeddings is true (granite-speech-4.1-2b-plus
    # is the live example). Resolve to whichever sub-flag actually
    # controls the head, defaulting to the top-level when neither is set.
    tie = txt.get("tie_word_embeddings")
    if tie is None:
        tie = config.get("tie_word_embeddings", False)
    tie = bool(tie)

    rope_params = txt.get("rope_parameters") or {}
    rope_theta = float(
        rope_params.get("rope_theta")
        if rope_params.get("rope_theta") is not None
        else txt.get("rope_theta", 10000.0)
    )

    # Frontend (torchaudio MelSpectrogram). preprocessor_config.json only
    # records the explicit melspec_kwargs; the remaining torchaudio
    # defaults (hann_periodic window, center=True, reflect pad, htk mel
    # scale, no normalization) are baked in here. Stage 2 confirmed
    # these against the reference frames.
    mks = preproc.get("melspec_kwargs") or {}
    if not mks:
        proc_audio = (processor.get("audio_processor") or {})
        mks = proc_audio.get("melspec_kwargs") or {}
    sample_rate = int(mks.get("sample_rate", 16000))
    n_fft = int(mks.get("n_fft", 512))
    win_length = int(mks.get("win_length", 400))
    hop_length = int(mks.get("hop_length", 160))
    n_mels = int(mks.get("n_mels", 80))

    return {
        # Encoder (Conformer, granite_speech_encoder)
        "enc_n_layers":         int(enc["num_layers"]),
        "enc_hidden":           int(enc["hidden_dim"]),
        "enc_n_heads":          int(enc["num_heads"]),
        "enc_head_dim":         int(enc["dim_head"]),
        "enc_input_dim":        int(enc["input_dim"]),
        "enc_output_dim":       int(enc["output_dim"]),
        "enc_feedforward_mult": int(enc["feedforward_mult"]),
        "enc_conv_kernel_size": int(enc["conv_kernel_size"]),
        "enc_conv_expansion":   int(enc["conv_expansion_factor"]),
        "enc_max_pos_emb":      int(enc["max_pos_emb"]),
        "enc_context_size":     int(enc["context_size"]),
        "enc_cat_hidden":       cat_hidden_layers,

        # Projector (BLIP-2 Q-Former)
        "prj_hidden":              int(prj["hidden_size"]),
        "prj_intermediate":        int(prj["intermediate_size"]),
        "prj_n_heads":             int(prj["num_attention_heads"]),
        "prj_n_layers":            int(prj["num_hidden_layers"]),
        "prj_encoder_hidden_size": int(prj["encoder_hidden_size"]),
        "prj_cross_attn_freq":     int(prj["cross_attention_frequency"]),
        "prj_hidden_act":          str(prj.get("hidden_act", "gelu")).lower(),
        "prj_layer_norm_eps":      float(prj["layer_norm_eps"]),
        "prj_max_pos_emb":         int(prj.get("max_position_embeddings", 2048)),
        "prj_pos_embed_type":      str(prj.get("position_embedding_type", "absolute")),

        # Text LM (Granite-4)
        "dec_n_layers":     int(txt["num_hidden_layers"]),
        "dec_hidden":       int(txt["hidden_size"]),
        "dec_intermediate": int(txt["intermediate_size"]),
        "dec_n_heads":      int(txt["num_attention_heads"]),
        "dec_n_kv_heads":   int(txt["num_key_value_heads"]),
        "dec_head_dim":     int(txt.get("head_dim", txt["hidden_size"] // txt["num_attention_heads"])),
        "dec_hidden_act":   str(txt["hidden_act"]).lower(),
        "dec_rms_norm_eps": float(txt["rms_norm_eps"]),
        "dec_rope_theta":   rope_theta,
        "dec_max_pos_emb":  int(txt["max_position_embeddings"]),
        "dec_vocab_size":   int(txt["vocab_size"]),
        "dec_tie_embeds":   tie,
        # Granite-4 scalar multipliers — silently degrade accuracy if missed.
        "dec_embedding_multiplier":  float(txt["embedding_multiplier"]),
        "dec_logits_scaling":        float(txt["logits_scaling"]),
        "dec_attention_multiplier":  float(txt["attention_multiplier"]),
        "dec_residual_multiplier":   float(txt["residual_multiplier"]),

        # Audio fusion
        "audio_token_id":   int(config["audio_token_index"]),
        "downsample_rate":  int(config["downsample_rate"]),
        "window_size":      int(config["window_size"]),

        # Frontend (torchaudio MelSpectrogram defaults locked at Stage 2)
        "fe_type":         "mel",
        "fe_sample_rate":  sample_rate,
        "fe_num_mels":     n_mels,
        "fe_n_fft":        n_fft,
        "fe_win_length":   win_length,
        "fe_hop_length":   hop_length,
        "fe_window":       "hann_periodic",
        # GraniteSpeechFeatureExtractor applies whisper-style per-utterance
        # normalization: clip(mel, 1e-10).log10() → max-8 floor → /4 + 1.
        # MelFrontend's whisper_mode does the same; we declare it here so
        # the loader takes that path.
        "fe_normalize":    "per_utterance",
        "fe_dither":       0.0,
        "fe_pre_emphasis": 0.0,
        "fe_f_min":        0.0,
        "fe_f_max":        float(sample_rate) / 2.0,
        "fe_pad_mode":     "reflect",
        "fe_center":       True,
        "fe_mel_norm":     "htk",
    }


# ---------------------------------------------------------------------------
# Tensor name mapping
#
# All mappings preserve PyTorch layout (no transpose). Conv1d weights
# carry shape (out_channels, in_channels/groups, kernel_size) which is
# what ggml im2col consumes.
# ---------------------------------------------------------------------------


def passthrough(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr)


# Encoder top-level (5 tensors; out_mid is the self-conditioned CTC
# residual bypass and is included even though the runtime may not need
# CTC outputs at inference — the residual feedback is part of the
# encoder graph).
ENCODER_TOP_TABLE: list[tuple[str, str]] = [
    ("encoder.input_linear.weight", "enc.input_linear.weight"),
    ("encoder.input_linear.bias",   "enc.input_linear.bias"),
    ("encoder.out.weight",          "enc.ctc_proj.weight"),
    ("encoder.out.bias",            "enc.ctc_proj.bias"),
    ("encoder.out_mid.weight",      "enc.ctc_bypass.weight"),
    ("encoder.out_mid.bias",        "enc.ctc_bypass.bias"),
]


# Encoder block tensors. Macaroni FFNs (ff1, ff2) flank the attention
# and conv modules. norm_* naming routes these to F32 via
# gguf_common.reference_dtype_for.
ENCODER_BLOCK_TABLE: list[tuple[str, str]] = [
    # FF1 (first half macaroni)
    ("ff1.pre_norm.weight",    "norm_ff1.weight"),
    ("ff1.pre_norm.bias",      "norm_ff1.bias"),
    ("ff1.up_proj.weight",     "ff1.up.weight"),
    ("ff1.up_proj.bias",       "ff1.up.bias"),
    ("ff1.down_proj.weight",   "ff1.down.weight"),
    ("ff1.down_proj.bias",     "ff1.down.bias"),
    # Multi-head self-attention with Shaw-style relative positional embedding.
    # to_q is separate (1024->1024). to_kv is fused (1024->2*1024) and has
    # no bias upstream. to_out (1024->1024) has bias.
    ("attn.pre_norm.weight",   "norm_attn.weight"),
    ("attn.pre_norm.bias",     "norm_attn.bias"),
    ("attn.to_q.weight",       "attn.q.weight"),
    ("attn.to_kv.weight",      "attn.kv.weight"),
    ("attn.to_out.weight",     "attn.out.weight"),
    ("attn.to_out.bias",       "attn.out.bias"),
    ("attn.rel_pos_emb.weight", "attn.rel_pos_emb.weight"),
    # Conv module (LN -> pointwise expand -> GLU -> depthwise -> BN -> SiLU -> pointwise contract).
    ("conv.norm.weight",       "norm_conv.weight"),
    ("conv.norm.bias",         "norm_conv.bias"),
    ("conv.up_conv.weight",    "conv.pointwise1.weight"),
    ("conv.up_conv.bias",      "conv.pointwise1.bias"),
    ("conv.depth_conv.conv.weight", "conv.depthwise.weight"),
    ("conv.batch_norm.weight",       "conv.bn.weight"),
    ("conv.batch_norm.bias",         "conv.bn.bias"),
    ("conv.batch_norm.running_mean", "conv.bn.running_mean"),
    ("conv.batch_norm.running_var",  "conv.bn.running_var"),
    ("conv.down_conv.weight",  "conv.pointwise2.weight"),
    ("conv.down_conv.bias",    "conv.pointwise2.bias"),
    # FF2 (second half macaroni)
    ("ff2.pre_norm.weight",    "norm_ff2.weight"),
    ("ff2.pre_norm.bias",      "norm_ff2.bias"),
    ("ff2.up_proj.weight",     "ff2.up.weight"),
    ("ff2.up_proj.bias",       "ff2.up.bias"),
    ("ff2.down_proj.weight",   "ff2.down.weight"),
    ("ff2.down_proj.bias",     "ff2.down.bias"),
    # Post-block LayerNorm
    ("post_norm.weight",       "norm_post.weight"),
    ("post_norm.bias",         "norm_post.bias"),
]


# PyTorch bookkeeping that the GGUF does not need.
ENCODER_BLOCK_SKIP_SUFFIXES = {
    "conv.batch_norm.num_batches_tracked",
}


# Projector top-level (linear lift + Q-Former wrapper LN + learned queries).
PROJECTOR_TOP_TABLE: list[tuple[str, str]] = [
    ("projector.query",                    "proj.query"),
    ("projector.linear.weight",            "proj.linear.weight"),
    ("projector.linear.bias",              "proj.linear.bias"),
    ("projector.qformer.layernorm.weight", "proj.qformer.final_norm.weight"),
    ("projector.qformer.layernorm.bias",   "proj.qformer.final_norm.bias"),
]


# Per-Q-Former-layer block. self+cross+ffn+final-LN per layer.
PROJECTOR_BLOCK_TABLE: list[tuple[str, str]] = [
    # Self-attention QKV
    ("attention.attention.query.weight", "self_attn.q.weight"),
    ("attention.attention.query.bias",   "self_attn.q.bias"),
    ("attention.attention.key.weight",   "self_attn.k.weight"),
    ("attention.attention.key.bias",     "self_attn.k.bias"),
    ("attention.attention.value.weight", "self_attn.v.weight"),
    ("attention.attention.value.bias",   "self_attn.v.bias"),
    ("attention.output.dense.weight",    "self_attn.out.weight"),
    ("attention.output.dense.bias",      "self_attn.out.bias"),
    ("attention.output.LayerNorm.weight", "norm_self_attn.weight"),
    ("attention.output.LayerNorm.bias",   "norm_self_attn.bias"),
    # Cross-attention QKV (K/V take encoder_hidden_size input on -plus)
    ("crossattention.attention.query.weight", "cross_attn.q.weight"),
    ("crossattention.attention.query.bias",   "cross_attn.q.bias"),
    ("crossattention.attention.key.weight",   "cross_attn.k.weight"),
    ("crossattention.attention.key.bias",     "cross_attn.k.bias"),
    ("crossattention.attention.value.weight", "cross_attn.v.weight"),
    ("crossattention.attention.value.bias",   "cross_attn.v.bias"),
    ("crossattention.output.dense.weight",    "cross_attn.out.weight"),
    ("crossattention.output.dense.bias",      "cross_attn.out.bias"),
    ("crossattention.output.LayerNorm.weight", "norm_cross_attn.weight"),
    ("crossattention.output.LayerNorm.bias",   "norm_cross_attn.bias"),
    # FFN (BLIP-2 calls these intermediate_query + output_query when the
    # Q-Former is operating on the query stream only — i.e. no separate
    # text path. Granite's Q-Former always runs in query-only mode.)
    ("intermediate_query.dense.weight", "ffn.up.weight"),
    ("intermediate_query.dense.bias",   "ffn.up.bias"),
    ("output_query.dense.weight",       "ffn.down.weight"),
    ("output_query.dense.bias",         "ffn.down.bias"),
    ("output_query.LayerNorm.weight",   "norm_ffn.weight"),
    ("output_query.LayerNorm.bias",     "norm_ffn.bias"),
]


# Text LM top-level. dec.output.weight only exists when the head is not
# tied; we add it conditionally.
TEXT_TOP_TABLE: list[tuple[str, str]] = [
    ("language_model.model.embed_tokens.weight", "dec.token_embd.weight"),
    ("language_model.model.norm.weight",         "dec.output_norm.weight"),
]


TEXT_BLOCK_TABLE: list[tuple[str, str]] = [
    ("input_layernorm.weight",         "norm_attn.weight"),
    ("post_attention_layernorm.weight", "norm_ffn.weight"),
    ("self_attn.q_proj.weight",        "attn.q.weight"),
    ("self_attn.k_proj.weight",        "attn.k.weight"),
    ("self_attn.v_proj.weight",        "attn.v.weight"),
    ("self_attn.o_proj.weight",        "attn.o.weight"),
    ("mlp.gate_proj.weight",           "ffn.gate.weight"),
    ("mlp.up_proj.weight",             "ffn.up.weight"),
    ("mlp.down_proj.weight",           "ffn.down.weight"),
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
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    config_path     = model_dir / "config.json"
    preproc_path    = model_dir / "preprocessor_config.json"
    processor_path  = model_dir / "processor_config.json"
    tpl_jinja       = model_dir / "chat_template.jinja"
    tpl_json        = model_dir / "chat_template.json"
    single_st       = model_dir / "model.safetensors"
    sharded_index   = model_dir / "model.safetensors.index.json"

    if not config_path.is_file():
        raise FileNotFoundError(f"missing required file: {config_path}")
    if not preproc_path.is_file() and not processor_path.is_file():
        # -plus drops preprocessor_config.json and parks the melspec
        # kwargs inside processor_config.json instead. We require at
        # least one of the two.
        raise FileNotFoundError(
            f"missing both {preproc_path.name} and {processor_path.name}; "
            f"need one to read melspec_kwargs")
    if not single_st.is_file() and not sharded_index.is_file():
        raise FileNotFoundError(
            f"missing weights: neither {single_st.name} nor "
            f"{sharded_index.name} present in {model_dir}")

    with config_path.open() as f:
        config = json.load(f)
    preproc: dict = {}
    if preproc_path.is_file():
        with preproc_path.open() as f:
            preproc = json.load(f)
    processor: dict = {}
    if processor_path.is_file():
        with processor_path.open() as f:
            processor = json.load(f)

    chat_template: str | None = None
    if tpl_jinja.is_file():
        chat_template = tpl_jinja.read_text(encoding="utf-8")
    elif tpl_json.is_file():
        with tpl_json.open() as f:
            chat_template = json.load(f).get("chat_template")

    hp = read_hparams(config, preproc, processor)

    print(f"Encoder: {hp['enc_n_layers']} layers, hidden={hp['enc_hidden']}, "
          f"heads={hp['enc_n_heads']}, cat_hidden_layers={hp['enc_cat_hidden']}")
    print(f"Projector: {hp['prj_n_layers']} layers Q-Former, hidden={hp['prj_hidden']}, "
          f"encoder_hidden_size={hp['prj_encoder_hidden_size']}")
    print(f"Text LM: {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, "
          f"heads={hp['dec_n_heads']}/{hp['dec_n_kv_heads']}, "
          f"tie_word_embeddings={hp['dec_tie_embeds']}")
    print(f"Multipliers: emb={hp['dec_embedding_multiplier']}, "
          f"logits={hp['dec_logits_scaling']}, "
          f"attn={hp['dec_attention_multiplier']}, "
          f"residual={hp['dec_residual_multiplier']}")

    print(f"Reading tokenizer from {model_dir}")
    tok = extract_tokenizer(model_dir, hp["dec_vocab_size"])

    # Sanity-check: the audio token id from the outer config should
    # exist as a CONTROL token in the resolved tokenizer (it lives in
    # added_tokens at id 100352 with special=True).
    audio_idx = hp["audio_token_id"]
    if audio_idx >= len(tok["tokens"]):
        raise ValueError(
            f"audio_token_index={audio_idx} >= vocab size {len(tok['tokens'])}"
        )
    if tok["types"][audio_idx] != TOKEN_TYPE_CONTROL:
        raise ValueError(
            f"audio_token id={audio_idx} ({tok['tokens'][audio_idx]!r}) "
            f"is not flagged as a special/control token")

    print(f"Opening safetensors (single or sharded) under {model_dir}")
    with _ShardedSafetensors(model_dir) as st:
        st_keys = set(st.keys())

        languages = LANG_BY_VARIANT.get(variant, [])
        if not languages:
            print(f"WARNING: no language list registered for variant {variant!r}; "
                  f"emitting empty general.languages", file=sys.stderr)

        # Count params for the size label. Exclude the absent lm_head
        # automatically (tied head case) and skip bookkeeping tensors.
        total = 0
        for k in st_keys:
            if any(k.endswith(s) for s in ENCODER_BLOCK_SKIP_SUFFIXES):
                continue
            total += st.get_tensor(k).numel()
        size_label = compute_size_label(total)
        print(f"Total params (BF16 floating + I64 bookkeeping skipped): "
              f"{total:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = gguf_writer(str(out_path), "granite_speech")

        # ---- general.* ----
        add_general_identity(
            writer,
            name={
                "granite-4.0-1b-speech":      "Granite Speech 4.0 1B",
                "granite-speech-4.1-2b":      "Granite Speech 4.1 2B",
                "granite-speech-4.1-2b-plus": "Granite Speech 4.1 2B Plus",
            }[variant],
            basename="granite-speech",
            size_label=size_label,
            file_type=REFERENCE_FILE_TYPE,
            languages=(languages if languages else None),
            author="IBM",
            organization="ibm-granite",
            license="apache-2.0",
            license_name="Apache License 2.0",
            license_link="https://www.apache.org/licenses/LICENSE-2.0",
            repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
        )

        # ---- stt.variant ----
        writer.add_string("stt.variant", variant)

        # ---- stt.capability.* ----
        # 1b / 2b advertise translation to/from English for the ASR languages,
        # plus English-to-Italian and English-to-Mandarin. 2b-plus narrows to
        # ASR only. Language detection is not exposed by Granite (the chat
        # template selects language explicitly).
        translation_caps = {
            "granite-4.0-1b-speech":      True,
            "granite-speech-4.1-2b":      True,
            "granite-speech-4.1-2b-plus": False,
        }
        can_translate = bool(translation_caps.get(variant, False))
        writer.add_bool("stt.capability.translate",
                        can_translate)
        writer.add_bool("stt.capability.lang_detect", False)
        if can_translate:
            writer.add_array("stt.translation.target_languages",
                             TRANSLATION_TARGET_LANG_BY_VARIANT[variant])
            writer.add_array("stt.translation.pairs",
                             granite_translation_pairs(languages))
        # -plus is the only variant exposing word timestamps and
        # speaker diarization (per its model card).
        writer.add_bool("stt.capability.word_timestamps",
                        variant == "granite-speech-4.1-2b-plus")
        writer.add_bool("stt.capability.speaker_diarization",
                        variant == "granite-speech-4.1-2b-plus")

        # ---- tokenizer.ggml.* (llama.cpp "gpt2" byte-level BPE) ----
        #
        # Pretokenizer flavor differs across granite-speech variants:
        #   - granite-4.0-1b-speech / granite-speech-4.1-2b  use a custom
        #     Split-then-ByteLevel sequence with a \p{N}{1,3} digit rule
        #     (the "granite" flavor in transcribe::unicode::pretokenize_granite).
        #   - granite-speech-4.1-2b-plus uses plain ByteLevel(use_regex=true),
        #     i.e. the standard GPT-2 split (\p{N}+ unlimited-digit), so
        #     "2024" stays one pretoken and BPE merges to ['20','24'] not
        #     ['202','4']. The C++ tokenizer pretokenize_gpt2 path handles
        #     that — we just have to declare it correctly in the GGUF.
        writer.add_string("tokenizer.ggml.model", "gpt2")
        pre_flavor = "gpt2" if variant == "granite-speech-4.1-2b-plus" else "granite"
        writer.add_string("tokenizer.ggml.pre",   pre_flavor)
        writer.add_array("tokenizer.ggml.tokens",     tok["tokens"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        writer.add_array("tokenizer.ggml.merges",     tok["merges"])
        if tok["bos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
        if tok["eos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
        if tok["pad_id"] is not None:
            writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])
        if tok["unk_id"] is not None:
            writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
        writer.add_bool("tokenizer.ggml.add_bos_token", False)
        writer.add_bool("tokenizer.ggml.add_eos_token", False)
        if chat_template is not None:
            writer.add_string("tokenizer.chat_template", chat_template)

        # ---- stt.granite.encoder.* ----
        writer.add_uint32("stt.granite.encoder.n_layers",         hp["enc_n_layers"])
        writer.add_uint32("stt.granite.encoder.hidden",           hp["enc_hidden"])
        writer.add_uint32("stt.granite.encoder.n_heads",          hp["enc_n_heads"])
        writer.add_uint32("stt.granite.encoder.head_dim",         hp["enc_head_dim"])
        writer.add_uint32("stt.granite.encoder.input_dim",        hp["enc_input_dim"])
        writer.add_uint32("stt.granite.encoder.output_dim",       hp["enc_output_dim"])
        writer.add_uint32("stt.granite.encoder.feedforward_mult", hp["enc_feedforward_mult"])
        writer.add_uint32("stt.granite.encoder.conv_kernel_size", hp["enc_conv_kernel_size"])
        writer.add_uint32("stt.granite.encoder.conv_expansion",   hp["enc_conv_expansion"])
        writer.add_uint32("stt.granite.encoder.max_pos_emb",      hp["enc_max_pos_emb"])
        writer.add_uint32("stt.granite.encoder.context_size",     hp["enc_context_size"])
        # cat_hidden_layers is a list. Emit as a uint32 array (empty
        # arrays are legal in GGUF). The loader treats an empty list as
        # "no concat; projector input == encoder hidden".
        writer.add_array("stt.granite.encoder.cat_hidden_layers",
                         [int(x) for x in hp["enc_cat_hidden"]])

        # ---- stt.granite.projector.* ----
        writer.add_uint32("stt.granite.projector.n_layers",            hp["prj_n_layers"])
        writer.add_uint32("stt.granite.projector.hidden",              hp["prj_hidden"])
        writer.add_uint32("stt.granite.projector.intermediate",        hp["prj_intermediate"])
        writer.add_uint32("stt.granite.projector.n_heads",             hp["prj_n_heads"])
        writer.add_uint32("stt.granite.projector.encoder_hidden_size", hp["prj_encoder_hidden_size"])
        writer.add_uint32("stt.granite.projector.cross_attn_frequency", hp["prj_cross_attn_freq"])
        writer.add_string("stt.granite.projector.hidden_act",          hp["prj_hidden_act"])
        writer.add_float32("stt.granite.projector.layer_norm_eps",     hp["prj_layer_norm_eps"])
        writer.add_uint32("stt.granite.projector.max_pos_emb",         hp["prj_max_pos_emb"])
        writer.add_string("stt.granite.projector.position_embedding_type",
                          hp["prj_pos_embed_type"])

        # ---- stt.granite.decoder.* (Granite-4 LM) ----
        writer.add_uint32("stt.granite.decoder.n_layers",       hp["dec_n_layers"])
        writer.add_uint32("stt.granite.decoder.hidden_size",    hp["dec_hidden"])
        writer.add_uint32("stt.granite.decoder.intermediate_size", hp["dec_intermediate"])
        writer.add_uint32("stt.granite.decoder.n_heads",        hp["dec_n_heads"])
        writer.add_uint32("stt.granite.decoder.n_kv_heads",     hp["dec_n_kv_heads"])
        writer.add_uint32("stt.granite.decoder.head_dim",       hp["dec_head_dim"])
        writer.add_string("stt.granite.decoder.hidden_act",     hp["dec_hidden_act"])
        writer.add_float32("stt.granite.decoder.rms_norm_eps",  hp["dec_rms_norm_eps"])
        writer.add_float32("stt.granite.decoder.rope_theta",    hp["dec_rope_theta"])
        writer.add_uint32("stt.granite.decoder.max_position_embeddings", hp["dec_max_pos_emb"])
        writer.add_bool("stt.granite.decoder.tie_word_embeddings", hp["dec_tie_embeds"])
        writer.add_uint32("stt.granite.decoder.vocab_size",     hp["dec_vocab_size"])
        # Granite-4 scalar multipliers — must be baked into the graph.
        writer.add_float32("stt.granite.decoder.embedding_multiplier",
                           hp["dec_embedding_multiplier"])
        writer.add_float32("stt.granite.decoder.logits_scaling",
                           hp["dec_logits_scaling"])
        writer.add_float32("stt.granite.decoder.attention_multiplier",
                           hp["dec_attention_multiplier"])
        writer.add_float32("stt.granite.decoder.residual_multiplier",
                           hp["dec_residual_multiplier"])

        # ---- audio fusion ----
        writer.add_uint32("stt.granite.audio_token_id",  hp["audio_token_id"])
        writer.add_uint32("stt.granite.downsample_rate", hp["downsample_rate"])
        writer.add_uint32("stt.granite.window_size",     hp["window_size"])

        # ---- stt.frontend.* (torchaudio MelSpectrogram) ----
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

        # ---- tensors ----
        n_added   = 0
        bytes_in  = 0
        bytes_out = 0

        # ---- frontend buffers (filterbank + window) ----
        import librosa as _lb
        mel_fb = _lb.filters.mel(
            sr=hp["fe_sample_rate"],
            n_fft=hp["fe_n_fft"],
            n_mels=hp["fe_num_mels"],
            fmin=hp["fe_f_min"],
            fmax=hp["fe_f_max"],
            norm=None,    # torchaudio MelSpectrogram default: no norm
            htk=True,     # torchaudio default mel scale
        ).astype(np.float32)
        writer.add_tensor(
            "frontend.mel_filterbank",
            np.ascontiguousarray(mel_fb),
            raw_dtype=GGMLQuantizationType.F32,
        )
        # torchaudio's default Hann window is periodic (denominator = N).
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

        # Encoder
        for src, dst in ENCODER_TOP_TABLE:
            add(src, dst)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, suffix_dst in ENCODER_BLOCK_TABLE:
                add(
                    f"encoder.layers.{i}.{suffix_src}",
                    f"enc.blocks.{i}.{suffix_dst}",
                )

        # Projector
        for src, dst in PROJECTOR_TOP_TABLE:
            add(src, dst)
        for i in range(hp["prj_n_layers"]):
            for suffix_src, suffix_dst in PROJECTOR_BLOCK_TABLE:
                add(
                    f"projector.qformer.encoder.layer.{i}.{suffix_src}",
                    f"proj.qformer.blocks.{i}.{suffix_dst}",
                )

        # Text LM
        for src, dst in TEXT_TOP_TABLE:
            add(src, dst)
        if not hp["dec_tie_embeds"]:
            # 1b / 2b: a distinct lm_head is shipped. -plus ties the head
            # to embed_tokens and the safetensors omits lm_head entirely.
            add("language_model.lm_head.weight", "dec.output.weight")
        for i in range(hp["dec_n_layers"]):
            for suffix_src, suffix_dst in TEXT_BLOCK_TABLE:
                add(
                    f"language_model.model.layers.{i}.{suffix_src}",
                    f"dec.blocks.{i}.{suffix_dst}",
                )

        expected = (
            len(ENCODER_TOP_TABLE)
            + hp["enc_n_layers"] * len(ENCODER_BLOCK_TABLE)
            + len(PROJECTOR_TOP_TABLE)
            + hp["prj_n_layers"] * len(PROJECTOR_BLOCK_TABLE)
            + len(TEXT_TOP_TABLE)
            + (0 if hp["dec_tie_embeds"] else 1)
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

        # Warn about unconsumed safetensors keys (excluding the explicit
        # skip list).
        consumed: set[str] = set()
        for src, _ in ENCODER_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, _ in ENCODER_BLOCK_TABLE:
                consumed.add(f"encoder.layers.{i}.{suffix_src}")
            for suffix in ENCODER_BLOCK_SKIP_SUFFIXES:
                consumed.add(f"encoder.layers.{i}.{suffix}")
        for src, _ in PROJECTOR_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["prj_n_layers"]):
            for suffix_src, _ in PROJECTOR_BLOCK_TABLE:
                consumed.add(
                    f"projector.qformer.encoder.layer.{i}.{suffix_src}"
                )
        for src, _ in TEXT_TOP_TABLE:
            consumed.add(src)
        if not hp["dec_tie_embeds"]:
            consumed.add("language_model.lm_head.weight")
        for i in range(hp["dec_n_layers"]):
            for suffix_src, _ in TEXT_BLOCK_TABLE:
                consumed.add(f"language_model.model.layers.{i}.{suffix_src}")
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
        description="Convert a Granite Speech checkpoint to a BF16 accuracy GGUF.",
    )
    p.add_argument("model", type=str,
                   help="HF repo id (e.g. ibm-granite/granite-4.0-1b-speech) "
                        "or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug "
                        "when converting from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch/tag/commit SHA) to pin the "
                        "download. Recommended for reproducibility.")
    p.add_argument("--variant", type=str, default=None,
                   help="stt.variant string (default: derived from slug)")
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
        out_path = REPO_ROOT / "models" / slug / gguf_name(slug, REFERENCE_DTYPE_LABEL)
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
