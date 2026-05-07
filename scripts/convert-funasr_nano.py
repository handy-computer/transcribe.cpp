#!/usr/bin/env python3
"""
convert-funasr_nano.py - convert a FunASR Fun-ASR-Nano checkpoint to a
GGUF that transcribe.cpp's loader can ingest.

Source format (FunASR-native checkpoint directory, HF or local):

    config.yaml              FunASR YAML config (model class FunASRNano)
    configuration.json       ModelScope-style entry pointer
    model.pt                 single torch.save() dict wrapping a 1261-key
                              state_dict (uniform BF16 across encoder,
                              audio_adaptor, and LLM).
    multilingual.tiktoken    Whisper-style tiktoken (CTC head only —
                              ignored at inference)
    Qwen3-0.6B/              tokenizer + Qwen3 LLM config files
        config.json
        generation_config.json
        merges.txt
        tokenizer.json
        tokenizer_config.json
        vocab.json

Architecture: audio-llm.

    audio (16 kHz)
      -> WavFrontend  (kaldi fbank 80 mels, 25/10 ms, hamming + LFR(7,6).
                        NO CMVN — Fun-ASR-Nano was trained on raw LFR
                        features.)
      -> SenseVoiceEncoderSmall  (50 SAN-M blocks + 20 tp_blocks; same
                                   architecture as sensevoice-small but
                                   independently fine-tuned. NO prefix-
                                   token embed table.)
                              => [T_lfr, 512]
      -> Audio adaptor (Transformer):
           linear1 (512 -> 2048) -> ReLU -> linear2 (2048 -> 1024)
           blocks[0..1] : MultiHeadedAttention (8 heads, head_dim=128) +
                          PositionwiseFeedForward (1024 -> 256 -> 1024)
                          with eps=1e-12 LayerNorms
                              => [T_lfr, 1024]
      -> Qwen3-0.6B LLM:
           audio embeddings spliced into the chat-template prompt at the
           <|startofspeech|>!!<|endofspeech|> position; LLM autoregressively
           generates the response.

The CTC head (`ctc_decoder.*` and `ctc.*` in the published config) is
NOT shipped with the released checkpoint — Layers 2..4 of `ctc_decoder`
and `ctc.ctc_lo.*` are missing from `model.pt`, and the partial CTC head
produces gibberish at inference. We treat the CTC path as dead and skip
it entirely.

KV emitted:

    general.architecture   = "funasr_nano"
    general.basename       = "fun-asr-nano"
    general.size_label     = computed from total params
    general.languages      = ["zh", "en", "ja"]

    stt.variant                       = "fun-asr-nano-2512"
    stt.funasr_nano.encoder.{n_blocks,tp_blocks,d_model,d_input,n_heads,
                              d_ff,kernel_size,sanm_shift,attention_type,
                              normalize_before}
    stt.funasr_nano.adaptor.{n_blocks,encoder_dim,llm_dim,ffn_dim,
                              n_heads,d_head,layer_norm_eps,activation}
    stt.funasr_nano.decoder.{n_layers,hidden_size,intermediate_size,
                              n_heads,n_kv_heads,head_dim,vocab_size,
                              max_position_embeddings,rms_norm_eps,
                              rope_theta,tie_word_embeddings}

    stt.frontend.{type,num_mels,sample_rate,n_fft,win_length,hop_length,
                  window,normalize,dither,upscale_samples,snip_edges,
                  lfr_m,lfr_n,fbank_style,apply_cmvn}

    tokenizer.ggml.model              = "gpt2"
    tokenizer.ggml.tokens             = full Qwen3 BPE vocabulary
    tokenizer.ggml.token_type
    tokenizer.ggml.merges
    tokenizer.ggml.bos_token_id / eos_token_id / padding_token_id
    tokenizer.chat_template            = Qwen3 Jinja chat template

CLI:

    uv run --project scripts/envs/funasr_nano \
      scripts/convert-funasr_nano.py FunAudioLLM/Fun-ASR-Nano-2512

Single-file, top-to-bottom — no hidden helpers.
"""

from __future__ import annotations

import argparse
import collections
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    encode_for_gguf,
    gguf_name,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reference dtype labels — the published checkpoint is uniform BF16 across
# encoder, adaptor, and LLM. The on-disk GGUF preserves BF16 for matrices
# / embeddings and routes biases and norm scales to F32 (loader requires
# F32 for those slots; matches the quantize policy's Norm bucket).
REFERENCE_DTYPE_LABEL = "BF16"
REFERENCE_FILE_TYPE = LlamaFileType.MOSTLY_BF16
REFERENCE_GGML_TYPE = GGMLQuantizationType.BF16


# ---------------------------------------------------------------------------
# fun_asr_nano import shim (FunASR 1.3.1 ships broken absolute imports)
# ---------------------------------------------------------------------------
#
# funasr/models/fun_asr_nano/model.py contains:
#     from ctc import CTC
#     from tools.utils import forced_align
# These should be relative imports. The funasr package's __init__ walks
# submodules and silently swallows ImportError, so the registration
# decorator never runs and AutoModel raises "FunASRNano is not registered".
# We don't actually need AutoModel here (we read state_dict directly), but
# having the shim ensures any later helper that imports fun_asr_nano.model
# resolves cleanly. Also, importing funasr is cheap and keeps the env
# healthy across other tools (e.g. dump_reference_funasr_nano_funasr.py).
def _patch_fun_asr_nano_imports() -> None:
    os.environ.setdefault("FUNASR_DISABLE_UPDATE", "1")
    try:
        import funasr  # noqa: F401
        sub_ctc = importlib.import_module("funasr.models.fun_asr_nano.ctc")
        sys.modules.setdefault("ctc", sub_ctc)
        sub_tools_pkg = importlib.import_module("funasr.models.fun_asr_nano.tools")
        sub_tools_utils = importlib.import_module("funasr.models.fun_asr_nano.tools.utils")
        sys.modules.setdefault("tools", sub_tools_pkg)
        sys.modules["tools.utils"] = sub_tools_utils
        importlib.import_module("funasr.models.fun_asr_nano.model")
    except Exception as e:
        # Non-fatal — this converter doesn't strictly need the model class.
        print(f"WARN: fun_asr_nano import shim failed: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Per-block tensor mapping (encoder)
# ---------------------------------------------------------------------------
# Same layout as sensevoice-small: 13 tensors per SAN-M block, applied to
# encoders0[0] (560-dim input projection block) and the 49 main encoders +
# 20 tp_encoders at 512-dim.

ENC_BLOCK_TABLE: list[tuple[str, str]] = [
    ("self_attn.linear_q_k_v.weight",  "attn.qkv.weight"),
    ("self_attn.linear_q_k_v.bias",    "attn.qkv.bias"),
    ("self_attn.linear_out.weight",    "attn.out.weight"),
    ("self_attn.linear_out.bias",      "attn.out.bias"),
    ("self_attn.fsmn_block.weight",    "attn.fsmn.weight"),
    ("feed_forward.w_1.weight",        "ffn.fc1.weight"),
    ("feed_forward.w_1.bias",          "ffn.fc1.bias"),
    ("feed_forward.w_2.weight",        "ffn.fc2.weight"),
    ("feed_forward.w_2.bias",          "ffn.fc2.bias"),
    ("norm1.weight",                   "norm_attn.weight"),
    ("norm1.bias",                     "norm_attn.bias"),
    ("norm2.weight",                   "norm_ffn.weight"),
    ("norm2.bias",                     "norm_ffn.bias"),
]


# ---------------------------------------------------------------------------
# Per-block tensor mapping (adaptor)
# ---------------------------------------------------------------------------
# 18 tensors per adaptor block: separate q/k/v projections, output proj,
# bottleneck FFN (1024 -> 256 -> 1024), pre-attn + pre-ffn LayerNorms.

ADAPTOR_BLOCK_TABLE: list[tuple[str, str]] = [
    ("self_attn.linear_q.weight",    "attn.q.weight"),
    ("self_attn.linear_q.bias",      "attn.q.bias"),
    ("self_attn.linear_k.weight",    "attn.k.weight"),
    ("self_attn.linear_k.bias",      "attn.k.bias"),
    ("self_attn.linear_v.weight",    "attn.v.weight"),
    ("self_attn.linear_v.bias",      "attn.v.bias"),
    ("self_attn.linear_out.weight",  "attn.out.weight"),
    ("self_attn.linear_out.bias",    "attn.out.bias"),
    ("feed_forward.w_1.weight",      "ffn.fc1.weight"),
    ("feed_forward.w_1.bias",        "ffn.fc1.bias"),
    ("feed_forward.w_2.weight",      "ffn.fc2.weight"),
    ("feed_forward.w_2.bias",        "ffn.fc2.bias"),
    ("norm1.weight",                 "norm_attn.weight"),
    ("norm1.bias",                   "norm_attn.bias"),
    ("norm2.weight",                 "norm_ffn.weight"),
    ("norm2.bias",                   "norm_ffn.bias"),
]


# ---------------------------------------------------------------------------
# Per-layer tensor mapping (LLM, Qwen3)
# ---------------------------------------------------------------------------
# 11 tensors per Qwen3 transformer layer. Qwen3 has q_norm/k_norm but no
# QKV biases; o_proj has no bias either (`attention_bias=false`).

LLM_TOP_TABLE: list[tuple[str, str]] = [
    ("llm.model.embed_tokens.weight", "dec.token_embd.weight"),
    ("llm.model.norm.weight",         "dec.output_norm.weight"),
]

LLM_BLOCK_TABLE: list[tuple[str, str]] = [
    ("input_layernorm.weight",          "norm_attn.weight"),
    ("post_attention_layernorm.weight", "norm_ffn.weight"),
    ("self_attn.q_proj.weight",         "attn.q.weight"),
    ("self_attn.k_proj.weight",         "attn.k.weight"),
    ("self_attn.v_proj.weight",         "attn.v.weight"),
    ("self_attn.o_proj.weight",         "attn.o.weight"),
    ("self_attn.q_norm.weight",         "attn.q_norm.weight"),
    ("self_attn.k_norm.weight",         "attn.k_norm.weight"),
    ("mlp.gate_proj.weight",            "ffn.gate.weight"),
    ("mlp.up_proj.weight",              "ffn.up.weight"),
    ("mlp.down_proj.weight",            "ffn.down.weight"),
]

# State_dict keys we deliberately drop:
#   - llm.lm_head.weight: bitwise tied to llm.model.embed_tokens.weight
#     (verified at convert time); store one copy under dec.token_embd.weight
#     and let the loader reuse it for the LM-head projection (llama.cpp
#     TENSOR_DUPLICATED convention).
#   - ctc_decoder.* / ctc.*: dead in the published checkpoint. The state
#     contains ctc_decoder.blocks.{0,1} only (5 layers configured) and
#     ctc.ctc_lo is entirely absent; the auxiliary CTC head produces
#     gibberish at inference. Stage 4 should NOT implement the CTC head.
SKIP_PREFIXES: tuple[str, ...] = ("ctc_decoder.", "ctc.")
SKIP_EXACT: set[str] = {"llm.lm_head.weight"}


# ---------------------------------------------------------------------------
# Per-tensor target dtype (mixed-precision GGUF)
# ---------------------------------------------------------------------------


def per_tensor_target_dtype(name: str) -> GGMLQuantizationType:
    """Decide the GGUF storage dtype for a single tensor.

    Rules (must stay in sync with tools/transcribe-quantize/policy.cpp's
    Norm bucket for this family):
      - .bias                                                   -> F32
      - norm_attn.weight, norm_ffn.weight                       -> F32
        (encoder + adaptor LayerNorm scales)
      - .after_norm.weight, .tp_norm.weight                     -> F32
        (encoder trailing LayerNorms)
      - dec.output_norm.weight                                  -> F32
        (Qwen3 final RMSNorm)
      - dec.layers.<i>.attn.q_norm.weight / k_norm.weight       -> F32
        (per-head Qwen3 RMSNorm)
      - everything else                                         -> BF16
        (preserves source dtype — matrices and embeddings)
    """
    if name.endswith(".bias"):
        return GGMLQuantizationType.F32
    if name.endswith(".norm_attn.weight") or name.endswith(".norm_ffn.weight"):
        return GGMLQuantizationType.F32
    if name.endswith(".after_norm.weight") or name.endswith(".tp_norm.weight"):
        return GGMLQuantizationType.F32
    if name.endswith(".output_norm.weight"):
        return GGMLQuantizationType.F32
    if name.endswith(".q_norm.weight") or name.endswith(".k_norm.weight"):
        return GGMLQuantizationType.F32
    # FSMN depthwise conv1d (3D weight). The C++ loader's CONV bucket
    # only accepts F32/F16; BF16 conv is not in the supported types
    # (matches sensevoice's policy). Upcast at convert time — the
    # storage cost is single-digit MB total across all 70 SAN-M blocks.
    if name.endswith(".attn.fsmn.weight"):
        return GGMLQuantizationType.F32
    return REFERENCE_GGML_TYPE


# ---------------------------------------------------------------------------
# Tokenizer extraction (Qwen3 byte-level BPE — same shape as qwen3_asr)
# ---------------------------------------------------------------------------


def extract_tokenizer(qwen_dir: Path, vocab_size: int) -> dict:
    vocab_path  = qwen_dir / "vocab.json"
    merges_path = qwen_dir / "merges.txt"
    tokcfg_path = qwen_dir / "tokenizer_config.json"

    with vocab_path.open(encoding="utf-8") as f:
        base_vocab: dict[str, int] = json.load(f)
    with merges_path.open(encoding="utf-8") as f:
        merges = [line.rstrip("\n") for line in f.readlines()]
        if merges and merges[0].startswith("#"):
            merges = merges[1:]
    with tokcfg_path.open(encoding="utf-8") as f:
        tokcfg = json.load(f)

    added = tokcfg.get("added_tokens_decoder", {}) or {}
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

    chat_template = tokcfg.get("chat_template")

    return {
        "tokens":         tokens,
        "types":          types,
        "merges":         merges,
        "bos_id":         _name_to_id("bos_token"),
        "eos_id":         _name_to_id("eos_token"),
        "pad_id":         _name_to_id("pad_token"),
        "chat_template":  chat_template,
    }


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


# Per-variant language coverage advertised by the publisher's model card.
# Stage 1 intake is the source of truth; this table mirrors it for the
# converter's `general.languages` KV. Add new sibling variants here.
VARIANT_LANGUAGES: dict[str, list[str]] = {
    "fun-asr-nano-2512":     ["zh", "en", "ja"],
    "fun-asr-mlt-nano-2512": [
        "zh", "en", "yue", "ja", "ko",
        "vi", "id", "th", "ms", "tl",
        "ar", "hi", "bg", "hr", "cs",
        "da", "nl", "et", "fi", "el",
        "hu", "ga", "lv", "lt", "mt",
        "pl", "pt", "ro", "sk", "sl", "sv",
    ],
}


def read_hparams(yaml_config: dict, qwen_config: dict, variant: str) -> dict:
    enc = yaml_config["audio_encoder_conf"]
    adp = yaml_config["audio_adaptor_conf"]
    fe  = yaml_config["frontend_conf"]

    n_mels = int(fe["n_mels"])
    lfr_m  = int(fe["lfr_m"])
    d_input = n_mels * lfr_m  # 80 * 7 = 560

    sample_rate     = int(fe["fs"])
    frame_length_ms = int(fe["frame_length"])
    frame_shift_ms  = int(fe["frame_shift"])
    win_length      = int(round(frame_length_ms * sample_rate / 1000))  # 400
    hop_length      = int(round(frame_shift_ms  * sample_rate / 1000))  # 160

    # Adaptor — config.yaml only declares class + dims; per-layer hparams
    # (heads, ffn_dim, eps) are class-defaults from
    # funasr.models.llm_asr.adaptor.Transformer. We bake the observed
    # values from the loaded checkpoint at write time below.
    return {
        # Encoder
        "enc_n_blocks":         int(enc["num_blocks"]),
        "enc_tp_blocks":        int(enc["tp_blocks"]),
        "enc_d_model":          int(enc["output_size"]),
        "enc_d_input":          d_input,
        "enc_n_heads":          int(enc["attention_heads"]),
        "enc_d_ff":             int(enc["linear_units"]),
        "enc_kernel_size":      int(enc["kernel_size"]),
        "enc_sanm_shift":       int(enc.get("sanm_shfit", 0)),
        "enc_attn_type":        str(enc["selfattention_layer_type"]),
        "enc_normalize_before": bool(enc.get("normalize_before", True)),

        # Adaptor (declared in YAML; remaining fields filled at write time)
        "adp_n_blocks":            int(adp.get("n_layer", 2)),
        "adp_encoder_dim":         int(adp.get("encoder_dim", 512)),
        "adp_llm_dim":             int(adp.get("llm_dim", 1024)),
        "adp_pre_ffn_dim":         int(adp.get("ffn_dim", 2048)),
        "adp_downsample_rate":     int(adp.get("downsample_rate", 1)),
        # When use_low_frame_rate=true, FunASR's data_load_speech computes
        # `fake_token_len = ((((T-1)/2 + 1) - 1)/2 + 1) / 2 + 1` (three
        # stride-2 stages) and the inference path slices only the first
        # `fake_token_len` frames of adaptor_out into the LLM prompt — the
        # rest of adaptor_out is silently dropped. The C++ runtime must
        # mirror this exactly; the K KV signals which formula to use.
        "adp_use_low_frame_rate":  bool(adp.get("use_low_frame_rate", False)),

        # Decoder (LLM, Qwen3-0.6B)
        "dec_n_layers":         int(qwen_config["num_hidden_layers"]),
        "dec_hidden":           int(qwen_config["hidden_size"]),
        "dec_intermediate":     int(qwen_config["intermediate_size"]),
        "dec_n_heads":          int(qwen_config["num_attention_heads"]),
        "dec_n_kv_heads":       int(qwen_config["num_key_value_heads"]),
        "dec_head_dim":         int(qwen_config["head_dim"]),
        "dec_vocab_size":       int(qwen_config["vocab_size"]),
        "dec_max_pos":          int(qwen_config["max_position_embeddings"]),
        "dec_rms_norm_eps":     float(qwen_config["rms_norm_eps"]),
        "dec_rope_theta":       float(qwen_config["rope_theta"]),
        "dec_tie_embeddings":   bool(qwen_config.get("tie_word_embeddings", True)),
        "dec_hidden_act":       str(qwen_config.get("hidden_act", "silu")),

        # Frontend (kaldi-style fbank + LFR; NO CMVN for funasr_nano)
        "fe_type":              "kaldi_fbank_lfr",
        "fe_sample_rate":       sample_rate,
        "fe_num_mels":          n_mels,
        "fe_win_length":        win_length,
        "fe_n_fft":             win_length,
        "fe_hop_length":        hop_length,
        "fe_window":            str(fe["window"]),
        "fe_normalize":         "none",
        "fe_dither":            0.0,
        "fe_upscale_samples":   True,
        "fe_snip_edges":        True,
        "fe_lfr_m":             lfr_m,
        "fe_lfr_n":             int(fe["lfr_n"]),
        "fe_fbank_style":       "kaldi_htk",
        "fe_apply_cmvn":        False,

        "languages":            VARIANT_LANGUAGES.get(variant, ["zh", "en", "ja"]),
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


def convert(model_dir: Path, out_path: Path, variant: str, display_name: str) -> None:
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (per-tensor BF16; norms/biases F32)")
    _patch_fun_asr_nano_imports()

    config_yaml = model_dir / "config.yaml"
    model_pt    = model_dir / "model.pt"
    qwen_dir    = model_dir / "Qwen3-0.6B"

    for p in (config_yaml, model_pt, qwen_dir):
        if not p.exists():
            raise FileNotFoundError(f"missing required path: {p}")
    if not qwen_dir.is_dir():
        raise NotADirectoryError(f"expected Qwen3-0.6B directory: {qwen_dir}")

    qwen_config_path = qwen_dir / "config.json"
    if not qwen_config_path.is_file():
        raise FileNotFoundError(f"missing Qwen3 config: {qwen_config_path}")

    print(f"Reading FunASR config from {config_yaml}")
    with config_yaml.open() as f:
        yaml_config = yaml.safe_load(f)
    print(f"Reading Qwen3 config from {qwen_config_path}")
    with qwen_config_path.open() as f:
        qwen_config = json.load(f)

    hp = read_hparams(yaml_config, qwen_config, variant)
    print(
        f"Encoder: {hp['enc_n_blocks']} main + {hp['enc_tp_blocks']} tp blocks; "
        f"d_model={hp['enc_d_model']} d_input={hp['enc_d_input']} "
        f"heads={hp['enc_n_heads']} d_ff={hp['enc_d_ff']} "
        f"kernel={hp['enc_kernel_size']}"
    )
    print(
        f"Adaptor: {hp['adp_n_blocks']} blocks; encoder_dim={hp['adp_encoder_dim']} "
        f"llm_dim={hp['adp_llm_dim']}"
    )
    print(
        f"Decoder (LLM): n_layers={hp['dec_n_layers']} hidden={hp['dec_hidden']} "
        f"heads={hp['dec_n_heads']}/{hp['dec_n_kv_heads']} "
        f"head_dim={hp['dec_head_dim']} vocab={hp['dec_vocab_size']} "
        f"tied={hp['dec_tie_embeddings']}"
    )

    print(f"Reading tokenizer from {qwen_dir}")
    tok = extract_tokenizer(qwen_dir, hp["dec_vocab_size"])
    print(f"Qwen3 vocab: {len(tok['tokens'])}; merges: {len(tok['merges'])}")

    print(f"Loading state_dict from {model_pt}")
    sd_outer = torch.load(str(model_pt), map_location="cpu", weights_only=False)
    if not isinstance(sd_outer, dict) or "state_dict" not in sd_outer:
        raise RuntimeError(
            f"unexpected model.pt structure: top-level keys "
            f"{list(sd_outer)[:5] if isinstance(sd_outer, dict) else type(sd_outer).__name__}"
        )
    sd = sd_outer["state_dict"]

    # Confirm dtype (uniform BF16 across encoder/adaptor/llm).
    dtypes = collections.Counter(v.dtype for v in sd.values())
    if dtypes != collections.Counter([torch.bfloat16] * sum(dtypes.values())):
        raise RuntimeError(
            f"unexpected weight dtypes in model.pt: {dict(dtypes)!r}; "
            "this converter only handles the published BF16 Fun-ASR-Nano "
            "checkpoint"
        )

    # Verify lm_head is bitwise tied to embed_tokens (so SKIP_EXACT is safe).
    embed = sd.get("llm.model.embed_tokens.weight")
    head  = sd.get("llm.lm_head.weight")
    if embed is None or head is None:
        raise RuntimeError("missing llm.model.embed_tokens.weight or llm.lm_head.weight")
    if not torch.equal(embed, head):
        raise RuntimeError(
            "llm.lm_head.weight is NOT bitwise tied to embed_tokens; "
            "the converter currently assumes the published 2512 checkpoint "
            "where they are equal. Update SKIP_EXACT and re-emit lm_head as "
            "dec.lm_head.weight if upstream un-ties them."
        )

    total_params = sum(int(v.numel()) for v in sd.values())
    size_label = compute_size_label(total_params)
    print(f"Total params: {total_params:,} -> size_label={size_label}")

    print(f"Writing GGUF to {out_path}")
    writer = GGUFWriter(str(out_path), "funasr_nano")

    # ----- general.* -----
    # FunASR Model Open Source License Agreement v1.1 attribution
    # requirement (`MODEL_LICENSE` 2.2): "you must attribute the source
    # and author information and retain relevant model names". Bake the
    # canonical attribution into the GGUF KV so downstream consumers
    # (anyone loading the converted file) see source + author + model
    # names without having to read external docs.
    writer.add_string("general.name",         display_name)
    writer.add_string("general.basename",     variant.rsplit("-", 1)[0] if "-" in variant else variant)
    writer.add_string("general.size_label",   size_label)
    writer.add_uint32("general.file_type",    int(REFERENCE_FILE_TYPE))
    writer.add_array ("general.languages",    hp["languages"])
    writer.add_string("general.author",       "Alibaba Group / FunAudioLLM")
    writer.add_string("general.organization", "FunAudioLLM")
    writer.add_string("general.license",      "FunASR-Model-License-1.1")
    writer.add_string("general.license.link",
                      "https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE")
    writer.add_string("general.url",
                      f"https://huggingface.co/FunAudioLLM/{display_name}")
    writer.add_string("general.source.url",
                      "https://github.com/modelscope/FunASR")
    # Model-name retention (license clause 2.2): list every canonical
    # upstream component the checkpoint stitches together. Anyone using,
    # copying, modifying, or sharing this GGUF must keep these visible.
    writer.add_array("general.tags", [
        "asr",
        "speech-recognition",
        "audio-llm",
        "FunASRNano",
        display_name,
        "SenseVoiceEncoderSmall",
        "Qwen3-0.6B",
    ])
    writer.add_string("general.description",
                      f"{display_name} (Alibaba FunAudioLLM): "
                      "SenseVoiceEncoderSmall encoder + 2-layer audio "
                      "adaptor + Qwen3-0.6B LLM. Bundled Qwen3-0.6B "
                      "weights are derivative of Qwen/Qwen3-0.6B "
                      "(Apache-2.0). Converted from FunAudioLLM/"
                      f"{display_name} model.pt; see "
                      "https://github.com/modelscope/FunASR/blob/main/"
                      "MODEL_LICENSE for FunASR redistribution terms.")

    # ----- stt.variant -----
    writer.add_string("stt.variant", variant)

    # ----- tokenizer.ggml.* -----
    # Qwen3 byte-level BPE — llama.cpp tags this as "gpt2".
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre",   "qwen2")
    writer.add_array ("tokenizer.ggml.tokens",     tok["tokens"])
    writer.add_array ("tokenizer.ggml.token_type", tok["types"])
    writer.add_array ("tokenizer.ggml.merges",     tok["merges"])
    if tok["bos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
    if tok["eos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
    if tok["pad_id"] is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])
    if tok["chat_template"]:
        writer.add_string("tokenizer.chat_template", tok["chat_template"])

    # ----- stt.funasr_nano.encoder.* -----
    writer.add_uint32("stt.funasr_nano.encoder.n_blocks",     hp["enc_n_blocks"])
    writer.add_uint32("stt.funasr_nano.encoder.tp_blocks",    hp["enc_tp_blocks"])
    writer.add_uint32("stt.funasr_nano.encoder.d_model",      hp["enc_d_model"])
    writer.add_uint32("stt.funasr_nano.encoder.d_input",      hp["enc_d_input"])
    writer.add_uint32("stt.funasr_nano.encoder.n_heads",      hp["enc_n_heads"])
    writer.add_uint32("stt.funasr_nano.encoder.d_ff",         hp["enc_d_ff"])
    writer.add_uint32("stt.funasr_nano.encoder.kernel_size",  hp["enc_kernel_size"])
    writer.add_uint32("stt.funasr_nano.encoder.sanm_shift",   hp["enc_sanm_shift"])
    writer.add_string("stt.funasr_nano.encoder.attention_type", hp["enc_attn_type"])
    writer.add_bool  ("stt.funasr_nano.encoder.normalize_before",
                      hp["enc_normalize_before"])

    # ----- stt.funasr_nano.adaptor.* -----
    # Adaptor structural shape is fully observable from the loaded
    # checkpoint. We bake the values seen here so the loader does not
    # depend on private FunASR class defaults.
    adp_blk0_q = sd["audio_adaptor.blocks.0.self_attn.linear_q.weight"]
    adp_n_heads_default = 8                          # MultiHeadedAttention default in funasr
    adp_d_head_default  = adp_blk0_q.shape[0] // adp_n_heads_default  # = 1024 / 8 = 128
    adp_block_ffn_dim   = int(sd["audio_adaptor.blocks.0.feed_forward.w_1.weight"].shape[0])
    adp_pre_ffn_obs     = int(sd["audio_adaptor.linear1.weight"].shape[0])
    writer.add_uint32 ("stt.funasr_nano.adaptor.n_blocks",          hp["adp_n_blocks"])
    writer.add_uint32 ("stt.funasr_nano.adaptor.encoder_dim",       hp["adp_encoder_dim"])
    writer.add_uint32 ("stt.funasr_nano.adaptor.llm_dim",           hp["adp_llm_dim"])
    writer.add_uint32 ("stt.funasr_nano.adaptor.pre_ffn_dim",       adp_pre_ffn_obs)
    writer.add_uint32 ("stt.funasr_nano.adaptor.block_ffn_dim",     adp_block_ffn_dim)
    writer.add_uint32 ("stt.funasr_nano.adaptor.n_heads",           adp_n_heads_default)
    writer.add_uint32 ("stt.funasr_nano.adaptor.d_head",            adp_d_head_default)
    writer.add_float32("stt.funasr_nano.adaptor.layer_norm_eps",    1e-12)
    writer.add_string ("stt.funasr_nano.adaptor.activation",        "relu")
    writer.add_uint32 ("stt.funasr_nano.adaptor.downsample_rate",   hp["adp_downsample_rate"])
    writer.add_bool   ("stt.funasr_nano.adaptor.use_low_frame_rate",
                       hp["adp_use_low_frame_rate"])

    # ----- stt.funasr_nano.decoder.* -----
    writer.add_uint32 ("stt.funasr_nano.decoder.n_layers",       hp["dec_n_layers"])
    writer.add_uint32 ("stt.funasr_nano.decoder.hidden_size",    hp["dec_hidden"])
    writer.add_uint32 ("stt.funasr_nano.decoder.intermediate_size",
                       hp["dec_intermediate"])
    writer.add_uint32 ("stt.funasr_nano.decoder.n_heads",        hp["dec_n_heads"])
    writer.add_uint32 ("stt.funasr_nano.decoder.n_kv_heads",     hp["dec_n_kv_heads"])
    writer.add_uint32 ("stt.funasr_nano.decoder.head_dim",       hp["dec_head_dim"])
    writer.add_uint32 ("stt.funasr_nano.decoder.vocab_size",     hp["dec_vocab_size"])
    writer.add_uint32 ("stt.funasr_nano.decoder.max_position_embeddings",
                       hp["dec_max_pos"])
    writer.add_float32("stt.funasr_nano.decoder.rms_norm_eps",   hp["dec_rms_norm_eps"])
    writer.add_float32("stt.funasr_nano.decoder.rope_theta",     hp["dec_rope_theta"])
    writer.add_bool   ("stt.funasr_nano.decoder.tie_word_embeddings",
                       hp["dec_tie_embeddings"])
    writer.add_string ("stt.funasr_nano.decoder.activation",     hp["dec_hidden_act"])

    # ----- stt.frontend.* -----
    writer.add_string ("stt.frontend.type",            hp["fe_type"])
    writer.add_uint32 ("stt.frontend.num_mels",        hp["fe_num_mels"])
    writer.add_uint32 ("stt.frontend.sample_rate",     hp["fe_sample_rate"])
    writer.add_uint32 ("stt.frontend.n_fft",           hp["fe_n_fft"])
    writer.add_uint32 ("stt.frontend.win_length",      hp["fe_win_length"])
    writer.add_uint32 ("stt.frontend.hop_length",      hp["fe_hop_length"])
    writer.add_string ("stt.frontend.window",          hp["fe_window"])
    writer.add_string ("stt.frontend.normalize",       hp["fe_normalize"])
    writer.add_float32("stt.frontend.dither",          hp["fe_dither"])
    writer.add_bool   ("stt.frontend.upscale_samples", hp["fe_upscale_samples"])
    writer.add_bool   ("stt.frontend.snip_edges",      hp["fe_snip_edges"])
    writer.add_uint32 ("stt.frontend.lfr_m",           hp["fe_lfr_m"])
    writer.add_uint32 ("stt.frontend.lfr_n",           hp["fe_lfr_n"])
    writer.add_string ("stt.frontend.fbank_style",     hp["fe_fbank_style"])
    writer.add_bool   ("stt.frontend.apply_cmvn",      hp["fe_apply_cmvn"])

    # ----- tensors -----
    n_added  = 0
    bytes_in = 0
    bytes_out = 0
    consumed: set[str] = set()

    def add_tensor(src_name: str, dst_name: str) -> None:
        nonlocal n_added, bytes_in, bytes_out
        if src_name not in sd:
            raise KeyError(f"state_dict missing tensor: {src_name!r}")
        consumed.add(src_name)
        t = sd[src_name]
        # encode_for_gguf needs F32 input regardless of the target type.
        arr_f32 = t.detach().to(dtype=torch.float32).contiguous().numpy()
        target = per_tensor_target_dtype(dst_name)
        encoded, raw_dtype = encode_for_gguf(np.ascontiguousarray(arr_f32), target)
        writer.add_tensor(dst_name, encoded, raw_dtype=raw_dtype)
        bytes_in  += int(arr_f32.nbytes)
        bytes_out += int(encoded.nbytes)
        n_added += 1

    # ----- encoder tensors -----
    # encoders0 (1 block — 560-dim input projection block).
    for src_suf, dst_suf in ENC_BLOCK_TABLE:
        add_tensor(f"audio_encoder.encoders0.0.{src_suf}",
                   f"enc.encoders0.0.{dst_suf}")

    # encoders (49 blocks = num_blocks - 1 since num_blocks counts encoders0[0]).
    for i in range(hp["enc_n_blocks"] - 1):
        for src_suf, dst_suf in ENC_BLOCK_TABLE:
            add_tensor(f"audio_encoder.encoders.{i}.{src_suf}",
                       f"enc.encoders.{i}.{dst_suf}")

    # after_norm (between main and tp tiers).
    add_tensor("audio_encoder.after_norm.weight", "enc.after_norm.weight")
    add_tensor("audio_encoder.after_norm.bias",   "enc.after_norm.bias")

    # tp_encoders (20 blocks).
    for i in range(hp["enc_tp_blocks"]):
        for src_suf, dst_suf in ENC_BLOCK_TABLE:
            add_tensor(f"audio_encoder.tp_encoders.{i}.{src_suf}",
                       f"enc.tp_encoders.{i}.{dst_suf}")

    # tp_norm (final encoder LayerNorm).
    add_tensor("audio_encoder.tp_norm.weight", "enc.tp_norm.weight")
    add_tensor("audio_encoder.tp_norm.bias",   "enc.tp_norm.bias")

    # ----- adaptor tensors -----
    add_tensor("audio_adaptor.linear1.weight", "adaptor.linear1.weight")
    add_tensor("audio_adaptor.linear1.bias",   "adaptor.linear1.bias")
    add_tensor("audio_adaptor.linear2.weight", "adaptor.linear2.weight")
    add_tensor("audio_adaptor.linear2.bias",   "adaptor.linear2.bias")
    for i in range(hp["adp_n_blocks"]):
        for src_suf, dst_suf in ADAPTOR_BLOCK_TABLE:
            add_tensor(f"audio_adaptor.blocks.{i}.{src_suf}",
                       f"adaptor.blocks.{i}.{dst_suf}")

    # ----- LLM tensors -----
    for src, dst in LLM_TOP_TABLE:
        add_tensor(src, dst)
    for i in range(hp["dec_n_layers"]):
        for src_suf, dst_suf in LLM_BLOCK_TABLE:
            add_tensor(f"llm.model.layers.{i}.{src_suf}",
                       f"dec.layers.{i}.{dst_suf}")

    # Mark intentionally-skipped state_dict keys as consumed so the
    # tail-of-file warning only fires on truly unexpected keys.
    for k in sd:
        if k in SKIP_EXACT:
            consumed.add(k)
        elif any(k.startswith(p) for p in SKIP_PREFIXES):
            consumed.add(k)

    expected = (
        len(ENC_BLOCK_TABLE) * (1 + (hp["enc_n_blocks"] - 1) + hp["enc_tp_blocks"])
        + 4   # after_norm + tp_norm (w+b each)
        + 4   # adaptor linear1/linear2 (w+b each)
        + len(ADAPTOR_BLOCK_TABLE) * hp["adp_n_blocks"]
        + len(LLM_TOP_TABLE)
        + len(LLM_BLOCK_TABLE) * hp["dec_n_layers"]
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

    unused = sorted(set(sd.keys()) - consumed)
    if unused:
        print(f"WARNING: {len(unused)} state_dict keys not consumed:",
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

    print(f"Done. Wrote {out_path} "
          f"({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


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


SLUG_TO_VARIANT = {
    "Fun-ASR-Nano-2512": "fun-asr-nano-2512",
}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a Fun-ASR-Nano (FunASR) checkpoint to a BF16 GGUF.",
    )
    p.add_argument("model", type=str,
                   help="HF repo id (e.g. FunAudioLLM/Fun-ASR-Nano-2512) "
                        "or local checkpoint dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug "
                        "when converting from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision to pin the download to (recommended)")
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

    raw_slug = slug_from_repo_id(repo_id) if repo_id else None
    variant = args.variant or (SLUG_TO_VARIANT.get(raw_slug) if raw_slug else None)
    if variant is None:
        if raw_slug:
            variant = raw_slug.lower()
        else:
            print("error: cannot infer variant; pass --variant or --repo-id",
                  file=sys.stderr)
            return 2

    out_path = args.out_path
    if out_path is None:
        # GGUF dir + filename use the upstream HF casing (`raw_slug`); the
        # `stt.variant` KV in the GGUF body stays kebab-case (`variant`)
        # so internal tooling (manifest, build/validate, family doc paths)
        # remains lowercase. Matches the qwen3_asr / parakeet / whisper
        # pattern.
        output_slug = raw_slug or variant
        out_path = REPO_ROOT / "models" / output_slug / gguf_name(output_slug, REFERENCE_DTYPE_LABEL)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_slug = raw_slug or variant

    convert(model_dir, out_path, variant, display_name=output_slug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
