#!/usr/bin/env python3
"""
convert-canary-qwen.py - convert a NeMo Canary-Qwen SALM (FastConformer
encoder + Qwen3-1.7B LM with LoRA) into a GGUF that transcribe.cpp's
loader can ingest. Preserves the source BF16 dtype; block quantization
(Q8_0, Q5_K_M, ...) goes through tools/transcribe-quantize.

Source format:
    HF model id (e.g. "nvidia/canary-qwen-2.5b") that NeMo's
    speechlm2.SALM.from_pretrained() can resolve, or a local SALM
    checkpoint.

Target format:
    Single .gguf at models/<slug>/<slug>-BF16.gguf. Reference dtype
    (BF16) is fixed by the SALM checkpoint header; the converter rejects
    sources with a different state-dict dtype rather than silently
    downcasting.

Architecture summary (audio-LLM SALM):

    audio (16 kHz mono)
      -> AudioToMelSpectrogramPreprocessor                 [B, n_mels, T]
      -> FastConformer encoder (32 layers, d_model=1024,
         dw-striding factor=8, rel_pos with untie_biases)  [B, T_enc, 1024]
      -> AudioPerceptionModule projection (1024 -> 2048)   [B, T_enc, 2048]
      -> SCATTER into Qwen3 LM input_embeds at positions
         where input_ids == audio_locator_id (151669)      [B, T_lm, 2048]
      -> Qwen3-1.7B LM (28 layers, GQA 16/8, head_dim=128,
         q/k RMSNorm, SwiGLU, 1D RoPE, tied embed)         logits over 151936

Pre-merge: the SALM checkpoint stores LoRA adapters separately on
q_proj/v_proj. We call peft.merge_and_unload() so the GGUF (and the
Stage 4 loader) sees a flat post-merge Qwen3 LM with no LoRA pairs.

KV emitted:

    general.architecture     = "canary_qwen"
    general.basename         = "canary-qwen"
    general.size_label       = "2.5B"
    general.languages        = ["en"]
    general.license          = "CC-BY-4.0"

    stt.variant              = "canary-qwen-2.5b"
    stt.capability.lang_detect = false
    stt.capability.translate   = false
    stt.capability.timestamps  = false
    stt.capability.streaming   = false

    tokenizer.ggml.model     = "gpt2"   (Qwen3 byte-level BPE)
    tokenizer.ggml.pre       = "qwen2"
    tokenizer.ggml.tokens    = full token list (151936 entries; slot
                               151669 carries `<|audioplaceholder|>`)
    tokenizer.ggml.token_type
    tokenizer.ggml.merges
    tokenizer.ggml.eos_token_id / padding_token_id
    tokenizer.chat_template  = Qwen3 chat template

    stt.canary_qwen.encoder.*       FastConformer hparams (mirror canary)
    stt.canary_qwen.perception.*    output_dim, audio_locator_id
    stt.canary_qwen.decoder.*       Qwen3 LM hparams (1D RoPE, no MRoPE)
    stt.frontend.*                  NeMo AudioToMelSpectrogramPreprocessor

CLI:

    uv run --project scripts/envs/canary_qwen \
      scripts/convert-canary-qwen.py nvidia/canary-qwen-2.5b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType

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

REFERENCE_DTYPE_LABEL = "BF16"
REFERENCE_FILE_TYPE = LlamaFileType.MOSTLY_BF16
REFERENCE_GGML_TYPE = GGMLQuantizationType.BF16


# ---------------------------------------------------------------------------
# Variant profiles
# ---------------------------------------------------------------------------

VARIANT_PROFILES: dict[str, dict[str, Any]] = {
    "canary-qwen-2.5b": {
        "size_label":   "2.5B",
        "license":      "CC-BY-4.0",
        "license_link": "https://creativecommons.org/licenses/by/4.0/",
        "languages":    ["en"],
        # Upstream tokenizer to pull vocab.json + merges.txt + chat_template
        # from. The SALM checkpoint does not ship these; the LM is
        # bit-identical to Qwen/Qwen3-1.7B at this revision.
        "lm_tokenizer_repo": "Qwen/Qwen3-1.7B",
        # SALM-added special token; pinned at Stage 2 from
        # model.tokenizer.text_to_ids('<|audioplaceholder|>') == [151669]
        "audio_locator_tag": "<|audioplaceholder|>",
        "audio_locator_id":  151669,
    },
}


# ---------------------------------------------------------------------------
# Model loading + LoRA merge
# ---------------------------------------------------------------------------


def load_salm(model_spec: str):
    """Load a SALM model via NeMo speechlm2 and merge LoRA into the LM
    base weights so the state_dict carries flat Qwen3 q_proj/v_proj
    with no separate LoRA A/B pairs."""
    from nemo.collections.speechlm2.models import SALM

    local = Path(model_spec).expanduser()
    if local.exists():
        print(f"Loading SALM from local path: {local}")
        model = SALM.from_pretrained(str(local), map_location="cpu")
    else:
        print(f"Loading SALM from HuggingFace: {model_spec}")
        model = SALM.from_pretrained(model_spec, map_location="cpu")
    model.eval()

    # Merge LoRA into the base Qwen3 LM. PEFT's merge_and_unload returns
    # a "clean" Qwen3ForCausalLM with the trained LoRA folded into
    # q_proj / v_proj. After this, model.llm.state_dict() has standard
    # Qwen3 keys (no .lora_A / .lora_B suffixes).
    llm = model.llm
    if hasattr(llm, "merge_and_unload"):
        print("Merging LoRA adapters into Qwen3 base weights...")
        merged = llm.merge_and_unload()
        model.llm = merged
    else:
        # Sanity-check: if the LM doesn't expose merge_and_unload, the
        # checkpoint may already be merged or use a non-PEFT adapter
        # mechanism. Reject loudly rather than silently shipping the LoRA
        # pairs as separate tensors.
        sd = llm.state_dict()
        lora_keys = [k for k in sd if ".lora_" in k]
        if lora_keys:
            raise RuntimeError(
                f"LM has {len(lora_keys)} LoRA-pair tensors but no "
                f"merge_and_unload(); converter cannot ship un-merged "
                f"adapters. Sample: {lora_keys[:3]}"
            )

    return model


# ---------------------------------------------------------------------------
# Tokenizer extraction (Qwen3 BPE + the SALM-added audio placeholder)
# ---------------------------------------------------------------------------
#
# Pull vocab.json, merges.txt, tokenizer_config.json, and chat_template
# from Qwen/Qwen3-1.7B (the LM backbone the SALM checkpoint inherits
# from). Then overlay the SALM-added <|audioplaceholder|> at id 151669.


def fetch_lm_tokenizer(lm_repo: str) -> Path:
    """Snapshot-download the LM tokenizer files into the standard HF
    cache (or $TRANSCRIBE_MODELS_DIR/<slug>/) and return the local dir."""
    from huggingface_hub import snapshot_download

    slug = slug_from_repo_id(lm_repo)
    models_root = os.environ.get("TRANSCRIBE_MODELS_DIR")
    local_dir = Path(models_root) / slug if models_root else None
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching LM tokenizer from {lm_repo} ...")
    resolved = snapshot_download(
        repo_id=lm_repo,
        allow_patterns=[
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "tokenizer.json",
            "chat_template.json",
            "special_tokens_map.json",
        ],
        local_dir=str(local_dir) if local_dir is not None else None,
    )
    return Path(resolved)


def extract_tokenizer(
    lm_dir: Path,
    vocab_size: int,
    audio_locator_tag: str,
    audio_locator_id: int,
) -> dict:
    """Build the canary-qwen tokenizer entry from the upstream Qwen3
    tokenizer files plus the SALM-added audio placeholder.

    The audio placeholder occupies a slot inside Qwen3's reserved-token
    block (151643..151935). At id 151669 in nvidia/canary-qwen-2.5b
    (verified at Stage 2). If the LM tokenizer already has a token at
    that id (e.g. a `<|reserved_NNN|>` placeholder), it is overwritten.
    """
    vocab_path  = lm_dir / "vocab.json"
    merges_path = lm_dir / "merges.txt"
    tokcfg_path = lm_dir / "tokenizer_config.json"

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

    # Overlay the SALM-added audio placeholder at its known slot.
    if audio_locator_id in tok_by_id:
        existing, _ = tok_by_id[audio_locator_id]
        if existing != audio_locator_tag:
            print(
                f"  overlaying tokenizer slot {audio_locator_id}: "
                f"{existing!r} -> {audio_locator_tag!r}"
            )
    else:
        print(
            f"  adding tokenizer slot {audio_locator_id} = "
            f"{audio_locator_tag!r}"
        )
    tok_by_id[audio_locator_id] = (audio_locator_tag, True)

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
    chat_template_json = lm_dir / "chat_template.json"
    if chat_template_json.is_file():
        with chat_template_json.open() as f:
            chat_template = json.load(f).get("chat_template", chat_template)

    return {
        "tokens":         tokens,
        "types":          types,
        "merges":         merges,
        "eos_id":         _name_to_id("eos_token"),
        "pad_id":         _name_to_id("pad_token"),
        "bos_id":         _name_to_id("bos_token"),
        "chat_template":  chat_template,
    }


# ---------------------------------------------------------------------------
# Hparams from model.cfg
# ---------------------------------------------------------------------------


def read_hparams(model) -> dict:
    """Extract hparams from the SALM model. SALM's cfg is the YAML
    training cfg (perception.{preprocessor,encoder,modality_adapter},
    perception.output_dim, lora.*); the LM hparams come from the loaded
    Qwen3ForCausalLM's HF config."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(model.cfg, resolve=True)
    perc = cfg["perception"]
    pre  = perc["preprocessor"]
    enc  = perc["encoder"]

    # Frontend (NeMo AudioToMelSpectrogramPreprocessor).
    sample_rate = int(pre["sample_rate"])
    win_length  = int(round(float(pre["window_size"])  * sample_rate))
    hop_length  = int(round(float(pre["window_stride"]) * sample_rate))

    # LM (Qwen3-1.7B). After merge_and_unload, model.llm is a flat
    # Qwen3ForCausalLM whose .config carries the stable hparams.
    lm_cfg = model.llm.config

    def _get(o, name, default=None):
        return getattr(o, name, default) if hasattr(o, name) else (
            o.get(name, default) if isinstance(o, dict) else default
        )

    return {
        # Encoder (FastConformer; mirrors canary's read_hparams).
        "enc_n_layers":             int(enc["n_layers"]),
        "enc_d_model":              int(enc["d_model"]),
        "enc_n_heads":              int(enc["n_heads"]),
        "enc_d_ff":                 int(enc["d_model"]) * int(enc["ff_expansion_factor"]),
        "enc_conv_kernel":          int(enc["conv_kernel_size"]),
        "enc_subsampling_factor":   int(enc["subsampling_factor"]),
        "enc_subsampling_channels": int(enc["subsampling_conv_channels"]),
        "enc_pos_emb_max_len":      int(enc["pos_emb_max_len"]),
        "enc_use_bias":             True,  # NeMo ConformerEncoder always biased

        # Perception projection.
        "perception_output_dim": int(perc["output_dim"]),

        # Audio injection.
        "audio_locator_tag": str(getattr(model, "audio_locator_tag", "<|audioplaceholder|>")),

        # LM (Qwen3-1.7B).
        "dec_n_layers":     int(_get(lm_cfg, "num_hidden_layers")),
        "dec_hidden":       int(_get(lm_cfg, "hidden_size")),
        "dec_intermediate": int(_get(lm_cfg, "intermediate_size")),
        "dec_n_heads":      int(_get(lm_cfg, "num_attention_heads")),
        "dec_n_kv_heads":   int(_get(lm_cfg, "num_key_value_heads")),
        "dec_head_dim":     int(_get(lm_cfg, "head_dim")),
        "dec_hidden_act":   str(_get(lm_cfg, "hidden_act", "silu")).lower(),
        "dec_rms_norm_eps": float(_get(lm_cfg, "rms_norm_eps")),
        "dec_rope_theta":   float(_get(lm_cfg, "rope_theta", 10000.0)),
        "dec_max_pos_emb":  int(_get(lm_cfg, "max_position_embeddings")),
        "dec_tie_embeddings": bool(_get(lm_cfg, "tie_word_embeddings", True)),
        "dec_vocab_size":   int(_get(lm_cfg, "vocab_size")),

        # Frontend (NeMo AudioToMelSpectrogramPreprocessor).
        "fe_type":         "mel",
        "fe_num_mels":     int(pre["features"]),
        "fe_sample_rate":  sample_rate,
        "fe_n_fft":        int(pre["n_fft"]),
        "fe_win_length":   win_length,
        "fe_hop_length":   hop_length,
        "fe_window":       str(pre["window"]),         # "hann" -> "hann" (canonicalized below if needed)
        "fe_normalize":    str(pre["normalize"]),      # "per_feature"
        "fe_dither":       0.0,                        # config declares 1e-5; inference uses 0.0
        "fe_log":          bool(pre.get("log", True)),
        "fe_pre_emphasis": 0.97,                       # NeMo FilterbankFeatures default
        "fe_f_min":        0.0,
        "fe_f_max":        float(sample_rate) / 2.0,
        "fe_pad_mode":     "reflect",
        "fe_center":       True,
        "fe_mel_norm":     "slaney",
    }


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------
#
# Encoder: mirrors canary's PRE_ENCODE_TABLE / ENCODER_BLOCK_TABLE
# (FastConformer is byte-for-byte the same module class). The PyTorch
# source prefix is `perception.encoder.` rather than `encoder.` because
# the encoder lives inside the perception module.
#
# LM: mirrors qwen3_asr's TEXT_TOP_TABLE / TEXT_BLOCK_TABLE. The PyTorch
# source prefix is `llm.model.` (post merge_and_unload). lm_head is tied
# to embed_tokens and skipped.


PRE_ENCODE_TABLE: list[tuple[str, str]] = [
    ("perception.encoder.pre_encode.conv.0.weight", "enc.pre_encode.conv.0.weight"),
    ("perception.encoder.pre_encode.conv.0.bias",   "enc.pre_encode.conv.0.bias"),
    ("perception.encoder.pre_encode.conv.2.weight", "enc.pre_encode.conv.2.weight"),
    ("perception.encoder.pre_encode.conv.2.bias",   "enc.pre_encode.conv.2.bias"),
    ("perception.encoder.pre_encode.conv.3.weight", "enc.pre_encode.conv.3.weight"),
    ("perception.encoder.pre_encode.conv.3.bias",   "enc.pre_encode.conv.3.bias"),
    ("perception.encoder.pre_encode.conv.5.weight", "enc.pre_encode.conv.5.weight"),
    ("perception.encoder.pre_encode.conv.5.bias",   "enc.pre_encode.conv.5.bias"),
    ("perception.encoder.pre_encode.conv.6.weight", "enc.pre_encode.conv.6.weight"),
    ("perception.encoder.pre_encode.conv.6.bias",   "enc.pre_encode.conv.6.bias"),
    ("perception.encoder.pre_encode.out.weight",    "enc.pre_encode.out.weight"),
    ("perception.encoder.pre_encode.out.bias",      "enc.pre_encode.out.bias"),
]

ENCODER_BLOCK_TABLE: list[tuple[str, str]] = [
    ("norm_feed_forward1.weight",       "norm_ff1.weight"),
    ("norm_feed_forward1.bias",         "norm_ff1.bias"),
    ("feed_forward1.linear1.weight",    "ff1.linear1.weight"),
    ("feed_forward1.linear1.bias",      "ff1.linear1.bias"),
    ("feed_forward1.linear2.weight",    "ff1.linear2.weight"),
    ("feed_forward1.linear2.bias",      "ff1.linear2.bias"),

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
    ("self_attn.linear_pos.weight",     "attn.linear_pos.weight"),
    ("self_attn.pos_bias_u",            "attn.pos_bias_u"),
    ("self_attn.pos_bias_v",            "attn.pos_bias_v"),

    ("norm_conv.weight",                "norm_conv.weight"),
    ("norm_conv.bias",                  "norm_conv.bias"),
    ("conv.pointwise_conv1.weight",     "conv.pointwise1.weight"),
    ("conv.pointwise_conv1.bias",       "conv.pointwise1.bias"),
    ("conv.depthwise_conv.weight",      "conv.depthwise.weight"),
    ("conv.depthwise_conv.bias",        "conv.depthwise.bias"),
    ("conv.pointwise_conv2.weight",     "conv.pointwise2.weight"),
    ("conv.pointwise_conv2.bias",       "conv.pointwise2.bias"),
    ("conv.batch_norm.weight",          "conv.bn.weight"),
    ("conv.batch_norm.bias",            "conv.bn.bias"),
    ("conv.batch_norm.running_mean",    "conv.bn.running_mean"),
    ("conv.batch_norm.running_var",     "conv.bn.running_var"),

    ("norm_feed_forward2.weight",       "norm_ff2.weight"),
    ("norm_feed_forward2.bias",         "norm_ff2.bias"),
    ("feed_forward2.linear1.weight",    "ff2.linear1.weight"),
    ("feed_forward2.linear1.bias",      "ff2.linear1.bias"),
    ("feed_forward2.linear2.weight",    "ff2.linear2.weight"),
    ("feed_forward2.linear2.bias",      "ff2.linear2.bias"),

    ("norm_out.weight",                 "norm_out.weight"),
    ("norm_out.bias",                   "norm_out.bias"),
]


# Perception projection: encoder d_model=1024 -> output_dim=2048, lives
# in AudioPerceptionModule. The exact tensor name is discovered from
# state_dict at convert time (NeMo SALM's AudioPerceptionModule has
# evolved; common names: `perception.proj.{weight,bias}`,
# `perception.out_proj.{weight,bias}`, `perception.audio_proj.{weight,bias}`).
PERCEPTION_PROJ_CANDIDATES = [
    ("perception.proj.weight",       "perception.proj.bias"),
    ("perception.out_proj.weight",   "perception.out_proj.bias"),
    ("perception.audio_proj.weight", "perception.audio_proj.bias"),
]


# Top-level shared embeddings. SALM keeps `embed_tokens` at the TOP
# level (referenced by the freeze_params regex `^embed_tokens\\..+$`)
# and the inner Qwen3 model's `model.embed_tokens` is bound to the same
# Parameter object — so it shows up only once in the state_dict, under
# the top-level name.
#
# `llm.lm_head.weight` is bitwise equal to `embed_tokens.weight` on
# nvidia/canary-qwen-2.5b (verified) — we skip it and the loader
# reuses dec.token_embd.weight for the output projection (qwen3_asr
# convention; tie_word_embeddings=true).
TEXT_TOP_TABLE: list[tuple[str, str]] = [
    ("embed_tokens.weight",   "dec.token_embd.weight"),
    ("llm.model.norm.weight", "dec.output_norm.weight"),
]


TEXT_BLOCK_TABLE: list[tuple[str, str]] = [
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


# State-dict keys we deliberately skip.
SKIP_EXACT = {
    # Tied LM head (tie_word_embeddings=true; bitwise equal to
    # embed_tokens.weight on nvidia/canary-qwen-2.5b — verified).
    "llm.lm_head.weight",
    # Rel-pos sinusoidal table; rebuilt at runtime by the encoder.
    "perception.encoder.pos_enc.pe",
}

# Note: `perception.preprocessor.featurizer.{fb,window}` were previously
# included here (and recomputed at convert time via librosa / numpy).
# That was a SUBTLE BUG: the SALM checkpoint stores its weights at BF16,
# so even when loaded at F32 those buffers carry BF16-truncated values
# (e.g. fb[0,1] = 0.0283203125 vs librosa's 0.02837754). The reference
# dumper reads exactly those truncated buffers, so matching the
# reference requires baking the SAME truncated values — not freshly
# recomputed F32 ones. We now extract them in `convert()` and DO NOT
# skip them here; the consume_preproc_buffers() helper marks them as
# consumed so the unused-key check stays clean.
SKIP_PREFIXES = (
    "perception.encoder.interctc.",  # InterCTC head, not used at inference
)

# Suffixes for non-parameter buffers we don't need.
SKIP_SUFFIXES = (
    ".num_batches_tracked",       # BN bookkeeping
)


def is_expected_unused(key: str) -> bool:
    if key in SKIP_EXACT:
        return True
    if key.startswith(SKIP_PREFIXES):
        return True
    for suf in SKIP_SUFFIXES:
        if key.endswith(suf):
            return True
    return False


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def tensor_to_fp32_numpy(t) -> np.ndarray:
    """Torch Tensor -> contiguous fp32 numpy. Source must be BF16
    (the SALM checkpoint header). Cast happens here so the cast site is
    visible at the call site rather than buried inside encode_for_gguf."""
    import torch

    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t).__name__}")
    if t.dtype == torch.float32:
        arr = t.detach().cpu().numpy()
    elif t.dtype == torch.bfloat16:
        arr = t.detach().to(dtype=torch.float32).cpu().numpy()
    elif t.dtype == torch.float16:
        arr = t.detach().to(dtype=torch.float32).cpu().numpy()
    else:
        raise ValueError(f"unsupported tensor dtype {t.dtype}; expected bf16 / f16 / f32")
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Variant resolution
# ---------------------------------------------------------------------------


def resolve_variant(model_spec: str, repo_id_arg: str | None) -> tuple[str, dict]:
    repo_id = repo_id_arg
    if repo_id is None and "/" in model_spec and not Path(model_spec).exists():
        repo_id = model_spec
    if repo_id is None:
        raise ValueError(
            "could not derive variant — provide --repo-id when converting "
            "from a local path"
        )
    slug = slug_from_repo_id(repo_id)
    if slug not in VARIANT_PROFILES:
        raise ValueError(
            f"unknown canary-qwen variant {slug!r}; known: {sorted(VARIANT_PROFILES)}"
        )
    return slug, VARIANT_PROFILES[slug]


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert(model_spec: str, out_path: Path, variant: str, profile: dict) -> None:
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    model = load_salm(model_spec)
    hp = read_hparams(model)
    sd = model.state_dict()
    sd_keys = set(sd.keys())

    # Sanity: vocab_size from the LM config must match the audio_locator_id slot.
    if hp["dec_vocab_size"] <= profile["audio_locator_id"]:
        raise ValueError(
            f"audio_locator_id {profile['audio_locator_id']} >= vocab_size "
            f"{hp['dec_vocab_size']}"
        )
    if hp["audio_locator_tag"] != profile["audio_locator_tag"]:
        print(
            f"WARNING: SALM model.audio_locator_tag {hp['audio_locator_tag']!r} "
            f"differs from variant profile {profile['audio_locator_tag']!r}; "
            f"using profile value."
        )

    # Discover the perception-projection tensor names.
    proj_w = proj_b = None
    for w_name, b_name in PERCEPTION_PROJ_CANDIDATES:
        if w_name in sd_keys:
            proj_w, proj_b = w_name, b_name
            break
    if proj_w is None:
        # Fall back to a wider scan: any "perception.*proj*.weight" with
        # shape (perception_output_dim, enc_d_model).
        cands = [k for k in sd_keys if k.startswith("perception.")
                 and k.endswith(".weight")
                 and "proj" in k.lower()]
        for k in cands:
            shp = tuple(sd[k].shape)
            if shp == (hp["perception_output_dim"], hp["enc_d_model"]):
                proj_w = k
                proj_b = k.removesuffix(".weight") + ".bias"
                if proj_b not in sd_keys:
                    proj_b = None  # bias may legitimately be absent
                break
    if proj_w is None:
        raise RuntimeError(
            "could not find perception projection weight in state_dict; "
            "expected a (perception_output_dim, enc_d_model) Linear under "
            "perception.*. Inspect the state_dict and add the name to "
            "PERCEPTION_PROJ_CANDIDATES."
        )
    print(f"Perception projection: {proj_w} -> enc.proj.weight ({tuple(sd[proj_w].shape)})")
    if proj_b is not None and proj_b in sd_keys:
        print(f"  + bias: {proj_b} -> enc.proj.bias ({tuple(sd[proj_b].shape)})")
    else:
        print("  (no bias on the perception projection)")

    print(f"Variant: {variant}")
    print(f"Encoder: {hp['enc_n_layers']} layers, d_model={hp['enc_d_model']}, "
          f"d_ff={hp['enc_d_ff']}, n_heads={hp['enc_n_heads']}")
    print(f"Perception output_dim: {hp['perception_output_dim']}")
    print(f"LM: {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, "
          f"heads={hp['dec_n_heads']}/{hp['dec_n_kv_heads']}, "
          f"head_dim={hp['dec_head_dim']}, vocab={hp['dec_vocab_size']}")
    print(f"Audio locator: {profile['audio_locator_tag']!r} = id {profile['audio_locator_id']}")

    # Tokenizer (Qwen3 BPE + audio placeholder).
    lm_dir = fetch_lm_tokenizer(profile["lm_tokenizer_repo"])
    tok = extract_tokenizer(
        lm_dir,
        vocab_size=hp["dec_vocab_size"],
        audio_locator_tag=profile["audio_locator_tag"],
        audio_locator_id=profile["audio_locator_id"],
    )
    print(f"Tokenizer: {len(tok['tokens'])} tokens, {len(tok['merges'])} merges")

    print(f"Writing GGUF to {out_path}")
    writer = GGUFWriter(str(out_path), "canary_qwen")

    # ---- general.* ----
    writer.add_string("general.basename",     "canary-qwen")
    writer.add_string("general.size_label",   profile["size_label"])
    writer.add_uint32("general.file_type",    int(REFERENCE_FILE_TYPE))
    writer.add_array ("general.languages",    profile["languages"])
    writer.add_string("general.license",      profile["license"])
    writer.add_string("general.license.link", profile["license_link"])

    # ---- stt.variant + capability KV ----
    writer.add_string("stt.variant", variant)
    writer.add_bool  ("stt.capability.lang_detect", False)
    writer.add_bool  ("stt.capability.translate",   False)
    writer.add_bool  ("stt.capability.timestamps",  False)
    writer.add_bool  ("stt.capability.streaming",   False)

    # ---- tokenizer.ggml.* (Qwen3 byte-level BPE) ----
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre",   "qwen2")
    writer.add_array ("tokenizer.ggml.tokens",     tok["tokens"])
    writer.add_array ("tokenizer.ggml.token_type", tok["types"])
    writer.add_array ("tokenizer.ggml.merges",     tok["merges"])
    if tok["eos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
    if tok["pad_id"] is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])
    if tok["bos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
    writer.add_bool("tokenizer.ggml.add_bos_token", False)
    if tok["chat_template"] is not None:
        writer.add_string("tokenizer.chat_template", tok["chat_template"])

    # ---- stt.canary_qwen.encoder.* (FastConformer, mirrors canary) ----
    writer.add_uint32("stt.canary_qwen.encoder.n_layers",             hp["enc_n_layers"])
    writer.add_uint32("stt.canary_qwen.encoder.d_model",              hp["enc_d_model"])
    writer.add_uint32("stt.canary_qwen.encoder.n_heads",              hp["enc_n_heads"])
    writer.add_uint32("stt.canary_qwen.encoder.d_ff",                 hp["enc_d_ff"])
    writer.add_uint32("stt.canary_qwen.encoder.conv_kernel",          hp["enc_conv_kernel"])
    writer.add_uint32("stt.canary_qwen.encoder.subsampling_factor",   hp["enc_subsampling_factor"])
    writer.add_uint32("stt.canary_qwen.encoder.subsampling_channels", hp["enc_subsampling_channels"])
    writer.add_uint32("stt.canary_qwen.encoder.pos_emb_max_len",      hp["enc_pos_emb_max_len"])
    writer.add_bool  ("stt.canary_qwen.encoder.use_bias",             hp["enc_use_bias"])

    # ---- stt.canary_qwen.perception.* ----
    writer.add_uint32("stt.canary_qwen.perception.output_dim",   hp["perception_output_dim"])
    writer.add_uint32("stt.canary_qwen.perception.audio_locator_id",
                      int(profile["audio_locator_id"]))
    writer.add_string("stt.canary_qwen.perception.audio_locator_tag",
                      str(profile["audio_locator_tag"]))
    writer.add_bool  ("stt.canary_qwen.perception.has_proj_bias",
                      bool(proj_b is not None and proj_b in sd_keys))

    # ---- stt.canary_qwen.decoder.* (Qwen3 LM, 1D RoPE, no MRoPE) ----
    writer.add_uint32 ("stt.canary_qwen.decoder.n_layers",       hp["dec_n_layers"])
    writer.add_uint32 ("stt.canary_qwen.decoder.hidden_size",    hp["dec_hidden"])
    writer.add_uint32 ("stt.canary_qwen.decoder.intermediate_size", hp["dec_intermediate"])
    writer.add_uint32 ("stt.canary_qwen.decoder.n_heads",        hp["dec_n_heads"])
    writer.add_uint32 ("stt.canary_qwen.decoder.n_kv_heads",     hp["dec_n_kv_heads"])
    writer.add_uint32 ("stt.canary_qwen.decoder.head_dim",       hp["dec_head_dim"])
    writer.add_string ("stt.canary_qwen.decoder.hidden_act",     hp["dec_hidden_act"])
    writer.add_float32("stt.canary_qwen.decoder.rms_norm_eps",   hp["dec_rms_norm_eps"])
    writer.add_float32("stt.canary_qwen.decoder.rope_theta",     hp["dec_rope_theta"])
    writer.add_uint32 ("stt.canary_qwen.decoder.max_position_embeddings",
                       hp["dec_max_pos_emb"])
    writer.add_bool   ("stt.canary_qwen.decoder.tie_word_embeddings",
                       hp["dec_tie_embeddings"])
    writer.add_uint32 ("stt.canary_qwen.decoder.vocab_size",     hp["dec_vocab_size"])

    # ---- stt.frontend.* (NeMo AudioToMelSpectrogramPreprocessor) ----
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
    writer.add_string ("stt.frontend.pad_mode",     hp["fe_pad_mode"])
    writer.add_bool   ("stt.frontend.center",       hp["fe_center"])
    writer.add_string ("stt.frontend.mel_norm",     hp["fe_mel_norm"])
    writer.add_bool   ("stt.frontend.log",          hp["fe_log"])

    # ---- frontend buffers (mel filterbank + window) ----
    # CRITICAL: Use the loaded model's runtime buffers, NOT freshly
    # computed librosa/torch versions. The SALM checkpoint stores its
    # weights at BF16; even when loaded at F32, the buffers
    # `featurizer.fb` and `featurizer.window` carry BF16-truncated
    # values (e.g. fb[0,1] = 0.0283203125 vs librosa's 0.02837754,
    # window[100] = 0.50390625 vs torch.hann_window(periodic=False)
    # value 0.5019684). The reference dumper reads exactly these
    # truncated buffers, so bit-identical tensor parity requires us to
    # do the same. Recomputing librosa/torch produces a different
    # filterbank/window and shows up as concentrated drift on the
    # extreme mel bins — propagated through 32 conformer blocks it
    # corrupts the LM input enough to derail generation.
    fe_buffers = {
        k: v for k, v in sd.items() if k.startswith("perception.preprocessor.")
    }
    fb_key  = "perception.preprocessor.featurizer.fb"
    win_key = "perception.preprocessor.featurizer.window"
    if fb_key not in fe_buffers or win_key not in fe_buffers:
        raise RuntimeError(
            f"SALM state_dict is missing {fb_key!r} / {win_key!r} — "
            f"cannot extract runtime mel filterbank / window."
        )
    mel_fb = tensor_to_fp32_numpy(sd[fb_key])
    if mel_fb.ndim == 3 and mel_fb.shape[0] == 1:
        mel_fb = mel_fb[0]
    hann = tensor_to_fp32_numpy(sd[win_key])

    n_added = 0
    bytes_in = 0
    bytes_out = 0

    writer.add_tensor(
        "frontend.mel_filterbank",
        np.ascontiguousarray(mel_fb),
        raw_dtype=GGMLQuantizationType.F32,
    )
    writer.add_tensor(
        "frontend.window",
        np.ascontiguousarray(hann),
        raw_dtype=GGMLQuantizationType.F32,
    )
    n_added += 2
    bytes_in  += mel_fb.nbytes + hann.nbytes
    bytes_out += mel_fb.nbytes + hann.nbytes

    # ---- weight tensors ----
    consumed: set[str] = set()
    # Mark the runtime preprocessor buffers we extracted above as consumed
    # so they don't trip the unused-key warning.
    consumed.add(fb_key)
    consumed.add(win_key)

    def add(src_name: str, dst_name: str) -> None:
        nonlocal n_added, bytes_in, bytes_out
        if src_name not in sd_keys:
            raise KeyError(f"state_dict missing tensor: {src_name!r}")
        arr = tensor_to_fp32_numpy(sd[src_name])
        target_type = reference_dtype_for(dst_name, REFERENCE_GGML_TYPE)
        encoded, raw_dtype = encode_for_gguf(arr, target_type)
        writer.add_tensor(dst_name, encoded, raw_dtype=raw_dtype)
        consumed.add(src_name)
        bytes_in  += int(arr.nbytes)
        bytes_out += int(encoded.nbytes)
        n_added += 1

    # Encoder pre_encode + per-block.
    for src, dst in PRE_ENCODE_TABLE:
        add(src, dst)
    for i in range(hp["enc_n_layers"]):
        for suffix_src, suffix_dst in ENCODER_BLOCK_TABLE:
            add(
                f"perception.encoder.layers.{i}.{suffix_src}",
                f"enc.blocks.{i}.{suffix_dst}",
            )

    # Perception projection.
    add(proj_w, "enc.proj.weight")
    if proj_b is not None and proj_b in sd_keys:
        add(proj_b, "enc.proj.bias")

    # LM top-level + per-block.
    for src, dst in TEXT_TOP_TABLE:
        add(src, dst)
    for i in range(hp["dec_n_layers"]):
        for suffix_src, suffix_dst in TEXT_BLOCK_TABLE:
            add(
                f"llm.model.layers.{i}.{suffix_src}",
                f"dec.blocks.{i}.{suffix_dst}",
            )

    expected = (
        2  # frontend filterbank + window
        + len(PRE_ENCODE_TABLE)
        + hp["enc_n_layers"] * len(ENCODER_BLOCK_TABLE)
        + 1  # perception proj weight
        + (1 if proj_b is not None and proj_b in sd_keys else 0)  # perception proj bias (optional)
        + len(TEXT_TOP_TABLE)
        + hp["dec_n_layers"] * len(TEXT_BLOCK_TABLE)
    )
    if n_added != expected:
        raise RuntimeError(
            f"tensor count mismatch: added {n_added}, expected {expected}"
        )
    print(f"Added {n_added} tensors "
          f"({bytes_in / (1024 * 1024):.1f} MB fp32 -> "
          f"{bytes_out / (1024 * 1024):.1f} MB on disk)")

    # Warn about unconsumed state_dict keys.
    unused = sorted(
        k for k in (sd_keys - consumed) if not is_expected_unused(k)
    )
    if unused:
        print(
            f"WARNING: {len(unused)} state_dict keys were not consumed:",
            file=sys.stderr,
        )
        for k in unused[:20]:
            shp = tuple(sd[k].shape)
            print(f"  {k}  shape={shp}", file=sys.stderr)
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
        description="Convert a NeMo Canary-Qwen SALM checkpoint to a BF16 GGUF.",
    )
    p.add_argument(
        "model",
        type=str,
        help="HF repo id (e.g. nvidia/canary-qwen-2.5b) or local SALM path",
    )
    p.add_argument(
        "out_path",
        type=Path,
        nargs="?",
        help="Output .gguf path. If omitted, derived from --repo-id or the "
             "model argument when it looks like an HF repo id.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HF repo id used to derive the output slug + variant when "
             "converting from a local path. Ignored if out_path is given.",
    )
    args = p.parse_args(argv[1:])

    variant, profile = resolve_variant(args.model, args.repo_id)

    out_path = args.out_path
    if out_path is None:
        out_path = REPO_ROOT / "models" / variant / gguf_name(variant, REFERENCE_DTYPE_LABEL)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    convert(args.model, out_path, variant, profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
