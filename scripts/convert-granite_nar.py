#!/usr/bin/env python3
"""
convert-granite_nar.py - convert IBM Granite Speech NLE (non-autoregressive
editor) HF checkpoint to a transcribe.cpp GGUF.

The NAR variant is structurally distinct from the AR granite family
(encoder-CTC + EncoderProjectorQFormer + bidirectional Granite-4 LM).
We emit a separate architecture key `granite_speech_nar` so the
C++ loader dispatches to src/arch/granite_nar/.

Tensor layout in HF state dict:

  encoder.input_linear.{weight,bias}
  encoder.out.{weight,bias}             # CTC head: 1024 -> 348
  encoder.out_mid.{weight,bias}         # bypass: 348 -> 1024
  encoder.out_bpe.{weight,bias}         # BPE CTC head: 1024 -> 100353 (NLE only)
  encoder.layers.{i}.{ff1, attn, conv, ff2, post_norm}.*
                                        # Same Conformer block as AR granite

  projector.layer_norms.{0..3}.{weight,bias}    # one per encoder_layer_idx
  projector.layer_projector.{weight,bias}        # 4096 -> 2048
  projector.out_norm.{weight,bias}               # LayerNorm 2048
  projector.out_linear.{weight,bias}             # 2048 -> 2048
  projector.query              [1, 3, 2048]      # 3 learned queries
  projector.window_positions   [1, 15, 2048]     # 15-frame window pos emb
  projector.qformer.layers.{0,1}.{attn_norm, cross_attention.*,
                                  mlp_norm, mlp.fc1, mlp.fc2}.*

  language_model.model.embed_tokens.weight   # tied lm_head; no separate output weight
  language_model.model.norm.weight
  language_model.model.layers.{i}.{input_layernorm, post_attention_layernorm,
                         self_attn.{q,k,v,o}_proj,
                         mlp.{gate,up,down}_proj}.*

Output GGUF tensor naming under three prefixes (enc., prj., dec.) to
mirror the AR granite layout where applicable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFWriter, GGUFValueType
from safetensors.torch import safe_open

REF_DTYPE = "BF16"
TOKEN_TYPE_NORMAL = 1
TOKEN_TYPE_CONTROL = 3
TOKEN_TYPE_USER_DEFINED = 4


def hf_resolve(model_arg: str, revision: str | None):
    """Return a local directory containing the HF model files."""
    p = Path(model_arg).expanduser().resolve()
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_arg, revision=revision))


def read_tokenizer(model_dir: Path) -> dict:
    """Read tokenizer.json (granite-4 BPE) → tokens / merges / types / specials."""
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    tcfg = json.loads((model_dir / "tokenizer_config.json").read_text())

    model = tok["model"]
    vocab = model["vocab"]  # str -> id
    merges = model.get("merges") or []
    added = tok.get("added_tokens") or []

    vocab_size = max(vocab.values()) + 1
    # added tokens may extend beyond the base vocab.
    for at in added:
        vocab_size = max(vocab_size, at["id"] + 1)

    tokens = [""] * vocab_size
    types = [TOKEN_TYPE_NORMAL] * vocab_size
    for piece, idx in vocab.items():
        tokens[idx] = piece
    added_ids = set()
    for at in added:
        i = at["id"]
        tokens[i] = at["content"]
        types[i] = TOKEN_TYPE_CONTROL if at.get("special", False) else TOKEN_TYPE_USER_DEFINED
        added_ids.add(i)

    # Normalize merges to "left right" form. Some new HF formats use
    # [left, right] pairs.
    norm_merges = []
    for m in merges:
        if isinstance(m, list):
            assert len(m) == 2
            norm_merges.append(f"{m[0]} {m[1]}")
        else:
            norm_merges.append(m)

    def _id(key):
        v = tcfg.get(key)
        if isinstance(v, dict):
            v = v.get("content")
        if v is None:
            return None
        for at in added:
            if at["content"] == v:
                return at["id"]
        return vocab.get(v)

    bos = _id("bos_token")
    eos = _id("eos_token")
    pad = _id("pad_token")
    unk = _id("unk_token")
    if pad is None:
        pad = vocab.get("<|pad|>")
    return {
        "tokens": tokens,
        "types":  types,
        "merges": norm_merges,
        "bos_id": bos,
        "eos_id": eos,
        "pad_id": pad,
        "unk_id": unk,
    }


def read_ctc_chars(config: dict) -> list[str]:
    """Build a 348-entry array: index -> single-char string.

    Index 0 = blank (empty string sentinel). Indices in the char2idx map
    are filled; gaps are empty strings.
    """
    ctc = (config.get("ctc_tokenizer_config") or {}).get("char2idx")
    if ctc is None:
        # Default: latin256 (32..255) + kana92 (0x30A1..0x30A1+91)
        ctc = {chr(n): n for n in range(32, 256)}
        ctc |= {chr(0x30A1 + n): 256 + n for n in range(92)}
    # Coerce values to ints.
    ctc = {k: int(v) for k, v in ctc.items()}
    vocab_size = max(ctc.values()) + 1
    bpe_dim = config.get("encoder_config", {}).get("output_dim", 348)
    vocab_size = max(vocab_size, bpe_dim)
    out = [""] * vocab_size
    for ch, idx in ctc.items():
        out[idx] = ch
    return out


def read_hparams(config: dict, preproc: dict, processor: dict | None) -> dict:
    enc = config["encoder_config"]
    prj = config["projector_config"]
    # NAR config renamed `llm_config` -> `text_config` between snapshots
    # 7d20732d (multi-file modeling) and 99a4df9 (single-file modeling).
    llm = config.get("text_config") or config["llm_config"]
    hp = {}

    # Encoder
    hp["enc_n_layers"]         = enc["num_layers"]
    hp["enc_hidden"]            = enc["hidden_dim"]
    hp["enc_n_heads"]           = enc["num_heads"]
    hp["enc_head_dim"]          = enc["dim_head"]
    hp["enc_input_dim"]         = enc["input_dim"]
    hp["enc_output_dim"]        = enc["output_dim"]
    hp["enc_bpe_output_dim"]    = enc.get("bpe_output_dim", 0)
    hp["enc_bpe_pool_window"]   = enc.get("bpe_pooling_window", 0)
    # BPE-CTC scheme:
    #   - 7d20732d snapshot: bpe_output_dim = vocab_size + 1, blank channel
    #     is at index 0, non-blank argmax mapped back via (argmax - 1).
    #   - 99a4df9  snapshot: bpe_output_dim = vocab_size, the blank channel
    #     IS one of the vocab channels (the BOS id 100257), no shift on
    #     decode. config["blank_token_id"] = 100257 in the new snapshot.
    # Read blank_token_id from the encoder_config (or fall back to root),
    # default to 0 to preserve the old scheme.
    hp["enc_bpe_blank_id"] = int(
        enc.get("blank_token_id") or config.get("blank_token_id") or 0
    )
    hp["enc_self_cond_layer"]   = enc.get("self_conditioning_layer", enc["num_layers"] // 2)
    hp["enc_feedforward_mult"]  = enc["feedforward_mult"]
    hp["enc_conv_kernel_size"]  = enc["conv_kernel_size"]
    hp["enc_conv_expansion"]    = enc["conv_expansion_factor"]
    hp["enc_max_pos_emb"]       = enc["max_pos_emb"]
    hp["enc_context_size"]      = enc["context_size"]

    # Multi-layer encoder feature consumption from NLEConfig.
    hp["enc_layer_indices"] = list(config.get("encoder_layer_indices") or [-1])

    # Projector
    hp["prj_n_layers"]            = prj["num_layers"]
    hp["prj_hidden"]              = prj["hidden_size"]
    hp["prj_mlp_ratio"]           = prj["mlp_ratio"]
    hp["prj_n_heads"]             = prj["num_heads"]
    hp["prj_encoder_dim"]         = prj["encoder_dim"]
    hp["prj_num_encoder_layers"]  = prj["num_encoder_layers"]
    hp["prj_layernorm_eps"]       = prj["layernorm_eps"]
    hp["prj_block_size"]          = prj["block_size"]
    hp["prj_downsample_rate"]     = prj["downsample_rate"]
    hp["prj_llm_dim"]             = prj["llm_dim"]
    hp["prj_attn_bias"]           = prj.get("attn_bias", True)
    hp["prj_mlp_bias"]            = prj.get("mlp_bias", True)

    # NLE scaling flag (project output / embedding_multiplier on LLM input)
    hp["scale_projected_embeddings"] = bool(config.get("scale_projected_embeddings", True))

    # Text LM (Granite-4)
    hp["dec_n_layers"]           = llm["num_hidden_layers"]
    hp["dec_hidden"]             = llm["hidden_size"]
    hp["dec_intermediate"]       = llm["intermediate_size"]
    hp["dec_n_heads"]            = llm["num_attention_heads"]
    hp["dec_n_kv_heads"]         = llm["num_key_value_heads"]
    hp["dec_head_dim"]           = llm.get("head_dim", llm["hidden_size"] // llm["num_attention_heads"])
    hp["dec_hidden_act"]         = llm["hidden_act"]
    hp["dec_rms_norm_eps"]       = llm["rms_norm_eps"]
    # 99a4df9 snapshot nests rope_theta under rope_parameters; the 7d20732d
    # snapshot stored it as a flat key.
    hp["dec_rope_theta"]         = (
        llm.get("rope_parameters", {}).get("rope_theta")
        or llm["rope_theta"]
    )
    hp["dec_max_pos_emb"]        = llm["max_position_embeddings"]
    hp["dec_tie_word_embeddings"] = bool(llm.get("tie_word_embeddings", True))
    hp["dec_vocab_size"]         = llm["vocab_size"]
    hp["dec_embedding_multiplier"] = float(llm["embedding_multiplier"])
    hp["dec_logits_scaling"]     = float(llm["logits_scaling"])
    hp["dec_attention_multiplier"] = float(llm["attention_multiplier"])
    hp["dec_residual_multiplier"] = float(llm["residual_multiplier"])
    hp["dec_bos_id"]             = llm["bos_token_id"]
    hp["dec_eos_id"]             = llm["eos_token_id"]
    hp["dec_pad_id"]             = llm.get("pad_token_id", 100256)

    # Frontend
    hp["fe_type"]         = "mel"
    hp["fe_sample_rate"]  = preproc.get("sampling_rate", 16000)
    hp["fe_num_mels"]     = preproc.get("n_mels", 80)
    hp["fe_n_fft"]        = preproc.get("n_fft", 512)
    hp["fe_win_length"]   = preproc.get("win_length", 400)
    hp["fe_hop_length"]   = preproc.get("hop_length", 160)
    hp["fe_window"]       = "hann_periodic"
    hp["fe_normalize"]    = "per_utterance"  # whisper-style log10/clamp/-8/4+1
    hp["fe_pad_mode"]     = "reflect"
    hp["fe_mel_norm"]     = "htk"

    return hp


def emit_mel_filterbank_and_window(writer: GGUFWriter, hp: dict) -> None:
    """Emit librosa-htk mel filterbank + periodic Hann window as tensors.

    Mirrors convert-granite.py so the C++ MelFrontend can load identical
    buffers at construction time.
    """
    import librosa
    n_fft = hp["fe_n_fft"]
    n_mels = hp["fe_num_mels"]
    win_length = hp["fe_win_length"]
    sr = hp["fe_sample_rate"]
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
                             htk=True, norm=None).astype(np.float32)
    writer.add_tensor("frontend.mel_filterbank", fb)
    # Periodic Hann window of win_length (torchaudio default for STFT).
    n = win_length
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)).astype(np.float32)
    writer.add_tensor("frontend.window", window)


# ----- Tensor name maps -----------------------------------------------------

# Top-level encoder tensors.
ENC_TOP_MAP = [
    ("encoder.input_linear.weight", "enc.input_linear.weight"),
    ("encoder.input_linear.bias",   "enc.input_linear.bias"),
    ("encoder.out.weight",          "enc.ctc_proj.weight"),
    ("encoder.out.bias",            "enc.ctc_proj.bias"),
    ("encoder.out_mid.weight",      "enc.ctc_bypass.weight"),
    ("encoder.out_mid.bias",        "enc.ctc_bypass.bias"),
    ("encoder.out_bpe.weight",      "enc.ctc_bpe.weight"),
    ("encoder.out_bpe.bias",        "enc.ctc_bpe.bias"),
]

# Encoder block suffix map. The HF Conformer block under encoder.layers.{i}
# uses the same naming as the AR granite encoder so we can mirror exactly.
ENC_BLOCK_MAP = [
    # FF1 macaron
    ("ff1.pre_norm.weight",          "norm_ff1.weight"),
    ("ff1.pre_norm.bias",            "norm_ff1.bias"),
    ("ff1.up_proj.weight",           "ff1_up.weight"),
    ("ff1.up_proj.bias",             "ff1_up.bias"),
    ("ff1.down_proj.weight",         "ff1_down.weight"),
    ("ff1.down_proj.bias",           "ff1_down.bias"),
    # Shaw block-local attention
    ("attn.pre_norm.weight",         "norm_attn.weight"),
    ("attn.pre_norm.bias",           "norm_attn.bias"),
    ("attn.to_q.weight",             "attn_q.weight"),
    ("attn.to_kv.weight",            "attn_kv.weight"),
    ("attn.to_out.weight",           "attn_out.weight"),
    ("attn.to_out.bias",             "attn_out.bias"),
    ("attn.rel_pos_emb.weight",      "attn_rel_pos_emb.weight"),
    # Conv module
    ("conv.norm.weight",             "norm_conv.weight"),
    ("conv.norm.bias",               "norm_conv.bias"),
    ("conv.up_conv.weight",          "conv_pointwise1.weight"),
    ("conv.up_conv.bias",            "conv_pointwise1.bias"),
    ("conv.depth_conv.conv.weight",  "conv_depthwise.weight"),
    ("conv.batch_norm.weight",       "conv_bn.weight"),
    ("conv.batch_norm.bias",         "conv_bn.bias"),
    ("conv.batch_norm.running_mean", "conv_bn.running_mean"),
    ("conv.batch_norm.running_var",  "conv_bn.running_var"),
    ("conv.down_conv.weight",        "conv_pointwise2.weight"),
    ("conv.down_conv.bias",          "conv_pointwise2.bias"),
    # FF2 macaron
    ("ff2.pre_norm.weight",          "norm_ff2.weight"),
    ("ff2.pre_norm.bias",            "norm_ff2.bias"),
    ("ff2.up_proj.weight",           "ff2_up.weight"),
    ("ff2.up_proj.bias",             "ff2_up.bias"),
    ("ff2.down_proj.weight",         "ff2_down.weight"),
    ("ff2.down_proj.bias",           "ff2_down.bias"),
    # Post-block LN
    ("post_norm.weight",             "norm_post.weight"),
    ("post_norm.bias",               "norm_post.bias"),
]
# Tensors we read but DON'T emit (bookkeeping for BN).
ENC_BLOCK_SKIP = {"conv.batch_norm.num_batches_tracked"}

# Projector top-level tensors.
PROJ_TOP_MAP = [
    ("projector.layer_projector.weight", "prj.layer_projector.weight"),
    ("projector.layer_projector.bias",   "prj.layer_projector.bias"),
    ("projector.out_norm.weight",        "prj.out_norm.weight"),
    ("projector.out_norm.bias",          "prj.out_norm.bias"),
    ("projector.out_linear.weight",      "prj.out_linear.weight"),
    ("projector.out_linear.bias",        "prj.out_linear.bias"),
    ("projector.query",                  "prj.query"),
    ("projector.window_positions",       "prj.window_positions"),
]

# Per-encoder-layer LayerNorm in the projector. There are
# num_encoder_layers of these (4 for NLE).
PROJ_LN_FMT = [
    ("projector.layer_norms.{i}.weight", "prj.layer_norms.{i}.weight"),
    ("projector.layer_norms.{i}.bias",   "prj.layer_norms.{i}.bias"),
]

# Per-Q-Former-layer.
PROJ_BLOCK_MAP = [
    ("attn_norm.weight",              "norm_attn.weight"),
    ("attn_norm.bias",                "norm_attn.bias"),
    ("cross_attention.q_proj.weight", "cross_attn_q.weight"),
    ("cross_attention.q_proj.bias",   "cross_attn_q.bias"),
    ("cross_attention.k_proj.weight", "cross_attn_k.weight"),
    ("cross_attention.k_proj.bias",   "cross_attn_k.bias"),
    ("cross_attention.v_proj.weight", "cross_attn_v.weight"),
    ("cross_attention.v_proj.bias",   "cross_attn_v.bias"),
    ("cross_attention.o_proj.weight", "cross_attn_o.weight"),
    ("cross_attention.o_proj.bias",   "cross_attn_o.bias"),
    ("mlp_norm.weight",               "norm_ffn.weight"),
    ("mlp_norm.bias",                 "norm_ffn.bias"),
    ("mlp.fc1.weight",                "ffn_fc1.weight"),
    ("mlp.fc1.bias",                  "ffn_fc1.bias"),
    ("mlp.fc2.weight",                "ffn_fc2.weight"),
    ("mlp.fc2.bias",                  "ffn_fc2.bias"),
]

# Granite-4 LLM. Mirrors AR convert-granite.py decoder layout (dec.*).
DEC_TOP_MAP = [
    ("language_model.model.embed_tokens.weight", "dec.token_embd.weight"),
    ("language_model.model.norm.weight",         "dec.output_norm.weight"),
]
DEC_BLOCK_MAP = [
    ("input_layernorm.weight",          "norm_attn.weight"),
    ("post_attention_layernorm.weight", "norm_ffn.weight"),
    ("self_attn.q_proj.weight",         "attn.q.weight"),
    ("self_attn.k_proj.weight",         "attn.k.weight"),
    ("self_attn.v_proj.weight",         "attn.v.weight"),
    ("self_attn.o_proj.weight",         "attn.o.weight"),
    ("mlp.gate_proj.weight",            "ffn.gate.weight"),
    ("mlp.up_proj.weight",              "ffn.up.weight"),
    ("mlp.down_proj.weight",            "ffn.down.weight"),
]


def _to_bf16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.bfloat16) if t.dtype != torch.bfloat16 else t.detach()


def _to_f32(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to(torch.float32).contiguous()


def add_bf16(writer: GGUFWriter, name: str, t: torch.Tensor) -> None:
    # ggml expects BF16 stored as uint16. The gguf writer handles dtype.
    arr = _to_bf16(t).view(torch.uint16).cpu().numpy().reshape(tuple(t.shape))
    from gguf import GGMLQuantizationType
    writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.BF16)


def add_f32(writer: GGUFWriter, name: str, t: torch.Tensor) -> None:
    arr = _to_f32(t).cpu().numpy()
    writer.add_tensor(name, arr)


# Choose dtype based on tensor name. BN / LayerNorm scalars and the
# float multipliers stay F32 (small + numerically sensitive). Linear
# weights stay BF16 (matches ggml's native granite ops).
F32_TENSOR_SUFFIXES = (
    "norm.weight", "norm.bias",
    "running_mean", "running_var",
    "norm_ff1.weight", "norm_ff1.bias",
    "norm_ff2.weight", "norm_ff2.bias",
    "norm_attn.weight", "norm_attn.bias",
    "norm_conv.weight", "norm_conv.bias",
    "norm_post.weight", "norm_post.bias",
    "norm_ffn.weight", "norm_ffn.bias",
    "out_norm.weight", "out_norm.bias",
    "conv_bn.weight", "conv_bn.bias",
    "frontend.mel_filterbank", "frontend.window",
)


CONV_KERNEL_SUFFIXES = (
    "conv_pointwise1.weight",
    "conv_pointwise2.weight",
    "conv_depthwise.weight",
)


def add_f16(writer: GGUFWriter, name: str, t: torch.Tensor) -> None:
    arr = _to_f32(t).cpu().numpy().astype(np.float16)
    from gguf import GGMLQuantizationType
    writer.add_tensor(name, arr, raw_dtype=GGMLQuantizationType.F16)


def add_tensor(writer: GGUFWriter, name: str, t: torch.Tensor) -> None:
    # Biases and norm scalars stay F32 for numerical sensitivity (matches
    # gguf_common.reference_dtype_for and the AR granite converter). Any
    # tensor whose path contains "norm" and ends in .weight is treated as
    # a norm scalar — covers prj.layer_norms.{i}.weight, dec.output_norm,
    # the encoder block norm_* tensors, and the projector out_norm /
    # qformer norm_attn / norm_ffn slots. Conv kernels must be F16 (the
    # ggml conv1d op rejects BF16).
    is_norm_weight = ("norm" in name) and name.endswith(".weight")
    is_f32 = name.endswith(".bias") or \
             is_norm_weight or \
             any(name.endswith(s) for s in F32_TENSOR_SUFFIXES)
    is_conv_kernel = any(name.endswith(s) for s in CONV_KERNEL_SUFFIXES)
    if is_f32:
        add_f32(writer, name, t)
    elif is_conv_kernel:
        add_f16(writer, name, t)
    else:
        add_bf16(writer, name, t)


# ----- Main -----------------------------------------------------------------


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("model", help="HF repo id or local model directory")
    p.add_argument("--repo-id", default=None,
                   help="HF repo id used to derive output filename (defaults to last segment of model)")
    p.add_argument("--revision", default=None,
                   help="HF revision (ignored for local paths)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: models/<repo-last-segment>/)")
    args = p.parse_args(argv)

    repo_id = args.repo_id or args.model
    variant = repo_id.split("/")[-1]

    model_dir = hf_resolve(args.model, args.revision)
    print(f"Source: {model_dir}")

    config = json.loads((model_dir / "config.json").read_text())
    preproc = json.loads((model_dir / "preprocessor_config.json").read_text())

    hp = read_hparams(config, preproc, None)
    tok = read_tokenizer(model_dir)
    ctc_chars = read_ctc_chars(config)

    print(f"variant: {variant}")
    print(f"  encoder: {hp['enc_n_layers']} layers, hidden={hp['enc_hidden']}, "
          f"layer_indices={hp['enc_layer_indices']}")
    print(f"  projector: {hp['prj_n_layers']} qformer layers, "
          f"hidden={hp['prj_hidden']}, n_heads={hp['prj_n_heads']}, "
          f"block_size={hp['prj_block_size']}, downsample={hp['prj_downsample_rate']}")
    print(f"  LLM: {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, "
          f"GQA {hp['dec_n_heads']}/{hp['dec_n_kv_heads']}, head={hp['dec_head_dim']}")
    print(f"  CTC vocab: {len(ctc_chars)} entries (blank=0, char range varies)")
    print(f"  LLM tokenizer: {len(tok['tokens'])} tokens, "
          f"{len(tok['merges'])} merges, "
          f"specials bos={tok['bos_id']} eos={tok['eos_id']} pad={tok['pad_id']}")

    outdir = Path(args.outdir or f"models/{variant}").expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"{variant}-{REF_DTYPE}.gguf"
    print(f"Writing GGUF: {out_path}")

    writer = GGUFWriter(str(out_path), "granite_speech_nar")

    # ---- general.* ----
    writer.add_string("general.basename", "granite-speech-nar")
    languages = ["en", "fr", "de", "es", "pt"]
    writer.add_array("general.languages", languages)

    # ---- stt.variant + capabilities ----
    writer.add_string("stt.variant", variant)
    writer.add_bool("stt.capability.translation", False)
    writer.add_bool("stt.capability.lang_detect", False)
    writer.add_bool("stt.capability.word_timestamps", False)
    writer.add_bool("stt.capability.speaker_diarization", False)

    # ---- tokenizer.ggml.* (granite-4 BPE) ----
    writer.add_string("tokenizer.ggml.model", "gpt2")
    writer.add_string("tokenizer.ggml.pre",   "granite")
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

    # ---- CTC char table (NLE-specific) ----
    writer.add_array("stt.granite_nar.ctc_chars", ctc_chars)

    # ---- stt.granite_nar.encoder.* ----
    writer.add_uint32("stt.granite_nar.encoder.n_layers", hp["enc_n_layers"])
    writer.add_uint32("stt.granite_nar.encoder.hidden", hp["enc_hidden"])
    writer.add_uint32("stt.granite_nar.encoder.n_heads", hp["enc_n_heads"])
    writer.add_uint32("stt.granite_nar.encoder.head_dim", hp["enc_head_dim"])
    writer.add_uint32("stt.granite_nar.encoder.input_dim", hp["enc_input_dim"])
    writer.add_uint32("stt.granite_nar.encoder.output_dim", hp["enc_output_dim"])
    writer.add_uint32("stt.granite_nar.encoder.bpe_output_dim", hp["enc_bpe_output_dim"])
    writer.add_uint32("stt.granite_nar.encoder.bpe_pool_window", hp["enc_bpe_pool_window"])
    writer.add_uint32("stt.granite_nar.encoder.bpe_blank_id", hp["enc_bpe_blank_id"])
    writer.add_uint32("stt.granite_nar.encoder.self_cond_layer", hp["enc_self_cond_layer"])
    writer.add_uint32("stt.granite_nar.encoder.feedforward_mult", hp["enc_feedforward_mult"])
    writer.add_uint32("stt.granite_nar.encoder.conv_kernel_size", hp["enc_conv_kernel_size"])
    writer.add_uint32("stt.granite_nar.encoder.conv_expansion", hp["enc_conv_expansion"])
    writer.add_uint32("stt.granite_nar.encoder.max_pos_emb", hp["enc_max_pos_emb"])
    writer.add_uint32("stt.granite_nar.encoder.context_size", hp["enc_context_size"])
    # encoder_layer_indices is a list of int32 (may include negatives).
    writer.add_array("stt.granite_nar.encoder.layer_indices",
                     [int(x) for x in hp["enc_layer_indices"]])

    # ---- stt.granite_nar.projector.* ----
    writer.add_uint32("stt.granite_nar.projector.n_layers", hp["prj_n_layers"])
    writer.add_uint32("stt.granite_nar.projector.hidden", hp["prj_hidden"])
    writer.add_uint32("stt.granite_nar.projector.mlp_ratio", hp["prj_mlp_ratio"])
    writer.add_uint32("stt.granite_nar.projector.n_heads", hp["prj_n_heads"])
    writer.add_uint32("stt.granite_nar.projector.encoder_dim", hp["prj_encoder_dim"])
    writer.add_uint32("stt.granite_nar.projector.num_encoder_layers", hp["prj_num_encoder_layers"])
    writer.add_uint32("stt.granite_nar.projector.block_size", hp["prj_block_size"])
    writer.add_uint32("stt.granite_nar.projector.downsample_rate", hp["prj_downsample_rate"])
    writer.add_uint32("stt.granite_nar.projector.llm_dim", hp["prj_llm_dim"])
    writer.add_float32("stt.granite_nar.projector.layernorm_eps", hp["prj_layernorm_eps"])
    writer.add_bool("stt.granite_nar.projector.attn_bias", hp["prj_attn_bias"])
    writer.add_bool("stt.granite_nar.projector.mlp_bias", hp["prj_mlp_bias"])
    writer.add_bool("stt.granite_nar.scale_projected_embeddings", hp["scale_projected_embeddings"])

    # ---- stt.granite_nar.text.* ----
    writer.add_uint32("stt.granite_nar.text.n_layers", hp["dec_n_layers"])
    writer.add_uint32("stt.granite_nar.text.hidden", hp["dec_hidden"])
    writer.add_uint32("stt.granite_nar.text.intermediate", hp["dec_intermediate"])
    writer.add_uint32("stt.granite_nar.text.n_heads", hp["dec_n_heads"])
    writer.add_uint32("stt.granite_nar.text.n_kv_heads", hp["dec_n_kv_heads"])
    writer.add_uint32("stt.granite_nar.text.head_dim", hp["dec_head_dim"])
    writer.add_string("stt.granite_nar.text.hidden_act", hp["dec_hidden_act"])
    writer.add_float32("stt.granite_nar.text.rms_norm_eps", hp["dec_rms_norm_eps"])
    writer.add_float32("stt.granite_nar.text.rope_theta", hp["dec_rope_theta"])
    writer.add_uint32("stt.granite_nar.text.max_position_embeddings", hp["dec_max_pos_emb"])
    writer.add_bool("stt.granite_nar.text.tie_word_embeddings", hp["dec_tie_word_embeddings"])
    writer.add_uint32("stt.granite_nar.text.vocab_size", hp["dec_vocab_size"])
    writer.add_float32("stt.granite_nar.text.embedding_multiplier", hp["dec_embedding_multiplier"])
    writer.add_float32("stt.granite_nar.text.logits_scaling", hp["dec_logits_scaling"])
    writer.add_float32("stt.granite_nar.text.attention_multiplier", hp["dec_attention_multiplier"])
    writer.add_float32("stt.granite_nar.text.residual_multiplier", hp["dec_residual_multiplier"])
    writer.add_uint32("stt.granite_nar.text.bos_id", hp["dec_bos_id"])
    writer.add_uint32("stt.granite_nar.text.eos_id", hp["dec_eos_id"])
    writer.add_uint32("stt.granite_nar.text.pad_id", hp["dec_pad_id"])

    # ---- stt.frontend.* ----
    writer.add_string("stt.frontend.type",        hp["fe_type"])
    writer.add_uint32("stt.frontend.sample_rate", hp["fe_sample_rate"])
    writer.add_uint32("stt.frontend.num_mels",    hp["fe_num_mels"])
    writer.add_uint32("stt.frontend.n_fft",       hp["fe_n_fft"])
    writer.add_uint32("stt.frontend.win_length",  hp["fe_win_length"])
    writer.add_uint32("stt.frontend.hop_length",  hp["fe_hop_length"])
    writer.add_string("stt.frontend.window",      hp["fe_window"])
    writer.add_string("stt.frontend.normalize",   hp["fe_normalize"])
    writer.add_string("stt.frontend.pad_mode",    hp["fe_pad_mode"])
    writer.add_string("stt.frontend.mel_norm",    hp["fe_mel_norm"])

    # ---- Frontend tensors (mel filterbank + window) ----
    emit_mel_filterbank_and_window(writer, hp)

    # ---- Read weights from safetensors ----
    sf_files = sorted(model_dir.glob("*.safetensors"))
    if not sf_files:
        raise SystemExit(f"no safetensors files in {model_dir}")

    n_emitted = 0
    consumed: set[str] = set()

    def emit(src_key: str, dst_key: str, source_handles, key_sets) -> None:
        nonlocal n_emitted
        for sh, keys in zip(source_handles, key_sets):
            if src_key in keys:
                t = sh.get_tensor(src_key)
                add_tensor(writer, dst_key, t)
                consumed.add(src_key)
                n_emitted += 1
                return
        raise KeyError(f"missing tensor: {src_key}")

    # Open shards.
    handles = [safe_open(sf, framework="pt") for sf in sf_files]
    key_sets = [set(sh.keys()) for sh in handles]
    try:
        all_keys = set()
        for ks in key_sets:
            all_keys |= ks

        # Encoder top-level.
        for src, dst in ENC_TOP_MAP:
            emit(src, dst, handles, key_sets)

        # Encoder layers.
        for i in range(hp["enc_n_layers"]):
            for src_suf, dst_suf in ENC_BLOCK_MAP:
                emit(f"encoder.layers.{i}.{src_suf}",
                     f"enc.blocks.{i}.{dst_suf}", handles, key_sets)
            for skip in ENC_BLOCK_SKIP:
                consumed.add(f"encoder.layers.{i}.{skip}")

        # Projector top-level.
        for src, dst in PROJ_TOP_MAP:
            emit(src, dst, handles, key_sets)
        # Per-encoder-layer LayerNorms.
        for j in range(hp["prj_num_encoder_layers"]):
            for src_fmt, dst_fmt in PROJ_LN_FMT:
                emit(src_fmt.format(i=j), dst_fmt.format(i=j), handles, key_sets)
        # Projector Q-Former layers.
        for i in range(hp["prj_n_layers"]):
            for src_suf, dst_suf in PROJ_BLOCK_MAP:
                emit(f"projector.qformer.layers.{i}.{src_suf}",
                     f"prj.blocks.{i}.{dst_suf}", handles, key_sets)

        # LLM top-level.
        for src, dst in DEC_TOP_MAP:
            emit(src, dst, handles, key_sets)
        # LLM layers.
        for i in range(hp["dec_n_layers"]):
            for src_suf, dst_suf in DEC_BLOCK_MAP:
                emit(f"language_model.model.layers.{i}.{src_suf}",
                     f"dec.blocks.{i}.{dst_suf}", handles, key_sets)

        leftover = sorted(all_keys - consumed)
        if leftover:
            print(f"WARNING: {len(leftover)} HF tensors not consumed:")
            for k in leftover[:20]:
                print(f"  {k}")
            if len(leftover) > 20:
                print(f"  ... and {len(leftover) - 20} more")
    finally:
        for sh in handles:
            sh.__exit__(None, None, None)

    print(f"Emitted {n_emitted} tensors")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Done: {out_path}")
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"sha256: {sha}")
    print(f"bytes:  {out_path.stat().st_size:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
