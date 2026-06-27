#!/usr/bin/env python3
"""
convert-vibevoice.py - convert the microsoft/VibeVoice-ASR checkpoint to a
BF16 reference GGUF that transcribe.cpp's loader can ingest. Block
quantization (Q8_0, Q5_K_M, ...) is a Stage 5 concern and goes through
tools/transcribe-quantize.

Source format:
    HuggingFace repo `microsoft/VibeVoice-ASR` — config.json + 8 BF16
    safetensors shards ONLY. No modeling code, processor, tokenizer, or
    generation_config ship with the weights; the reference implementation
    lives in the `vibevoice` package (vendored under
    models/_vendor/VibeVoice) and the tokenizer is the stock
    `Qwen/Qwen2.5-7B` byte-level BPE, pulled separately here.

Architecture: audio-llm. Raw 24 kHz waveform feeds two parallel causal-conv
VAE encoders (acoustic vae_dim 64, semantic vae_dim 128); each encoder's
latent `.mean` is projected to the Qwen2.5 hidden size (3584) by a
SpeechConnector, the two projections are summed, and the result is scattered
into the Qwen2.5-7B LM token stream at the speech-pad positions. State dict
prefixes:

    model.language_model.*          -> Qwen2.5-7B causal LM (28 layers, GQA
                                       28/4, head_dim 128, q/k/v biases,
                                       SwiGLU, rope_theta 1e6)
    lm_head.weight                  -> output projection (untied)
    model.acoustic_tokenizer.encoder.*  -> acoustic VAE encoder  (KEPT)
    model.acoustic_tokenizer.decoder.*  -> acoustic VAE decoder  (DROPPED:
                                           TTS-only, never on the ASR path)
    model.semantic_tokenizer.encoder.*  -> semantic VAE encoder  (KEPT;
                                           semantic tokenizer is encoder-only)
    model.acoustic_connector.*      -> acoustic latent (64)  -> 3584 projector
    model.semantic_connector.*      -> semantic latent (128) -> 3584 projector

The acoustic-tokenizer DECODER (276 tensors) and the config's diffusion-head
are the VibeVoice TTS generation path; ASR calls only `.encode(...).mean`, so
they are dropped. This is confirmed by the Stage-2 oracle dumper
(scripts/dump_reference_vibevoice_author.py): the ASR forward only touches the
two encoders, the two connectors, and the LM.

Layout conversions: NONE. Linear weights stay (out, in); conv1d kernels stay
(out, in, k) which is what ggml expects.

KV emitted (see body for the full list):
    general.architecture   = "vibevoice"
    stt.variant            = "vibevoice-asr"
    tokenizer.ggml.*       = Qwen2.5-7B gpt2 byte-level BPE + chat template
    stt.vibevoice.decoder.*       Qwen2.5 LM hparams
    stt.vibevoice.acoustic.*      acoustic VAE encoder hparams
    stt.vibevoice.semantic.*      semantic VAE encoder hparams
    stt.vibevoice.speech_*_token_id / speech_tok_compress_ratio / system_prompt
    stt.frontend.*                raw-waveform frontend (sample_rate only)

CLI:
    uv run --project scripts/envs/vibevoice \
      scripts/convert-vibevoice.py microsoft/VibeVoice-ASR \
        --revision d0c9efdb8d614685062c04425d91e01b6f37d944

Single-file, top-to-bottom — no hidden helpers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType
from huggingface_hub import snapshot_download
from safetensors import safe_open


class _ShardedSafetensors:
    """Multi-file safe_open shim.

    Reads model.safetensors.index.json, opens each shard lazily, and exposes
    the same keys()/get_tensor() surface as safe_open so the converter
    streams one tensor at a time and never materializes the 9B model.
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
    encode_for_gguf,
    gguf_name,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DTYPE_LABEL = "BF16"
REFERENCE_FILE_TYPE = LlamaFileType.MOSTLY_BF16
REFERENCE_GGML_TYPE = GGMLQuantizationType.BF16

# The HF weights repo ships no tokenizer; the processor loads the stock
# Qwen2.5-7B byte-level BPE (vocab padded to 152064). We pull just the
# tokenizer files, not the 15 GB of Qwen weights.
TOKENIZER_REPO = "Qwen/Qwen2.5-7B"
TOKENIZER_FILES = ["vocab.json", "merges.txt", "tokenizer_config.json"]

# Constants lifted verbatim from the vendored VibeVoice ASR processor /
# tokenizer (models/_vendor/VibeVoice/vibevoice/...). The processor builds the
# LM prompt dynamically (it interpolates the audio duration and a speech-pad
# run of length ceil(n_samples / compress_ratio)); we bake the static pieces
# into the GGUF so the C++ runtime can reconstruct it.
SYSTEM_PROMPT = (
    "You are a helpful assistant that transcribes audio input into text "
    "output in JSON format."
)
CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' "
    "+ message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if "
    "add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)
# Speech markers REUSE existing Qwen2.5 special tokens (no new vocab).
SPEECH_START_TOKEN = "<|object_ref_start|>"
SPEECH_END_TOKEN = "<|object_ref_end|>"
SPEECH_PAD_TOKEN = "<|box_start|>"
SPEECH_TOK_COMPRESS_RATIO = 3200  # product of encoder_ratios [8,5,5,4,2,2]
TARGET_SAMPLE_RATE = 24000
GENERATION_STOP_STRING = "}]"


# ---------------------------------------------------------------------------
# Tokenizer extraction (Qwen2.5 / Qwen2Tokenizer — byte-level BPE)
# ---------------------------------------------------------------------------


def extract_tokenizer(tok_dir: Path, vocab_size: int) -> dict:
    vocab_path = tok_dir / "vocab.json"
    merges_path = tok_dir / "merges.txt"
    tokcfg_path = tok_dir / "tokenizer_config.json"

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
            f"tokenizer has id {max_id} but config vocab_size={vocab_size}")

    content_to_id = {content: tid for tid, (content, _) in tok_by_id.items()}

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

    def _name_to_id(name_field: str) -> int | None:
        tok = tokcfg.get(name_field)
        if isinstance(tok, dict):
            tok = tok.get("content")
        if not tok:
            return None
        return content_to_id.get(tok)

    def _content_id(content: str) -> int:
        tid = content_to_id.get(content)
        if tid is None:
            raise KeyError(f"expected token {content!r} in {TOKENIZER_REPO} vocab")
        return tid

    return {
        "tokens": tokens,
        "types": types,
        "merges": merges,
        "eos_id": _name_to_id("eos_token"),
        "pad_id": _name_to_id("pad_token"),
        "bos_id": _name_to_id("bos_token"),
        "speech_start_id": _content_id(SPEECH_START_TOKEN),
        "speech_end_id": _content_id(SPEECH_END_TOKEN),
        "speech_pad_id": _content_id(SPEECH_PAD_TOKEN),
        "im_start_id": _content_id("<|im_start|>"),
        "im_end_id": _content_id("<|im_end|>"),
    }


# ---------------------------------------------------------------------------
# Per-tensor reference dtype policy (BF16 reference)
# ---------------------------------------------------------------------------
#
# Local rather than gguf_common.reference_dtype_for because VibeVoice carries
# tensor shapes that family's parakeet-tuned heuristics don't cover: the VAE
# encoders' layer-scale `gamma` / `ffn_gamma` vectors and their bare
# `norm.weight` / `ffn_norm.weight` RMSNorms must stay F32, and all of their
# causal-conv kernels go to F16 (the loader has no BF16 conv path; F16 also
# has more mantissa than BF16, so this is the accuracy-preserving choice).


def vibe_dtype_for(name: str) -> GGMLQuantizationType:
    if name.endswith(".bias"):
        return GGMLQuantizationType.F32
    if name.endswith(".gamma") or name.endswith("_gamma"):
        return GGMLQuantizationType.F32
    if (name.endswith("norm.weight")
            or name.endswith("norm_attn.weight")
            or name.endswith("norm_ffn.weight")):
        return GGMLQuantizationType.F32
    if ".conv." in name and name.endswith(".weight"):
        return GGMLQuantizationType.F16
    return REFERENCE_GGML_TYPE


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------


def passthrough(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr)


# Qwen2.5 LM top-level (untied head: lm_head.weight ships separately).
TEXT_TOP_TABLE: list[tuple[str, str]] = [
    ("model.language_model.embed_tokens.weight", "dec.token_embd.weight"),
    ("model.language_model.norm.weight",         "dec.output_norm.weight"),
    ("lm_head.weight",                           "dec.output.weight"),
]

# Per-LM-layer block (Qwen2: q/k/v carry biases, o_proj does not).
TEXT_BLOCK_TABLE: list[tuple[str, str]] = [
    ("input_layernorm.weight",          "norm_attn.weight"),
    ("post_attention_layernorm.weight", "norm_ffn.weight"),
    ("self_attn.q_proj.weight",         "attn.q.weight"),
    ("self_attn.q_proj.bias",           "attn.q.bias"),
    ("self_attn.k_proj.weight",         "attn.k.weight"),
    ("self_attn.k_proj.bias",           "attn.k.bias"),
    ("self_attn.v_proj.weight",         "attn.v.weight"),
    ("self_attn.v_proj.bias",           "attn.v.bias"),
    ("self_attn.o_proj.weight",         "attn.o.weight"),
    ("mlp.gate_proj.weight",            "ffn.gate.weight"),
    ("mlp.up_proj.weight",              "ffn.up.weight"),
    ("mlp.down_proj.weight",            "ffn.down.weight"),
]

# SpeechConnector (fc1 -> RMSNorm -> fc2), one per stream.
CONNECTOR_SUFFIXES = ["fc1.weight", "fc1.bias", "norm.weight",
                      "fc2.weight", "fc2.bias"]

# VAE encoder prefixes: every key under these is copied verbatim (suffix
# preserved) under the enc.<stream>. namespace. The decoder prefix is dropped.
ENC_PREFIXES = {
    "acoustic": "model.acoustic_tokenizer.encoder.",
    "semantic": "model.semantic_tokenizer.encoder.",
}
DROP_PREFIX = "model.acoustic_tokenizer.decoder."


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


def read_hparams(config: dict) -> dict:
    dec = config["decoder_config"]
    ac = config["acoustic_tokenizer_config"]
    sc = config["semantic_tokenizer_config"]

    n_heads = int(dec["num_attention_heads"])
    hidden = int(dec["hidden_size"])

    def enc_hp(cfg: dict) -> dict:
        return {
            "vae_dim": int(cfg["vae_dim"]),
            "n_filters": int(cfg["encoder_n_filters"]),
            "ratios": [int(r) for r in cfg["encoder_ratios"]],
            "depths": str(cfg["encoder_depths"]),
            "mixer": str(cfg["mixer_layer"]),
            "layernorm": str(cfg["layernorm"]),
            "layernorm_eps": float(cfg.get("layernorm_eps", 1e-6)),
            "layer_scale_init": float(cfg.get("layer_scale_init_value", 1e-6)),
            "causal": bool(cfg["causal"]),
            "fix_std": float(cfg.get("fix_std", 0.0)),
            "std_dist": str(cfg.get("std_dist_type", "none")),
            "disable_last_norm": bool(cfg.get("disable_last_norm", False)),
            "pad_mode": str(cfg.get("pad_mode", "constant")),
            "conv_bias": bool(cfg.get("conv_bias", True)),
        }

    return {
        "dec_n_layers":     int(dec["num_hidden_layers"]),
        "dec_hidden":       hidden,
        "dec_intermediate": int(dec["intermediate_size"]),
        "dec_n_heads":      n_heads,
        "dec_n_kv_heads":   int(dec["num_key_value_heads"]),
        "dec_head_dim":     int(dec.get("head_dim", hidden // n_heads)),
        "dec_hidden_act":   str(dec.get("hidden_act", "silu")).lower(),
        "dec_rms_norm_eps": float(dec["rms_norm_eps"]),
        "dec_rope_theta":   float(dec["rope_theta"]),
        "dec_max_pos_emb":  int(dec["max_position_embeddings"]),
        "dec_tie_embeddings": bool(dec.get("tie_word_embeddings", False)),
        "dec_vocab_size":   int(dec["vocab_size"]),
        "acoustic_vae_dim": int(config["acoustic_vae_dim"]),
        "semantic_vae_dim": int(config["semantic_vae_dim"]),
        "acoustic":         enc_hp(ac),
        "semantic":         enc_hp(sc),
    }


def compute_size_label(total_params: int) -> str:
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.0f}M"
    return f"{total_params / 1_000:.0f}K"


def load_intake_capabilities() -> tuple[list[str], bool]:
    """Languages + lang-detect flag from the Stage-1 intake (single source)."""
    intake_path = (REPO_ROOT / "reports" / "porting" / "vibevoice"
                   / "vibevoice-asr" / "intake.json")
    with intake_path.open() as f:
        intake = json.load(f)
    caps = intake.get("capabilities", {})
    return (list(caps.get("languages", [])),
            bool(caps.get("language_detection", False)))


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert(model_dir: Path, tok_dir: Path, out_path: Path, variant: str) -> None:
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    with (model_dir / "config.json").open() as f:
        config = json.load(f)
    hp = read_hparams(config)
    languages, lang_detect = load_intake_capabilities()

    print(f"LM: Qwen2.5 {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, "
          f"heads={hp['dec_n_heads']}/{hp['dec_n_kv_heads']}, "
          f"vocab={hp['dec_vocab_size']}, tie={hp['dec_tie_embeddings']}")
    print(f"Acoustic VAE: vae_dim={hp['acoustic_vae_dim']} "
          f"ratios={hp['acoustic']['ratios']} depths={hp['acoustic']['depths']}")
    print(f"Semantic VAE: vae_dim={hp['semantic_vae_dim']} "
          f"ratios={hp['semantic']['ratios']} depths={hp['semantic']['depths']}")

    print(f"Reading Qwen2.5 tokenizer from {tok_dir}")
    tok = extract_tokenizer(tok_dir, hp["dec_vocab_size"])
    print(f"  eos={tok['eos_id']} pad={tok['pad_id']} "
          f"speech_start={tok['speech_start_id']} speech_end={tok['speech_end_id']} "
          f"speech_pad={tok['speech_pad_id']}")

    with _ShardedSafetensors(model_dir) as st:
        st_keys = set(st.keys())

        # Build the full (src -> dst) plan up front so we can validate
        # coverage before writing a 13 GB file.
        plan: list[tuple[str, str]] = list(TEXT_TOP_TABLE)
        for i in range(hp["dec_n_layers"]):
            for s_src, s_dst in TEXT_BLOCK_TABLE:
                plan.append((f"model.language_model.layers.{i}.{s_src}",
                             f"dec.blocks.{i}.{s_dst}"))
        for stream in ("acoustic", "semantic"):
            for suffix in CONNECTOR_SUFFIXES:
                plan.append((f"model.{stream}_connector.{suffix}",
                             f"conn.{stream}.{suffix}"))
        for stream, prefix in ENC_PREFIXES.items():
            for k in sorted(st_keys):
                if k.startswith(prefix):
                    plan.append((k, f"enc.{stream}." + k[len(prefix):]))

        planned_src = {s for s, _ in plan}
        dropped = sorted(k for k in st_keys if k.startswith(DROP_PREFIX))
        unexpected = sorted(st_keys - planned_src - set(dropped))
        if unexpected:
            raise RuntimeError(
                "unplanned safetensors keys (neither mapped nor dropped):\n  "
                + "\n  ".join(unexpected[:30]))
        missing = sorted(planned_src - st_keys)
        if missing:
            raise KeyError("planned tensors absent from checkpoint:\n  "
                           + "\n  ".join(missing[:30]))
        print(f"Plan: {len(plan)} tensors kept, {len(dropped)} dropped "
              f"(acoustic-tokenizer decoder / TTS dead weight)")

        total = sum(st.get_tensor(s).numel() for s, _ in plan)
        size_label = compute_size_label(total)
        print(f"Kept params: {total:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = GGUFWriter(str(out_path), "vibevoice")

        # ---- general.* ----
        writer.add_string("general.basename",   "vibevoice")
        writer.add_string("general.size_label", size_label)
        writer.add_uint32("general.file_type",  int(REFERENCE_FILE_TYPE))
        writer.add_array("general.languages",    languages)

        # ---- stt.variant + capabilities ----
        writer.add_string("stt.variant", variant)
        writer.add_bool("stt.capability.lang_detect", lang_detect)
        writer.add_bool("stt.capability.diarize", True)

        # ---- tokenizer.ggml.* (Qwen2.5 byte-level BPE) ----
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
        writer.add_string("tokenizer.chat_template", CHAT_TEMPLATE)

        # ---- stt.vibevoice.decoder.* (Qwen2.5 LM) ----
        writer.add_uint32("stt.vibevoice.decoder.n_layers",          hp["dec_n_layers"])
        writer.add_uint32("stt.vibevoice.decoder.hidden_size",       hp["dec_hidden"])
        writer.add_uint32("stt.vibevoice.decoder.intermediate_size", hp["dec_intermediate"])
        writer.add_uint32("stt.vibevoice.decoder.n_heads",           hp["dec_n_heads"])
        writer.add_uint32("stt.vibevoice.decoder.n_kv_heads",        hp["dec_n_kv_heads"])
        writer.add_uint32("stt.vibevoice.decoder.head_dim",          hp["dec_head_dim"])
        writer.add_string("stt.vibevoice.decoder.hidden_act",        hp["dec_hidden_act"])
        writer.add_float32("stt.vibevoice.decoder.rms_norm_eps",     hp["dec_rms_norm_eps"])
        writer.add_float32("stt.vibevoice.decoder.rope_theta",       hp["dec_rope_theta"])
        writer.add_uint32("stt.vibevoice.decoder.max_position_embeddings",
                          hp["dec_max_pos_emb"])
        writer.add_bool("stt.vibevoice.decoder.tie_word_embeddings",
                        hp["dec_tie_embeddings"])
        writer.add_uint32("stt.vibevoice.decoder.vocab_size",        hp["dec_vocab_size"])

        # ---- stt.vibevoice.{acoustic,semantic}.* (VAE encoders + connectors) ----
        for stream in ("acoustic", "semantic"):
            e = hp[stream]
            base = f"stt.vibevoice.{stream}"
            writer.add_uint32(f"{base}.vae_dim",            e["vae_dim"])
            writer.add_uint32(f"{base}.n_filters",          e["n_filters"])
            writer.add_array(f"{base}.encoder_ratios",      e["ratios"])
            writer.add_string(f"{base}.encoder_depths",     e["depths"])
            writer.add_string(f"{base}.mixer_layer",        e["mixer"])
            writer.add_string(f"{base}.layernorm",          e["layernorm"])
            writer.add_float32(f"{base}.layernorm_eps",     e["layernorm_eps"])
            writer.add_float32(f"{base}.layer_scale_init",  e["layer_scale_init"])
            writer.add_bool(f"{base}.causal",               e["causal"])
            writer.add_float32(f"{base}.fix_std",           e["fix_std"])
            writer.add_string(f"{base}.std_dist_type",      e["std_dist"])
            writer.add_bool(f"{base}.disable_last_norm",    e["disable_last_norm"])
            writer.add_string(f"{base}.pad_mode",           e["pad_mode"])
            writer.add_bool(f"{base}.conv_bias",            e["conv_bias"])

        # ---- speech fusion + prompt metadata ----
        writer.add_uint32("stt.vibevoice.speech_start_token_id", tok["speech_start_id"])
        writer.add_uint32("stt.vibevoice.speech_end_token_id",   tok["speech_end_id"])
        writer.add_uint32("stt.vibevoice.speech_pad_token_id",   tok["speech_pad_id"])
        writer.add_uint32("stt.vibevoice.im_start_token_id",     tok["im_start_id"])
        writer.add_uint32("stt.vibevoice.im_end_token_id",       tok["im_end_id"])
        writer.add_uint32("stt.vibevoice.speech_tok_compress_ratio",
                          SPEECH_TOK_COMPRESS_RATIO)
        writer.add_string("stt.vibevoice.system_prompt", SYSTEM_PROMPT)
        writer.add_string("stt.vibevoice.generation_stop_string",
                          GENERATION_STOP_STRING)

        # ---- stt.frontend.* (raw 24 kHz waveform, no STFT/mel) ----
        writer.add_string("stt.frontend.type",        "raw_waveform")
        writer.add_uint32("stt.frontend.sample_rate", TARGET_SAMPLE_RATE)

        # ---- tensors ----
        n_added = 0
        bytes_in = 0
        bytes_out = 0

        def add(src_name: str, dst_name: str, transform=passthrough) -> None:
            nonlocal n_added, bytes_in, bytes_out
            t = st.get_tensor(src_name)
            if t.dtype != torch.bfloat16:
                raise ValueError(
                    f"{src_name}: expected torch.bfloat16, got {t.dtype}")
            arr = transform(t.float().numpy())
            if arr.dtype != np.float32:
                raise ValueError(
                    f"{src_name}: expected float32 after transform, got {arr.dtype}")
            target_type = vibe_dtype_for(dst_name)
            encoded, raw_dtype = encode_for_gguf(arr, target_type)
            writer.add_tensor(dst_name, encoded, raw_dtype=raw_dtype)
            bytes_in += int(arr.nbytes)
            bytes_out += int(encoded.nbytes)
            n_added += 1

        for src, dst in plan:
            add(src, dst)

        if n_added != len(plan):
            raise RuntimeError(
                f"tensor count mismatch: added {n_added}, planned {len(plan)}")
        print(f"Added {n_added} tensors "
              f"({bytes_in / (1024 * 1024):.1f} MB fp32 -> "
              f"{bytes_out / (1024 * 1024):.1f} MB on disk)")

        print("Writing header + KV + tensor info...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        print("Writing tensor data...")
        writer.write_tensors_to_file()
        writer.close()

    print(f"Done. Wrote {out_path} "
          f"({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _looks_like_repo_id(s: str) -> bool:
    return "/" in s and not Path(s).exists()


def _download_snapshot(repo_id: str, revision: str | None,
                       allow_patterns: list[str] | None = None,
                       max_workers: int | None = None) -> Path:
    slug = slug_from_repo_id(repo_id)
    models_root = os.environ.get("TRANSCRIBE_MODELS_DIR")
    local_dir = Path(models_root) / slug if models_root and allow_patterns is None else None
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
    if revision:
        print(f"Downloading {repo_id}@{revision} from Hugging Face "
              f"(max_workers={max_workers or 'default'})...", flush=True)
    else:
        print(f"Downloading {repo_id} from Hugging Face "
              f"(no revision pin; reproducibility depends on upstream)...",
              flush=True)
    kwargs: dict = dict(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir) if local_dir is not None else None,
    )
    if max_workers is not None:
        kwargs["max_workers"] = max_workers
    resolved = snapshot_download(**kwargs)
    return Path(resolved)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert microsoft/VibeVoice-ASR to a BF16 reference GGUF.")
    p.add_argument("model", type=str,
                   help="HF repo id (microsoft/VibeVoice-ASR) or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug "
                        "when converting from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch / tag / commit SHA) to pin the "
                        "download to. Ignored when `model` is a local dir.")
    p.add_argument("--tokenizer-repo", type=str, default=TOKENIZER_REPO,
                   help=f"HF repo for the byte-level BPE tokenizer "
                        f"(default: {TOKENIZER_REPO}).")
    p.add_argument("--variant", type=str, default=None,
                   help="stt.variant string (default: derived from slug)")
    p.add_argument("--max-workers", type=int, default=None,
                   help="cap snapshot_download parallel workers (default: "
                        "hub default of 8). Lower (e.g. 4) reduces parallel-TLS "
                        "saturation on a congested link.")
    args = p.parse_args(argv[1:])

    if _looks_like_repo_id(args.model):
        repo_id = args.repo_id or args.model
        model_dir = _download_snapshot(args.model, args.revision,
                                       max_workers=args.max_workers)
    else:
        model_dir = Path(args.model)
        if not model_dir.is_dir():
            print(f"error: {model_dir} is not a directory and not an HF repo id",
                  file=sys.stderr)
            return 2
        repo_id = args.repo_id

    tok_dir = _download_snapshot(args.tokenizer_repo, None,
                                 allow_patterns=TOKENIZER_FILES)

    out_path = args.out_path
    if out_path is None:
        if not repo_id:
            print("error: provide out_path, --repo-id, or an HF repo id as model",
                  file=sys.stderr)
            return 2
        slug = slug_from_repo_id(repo_id)
        out_path = REPO_ROOT / "models" / slug / gguf_name(slug, REFERENCE_DTYPE_LABEL)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    variant = args.variant
    if variant is None:
        variant = slug_from_repo_id(repo_id).lower() if repo_id else "vibevoice-asr"

    convert(model_dir, tok_dir, out_path, variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
