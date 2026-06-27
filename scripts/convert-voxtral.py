#!/usr/bin/env -S uv run --project scripts/envs/voxtral python
"""
convert-voxtral.py - convert a Voxtral (2507) checkpoint to a BF16 reference GGUF.

Voxtral is an audio-LLM (token injection, no cross-attention):
    audio_tower (Whisper-large-v3 encoder, bidirectional)
        conv1 (128->1280, k3 s1) -> GELU -> conv2 (1280->1280, k3 s2) -> GELU
        + fixed sinusoidal embed_positions (1500x1280, F32)
        32 pre-LN layers (d_model 1280, 20 heads, head_dim 64, ffn 5120, GELU;
        k_proj has NO bias) -> final layer_norm
    multi_modal_projector
        reshape (T,1280)->(T/4,5120) [4x frame grouping] -> Linear(5120->H)
        -> GELU -> Linear(H->H)   (both bias=False)
    language_model (Llama / Ministral text decoder)
        masked_scatter audio embeds at input_ids == audio_token_id (24)
        30 layers, GQA 32/8, head_dim 128, SwiGLU, RMSNorm, NEOX RoPE theta 1e8,
        UNTIED lm_head.

Tokenizer: Mistral tekken (mistral-common). No HF tokenizer.json. We mirror
llama.cpp's MistralVocab representation: tokenizer.ggml.model="gpt2",
tokenizer.ggml.pre="tekken", a 131072-entry token list (ids 0..999 CONTROL,
1000.. NORMAL byte-level), and a reconstructed gpt2-style merges list (tekken is
rank-based / tiktoken, so merges are synthesized from the mergeable ranks).

Output dtype: BF16 (reference dtype; preserves the upstream weights). Conv
kernels are emitted F16 (the loader has no BF16 conv kernel); norms / biases /
the sinusoidal pos-emb stay F32 (see scripts/lib/gguf_common.reference_dtype_for).

Usage:
    uv run --project scripts/envs/voxtral scripts/convert-voxtral.py \\
      mistralai/Voxtral-Mini-3B-2507 --repo-id mistralai/Voxtral-Mini-3B-2507
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

# Capability languages (model card order). English-only acceptance, but the
# model is natively multilingual; advertise the full set in the GGUF.
LANGUAGES = ["en", "fr", "de", "es", "it", "pt", "nl", "hi"]


# ---------------------------------------------------------------------------
# Sharded safetensors reader
# ---------------------------------------------------------------------------


class _ShardedSafetensors:
    """Open a single- or multi-shard safetensors checkpoint and serve tensors
    by name. Mirrors the pattern used by the other converters."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self._stack = ExitStack()
        self._handles: dict[str, object] = {}
        self._key_to_file: dict[str, str] = {}

    def __enter__(self):
        index = self.model_dir / "model.safetensors.index.json"
        single = self.model_dir / "model.safetensors"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            files = sorted(set(weight_map.values()))
            self._key_to_file = weight_map
        elif single.is_file():
            files = ["model.safetensors"]
            self._key_to_file = {}  # filled below
        else:
            raise FileNotFoundError(
                f"no safetensors (index or single) under {self.model_dir}")
        for fn in files:
            h = self._stack.enter_context(
                safe_open(self.model_dir / fn, framework="pt", device="cpu"))
            self._handles[fn] = h
            if not self._key_to_file:
                for k in h.keys():
                    self._key_to_file[k] = fn
        return self

    def __exit__(self, *exc):
        self._stack.close()

    def keys(self):
        return self._key_to_file.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        fn = self._key_to_file[name]
        return self._handles[fn].get_tensor(name)


# ---------------------------------------------------------------------------
# Tekken tokenizer extraction (replicates llama.cpp gguf-py MistralVocab)
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
    """gpt2 byte<->unicode map (copied from transformers / llama.cpp)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _token_bytes_to_string(b: bytes, byte_encoder: dict[int, str]) -> str:
    return "".join(byte_encoder[ord(c)] for c in b.decode("latin-1"))


def extract_tekken(model_dir: Path, expected_vocab: int) -> dict:
    """Load tekken.json via mistral-common and produce the GGUF token list +
    reconstructed merges, exactly as llama.cpp's MistralVocab does."""
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    tekken_path = model_dir / "tekken.json"
    if not tekken_path.is_file():
        raise FileNotFoundError(f"missing tekken.json under {model_dir}")
    tk = Tekkenizer.from_file(str(tekken_path))

    n_words = int(tk.n_words)
    num_special = int(tk.num_special_tokens)
    if n_words != expected_vocab:
        raise ValueError(
            f"tekken n_words={n_words} != config vocab_size={expected_vocab}")

    byte_encoder = _bytes_to_unicode()
    tokens: list[str] = []
    types: list[int] = []
    # ids 0..num_special-1: the 1000 control slots (named + filler).
    for tid in range(num_special):
        tokens.append(tk.id_to_piece(tid))
        types.append(TOKEN_TYPE_CONTROL)
    # ids num_special..n_words-1: byte-level BPE tokens (NORMAL).
    for tok in tk._tekken_token2id_nospecial:
        tokens.append(_token_bytes_to_string(tok, byte_encoder))
        types.append(TOKEN_TYPE_NORMAL)
    if len(tokens) != n_words:
        raise RuntimeError(
            f"token count {len(tokens)} != n_words {n_words}")

    # Reconstruct gpt2-style merges from the rank-based (tiktoken) model.
    mergeable = tk._model._mergeable_ranks  # dict[bytes,int]
    by_rank = {rank: tb for tb, rank in mergeable.items()}
    triples: list[tuple[bytes, bytes, int]] = []
    for i in range(256, n_words - num_special):
        merged = by_rank[i]
        local: list[tuple[bytes, bytes, int]] = []
        for j in range(1, len(merged)):
            left, right = merged[:j], merged[j:]
            if left in mergeable and right in mergeable and (left + right) in mergeable:
                local.append((left, right, i))
        if not local:
            raise ValueError(f"no valid merge for rank {i}: {merged!r}")
        local.sort(key=lambda x: (mergeable[x[0]], mergeable[x[1]]))
        triples.extend(local)
    triples.sort(key=lambda x: x[2])
    merges = [
        " ".join(
            "".join(chr(ord(c) + 256) if c == " " else c
                    for c in _token_bytes_to_string(part, byte_encoder))
            for part in (left, right)
        )
        for left, right, _ in triples
    ]

    pad_id = int(tk.pad_id) if int(tk.pad_id) != -1 else int(tk.eos_id)
    return {
        "tokens": tokens,
        "types": types,
        "merges": merges,
        "bos_id": int(tk.bos_id),
        "eos_id": int(tk.eos_id),
        "unk_id": int(tk.unk_id),
        "pad_id": pad_id,
        "vocab_size": n_words,
    }


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


def read_hparams(config: dict, preproc: dict) -> dict:
    aenc = config["audio_config"]
    tdec = config["text_config"]

    sample_rate = int(preproc.get("sampling_rate", 16000))
    hop_length = int(preproc["hop_length"])
    n_fft = int(preproc["n_fft"])
    n_mels = int(preproc["feature_size"])
    chunk_len = int(preproc.get("chunk_length", 30))
    n_samples = int(preproc.get("n_samples", chunk_len * sample_rate))
    nb_max_frm = int(preproc.get("nb_max_frames", n_samples // hop_length))

    text_hidden = int(tdec["hidden_size"])
    enc_d_model = int(aenc["hidden_size"])
    # Projector input = audio hidden * downsample_factor (4x frame grouping).
    downsample = 4
    proj_in = enc_d_model * downsample

    return {
        # Audio encoder (Whisper-large-v3)
        "enc_n_layers": int(aenc["num_hidden_layers"]),
        "enc_d_model": enc_d_model,
        "enc_n_heads": int(aenc["num_attention_heads"]),
        "enc_head_dim": int(aenc["head_dim"]),
        "enc_ffn_dim": int(aenc["intermediate_size"]),
        "enc_n_mels": int(aenc["num_mel_bins"]),
        "enc_max_src_pos": int(aenc["max_source_positions"]),
        "enc_activation": str(aenc["activation_function"]).lower(),

        # Projector
        "proj_in": proj_in,
        "proj_downsample": downsample,
        "proj_hidden_act": str(config.get("projector_hidden_act", "gelu")).lower(),

        # Text LM (Llama / Ministral)
        "dec_n_layers": int(tdec["num_hidden_layers"]),
        "dec_hidden": text_hidden,
        "dec_intermediate": int(tdec["intermediate_size"]),
        "dec_n_heads": int(tdec["num_attention_heads"]),
        "dec_n_kv_heads": int(tdec["num_key_value_heads"]),
        "dec_head_dim": int(tdec["head_dim"]),
        "dec_hidden_act": str(tdec["hidden_act"]).lower(),
        "dec_rms_norm_eps": float(tdec["rms_norm_eps"]),
        "dec_rope_theta": float(tdec["rope_theta"]),
        "dec_max_pos_emb": int(tdec["max_position_embeddings"]),
        "dec_tie_embeddings": bool(tdec.get("tie_word_embeddings", False)),
        "dec_vocab_size": int(tdec["vocab_size"]),

        # Fusion
        "audio_token_id": int(config["audio_token_id"]),

        # Frontend (Whisper feature extractor)
        "fe_type": "mel",
        "fe_sample_rate": sample_rate,
        "fe_num_mels": n_mels,
        "fe_n_fft": n_fft,
        "fe_win_length": n_fft,
        "fe_hop_length": hop_length,
        "fe_window": "hann_periodic",
        "fe_normalize": "per_utterance",
        "fe_dither": float(preproc.get("dither", 0.0)),
        "fe_pre_emphasis": 0.0,
        "fe_f_min": 0.0,
        "fe_f_max": float(sample_rate) / 2.0,
        "fe_pad_mode": "reflect",
        "fe_center": True,
        "fe_mel_norm": "slaney",
        "fe_chunk_length": chunk_len,
        "fe_n_samples": n_samples,
        "fe_nb_max_frm": nb_max_frm,

        "languages": LANGUAGES,
    }


# ---------------------------------------------------------------------------
# Tensor name mapping (HF -> GGUF)
# ---------------------------------------------------------------------------


# Audio encoder top-level. Conv named ".conv.N." so gguf_common routes the
# kernel to F16 (loader has no BF16 conv); pos-emb named ".pos_emb.weight" so
# it stays F32 (it already is F32 in the checkpoint).
ENC_TOP_TABLE: list[tuple[str, str]] = [
    ("audio_tower.conv1.weight", "enc.conv.0.weight"),
    ("audio_tower.conv1.bias",   "enc.conv.0.bias"),
    ("audio_tower.conv2.weight", "enc.conv.1.weight"),
    ("audio_tower.conv2.bias",   "enc.conv.1.bias"),
    ("audio_tower.embed_positions.weight", "enc.pos_emb.weight"),
    ("audio_tower.layer_norm.weight", "enc.ln_post.weight"),
    ("audio_tower.layer_norm.bias",   "enc.ln_post.bias"),
]

# Per encoder layer. k_proj has NO bias (Whisper convention).
ENC_BLOCK_TABLE: list[tuple[str, str]] = [
    ("self_attn_layer_norm.weight", "norm_attn.weight"),
    ("self_attn_layer_norm.bias",   "norm_attn.bias"),
    ("self_attn.q_proj.weight",     "attn.q.weight"),
    ("self_attn.q_proj.bias",       "attn.q.bias"),
    ("self_attn.k_proj.weight",     "attn.k.weight"),
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

# Multimodal projector (no biases).
PROJ_TABLE: list[tuple[str, str]] = [
    ("multi_modal_projector.linear_1.weight", "proj.linear_1.weight"),
    ("multi_modal_projector.linear_2.weight", "proj.linear_2.weight"),
]

# Text LM top-level. lm_head is UNTIED -> separate dec.output.weight.
TEXT_TOP_TABLE: list[tuple[str, str]] = [
    ("language_model.model.embed_tokens.weight", "dec.token_embd.weight"),
    ("language_model.model.norm.weight",         "dec.output_norm.weight"),
    ("language_model.lm_head.weight",            "dec.output.weight"),
]

# Per text layer (Llama; no attention biases).
TEXT_BLOCK_TABLE: list[tuple[str, str]] = [
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


def compute_size_label(total_params: int) -> str:
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.1f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.0f}M"
    return f"{total_params / 1_000:.0f}K"


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def convert(model_dir: Path, out_path: Path, variant: str, repo_id: str | None = None) -> None:
    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    config = json.loads((model_dir / "config.json").read_text())
    preproc = json.loads((model_dir / "preprocessor_config.json").read_text())
    hp = read_hparams(config, preproc)
    print(f"Audio encoder: {hp['enc_n_layers']} layers, d_model={hp['enc_d_model']}, "
          f"heads={hp['enc_n_heads']}")
    print(f"Text LM: {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, "
          f"heads={hp['dec_n_heads']}/{hp['dec_n_kv_heads']}, "
          f"tie_word_embeddings={hp['dec_tie_embeddings']}")

    print(f"Reading tekken tokenizer from {model_dir}")
    tok = extract_tekken(model_dir, hp["dec_vocab_size"])
    print(f"  tokens={len(tok['tokens'])} merges={len(tok['merges'])} "
          f"bos={tok['bos_id']} eos={tok['eos_id']} pad={tok['pad_id']} unk={tok['unk_id']}")
    if tok["bos_id"] != config.get("bos_token_id", 1) and "bos_token_id" in config:
        raise ValueError("bos id mismatch tekken vs config")

    with _ShardedSafetensors(model_dir) as st:
        st_keys = set(st.keys())

        total = sum(st.get_tensor(k).numel() for k in st_keys)
        size_label = compute_size_label(total)
        print(f"Total params: {total:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = GGUFWriter(str(out_path), "voxtral")

        # ---- general.* ----
        _VARIANT_TABLE = {
            "voxtral-mini-3b-2507":  ("Voxtral Mini 3B",  "2507"),
            "voxtral-small-24b-2507": ("Voxtral Small 24B", "2507"),
        }
        if variant not in _VARIANT_TABLE:
            raise ValueError(f"unknown voxtral variant slug: {variant!r}")
        _disp_name, _disp_version = _VARIANT_TABLE[variant]
        add_general_identity(
            writer,
            name=_disp_name,
            basename="voxtral",
            version=_disp_version,
            size_label=size_label,
            file_type=int(REFERENCE_FILE_TYPE),
            languages=hp["languages"],
            author="Mistral AI",
            organization="mistralai",
            license="apache-2.0",
            license_name="Apache License 2.0",
            license_link="https://www.apache.org/licenses/LICENSE-2.0",
            repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
        )
        writer.add_string("stt.variant", variant)

        # ---- stt.capability.* ----
        # Auto language detection (default transcription mode) and speech
        # translation are both in scope for this port (user-signed).
        writer.add_bool("stt.capability.lang_detect", True)
        writer.add_bool("stt.capability.translate", True)

        # ---- tokenizer.ggml.* (Mistral tekken -> llama.cpp gpt2 BPE) ----
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_string("tokenizer.ggml.pre", "tekken")
        writer.add_array("tokenizer.ggml.tokens", tok["tokens"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        writer.add_array("tokenizer.ggml.merges", tok["merges"])
        writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
        writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
        writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
        writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])
        writer.add_bool("tokenizer.ggml.add_bos_token", True)
        writer.add_bool("tokenizer.ggml.add_eos_token", False)

        # ---- stt.voxtral.encoder.* ----
        writer.add_uint32("stt.voxtral.encoder.n_layers", hp["enc_n_layers"])
        writer.add_uint32("stt.voxtral.encoder.d_model", hp["enc_d_model"])
        writer.add_uint32("stt.voxtral.encoder.n_heads", hp["enc_n_heads"])
        writer.add_uint32("stt.voxtral.encoder.head_dim", hp["enc_head_dim"])
        writer.add_uint32("stt.voxtral.encoder.ffn_dim", hp["enc_ffn_dim"])
        writer.add_uint32("stt.voxtral.encoder.num_mel_bins", hp["enc_n_mels"])
        writer.add_uint32("stt.voxtral.encoder.max_source_positions", hp["enc_max_src_pos"])
        writer.add_string("stt.voxtral.encoder.activation", hp["enc_activation"])

        # ---- stt.voxtral.projector.* ----
        writer.add_uint32("stt.voxtral.projector.downsample_factor", hp["proj_downsample"])
        writer.add_uint32("stt.voxtral.projector.input_dim", hp["proj_in"])
        writer.add_string("stt.voxtral.projector.hidden_act", hp["proj_hidden_act"])

        # ---- stt.voxtral.decoder.* ----
        writer.add_uint32("stt.voxtral.decoder.n_layers", hp["dec_n_layers"])
        writer.add_uint32("stt.voxtral.decoder.hidden_size", hp["dec_hidden"])
        writer.add_uint32("stt.voxtral.decoder.intermediate_size", hp["dec_intermediate"])
        writer.add_uint32("stt.voxtral.decoder.n_heads", hp["dec_n_heads"])
        writer.add_uint32("stt.voxtral.decoder.n_kv_heads", hp["dec_n_kv_heads"])
        writer.add_uint32("stt.voxtral.decoder.head_dim", hp["dec_head_dim"])
        writer.add_string("stt.voxtral.decoder.hidden_act", hp["dec_hidden_act"])
        writer.add_float32("stt.voxtral.decoder.rms_norm_eps", hp["dec_rms_norm_eps"])
        writer.add_float32("stt.voxtral.decoder.rope_theta", hp["dec_rope_theta"])
        writer.add_uint32("stt.voxtral.decoder.max_position_embeddings", hp["dec_max_pos_emb"])
        writer.add_bool("stt.voxtral.decoder.tie_word_embeddings", hp["dec_tie_embeddings"])
        writer.add_uint32("stt.voxtral.decoder.vocab_size", hp["dec_vocab_size"])
        writer.add_uint32("stt.voxtral.audio_token_id", hp["audio_token_id"])

        # ---- stt.frontend.* (Whisper feature extractor) ----
        writer.add_string("stt.frontend.type", hp["fe_type"])
        writer.add_uint32("stt.frontend.num_mels", hp["fe_num_mels"])
        writer.add_uint32("stt.frontend.sample_rate", hp["fe_sample_rate"])
        writer.add_uint32("stt.frontend.n_fft", hp["fe_n_fft"])
        writer.add_uint32("stt.frontend.win_length", hp["fe_win_length"])
        writer.add_uint32("stt.frontend.hop_length", hp["fe_hop_length"])
        writer.add_string("stt.frontend.window", hp["fe_window"])
        writer.add_string("stt.frontend.normalize", hp["fe_normalize"])
        writer.add_float32("stt.frontend.dither", hp["fe_dither"])
        writer.add_float32("stt.frontend.pre_emphasis", hp["fe_pre_emphasis"])
        writer.add_float32("stt.frontend.f_min", hp["fe_f_min"])
        writer.add_float32("stt.frontend.f_max", hp["fe_f_max"])
        writer.add_string("stt.frontend.pad_mode", hp["fe_pad_mode"])
        writer.add_bool("stt.frontend.center", hp["fe_center"])
        writer.add_string("stt.frontend.mel_norm", hp["fe_mel_norm"])
        writer.add_uint32("stt.frontend.chunk_length", hp["fe_chunk_length"])
        writer.add_uint32("stt.frontend.n_samples", hp["fe_n_samples"])
        writer.add_uint32("stt.frontend.nb_max_frames", hp["fe_nb_max_frm"])

        # ---- tensors ----
        n_added = 0
        bytes_in = bytes_out = 0

        # Frontend buffers baked in (bit-identical filterbank + window in C++).
        import librosa as _lb
        mel_fb = _lb.filters.mel(
            sr=hp["fe_sample_rate"], n_fft=hp["fe_n_fft"], n_mels=hp["fe_num_mels"],
            fmin=hp["fe_f_min"], fmax=hp["fe_f_max"], norm="slaney", htk=False,
        ).astype(np.float32)
        writer.add_tensor("frontend.mel_filterbank", np.ascontiguousarray(mel_fb),
                          raw_dtype=GGMLQuantizationType.F32)
        N = int(hp["fe_win_length"])
        hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / N)).astype(np.float32)
        writer.add_tensor("frontend.window", np.ascontiguousarray(hann),
                          raw_dtype=GGMLQuantizationType.F32)
        n_added += 2
        bytes_in += mel_fb.nbytes + hann.nbytes
        bytes_out += mel_fb.nbytes + hann.nbytes

        def add(src_name: str, dst_name: str) -> None:
            nonlocal n_added, bytes_in, bytes_out
            if src_name not in st_keys:
                raise KeyError(f"safetensors missing tensor: {src_name!r}")
            t = st.get_tensor(src_name)
            # embed_positions is F32 in the checkpoint; everything else BF16.
            if t.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError(f"{src_name}: unexpected dtype {t.dtype}")
            arr = np.ascontiguousarray(t.float().numpy())
            target_type = reference_dtype_for(dst_name, REFERENCE_GGML_TYPE)
            encoded, raw_dtype = encode_for_gguf(arr, target_type)
            writer.add_tensor(dst_name, encoded, raw_dtype=raw_dtype)
            bytes_in += int(arr.nbytes)
            bytes_out += int(encoded.nbytes)
            n_added += 1

        for src, dst in ENC_TOP_TABLE:
            add(src, dst)
        for i in range(hp["enc_n_layers"]):
            for s, d in ENC_BLOCK_TABLE:
                add(f"audio_tower.layers.{i}.{s}", f"enc.blocks.{i}.{d}")
        for src, dst in PROJ_TABLE:
            add(src, dst)
        for src, dst in TEXT_TOP_TABLE:
            add(src, dst)
        for i in range(hp["dec_n_layers"]):
            for s, d in TEXT_BLOCK_TABLE:
                add(f"language_model.model.layers.{i}.{s}", f"dec.blocks.{i}.{d}")

        expected = (
            len(ENC_TOP_TABLE)
            + hp["enc_n_layers"] * len(ENC_BLOCK_TABLE)
            + len(PROJ_TABLE)
            + len(TEXT_TOP_TABLE)
            + hp["dec_n_layers"] * len(TEXT_BLOCK_TABLE)
            + 2  # frontend buffers
        )
        if n_added != expected:
            raise RuntimeError(f"tensor count mismatch: added {n_added}, expected {expected}")

        # Verify every safetensors key was consumed.
        consumed = set()
        for src, _ in ENC_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["enc_n_layers"]):
            for s, _ in ENC_BLOCK_TABLE:
                consumed.add(f"audio_tower.layers.{i}.{s}")
        for src, _ in PROJ_TABLE:
            consumed.add(src)
        for src, _ in TEXT_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["dec_n_layers"]):
            for s, _ in TEXT_BLOCK_TABLE:
                consumed.add(f"language_model.model.layers.{i}.{s}")
        unused = sorted(st_keys - consumed)
        if unused:
            raise RuntimeError(
                f"{len(unused)} safetensors keys not consumed: {unused[:10]}")

        print(f"Added {n_added} tensors "
              f"({bytes_in / (1024 * 1024):.1f} MB fp32 -> "
              f"{bytes_out / (1024 * 1024):.1f} MB on disk)")

        print("Writing header + KV + tensors...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    print(f"Done. Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _looks_like_repo_id(s: str) -> bool:
    return "/" in s and not Path(s).exists()


def _download_snapshot(repo_id: str, revision: str | None) -> Path:
    slug = slug_from_repo_id(repo_id)
    models_root = os.environ.get("TRANSCRIBE_MODELS_DIR")
    local_dir = Path(models_root) / slug if models_root else None
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
    resolved = snapshot_download(
        repo_id=repo_id, revision=revision,
        local_dir=str(local_dir) if local_dir is not None else None,
    )
    return Path(resolved)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a Voxtral (2507) checkpoint to a BF16 reference GGUF.")
    p.add_argument("model", type=str, help="HF repo id or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch/tag/commit) to pin the download")
    p.add_argument("--variant", type=str, default=None,
                   help="stt.variant string (default: derived from slug)")
    args = p.parse_args(argv[1:])

    if _looks_like_repo_id(args.model):
        repo_id = args.repo_id or args.model
        model_dir = _download_snapshot(args.model, args.revision)
    else:
        model_dir = Path(args.model)
        if not model_dir.is_dir():
            print(f"error: {model_dir} is not a directory or HF repo id", file=sys.stderr)
            return 2
        repo_id = args.repo_id

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
    if variant is None and repo_id:
        variant = slug_from_repo_id(repo_id).lower()
    if variant is None:
        variant = out_path.stem.lower()
        for q in ("-bf16", "-f32", "-f16"):
            if variant.endswith(q):
                variant = variant[: -len(q)]
                break

    convert(model_dir, out_path, variant, repo_id=repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
