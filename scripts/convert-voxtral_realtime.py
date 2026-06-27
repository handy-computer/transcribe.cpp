#!/usr/bin/env -S uv run --project scripts/envs/voxtral_realtime python
"""
convert-voxtral_realtime.py - convert a Voxtral Realtime (2602) checkpoint to a
BF16 reference GGUF.

Voxtral Realtime is a STREAMING audio-LLM, architecturally DISTINCT from the
2507 `voxtral` models (own converter convert-voxtral.py). It shares only the
projector shape, the tekken tokenizer, and the family brand. Differences:

    audio_tower (causal RoPE encoder, NOT Whisper bidirectional):
        embedder: causal Conv1d 128->1280 (k3 s1) -> GELU
                  causal Conv1d 1280->1280 (k3 s2) -> GELU
        32 pre-norm RMSNorm layers, RoPE theta 1e6 head_dim 64, causal +
        sliding-window(750) attention (q/v/o have bias, k NONE), SwiGLU/silu
        MLP (bias only on down_proj) -> final RMSNorm
    multi_modal_projector:
        reshape (T,1280)->(T/4,5120) [4x] -> Linear(5120->3072) -> GELU
        -> Linear(3072->3072)   (both bias=False)
    time conditioning:
        t = sinusoid(num_delay_tokens) [theta 10000, dim 3072]; each decoder
        layer applies post_attention_layernorm(h) * (1 + ada_rms_norm(t)),
        where ada_rms_norm = Linear(3072->32) -> GELU -> Linear(32->3072).
    language_model (26-layer Ministral text decoder):
        ADDITIVE audio fusion (inputs_embeds += audio_embeds, NOT
        masked_scatter); GQA 32/8 head_dim 128, SwiGLU, RMSNorm, NEOX RoPE
        theta 1e6, TIED lm_head (lm_head.weight == embed_tokens.weight).

Source: loaded via transformers (VoxtralRealtimeForConditionalGeneration), so
permute_for_rope is already applied (HF split-halves / NEOX layout) and weights
are the same the Stage-2 oracle dumped. The checkpoint ships original-Mistral
`consolidated.safetensors`; transformers maps it to the HF tree on load.

Output dtype: BF16 (reference dtype; preserves the upstream weights). Conv
kernels are emitted F16 (loader has no BF16 conv); norms / biases / frontend
buffers / the baked time-embedding inv_freq stay F32
(see scripts/lib/gguf_common.reference_dtype_for).

Usage:
    uv run --project scripts/envs/voxtral_realtime scripts/convert-voxtral_realtime.py \\
      mistralai/Voxtral-Mini-4B-Realtime-2602 --repo-id mistralai/Voxtral-Mini-4B-Realtime-2602
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.hf_source import download_snapshot, looks_like_repo_id  # noqa: E402
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

# Model card languages (13). Multilingual streaming model; English-only
# acceptance gate, but advertise the full set in the GGUF.
LANGUAGES = ["en", "fr", "es", "de", "ru", "zh", "ja", "it", "pt", "nl", "ar", "hi", "ko"]


# ---------------------------------------------------------------------------
# Tekken tokenizer extraction (identical mechanism to convert-voxtral.py)
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
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
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    tekken_path = model_dir / "tekken.json"
    if not tekken_path.is_file():
        raise FileNotFoundError(f"missing tekken.json under {model_dir}")
    tk = Tekkenizer.from_file(str(tekken_path))

    n_words = int(tk.n_words)
    num_special = int(tk.num_special_tokens)
    if n_words != expected_vocab:
        raise ValueError(f"tekken n_words={n_words} != config vocab_size={expected_vocab}")

    byte_encoder = _bytes_to_unicode()
    tokens: list[str] = []
    types: list[int] = []
    for tid in range(num_special):
        tokens.append(tk.id_to_piece(tid))
        types.append(TOKEN_TYPE_CONTROL)
    for tok in tk._tekken_token2id_nospecial:
        tokens.append(_token_bytes_to_string(tok, byte_encoder))
        types.append(TOKEN_TYPE_NORMAL)
    if len(tokens) != n_words:
        raise RuntimeError(f"token count {len(tokens)} != n_words {n_words}")

    mergeable = tk._model._mergeable_ranks
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
        "tokens": tokens, "types": types, "merges": merges,
        "bos_id": int(tk.bos_id), "eos_id": int(tk.eos_id),
        "unk_id": int(tk.unk_id), "pad_id": pad_id, "vocab_size": n_words,
    }


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


def _rope_theta(cfg: dict) -> float:
    rp = cfg.get("rope_parameters") or {}
    return float(rp.get("rope_theta", cfg.get("rope_theta", 1000000.0)))


def read_hparams(config: dict, fe: dict) -> dict:
    aenc = config["audio_config"]
    tdec = config["text_config"]
    downsample = int(config["downsample_factor"])
    enc_d_model = int(aenc["hidden_size"])

    n_fft = int(fe["n_fft"])
    n_mels = int(fe["feature_size"])
    hop = int(fe["hop_length"])
    win = int(fe["win_length"])
    sr = int(fe.get("sampling_rate", 16000))

    return {
        # Audio encoder (causal RoPE, sliding-window)
        "enc_n_layers": int(aenc["num_hidden_layers"]),
        "enc_d_model": enc_d_model,
        "enc_n_heads": int(aenc["num_attention_heads"]),
        "enc_n_kv_heads": int(aenc["num_key_value_heads"]),
        "enc_head_dim": int(aenc["head_dim"]),
        "enc_ffn_dim": int(aenc["intermediate_size"]),
        "enc_n_mels": int(aenc["num_mel_bins"]),
        "enc_max_pos": int(aenc["max_position_embeddings"]),
        "enc_sliding_window": int(aenc["sliding_window"]),
        "enc_rope_theta": _rope_theta(aenc),
        "enc_rms_norm_eps": float(aenc["rms_norm_eps"]),
        "enc_hidden_act": str(aenc["hidden_act"]).lower(),   # silu (MLP)

        # Projector
        "proj_in": enc_d_model * downsample,
        "proj_downsample": downsample,
        "proj_hidden_act": str(config.get("projector_hidden_act", "gelu")).lower(),
        "audio_length_per_tok": int(config["audio_length_per_tok"]),

        # Text decoder (Ministral)
        "dec_n_layers": int(tdec["num_hidden_layers"]),
        "dec_hidden": int(tdec["hidden_size"]),
        "dec_intermediate": int(tdec["intermediate_size"]),
        "dec_n_heads": int(tdec["num_attention_heads"]),
        "dec_n_kv_heads": int(tdec["num_key_value_heads"]),
        "dec_head_dim": int(tdec["head_dim"]),
        "dec_hidden_act": str(tdec["hidden_act"]).lower(),
        "dec_rms_norm_eps": float(tdec["rms_norm_eps"]),
        "dec_rope_theta": _rope_theta(tdec),
        "dec_sliding_window": int(tdec["sliding_window"]),
        "dec_max_pos_emb": int(tdec["max_position_embeddings"]),
        "dec_tie_embeddings": bool(tdec.get("tie_word_embeddings", True)),
        "dec_vocab_size": int(tdec["vocab_size"]),

        # Delay-token time conditioning
        "default_num_delay_tokens": int(config["default_num_delay_tokens"]),
        "time_embed_theta": 10000.0,    # VoxtralRealtimeTimeEmbedding default
        "time_embed_dim": int(tdec["hidden_size"]),
        "ada_hidden": 32,               # VoxtralRealtimeTextAdaRmsNorm bottleneck

        # Frontend (streaming log-mel; FIXED global_log_mel_max)
        "fe_type": "mel",
        "fe_sample_rate": sr,
        "fe_num_mels": n_mels,
        "fe_n_fft": n_fft,
        "fe_win_length": win,
        "fe_hop_length": hop,
        "fe_window": "hann_periodic",
        "fe_normalize": "global",
        "fe_global_log_mel_max": float(fe.get("global_log_mel_max", 1.5)),
        "fe_dither": 0.0,
        "fe_pre_emphasis": 0.0,
        "fe_f_min": 0.0,
        "fe_f_max": 8000.0,
        "fe_pad_mode": "reflect",
        "fe_center": True,
        "fe_mel_norm": "slaney",

        "languages": LANGUAGES,
    }


# ---------------------------------------------------------------------------
# Tensor name mapping (HF state_dict -> GGUF)
# ---------------------------------------------------------------------------

# Audio encoder top-level. Conv named ".conv.N." so gguf_common routes the
# kernel to F16. Final RMSNorm named ".final_norm.weight" so it stays F32.
ENC_TOP_TABLE: list[tuple[str, str]] = [
    ("model.audio_tower.embedder.conv1.weight", "enc.conv.0.weight"),
    ("model.audio_tower.embedder.conv1.bias",   "enc.conv.0.bias"),
    ("model.audio_tower.embedder.conv2.weight", "enc.conv.1.weight"),
    ("model.audio_tower.embedder.conv2.bias",   "enc.conv.1.bias"),
    ("model.audio_tower.norm.weight",           "enc.final_norm.weight"),
]

# Per encoder layer. q/v/o have bias; k has NO bias. MLP is SwiGLU/silu with
# bias only on down_proj.
ENC_BLOCK_TABLE: list[tuple[str, str]] = [
    ("self_attn_layer_norm.weight", "norm_attn.weight"),
    ("self_attn.q_proj.weight",     "attn.q.weight"),
    ("self_attn.q_proj.bias",       "attn.q.bias"),
    ("self_attn.k_proj.weight",     "attn.k.weight"),
    ("self_attn.v_proj.weight",     "attn.v.weight"),
    ("self_attn.v_proj.bias",       "attn.v.bias"),
    ("self_attn.o_proj.weight",     "attn.out.weight"),
    ("self_attn.o_proj.bias",       "attn.out.bias"),
    ("final_layer_norm.weight",     "norm_ffn.weight"),
    ("mlp.gate_proj.weight",        "ffn.gate.weight"),
    ("mlp.up_proj.weight",          "ffn.up.weight"),
    ("mlp.down_proj.weight",        "ffn.down.weight"),
    ("mlp.down_proj.bias",          "ffn.down.bias"),
]

# Multimodal projector (no biases).
PROJ_TABLE: list[tuple[str, str]] = [
    ("model.multi_modal_projector.linear_1.weight", "proj.linear_1.weight"),
    ("model.multi_modal_projector.linear_2.weight", "proj.linear_2.weight"),
]

# Text decoder top-level. lm_head is TIED -> emit token_embd only.
TEXT_TOP_TABLE: list[tuple[str, str]] = [
    ("model.language_model.embed_tokens.weight", "dec.token_embd.weight"),
    ("model.language_model.norm.weight",         "dec.output_norm.weight"),
]

# Per text layer (Ministral; no attention biases). ada_rms_norm linears are
# BF16 (small linears), not norms.
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
    ("ada_rms_norm.linear1.weight",     "ada.linear_1.weight"),
    ("ada_rms_norm.linear2.weight",     "ada.linear_2.weight"),
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
    from transformers import VoxtralRealtimeForConditionalGeneration

    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    config = json.loads((model_dir / "config.json").read_text())
    proc = json.loads((model_dir / "processor_config.json").read_text())
    fe = proc["feature_extractor"]
    hp = read_hparams(config, fe)
    print(f"Audio encoder: {hp['enc_n_layers']} layers d_model={hp['enc_d_model']} "
          f"heads={hp['enc_n_heads']} hd={hp['enc_head_dim']} sw={hp['enc_sliding_window']}")
    print(f"Text decoder: {hp['dec_n_layers']} layers hidden={hp['dec_hidden']} "
          f"heads={hp['dec_n_heads']}/{hp['dec_n_kv_heads']} hd={hp['dec_head_dim']} "
          f"tie={hp['dec_tie_embeddings']}")

    print(f"Reading tekken tokenizer from {model_dir}")
    tok = extract_tekken(model_dir, hp["dec_vocab_size"])
    print(f"  tokens={len(tok['tokens'])} merges={len(tok['merges'])} "
          f"bos={tok['bos_id']} eos={tok['eos_id']} pad={tok['pad_id']} unk={tok['unk_id']}")

    print("Loading weights via transformers (applies permute_for_rope / NEOX)...")
    model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.bfloat16,
    ).eval()
    sd = dict(model.state_dict())

    # Tied lm_head must equal embed_tokens; emit token_embd only.
    if "lm_head.weight" in sd:
        emb = sd["model.language_model.embed_tokens.weight"]
        if not torch.equal(sd["lm_head.weight"], emb):
            raise ValueError("lm_head.weight != embed_tokens.weight but tie_word_embeddings expected")

    total = sum(v.numel() for v in sd.values())
    size_label = compute_size_label(total)
    print(f"Total params: {total:,} -> size_label={size_label}")

    print(f"Writing GGUF to {out_path}")
    writer = GGUFWriter(str(out_path), "voxtral_realtime")

    # ---- general.* ----
    add_general_identity(
        writer,
        name="Voxtral Mini 4B Realtime",
        basename="voxtral_realtime",
        version="2602",
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
    writer.add_bool("stt.capability.lang_detect", True)
    writer.add_bool("stt.capability.translate", False)
    writer.add_bool("stt.capability.streaming", True)

    # ---- tokenizer.ggml.* ----
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
    # Streaming control tokens (mistral-common tekken).
    writer.add_uint32("stt.voxtral_realtime.streaming_pad_token_id", 32)

    # ---- stt.voxtral_realtime.encoder.* ----
    p = "stt.voxtral_realtime.encoder."
    writer.add_uint32(p + "n_layers", hp["enc_n_layers"])
    writer.add_uint32(p + "d_model", hp["enc_d_model"])
    writer.add_uint32(p + "n_heads", hp["enc_n_heads"])
    writer.add_uint32(p + "n_kv_heads", hp["enc_n_kv_heads"])
    writer.add_uint32(p + "head_dim", hp["enc_head_dim"])
    writer.add_uint32(p + "ffn_dim", hp["enc_ffn_dim"])
    writer.add_uint32(p + "num_mel_bins", hp["enc_n_mels"])
    writer.add_uint32(p + "max_position_embeddings", hp["enc_max_pos"])
    writer.add_uint32(p + "sliding_window", hp["enc_sliding_window"])
    writer.add_float32(p + "rope_theta", hp["enc_rope_theta"])
    writer.add_float32(p + "rms_norm_eps", hp["enc_rms_norm_eps"])
    writer.add_string(p + "hidden_act", hp["enc_hidden_act"])

    # ---- stt.voxtral_realtime.projector.* ----
    p = "stt.voxtral_realtime.projector."
    writer.add_uint32(p + "downsample_factor", hp["proj_downsample"])
    writer.add_uint32(p + "input_dim", hp["proj_in"])
    writer.add_string(p + "hidden_act", hp["proj_hidden_act"])
    writer.add_uint32(p + "audio_length_per_tok", hp["audio_length_per_tok"])

    # ---- stt.voxtral_realtime.decoder.* ----
    p = "stt.voxtral_realtime.decoder."
    writer.add_uint32(p + "n_layers", hp["dec_n_layers"])
    writer.add_uint32(p + "hidden_size", hp["dec_hidden"])
    writer.add_uint32(p + "intermediate_size", hp["dec_intermediate"])
    writer.add_uint32(p + "n_heads", hp["dec_n_heads"])
    writer.add_uint32(p + "n_kv_heads", hp["dec_n_kv_heads"])
    writer.add_uint32(p + "head_dim", hp["dec_head_dim"])
    writer.add_string(p + "hidden_act", hp["dec_hidden_act"])
    writer.add_float32(p + "rms_norm_eps", hp["dec_rms_norm_eps"])
    writer.add_float32(p + "rope_theta", hp["dec_rope_theta"])
    writer.add_uint32(p + "sliding_window", hp["dec_sliding_window"])
    writer.add_uint32(p + "max_position_embeddings", hp["dec_max_pos_emb"])
    writer.add_bool(p + "tie_word_embeddings", hp["dec_tie_embeddings"])
    writer.add_uint32(p + "vocab_size", hp["dec_vocab_size"])

    # ---- stt.voxtral_realtime.time.* (delay-token conditioning) ----
    p = "stt.voxtral_realtime.time."
    writer.add_uint32(p + "default_num_delay_tokens", hp["default_num_delay_tokens"])
    writer.add_float32(p + "embed_theta", hp["time_embed_theta"])
    writer.add_uint32(p + "embed_dim", hp["time_embed_dim"])
    writer.add_uint32(p + "ada_hidden", hp["ada_hidden"])

    # ---- stt.frontend.* (streaming log-mel; fixed global max) ----
    p = "stt.frontend."
    writer.add_string(p + "type", hp["fe_type"])
    writer.add_uint32(p + "num_mels", hp["fe_num_mels"])
    writer.add_uint32(p + "sample_rate", hp["fe_sample_rate"])
    writer.add_uint32(p + "n_fft", hp["fe_n_fft"])
    writer.add_uint32(p + "win_length", hp["fe_win_length"])
    writer.add_uint32(p + "hop_length", hp["fe_hop_length"])
    writer.add_string(p + "window", hp["fe_window"])
    writer.add_string(p + "normalize", hp["fe_normalize"])
    writer.add_float32(p + "global_log_mel_max", hp["fe_global_log_mel_max"])
    writer.add_float32(p + "dither", hp["fe_dither"])
    writer.add_float32(p + "pre_emphasis", hp["fe_pre_emphasis"])
    writer.add_float32(p + "f_min", hp["fe_f_min"])
    writer.add_float32(p + "f_max", hp["fe_f_max"])
    writer.add_string(p + "pad_mode", hp["fe_pad_mode"])
    writer.add_bool(p + "center", hp["fe_center"])
    writer.add_string(p + "mel_norm", hp["fe_mel_norm"])

    # ---- tensors ----
    n_added = 0
    bytes_in = bytes_out = 0

    # Frontend buffers baked in (bit-identical filterbank + window in C++).
    # mel_filter_bank(num_freq=1+n_fft//2, n_mels, 0..8000, slaney) matches the
    # HF VoxtralRealtimeFeatureExtractor; periodic Hann of length n_fft.
    import librosa as _lb
    mel_fb = _lb.filters.mel(
        sr=hp["fe_sample_rate"], n_fft=hp["fe_n_fft"], n_mels=hp["fe_num_mels"],
        fmin=hp["fe_f_min"], fmax=hp["fe_f_max"], norm="slaney", htk=False,
    ).astype(np.float32)
    writer.add_tensor("frontend.mel_filterbank", np.ascontiguousarray(mel_fb),
                      raw_dtype=GGMLQuantizationType.F32)
    N = int(hp["fe_n_fft"])
    hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / N)).astype(np.float32)
    writer.add_tensor("frontend.window", np.ascontiguousarray(hann),
                      raw_dtype=GGMLQuantizationType.F32)

    # Baked time-embedding inv_freq (non-persistent buffer; recompute exactly).
    # inv_freq = exp(-log(theta) * arange(dim//2) / (dim//2)); t_cond =
    # cat(cos(t*inv_freq), sin(t*inv_freq)).
    half = hp["time_embed_dim"] // 2
    inv_freq = np.exp(-math.log(hp["time_embed_theta"]) * np.arange(half) / half).astype(np.float32)
    writer.add_tensor("dec.time_embed.inv_freq", np.ascontiguousarray(inv_freq),
                      raw_dtype=GGMLQuantizationType.F32)
    n_added += 3
    for a in (mel_fb, hann, inv_freq):
        bytes_in += a.nbytes
        bytes_out += a.nbytes

    consumed: set[str] = set()

    def add(src_name: str, dst_name: str) -> None:
        nonlocal n_added, bytes_in, bytes_out
        if src_name not in sd:
            raise KeyError(f"state_dict missing tensor: {src_name!r}")
        t = sd[src_name]
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{src_name}: unexpected dtype {t.dtype}")
        arr = np.ascontiguousarray(t.float().numpy())
        target_type = reference_dtype_for(dst_name, REFERENCE_GGML_TYPE)
        encoded, raw_dtype = encode_for_gguf(arr, target_type)
        writer.add_tensor(dst_name, encoded, raw_dtype=raw_dtype)
        bytes_in += int(arr.nbytes)
        bytes_out += int(encoded.nbytes)
        n_added += 1
        consumed.add(src_name)

    for src, dst in ENC_TOP_TABLE:
        add(src, dst)
    for i in range(hp["enc_n_layers"]):
        for s, d in ENC_BLOCK_TABLE:
            add(f"model.audio_tower.layers.{i}.{s}", f"enc.blocks.{i}.{d}")
    for src, dst in PROJ_TABLE:
        add(src, dst)
    for src, dst in TEXT_TOP_TABLE:
        add(src, dst)
    for i in range(hp["dec_n_layers"]):
        for s, d in TEXT_BLOCK_TABLE:
            add(f"model.language_model.layers.{i}.{s}", f"dec.blocks.{i}.{d}")

    # lm_head is tied (verified above); count it consumed without emitting.
    consumed.add("lm_head.weight")

    unused = sorted(set(sd.keys()) - consumed)
    if unused:
        raise RuntimeError(f"{len(unused)} state_dict keys not consumed: {unused[:10]}")

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


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a Voxtral Realtime (2602) checkpoint to a BF16 reference GGUF.")
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

    if looks_like_repo_id(args.model):
        repo_id = args.repo_id or args.model
        model_dir = download_snapshot(args.model, args.revision)
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
