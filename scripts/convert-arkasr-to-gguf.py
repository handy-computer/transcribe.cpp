#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#   "gguf",
#   "huggingface-hub",
#   "numpy",
#   "safetensors",
#   "torch",
#   "transformers",
# ]
# ///
"""Convert ARK-ASR 0.6B or 3B to GGUF format for transcribe.cpp.

Architecture (see PLAN.md §ARK):
  * Audio encoder: Whisper-large-v3 (128 mel, d_model 1280, 32 layers, 20 heads)
    with *partial interleaved RoPE* (rot_dim 32 of head_dim 64, theta 10000,
    Q+K only); k_proj has no bias; the encoder's own final LayerNorm is dropped
    (nn.Identity).
  * Adapter: LayerNorm(1280) -> merge 4 consecutive frames (->5120) ->
    Linear 5120->4096 -> GELU -> Linear 4096->2048.
  * Decoder: stock Qwen2.5-3B (2048 hid, 36 L, GQA 16/2, SwiGLU, RMSNorm,
    rope_theta 1e6, vocab 151936, tied embeddings).

Audio placeholder token <|audio|> (151663) embeddings are replaced by the first
N = ((mel_frames+1)//2)//4 adapter frames.

Streams tensors one-at-a-time via safe_open (BF16->F16). ~9 GB peak RAM.

Usage:
    uv run --python 3.12 scripts/convert-arkasr-to-gguf.py --input AutoArk-AI/ARK-ASR-3B \
        --output arkasr-3b-f16.gguf --outtype f16
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Error: gguf package not found. Install with: pip install gguf")
    sys.exit(1)

try:
    from safetensors import safe_open
except ImportError:
    print("Error: safetensors package not found. Install with: pip install safetensors")
    sys.exit(1)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


def load_model_dir(model_id: str, revision: str | None = None) -> Path:
    p = Path(model_id)
    if p.is_dir():
        return p
    if snapshot_download is None:
        print("Error: huggingface_hub not installed and --input is not a local dir")
        sys.exit(1)
    print(f"Downloading model from HuggingFace: {model_id}")
    return Path(snapshot_download(
        model_id, revision=revision,
        allow_patterns=["*.safetensors", "*.json", "merges.txt", "*.jinja"],
    ))


def map_tensor_name(name: str):
    """Map an HF tensor name to a GGUF name, or None to skip it."""
    # --- decoder (Qwen2.5-3B) ---
    if name == "model.embed_tokens.weight":
        return "dec.embed.weight"
    if name == "model.norm.weight":
        return "dec.norm.weight"
    if name == "lm_head.weight":
        return None  # tied to dec.embed.weight; reused at runtime
    if name.startswith("model.layers."):
        rest = name[len("model.layers."):]
        idx, _, sub = rest.partition(".")
        m = {
            "self_attn.q_proj.weight": "attn.q.weight",
            "self_attn.q_proj.bias": "attn.q.bias",
            "self_attn.k_proj.weight": "attn.k.weight",
            "self_attn.k_proj.bias": "attn.k.bias",
            "self_attn.v_proj.weight": "attn.v.weight",
            "self_attn.v_proj.bias": "attn.v.bias",
            "self_attn.o_proj.weight": "attn.o.weight",
            "input_layernorm.weight": "attn_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate_proj.weight": "ffn.gate.weight",
            "mlp.up_proj.weight": "ffn.up.weight",
            "mlp.down_proj.weight": "ffn.down.weight",
        }.get(sub)
        return f"dec.blk.{idx}.{m}" if m else None

    # --- adapter ---
    if name == "audio_encoder.layer_norm.weight":
        return "adapter.ln.weight"
    if name == "audio_encoder.layer_norm.bias":
        return "adapter.ln.bias"
    if name == "audio_encoder.adapting.0.weight":
        return "adapter.fc1.weight"
    if name == "audio_encoder.adapting.0.bias":
        return "adapter.fc1.bias"
    if name == "audio_encoder.adapting.2.weight":
        return "adapter.fc2.weight"
    if name == "audio_encoder.adapting.2.bias":
        return "adapter.fc2.bias"

    # --- whisper encoder ---
    if name.startswith("audio_encoder.whisper."):
        rest = name[len("audio_encoder.whisper."):]
        if rest == "conv1.weight":
            return "enc.conv1.weight"
        if rest == "conv1.bias":
            return "enc.conv1.bias"
        if rest == "conv2.weight":
            return "enc.conv2.weight"
        if rest == "conv2.bias":
            return "enc.conv2.bias"
        if rest == "embed_positions.weight":
            return None  # unused (use_rope=True)
        if rest in ("layer_norm.weight", "layer_norm.bias"):
            return None  # encoder final LN replaced by nn.Identity
        if rest.startswith("layers."):
            r2 = rest[len("layers."):]
            idx, _, sub = r2.partition(".")
            m = {
                "self_attn.q_proj.weight": "attn.q.weight",
                "self_attn.q_proj.bias": "attn.q.bias",
                "self_attn.k_proj.weight": "attn.k.weight",  # no bias
                "self_attn.v_proj.weight": "attn.v.weight",
                "self_attn.v_proj.bias": "attn.v.bias",
                "self_attn.out_proj.weight": "attn.o.weight",
                "self_attn.out_proj.bias": "attn.o.bias",
                "self_attn_layer_norm.weight": "attn_ln.weight",
                "self_attn_layer_norm.bias": "attn_ln.bias",
                "fc1.weight": "fc1.weight",
                "fc1.bias": "fc1.bias",
                "fc2.weight": "fc2.weight",
                "fc2.bias": "fc2.bias",
                "final_layer_norm.weight": "ffn_ln.weight",
                "final_layer_norm.bias": "ffn_ln.bias",
            }.get(sub)
            return f"enc.blk.{idx}.{m}" if m else None
        return None
    return None


def build_mel_and_window(n_fft: int, n_mels: int, sr: int):
    """Whisper feature-extractor mel filterbank (n_freqs, n_mels) + hann window.

    Mirrors transformers.WhisperFeatureExtractor: slaney mel scale + slaney
    norm, 0..sr/2 Hz, and torch.hann_window(periodic=True)."""
    from transformers.audio_utils import mel_filter_bank
    n_freqs = 1 + n_fft // 2
    fb = mel_filter_bank(
        num_frequency_bins=n_freqs,
        num_mel_filters=n_mels,
        min_frequency=0.0,
        max_frequency=float(sr) / 2.0,
        sampling_rate=sr,
        norm="slaney",
        mel_scale="slaney",
    )  # (n_freqs, n_mels) float32
    fb = np.ascontiguousarray(fb, dtype=np.float32)
    # periodic hann window of length n_fft
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    return fb, win


def main():
    ap = argparse.ArgumentParser(description="Convert ARK-ASR-3B to GGUF")
    ap.add_argument("--input", required=True, help="HF model ID or local dir")
    ap.add_argument("--output", required=True, help="Output GGUF path")
    ap.add_argument("--outtype", default="f16", choices=["f16", "f32", "q8_0"],
                    help="dtype for 2D weights (default f16). q8_0 quantizes 2D "
                         "decoder/encoder/adapter weights; keeps the tied embedding, "
                         "conv kernels (F16) and norms/filterbank (F32).")
    ap.add_argument("--variant", required=True, choices=["ark-asr-0.6b", "ark-asr-3b"])
    ap.add_argument("--revision", help="Pinned Hugging Face source revision")
    args = ap.parse_args()

    mdir = load_model_dir(args.input, args.revision)
    with open(mdir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    wc = cfg["whisper_config"]
    expected_hidden = {"ark-asr-0.6b": 896, "ark-asr-3b": 2048}[args.variant]
    if cfg["model_type"] != "arkasr" or cfg["hidden_size"] != expected_hidden:
        raise ValueError("Checkpoint does not match the selected ARK variant")
    if not cfg.get("tie_word_embeddings") or not cfg.get("use_rope"):
        raise ValueError("ARK runtime requires tied embeddings and RoPE")

    w = GGUFWriter(args.output, "arkasr")
    w.add_name(args.variant.upper())
    w.add_string("stt.variant", args.variant)
    w.add_bool("stt.capability.lang_detect", True)
    w.add_array("general.languages", ["zh", "en", "de", "ja", "fr", "ko", "es", "pl", "it", "ro", "hu", "cs", "nl", "fi", "hr", "sk", "sl", "et", "lt"] if args.variant == "ark-asr-0.6b" else ["zh", "en", "de", "ja", "fr", "ko", "es"])
    w.add_description("ARK-ASR (Whisper-RoPE encoder + Qwen2 decoder)")

    # ---- decoder (Qwen2.5-3B) hparams ----
    hid = int(cfg["hidden_size"])
    n_heads = int(cfg["num_attention_heads"])
    w.add_uint32("arkasr.llm.hidden_size", hid)
    w.add_uint32("arkasr.llm.num_layers", int(cfg["num_hidden_layers"]))
    w.add_uint32("arkasr.llm.num_heads", n_heads)
    w.add_uint32("arkasr.llm.num_kv_heads", int(cfg["num_key_value_heads"]))
    w.add_uint32("arkasr.llm.head_dim", hid // n_heads)
    w.add_uint32("arkasr.llm.intermediate_size", int(cfg["intermediate_size"]))
    w.add_uint32("arkasr.llm.vocab_size", int(cfg["vocab_size"]))
    w.add_uint32("arkasr.llm.max_position_embeddings", int(cfg.get("max_position_embeddings", 32768)))
    w.add_float32("arkasr.llm.rope_theta", float(cfg.get("rope_theta", 1000000.0)))
    w.add_float32("arkasr.llm.rms_norm_eps", float(cfg.get("rms_norm_eps", 1e-6)))

    # ---- whisper encoder hparams ----
    d_model = int(wc["d_model"])
    e_heads = int(wc["encoder_attention_heads"])
    e_head_dim = d_model // e_heads
    w.add_uint32("arkasr.whisper.d_model", d_model)
    w.add_uint32("arkasr.whisper.num_layers", int(wc["encoder_layers"]))
    w.add_uint32("arkasr.whisper.num_heads", e_heads)
    w.add_uint32("arkasr.whisper.head_dim", e_head_dim)
    w.add_uint32("arkasr.whisper.ffn_dim", int(wc["encoder_ffn_dim"]))
    w.add_uint32("arkasr.whisper.num_mel_bins", int(wc["num_mel_bins"]))
    w.add_uint32("arkasr.whisper.max_source_positions", int(wc["max_source_positions"]))
    # RotaryEmbedding(dim = head_dim // 2), base 10000 (rope_ratio 1)
    w.add_uint32("arkasr.whisper.rot_dim", e_head_dim // 2)
    w.add_float32("arkasr.whisper.rope_theta", 10000.0)
    w.add_float32("arkasr.whisper.ln_eps", 1e-5)

    # ---- adapter / audio / tokenizer hparams ----
    w.add_uint32("arkasr.adapter.merge_factor", int(cfg.get("merge_factor", 4)))
    w.add_uint32("arkasr.audio_token_id", int(cfg["audio_token_id"]))
    w.add_uint32("arkasr.bos_token_id", int(cfg.get("bos_token_id", 151643)))
    w.add_uint32("arkasr.eos_token_id", int(cfg.get("eos_token_id", 151645)))
    w.add_uint32("arkasr.pad_token_id", int(cfg.get("pad_token_id", 151643)))
    w.add_uint32("arkasr.n_fft", 400)
    w.add_uint32("arkasr.hop_length", 160)
    w.add_uint32("arkasr.sample_rate", 16000)

    # ---- mel filterbank + window ----
    fb, win = build_mel_and_window(400, int(wc["num_mel_bins"]), 16000)
    w.add_tensor("enc.mel_filters", fb, raw_dtype=GGMLQuantizationType.F32)
    w.add_tensor("enc.mel_window", win, raw_dtype=GGMLQuantizationType.F32)

    # ---- tokenizer: GPT-2 byte-level BPE (vocab + merges + specials) ----
    vocab_size = int(cfg["vocab_size"])
    tokens = [f"[PAD{i}]" for i in range(vocab_size)]
    vj = mdir / "vocab.json"
    if vj.exists():
        with open(vj, encoding="utf-8") as f:
            vmap = json.load(f)
        for tok, idx in vmap.items():
            if 0 <= idx < vocab_size:
                tokens[idx] = tok
    aj = mdir / "added_tokens.json"
    if aj.exists():
        with open(aj, encoding="utf-8") as f:
            for tok, idx in json.load(f).items():
                if 0 <= idx < vocab_size:
                    tokens[idx] = tok
    w.add_tokenizer_model("gpt2")
    w.add_token_list(tokens)
    mt = mdir / "merges.txt"
    if mt.exists():
        merges = []
        with open(mt, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line and not line.startswith("#"):
                    merges.append(line)
        if merges:
            w.add_token_merges(merges)
    else:
        print("WARNING: merges.txt missing — decode-only is fine, encode falls back to per-byte")

    # ---- stream tensors ----
    index_path = mdir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
    else:
        # single-file checkpoint
        only = next(mdir.glob("*.safetensors"))
        weight_map = {}
        with safe_open(str(only), framework="pt") as h:
            for k in h.keys():
                weight_map[k] = only.name

    handles = {}
    def get_handle(fname):
        if fname not in handles:
            handles[fname] = safe_open(str(mdir / fname), framework="pt")
        return handles[fname]

    # q8_0: keep these at F16/F32 (sampling-critical / conv / filterbank).
    def keep_unquantized(name):
        return (name == "dec.embed.weight" or name.endswith("conv1.weight") or name.endswith("conv2.weight"))

    try:
        from gguf import quants as _gguf_quants
    except ImportError:
        _gguf_quants = None

    n_written = 0
    n_quant = 0
    for hf_name in sorted(weight_map.keys()):
        gguf_name = map_tensor_name(hf_name)
        if gguf_name is None:
            continue
        t = get_handle(weight_map[hf_name]).get_tensor(hf_name)
        if t.ndim <= 1:
            data = np.ascontiguousarray(t.to(torch.float32).numpy())
            w.add_tensor(gguf_name, data, raw_dtype=GGMLQuantizationType.F32)
        elif args.outtype == "q8_0" and t.ndim == 2 and not keep_unquantized(gguf_name) \
                and (t.shape[-1] % 32 == 0) and _gguf_quants is not None:
            arr = np.ascontiguousarray(t.to(torch.float32).numpy())
            qd = _gguf_quants.quantize(arr, GGMLQuantizationType.Q8_0)
            w.add_tensor(gguf_name, qd, raw_dtype=GGMLQuantizationType.Q8_0)
            n_quant += 1
        else:
            dt = torch.float32 if args.outtype == "f32" else torch.float16
            qt = GGMLQuantizationType.F32 if args.outtype == "f32" else GGMLQuantizationType.F16
            data = np.ascontiguousarray(t.to(dt).numpy())
            w.add_tensor(gguf_name, data, raw_dtype=qt)
        n_written += 1
        if n_written % 50 == 0:
            print(f"  ... {n_written} tensors ({n_quant} q8_0)")

    print(f"Writing GGUF ({n_written} tensors) -> {args.output}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("Done.")


if __name__ == "__main__":
    main()
