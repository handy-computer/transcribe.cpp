#!/usr/bin/env python3
"""
convert-moss.py - convert a MOSS-Transcribe-Diarize HuggingFace checkpoint
to a BF16 reference GGUF that transcribe.cpp's loader can ingest. Block
quantization (Q8_0, Q5_K_M, ...) is a Stage 5 concern via
tools/transcribe-quantize.

Source format:
    HuggingFace repo/dir (OpenMOSS-Team/MOSS-Transcribe-Diarize), with:

      config.json              MossTranscribeDiarizeConfig (audio_config +
                               text_config + fusion params)
      generation_config.json   Generation defaults
      preprocessor_config.json WhisperFeatureExtractor parameters
      processor_config.json    audio_tokens_per_second / merge_size /
                               time_marker params (time-marker injection)
      tokenizer.json           Qwen2 BPE + 29 added special tokens (audio
                               placeholders at 151669/151670/151671)
      vocab.json / merges.txt  BPE base vocab + merges
      chat_template.jinja      Jinja chat template
      model-00000-of-00001.safetensors   BF16 weights (683 tensors)

Architecture: audio-llm. HF state dict has three top-level prefixes:

    model.whisper_encoder.*   -> HF WhisperEncoder (24 layers, gelu,
                                 LayerNorm; conv1/conv2 + learned pos
                                 embeddings). Emitted with the SAME tensor
                                 names as the `whisper` family so the C++
                                 encoder graph can be shared.
    model.vq_adaptor.*        -> VQAdaptor bridge: Linear(4096->1024) ->
                                 SiLU -> Linear(1024->1024) -> LayerNorm.
                                 Consumes the 4x-time-merged encoder output.
    model.language_model.*    -> Qwen3-0.6B causal LM (28 layers, GQA 16/8,
                                 head_dim=128, q/k RMSNorm, SwiGLU, standard
                                 RoPE theta 1e6). Emitted with the SAME
                                 names as the `qwen3_asr` text LM.
    lm_head.weight            -> NOT stored (tie_word_embeddings=true; the
                                 loader reuses dec.token_embd.weight).

Layout conversions: NONE. Linear weights are PyTorch (out, in); Conv1d
kernels are (out, in, k) — both already match ggml.

KV emitted:

    general.architecture   = "moss"
    general.basename       = "moss-transcribe-diarize"
    general.languages      = ["en", "zh"]

    stt.variant            = "moss-transcribe-diarize"
    stt.capability.speaker_diarization = true

    tokenizer.ggml.*       (llama.cpp "gpt2" byte-level BPE, table padded to
                            the 151936-wide embedding/logits vocab)
    stt.moss.encoder.*     Whisper encoder hparams
    stt.moss.decoder.*     Qwen3 text-LM hparams
    stt.moss.adaptor.*     VQAdaptor dims + merge size
    stt.moss.audio_token_id / audio_tokens_per_second /
      time_marker_every_seconds / enable_time_marker  (audio-span +
      time-marker construction — see modeling/processing notes)
    stt.frontend.*         Whisper frontend parameters

CLI:

    uv run --project scripts/envs/moss \
      scripts/convert-moss.py OpenMOSS-Team/MOSS-Transcribe-Diarize \
        --revision d7231bbae2587a4af278735eb765b318c4f64edd

Single-file, top-to-bottom.
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


class _ShardedSafetensors:
    """Multi-file safe_open shim: reads model.safetensors.index.json (or a
    single model.safetensors) and exposes keys()/get_tensor()."""

    def __init__(self, model_dir: Path) -> None:
        self._stack = ExitStack()
        single = model_dir / "model.safetensors"
        if single.is_file():
            sf = self._stack.enter_context(safe_open(str(single), framework="pt"))
            self._handles = {"__single__": sf}
            self._shard_for = {k: "__single__" for k in sf.keys()}
            return
        index_path = model_dir / "model.safetensors.index.json"
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
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


# ---------------------------------------------------------------------------
# Tokenizer extraction (Qwen2 byte-level BPE)
# ---------------------------------------------------------------------------
#
# Base vocab (vocab.json, 151643 tokens) + merges (merges.txt, 151388) +
# the 29 added special tokens from tokenizer.json (ids 151643..151671,
# including audio placeholders). The embedding/logits table is padded to
# text_config.vocab_size (151936); unused slots become <|unused_N|>
# placeholders so the token table length matches the model's logit width.


def extract_tokenizer(model_dir: Path, vocab_size: int) -> dict:
    with (model_dir / "vocab.json").open(encoding="utf-8") as f:
        base_vocab: dict[str, int] = json.load(f)
    with (model_dir / "merges.txt").open(encoding="utf-8") as f:
        merges = [ln.rstrip("\n") for ln in f]
        if merges and merges[0].startswith("#"):
            merges = merges[1:]
    with (model_dir / "tokenizer.json").open(encoding="utf-8") as f:
        tok_json = json.load(f)
    with (model_dir / "tokenizer_config.json").open(encoding="utf-8") as f:
        tokcfg = json.load(f)

    # id -> (token_str, is_special)
    tok_by_id: dict[int, tuple[str, bool]] = {}
    for tok, tid in base_vocab.items():
        tok_by_id[int(tid)] = (tok, False)
    for entry in tok_json.get("added_tokens", []):
        tok_by_id[int(entry["id"])] = (entry["content"], bool(entry.get("special", False)))

    content_to_id = {v[0]: k for k, v in tok_by_id.items()}

    max_id = max(tok_by_id.keys())
    if max_id + 1 > vocab_size:
        raise ValueError(f"tokenizer id {max_id} exceeds vocab_size={vocab_size}")

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
        tok = tokcfg.get(field)
        if isinstance(tok, dict):
            tok = tok.get("content")
        if not tok:
            return None
        return content_to_id.get(tok)

    return {
        "tokens": tokens,
        "types": types,
        "merges": merges,
        "eos_id": _name_to_id("eos_token"),      # <|im_end|> = 151645
        "pad_id": _name_to_id("pad_token"),       # <|endoftext|> = 151643
        "bos_id": None,                            # Qwen: no BOS prepended
        "audio_pad_id": content_to_id.get("<|audio_pad|>"),
        "audio_start_id": content_to_id.get("<|audio_start|>"),
        "audio_end_id": content_to_id.get("<|audio_end|>"),
    }


# ---------------------------------------------------------------------------
# Hparams
# ---------------------------------------------------------------------------


def read_hparams(config: dict, preproc: dict, proc: dict) -> dict:
    aenc = config["audio_config"]
    tdec = config["text_config"]

    sample_rate = int(preproc.get("sampling_rate", 16000))
    hop_length = int(preproc["hop_length"])
    n_fft = int(preproc["n_fft"])
    n_mels = int(preproc["feature_size"])
    chunk_len = int(preproc.get("chunk_length", 30))
    n_samples = int(preproc.get("n_samples", chunk_len * sample_rate))
    nb_max_frm = int(preproc.get("nb_max_frames", n_samples // hop_length))

    return {
        # Audio encoder (HF WhisperEncoder)
        "enc_n_layers":    int(aenc["encoder_layers"]),
        "enc_d_model":     int(aenc["d_model"]),
        "enc_n_heads":     int(aenc["encoder_attention_heads"]),
        "enc_ffn_dim":     int(aenc["encoder_ffn_dim"]),
        "enc_n_mels":      int(aenc["num_mel_bins"]),
        "enc_max_src_pos": int(aenc["max_source_positions"]),
        "enc_activation":  str(aenc["activation_function"]).lower(),
        "enc_scale_embedding": bool(aenc.get("scale_embedding", False)),

        # Text LM (Qwen3-0.6B)
        "dec_n_layers":     int(tdec["num_hidden_layers"]),
        "dec_hidden":       int(tdec["hidden_size"]),
        "dec_intermediate": int(tdec["intermediate_size"]),
        "dec_n_heads":      int(tdec["num_attention_heads"]),
        "dec_n_kv_heads":   int(tdec["num_key_value_heads"]),
        "dec_head_dim":     int(tdec["head_dim"]),
        "dec_hidden_act":   str(tdec["hidden_act"]).lower(),
        "dec_rms_norm_eps": float(tdec["rms_norm_eps"]),
        "dec_rope_theta":   float(tdec["rope_theta"]),
        "dec_max_pos_emb":  int(tdec["max_position_embeddings"]),
        "dec_tie_embeddings": bool(config.get("tie_word_embeddings", True)),
        "dec_vocab_size":   int(tdec["vocab_size"]),

        # Adaptor + fusion
        "adaptor_input_dim": int(config["adaptor_input_dim"]),
        "audio_merge_size":  int(config["audio_merge_size"]),
        "audio_token_id":    int(config["audio_token_id"]),

        # Processor time-marker injection (audio-span construction)
        "audio_tokens_per_second":  float(proc["audio_tokens_per_second"]),
        "time_marker_every_seconds": int(proc["time_marker_every_seconds"]),
        "enable_time_marker":        bool(proc["enable_time_marker"]),

        # Frontend (Whisper feature extractor)
        "fe_sample_rate": sample_rate,
        "fe_num_mels":    n_mels,
        "fe_n_fft":       n_fft,
        "fe_win_length":  n_fft,
        "fe_hop_length":  hop_length,
        "fe_window":      "hann_periodic",
        "fe_normalize":   "per_utterance",
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

        "languages": ["en", "zh"],
    }


# ---------------------------------------------------------------------------
# Tensor name maps (source HF name -> GGUF name)
# ---------------------------------------------------------------------------

# Whisper encoder top-level (matches the `whisper` family names exactly).
ENC_TOP_TABLE: list[tuple[str, str]] = [
    ("model.whisper_encoder.conv1.weight",           "enc.conv.0.weight"),
    ("model.whisper_encoder.conv1.bias",             "enc.conv.0.bias"),
    ("model.whisper_encoder.conv2.weight",           "enc.conv.1.weight"),
    ("model.whisper_encoder.conv2.bias",             "enc.conv.1.bias"),
    ("model.whisper_encoder.embed_positions.weight", "enc.pos_emb.weight"),
    ("model.whisper_encoder.layer_norm.weight",      "enc.final_norm.weight"),
    ("model.whisper_encoder.layer_norm.bias",        "enc.final_norm.bias"),
]

# Whisper attention: q / v / out have bias, k does NOT.
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

# VQAdaptor: Sequential(Linear[0], SiLU[1], Linear[2], LayerNorm[3]).
# norm_out.* hits the F32 bucket via the "norm_" rule in reference_dtype_for.
ADAPTOR_TABLE: list[tuple[str, str]] = [
    ("model.vq_adaptor.layers.0.weight", "adaptor.fc1.weight"),
    ("model.vq_adaptor.layers.0.bias",   "adaptor.fc1.bias"),
    ("model.vq_adaptor.layers.2.weight", "adaptor.fc2.weight"),
    ("model.vq_adaptor.layers.2.bias",   "adaptor.fc2.bias"),
    ("model.vq_adaptor.layers.3.weight", "adaptor.norm_out.weight"),
    ("model.vq_adaptor.layers.3.bias",   "adaptor.norm_out.bias"),
]

# Qwen3 text LM top-level (tied head: lm_head not stored).
DEC_TOP_TABLE: list[tuple[str, str]] = [
    ("model.language_model.embed_tokens.weight", "dec.token_embd.weight"),
    ("model.language_model.norm.weight",         "dec.output_norm.weight"),
]

# Qwen3 per-layer (q/k RMSNorm, no attention biases).
DEC_BLOCK_TABLE: list[tuple[str, str]] = [
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

# tie_word_embeddings=true — lm_head.weight is not present in the state dict.
SKIP_EXACT: set[str] = set()


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

    config_path = model_dir / "config.json"
    preproc_path = model_dir / "preprocessor_config.json"
    proc_path = model_dir / "processor_config.json"
    chat_template_path = model_dir / "chat_template.jinja"

    for p in (config_path, preproc_path, proc_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file: {p}")

    with config_path.open() as f:
        config = json.load(f)
    with preproc_path.open() as f:
        preproc = json.load(f)
    with proc_path.open() as f:
        proc = json.load(f)
    chat_template = None
    if chat_template_path.is_file():
        chat_template = chat_template_path.read_text(encoding="utf-8")

    hp = read_hparams(config, preproc, proc)
    print(f"Audio encoder: {hp['enc_n_layers']} layers, d_model={hp['enc_d_model']}, "
          f"heads={hp['enc_n_heads']}, act={hp['enc_activation']}")
    print(f"Adaptor: {hp['adaptor_input_dim']} -> {hp['dec_hidden']} "
          f"(merge={hp['audio_merge_size']})")
    print(f"Text LM: {hp['dec_n_layers']} layers, hidden={hp['dec_hidden']}, "
          f"heads={hp['dec_n_heads']}/{hp['dec_n_kv_heads']}, "
          f"tie_embeddings={hp['dec_tie_embeddings']}")
    print(f"Time-marker: every {hp['time_marker_every_seconds']}s, "
          f"{hp['audio_tokens_per_second']} tok/s, enabled={hp['enable_time_marker']}")

    print(f"Reading tokenizer from {model_dir}")
    tok = extract_tokenizer(model_dir, hp["dec_vocab_size"])
    if tok["audio_pad_id"] != hp["audio_token_id"]:
        raise ValueError(
            f"audio_pad token id mismatch: config={hp['audio_token_id']} "
            f"tokenizer={tok['audio_pad_id']}")

    print(f"Opening safetensors under {model_dir}")
    with _ShardedSafetensors(model_dir) as st:
        st_keys = set(st.keys())

        total = sum(st.get_tensor(k).numel() for k in st_keys if k not in SKIP_EXACT)
        size_label = compute_size_label(total)
        print(f"Total params: {total:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = gguf_writer(str(out_path), "moss")

        # ---- general.* ----
        add_general_identity(
            writer,
            name="MOSS-Transcribe-Diarize 0.9B",
            basename="moss-transcribe-diarize",
            size_label=size_label,
            file_type=REFERENCE_FILE_TYPE,
            languages=hp["languages"],
            author="MOSI.AI",
            organization="OpenMOSS-Team",
            license="apache-2.0",
            license_name="Apache License 2.0",
            license_link="https://www.apache.org/licenses/LICENSE-2.0",
            repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
        )

        writer.add_string("stt.variant", variant)

        # Speaker diarization is a real model capability (emergent [Sxx]
        # text). Matches intake capabilities.speaker_diarization=true.
        writer.add_bool("stt.capability.speaker_diarization", True)

        # ---- tokenizer.ggml.* (llama.cpp "gpt2" byte-level BPE) ----
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_string("tokenizer.ggml.pre", "qwen2")
        writer.add_array("tokenizer.ggml.tokens", tok["tokens"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        writer.add_array("tokenizer.ggml.merges", tok["merges"])
        if tok["eos_id"] is not None:
            writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
        if tok["pad_id"] is not None:
            writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])
        writer.add_bool("tokenizer.ggml.add_bos_token", False)
        if chat_template is not None:
            writer.add_string("tokenizer.chat_template", chat_template)

        # ---- stt.moss.encoder.* (Whisper) ----
        writer.add_uint32("stt.moss.encoder.n_layers",            hp["enc_n_layers"])
        writer.add_uint32("stt.moss.encoder.d_model",             hp["enc_d_model"])
        writer.add_uint32("stt.moss.encoder.n_heads",             hp["enc_n_heads"])
        writer.add_uint32("stt.moss.encoder.ffn_dim",             hp["enc_ffn_dim"])
        writer.add_uint32("stt.moss.encoder.num_mel_bins",        hp["enc_n_mels"])
        writer.add_uint32("stt.moss.encoder.max_source_positions", hp["enc_max_src_pos"])
        writer.add_string("stt.moss.encoder.activation",          hp["enc_activation"])
        writer.add_bool("stt.moss.encoder.scale_embedding",       hp["enc_scale_embedding"])

        # ---- stt.moss.decoder.* (Qwen3 text LM) ----
        writer.add_uint32("stt.moss.decoder.n_layers",       hp["dec_n_layers"])
        writer.add_uint32("stt.moss.decoder.hidden_size",    hp["dec_hidden"])
        writer.add_uint32("stt.moss.decoder.intermediate_size", hp["dec_intermediate"])
        writer.add_uint32("stt.moss.decoder.n_heads",        hp["dec_n_heads"])
        writer.add_uint32("stt.moss.decoder.n_kv_heads",     hp["dec_n_kv_heads"])
        writer.add_uint32("stt.moss.decoder.head_dim",       hp["dec_head_dim"])
        writer.add_string("stt.moss.decoder.hidden_act",     hp["dec_hidden_act"])
        writer.add_float32("stt.moss.decoder.rms_norm_eps",  hp["dec_rms_norm_eps"])
        writer.add_float32("stt.moss.decoder.rope_theta",    hp["dec_rope_theta"])
        writer.add_uint32("stt.moss.decoder.max_position_embeddings", hp["dec_max_pos_emb"])
        writer.add_bool("stt.moss.decoder.tie_word_embeddings", hp["dec_tie_embeddings"])
        writer.add_uint32("stt.moss.decoder.vocab_size",     hp["dec_vocab_size"])

        # ---- stt.moss.adaptor.* + fusion / time-marker ----
        writer.add_uint32("stt.moss.adaptor.input_dim",      hp["adaptor_input_dim"])
        writer.add_uint32("stt.moss.audio_merge_size",       hp["audio_merge_size"])
        writer.add_uint32("stt.moss.audio_token_id",         hp["audio_token_id"])
        writer.add_float32("stt.moss.audio_tokens_per_second", hp["audio_tokens_per_second"])
        writer.add_uint32("stt.moss.time_marker_every_seconds", hp["time_marker_every_seconds"])
        writer.add_bool("stt.moss.enable_time_marker",       hp["enable_time_marker"])

        # ---- stt.frontend.* (Whisper feature extractor) ----
        writer.add_string("stt.frontend.type",          "mel")
        writer.add_uint32("stt.frontend.num_mels",      hp["fe_num_mels"])
        writer.add_uint32("stt.frontend.sample_rate",   hp["fe_sample_rate"])
        writer.add_uint32("stt.frontend.n_fft",         hp["fe_n_fft"])
        writer.add_uint32("stt.frontend.win_length",    hp["fe_win_length"])
        writer.add_uint32("stt.frontend.hop_length",    hp["fe_hop_length"])
        writer.add_string("stt.frontend.window",        hp["fe_window"])
        writer.add_string("stt.frontend.normalize",     hp["fe_normalize"])
        writer.add_float32("stt.frontend.dither",       hp["fe_dither"])
        writer.add_float32("stt.frontend.pre_emphasis", hp["fe_pre_emphasis"])
        writer.add_float32("stt.frontend.f_min",        hp["fe_f_min"])
        writer.add_float32("stt.frontend.f_max",        hp["fe_f_max"])
        writer.add_string("stt.frontend.pad_mode",      hp["fe_pad_mode"])
        writer.add_bool("stt.frontend.center",          hp["fe_center"])
        writer.add_string("stt.frontend.mel_norm",      hp["fe_mel_norm"])
        writer.add_uint32("stt.frontend.chunk_length",  hp["fe_chunk_length"])
        writer.add_uint32("stt.frontend.n_samples",     hp["fe_n_samples"])
        writer.add_uint32("stt.frontend.nb_max_frames", hp["fe_nb_max_frm"])

        # ---- tensors ----
        n_added = bytes_in = bytes_out = 0

        # Frontend buffers: slaney mel filterbank + Whisper periodic Hann
        # window, baked in so the C++ MelFrontend is bit-identical.
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
            if t.dtype != torch.bfloat16:
                raise ValueError(f"{src_name}: expected torch.bfloat16, got {t.dtype}")
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
                add(f"model.whisper_encoder.layers.{i}.{s}", f"enc.blocks.{i}.{d}")
        for src, dst in ADAPTOR_TABLE:
            add(src, dst)
        for src, dst in DEC_TOP_TABLE:
            add(src, dst)
        for i in range(hp["dec_n_layers"]):
            for s, d in DEC_BLOCK_TABLE:
                add(f"model.language_model.layers.{i}.{s}", f"dec.blocks.{i}.{d}")

        expected = (
            len(ENC_TOP_TABLE)
            + hp["enc_n_layers"] * len(ENC_BLOCK_TABLE)
            + len(ADAPTOR_TABLE)
            + len(DEC_TOP_TABLE)
            + hp["dec_n_layers"] * len(DEC_BLOCK_TABLE)
            + 2  # frontend buffers
        )
        if n_added != expected:
            raise RuntimeError(f"tensor count mismatch: added {n_added}, expected {expected}")
        print(f"Added {n_added} tensors "
              f"({bytes_in / (1024 * 1024):.1f} MB fp32 -> "
              f"{bytes_out / (1024 * 1024):.1f} MB on disk)")

        # Warn about unconsumed safetensors keys.
        consumed = set(SKIP_EXACT)
        for src, _ in ENC_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["enc_n_layers"]):
            for s, _ in ENC_BLOCK_TABLE:
                consumed.add(f"model.whisper_encoder.layers.{i}.{s}")
        for src, _ in ADAPTOR_TABLE:
            consumed.add(src)
        for src, _ in DEC_TOP_TABLE:
            consumed.add(src)
        for i in range(hp["dec_n_layers"]):
            for s, _ in DEC_BLOCK_TABLE:
                consumed.add(f"model.language_model.layers.{i}.{s}")
        unused = sorted(st_keys - consumed)
        if unused:
            print(f"WARNING: {len(unused)} safetensors keys not consumed:", file=sys.stderr)
            for k in unused[:20]:
                print(f"  {k}", file=sys.stderr)

        print("Writing header + KV + tensor info...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        print("Writing tensor data...")
        writer.write_tensors_to_file()
        writer.close()

    print(f"Done. Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a MOSS-Transcribe-Diarize checkpoint to a BF16 GGUF.")
    p.add_argument("model", type=str,
                   help="HF repo id (OpenMOSS-Team/MOSS-Transcribe-Diarize) or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the output slug from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch/tag/commit SHA) to pin the download to. "
                        "Ignored when `model` is a local directory.")
    p.add_argument("--variant", type=str, default=None,
                   help="stt.variant string (default: derived from slug)")
    args = p.parse_args(argv[1:])

    if looks_like_repo_id(args.model):
        repo_id = args.repo_id or args.model
        model_dir = download_snapshot(args.model, args.revision)
    else:
        model_dir = Path(args.model)
        if not model_dir.is_dir():
            print(f"error: {model_dir} is not a directory and not an HF repo id", file=sys.stderr)
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

    variant = args.variant or (slug_from_repo_id(repo_id).lower() if repo_id else out_path.stem)
    convert(model_dir, out_path, variant, repo_id=repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
