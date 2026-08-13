#!/usr/bin/env python3
"""
convert-crisperwhisper.py - convert a CrisperWhisper 2.0 HuggingFace
directory to a GGUF that transcribe.cpp's loader will ingest. Preserves
the source/reference dtype (BF16 for all four open variants); block
quantization (Q8_0, Q5_K_M, ...) goes through tools/transcribe-quantize
later.

Source format:
    A HuggingFace directory (or repo id), e.g. nyralabs/CrisperWhisper2.0_small:

      config.json               model_type=whisper; d_model, layers, heads, ...
      generation_config.json    special-token ids, suppress_tokens, lang_to_id,
                                alignment_heads (word timing)
      preprocessor_config.json  WhisperFeatureExtractor parameters. Unlike
                                whisper-tiny..large-v2 it carries NO
                                `mel_filters` array, so the slaney filterbank is
                                rebuilt here exactly as HF would at load time.
      tokenizer.json            full HF-fast tokenizer (BPE + 1639 added tokens)
      model.safetensors         BF16 weights (481 tensors for small)

Relationship to `whisper`
-------------------------
The GRAPH is stock Whisper: encoder, decoder, cross-attention, KV cache,
positional embeddings and the mel frontend are identical, and the emitted
tensor names are deliberately byte-identical to convert-whisper.py's so
src/arch/crisperwhisper/ can reuse the whisper graph code and so the
Stage 2 oracle dumps line up name-for-name.

What differs is everything ABOVE the graph, which is why this is a separate
family and a separate GGUF architecture rather than a whisper variant:

  * 31 added tokens (32 on turbo), none of them flagged `special`, carrying
    the verbatim/intended mode contract, the six prompt markers and 15
    vocal-event tokens that are INTENDED OUTPUT in verbatim mode.
  * the mode tags are prepended BEFORE the Whisper prefix, with no
    <|startofprev|> wrapper.
  * <|notimestamps|> is always forced; the model never emits timestamp
    tokens, so there are no segment timestamps to parse.
  * word timing is a Viterbi alignment over generation_config.alignment_heads,
    not Whisper's median-filtered DTW.
  * long-form is <ctx> conditional continuation over 30 s windows at a 26 s
    stride, not timestamp-token stitching.

The decode-contract ids and parameters those paths need are emitted as
`stt.crisperwhisper.*` KV below. Nothing is hardcoded: every token id is
resolved from tokenizer.json by literal content, because the turbo variant
shifts every id above 50357 by +1 (128 mel bins, 100 languages, one extra
language token).

Dropped tensors
---------------
`encoder_blank_head.{weight,bias}` [1, d_model] / [1] is present in all four
checkpoints and is NOT emitted. Evidence it is untrained training scaffold
rather than an inference weight:

  * unreferenced by the publisher's `crisperwhisper` package and by
    transformers' Whisper modelling code (the blank probabilities the Viterbi
    aligner uses come from `word_timing.blank_logp_from_mel_energy` /
    `blank_logp_from_space_attention`, never from a learned head);
  * `weight.std() = 0.0207` against `config.init_std = 0.02`, `mean ~= 1e-3`;
  * `bias` is exactly 0.0, i.e. still at its zero-init;
  * `config.architectures` names WhisperForConditionalGenerationWithAttentionLoss,
    a training-time subclass that exists nowhere in the published inference
    package (HF Auto classes dispatch on `model_type: whisper`).

Emitting it would put an unused, untrained tensor in every quant of every
variant. See docs/porting/families/crisperwhisper.md (Notes).

Tensor naming (identical to convert-whisper.py):
    Encoder top-level
      enc.conv.0.weight / .bias          [d_model, n_mels, 3] / [d_model]
      enc.conv.1.weight / .bias          [d_model, d_model, 3] / [d_model]
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
      dec.blocks.{i}.self_attn.{q,k,v,out}.*   (k has no bias)
      dec.blocks.{i}.norm_cross.weight/bias
      dec.blocks.{i}.cross_attn.{q,k,v,out}.*  (k has no bias)
      dec.blocks.{i}.norm_ffn.weight/bias
      dec.blocks.{i}.ffn.{fc1,fc2}.weight/bias
    Frontend
      frontend.mel_filterbank  [n_mels, n_fft/2+1]  (rebuilt, slaney)
      frontend.window          [win_length=400]     (hann_periodic)

CLI:
    uv run --project scripts/envs/crisperwhisper \
      scripts/convert-crisperwhisper.py nyralabs/CrisperWhisper2.0_small \
      --revision 57750c47fde52dc1b016ec2bd4bf4704944cf3df

Single-file, top-to-bottom — no hidden helpers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
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


# ---------------------------------------------------------------------------
# Variant identity
# ---------------------------------------------------------------------------
# The upstream repo slugs (CrisperWhisper2.0_small) are not usable as
# transcribe.cpp variant keys, so map them explicitly. The variant key is what
# names the models/ directory, the intake, the golden manifest and every WER
# report, so it must not be derived by string munging.

VARIANTS: dict[str, dict[str, str]] = {
    "CrisperWhisper2.0_small":  {"variant": "crisperwhisper-2.0-small",
                                 "display": "CrisperWhisper 2.0 Small"},
    "CrisperWhisper2.0_medium": {"variant": "crisperwhisper-2.0-medium",
                                 "display": "CrisperWhisper 2.0 Medium"},
    "CrisperWhisper2.0_large":  {"variant": "crisperwhisper-2.0-large",
                                 "display": "CrisperWhisper 2.0 Large"},
    "CrisperWhisper2.0_turbo":  {"variant": "crisperwhisper-2.0-turbo",
                                 "display": "CrisperWhisper 2.0 Turbo"},
}


# ---------------------------------------------------------------------------
# Reference dtype
# ---------------------------------------------------------------------------
# All four open CrisperWhisper 2.0 checkpoints are uniformly BF16 — there are
# no F32 norms or biases in the safetensors header. Detect rather than assume,
# so a future F32 re-upload converts correctly instead of silently mislabelling.


_FORCED_DTYPES = {
    "f32":  ("F32",  LlamaFileType.ALL_F32,     GGMLQuantizationType.F32),
    "f16":  ("F16",  LlamaFileType.MOSTLY_F16,  GGMLQuantizationType.F16),
    "bf16": ("BF16", LlamaFileType.MOSTLY_BF16, GGMLQuantizationType.BF16),
}


def detect_reference_dtype(
    safetensors_path: Path,
    force: str | None = None,
) -> tuple[str, LlamaFileType, GGMLQuantizationType]:
    if force is not None:
        return _FORCED_DTYPES[force]
    with safe_open(str(safetensors_path), framework="pt") as st:
        dtypes = {str(st.get_slice(k).get_dtype()) for k in st.keys()}
    if dtypes == {"BF16"}:
        return ("BF16", LlamaFileType.MOSTLY_BF16, GGMLQuantizationType.BF16)
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
# Tokenizer extraction (GPT-2 byte-level BPE + whisper's + nyra's added tokens)
# ---------------------------------------------------------------------------
#
# The base BPE vocab is byte-identical to openai/whisper-small's; only
# `added_tokens` differ. 1639 added tokens on small/medium/large: 99 language
# tokens, 2 task tokens, ~5 auxiliary control tokens, 1501 timestamp tokens
# (dead — the model is always decoded with <|notimestamps|>), and nyra's 31.
#
# All 31 nyra tokens carry `special: false`, so `skip_special_tokens` does NOT
# strip them. That is deliberate for the 15 vocal-event tokens, which are
# intended verbatim-mode output; the 16 mode tags and prompt markers must be
# stripped by explicit id (see stt.crisperwhisper.prompt_artifact_token_ids).

EVENT_TOKEN_CONTENTS = (
    "[UM]", "[UH]", "[laughter]", "[sniff]", "[throatclearing]", "[cough]",
    "[sigh]", "[breath]", "[lipsmack]", "[yawn]", "[noise]", "[crying]",
    "[fart]", "[scream]", "[sneeze]",
)

# crisperwhisper/prompt.py::PROMPT_MARKER_TOKENS, in that file's order.
MARKER_TOKENS = (
    ("verbatimize_start", "<vtx>"),
    ("verbatimize_end",   "<evtx>"),
    ("hotword_start",     "<htx>"),
    ("hotword_end",       "<ehtx>"),
    ("context_start",     "<ctx>"),
    ("context_end",       "<ectx>"),
)

_MODE_TAG_RE = re.compile(r"^\[(verbatim|intended)_(\d+)\]$")


def extract_tokenizer(model_dir: Path, vocab_size: int) -> dict:
    with (model_dir / "tokenizer.json").open(encoding="utf-8") as f:
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
        tok_by_id[int(entry["id"])] = (entry["content"],
                                       bool(entry.get("special", False)))

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

    content_to_id = {tok: int(tid) for tok, tid in base_vocab.items()}
    for entry in added_tokens:
        content_to_id[entry["content"]] = int(entry["id"])

    def tok_id(content: str) -> int | None:
        return content_to_id.get(content)

    def require(content: str) -> int:
        tid = content_to_id.get(content)
        if tid is None:
            raise ValueError(
                f"tokenizer.json is missing the required CrisperWhisper token "
                f"{content!r}; this checkpoint does not carry the 2.0 decode "
                f"contract"
            )
        return tid

    # Mode tags: discovered, not hardcoded. verbatim_tag_count /
    # intended_tag_count are constructor arguments upstream (both default 5),
    # so read whatever this checkpoint actually ships and sort by index.
    mode_tags: dict[str, list[int]] = {"verbatim": [], "intended": []}
    for content, tid in content_to_id.items():
        m = _MODE_TAG_RE.match(content)
        if m:
            mode_tags[m.group(1)].append((int(m.group(2)), tid))
    for mode in ("verbatim", "intended"):
        if not mode_tags[mode]:
            raise ValueError(
                f"tokenizer.json has no [{mode}_N] mode tags; this checkpoint "
                f"does not carry the CrisperWhisper 2.0 mode contract"
            )
        mode_tags[mode] = [tid for _, tid in sorted(mode_tags[mode])]

    return {
        "tokens":            tokens,
        "types":             types,
        "merges":            merges,
        "bos_id":            require("<|endoftext|>"),
        "eos_id":            require("<|endoftext|>"),
        "pad_id":            require("<|endoftext|>"),
        "sot_id":            require("<|startoftranscript|>"),
        "transcribe_id":     require("<|transcribe|>"),
        "translate_id":      tok_id("<|translate|>"),
        "no_timestamps_id":  require("<|notimestamps|>"),
        "prev_sot_id":       tok_id("<|startofprev|>"),
        "verbatim_tag_ids":  mode_tags["verbatim"],
        "intended_tag_ids":  mode_tags["intended"],
        "event_ids":         [require(c) for c in EVENT_TOKEN_CONTENTS],
        "marker_ids":        {role: require(c) for role, c in MARKER_TOKENS},
    }


# ---------------------------------------------------------------------------
# Hparams from config.json + generation_config.json + preprocessor_config.json
# ---------------------------------------------------------------------------


def read_hparams(config: dict, gen_config: dict, preproc: dict) -> dict:
    if config.get("model_type") != "whisper":
        raise ValueError(
            f"expected config.model_type=whisper, got {config.get('model_type')!r}"
        )

    d_model              = int(config["d_model"])
    max_source_positions = int(config["max_source_positions"])
    max_target_positions = int(config["max_target_positions"])
    vocab_size           = int(config["vocab_size"])

    decoder_start_id = int(gen_config["decoder_start_token_id"])
    no_ts_id         = int(gen_config["no_timestamps_token_id"])
    suppress_tokens  = [int(x) for x in gen_config.get("suppress_tokens", []) or []]
    begin_suppress   = [int(x) for x in gen_config.get("begin_suppress_tokens", []) or []]

    # Word timing: 10 (layer, head) pairs selecting the cross-attention rows the
    # Viterbi aligner consumes. Different per variant — turbo has 4 decoder
    # layers, so its pairs cannot be reused from the others. Flattened to a
    # uint32 array of 2N entries in the GGUF.
    alignment_heads = gen_config.get("alignment_heads") or []
    if not alignment_heads:
        raise ValueError(
            "generation_config.json has no alignment_heads; word timestamps "
            "are a MUST PASS capability for this family and have no fallback"
        )
    dec_layers = int(config["decoder_layers"])
    dec_heads  = int(config["decoder_attention_heads"])
    for layer, head in alignment_heads:
        if not (0 <= int(layer) < dec_layers and 0 <= int(head) < dec_heads):
            raise ValueError(
                f"alignment_heads entry ({layer}, {head}) is out of range for "
                f"{dec_layers} decoder layers x {dec_heads} heads"
            )

    # Languages ordered by language-token id (en, zh, de, ... — Whisper's
    # canonical frequency order), not by lang_to_id dict order (alphabetical).
    # The token-id order is what the intake and the golden manifest declare.
    lang_to_id = gen_config.get("lang_to_id") or {}
    languages = [tok[2:-2] for tok, _ in sorted(lang_to_id.items(),
                                                key=lambda kv: kv[1])] or ["en"]

    sample_rate  = int(preproc.get("sampling_rate", 16000))
    n_fft        = int(preproc["n_fft"])
    hop_length   = int(preproc["hop_length"])
    feature_size = int(preproc["feature_size"])
    chunk_length = int(preproc.get("chunk_length", 30))
    n_samples    = int(preproc.get("n_samples", chunk_length * sample_rate))
    nb_max_frm   = int(preproc.get("nb_max_frames", n_samples // hop_length))
    if feature_size != int(config["num_mel_bins"]):
        raise ValueError(
            f"preprocessor feature_size={feature_size} != "
            f"config.num_mel_bins={config['num_mel_bins']}"
        )

    return {
        "d_model":              d_model,
        "enc_n_layers":         int(config["encoder_layers"]),
        "enc_n_heads":          int(config["encoder_attention_heads"]),
        "enc_ffn_dim":          int(config["encoder_ffn_dim"]),
        "dec_n_layers":         dec_layers,
        "dec_n_heads":          dec_heads,
        "dec_ffn_dim":          int(config["decoder_ffn_dim"]),
        "num_mel_bins":         int(config["num_mel_bins"]),
        "max_source_positions": max_source_positions,
        "max_target_positions": max_target_positions,
        "vocab_size":           vocab_size,
        "activation":           str(config["activation_function"]).lower(),
        "scale_embedding":      bool(config.get("scale_embedding", False)),

        "decoder_start_token_id": decoder_start_id,
        "no_timestamps_token_id": no_ts_id,
        "suppress_tokens":        suppress_tokens,
        "begin_suppress_tokens":  begin_suppress,
        "alignment_heads":        [[int(l), int(h)] for l, h in alignment_heads],

        "fe_type":        "mel",
        "fe_sample_rate": sample_rate,
        "fe_num_mels":    feature_size,
        "fe_n_fft":       n_fft,
        "fe_win_length":  n_fft,            # WhisperFeatureExtractor: win=n_fft
        "fe_hop_length":  hop_length,
        "fe_window":      "hann_periodic",
        "fe_normalize":   "whisper_logmel", # log10 + dynamic-range clamp + scale
        "fe_dither":      float(preproc.get("dither", 0.0)),
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
# Long-form defaults (crisperwhisper/longform/base.py::LongformConfig)
# ---------------------------------------------------------------------------
# These live in the package, not in the checkpoint, but they are a property of
# how the model was TRAINED (the 26 s stride / 4 s overlap geometry and the
# 12-word context window are what the <ctx> prompt saw during training), so the
# GGUF carries them rather than leaving Stage 4 to hardcode magic numbers.
# Pinned against crisperwhisper==2.0.2.

LONGFORM_CHUNK_DURATION = 30.0   # seconds; one Whisper encoder window
LONGFORM_STRIDE         = 26.0   # seconds; 4 s overlap
LONGFORM_CONTEXT_WORDS  = 12     # trailing confirmed words fed via <ctx>
LONGFORM_DROP_WORDS     = 2      # cap on overlap-aware trailing-word drop
DEFAULT_MODE            = "verbatim"  # model.py::transcribe(mode="verbatim")


# ---------------------------------------------------------------------------
# Tensor name mapping (identical to convert-whisper.py)
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


# Deliberately not emitted — see the module docstring for the evidence.
DROPPED_TENSORS: dict[str, str] = {
    "encoder_blank_head.weight":
        "untrained training scaffold (std==init_std, unreferenced by the "
        "inference package)",
    "encoder_blank_head.bias":
        "untrained training scaffold (still exactly 0.0)",
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


def convert(model_dir: Path, out_path: Path, variant: str, display_name: str,
            repo_id: str | None = None, force_dtype: str | None = None) -> None:
    config_path      = model_dir / "config.json"
    gen_config_path  = model_dir / "generation_config.json"
    preproc_path     = model_dir / "preprocessor_config.json"
    safetensors_path = model_dir / "model.safetensors"

    for p in (config_path, gen_config_path, preproc_path, safetensors_path):
        if not p.is_file():
            raise FileNotFoundError(f"missing required file: {p}")

    REFERENCE_DTYPE_LABEL, REFERENCE_FILE_TYPE, REFERENCE_GGML_TYPE = \
        detect_reference_dtype(safetensors_path, force_dtype)
    if force_dtype is not None:
        print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (FORCED — diagnostic build, "
              f"not a shippable artifact)")
    else:
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
    print(f"Languages: {len(hp['languages'])}; "
          f"alignment_heads: {len(hp['alignment_heads'])}")
    print(f"Variant: {variant}")

    print(f"Reading tokenizer from {model_dir}")
    tok = extract_tokenizer(model_dir, hp["vocab_size"])
    print(f"Mode tags: {len(tok['verbatim_tag_ids'])} verbatim "
          f"{tok['verbatim_tag_ids'][0]}..{tok['verbatim_tag_ids'][-1]}, "
          f"{len(tok['intended_tag_ids'])} intended "
          f"{tok['intended_tag_ids'][0]}..{tok['intended_tag_ids'][-1]}; "
          f"{len(tok['event_ids'])} event tokens; "
          f"{len(tok['marker_ids'])} prompt markers")

    print(f"Opening safetensors at {safetensors_path}")
    with safe_open(str(safetensors_path), framework="pt") as st:
        st_keys = set(st.keys())

        missing_drops = set(DROPPED_TENSORS) - st_keys
        if missing_drops:
            print(f"note: expected-but-absent dropped tensors: "
                  f"{sorted(missing_drops)}", file=sys.stderr)

        # Size label counts the params that ship in the GGUF, so the dropped
        # training scaffold does not inflate it.
        total = sum(st.get_tensor(k).numel()
                    for k in st_keys - set(DROPPED_TENSORS))
        size_label = compute_size_label(total)
        print(f"Total params: {total:,} -> size_label={size_label}")

        print(f"Writing GGUF to {out_path}")
        writer = gguf_writer(str(out_path), "crisperwhisper")

        # ---- general.* ----
        # Dual license: MIT for the inference code (Part A), nyra health
        # Non-Commercial Research License v1.0 for the weights AND their
        # outputs (Part B). A GGUF is a derivative of the weights, so the
        # non-commercial terms are the ones that bind this file.
        add_general_identity(
            writer,
            name=display_name,
            basename="crisperwhisper",
            version="2.0",
            size_label=size_label,
            file_type=REFERENCE_FILE_TYPE,
            languages=hp["languages"],
            author="nyra health GmbH",
            organization="nyralabs",
            license="other",
            license_name="nyra health Non-Commercial Research License v1.0",
            license_link=(f"https://huggingface.co/{repo_id}/blob/main/LICENSE.md"
                          if repo_id else None),
            repo_url=(f"https://huggingface.co/{repo_id}" if repo_id else None),
            source_url="https://github.com/nyrahealth/CrisperWhisper",
            description=(
                "CrisperWhisper 2.0 — verbatim ASR with disfluency retention and "
                "word-level timestamps. NON-COMMERCIAL USE ONLY: the model "
                "weights and any transcripts they generate are licensed under "
                "the nyra health Non-Commercial Research License v1.0."
            ),
            tags=["speech-recognition", "verbatim", "disfluency",
                  "word-timestamps", "non-commercial"],
        )

        # ---- stt.variant ----
        writer.add_string("stt.variant", variant)

        # ---- stt.capability.* ----
        # Language detection is inherited from Whisper (all 99 <|lang|> tokens
        # plus forced_decoder_ids [[1, null]]). Translation is NOT advertised:
        # <|translate|> survives in the vocab but the author package exposes no
        # translate path and neither the card nor DOCS claims it.
        # Timestamps are word-only — the model is always decoded with
        # <|notimestamps|> and never emits timestamp tokens.
        writer.add_bool("stt.capability.lang_detect",         True)
        writer.add_bool("stt.capability.translate",           False)
        writer.add_bool("stt.capability.timestamps",          True)
        writer.add_bool("stt.capability.word_timestamps",     True)
        writer.add_bool("stt.capability.segment_timestamps",  False)
        writer.add_bool("stt.capability.streaming",           False)
        writer.add_bool("stt.capability.speaker_diarization", False)

        # ---- tokenizer.ggml.* (llama.cpp "gpt2" byte-level BPE) ----
        writer.add_string("tokenizer.ggml.model", "gpt2")
        writer.add_string("tokenizer.ggml.pre",   "gpt2")
        writer.add_array("tokenizer.ggml.tokens",     tok["tokens"])
        writer.add_array("tokenizer.ggml.token_type", tok["types"])
        writer.add_array("tokenizer.ggml.merges",     tok["merges"])
        writer.add_uint32("tokenizer.ggml.bos_token_id",     tok["bos_id"])
        writer.add_uint32("tokenizer.ggml.eos_token_id",     tok["eos_id"])
        writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])
        writer.add_bool("tokenizer.ggml.add_bos_token", False)

        # ---- stt.crisperwhisper.encoder.* ----
        writer.add_uint32("stt.crisperwhisper.encoder.n_layers",     hp["enc_n_layers"])
        writer.add_uint32("stt.crisperwhisper.encoder.d_model",      hp["d_model"])
        writer.add_uint32("stt.crisperwhisper.encoder.n_heads",      hp["enc_n_heads"])
        writer.add_uint32("stt.crisperwhisper.encoder.ffn_dim",      hp["enc_ffn_dim"])
        writer.add_uint32("stt.crisperwhisper.encoder.num_mel_bins", hp["num_mel_bins"])
        writer.add_uint32("stt.crisperwhisper.encoder.max_source_positions",
                          hp["max_source_positions"])
        writer.add_string("stt.crisperwhisper.encoder.activation",   hp["activation"])

        # ---- stt.crisperwhisper.decoder.* ----
        writer.add_uint32("stt.crisperwhisper.decoder.n_layers",  hp["dec_n_layers"])
        writer.add_uint32("stt.crisperwhisper.decoder.d_model",   hp["d_model"])
        writer.add_uint32("stt.crisperwhisper.decoder.n_heads",   hp["dec_n_heads"])
        writer.add_uint32("stt.crisperwhisper.decoder.ffn_dim",   hp["dec_ffn_dim"])
        writer.add_uint32("stt.crisperwhisper.decoder.max_target_positions",
                          hp["max_target_positions"])
        writer.add_uint32("stt.crisperwhisper.decoder.vocab_size", hp["vocab_size"])
        writer.add_string("stt.crisperwhisper.decoder.activation", hp["activation"])
        writer.add_bool("stt.crisperwhisper.decoder.tie_word_embeddings", True)
        writer.add_bool("stt.crisperwhisper.decoder.scale_embedding",
                        hp["scale_embedding"])

        # ---- Whisper-inherited prompt prefix / suppression ----
        writer.add_uint32("stt.crisperwhisper.decoder_start_token_id",
                          hp["decoder_start_token_id"])
        writer.add_uint32("stt.crisperwhisper.no_timestamps_token_id",
                          hp["no_timestamps_token_id"])
        writer.add_uint32("stt.crisperwhisper.sot_token_id",        tok["sot_id"])
        writer.add_uint32("stt.crisperwhisper.transcribe_token_id", tok["transcribe_id"])
        if tok["translate_id"] is not None:
            writer.add_uint32("stt.crisperwhisper.translate_token_id",
                              tok["translate_id"])
        if tok["prev_sot_id"] is not None:
            writer.add_uint32("stt.crisperwhisper.prev_sot_token_id",
                              tok["prev_sot_id"])
        if hp["suppress_tokens"]:
            writer.add_array("stt.crisperwhisper.suppress_tokens",
                             hp["suppress_tokens"])
        if hp["begin_suppress_tokens"]:
            writer.add_array("stt.crisperwhisper.begin_suppress_tokens",
                             hp["begin_suppress_tokens"])

        # ---- CrisperWhisper decode contract ----
        # The decoder input is, literally:
        #   encode("[verbatim_1]...[verbatim_5]"
        #          [+ " <ctx> " + last N confirmed words + " <ectx>"]
        #          [+ " <htx> " + hotwords + " <ehtx>"])            # Pro only
        #   + [<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]
        # Mode tags come BEFORE the Whisper prefix with no <|startofprev|>
        # wrapper, and context comes BEFORE hotwords (training order — emitting
        # them the other way round silently changes the output).
        writer.add_bool("stt.crisperwhisper.mode_tags_before_prefix", True)
        writer.add_bool("stt.crisperwhisper.always_no_timestamps",    True)
        writer.add_string("stt.crisperwhisper.mode.default",          DEFAULT_MODE)
        writer.add_array("stt.crisperwhisper.mode.verbatim_token_ids",
                         tok["verbatim_tag_ids"])
        writer.add_array("stt.crisperwhisper.mode.intended_token_ids",
                         tok["intended_tag_ids"])
        for role, tid in tok["marker_ids"].items():
            writer.add_uint32(f"stt.crisperwhisper.marker.{role}_token_id", tid)

        # 15 vocal-event tokens. `special: false` upstream, and that is
        # deliberate: they are intended verbatim-mode OUTPUT and must survive
        # decoding. Carried so the runtime can distinguish them from the
        # prompt artifacts below without a hardcoded id range.
        writer.add_array("stt.crisperwhisper.event_token_ids", tok["event_ids"])

        # Mode tags + prompt markers: echoed into the output by the model and
        # stripped by explicit id upstream
        # (crisperwhisper/prompt.py::strip_prompt_artifacts), NOT by the
        # tokenizer's special-token machinery.
        writer.add_array(
            "stt.crisperwhisper.prompt_artifact_token_ids",
            tok["verbatim_tag_ids"] + tok["intended_tag_ids"]
            + [tok["marker_ids"][role] for role, _ in MARKER_TOKENS],
        )

        # ---- word timing (Viterbi over supervised cross-attention) ----
        writer.add_array(
            "stt.crisperwhisper.word_timing.alignment_heads",
            [v for pair in hp["alignment_heads"] for v in pair],
        )

        # ---- long-form (<ctx> conditional continuation) ----
        writer.add_float32("stt.crisperwhisper.longform.chunk_duration",
                           LONGFORM_CHUNK_DURATION)
        writer.add_float32("stt.crisperwhisper.longform.stride",
                           LONGFORM_STRIDE)
        writer.add_uint32("stt.crisperwhisper.longform.context_words",
                          LONGFORM_CONTEXT_WORDS)
        writer.add_uint32("stt.crisperwhisper.longform.drop_words",
                          LONGFORM_DROP_WORDS)

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
        # CrisperWhisper's preprocessor_config.json carries no `mel_filters`
        # array (unlike whisper tiny..large-v2), so WhisperFeatureExtractor
        # builds the filterbank at load time. Reproduce that exact call here so
        # the GGUF carries the same filters HF would build.
        if "mel_filters" in preproc:
            mel_fb = np.asarray(preproc["mel_filters"], dtype=np.float32)
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
            print("computed mel filterbank via transformers.audio_utils "
                  "(no mel_filters in preprocessor_config.json)")
        if mel_fb.shape != (hp["fe_num_mels"], hp["fe_n_fft"] // 2 + 1):
            raise ValueError(
                f"mel_filters shape {mel_fb.shape} does not match "
                f"[num_mels={hp['fe_num_mels']}, n_fft/2+1="
                f"{hp['fe_n_fft'] // 2 + 1}]"
            )
        writer.add_tensor("frontend.mel_filterbank",
                          np.ascontiguousarray(mel_fb),
                          raw_dtype=GGMLQuantizationType.F32)
        # WhisperFeatureExtractor uses a periodic Hann window of length n_fft
        # (denominator = N, not N-1), i.e. torch.hann_window(N, periodic=True).
        N = int(hp["fe_win_length"])
        hann = (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(N) / N)).astype(np.float32)
        writer.add_tensor("frontend.window",
                          np.ascontiguousarray(hann),
                          raw_dtype=GGMLQuantizationType.F32)
        n_added += 2
        bytes_in  += mel_fb.nbytes + hann.nbytes
        bytes_out += mel_fb.nbytes + hann.nbytes

        def add(src_name: str, dst_name: str, transform=passthrough) -> None:
            nonlocal n_added, bytes_in, bytes_out
            if src_name not in st_keys:
                raise KeyError(f"safetensors missing tensor: {src_name!r}")
            t = st.get_tensor(src_name)
            if t.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(
                    f"{src_name}: expected float32, float16, or bfloat16, "
                    f"got {t.dtype}"
                )
            # encode_for_gguf wants fp32 input. BF16/F16 -> F32 is exact, so we
            # upcast on read and let encode_for_gguf re-pack to the per-tensor
            # target dtype (which respects REFERENCE_GGML_TYPE).
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

        for src, dst in ENCODER_TOP_TABLE:
            add(src, dst)
        for i in range(hp["enc_n_layers"]):
            for suffix_src, suffix_dst in ENCODER_BLOCK_TABLE:
                add(f"model.encoder.layers.{i}.{suffix_src}",
                    f"enc.blocks.{i}.{suffix_dst}")

        for src, dst in DECODER_TOP_TABLE:
            add(src, dst)
        for i in range(hp["dec_n_layers"]):
            for suffix_src, suffix_dst in DECODER_BLOCK_TABLE:
                add(f"model.decoder.layers.{i}.{suffix_src}",
                    f"dec.blocks.{i}.{suffix_dst}")

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
        print(f"Added {n_added} tensors "
              f"({bytes_in / (1024 * 1024):.1f} MB fp32 -> "
              f"{bytes_out / (1024 * 1024):.1f} MB on disk)")
        for name, why in DROPPED_TENSORS.items():
            if name in st_keys:
                print(f"Dropped {name}: {why}")

        # Every safetensors key must be either consumed or explicitly dropped.
        # An unexpected leftover is a checkpoint change we have not reviewed,
        # so fail rather than print a warning nobody reads.
        consumed: set[str] = set(DROPPED_TENSORS)
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
            raise RuntimeError(
                f"{len(unused)} safetensors keys neither converted nor listed "
                f"in DROPPED_TENSORS: {unused[:20]}"
                + (f" ... and {len(unused) - 20} more" if len(unused) > 20 else "")
            )

        print("Writing header + KV + tensor info...")
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        print("Writing tensor data...")
        writer.write_tensors_to_file()
        writer.close()

    print(f"Done. Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a CrisperWhisper 2.0 checkpoint to a "
                    "reference-dtype GGUF.",
    )
    p.add_argument("model", type=str,
                   help="HF repo id (e.g. nyralabs/CrisperWhisper2.0_small) "
                        "or local dir")
    p.add_argument("out_path", type=Path, nargs="?",
                   help="Output .gguf path (derived from --repo-id when omitted)")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF repo id used to derive the variant when converting "
                        "from a local path")
    p.add_argument("--revision", type=str, default=None,
                   help="HF revision (branch / tag / commit SHA) to pin the "
                        "download to. Recommended for reproducibility; the "
                        "intake records the canonical pinned revision.")
    p.add_argument("--variant", type=str, default=None,
                   help="stt.variant string (default: derived from the repo slug)")
    p.add_argument("--force-reference-dtype", type=str, default=None,
                   choices=["f32", "f16", "bf16"],
                   help="DIAGNOSTIC ONLY. Override the detected reference dtype. "
                        "The shipped GGUF must always use the detected dtype "
                        "(BF16 for every CrisperWhisper 2.0 checkpoint); this "
                        "flag exists so numerical validation can attribute drift "
                        "to storage precision by rebuilding the same weights at "
                        "F32. Never use it for a published artifact.")
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

    if not repo_id and not args.variant:
        print("error: pass an HF repo id, --repo-id, or --variant so the "
              "variant key can be resolved", file=sys.stderr)
        return 2

    upstream_slug = slug_from_repo_id(repo_id) if repo_id else None
    if upstream_slug and upstream_slug not in VARIANTS:
        print(f"error: unknown CrisperWhisper repo slug {upstream_slug!r}; "
              f"add it to VARIANTS (known: {sorted(VARIANTS)})", file=sys.stderr)
        return 2

    variant = args.variant or VARIANTS[upstream_slug]["variant"]
    display_name = (VARIANTS[upstream_slug]["display"] if upstream_slug
                    else variant)

    out_path = args.out_path
    if out_path is None:
        ref_label, _, _ = detect_reference_dtype(model_dir / "model.safetensors",
                                                 args.force_reference_dtype)
        out_path = REPO_ROOT / "models" / variant / gguf_name(variant, ref_label)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    convert(model_dir, out_path, variant, display_name, repo_id=repo_id,
            force_dtype=args.force_reference_dtype)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
