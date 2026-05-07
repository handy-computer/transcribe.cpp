#!/usr/bin/env python3
"""
convert-canary.py - convert a NeMo Canary multitask AED model into a
GGUF that transcribe.cpp's loader can ingest end-to-end. Loads via
nemo.collections.asr.models.EncDecMultiTaskModel.

    uv run --project scripts/envs/canary \
      scripts/convert-canary.py nvidia/canary-180m-flash

Source format:
    HF model id (e.g. "nvidia/canary-1b-flash") that NeMo's
    EncDecMultiTaskModel.from_pretrained() can resolve, or a local
    .nemo path / extracted directory.

Target format:
    Single .gguf at models/<slug>/<slug>-F32.gguf. Reference dtype is
    fp32 for every published canary checkpoint we know about; the
    converter rejects sources with a different state-dict dtype rather
    than silently downcasting.

Architecture summary (encoder-decoder):
    preprocessor   AudioToMelSpectrogramPreprocessor (runtime, not exported)
    encoder        FastConformer (same module as parakeet)
    encoder_decoder_proj  Linear from encoder d_model to decoder hidden
                          (Identity / absent when dims already match)
    transf_encoder optional Transformer encoder on top of FastConformer.
                   All four shipping variants set num_layers=0 so the
                   submodule has no parameters; we assert that here.
    transf_decoder Transformer decoder, pre-LN, sinusoidal pos enc.
                   _embedding.{token_embedding, position_embedding, layer_norm}
                   _decoder.layers[i].{layer_norm_1, first_sub_layer (self),
                                       layer_norm_2, second_sub_layer (cross),
                                       layer_norm_3, third_sub_layer (FFN)}
                   _decoder.final_layer_norm
    log_softmax    TokenClassifier with a single Linear -> vocab.

Variant differences (captured in VARIANT_PROFILES):
    canary-1b           prompt=canary  (4-slot), 24×24 layers, vocab 4128,
                        spl_tokens=32, no language tokens, CC-BY-NC-4.0
    canary-1b-v2        prompt=canary2 (5-slot), 32× / 4×, vocab 16384,
                        25 European languages, separate timestamps aligner
    canary-1b-flash     prompt=canary2, 32× / 4×, vocab 5248, 4 languages
    canary-180m-flash   prompt=canary2, 17× / 4×, vocab 5248, 4 languages

Tokenizer:
    NeMo's CanaryTokenizer is an aggregate over per-language SentencePiece
    sub-tokenizers plus a shared spl_tokens SP. We emit a flat
    `tokenizer.ggml.tokens` array of length vocab_size with the resolved
    piece per id, plus per-sub-tokenizer offsets/codes/sizes under
    stt.canary.tokenizer.* so a future C++ encoder can route text-side
    encoding to the right sub-vocab. SP scores are emitted when the
    underlying SP processor exposes them; CONTROL is used for every entry
    in CanaryTokenizer.special_tokens (and id 0 = <unk>).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from gguf import GGUFWriter, LlamaFileType

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.gguf_common import (  # noqa: E402
    TOKEN_TYPE_CONTROL,
    TOKEN_TYPE_NORMAL,
    TOKEN_TYPE_UNKNOWN,
    gguf_name,
    slug_from_repo_id,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_DTYPE_LABEL = "F32"
REFERENCE_FILE_TYPE = LlamaFileType.ALL_F32


# ---------------------------------------------------------------------------
# Variant profiles
# ---------------------------------------------------------------------------
#
# The HF slug uniquely identifies a variant. Profiles carry the bits the
# converter cannot derive from cfg/state_dict alone (display version
# label, license-derived size_label, the prompt format used by the
# multitask interface).

VARIANT_PROFILES: dict[str, dict[str, Any]] = {
    "canary-1b": {
        "version":        "v1",
        "size_label":     "1B",
        "prompt_format":  "canary",
    },
    "canary-1b-v2": {
        "version":        "v2",
        "size_label":     "1B",
        "prompt_format":  "canary2",
    },
    "canary-1b-flash": {
        "version":        "1b-flash",
        "size_label":     "1B",
        "prompt_format":  "canary2",
    },
    "canary-180m-flash": {
        "version":        "180m-flash",
        "size_label":     "180M",
        "prompt_format":  "canary2",
    },
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_canary_model(model_spec: str):
    """Load a Canary multitask model via NeMo. Accepts an HF id or
    local .nemo path / extracted directory."""
    from nemo.collections.asr.models import EncDecMultiTaskModel

    local = Path(model_spec).expanduser()
    if local.exists():
        print(f"Loading Canary from local path: {local}")
        model = EncDecMultiTaskModel.restore_from(str(local), map_location="cpu")
    else:
        print(f"Loading Canary from HuggingFace: {model_spec}")
        model = EncDecMultiTaskModel.from_pretrained(model_spec, map_location="cpu")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Tokenizer extraction
# ---------------------------------------------------------------------------
#
# CanaryTokenizer aggregates spl_tokens + per-language SP. We emit a
# flat token list of length vocab_size; for each id we ask the
# CanaryTokenizer for the piece, route to the underlying SP for the
# score when the id is not a special token, and tag types accordingly.


def _extract_aggregate(tk) -> dict:
    """CanaryTokenizer (canary-1b, canary-1b-flash, canary-180m-flash):
    aggregate over spl_tokens + per-language SP. Routing info is real
    (lang_codes / lang_offsets / lang_sizes)."""
    vocab_size = int(tk.vocab_size)
    special_ids = set(tk.special_tokens.values()) | {0}  # id 0 is <unk>

    tokens, scores, types = [], [], []
    for i in range(vocab_size):
        try:
            piece = tk.ids_to_tokens([i])[0]
        except Exception:
            piece = f"<unused_{i}>"
        if i == 0:
            ttype = TOKEN_TYPE_UNKNOWN
        elif i in special_ids:
            ttype = TOKEN_TYPE_CONTROL
        else:
            ttype = TOKEN_TYPE_NORMAL
        tokens.append(piece)
        scores.append(0.0)
        types.append(ttype)

    offsets, codes, sizes = [], [], []
    for name, sp in tk.tokenizers_dict.items():
        offsets.append(int(tk.token_id_offset[name]))
        codes.append(str(name))
        sizes.append(int(sp.vocab_size))

    return {
        "tokens": tokens, "scores": scores, "types": types,
        "offsets": offsets, "codes": codes, "sizes": sizes,
        "specials": dict(tk.special_tokens),
        "unk_id": 0,
        "bos_id": int(tk.bos_id) if tk.bos_id is not None else None,
        "eos_id": int(tk.eos_id) if tk.eos_id is not None else None,
        "pad_id": int(tk.pad_id) if tk.pad_id is not None else None,
    }


def _extract_bpe(tk) -> dict:
    """CanaryBPETokenizer (canary-1b-v2): a single SentencePiece model
    where every special token (BOS, EOS, language tags, task tags,
    timestamp tags) is baked into the SP vocabulary as a CONTROL piece.
    No per-language routing — all text-side encoding goes through the
    one SP. We surface a single sub-tokenizer entry ("all") so the KV
    schema stays uniform across variants."""
    sp = tk.tokenizer
    vocab_size = int(sp.GetPieceSize())
    if vocab_size != int(tk.vocab_size):
        raise RuntimeError(
            f"SP piece size {vocab_size} != tk.vocab_size {tk.vocab_size}"
        )

    tokens, scores, types = [], [], []
    specials: dict[str, int] = {}
    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        score = float(sp.GetScore(i))
        if sp.IsUnknown(i):
            ttype = TOKEN_TYPE_UNKNOWN
        elif sp.IsControl(i):
            ttype = TOKEN_TYPE_CONTROL
            specials[piece] = i
        else:
            ttype = TOKEN_TYPE_NORMAL
        tokens.append(piece)
        scores.append(score)
        types.append(ttype)

    return {
        "tokens": tokens, "scores": scores, "types": types,
        "offsets": [0], "codes": ["all"], "sizes": [vocab_size],
        "specials": specials,
        "unk_id": int(tk.unk_id) if getattr(tk, "unk_id", None) is not None else None,
        "bos_id": int(tk.bos_id) if tk.bos_id is not None else None,
        "eos_id": int(tk.eos_id) if tk.eos_id is not None else None,
        "pad_id": int(tk.pad_id) if tk.pad_id is not None else None,
    }


def extract_tokenizer(tk) -> dict:
    """Dispatch by tokenizer class. The two canary tokenizer flavors
    have different surfaces; both project onto the same flat
    `tokens / scores / types` plus optional `offsets / codes / sizes`
    routing arrays."""
    cls = type(tk).__name__
    if cls == "CanaryTokenizer":
        return _extract_aggregate(tk)
    if cls == "CanaryBPETokenizer":
        return _extract_bpe(tk)
    raise NotImplementedError(
        f"unsupported canary tokenizer class: {cls!r}; "
        f"converter handles CanaryTokenizer + CanaryBPETokenizer"
    )


# ---------------------------------------------------------------------------
# Hparams from model.cfg
# ---------------------------------------------------------------------------


def read_hparams(config: dict) -> dict:
    enc  = config["encoder"]
    pre  = config["preprocessor"]
    tdec = config["transf_decoder"]["config_dict"]
    head = config["head"]
    tenc = config.get("transf_encoder") or {}

    target = pre.get("_target_", "")
    if "MelSpectrogram" not in target:
        raise ValueError(
            f"unsupported preprocessor _target_: {target!r}; "
            f"converter only handles AudioToMelSpectrogramPreprocessor"
        )

    sample_rate   = int(pre["sample_rate"])
    window_size   = float(pre["window_size"])
    window_stride = float(pre["window_stride"])
    win_length    = int(round(window_size  * sample_rate))
    hop_length    = int(round(window_stride * sample_rate))

    # transf_encoder is configured but disabled across all four shipping
    # canary variants. If a future variant turns it on we need new
    # tensor-mapping code, so reject early instead of silently emitting
    # an incomplete GGUF.
    transf_enc_layers = int(tenc.get("num_layers") or 0)

    return {
        "enc_n_layers":             int(enc["n_layers"]),
        "enc_d_model":              int(enc["d_model"]),
        "enc_n_heads":              int(enc["n_heads"]),
        "enc_d_ff":                 int(enc["d_model"]) * int(enc["ff_expansion_factor"]),
        "enc_conv_kernel":          int(enc["conv_kernel_size"]),
        "enc_subsampling_factor":   int(enc["subsampling_factor"]),
        "enc_subsampling_channels": int(enc["subsampling_conv_channels"]),
        "enc_pos_emb_max_len":      int(enc["pos_emb_max_len"]),
        "enc_use_bias":             bool(enc.get("use_bias", False)),

        "dec_n_layers":             int(tdec["num_layers"]),
        "dec_d_model":              int(tdec["hidden_size"]),
        "dec_n_heads":              int(tdec["num_attention_heads"]),
        "dec_d_ff":                 int(tdec["inner_size"]),
        "dec_max_position":         int(tdec["max_sequence_length"]),
        "dec_activation":           str(tdec["hidden_act"]),
        "dec_pre_ln":               bool(tdec.get("pre_ln", True)),
        "dec_learn_pos":            bool(tdec.get("learn_positional_encodings", False)),
        "dec_pre_ln_final_layer_norm": bool(
            config["transf_decoder"].get("pre_ln_final_layer_norm", True)
        ),

        "head_num_classes": int(head["num_classes"]),
        "head_activation":  str(head.get("activation", "relu")),
        "head_log_softmax": bool(head.get("log_softmax", True)),

        "transf_enc_layers": transf_enc_layers,

        "fe_type":         "mel",
        "fe_num_mels":     int(pre["features"]),
        "fe_sample_rate":  sample_rate,
        "fe_n_fft":        int(pre["n_fft"]),
        "fe_win_length":   win_length,
        "fe_hop_length":   hop_length,
        "fe_window":       str(pre["window"]),
        "fe_normalize":    str(pre["normalize"]),
        "fe_dither":       float(pre["dither"]),
        "fe_log":          bool(pre.get("log", True)),
        "fe_pre_emphasis": 0.97,  # NeMo FilterbankFeatures default
        "fe_f_min":        0.0,
        "fe_f_max":        float(sample_rate) / 2.0,
    }


# ---------------------------------------------------------------------------
# Tensor name mapping
# ---------------------------------------------------------------------------
#
# Encoder block table mirrors src/arch/parakeet/weights.cpp — the
# FastConformer module class is shared verbatim between parakeet and
# canary so the loader's read_parakeet_block_*() helpers work as-is
# when Stage 4 wires up the canary arch.

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


PRE_ENCODE_TABLE: list[tuple[str, str]] = [
    ("encoder.pre_encode.conv.0.weight", "enc.pre_encode.conv.0.weight"),
    ("encoder.pre_encode.conv.0.bias",   "enc.pre_encode.conv.0.bias"),
    ("encoder.pre_encode.conv.2.weight", "enc.pre_encode.conv.2.weight"),
    ("encoder.pre_encode.conv.2.bias",   "enc.pre_encode.conv.2.bias"),
    ("encoder.pre_encode.conv.3.weight", "enc.pre_encode.conv.3.weight"),
    ("encoder.pre_encode.conv.3.bias",   "enc.pre_encode.conv.3.bias"),
    ("encoder.pre_encode.conv.5.weight", "enc.pre_encode.conv.5.weight"),
    ("encoder.pre_encode.conv.5.bias",   "enc.pre_encode.conv.5.bias"),
    ("encoder.pre_encode.conv.6.weight", "enc.pre_encode.conv.6.weight"),
    ("encoder.pre_encode.conv.6.bias",   "enc.pre_encode.conv.6.bias"),
    ("encoder.pre_encode.out.weight",    "enc.pre_encode.out.weight"),
    ("encoder.pre_encode.out.bias",      "enc.pre_encode.out.bias"),
]


# Decoder layer table — sub-layer naming mirrors NeMo's
# TransformerDecoderLayer (first/second/third) but the GGUF names are
# the conventional self_attn / cross_attn / ffn for Stage 4 readability.
DECODER_BLOCK_TABLE: list[tuple[str, str]] = [
    # pre-self-attn LN
    ("layer_norm_1.weight",                  "norm1.weight"),
    ("layer_norm_1.bias",                    "norm1.bias"),
    # self-attention
    ("first_sub_layer.query_net.weight",     "self_attn.q.weight"),
    ("first_sub_layer.query_net.bias",       "self_attn.q.bias"),
    ("first_sub_layer.key_net.weight",       "self_attn.k.weight"),
    ("first_sub_layer.key_net.bias",         "self_attn.k.bias"),
    ("first_sub_layer.value_net.weight",     "self_attn.v.weight"),
    ("first_sub_layer.value_net.bias",       "self_attn.v.bias"),
    ("first_sub_layer.out_projection.weight","self_attn.o.weight"),
    ("first_sub_layer.out_projection.bias",  "self_attn.o.bias"),
    # pre-cross-attn LN
    ("layer_norm_2.weight",                  "norm2.weight"),
    ("layer_norm_2.bias",                    "norm2.bias"),
    # cross-attention (second_sub_layer)
    ("second_sub_layer.query_net.weight",     "cross_attn.q.weight"),
    ("second_sub_layer.query_net.bias",       "cross_attn.q.bias"),
    ("second_sub_layer.key_net.weight",       "cross_attn.k.weight"),
    ("second_sub_layer.key_net.bias",         "cross_attn.k.bias"),
    ("second_sub_layer.value_net.weight",     "cross_attn.v.weight"),
    ("second_sub_layer.value_net.bias",       "cross_attn.v.bias"),
    ("second_sub_layer.out_projection.weight","cross_attn.o.weight"),
    ("second_sub_layer.out_projection.bias",  "cross_attn.o.bias"),
    # pre-FFN LN
    ("layer_norm_3.weight",                  "norm3.weight"),
    ("layer_norm_3.bias",                    "norm3.bias"),
    # feed-forward (third_sub_layer)
    ("third_sub_layer.dense_in.weight",      "ffn.up.weight"),
    ("third_sub_layer.dense_in.bias",        "ffn.up.bias"),
    ("third_sub_layer.dense_out.weight",     "ffn.down.weight"),
    ("third_sub_layer.dense_out.bias",       "ffn.down.bias"),
]


EXPECTED_UNUSED_PREFIXES = (
    "preprocessor.",            # mel filterbank + window — C++ recomputes
)
EXPECTED_UNUSED_SUFFIXES = (
    ".num_batches_tracked",     # BN bookkeeping (scalar int64)
)


def is_expected_unused(key: str) -> bool:
    return (
        key.startswith(EXPECTED_UNUSED_PREFIXES)
        or key.endswith(EXPECTED_UNUSED_SUFFIXES)
    )


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------


def tensor_to_fp32_numpy(t) -> np.ndarray:
    """Torch Tensor -> contiguous fp32 numpy. Source must be fp32; any
    drop in precision should surface here rather than silently
    collapse into the target dtype."""
    import torch
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(t).__name__}")
    if t.dtype != torch.float32:
        raise ValueError(f"expected fp32 tensor, got {t.dtype}")
    arr = t.detach().cpu().numpy()
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Variant resolution
# ---------------------------------------------------------------------------


def resolve_variant(model_spec: str, repo_id_arg: str | None) -> tuple[str, dict]:
    """Pick the variant name + profile from the HF repo id (or local
    path, if --repo-id is provided). Errors loudly when the slug is not
    one of the four supported canary variants."""
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
            f"unknown canary variant {slug!r}; known: {sorted(VARIANT_PROFILES)}"
        )
    return slug, VARIANT_PROFILES[slug]


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


def convert(model_spec: str, out_path: Path, variant: str, profile: dict, languages: list[str]) -> None:
    from omegaconf import OmegaConf

    print(f"Output dtype: {REFERENCE_DTYPE_LABEL} (source/reference dtype)")

    model = load_canary_model(model_spec)

    config = OmegaConf.to_container(model.cfg, resolve=True)
    hp = read_hparams(config)

    if hp["transf_enc_layers"] != 0:
        raise NotImplementedError(
            f"variant {variant} has transf_encoder.num_layers="
            f"{hp['transf_enc_layers']} — converter only supports the "
            f"published Canary variants where this submodule is empty"
        )
    if hp["head_num_classes"] != int(model.tokenizer.vocab_size):
        raise ValueError(
            f"head.num_classes ({hp['head_num_classes']}) != tokenizer "
            f"vocab_size ({model.tokenizer.vocab_size})"
        )

    tok = extract_tokenizer(model.tokenizer)
    if len(tok["tokens"]) != hp["head_num_classes"]:
        raise RuntimeError(
            f"tokenizer length {len(tok['tokens'])} != head vocab "
            f"{hp['head_num_classes']}"
        )

    sd = model.state_dict()
    sd_keys = set(sd.keys())

    has_proj = "encoder_decoder_proj.weight" in sd_keys
    if has_proj:
        proj_w = sd["encoder_decoder_proj.weight"]
        proj_in, proj_out = int(proj_w.shape[1]), int(proj_w.shape[0])
        if proj_in != hp["enc_d_model"] or proj_out != hp["dec_d_model"]:
            raise ValueError(
                f"encoder_decoder_proj shape mismatch: got "
                f"{tuple(proj_w.shape)}, expected ({hp['dec_d_model']}, "
                f"{hp['enc_d_model']})"
            )

    print(f"Variant: {variant}")
    print(f"Encoder: {hp['enc_n_layers']} layers, d_model={hp['enc_d_model']}, "
          f"d_ff={hp['enc_d_ff']}, n_heads={hp['enc_n_heads']}")
    print(f"Decoder: {hp['dec_n_layers']} layers, d_model={hp['dec_d_model']}, "
          f"d_ff={hp['dec_d_ff']}, n_heads={hp['dec_n_heads']}, "
          f"max_pos={hp['dec_max_position']}")
    print(f"Vocab: {hp['head_num_classes']} ({len(tok['codes'])} sub-tokenizers)")
    print(f"encoder_decoder_proj: {'present' if has_proj else 'absent (dims match)'}")

    print(f"Writing GGUF to {out_path}")
    writer = GGUFWriter(str(out_path), "canary")

    # ----- general.* -----
    writer.add_string("general.basename",   "canary")
    writer.add_string("general.size_label", profile["size_label"])
    writer.add_string("general.version",    profile["version"])
    writer.add_uint32("general.file_type",  int(REFERENCE_FILE_TYPE))
    writer.add_array ("general.languages",  languages)

    # ----- stt.variant + capability KV -----
    writer.add_string("stt.variant", variant)
    writer.add_bool  ("stt.capability.translate",   True)
    writer.add_bool  ("stt.capability.lang_detect", False)
    # All shipping variants except canary-1b expose timestamp tokens.
    has_timestamps = "<|timestamp|>" in tok["specials"]
    writer.add_bool  ("stt.capability.timestamps",  has_timestamps)

    # ----- tokenizer.ggml.* -----
    # We use a custom "canary" tokenizer label since the pieces are an
    # aggregate over multiple SP sub-vocabs, not a single SPM.
    writer.add_string("tokenizer.ggml.model",     "canary")
    writer.add_array ("tokenizer.ggml.tokens",     tok["tokens"])
    writer.add_array ("tokenizer.ggml.scores",     tok["scores"])
    writer.add_array ("tokenizer.ggml.token_type", tok["types"])
    if tok["unk_id"] is not None:
        writer.add_uint32("tokenizer.ggml.unknown_token_id", tok["unk_id"])
    if tok["bos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", tok["bos_id"])
    if tok["eos_id"] is not None:
        writer.add_uint32("tokenizer.ggml.eos_token_id", tok["eos_id"])
    if tok["pad_id"] is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", tok["pad_id"])

    # ----- stt.canary.tokenizer.* (multi-SP routing) -----
    writer.add_array ("stt.canary.tokenizer.lang_codes",   tok["codes"])
    writer.add_array ("stt.canary.tokenizer.lang_offsets", tok["offsets"])
    writer.add_array ("stt.canary.tokenizer.lang_sizes",   tok["sizes"])
    writer.add_string("stt.canary.tokenizer.prompt_format", profile["prompt_format"])

    # ----- stt.canary.special.* (named special-token IDs) -----
    # Only emit a stable subset that the loader is likely to need; the
    # full set is reachable via the flat tokens list anyway.
    named_specials = [
        "<|nospeech|>",
        "<|startoftranscript|>",
        "<|endoftext|>",
        "<|startofcontext|>",
        "<|pnc|>",
        "<|nopnc|>",
        "<|itn|>",
        "<|noitn|>",
        "<|timestamp|>",
        "<|notimestamp|>",
        "<|diarize|>",
        "<|nodiarize|>",
        "<|spkchange|>",
        "<|audioseparator|>",
        "<pad>",
    ]
    for name in named_specials:
        if name in tok["specials"]:
            kv = name.strip("<>|")  # "startoftranscript", "pnc", ...
            kv = kv.replace("|", "_")  # safety
            writer.add_uint32(f"stt.canary.special.{kv}_id", int(tok["specials"][name]))

    # Per-language tag IDs (only present in canary2 variants where the
    # special-tokens table includes BCP-47 language tags).
    for code in languages:
        tag = f"<|{code}|>"
        if tag in tok["specials"]:
            writer.add_uint32(
                f"stt.canary.special.lang.{code}_id",
                int(tok["specials"][tag]),
            )

    # ----- stt.canary.encoder.* (FastConformer) -----
    writer.add_uint32("stt.canary.encoder.n_layers",             hp["enc_n_layers"])
    writer.add_uint32("stt.canary.encoder.d_model",              hp["enc_d_model"])
    writer.add_uint32("stt.canary.encoder.n_heads",              hp["enc_n_heads"])
    writer.add_uint32("stt.canary.encoder.d_ff",                 hp["enc_d_ff"])
    writer.add_uint32("stt.canary.encoder.conv_kernel",          hp["enc_conv_kernel"])
    writer.add_uint32("stt.canary.encoder.subsampling_factor",   hp["enc_subsampling_factor"])
    writer.add_uint32("stt.canary.encoder.subsampling_channels", hp["enc_subsampling_channels"])
    writer.add_uint32("stt.canary.encoder.pos_emb_max_len",      hp["enc_pos_emb_max_len"])
    writer.add_bool  ("stt.canary.encoder.use_bias",             hp["enc_use_bias"])

    # ----- stt.canary.decoder.* (Transformer decoder) -----
    writer.add_uint32("stt.canary.decoder.n_layers",             hp["dec_n_layers"])
    writer.add_uint32("stt.canary.decoder.d_model",              hp["dec_d_model"])
    writer.add_uint32("stt.canary.decoder.n_heads",              hp["dec_n_heads"])
    writer.add_uint32("stt.canary.decoder.d_ff",                 hp["dec_d_ff"])
    writer.add_uint32("stt.canary.decoder.max_position",         hp["dec_max_position"])
    writer.add_uint32("stt.canary.decoder.vocab_size",           hp["head_num_classes"])
    writer.add_string("stt.canary.decoder.activation",           hp["dec_activation"])
    writer.add_bool  ("stt.canary.decoder.pre_ln",               hp["dec_pre_ln"])
    writer.add_bool  ("stt.canary.decoder.learn_positional_encodings", hp["dec_learn_pos"])
    writer.add_bool  ("stt.canary.decoder.encoder_decoder_proj",  has_proj)

    # ----- stt.frontend.* -----
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
    writer.add_bool   ("stt.frontend.log",          hp["fe_log"])

    # ----- tensors -----
    consumed: set[str] = set()
    n_added = 0
    bytes_out = 0

    def add(nemo_name: str, gguf_name: str) -> None:
        nonlocal n_added, bytes_out
        if nemo_name not in sd_keys:
            raise KeyError(f"state_dict missing tensor: {nemo_name!r}")
        arr = tensor_to_fp32_numpy(sd[nemo_name])
        writer.add_tensor(gguf_name, arr)
        consumed.add(nemo_name)
        bytes_out += int(arr.nbytes)
        n_added += 1

    # pre_encode
    for nemo_name, gguf_name in PRE_ENCODE_TABLE:
        add(nemo_name, gguf_name)

    # encoder layers
    for i in range(hp["enc_n_layers"]):
        for suffix_nemo, suffix_gguf in ENCODER_BLOCK_TABLE:
            add(
                f"encoder.layers.{i}.{suffix_nemo}",
                f"enc.blocks.{i}.{suffix_gguf}",
            )

    # encoder_decoder_proj (Linear); present only when dims differ
    if has_proj:
        add("encoder_decoder_proj.weight", "enc.proj.weight")
        add("encoder_decoder_proj.bias",   "enc.proj.bias")

    # decoder embeddings
    add("transf_decoder._embedding.token_embedding.weight",       "dec.embed.token.weight")
    add("transf_decoder._embedding.position_embedding.pos_enc",   "dec.embed.pos_enc")
    add("transf_decoder._embedding.layer_norm.weight",            "dec.embed.norm.weight")
    add("transf_decoder._embedding.layer_norm.bias",              "dec.embed.norm.bias")

    # decoder layers
    for i in range(hp["dec_n_layers"]):
        for suffix_nemo, suffix_gguf in DECODER_BLOCK_TABLE:
            add(
                f"transf_decoder._decoder.layers.{i}.{suffix_nemo}",
                f"dec.layer.{i}.{suffix_gguf}",
            )

    # decoder final layer norm
    add("transf_decoder._decoder.final_layer_norm.weight", "dec.norm.weight")
    add("transf_decoder._decoder.final_layer_norm.bias",   "dec.norm.bias")

    # output projection (TokenClassifier head, single hidden layer)
    add("log_softmax.mlp.layer0.weight", "dec.head.weight")
    add("log_softmax.mlp.layer0.bias",   "dec.head.bias")

    expected = (
        len(PRE_ENCODE_TABLE)
        + hp["enc_n_layers"] * len(ENCODER_BLOCK_TABLE)
        + (2 if has_proj else 0)
        + 4                                         # decoder embeddings
        + hp["dec_n_layers"] * len(DECODER_BLOCK_TABLE)
        + 2                                         # final layer norm
        + 2                                         # output head
    )
    if n_added != expected:
        raise RuntimeError(
            f"tensor count mismatch: added {n_added}, expected {expected}"
        )
    print(f"Added {n_added} tensors ({bytes_out / (1024 * 1024):.1f} MB)")

    unused = sorted(
        k for k in (sd_keys - consumed) if not is_expected_unused(k)
    )
    if unused:
        print(
            f"WARNING: {len(unused)} state_dict keys were not consumed:",
            file=sys.stderr,
        )
        for k in unused[:10]:
            print(f"  {k}", file=sys.stderr)
        if len(unused) > 10:
            print(f"  ... and {len(unused) - 10} more", file=sys.stderr)

    print("Writing header + KV + tensor info...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    print("Writing tensor data...")
    writer.write_tensors_to_file()
    writer.close()

    print(f"Done. Wrote {out_path} ({out_path.stat().st_size / (1024 * 1024):.1f} MB)")


def languages_from_manifest(variant: str) -> list[str]:
    """Read the canary capability list from the golden manifest. The
    manifest is the single source of truth across the porting pipeline
    so the converter doesn't drift from intake/preflight."""
    import json
    p = REPO_ROOT / "tests" / "golden" / "canary" / f"{variant}.manifest.json"
    if not p.exists():
        raise FileNotFoundError(
            f"manifest not found at {p} — run porting-1-intake / "
            f"porting-2-oracle for variant {variant} before converting"
        )
    data = json.loads(p.read_text())
    langs = data.get("capabilities", {}).get("languages")
    if not langs:
        raise ValueError(f"manifest {p} has no capabilities.languages")
    return list(langs)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Convert a NeMo Canary multitask AED model to a GGUF.",
    )
    p.add_argument(
        "model",
        type=str,
        help="HF repo id (e.g. nvidia/canary-1b-flash) or local .nemo path",
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
    languages = languages_from_manifest(variant)

    out_path = args.out_path
    if out_path is None:
        out_path = REPO_ROOT / "models" / variant / gguf_name(variant, REFERENCE_DTYPE_LABEL)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    convert(args.model, out_path, variant, profile, languages)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
