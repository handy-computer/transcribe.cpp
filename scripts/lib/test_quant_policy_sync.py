#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gguf>=0.10",
# ]
# ///
"""Pin the converter-side quant bucketing against the canonical policy.

There are three hand-maintained copies of "what bucket does tensor X belong
to" in this repo:

  1. tools/transcribe-quantize/policy.cpp::classify_tensor  — CANONICAL.
     The C++ quantizer's per-tensor bucket table (Norm / Conv / ConvPw /
     Linear / Embed). It is the source of truth and mirrors the loader's
     dtype allowlist (src/transcribe-weights-util.h).
  2. scripts/lib/gguf_common.py::reference_dtype_for         — this test.
     The Python helper every converter uses to pick a per-tensor dtype when
     emitting a reference-tier (F32/F16/BF16) GGUF. It re-implements a SUBSET
     of (1): it only needs the Norm (-> F32) and Conv (-> F16 when the
     reference dtype is BF16, which the loader has no conv kernel for) rules,
     because at the reference dtype every other tensor just keeps that dtype.
  3. scripts/convert-funasr_nano.py::per_tensor_target_dtype — a family-local
     fork of (2). Not exercised here (importing it pulls torch); documented in
     that file. If it ever folds back into the shared helper, delete it.

Copies (1) and (2) carry "keep in sync" comments but nothing enforced them.
This module is that enforcement, for the Python copy: it locks the dtype
`reference_dtype_for` assigns to a representative tensor from every bucket and
every family-specific override, so the helper cannot silently regress or drift
further from policy.cpp. The name lists below are a transcription of
policy.cpp::classify_tensor — keep them in sync with that file (now there is a
single, *tested* transcription instead of a silent second implementation).

It is intentionally NOT a cross-process check against the C++ binary: that
would need a built transcribe-quantize and the multi-GB reference GGUFs, which
do not belong in a fast unit test. The canonical buckets are transcribed here
as data and reviewed against policy.cpp.

Run standalone (exit-code driven):  uv run scripts/lib/test_quant_policy_sync.py
Or under pytest:                    pytest scripts/lib/test_quant_policy_sync.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from gguf import GGMLQuantizationType as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo scripts/
from lib.gguf_common import reference_dtype_for  # noqa: E402


# ---------------------------------------------------------------------------
# Contract corpus — transcribed from policy.cpp::classify_tensor.
#
# Each list groups tensor names by the bucket reference_dtype_for IMPLEMENTS
# today. The asserts below check the helper actually routes them that way at
# both reference dtypes the converters use.
# ---------------------------------------------------------------------------

# Norm bucket: biases, LayerNorm/RMSNorm scales, positional tables, frontend
# buffers. Loader requires F32 in these slots -> F32 at every reference dtype.
NORM = [
    "dec.layers.0.attn.linear_q.bias",      # .bias
    "enc.blocks.3.norm_ff1.weight",         # norm_ prefix
    "enc.blocks.3.conv.bn.weight",          # .bn. batchnorm
    "dec.final_norm.weight",                # cohere final norm (dot separator)
    "dec.embed.norm.weight",                # cohere embed norm (dot separator)
    "dec.layers.0.attn.q_norm.weight",      # qwen3 per-head q_norm
    "dec.layers.0.attn.k_norm.weight",      # qwen3 per-head k_norm
    "dec.output_norm.weight",               # qwen3 final RMSNorm
    "enc.layers.0.ln_post.weight",          # qwen3 encoder ln_post
    "enc.layers.0.ln_pre.weight",           # qwen3 encoder ln_pre
    "enc.blocks.0.self_attn.pos_bias_u",    # conformer rel-pos bias u
    "enc.blocks.0.self_attn.pos_bias_v",    # conformer rel-pos bias v
    "dec.pos_enc",                          # cohere sinusoidal pos table
    "enc.pos_emb.weight",                   # whisper encoder pos_emb
    "dec.pos_emb.weight",                   # whisper decoder pos_emb
    "frontend.mel_filterbank",              # mel frontend buffer
    "frontend.window",                      # window frontend buffer
]

# Conv bucket: 2D / depthwise / 1x1 pointwise conv kernels. The loader has no
# BF16 conv kernel, so at BF16 reference these downcast to F16; at F32/F16
# reference they keep the reference dtype.
CONV = [
    "enc.blocks.3.conv.pointwise1.weight",  # conformer 1x1 pointwise
    "enc.blocks.3.conv.pointwise2.weight",  # conformer 1x1 pointwise
    "enc.pre_encode.conv.0.weight",         # pre-encode subsampling conv
    "enc.blocks.3.conv.depthwise.weight",   # conformer depthwise conv
]

# Linear / Embed: ggml_mul_mat operands and the decoder token embedding.
# Keep the reference dtype unchanged (block quantization is a Stage-5 concern).
LINEAR = [
    "enc.blocks.3.attn.linear_q.weight",    # attention projection
    "dec.layers.0.ffn.up.weight",           # FFN matrix
    "enc.blocks.3.attn.linear_out.weight",  # attention output projection
    "dec.embed.token.weight",               # cohere tied embedding (Embed)
    "dec.token_embd.weight",                # llama-style embedding (Embed)
]

# KNOWN DRIFT — policy.cpp::classify_tensor places these in the Norm (F32) or
# Conv (F16) bucket, but reference_dtype_for does NOT implement the matching
# rule, so it currently returns the reference dtype unchanged.
#
# This is SAFE today only because every family that emits one of these names
# either:
#   (a) converts at F32 reference, where reference_dtype_for is a no-op
#       (sensevoice: cmvn/after_norm/tp_norm/enc.embed; canary: norm{1,2,3}/
#       dec.norm; moonshine-streaming as shipped: enc.embedder.comp.log_k),
#   (b) special-cases the tensor to F32 in its own converter, bypassing this
#       helper (voxtral-realtime: dec.time_embed.inv_freq, emitted F32), or
#   (c) uses a family-local converter that does not call this helper at all
#       (granite_nar: conv_bn.*, prj.*, conv_pointwise/conv_depthwise).
#
# Pinned here so the gap is visible and locked. If a BF16/F16-reference family
# ever emits one of these names through reference_dtype_for, the helper will
# silently mis-store the tensor (e.g. a norm at BF16 where the loader wants
# F32). The fix is to add the rule to reference_dtype_for and MOVE the name
# into NORM / CONV above — this test will fail (the name no longer returns the
# reference dtype) until you do, which is the intended tripwire.
#
# Comment on each = the canonical policy.cpp bucket it *should* map to.
KNOWN_DRIFT = [
    "enc.blocks.3.conv_bn.weight",          # Norm (granite_nar conv_bn)
    "enc.blocks.3.conv_bn.running_mean",    # Norm (granite_nar BN running stat)
    "enc.blocks.3.conv_bn.running_var",     # Norm (granite_nar BN running stat)
    "prj.out_norm.weight",                  # Norm (granite_nar projector LN)
    "prj.layer_norms.2.weight",             # Norm (granite_nar per-layer LN)
    "dec.layer.5.norm1.weight",             # Norm (canary decoder LN)
    "dec.layer.5.norm2.weight",             # Norm (canary decoder LN)
    "dec.layer.5.norm3.weight",             # Norm (canary decoder LN)
    "dec.norm.weight",                      # Norm (canary final decoder LN)
    "prj.query",                            # Norm (granite_nar projector query)
    "prj.window_positions",                 # Norm (granite_nar window bias)
    "frontend.cmvn.shift",                  # Norm (sensevoice CMVN)
    "frontend.cmvn.scale",                  # Norm (sensevoice CMVN)
    "enc.embed.weight",                     # Norm (sensevoice prefix-token table)
    "enc.after_norm.weight",                # Norm (sensevoice trailing LN)
    "tp_encoders.tp_norm.weight",           # Norm (sensevoice tp LN)
    "enc.embedder.comp.log_k",              # Norm (moonshine-streaming asinh scalar)
    "dec.time_embed.inv_freq",              # Norm (voxtral-realtime time-embed table)
    "enc.blocks.3.conv_pointwise1.weight",  # ConvPw (granite_nar underscore form)
    "enc.blocks.3.conv_depthwise.weight",   # Conv (granite_nar underscore form)
    "enc.blocks.3.attn.fsmn.weight",        # Conv (sensevoice FSMN depthwise)
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_f32_reference_is_all_f32() -> None:
    """The safety net the F32-reference families rely on: at F32 reference,
    every tensor — whatever its bucket — is stored F32. This is what makes the
    KNOWN_DRIFT names safe for sensevoice / canary / moonshine-streaming."""
    for name in NORM + CONV + LINEAR + KNOWN_DRIFT:
        got = reference_dtype_for(name, T.F32)
        assert got == T.F32, f"{name}: F32 reference must stay F32, got {got.name}"


def test_norm_bucket_is_f32_at_bf16() -> None:
    """Implemented Norm rules: F32 regardless of reference dtype."""
    for name in NORM:
        got = reference_dtype_for(name, T.BF16)
        assert got == T.F32, f"{name}: Norm bucket must be F32, got {got.name}"


def test_conv_bucket_downcasts_to_f16_at_bf16() -> None:
    """Implemented Conv rules: BF16 reference -> F16 (no BF16 conv kernel);
    F16 reference keeps F16."""
    for name in CONV:
        got_bf16 = reference_dtype_for(name, T.BF16)
        assert got_bf16 == T.F16, f"{name}: Conv at BF16 ref must be F16, got {got_bf16.name}"
        got_f16 = reference_dtype_for(name, T.F16)
        assert got_f16 == T.F16, f"{name}: Conv at F16 ref must be F16, got {got_f16.name}"


def test_linear_keeps_reference_dtype() -> None:
    """Linear / Embed operands keep the reference dtype (no quant at convert)."""
    for name in LINEAR:
        assert reference_dtype_for(name, T.BF16) == T.BF16, f"{name}: must keep BF16"
        assert reference_dtype_for(name, T.F16) == T.F16, f"{name}: must keep F16"


def test_known_drift_is_pinned() -> None:
    """Lock the documented gaps: these C++ Norm/Conv tensors currently fall
    through reference_dtype_for to the reference dtype. If you implement the
    missing rule, this assert flips — move the name into NORM / CONV."""
    for name in KNOWN_DRIFT:
        got = reference_dtype_for(name, T.BF16)
        assert got == T.BF16, (
            f"{name}: KNOWN_DRIFT no longer returns the reference dtype "
            f"(got {got.name}) — reference_dtype_for grew a rule for it; "
            f"move it from KNOWN_DRIFT into NORM or CONV."
        )


_TESTS = [
    test_f32_reference_is_all_f32,
    test_norm_bucket_is_f32_at_bf16,
    test_conv_bucket_downcasts_to_f16_at_bf16,
    test_linear_keeps_reference_dtype,
    test_known_drift_is_pinned,
]


def main() -> int:
    failures = 0
    for t in _TESTS:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        else:
            print(f"ok   {t.__name__}")
    n_names = len(NORM) + len(CONV) + len(LINEAR) + len(KNOWN_DRIFT)
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} checks passed "
          f"over {n_names} tensor names "
          f"({len(KNOWN_DRIFT)} pinned as known drift).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
