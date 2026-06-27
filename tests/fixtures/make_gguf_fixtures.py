#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
make_gguf_fixtures.py - emit synthetic GGUF files for the loader tests.

We deliberately implement the GGUF binary format directly here instead
of pulling in a third-party gguf-py package. The format is documented
in ggml/include/gguf.h, and a from-scratch writer keeps the test
surface hermetic and uv-friendly (no dependency resolution at
configure time). Tensor data emission uses Python struct only — no
numpy dep — because the toy tensors are tiny (~3000 fp32 elements).

Six fixtures are emitted:

  arch_parakeet.gguf       -- valid header, KV pairs:
                                general.architecture = "parakeet"
                                stt.variant          = "tdt-0.6b-v2"
                              No tokenizer KV. Used to assert the loader
                              recognizes the architecture and dispatches
                              to the Parakeet handler, which now
                              requires tokenizer KV and so returns
                              TRANSCRIBE_ERR_GGUF for this fixture.

  arch_unknown.gguf        -- valid header, KV pair:
                                general.architecture = "banana"
                              Used to assert the loader returns
                              TRANSCRIBE_ERR_UNSUPPORTED_ARCH for an
                              arch the registry does not know about.

  corrupt_magic.gguf       -- four bytes of garbage where "GGUF" should
                              be, followed by an otherwise plausible
                              header. Used to assert the loader returns
                              TRANSCRIBE_ERR_GGUF when
                              gguf_init_from_file rejects the file.

  tokenizer_minimal.gguf   -- a structurally complete minimal Parakeet
                              model: parakeet arch, stt.variant
                              "tdt-0.6b-v2", full tokenizer payload, the
                              full stt.parakeet.* / stt.frontend.* KV
                              block (toy hparams: 2 encoder layers,
                              d_model=8, n_heads=2, ...), and all 78
                              weight tensors with deterministic toy
                              float32 data. No capability KV and no
                              general.languages — exercises the
                              family-default code path. Despite the
                              "tokenizer_" name, this is now a complete
                              Parakeet model the loader walks
                              end-to-end (tokenizer + capabilities +
                              hparams + weights). Both
                              transcribe_tokenizer_smoke and
                              transcribe_parakeet_smoke load this file
                              and assert different things about it.

  tokenizer_minimal_v3.gguf -- structurally identical to
                              tokenizer_minimal.gguf except:
                                stt.variant              = "tdt-0.6b-v3"
                                stt.capability.lang_detect = true
                                general.languages = ["en","de","fr","es"]
                              Same toy vocabulary, same hparams, same
                              tensor catalog. The only behavioral
                              difference at the loader level is the
                              variant string + capability + language
                              KV — proves the v2 / v3 split is fully
                              KV-driven and shares the same code path.

  tokenizer_minimal_streaming.gguf
                           -- structurally identical to
                              tokenizer_minimal.gguf except:
                                stt.variant            = "tdt-0.6b-stream-toy"
                                stt.capability.streaming = true
                              The parakeet family has no streaming
                              hooks wired today, so this fixture pins
                              the "capability KV says yes but hooks
                              are missing" path: caps.supports_streaming
                              reads true at the model surface, and
                              transcribe_stream_begin still returns
                              NOT_IMPLEMENTED because the dispatcher
                              also checks that all three required
                              hooks (begin/feed/finalize) are wired.

  arch_cohere_minimal.gguf -- a structurally complete minimal Cohere ASR
                              model: cohere_asr arch, stt.variant
                              "cohere-asr-toy", full tokenizer payload
                              (BPE, 16-token toy vocab), full
                              stt.cohere.* / stt.frontend.* KV block
                              (toy hparams: 2 encoder layers + 2 decoder
                              layers, d_model=8, dec_hidden=8, ...), and
                              all weight tensors with deterministic toy
                              float32 data including the encoder-decoder
                              projection, decoder embed/blocks/final-norm,
                              and the head bias (head weight is tied to
                              dec.embed.token.weight per the family
                              default). Exercises the full
                              build_cohere_weights walk: pre_encode +
                              N_enc Conformer blocks (with FFN bias) +
                              enc_dec_proj + N_dec Transformer blocks
                              with self/cross-attention + tied head.

Run:

    uv run tests/fixtures/make_gguf_fixtures.py [output_dir]

If output_dir is omitted the script writes next to itself
(tests/fixtures/). It is also invoked from CMake as a custom target so
a local `cmake --build build --target fixtures` regenerates the files.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

# These match the constants in ggml/include/gguf.h. Pinned here so the
# generator does not silently drift if upstream renumbers (it would just
# stop matching the loader, which the loader tests would catch).
GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32

# gguf_type enum values (only the ones we use).
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9

# ggml_type enum values used for tensor data. Pinned here so we are not
# at the mercy of upstream renumbering — a mismatch would surface as a
# loader test failure (the most useful possible signal).
GGML_TYPE_F32 = 0

# Bytes per element for each ggml_type we emit.
GGML_TYPE_SIZE = {
    GGML_TYPE_F32: 4,
}


def _pack_string(s: str) -> bytes:
    # GGUF strings: uint64 length followed by raw bytes (NO trailing null).
    data = s.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _pack_kv_string(key: str, value: str) -> bytes:
    return (
        _pack_string(key)
        + struct.pack("<i", GGUF_TYPE_STRING)
        + _pack_string(value)
    )


def _pack_kv_uint32(key: str, value: int) -> bytes:
    return (
        _pack_string(key)
        + struct.pack("<i", GGUF_TYPE_UINT32)
        + struct.pack("<I", value)
    )


def _pack_kv_int32(key: str, value: int) -> bytes:
    return (
        _pack_string(key)
        + struct.pack("<i", GGUF_TYPE_INT32)
        + struct.pack("<i", value)
    )


def _pack_kv_bool(key: str, value: bool) -> bytes:
    # GGUF stores bool values as int8 on the wire (see gguf.h: "All bool
    # values are stored as int8_t").
    return (
        _pack_string(key)
        + struct.pack("<i", GGUF_TYPE_BOOL)
        + struct.pack("<b", 1 if value else 0)
    )


def _pack_kv_float32(key: str, value: float) -> bytes:
    return (
        _pack_string(key)
        + struct.pack("<i", GGUF_TYPE_FLOAT32)
        + struct.pack("<f", value)
    )


def _pack_kv_array_string(key: str, values: list[str]) -> bytes:
    body = (
        _pack_string(key)
        + struct.pack("<i", GGUF_TYPE_ARRAY)
        + struct.pack("<i", GGUF_TYPE_STRING)
        + struct.pack("<Q", len(values))
    )
    for v in values:
        body += _pack_string(v)
    return body


def _pack_kv_array_float32(key: str, values: list[float]) -> bytes:
    body = (
        _pack_string(key)
        + struct.pack("<i", GGUF_TYPE_ARRAY)
        + struct.pack("<i", GGUF_TYPE_FLOAT32)
        + struct.pack("<Q", len(values))
    )
    for v in values:
        body += struct.pack("<f", v)
    return body


def _pack_kv_array_int32(key: str, values: list[int]) -> bytes:
    body = (
        _pack_string(key)
        + struct.pack("<i", GGUF_TYPE_ARRAY)
        + struct.pack("<i", GGUF_TYPE_INT32)
        + struct.pack("<Q", len(values))
    )
    for v in values:
        body += struct.pack("<i", v)
    return body


def _build_header(magic: bytes, kv_blobs: list[bytes]) -> bytes:
    """Build a complete GGUF file with N pre-encoded KV blobs and zero tensors.

    Each entry in `kv_blobs` is the full key+type+value byte sequence for
    one KV pair, as produced by one of the _pack_kv_* helpers above.
    Mixing string / array / scalar KV in one file would otherwise need a
    polymorphic representation that the test fixtures do not benefit from.

    Pads to GGUF_DEFAULT_ALIGNMENT after the KV section because the loader
    seeks to that aligned offset before declaring "data section starts
    here", even when there are no tensors. Without the padding the seek
    lands past EOF — POSIX permits that, but we are explicit.
    """
    body = bytearray()
    body += magic
    body += struct.pack("<I", GGUF_VERSION)
    body += struct.pack("<q", 0)  # n_tensors
    body += struct.pack("<q", len(kv_blobs))  # n_kv
    for blob in kv_blobs:
        body += blob
    # Pad to alignment so the (empty) tensor data section starts on a
    # boundary the loader is happy with.
    pad = (-len(body)) % GGUF_DEFAULT_ALIGNMENT
    body += b"\x00" * pad
    return bytes(body)


def _string_kvs(pairs: list[tuple[str, str]]) -> list[bytes]:
    return [_pack_kv_string(k, v) for k, v in pairs]


# ---------------------------------------------------------------------------
# Tensor emission
# ---------------------------------------------------------------------------
#
# A "Tensor" here is just a (name, ne, dtype, data_bytes) tuple. ne is
# fast-to-slow dim order matching ggml_tensor::ne[]. data_bytes is the
# raw little-endian bytes of the tensor's elements, length must equal
# product(ne) * GGML_TYPE_SIZE[dtype].
#
# _build_full_gguf assembles header + KV section + tensor info section
# + aligned tensor data blob in one pass. The layout follows
# ggml/include/gguf.h section headers exactly (we read the format from
# there, not from any third-party gguf-py).


class Tensor:
    __slots__ = ("name", "ne", "dtype", "data")

    def __init__(
        self, name: str, ne: list[int], dtype: int, data: bytes
    ) -> None:
        nbytes = 1
        for d in ne:
            nbytes *= d
        nbytes *= GGML_TYPE_SIZE[dtype]
        if len(data) != nbytes:
            raise ValueError(
                f"tensor {name!r}: ne={ne} dtype={dtype} expects "
                f"{nbytes} bytes, got {len(data)}"
            )
        if not (1 <= len(ne) <= 4):
            raise ValueError(
                f"tensor {name!r}: ne must have 1..4 dims, got {len(ne)}"
            )
        self.name = name
        self.ne = list(ne)
        self.dtype = dtype
        self.data = data


def _ggml_pad(x: int, align: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    """Round x up to the next multiple of align."""
    return (x + align - 1) // align * align


def _pack_tensor_info(t: Tensor, data_offset: int) -> bytes:
    body = _pack_string(t.name)
    body += struct.pack("<I", len(t.ne))
    for d in t.ne:
        body += struct.pack("<q", d)
    body += struct.pack("<i", t.dtype)
    body += struct.pack("<Q", data_offset)
    return body


def _build_full_gguf(
    magic: bytes,
    kv_blobs: list[bytes],
    tensors: list[Tensor],
) -> bytes:
    """Assemble a complete GGUF file with KV pairs AND tensor data.

    Layout (from gguf.h):
      header (24 bytes: magic, version, n_tensors, n_kv)
      KV section (concatenated kv_blobs)
      tensor info section (one entry per tensor)
      [pad to alignment]
      tensor data blob (each tensor's bytes at its declared offset
                        within the blob)

    Each tensor's data offset within the data blob is aligned to
    GGUF_DEFAULT_ALIGNMENT. ggml_get_alignment defaults to 32 when
    general.alignment is absent (we don't write it), so emitting at the
    same alignment matches what the loader expects.
    """
    body = bytearray()
    body += magic
    body += struct.pack("<I", GGUF_VERSION)
    body += struct.pack("<q", len(tensors))  # n_tensors
    body += struct.pack("<q", len(kv_blobs))  # n_kv

    for blob in kv_blobs:
        body += blob

    # Compute per-tensor offsets within the (yet-to-be-written) data
    # blob. Each tensor starts on a GGUF_DEFAULT_ALIGNMENT boundary;
    # the size of each tensor is its raw byte count (no internal
    # padding required because the next tensor's start is realigned).
    offsets: list[int] = []
    cursor = 0
    for t in tensors:
        cursor = _ggml_pad(cursor)
        offsets.append(cursor)
        cursor += len(t.data)
    data_blob_size = _ggml_pad(cursor)

    for t, off in zip(tensors, offsets):
        body += _pack_tensor_info(t, off)

    # Pad from end of tensor info section to start of data blob.
    body_len = len(body)
    data_blob_start = _ggml_pad(body_len)
    body += b"\x00" * (data_blob_start - body_len)

    # Emit each tensor's bytes at its declared offset within the data
    # blob. The data_blob_start above already aligned us to a 32-byte
    # boundary, so each per-tensor offset (which is also a multiple of
    # 32) lands at the right absolute position relative to data_blob_start.
    data_blob = bytearray(data_blob_size)
    for t, off in zip(tensors, offsets):
        data_blob[off : off + len(t.data)] = t.data
    body += bytes(data_blob)

    return bytes(body)


def _f32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _f32_seq(num_elements: int, tensor_index: int) -> bytes:
    """Build deterministic float32 data for a toy tensor.

    The first element of tensor i is exactly float(i). Subsequent
    elements step by 0.0001. This means a C++ test that loads the
    fixture and reads `tensor.data[0]` for a known tensor index can
    assert an exact value, proving the bytes were copied (not zeroed
    or filled with garbage). Different tensors have different first
    elements, so a wrong-index swap shows up as a wrong first element.
    """
    return _f32_bytes(
        [tensor_index + j * 0.0001 for j in range(num_elements)]
    )


# ---------------------------------------------------------------------------
# Toy parakeet hparams + tensor catalog
# ---------------------------------------------------------------------------
#
# The numeric values here are pinned by the parakeet_smoke test (and by
# the loader's read_parakeet_hparams cross-field validation). They are
# small enough that the resulting fixture is ~20 KB but structurally
# complete: every named slot in ParakeetWeights gets a real tensor with
# the right shape, so the C++ ingest code path is exercised end-to-end.

PARAKEET_HP = {
    "enc_n_layers":             2,
    "enc_d_model":              8,
    "enc_n_heads":              2,
    "enc_d_ff":                 16,
    "enc_conv_kernel":          3,
    "enc_subsampling_factor":   2,
    "enc_subsampling_channels": 4,
    "enc_pos_emb_max_len":      32,
    "pred_hidden":              4,
    "pred_n_layers":            1,
    # pred_vocab is the embed table row count, which is (raw vocab + 1)
    # for the prepended "no previous token" / start row. The toy vocab
    # is 16 tokens, so pred_vocab is 17.
    "pred_vocab":               17,
    "joint_hidden":             4,
    "joint_num_extra_outputs":  2,
    # Joint output activation. Real Parakeet 0.6B v2/v3 ship "relu"
    # (verified from config.json on both variants); we mirror that
    # in the toy fixture so the loader exercises the same allow-list
    # branch.
    "joint_activation":         "relu",
    # TDT durations. The loader requires len(durations) ==
    # joint_num_extra_outputs, so a 2-output toy needs a 2-entry
    # durations list. The exact values are not interesting at the
    # loader level — the decoder uses them, not the loader.
    "tdt_durations":            [0, 1],
    # TDT max_symbols stuck-prevention cap. Mirror the real default.
    "tdt_max_symbols":          10,
    # Full stt.frontend.* block. Toy values that satisfy the loader's
    # cross-field invariants (win_length <= n_fft, f_max > f_min, etc.)
    # but are otherwise a miniature of NeMo Parakeet's mel pipeline.
    "fe_type":         "mel",
    "fe_num_mels":     4,
    "fe_sample_rate":  16000,
    "fe_n_fft":        16,
    "fe_win_length":   8,
    "fe_hop_length":   4,
    "fe_window":       "hann",
    "fe_normalize":    "per_feature",
    "fe_dither":       1e-5,
    "fe_pre_emphasis": 0.97,
    "fe_f_min":        0.0,
    "fe_f_max":        8000.0,
}


def _parakeet_hparams_kv() -> list[bytes]:
    """KV blobs for the stt.parakeet.* / stt.frontend.* hparams the
    loader's read_parakeet_hparams() requires."""
    hp = PARAKEET_HP
    return [
        _pack_kv_uint32("stt.parakeet.encoder.n_layers",             hp["enc_n_layers"]),
        _pack_kv_uint32("stt.parakeet.encoder.d_model",              hp["enc_d_model"]),
        _pack_kv_uint32("stt.parakeet.encoder.n_heads",              hp["enc_n_heads"]),
        _pack_kv_uint32("stt.parakeet.encoder.d_ff",                 hp["enc_d_ff"]),
        _pack_kv_uint32("stt.parakeet.encoder.conv_kernel",          hp["enc_conv_kernel"]),
        _pack_kv_uint32("stt.parakeet.encoder.subsampling_factor",   hp["enc_subsampling_factor"]),
        _pack_kv_uint32("stt.parakeet.encoder.subsampling_channels", hp["enc_subsampling_channels"]),
        _pack_kv_uint32("stt.parakeet.encoder.pos_emb_max_len",      hp["enc_pos_emb_max_len"]),
        _pack_kv_uint32("stt.parakeet.predictor.hidden",             hp["pred_hidden"]),
        _pack_kv_uint32("stt.parakeet.predictor.n_layers",           hp["pred_n_layers"]),
        _pack_kv_uint32("stt.parakeet.predictor.vocab",              hp["pred_vocab"]),
        _pack_kv_uint32("stt.parakeet.joint.hidden",                 hp["joint_hidden"]),
        _pack_kv_uint32("stt.parakeet.joint.num_extra_outputs",      hp["joint_num_extra_outputs"]),
        _pack_kv_string("stt.parakeet.joint.activation",             hp["joint_activation"]),
        _pack_kv_array_int32("stt.parakeet.tdt.durations",           hp["tdt_durations"]),
        _pack_kv_uint32("stt.parakeet.tdt.max_symbols",              hp["tdt_max_symbols"]),
        _pack_kv_string ("stt.frontend.type",         hp["fe_type"]),
        _pack_kv_uint32 ("stt.frontend.num_mels",     hp["fe_num_mels"]),
        _pack_kv_uint32 ("stt.frontend.sample_rate",  hp["fe_sample_rate"]),
        _pack_kv_uint32 ("stt.frontend.n_fft",        hp["fe_n_fft"]),
        _pack_kv_uint32 ("stt.frontend.win_length",   hp["fe_win_length"]),
        _pack_kv_uint32 ("stt.frontend.hop_length",   hp["fe_hop_length"]),
        _pack_kv_string ("stt.frontend.window",       hp["fe_window"]),
        _pack_kv_string ("stt.frontend.normalize",    hp["fe_normalize"]),
        _pack_kv_float32("stt.frontend.dither",       hp["fe_dither"]),
        _pack_kv_float32("stt.frontend.pre_emphasis", hp["fe_pre_emphasis"]),
        _pack_kv_float32("stt.frontend.f_min",        hp["fe_f_min"]),
        _pack_kv_float32("stt.frontend.f_max",        hp["fe_f_max"]),
    ]


# Index list of (name, ne) pairs in the canonical order. The C++
# parakeet_smoke test relies on the index of a few specific tensors so
# it can assert exact first-element values via _f32_seq above.
def _parakeet_pre_encode_F_prime(n_mels: int, causal: bool = False) -> int:
    """Mirror of src/arch/parakeet/weights.cpp's pre_encode_F_prime
    lambda. The pre-encode subsampling stack is three stride-2 / k=3
    convs; the loader traces them on the configured padding so the
    flattened (channels × F') feed into the pre_encode.out linear
    matches both offline and cache-aware variants.

    causal=False: offline / ChunkedLimitedWithRc branch. Total per-axis
    pad = (k-1)/2 + (k-1)/2 = 2 for k=3.
    causal=True: ChunkedLimited (cache-aware, nemotron) branch.
    Total per-axis pad = (k-1) + (s-1) = 3 for k=3, s=2.

    For real parakeet sizes (n_mels=128) the offline trace produces
    F' = 16, matching the old `n_mels // subs` shortcut. For toy
    sizes (n_mels=4) the trace produces F' = 1 (offline) / F' = 2
    (causal) while the shortcut gives 2 — that divergence is why
    this function exists.
    """
    k = 3
    s = 2
    total_pad = ((k - 1) + (s - 1)) if causal else ((k - 1) // 2 + (k - 1) // 2)
    dim = n_mels
    for _ in range(3):
        dim = ((dim + total_pad - k) // s) + 1
    return dim


def _parakeet_tensor_descriptors(causal_pre_encode: bool = False) -> list[tuple[str, list[int]]]:
    hp = PARAKEET_HP
    d_model      = hp["enc_d_model"]
    d_ff         = hp["enc_d_ff"]
    n_heads      = hp["enc_n_heads"]
    head_dim     = d_model // n_heads
    k            = hp["enc_conv_kernel"]
    channels     = hp["enc_subsampling_channels"]
    n_mels       = hp["fe_num_mels"]
    subs         = hp["enc_subsampling_factor"]
    pre_in       = channels * _parakeet_pre_encode_F_prime(n_mels, causal_pre_encode)
    n_layers     = hp["enc_n_layers"]
    pred_h       = hp["pred_hidden"]
    pred_v       = hp["pred_vocab"]
    pred_layers  = hp["pred_n_layers"]
    gates        = 4 * pred_h
    joint_h      = hp["joint_hidden"]
    joint_n      = (pred_v - 1) + hp["joint_num_extra_outputs"] + 1

    out: list[tuple[str, list[int]]] = []

    # ----- pre_encode -----
    out += [
        ("enc.pre_encode.conv.0.weight", [3, 3, 1, channels]),
        ("enc.pre_encode.conv.0.bias",   [channels]),
        ("enc.pre_encode.conv.2.weight", [3, 3, 1, channels]),
        ("enc.pre_encode.conv.2.bias",   [channels]),
        ("enc.pre_encode.conv.3.weight", [1, 1, channels, channels]),
        ("enc.pre_encode.conv.3.bias",   [channels]),
        ("enc.pre_encode.conv.5.weight", [3, 3, 1, channels]),
        ("enc.pre_encode.conv.5.bias",   [channels]),
        ("enc.pre_encode.conv.6.weight", [1, 1, channels, channels]),
        ("enc.pre_encode.conv.6.bias",   [channels]),
        ("enc.pre_encode.out.weight",    [pre_in, d_model]),
        ("enc.pre_encode.out.bias",      [d_model]),
    ]

    # ----- encoder blocks -----
    for i in range(n_layers):
        out += [
            (f"enc.blocks.{i}.norm_ff1.weight",        [d_model]),
            (f"enc.blocks.{i}.norm_ff1.bias",          [d_model]),
            (f"enc.blocks.{i}.ff1.linear1.weight",     [d_model, d_ff]),
            (f"enc.blocks.{i}.ff1.linear2.weight",     [d_ff, d_model]),

            (f"enc.blocks.{i}.norm_attn.weight",       [d_model]),
            (f"enc.blocks.{i}.norm_attn.bias",         [d_model]),
            (f"enc.blocks.{i}.attn.linear_q.weight",   [d_model, d_model]),
            (f"enc.blocks.{i}.attn.linear_k.weight",   [d_model, d_model]),
            (f"enc.blocks.{i}.attn.linear_v.weight",   [d_model, d_model]),
            (f"enc.blocks.{i}.attn.linear_out.weight", [d_model, d_model]),
            (f"enc.blocks.{i}.attn.linear_pos.weight", [d_model, d_model]),
            (f"enc.blocks.{i}.attn.pos_bias_u",        [head_dim, n_heads]),
            (f"enc.blocks.{i}.attn.pos_bias_v",        [head_dim, n_heads]),

            (f"enc.blocks.{i}.norm_conv.weight",       [d_model]),
            (f"enc.blocks.{i}.norm_conv.bias",         [d_model]),
            (f"enc.blocks.{i}.conv.pointwise1.weight", [1, d_model, 2 * d_model]),
            (f"enc.blocks.{i}.conv.depthwise.weight",  [k, 1, d_model]),
            (f"enc.blocks.{i}.conv.pointwise2.weight", [1, d_model, d_model]),
            (f"enc.blocks.{i}.conv.bn.weight",         [d_model]),
            (f"enc.blocks.{i}.conv.bn.bias",           [d_model]),
            (f"enc.blocks.{i}.conv.bn.running_mean",   [d_model]),
            (f"enc.blocks.{i}.conv.bn.running_var",    [d_model]),

            (f"enc.blocks.{i}.norm_ff2.weight",        [d_model]),
            (f"enc.blocks.{i}.norm_ff2.bias",          [d_model]),
            (f"enc.blocks.{i}.ff2.linear1.weight",     [d_model, d_ff]),
            (f"enc.blocks.{i}.ff2.linear2.weight",     [d_ff, d_model]),

            (f"enc.blocks.{i}.norm_out.weight",        [d_model]),
            (f"enc.blocks.{i}.norm_out.bias",          [d_model]),
        ]

    # ----- predictor -----
    out += [("pred.embed.weight", [pred_h, pred_v])]
    for i in range(pred_layers):
        out += [
            (f"pred.lstm.{i}.Wx",   [pred_h, gates]),
            (f"pred.lstm.{i}.Wh",   [pred_h, gates]),
            (f"pred.lstm.{i}.bias", [gates]),
        ]

    # ----- joint -----
    out += [
        ("joint.enc.weight",  [d_model, joint_h]),
        ("joint.enc.bias",    [joint_h]),
        ("joint.pred.weight", [pred_h,  joint_h]),
        ("joint.pred.bias",   [joint_h]),
        ("joint.out.weight",  [joint_h, joint_n]),
        ("joint.out.bias",    [joint_n]),
    ]

    return out


def _parakeet_tensors(causal_pre_encode: bool = False) -> list[Tensor]:
    descriptors = _parakeet_tensor_descriptors(causal_pre_encode=causal_pre_encode)
    tensors: list[Tensor] = []
    for idx, (name, ne) in enumerate(descriptors):
        n_elem = 1
        for d in ne:
            n_elem *= d
        tensors.append(
            Tensor(name, ne, GGML_TYPE_F32, _f32_seq(n_elem, idx))
        )
    return tensors


# ---------------------------------------------------------------------------
# Toy cohere hparams + tensor catalog
# ---------------------------------------------------------------------------
#
# Same role as PARAKEET_HP / _parakeet_tensor_descriptors above but for
# the Cohere ASR family (Conformer encoder + autoregressive Transformer
# decoder with cross-attention + tied head). Toy values keep the fixture
# small (~50 KB) but every CohereWeights slot is populated, so
# build_cohere_weights() walks the full tensor catalog end-to-end.
#
# Numeric values are pinned by transcribe_cohere_smoke (the loader's
# read_cohere_hparams cross-field validation also enforces several
# invariants here: enc_d_model % enc_n_heads == 0,
# dec_hidden % dec_n_heads == 0, win_length <= n_fft, n_fft is a power
# of 2, fe_num_mels divisible by enc_subsampling_factor, etc.).

COHERE_HP = {
    # Encoder.
    "enc_n_layers":             2,
    "enc_d_model":              8,
    "enc_n_heads":              2,
    "enc_d_ff":                 16,
    "enc_conv_kernel":          3,
    "enc_subsampling_factor":   2,
    "enc_subsampling_channels": 4,
    "enc_pos_emb_max_len":      32,
    "enc_use_bias":             True,

    # Decoder.
    "dec_n_layers":   2,
    "dec_hidden":     8,
    "dec_n_heads":    2,
    "dec_inner":      16,
    "dec_max_seq":    32,
    "dec_activation": "relu",

    # Decoder start token id (separate from BOS in Cohere — explicit).
    "decoder_start_token_id": 1,

    # Head defaults: log_softmax true, tied_weights true. tied means the
    # loader does NOT load head.weight; head.bias is the only head
    # tensor in the catalog. Both KVs are optional but we set them
    # explicitly so a future default flip in the loader does not
    # silently change what this fixture exercises.
    "head_log_softmax":  True,
    "head_tied_weights": True,

    # Vocab size (matches TOY_VOCAB length below). dec.embed.token.weight
    # has ne[1] = vocab_size, head.bias has ne[0] = vocab_size.
    "vocab_size": 16,

    # Frontend. Same toy mel pipeline as the Parakeet fixture; the
    # loader cross-field invariants are: win_length <= n_fft, n_fft is
    # a power of 2, fe_num_mels % enc_subsampling_factor == 0,
    # f_max > f_min >= 0, type=="mel", window=="hann",
    # normalize=="per_feature", pad_mode in {"reflect","constant"}.
    "fe_type":         "mel",
    "fe_num_mels":     4,
    "fe_sample_rate":  16000,
    "fe_n_fft":        16,
    "fe_win_length":   8,
    "fe_hop_length":   4,
    "fe_window":       "hann",
    "fe_normalize":    "per_feature",
    "fe_dither":       1e-5,
    "fe_pre_emphasis": 0.97,
    "fe_f_min":        0.0,
    "fe_f_max":        8000.0,
    "fe_pad_mode":     "reflect",
}


def _cohere_hparams_kv() -> list[bytes]:
    """KV blobs for the stt.cohere.* / stt.frontend.* hparams the
    loader's read_cohere_hparams() requires (plus the optional ones we
    set explicitly)."""
    hp = COHERE_HP
    return [
        # Encoder.
        _pack_kv_uint32("stt.cohere.encoder.n_layers",             hp["enc_n_layers"]),
        _pack_kv_uint32("stt.cohere.encoder.d_model",              hp["enc_d_model"]),
        _pack_kv_uint32("stt.cohere.encoder.n_heads",              hp["enc_n_heads"]),
        _pack_kv_uint32("stt.cohere.encoder.d_ff",                 hp["enc_d_ff"]),
        _pack_kv_uint32("stt.cohere.encoder.conv_kernel",          hp["enc_conv_kernel"]),
        _pack_kv_uint32("stt.cohere.encoder.subsampling_factor",   hp["enc_subsampling_factor"]),
        _pack_kv_uint32("stt.cohere.encoder.subsampling_channels", hp["enc_subsampling_channels"]),
        _pack_kv_uint32("stt.cohere.encoder.pos_emb_max_len",      hp["enc_pos_emb_max_len"]),
        _pack_kv_bool  ("stt.cohere.encoder.use_bias",             hp["enc_use_bias"]),
        # Decoder.
        _pack_kv_uint32("stt.cohere.decoder.n_layers",   hp["dec_n_layers"]),
        _pack_kv_uint32("stt.cohere.decoder.hidden_size", hp["dec_hidden"]),
        _pack_kv_uint32("stt.cohere.decoder.n_heads",    hp["dec_n_heads"]),
        _pack_kv_uint32("stt.cohere.decoder.inner_size",  hp["dec_inner"]),
        _pack_kv_uint32("stt.cohere.decoder.max_seq_len", hp["dec_max_seq"]),
        _pack_kv_string("stt.cohere.decoder.activation", hp["dec_activation"]),
        _pack_kv_uint32("stt.cohere.decoder_start_token_id", hp["decoder_start_token_id"]),
        # Head.
        _pack_kv_bool("stt.cohere.head.log_softmax",  hp["head_log_softmax"]),
        _pack_kv_bool("stt.cohere.head.tied_weights", hp["head_tied_weights"]),
        # Frontend.
        _pack_kv_string ("stt.frontend.type",         hp["fe_type"]),
        _pack_kv_uint32 ("stt.frontend.num_mels",     hp["fe_num_mels"]),
        _pack_kv_uint32 ("stt.frontend.sample_rate",  hp["fe_sample_rate"]),
        _pack_kv_uint32 ("stt.frontend.n_fft",        hp["fe_n_fft"]),
        _pack_kv_uint32 ("stt.frontend.win_length",   hp["fe_win_length"]),
        _pack_kv_uint32 ("stt.frontend.hop_length",   hp["fe_hop_length"]),
        _pack_kv_string ("stt.frontend.window",       hp["fe_window"]),
        _pack_kv_string ("stt.frontend.normalize",    hp["fe_normalize"]),
        _pack_kv_float32("stt.frontend.dither",       hp["fe_dither"]),
        _pack_kv_float32("stt.frontend.pre_emphasis", hp["fe_pre_emphasis"]),
        _pack_kv_float32("stt.frontend.f_min",        hp["fe_f_min"]),
        _pack_kv_float32("stt.frontend.f_max",        hp["fe_f_max"]),
        _pack_kv_string ("stt.frontend.pad_mode",     hp["fe_pad_mode"]),
    ]


def _cohere_tensor_descriptors() -> list[tuple[str, list[int]]]:
    hp = COHERE_HP
    d_model       = hp["enc_d_model"]
    d_ff          = hp["enc_d_ff"]
    n_heads       = hp["enc_n_heads"]
    head_dim      = d_model // n_heads
    k             = hp["enc_conv_kernel"]
    channels      = hp["enc_subsampling_channels"]
    n_mels        = hp["fe_num_mels"]
    subs          = hp["enc_subsampling_factor"]
    pre_in        = channels * (n_mels // subs)
    enc_layers    = hp["enc_n_layers"]
    dec_h         = hp["dec_hidden"]
    dec_in        = hp["dec_inner"]
    dec_layers    = hp["dec_n_layers"]
    dec_max_seq   = hp["dec_max_seq"]
    vocab         = hp["vocab_size"]

    out: list[tuple[str, list[int]]] = []

    # ----- pre_encode -----
    out += [
        ("enc.pre_encode.conv.0.weight", [3, 3, 1, channels]),
        ("enc.pre_encode.conv.0.bias",   [channels]),
        ("enc.pre_encode.conv.2.weight", [3, 3, 1, channels]),
        ("enc.pre_encode.conv.2.bias",   [channels]),
        ("enc.pre_encode.conv.3.weight", [1, 1, channels, channels]),
        ("enc.pre_encode.conv.3.bias",   [channels]),
        ("enc.pre_encode.conv.5.weight", [3, 3, 1, channels]),
        ("enc.pre_encode.conv.5.bias",   [channels]),
        ("enc.pre_encode.conv.6.weight", [1, 1, channels, channels]),
        ("enc.pre_encode.conv.6.bias",   [channels]),
        ("enc.pre_encode.out.weight",    [pre_in, d_model]),
        ("enc.pre_encode.out.bias",      [d_model]),
    ]

    # ----- encoder blocks (Conformer with FFN bias) -----
    for i in range(enc_layers):
        out += [
            (f"enc.blocks.{i}.norm_ff1.weight",       [d_model]),
            (f"enc.blocks.{i}.norm_ff1.bias",         [d_model]),
            (f"enc.blocks.{i}.ff1.linear1.weight",    [d_model, d_ff]),
            (f"enc.blocks.{i}.ff1.linear1.bias",      [d_ff]),
            (f"enc.blocks.{i}.ff1.linear2.weight",    [d_ff,    d_model]),
            (f"enc.blocks.{i}.ff1.linear2.bias",      [d_model]),

            (f"enc.blocks.{i}.norm_attn.weight",      [d_model]),
            (f"enc.blocks.{i}.norm_attn.bias",        [d_model]),
            (f"enc.blocks.{i}.attn.linear_q.weight",  [d_model, d_model]),
            (f"enc.blocks.{i}.attn.linear_q.bias",    [d_model]),
            (f"enc.blocks.{i}.attn.linear_k.weight",  [d_model, d_model]),
            (f"enc.blocks.{i}.attn.linear_k.bias",    [d_model]),
            (f"enc.blocks.{i}.attn.linear_v.weight",  [d_model, d_model]),
            (f"enc.blocks.{i}.attn.linear_v.bias",    [d_model]),
            (f"enc.blocks.{i}.attn.linear_out.weight",[d_model, d_model]),
            (f"enc.blocks.{i}.attn.linear_out.bias",  [d_model]),
            (f"enc.blocks.{i}.attn.linear_pos.weight",[d_model, d_model]),
            (f"enc.blocks.{i}.attn.pos_bias_u",       [head_dim, n_heads]),
            (f"enc.blocks.{i}.attn.pos_bias_v",       [head_dim, n_heads]),

            (f"enc.blocks.{i}.norm_conv.weight",      [d_model]),
            (f"enc.blocks.{i}.norm_conv.bias",        [d_model]),
            (f"enc.blocks.{i}.conv.pointwise1.weight",[1, d_model, 2 * d_model]),
            (f"enc.blocks.{i}.conv.pointwise1.bias",  [2 * d_model]),
            (f"enc.blocks.{i}.conv.depthwise.weight", [k, 1,        d_model]),
            (f"enc.blocks.{i}.conv.depthwise.bias",   [d_model]),
            (f"enc.blocks.{i}.conv.pointwise2.weight",[1, d_model,  d_model]),
            (f"enc.blocks.{i}.conv.pointwise2.bias",  [d_model]),
            (f"enc.blocks.{i}.conv.bn.weight",        [d_model]),
            (f"enc.blocks.{i}.conv.bn.bias",          [d_model]),
            (f"enc.blocks.{i}.conv.bn.running_mean",  [d_model]),
            (f"enc.blocks.{i}.conv.bn.running_var",   [d_model]),

            (f"enc.blocks.{i}.norm_ff2.weight",       [d_model]),
            (f"enc.blocks.{i}.norm_ff2.bias",         [d_model]),
            (f"enc.blocks.{i}.ff2.linear1.weight",    [d_model, d_ff]),
            (f"enc.blocks.{i}.ff2.linear1.bias",      [d_ff]),
            (f"enc.blocks.{i}.ff2.linear2.weight",    [d_ff,    d_model]),
            (f"enc.blocks.{i}.ff2.linear2.bias",      [d_model]),

            (f"enc.blocks.{i}.norm_out.weight",       [d_model]),
            (f"enc.blocks.{i}.norm_out.bias",         [d_model]),
        ]

    # ----- encoder-decoder projection -----
    out += [
        ("enc_dec_proj.weight", [d_model, dec_h]),
        ("enc_dec_proj.bias",   [dec_h]),
    ]

    # ----- decoder embedding -----
    out += [
        ("dec.embed.token.weight", [dec_h, vocab]),
        ("dec.embed.pos_enc",      [dec_h, dec_max_seq]),
        ("dec.embed.norm.weight",  [dec_h]),
        ("dec.embed.norm.bias",    [dec_h]),
    ]

    # ----- decoder blocks (Transformer: self-attn + cross-attn + FFN) -----
    for i in range(dec_layers):
        out += [
            (f"dec.blocks.{i}.norm_self.weight",       [dec_h]),
            (f"dec.blocks.{i}.norm_self.bias",         [dec_h]),
            (f"dec.blocks.{i}.self_attn.q.weight",     [dec_h, dec_h]),
            (f"dec.blocks.{i}.self_attn.q.bias",       [dec_h]),
            (f"dec.blocks.{i}.self_attn.k.weight",     [dec_h, dec_h]),
            (f"dec.blocks.{i}.self_attn.k.bias",       [dec_h]),
            (f"dec.blocks.{i}.self_attn.v.weight",     [dec_h, dec_h]),
            (f"dec.blocks.{i}.self_attn.v.bias",       [dec_h]),
            (f"dec.blocks.{i}.self_attn.out.weight",   [dec_h, dec_h]),
            (f"dec.blocks.{i}.self_attn.out.bias",     [dec_h]),

            (f"dec.blocks.{i}.norm_cross.weight",      [dec_h]),
            (f"dec.blocks.{i}.norm_cross.bias",        [dec_h]),
            (f"dec.blocks.{i}.cross_attn.q.weight",    [dec_h, dec_h]),
            (f"dec.blocks.{i}.cross_attn.q.bias",      [dec_h]),
            (f"dec.blocks.{i}.cross_attn.k.weight",    [dec_h, dec_h]),
            (f"dec.blocks.{i}.cross_attn.k.bias",      [dec_h]),
            (f"dec.blocks.{i}.cross_attn.v.weight",    [dec_h, dec_h]),
            (f"dec.blocks.{i}.cross_attn.v.bias",      [dec_h]),
            (f"dec.blocks.{i}.cross_attn.out.weight",  [dec_h, dec_h]),
            (f"dec.blocks.{i}.cross_attn.out.bias",    [dec_h]),

            (f"dec.blocks.{i}.norm_ff.weight",         [dec_h]),
            (f"dec.blocks.{i}.norm_ff.bias",           [dec_h]),
            (f"dec.blocks.{i}.ff.dense_in.weight",     [dec_h,  dec_in]),
            (f"dec.blocks.{i}.ff.dense_in.bias",       [dec_in]),
            (f"dec.blocks.{i}.ff.dense_out.weight",    [dec_in, dec_h]),
            (f"dec.blocks.{i}.ff.dense_out.bias",      [dec_h]),
        ]

    # ----- decoder final norm -----
    out += [
        ("dec.final_norm.weight", [dec_h]),
        ("dec.final_norm.bias",   [dec_h]),
    ]

    # ----- head (bias only; weight is tied to dec.embed.token.weight) -----
    out += [("head.bias", [vocab])]

    return out


def _cohere_tensors() -> list[Tensor]:
    descriptors = _cohere_tensor_descriptors()
    tensors: list[Tensor] = []
    for idx, (name, ne) in enumerate(descriptors):
        n_elem = 1
        for d in ne:
            n_elem *= d
        tensors.append(
            Tensor(name, ne, GGML_TYPE_F32, _f32_seq(n_elem, idx))
        )
    return tensors


# Token IDs the cohere fixture pins for transcribe_cohere_smoke. Picked
# to be valid token ids in TOY_VOCAB (the Parakeet vocab is reused).
COHERE_TOKEN_IDS = {
    "decoder_start": 1,  # <s>
    "bos":           1,  # <s>
    "eos":           2,  # </s>
    "pad":           0,  # <unk> (reuse; the fixture only needs a valid id)
    "unk":           0,  # <unk>
}


# ---------------------------------------------------------------------------
# Toy qwen3_asr hparams + tensor catalog
# ---------------------------------------------------------------------------
#
# Audio-LLM pattern: audio encoder (3x Conv2d subsampler + N transformer
# blocks + LN/proj1/proj2 head) and a Qwen3 causal LM decoder with
# GQA + per-head Q/K-RMSNorm + SwiGLU + tied lm_head. All cross-field
# invariants the loader enforces are honored here:
#
#   enc_d_model % enc_n_heads == 0
#   dec_n_heads % dec_n_kv_heads == 0
#   dec_hidden_act in {silu, swish}
#   enc_activation == "gelu"
#   fe_type == "mel"
#   dec_tie_word_embeddings == true  (Phase 1.2 load-time assertion)
#   mrope_section_{t,h,w} sum to dec_head_dim / 2
#   enc_output_dim == dec_hidden     (Phase 4.1 load-time assertion)
#
# Tokenizer carries the chat-template pieces the Phase 1.6 resolver
# requires (<|im_start|>, <|im_end|>, Ċ, system, user, assistant) and
# valid in-range ids for audio_{start,end,pad}_token_id.

QWEN3_ASR_HP = {
    # Audio encoder.
    "enc_n_layers":              2,
    "enc_d_model":               16,
    "enc_n_heads":               2,   # head_dim = 8
    "enc_ffn_dim":               32,
    "enc_num_mel_bins":          8,
    "enc_downsample_hidden":     16,
    "enc_output_dim":            16,  # must equal dec_hidden
    "enc_max_source_positions":  64,
    "enc_n_window":              2,
    "enc_n_window_infer":        4,
    "enc_conv_chunksize":        8,
    "enc_activation":            "gelu",

    # Text LM.
    "dec_n_layers":                2,
    "dec_hidden":                  16,
    "dec_intermediate":            32,
    "dec_n_heads":                 2,   # GQA: 2 / 1
    "dec_n_kv_heads":              1,
    "dec_head_dim":                8,
    "dec_hidden_act":              "silu",
    "dec_rms_norm_eps":            1e-6,
    "dec_rope_theta":              10000.0,
    "dec_rope_mrope_section_t":    2,   # 2 + 1 + 1 == 4 == head_dim/2
    "dec_rope_mrope_section_h":    1,
    "dec_rope_mrope_section_w":    1,
    "dec_rope_mrope_interleaved":  True,
    "dec_max_position_embeddings": 128,
    "dec_tie_word_embeddings":     True,
    "dec_vocab_size":              32,

    # Audio-injection ids (indices into the toy vocab).
    "audio_start_token_id": 16,
    "audio_end_token_id":   17,
    "audio_token_id":       18,

    # Frontend (Whisper-style toy mel; same shape as Qwen3-ASR 0.6B in
    # structure, not in numeric size).
    "fe_type":         "mel",
    "fe_num_mels":     8,
    "fe_sample_rate":  16000,
    "fe_n_fft":        16,
    "fe_win_length":   8,
    "fe_hop_length":   4,
    "fe_window":       "hann_periodic",
    "fe_normalize":    "per_utterance",
    "fe_dither":       0.0,
    "fe_pre_emphasis": 0.0,
    "fe_f_min":        0.0,
    "fe_f_max":        8000.0,
    "fe_pad_mode":     "reflect",
    "fe_mel_norm":     "slaney",
    "fe_center":       True,
    "fe_chunk_length": 30,
    "fe_n_samples":    480000,
    "fe_nb_max_frames": 3000,
}


# 32-token toy vocab. IDs chosen so:
#   id  3 -> "<|im_end|>"     (also used as eos_token_id)
#   id  4 -> "\xc4\x8a"       (Ċ: byte-level encoded newline)
#   id 16 -> "<|audio_start|>"
#   id 17 -> "<|audio_end|>"
#   id 18 -> "<|audio_pad|>"
# These pins line up with QWEN3_ASR_HP["audio_*_token_id"] and the
# chat-template pieces resolve_chat_tokens() looks up at load.
QWEN3_ASR_VOCAB: list[str] = [
    "<|unk|>",         # 0
    "<|pad|>",         # 1
    "<|im_start|>",    # 2
    "<|im_end|>",      # 3  (eos)
    "\u010a",          # 4  (Ċ -- byte-level \n; UTF-8: c4 8a)
    "system",          # 5
    "user",            # 6
    "assistant",       # 7
    "\u2581hello",     # 8
    "\u2581world",     # 9
    "\u2581foo",       # 10
    "\u2581bar",       # 11
    "\u2581baz",       # 12
    "\u2581the",       # 13
    "\u2581a",         # 14
    ".",               # 15
    "<|audio_start|>", # 16
    "<|audio_end|>",   # 17
    "<|audio_pad|>",   # 18
    "<|unused_19|>",   # 19
    "<|unused_20|>",   # 20
    "<|unused_21|>",   # 21
    "<|unused_22|>",   # 22
    "<|unused_23|>",   # 23
    "<|unused_24|>",   # 24
    "<|unused_25|>",   # 25
    "<|unused_26|>",   # 26
    "<|unused_27|>",   # 27
    "<|unused_28|>",   # 28
    "<|unused_29|>",   # 29
    "<|unused_30|>",   # 30
    "<|unused_31|>",   # 31
]

# llama.cpp token_type codes: 1=NORMAL, 3=CONTROL, 4=USER_DEFINED,
# 5=UNUSED. Keep it simple here — the loader only checks that the
# lengths match n_tokens; the smoke test does not introspect types.
QWEN3_ASR_TOKEN_TYPES = [3 if QWEN3_ASR_VOCAB[i].startswith("<|") else 1
                         for i in range(len(QWEN3_ASR_VOCAB))]


def _qwen3_asr_hparams_kv() -> list[bytes]:
    hp = QWEN3_ASR_HP
    return [
        # Audio encoder.
        _pack_kv_uint32("stt.qwen3_asr.encoder.n_layers",             hp["enc_n_layers"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.d_model",              hp["enc_d_model"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.n_heads",              hp["enc_n_heads"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.ffn_dim",              hp["enc_ffn_dim"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.num_mel_bins",         hp["enc_num_mel_bins"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.downsample_hidden",    hp["enc_downsample_hidden"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.output_dim",           hp["enc_output_dim"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.max_source_positions", hp["enc_max_source_positions"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.n_window",             hp["enc_n_window"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.n_window_infer",       hp["enc_n_window_infer"]),
        _pack_kv_uint32("stt.qwen3_asr.encoder.conv_chunksize",       hp["enc_conv_chunksize"]),
        _pack_kv_string("stt.qwen3_asr.encoder.activation",           hp["enc_activation"]),
        # Text LM.
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.n_layers",                hp["dec_n_layers"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.hidden_size",             hp["dec_hidden"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.intermediate_size",       hp["dec_intermediate"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.n_heads",                 hp["dec_n_heads"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.n_kv_heads",              hp["dec_n_kv_heads"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.head_dim",                hp["dec_head_dim"]),
        _pack_kv_string ("stt.qwen3_asr.decoder.hidden_act",              hp["dec_hidden_act"]),
        _pack_kv_float32("stt.qwen3_asr.decoder.rms_norm_eps",            hp["dec_rms_norm_eps"]),
        _pack_kv_float32("stt.qwen3_asr.decoder.rope_theta",              hp["dec_rope_theta"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.rope_mrope_section_t",    hp["dec_rope_mrope_section_t"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.rope_mrope_section_h",    hp["dec_rope_mrope_section_h"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.rope_mrope_section_w",    hp["dec_rope_mrope_section_w"]),
        _pack_kv_bool   ("stt.qwen3_asr.decoder.rope_mrope_interleaved",  hp["dec_rope_mrope_interleaved"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.max_position_embeddings", hp["dec_max_position_embeddings"]),
        _pack_kv_bool   ("stt.qwen3_asr.decoder.tie_word_embeddings",     hp["dec_tie_word_embeddings"]),
        _pack_kv_uint32 ("stt.qwen3_asr.decoder.vocab_size",              hp["dec_vocab_size"]),
        # Audio-injection ids.
        _pack_kv_uint32("stt.qwen3_asr.audio_token_id",       hp["audio_token_id"]),
        _pack_kv_uint32("stt.qwen3_asr.audio_start_token_id", hp["audio_start_token_id"]),
        _pack_kv_uint32("stt.qwen3_asr.audio_end_token_id",   hp["audio_end_token_id"]),
        # Frontend.
        _pack_kv_string ("stt.frontend.type",          hp["fe_type"]),
        _pack_kv_uint32 ("stt.frontend.num_mels",      hp["fe_num_mels"]),
        _pack_kv_uint32 ("stt.frontend.sample_rate",   hp["fe_sample_rate"]),
        _pack_kv_uint32 ("stt.frontend.n_fft",         hp["fe_n_fft"]),
        _pack_kv_uint32 ("stt.frontend.win_length",    hp["fe_win_length"]),
        _pack_kv_uint32 ("stt.frontend.hop_length",    hp["fe_hop_length"]),
        _pack_kv_string ("stt.frontend.window",        hp["fe_window"]),
        _pack_kv_string ("stt.frontend.normalize",     hp["fe_normalize"]),
        _pack_kv_float32("stt.frontend.dither",        hp["fe_dither"]),
        _pack_kv_float32("stt.frontend.pre_emphasis",  hp["fe_pre_emphasis"]),
        _pack_kv_float32("stt.frontend.f_min",         hp["fe_f_min"]),
        _pack_kv_float32("stt.frontend.f_max",         hp["fe_f_max"]),
        _pack_kv_string ("stt.frontend.pad_mode",      hp["fe_pad_mode"]),
        _pack_kv_string ("stt.frontend.mel_norm",      hp["fe_mel_norm"]),
        _pack_kv_bool   ("stt.frontend.center",        hp["fe_center"]),
        _pack_kv_uint32 ("stt.frontend.chunk_length",  hp["fe_chunk_length"]),
        _pack_kv_uint32 ("stt.frontend.n_samples",     hp["fe_n_samples"]),
        _pack_kv_uint32 ("stt.frontend.nb_max_frames", hp["fe_nb_max_frames"]),
    ]


def _qwen3_asr_tensor_descriptors() -> list[tuple[str, list[int]]]:
    hp        = QWEN3_ASR_HP
    d_model   = hp["enc_d_model"]
    ffn_dim   = hp["enc_ffn_dim"]
    ds_h      = hp["enc_downsample_hidden"]
    n_mels    = hp["enc_num_mel_bins"]
    out_dim   = hp["enc_output_dim"]
    mel_ds1   = (n_mels + 1) // 2
    mel_ds2   = (mel_ds1 + 1) // 2
    mel_ds3   = (mel_ds2 + 1) // 2
    conv_out_in = ds_h * mel_ds3
    enc_layers = hp["enc_n_layers"]

    dec_h   = hp["dec_hidden"]
    dec_nh  = hp["dec_n_heads"]
    dec_nkv = hp["dec_n_kv_heads"]
    dec_hd  = hp["dec_head_dim"]
    dec_im  = hp["dec_intermediate"]
    q_out   = dec_nh  * dec_hd
    kv_out  = dec_nkv * dec_hd
    dec_layers = hp["dec_n_layers"]
    vocab      = hp["dec_vocab_size"]

    out: list[tuple[str, list[int]]] = []

    # Encoder subsample.
    out += [
        ("enc.conv.0.weight", [3, 3, 1,    ds_h]),
        ("enc.conv.0.bias",   [ds_h]),
        ("enc.conv.1.weight", [3, 3, ds_h, ds_h]),
        ("enc.conv.1.bias",   [ds_h]),
        ("enc.conv.2.weight", [3, 3, ds_h, ds_h]),
        ("enc.conv.2.bias",   [ds_h]),
        ("enc.conv_out.weight", [conv_out_in, d_model]),
    ]

    # Encoder blocks.
    for i in range(enc_layers):
        out += [
            (f"enc.blocks.{i}.norm_attn.weight",    [d_model]),
            (f"enc.blocks.{i}.norm_attn.bias",      [d_model]),
            (f"enc.blocks.{i}.attn.q.weight",       [d_model, d_model]),
            (f"enc.blocks.{i}.attn.q.bias",         [d_model]),
            (f"enc.blocks.{i}.attn.k.weight",       [d_model, d_model]),
            (f"enc.blocks.{i}.attn.k.bias",         [d_model]),
            (f"enc.blocks.{i}.attn.v.weight",       [d_model, d_model]),
            (f"enc.blocks.{i}.attn.v.bias",         [d_model]),
            (f"enc.blocks.{i}.attn.out.weight",     [d_model, d_model]),
            (f"enc.blocks.{i}.attn.out.bias",       [d_model]),
            (f"enc.blocks.{i}.norm_ffn.weight",     [d_model]),
            (f"enc.blocks.{i}.norm_ffn.bias",       [d_model]),
            (f"enc.blocks.{i}.ffn.fc1.weight",      [d_model, ffn_dim]),
            (f"enc.blocks.{i}.ffn.fc1.bias",        [ffn_dim]),
            (f"enc.blocks.{i}.ffn.fc2.weight",      [ffn_dim, d_model]),
            (f"enc.blocks.{i}.ffn.fc2.bias",        [d_model]),
        ]

    # Encoder head.
    out += [
        ("enc.ln_post.weight", [d_model]),
        ("enc.ln_post.bias",   [d_model]),
        ("enc.proj1.weight",   [d_model, d_model]),
        ("enc.proj1.bias",     [d_model]),
        ("enc.proj2.weight",   [d_model, out_dim]),
        ("enc.proj2.bias",     [out_dim]),
    ]

    # Decoder embed (tied to lm_head).
    out += [("dec.token_embd.weight", [dec_h, vocab])]

    # Decoder blocks.
    for i in range(dec_layers):
        out += [
            (f"dec.blocks.{i}.norm_attn.weight",    [dec_h]),
            (f"dec.blocks.{i}.norm_ffn.weight",     [dec_h]),
            (f"dec.blocks.{i}.attn.q.weight",       [dec_h, q_out]),
            (f"dec.blocks.{i}.attn.k.weight",       [dec_h, kv_out]),
            (f"dec.blocks.{i}.attn.v.weight",       [dec_h, kv_out]),
            (f"dec.blocks.{i}.attn.o.weight",       [q_out, dec_h]),
            (f"dec.blocks.{i}.attn.q_norm.weight",  [dec_hd]),
            (f"dec.blocks.{i}.attn.k_norm.weight",  [dec_hd]),
            (f"dec.blocks.{i}.ffn.gate.weight",     [dec_h, dec_im]),
            (f"dec.blocks.{i}.ffn.up.weight",       [dec_h, dec_im]),
            (f"dec.blocks.{i}.ffn.down.weight",     [dec_im, dec_h]),
        ]

    # Decoder final norm (no tied lm_head tensor; the graph reuses
    # dec.token_embd.weight).
    out += [("dec.output_norm.weight", [dec_h])]

    return out


def _qwen3_asr_tensors() -> list[Tensor]:
    descriptors = _qwen3_asr_tensor_descriptors()
    tensors: list[Tensor] = []
    for idx, (name, ne) in enumerate(descriptors):
        n_elem = 1
        for d in ne:
            n_elem *= d
        tensors.append(
            Tensor(name, ne, GGML_TYPE_F32, _f32_seq(n_elem, idx))
        )
    return tensors


# Minimal non-empty Qwen3-ASR chat-template string. The real template
# is ~80 lines of Jinja; the fixture just needs a non-empty string so
# the optional read succeeds. The smoke test does not render it.
QWEN3_ASR_CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n"
    "{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"wrote {path} ({len(data)} bytes)")


# Sixteen-token toy vocabulary. Indices used downstream by the tokenizer
# smoke test, so they are pinned here:
#
#   0  <unk>           special, unk_id
#   1  <s>             special, bos_id
#   2  </s>            special, eos_id
#   3  ▁hello          word "hello" with the SentencePiece word-boundary
#                       marker U+2581
#   4  ▁world          word "world" same
#   5  ▁foo
#   6  ▁bar
#   7  ▁baz
#   8  s               continuation piece (no leading marker)
#   9  ed              continuation piece
#  10  ing             continuation piece
#  11  ▁the
#  12  ▁a
#  13  ▁of
#  14  .               punctuation
#  15  <blank>         CTC / transducer blank id
TOY_VOCAB: list[str] = [
    "<unk>",
    "<s>",
    "</s>",
    "\u2581hello",
    "\u2581world",
    "\u2581foo",
    "\u2581bar",
    "\u2581baz",
    "s",
    "ed",
    "ing",
    "\u2581the",
    "\u2581a",
    "\u2581of",
    ".",
    "<blank>",
]
# Sentinel scores. The values are not interesting; the test only checks
# that read-back round-trips and that length-must-match is enforced.
TOY_SCORES = [-float(i) for i in range(len(TOY_VOCAB))]
# Token types: 1 == NORMAL, 3 == CONTROL (matches the llama.cpp /
# whisper.cpp convention). The test does not assert specific values, only
# that the array survives the round trip.
TOY_TOKEN_TYPES = [
    3, 3, 3,        # <unk> <s> </s>
    1, 1, 1, 1, 1,  # ▁hello ▁world ▁foo ▁bar ▁baz
    1, 1, 1,        # s ed ing
    1, 1, 1,        # ▁the ▁a ▁of
    1,              # .
    3,              # <blank>
]


def emit_fixtures(out_dir: Path) -> None:
    # Recognized architecture; no tokenizer payload. The Parakeet handler
    # rejects this with TRANSCRIBE_ERR_GGUF in 2B because tokenizer load
    # is now part of the contract.
    _write(
        out_dir / "arch_parakeet.gguf",
        _build_header(
            GGUF_MAGIC,
            _string_kvs(
                [
                    ("general.architecture", "parakeet"),
                    ("stt.variant", "tdt-0.6b-v2"),
                ]
            ),
        ),
    )

    # Recognized SenseVoice arch; no tokenizer / hparam payload. The
    # SenseVoice handler is reached, then rejects with
    # TRANSCRIBE_ERR_GGUF because the rest of the contract is missing.
    _write(
        out_dir / "arch_sensevoice.gguf",
        _build_header(
            GGUF_MAGIC,
            _string_kvs(
                [
                    ("general.architecture", "sensevoice"),
                    ("stt.variant", "sensevoice-small"),
                ]
            ),
        ),
    )

    # Same for FunASR-Nano (audio-llm arch).
    _write(
        out_dir / "arch_funasr_nano.gguf",
        _build_header(
            GGUF_MAGIC,
            _string_kvs(
                [
                    ("general.architecture", "funasr_nano"),
                    ("stt.variant", "fun-asr-nano-2512"),
                ]
            ),
        ),
    )

    # Architecture string the registry has never heard of.
    _write(
        out_dir / "arch_unknown.gguf",
        _build_header(
            GGUF_MAGIC,
            _string_kvs([("general.architecture", "banana")]),
        ),
    )

    # Bad magic. Everything after the magic is a structurally valid
    # header so we are testing magic-rejection specifically, not "the
    # file is short and the loader fails on something else first".
    _write(
        out_dir / "corrupt_magic.gguf",
        _build_header(
            b"NOPE",
            _string_kvs([("general.architecture", "parakeet")]),
        ),
    )

    # Recognized arch + complete minimal Parakeet model. The fixture
    # has tokenizer + capabilities (defaulted) + parakeet hparams +
    # all 78 weight tensors with deterministic toy fp32 data. The
    # loader walks every code path on this file: arch dispatch,
    # tokenizer ingest, capability KV (absent → family defaults),
    # languages KV (absent → information gap), parakeet hparams,
    # second-pass GGUF open with tensor data, build_parakeet_weights.
    assert len(TOY_VOCAB) == 16
    assert len(TOY_SCORES) == len(TOY_VOCAB)
    assert len(TOY_TOKEN_TYPES) == len(TOY_VOCAB)
    tokenizer_kv = [
        _pack_kv_string("tokenizer.ggml.model", "unigram"),
        _pack_kv_array_string("tokenizer.ggml.tokens", TOY_VOCAB),
        _pack_kv_array_float32("tokenizer.ggml.scores", TOY_SCORES),
        _pack_kv_array_int32("tokenizer.ggml.token_type", TOY_TOKEN_TYPES),
        _pack_kv_uint32("tokenizer.ggml.unknown_token_id", 0),
        _pack_kv_uint32("tokenizer.ggml.bos_token_id", 1),
        _pack_kv_uint32("tokenizer.ggml.eos_token_id", 2),
        _pack_kv_uint32("tokenizer.ggml.blank_token_id", 15),
    ]
    parakeet_hparams_kv = _parakeet_hparams_kv()
    parakeet_tensors    = _parakeet_tensors()

    _write(
        out_dir / "tokenizer_minimal.gguf",
        _build_full_gguf(
            GGUF_MAGIC,
            [
                _pack_kv_string("general.architecture", "parakeet"),
                _pack_kv_string("stt.variant", "tdt-0.6b-v2"),
                *tokenizer_kv,
                *parakeet_hparams_kv,
            ],
            parakeet_tensors,
        ),
    )

    # v3 fixture: same toy vocabulary, same hparams, same tensor
    # catalog — only the descriptive metadata (variant string) and
    # the capability + language KV differ. Pins "v2 vs v3 are the
    # same code path" structurally: build_parakeet_weights walks the
    # same 78 tensors regardless of which fixture it's called on,
    # and only the post-load capability struct differs.
    _write(
        out_dir / "tokenizer_minimal_v3.gguf",
        _build_full_gguf(
            GGUF_MAGIC,
            [
                _pack_kv_string("general.architecture", "parakeet"),
                _pack_kv_string("stt.variant", "tdt-0.6b-v3"),
                # Capability KV: v3 adds language detect.
                _pack_kv_bool("stt.capability.lang_detect", True),
                # Languages: toy four-language list, not the full 25
                # NeMo emits. The real list arrives in 2C with the
                # converter; the test only needs to assert that the
                # array round-trips and the count matches.
                _pack_kv_array_string(
                    "general.languages",
                    ["en", "de", "fr", "es"],
                ),
                *tokenizer_kv,
                *parakeet_hparams_kv,
            ],
            parakeet_tensors,
        ),
    )

    # Streaming-capability variant. Same toy vocabulary, same hparams,
    # same tensor catalog as v2 — but adds stt.capability.streaming = true
    # to flip caps.supports_streaming on at the model surface. The
    # parakeet family has no streaming hooks wired, so this fixture
    # exists specifically to exercise the dispatcher's "all three
    # required hooks (begin/feed/finalize) must be present" guard:
    # capability KV says yes, transcribe_stream_begin still returns
    # NOT_IMPLEMENTED.
    _write(
        out_dir / "tokenizer_minimal_streaming.gguf",
        _build_full_gguf(
            GGUF_MAGIC,
            [
                _pack_kv_string("general.architecture", "parakeet"),
                _pack_kv_string("stt.variant", "tdt-0.6b-stream-toy"),
                _pack_kv_bool("stt.capability.streaming", True),
                *tokenizer_kv,
                *parakeet_hparams_kv,
            ],
            parakeet_tensors,
        ),
    )

    # Translation-capability KV override. Same toy parakeet vocabulary,
    # hparams, and tensor catalog — but carries stt.capability.translate
    # = true. The parakeet family default is supports_translate=false, so
    # this fixture pins that the loader reads the canonical capability KV
    # and flips the flag on. This is the key the granite / medasr /
    # granite_nar converters emit; before the rename they wrote a
    # misspelled stt.capability.translation that the loader never read, so
    # granite -plus advertised translation it should not have.
    _write(
        out_dir / "tokenizer_minimal_translate.gguf",
        _build_full_gguf(
            GGUF_MAGIC,
            [
                _pack_kv_string("general.architecture", "parakeet"),
                _pack_kv_string("stt.variant", "tdt-0.6b-translate-toy"),
                _pack_kv_bool("stt.capability.translate", True),
                _pack_kv_array_string("stt.translation.target_languages", ["en"]),
                *tokenizer_kv,
                *parakeet_hparams_kv,
            ],
            parakeet_tensors,
        ),
    )

    # Cache-aware streaming variant (ChunkedLimited, nemotron-style).
    # Adds the att_context_style + flat (left, right) menu so the
    # parakeet stream_begin hook routes into the cache-aware path and
    # reaches the att_context_right validation. Built specifically to
    # exercise the "sentinel value < -1 is INVALID_ARG" reject; the
    # encoder graph never runs because the test asserts rejection
    # before any compute.
    # Cache-aware uses the causal-pre-encode pad in the loader, which
    # changes the expected pre_in dimension of enc.pre_encode.out.weight.
    # Build a separate tensor list for this fixture so the shape check
    # the loader runs at load time passes.
    parakeet_tensors_causal = _parakeet_tensors(causal_pre_encode=True)
    _write(
        out_dir / "tokenizer_minimal_streaming_cache_aware.gguf",
        _build_full_gguf(
            GGUF_MAGIC,
            [
                _pack_kv_string("general.architecture", "parakeet"),
                _pack_kv_string("stt.variant", "tdt-0.6b-cache-aware-toy"),
                _pack_kv_bool("stt.capability.streaming", True),
                _pack_kv_string("stt.parakeet.encoder.att_context_style",
                                "chunked_limited"),
                # Menu mirrors nemotron-speech-streaming-en-0.6b:
                # right ∈ {13, 6, 1, 0}, left = 70 throughout. Flat
                # (left, right) pairs; choices[0] is the default.
                _pack_kv_array_int32(
                    "stt.parakeet.encoder.att_context_size_choices",
                    [70, 13, 70, 6, 70, 1, 70, 0]),
                _pack_kv_int32("stt.parakeet.encoder.att_context_left",  70),
                _pack_kv_int32("stt.parakeet.encoder.att_context_right", 13),
                *tokenizer_kv,
                *parakeet_hparams_kv,
            ],
            parakeet_tensors_causal,
        ),
    )

    # Chunked-attention buffered streaming variant (ChunkedLimitedWithRc,
    # parakeet-unified-style). Adds the three context menus so the
    # parakeet stream_begin hook routes into the buffered path and
    # reaches the per-field ms validation. Same purpose as above:
    # exercise rejection of sentinel values < -1.
    _write(
        out_dir / "tokenizer_minimal_streaming_buffered.gguf",
        _build_full_gguf(
            GGUF_MAGIC,
            [
                _pack_kv_string("general.architecture", "parakeet"),
                _pack_kv_string("stt.variant", "tdt-0.6b-buffered-toy"),
                _pack_kv_bool("stt.capability.streaming", True),
                _pack_kv_string("stt.parakeet.encoder.att_context_style",
                                "chunked_limited_with_rc"),
                # Mirrors parakeet-unified-en-0.6b's menu shape.
                _pack_kv_array_int32(
                    "stt.parakeet.encoder.att_chunk_left_choices",  [70]),
                _pack_kv_array_int32(
                    "stt.parakeet.encoder.att_chunk_chunk_choices", [1, 2, 7, 13]),
                _pack_kv_array_int32(
                    "stt.parakeet.encoder.att_chunk_right_choices", [0, 1, 2, 4, 7, 13]),
                *tokenizer_kv,
                *parakeet_hparams_kv,
            ],
            parakeet_tensors,
        ),
    )

    # Cohere ASR minimal fixture. Uses the Cohere tokenizer model
    # name "bpe" (vs Parakeet's "unigram"); same toy vocabulary
    # otherwise. The tensor catalog is the full Cohere weight set
    # (encoder + enc_dec_proj + decoder + final norm + head bias)
    # at toy dims, so build_cohere_weights walks every code path.
    cohere_tokenizer_kv = [
        _pack_kv_string("tokenizer.ggml.model", "bpe"),
        _pack_kv_array_string("tokenizer.ggml.tokens", TOY_VOCAB),
        _pack_kv_array_float32("tokenizer.ggml.scores", TOY_SCORES),
        _pack_kv_array_int32("tokenizer.ggml.token_type", TOY_TOKEN_TYPES),
        _pack_kv_uint32("tokenizer.ggml.unknown_token_id", COHERE_TOKEN_IDS["unk"]),
        _pack_kv_uint32("tokenizer.ggml.bos_token_id",     COHERE_TOKEN_IDS["bos"]),
        _pack_kv_uint32("tokenizer.ggml.eos_token_id",     COHERE_TOKEN_IDS["eos"]),
    ]
    cohere_hparams_kv = _cohere_hparams_kv()
    cohere_tensors    = _cohere_tensors()

    _write(
        out_dir / "arch_cohere_minimal.gguf",
        _build_full_gguf(
            GGUF_MAGIC,
            [
                _pack_kv_string("general.architecture", "cohere_asr"),
                _pack_kv_string("stt.variant",          "cohere-asr-toy"),
                *cohere_tokenizer_kv,
                *cohere_hparams_kv,
            ],
            cohere_tensors,
        ),
    )

    # Qwen3-ASR minimal fixture. Byte-level BPE tokenizer ("gpt2" model
    # tag), 32-token toy vocab with the chat-template pieces the Phase
    # 1.6 resolver requires, 2 encoder blocks + 2 decoder blocks at toy
    # dims, all cross-field invariants honored. Exercises the full
    # load pipeline: tokenizer, chat_template KV, chat-token resolution,
    # hparams (including the tie_word_embeddings + mrope_section
    # invariants), build_qwen3_asr_weights, backend alloc, ffn_gate_up
    # packing, final capabilities.
    qwen3_asr_tokenizer_kv = [
        _pack_kv_string("tokenizer.ggml.model", "gpt2"),
        _pack_kv_array_string("tokenizer.ggml.tokens", QWEN3_ASR_VOCAB),
        _pack_kv_array_int32("tokenizer.ggml.token_type", QWEN3_ASR_TOKEN_TYPES),
        _pack_kv_uint32("tokenizer.ggml.unknown_token_id", 0),
        # eos = <|im_end|>. No bos for Qwen3 (chat template provides it).
        _pack_kv_uint32("tokenizer.ggml.eos_token_id", 3),
    ]
    qwen3_asr_hparams_kv = _qwen3_asr_hparams_kv()
    qwen3_asr_tensors    = _qwen3_asr_tensors()

    _write(
        out_dir / "arch_qwen3_asr_minimal.gguf",
        _build_full_gguf(
            GGUF_MAGIC,
            [
                _pack_kv_string("general.architecture", "qwen3_asr"),
                _pack_kv_string("stt.variant",          "qwen3-asr-toy"),
                _pack_kv_string("tokenizer.chat_template",
                                QWEN3_ASR_CHAT_TEMPLATE),
                # Capabilities: match the family note. The loader reads
                # general.languages as factual detection coverage; the
                # run() handler rejects any explicit language hint with
                # TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE until the prompt
                # renderer honors them (Phase 1.1 Option A).
                _pack_kv_bool("stt.capability.lang_detect", True),
                _pack_kv_array_string(
                    "general.languages",
                    ["en", "zh", "ja", "de"],
                ),
                *qwen3_asr_tokenizer_kv,
                *qwen3_asr_hparams_kv,
            ],
            qwen3_asr_tensors,
        ),
    )


def main(argv: list[str]) -> int:
    if len(argv) > 2:
        print(f"usage: {argv[0]} [output_dir]", file=sys.stderr)
        return 2
    out_dir = Path(argv[1]) if len(argv) == 2 else Path(__file__).resolve().parent
    emit_fixtures(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
