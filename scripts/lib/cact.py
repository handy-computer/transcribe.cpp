"""Reader for Cactus Needle-3 `.cact` archives (Whistle, Needle).

Format (documented in cactus-compute/needle needle/model/export.py):
  header      48 u32 + f32 rope_theta (196 bytes)
  codebooks   codebook_len f32: Lloyd-Max unit-sphere codebooks cb2[4] | cb3[8] | cb4[16],
              already scaled by 1/sqrt(group)
  directory   num_tensors x 44-byte records, no names:
              u8 dtype, u8 ndim, u16 pad, u32 shape[4], u64 offset, u64 nbytes,
              u32 group_size, u32 bits   (dtype 1=FP16 2=FP32 3=CQ 4=RAW)
  blobs       64-byte aligned

CQ blob for a logical [out, in] matrix (in padded to a multiple of group):
packed LSB-first indices (out * in_pad * bits / 8 bytes), then per-group L2
norms (out * in_pad / group FP16). Reconstruction per group:
    w_group = (codebook_bits[idx] * norm) @ H     H = Walsh-Hadamard(group) / sqrt(group)
Ternary (bits=5) and binary (bits=1) records are not used by Whistle and raise.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAG = 0x05E12A84
FP16, FP32, CQ, RAW = 1, 2, 3, 4
_HDR_FMT = "<48If"
_REC_FMT = "<BBHIIIIQQII"

HEADER_FIELDS = (
    "tag", "num_tensors", "codebook_len", "kv_window", "kv_bits",
    "vocab", "out_vocab", "d_model", "num_heads", "num_kv_heads", "num_layers",
    "qk_head_dim", "v_head_dim", "max_seq_len", "hada_n", "mhc_lanes",
    "sliding_window", "global_mask_lo", "global_mask_hi", "qkv_conv_taps",
    "engram_slots", "engram_sub_dim", "num_engram_tables", "engram_conv_taps",
    "engram_conv_dilation", "engram_seed_heads", "num_engram_orders",
)


@dataclass
class Record:
    index: int
    dtype: int
    shape: tuple
    offset: int
    nbytes: int
    group: int
    bits: int

    @property
    def kind(self) -> str:
        if self.dtype == CQ:
            return f"CQ{self.bits}"
        return {FP16: "F16", FP32: "F32", RAW: "RAW"}[self.dtype]


def _hadamard(n: int) -> np.ndarray:
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _unpack_lsb(packed: np.ndarray, bits: int, in_pad: int) -> np.ndarray:
    out = packed.shape[0]
    chunks = packed.reshape(out, in_pad // 8, bits).astype(np.uint64)
    word = np.zeros(chunks.shape[:-1], np.uint64)
    for b in range(bits):
        word |= chunks[..., b] << np.uint64(8 * b)
    idx = np.empty((out, in_pad // 8, 8), np.uint8)
    mask = np.uint64((1 << bits) - 1)
    for i in range(8):
        idx[..., i] = (word >> np.uint64(i * bits)) & mask
    return idx.reshape(out, in_pad)


class Cact:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.buf = self.path.read_bytes()
        hdr = struct.unpack(_HDR_FMT, self.buf[: struct.calcsize(_HDR_FMT)])
        if hdr[0] != TAG:
            raise ValueError(f"{path}: not a Needle-3 .cact (tag {hdr[0]:#x})")
        self.header = dict(zip(HEADER_FIELDS, hdr[: len(HEADER_FIELDS)]))
        n_orders = self.header["num_engram_orders"]
        self.header["engram_orders"] = list(hdr[27:27 + n_orders])
        n_sites = hdr[31]
        self.header["engram_sites"] = list(hdr[32:32 + n_sites])
        self.header["rope_theta"] = float(hdr[48])
        off = struct.calcsize(_HDR_FMT)
        cbl = self.header["codebook_len"]
        cb = np.frombuffer(self.buf, "<f4", cbl, off).astype(np.float64)
        if cbl != 28:
            raise ValueError(f"unexpected codebook_len {cbl}")
        self.codebooks = {2: cb[0:4], 3: cb[4:12], 4: cb[12:28]}
        off += 4 * cbl
        rs = struct.calcsize(_REC_FMT)
        self.records = []
        for i in range(self.header["num_tensors"]):
            r = struct.unpack(_REC_FMT, self.buf[off + i * rs: off + (i + 1) * rs])
            self.records.append(Record(i, r[0], tuple(r[3:3 + r[1]]), r[7], r[8], r[9], r[10]))

    def raw(self, i: int) -> bytes:
        r = self.records[i]
        return self.buf[r.offset: r.offset + r.nbytes]

    def array(self, i: int) -> np.ndarray:
        """FP16/FP32 tensors in their stored dtype; CQ tensors dequantized to float32 [out, in]."""
        r = self.records[i]
        if r.dtype == FP16:
            return np.frombuffer(self.raw(i), "<f2").reshape(r.shape).copy()
        if r.dtype == FP32:
            return np.frombuffer(self.raw(i), "<f4").reshape(r.shape).copy()
        if r.dtype == CQ:
            return self.dequant(i)
        raise ValueError(f"tensor {i} is RAW")

    def dequant(self, i: int) -> np.ndarray:
        r = self.records[i]
        if r.bits not in self.codebooks:
            raise NotImplementedError(f"tensor {i}: CQ bits={r.bits} not supported")
        out, in_dim = r.shape
        g, bits = r.group, r.bits
        in_pad = (in_dim + g - 1) // g * g
        n_packed = out * in_pad * bits // 8
        n_groups = out * in_pad // g
        blob = self.raw(i)
        if len(blob) != n_packed + 2 * n_groups:
            raise ValueError(f"tensor {i}: blob {len(blob)} != {n_packed} + {2 * n_groups}")
        packed = np.frombuffer(blob, np.uint8, n_packed).reshape(out, in_pad * bits // 8)
        norms = np.frombuffer(blob, "<f2", n_groups, n_packed).astype(np.float64)
        idx = _unpack_lsb(packed, bits, in_pad)
        unit = self.codebooks[bits][idx].reshape(out, in_pad // g, g)
        rot = unit * norms.reshape(out, in_pad // g, 1)
        w = (rot @ _hadamard(g)).reshape(out, in_pad)
        return w[:, :in_dim].astype(np.float32)

    def tokenizer(self) -> dict:
        """The trailing RAW SentencePiece dump (format in export.py)."""
        raw_idx = [r.index for r in self.records if r.dtype == RAW]
        if len(raw_idx) != 1 or raw_idx[0] != len(self.records) - 1:
            raise ValueError("expected exactly one trailing RAW tokenizer record")
        t = self.raw(raw_idx[0])
        n, pad, eos, bos, unk, adp, bfb, _ = struct.unpack("<IIIIIBBH", t[:24])
        p, pieces, scores, types = 24, [], [], []
        for _ in range(n):
            sc, ty, ln = struct.unpack("<fBH", t[p:p + 7])
            p += 7
            pieces.append(t[p:p + ln].decode("utf-8"))
            scores.append(sc)
            types.append(ty)
            p += ln
        if p != len(t):
            raise ValueError(f"tokenizer blob has {len(t) - p} trailing bytes")
        return {"pieces": pieces, "scores": scores, "types": types, "pad": pad, "eos": eos,
                "bos": bos, "unk": unk, "add_dummy_prefix": adp, "byte_fallback": bfb}
