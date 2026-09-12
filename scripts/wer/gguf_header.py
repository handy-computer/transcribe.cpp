#!/usr/bin/env python3
"""gguf_header.py — read a GGUF's identity without downloading the weights.

Two of the things the index needs about a model, its loader family and its
parameter count, are stamped in the GGUF and nowhere else. Most of the
shipped models are no longer on this disk: they were converted, uploaded,
and the local copy deleted. Downloading a multi-GB file to read a header is
absurd, so this parses the header out of the first few MB over an HTTP
range request instead.

The header is: magic, version, counts, the key/value block, then one info
record per tensor (name, dims, type, offset) before the tensor data starts.
Everything this module wants lives in that prefix. Only the KV block's size
varies much, because a tokenizer's token list is stored there, so the
fetcher grows its window when a read runs past the end rather than guessing
a size up front.

    from gguf_header import identity
    identity(pathlib.Path("models/whisper-tiny/whisper-tiny-F32.gguf"))
    identity_remote("handy-computer/whisper-medium-gguf", "whisper-medium-Q8_0.gguf")
"""
from __future__ import annotations

import math
import os
import pathlib
import struct

# GGUF metadata value types, from the spec's gguf_metadata_value_type enum.
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY,
 UINT64, INT64, FLOAT64) = range(13)

FIXED = {UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4, INT32: 4,
         FLOAT32: 4, BOOL: 1, UINT64: 8, INT64: 8, FLOAT64: 8}


class Window:
    """A growable byte window over a file or a remote object."""

    def __init__(self, fetch, initial: int = 1 << 20):
        self._fetch = fetch            # fetch(offset, length) -> bytes
        self._buf = fetch(0, initial)
        self.pos = 0

    def _need(self, end: int) -> None:
        while end > len(self._buf):
            more = max(len(self._buf), end - len(self._buf))
            chunk = self._fetch(len(self._buf), more)
            if not chunk:
                raise EOFError(f"ran past the end of the object at {end}")
            self._buf += chunk

    def take(self, n: int) -> bytes:
        self._need(self.pos + n)
        out = self._buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", errors="replace")

    def skip_value(self, vtype: int) -> None:
        """Advance past one metadata value without materialising it."""
        if vtype in FIXED:
            self.take(FIXED[vtype])
        elif vtype == STRING:
            self.take(self.u64())
        elif vtype == ARRAY:
            elem, n = self.u32(), self.u64()
            if elem in FIXED:
                self.take(FIXED[elem] * n)
            else:
                for _ in range(n):
                    self.skip_value(elem)
        else:
            raise ValueError(f"unknown GGUF value type {vtype}")


def _parse(win: Window) -> dict:
    if win.take(4) != b"GGUF":
        raise ValueError("not a GGUF file")
    version = win.u32()
    n_tensors, n_kv = win.u64(), win.u64()

    wanted = {"general.architecture", "general.basename", "general.size_label"}
    kv: dict[str, str] = {}
    for _ in range(n_kv):
        key = win.string()
        vtype = win.u32()
        if key in wanted and vtype == STRING:
            kv[key] = win.string()
        else:
            win.skip_value(vtype)

    params = 0
    for _ in range(n_tensors):
        win.string()                       # tensor name
        dims = [win.u64() for _ in range(win.u32())]
        win.u32()                          # ggml type
        win.u64()                          # offset into the data blob
        params += math.prod(dims)

    return {"version": version, "n_tensors": n_tensors,
            "family": kv.get("general.architecture"),
            "basename": kv.get("general.basename"),
            "size_label": kv.get("general.size_label"),
            "params": params, "params_m": round(params / 1e6, 1)}


def identity(path: pathlib.Path) -> dict:
    """Identity from a local GGUF, reading only as much as the header needs."""
    with open(path, "rb") as f:
        def fetch(offset: int, length: int) -> bytes:
            f.seek(offset)
            return f.read(length)
        return _parse(Window(fetch))


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = pathlib.Path("~/.cache/huggingface/token").expanduser()
    return p.read_text().strip() if p.exists() else None


def identity_remote(repo: str, filename: str, revision: str = "main") -> dict:
    """Identity from a GGUF on the Hub, over HTTP range requests.

    Costs a few MB rather than the whole file. Works on private repos when a
    token is available, which is the normal case for handy-computer/*."""
    import requests

    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    headers = {}
    tok = hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    session = requests.Session()

    def fetch(offset: int, length: int) -> bytes:
        h = dict(headers, Range=f"bytes={offset}-{offset + length - 1}")
        r = session.get(url, headers=h, timeout=60)
        if r.status_code not in (200, 206):
            raise RuntimeError(f"{repo}/{filename}: HTTP {r.status_code}")
        return r.content

    return _parse(Window(fetch))
