#!/usr/bin/env python3
"""Read selected GGUF metadata without downloading the weights.

The header -- magic, counts, the KV block, then one info record per tensor --
is the first few MB of the file, so a local read touches only that prefix and a
remote read is a couple of HTTP range requests rather than a multi-GB download.

Returns the keys the catalog needs: architecture and parameter count for
identity, and the capability surface the loader itself dispatches on
(stt.capability.*, general.languages, stt.translation.*).
"""
from __future__ import annotations

import math
import os
import pathlib
import struct

(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY,
 UINT64, INT64, FLOAT64) = range(13)
FIXED = {UINT8: 1, INT8: 1, UINT16: 2, INT16: 2, UINT32: 4, INT32: 4,
         FLOAT32: 4, BOOL: 1, UINT64: 8, INT64: 8, FLOAT64: 8}

WANTED = {
    "general.architecture", "general.basename", "general.size_label",
    "general.languages", "stt.variant",
    "stt.capability.translate", "stt.capability.lang_detect",
    "stt.capability.streaming", "stt.capability.speaker_diarization",
    "stt.capability.timestamps", "stt.capability.word_timestamps",
    "stt.translation.target_languages", "stt.translation.pairs",
    "stt.sortformer.max_speakers",
    "stt.parakeet.encoder.att_chunk_left_choices",
    "stt.parakeet.encoder.att_chunk_chunk_choices",
    "stt.parakeet.encoder.att_chunk_right_choices",
}


class Window:
    def __init__(self, fetch, initial: int = 1 << 20):
        self._fetch, self._buf, self.pos = fetch, fetch(0, initial), 0

    def _need(self, end: int) -> None:
        while end > len(self._buf):
            chunk = self._fetch(len(self._buf), max(len(self._buf), end - len(self._buf)))
            if not chunk:
                raise EOFError(f"ran past the end of the object at {end}")
            self._buf += chunk

    def take(self, n: int) -> bytes:
        self._need(self.pos + n)
        out = self._buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def i32(self): return struct.unpack("<i", self.take(4))[0]
    def string(self): return self.take(self.u64()).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        """Materialise one metadata value (only the shapes the catalog needs)."""
        if vtype == STRING:
            return self.string()
        if vtype == BOOL:
            return self.take(1) != b"\x00"
        if vtype in (UINT32, INT32):
            return self.i32() if vtype == INT32 else struct.unpack("<I", self.take(4))[0]
        if vtype in (UINT64, INT64):
            return self.u64()
        if vtype == ARRAY:
            elem, n = self.u32(), self.u64()
            return [self.value(elem) for _ in range(n)]
        if vtype in FIXED:
            self.take(FIXED[vtype])
            return None
        raise ValueError(f"unknown GGUF value type {vtype}")

    def skip_value(self, vtype: int) -> None:
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
    win.u32()                                   # version
    n_tensors, n_kv = win.u64(), win.u64()
    kv: dict = {}
    for _ in range(n_kv):
        key = win.string()
        vtype = win.u32()
        if key in WANTED:
            kv[key] = win.value(vtype)
        else:
            win.skip_value(vtype)
    params = 0
    for _ in range(n_tensors):
        win.string()
        dims = [win.u64() for _ in range(win.u32())]
        win.u32()
        win.u64()
        params += math.prod(dims)
    kv["_params"] = params
    kv["_n_tensors"] = n_tensors
    return kv


def read_local(path: pathlib.Path) -> dict:
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


def read_remote(repo: str, filename: str, revision: str = "main") -> dict:
    import requests
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
    headers = {}
    tok = hf_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    session = requests.Session()

    def fetch(offset: int, length: int) -> bytes:
        r = session.get(url, headers=dict(headers, Range=f"bytes={offset}-{offset + length - 1}"),
                        timeout=60)
        if r.status_code not in (200, 206):
            raise RuntimeError(f"{repo}/{filename}: HTTP {r.status_code}")
        return r.content

    return _parse(Window(fetch))
