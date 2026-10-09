"""ctypes binding to the closed Cactus Needle engine that runs Whistle.

The engine is the Whistle oracle (reference_framework
author_repo_cactus_needle_engine). It is loaded directly from the pinned
cactus-needle wheel instead of through the `cactus-needle` Python package,
because that wrapper sends usage telemetry, re-fetches config.json on every
load and caches engines and weights under ~/.cache/cactus-needle.

Artifacts, all pinned:
  engine   Cactus-Compute/needle3@ENGINE_REVISION python/<wheel>, extracted
           to <sandbox>/refs/cactus/needle3-engine/<version>/wheel/
  weights  Cactus-Compute/whistle@WEIGHTS_REVISION whistle.cact (HF cache)

The C API is declared in needle.h (same folder as the CLI in the needle3
repo). One process-global speech model; not thread-safe.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import zipfile
from pathlib import Path

import numpy as np

ENGINE_REPO = "Cactus-Compute/needle3"
ENGINE_REVISION = "2ae11323dc000f5e70c49f7403efa6af12ba9e67"
ENGINE_VERSION = "3.2.0"
WEIGHTS_REPO = "Cactus-Compute/whistle"
WEIGHTS_REVISION = "b358ddadd89b7a713b5aa131f23032d3cca1b251"
SAMPLE_RATE = 16000
MAX_SECONDS = 30
FRAME_SECONDS = 0.08

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = REPO_ROOT.parent / "refs" / "cactus" / "needle3-engine" / ENGINE_VERSION

_WHEEL_TAGS = {
    ("Darwin", "arm64"): ("macosx_11_0_arm64", "libneedle3.dylib"),
    ("Darwin", "x86_64"): ("macosx_11_0_x86_64", "libneedle3.dylib"),
    ("Linux", "x86_64"): ("manylinux2014_x86_64", "libneedle3.so"),
    ("Linux", "aarch64"): ("manylinux2014_aarch64", "libneedle3.so"),
}


def provenance() -> dict:
    return {
        "framework": "cactus_needle_engine",
        "engine": f"{ENGINE_REPO}@{ENGINE_REVISION} cactus_needle-{ENGINE_VERSION}",
        "model": f"{WEIGHTS_REPO}@{WEIGHTS_REVISION} whistle.cact",
    }


def engine_lib_path() -> Path:
    override = os.environ.get("WHISTLE_ENGINE_LIB")
    if override:
        return Path(override)
    key = (platform.system(), platform.machine())
    if key not in _WHEEL_TAGS:
        raise RuntimeError(f"no pinned Needle engine wheel for {key}")
    tag, lib = _WHEEL_TAGS[key]
    path = ENGINE_DIR / "wheel" / "needle" / lib
    if not path.exists():
        from huggingface_hub import hf_hub_download

        wheel = hf_hub_download(
            ENGINE_REPO,
            f"python/cactus_needle-{ENGINE_VERSION}-py3-none-{tag}.whl",
            revision=ENGINE_REVISION,
        )
        with zipfile.ZipFile(wheel) as zf:
            zf.extract(f"needle/{lib}", ENGINE_DIR / "wheel")
    return path


def weights_path() -> Path:
    override = os.environ.get("WHISTLE_CACT")
    if override:
        return Path(override)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(WEIGHTS_REPO, "whistle.cact", revision=WEIGHTS_REVISION))


def load_audio(path: str | Path) -> np.ndarray:
    import soundfile as sf

    pcm, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: {sr} Hz, expected {SAMPLE_RATE} (resample upstream)")
    return np.ascontiguousarray(pcm.mean(axis=1), dtype=np.float32)


class Engine:
    def __init__(self, lib_path: Path | None = None, cact: Path | None = None,
                 buffer_size: int = 1 << 20):
        self.lib_path = Path(lib_path or engine_lib_path())
        self.cact = Path(cact or weights_path())
        lib = ctypes.CDLL(str(self.lib_path))
        f32p = ctypes.POINTER(ctypes.c_float)
        lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
        lib.needle_load.restype = ctypes.c_int
        lib.needle_transcribe.argtypes = [f32p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
                                          ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.needle_transcribe.restype = ctypes.c_int
        lib.needle_embed.argtypes = [ctypes.c_char_p, f32p, ctypes.c_int, f32p, ctypes.c_int]
        lib.needle_embed.restype = ctypes.c_int
        lib.needle_last_error.argtypes = []
        lib.needle_last_error.restype = ctypes.c_char_p
        self._lib = lib
        self._buf = ctypes.create_string_buffer(buffer_size)
        data = self.cact.read_bytes()
        if lib.needle_load(data, len(data)) < 0:
            raise RuntimeError(self._error())

    def _error(self) -> str:
        msg = self._lib.needle_last_error()
        return msg.decode("utf-8", "replace") if msg else "unknown engine error"

    @staticmethod
    def _ptr(pcm: np.ndarray):
        pcm = np.ascontiguousarray(pcm, dtype=np.float32)
        if pcm.size == 0:
            pcm = np.zeros(1, dtype=np.float32)
        return pcm, pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def transcribe(self, pcm: np.ndarray, language: str | None = None,
                   keywords: str | None = None, word_timestamps: bool = False) -> dict:
        """needle_transcribe. Returns the engine JSON plus `n_tokens` (the return value)."""
        pcm, ptr = self._ptr(pcm)
        n = self._lib.needle_transcribe(
            ptr, int(pcm.size),
            language.encode() if language else None,
            keywords.encode() if keywords else None,
            int(bool(word_timestamps)), self._buf, len(self._buf))
        if n < 0:
            raise RuntimeError(self._error())
        out = json.loads(self._buf.value.decode("utf-8", "replace"))
        out["n_tokens"] = n
        return out

    def embed(self, pcm: np.ndarray) -> np.ndarray:
        """needle_embed: encoder output, one row per 80 ms frame, as [frames, width]."""
        pcm, ptr = self._ptr(pcm)
        size = self._lib.needle_embed(None, ptr, int(pcm.size), None, 0)
        if size < 0:
            raise RuntimeError(self._error())
        out = np.zeros(size, dtype=np.float32)
        got = self._lib.needle_embed(None, ptr, int(pcm.size),
                                     out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), size)
        if got != size:
            raise RuntimeError(self._error())
        return out
