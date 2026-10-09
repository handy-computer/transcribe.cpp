"""Published Silero .bin reference: expand stored weights into upstream JIT.

Independent of the native loader. The binaries and upstream JIT are pinned;
all 15 payloads must exactly equal upstream weights rounded to the wire type.
No STFT regeneration or restoration of precision lost to F16 storage.
"""
from __future__ import annotations

import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

HF_REVISION = "9ffd54a1e1ee413ddf265af9913beaf518d1639b"
BIN_SHA256 = {
    (5, 1, 2): "29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf",
    (6, 2, 0): "2aa269b785eeb53a82983a20501ddf7c1d9c48e33ab63a41391ac6c9f7fb6987",
}
V5_JIT_SHA256 = "85c48e1f0ecb604e5d2a268f3ccfb912d4f7e935acdc86af5a3fc5b0aea7b29a"
V6_JIT_SHA256 = "e1122837f4154c511485fe0b9c64455f7b929c96fbb8d79fbdb336383ebd3720"
V5_JIT_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.jit"


def checked_bytes(path: Path, sha256: str) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path}: SHA256 mismatch (expected {sha256})")
    return data


def load_stored_model(path: Path, v5_jit: Path | None = None):
    import importlib.resources
    import silero_vad

    data = path.read_bytes()
    offset = 0

    def take(n):
        nonlocal offset
        if n < 0 or offset + n > len(data):
            raise ValueError("truncated binary")
        value = data[offset:offset + n]
        offset += n
        return value

    def integer():
        return struct.unpack("<i", take(4))[0]

    if integer() != 0x67676d6c or integer() != 10 or take(10) != b"silero-16k":
        raise ValueError("not a published silero-16k binary")
    version = tuple(integer() for _ in range(3))
    if version not in BIN_SHA256:
        raise ValueError(f"unsupported binary version {version}")
    bin_sha = hashlib.sha256(data).hexdigest()
    if bin_sha != BIN_SHA256[version]:
        raise ValueError(f"{path}: not the pinned official file for {version}")
    geometry = [integer() for _ in range(19)]
    if geometry != [512, 64, 4, 129, 128, 3, 128, 64, 3, 64, 64, 3, 64, 128, 3, 128, 128, 128, 1]:
        raise ValueError("unexpected binary geometry")

    if version == (5, 1, 2):
        if v5_jit is None:
            v5_jit = Path(__file__).resolve().parents[2] / "build/reference/silero_vad/silero-v5.1.2.jit"
            if not v5_jit.exists():
                v5_jit.parent.mkdir(parents=True, exist_ok=True)
                # Explicit reference command only; runtime never downloads weights.
                downloaded = urllib.request.urlopen(V5_JIT_URL).read()
                if hashlib.sha256(downloaded).hexdigest() != V5_JIT_SHA256:
                    raise ValueError("downloaded v5 JIT SHA256 mismatch")
                v5_jit.write_bytes(downloaded)
        checked_bytes(v5_jit, V5_JIT_SHA256)
        model = torch.jit.load(str(v5_jit), map_location="cpu")
        jit_sha = V5_JIT_SHA256
    else:
        jit = Path(str(importlib.resources.files("silero_vad.data").joinpath("silero_vad.jit")))
        checked_bytes(jit, V6_JIT_SHA256)
        model = silero_vad.load_silero_vad()
        jit_sha = V6_JIT_SHA256

    sub = model._model
    if (sub.stft.filter_length, sub.stft.hop_length, tuple(sub.stft.padding.padding)) != (256, 128, (0, 64)):
        raise ValueError("upstream STFT geometry/padding changed")
    for i, stride in enumerate((1, 2, 2, 1)):
        conv = getattr(sub.encoder, str(i)).reparam_conv
        if tuple(conv.stride) != (stride,) or tuple(conv.padding) != (1,):
            raise ValueError("upstream encoder geometry changed")

    state = model.state_dict()
    expected = {name for name in state if name.startswith("_model.")}
    seen = set()
    types = {}
    while offset < len(data):
        rank, name_len, kind = integer(), integer(), integer()
        if rank not in range(4) or name_len not in range(1, 129) or kind not in (0, 1):
            raise ValueError("invalid tensor header")
        ne = [integer() for _ in range(rank)]
        name = take(name_len).decode("ascii")
        if name not in expected or name in seen:
            raise ValueError(f"unexpected/duplicate tensor {name}")
        original = state[name].numpy()
        shape = original.shape if name.endswith("forward_basis_buffer") else original.squeeze().shape
        if ne != list(reversed(shape)):
            raise ValueError(f"{name}: wrong wire shape {ne}")
        dtype = np.dtype("<f4" if kind == 0 else "<f2")
        count = int(original.size)
        stored = np.frombuffer(take(count * dtype.itemsize), dtype=dtype)
        if not np.array_equal(stored, original.reshape(-1).astype(dtype)):
            raise ValueError(f"{name}: binary payload != rounded upstream weights")
        state[name] = torch.from_numpy(stored.astype(np.float32).copy()).reshape(original.shape)
        seen.add(name)
        types[name] = "f32" if kind == 0 else "f16"
    if seen != expected or len(seen) != 15:
        raise ValueError("missing tensors")
    model.load_state_dict(state)
    model.reset_states()
    return model, {
        "binary_version": ".".join(map(str, version)),
        "binary_sha256": bin_sha,
        "binary_hf_repo": "ggml-org/whisper-vad",
        "binary_hf_revision": HF_REVISION,
        "upstream_jit_sha256": jit_sha,
        "stored_tensor_types": types,
        "weight_loading": "stored F16/F32 values expanded to F32, exact rounded upstream match",
    }
