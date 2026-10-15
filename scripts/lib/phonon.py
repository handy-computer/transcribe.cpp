# Container decoding adapted from Copyright 2026 Fermion Research,
# licensed under Apache-2.0 (phonon-LICENSE-Apache-2.0.txt).
# Modified for strict validation, pinned conversion and dense F32 output.
"""Conversion-time reader for Fermion's five-value Parakeet container.

Format reference (Apache-2.0): FermionResearch/Phonon-2@7e153e4,
fermion_container.py. No publisher code is executed by the converter.
"""
from __future__ import annotations

import hashlib
import json
import math
import tarfile
import tempfile
from pathlib import Path

import numpy as np

REPO = "FermionResearch/Phonon-2"
REVISION = "7e153e4054c0a10db6d47d6fff5c062d4c77c154"
BASE_REPO = "nvidia/parakeet-tdt-0.6b-v3"
BASE_REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
ARCHIVE = "phonon-2.bps.tar.zst"
ARCHIVE_SHA256 = "98125795b6dda72f5c6eee9ba33d19815df65dcb18b50a357bf9f73c9935309e"
CONTAINER_SHA256 = "4b6bfa3a12cc3c4e0a54f2ab3ec4ca7a842b09e5c7ecfc8e7ca0ac6cc8c11468"


def verify_sha256(path: Path, expected: str) -> None:
    with path.open("rb") as fh:
        actual = hashlib.file_digest(fh, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")


def _decode_record(kind: str, shape: tuple[int, ...], blob: bytes) -> np.ndarray:
    total = math.prod(shape)
    rows = shape[0] if shape else 1
    cols = total // rows
    if kind == "fp16":
        if len(blob) != total * 2:
            raise ValueError("invalid fp16 record length")
        return np.frombuffer(blob, dtype="<f2").reshape(shape).astype(np.float32)
    if kind in ("int6", "int8"):
        payload_size = ((total + 3) // 4) * 3 if kind == "int6" else total
        if len(blob) != payload_size + rows * 2:
            raise ValueError(f"invalid {kind} record length")
        if kind == "int6":
            b = np.frombuffer(blob[:payload_size], dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
            packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
            q = np.stack([(packed >> shift) & 63 for shift in (0, 6, 12, 18)], axis=1)
            q = q.ravel()[:total].astype(np.int32) - 32
        else:
            q = np.frombuffer(blob[:payload_size], dtype=np.int8)
        scales = np.frombuffer(blob[payload_size:], dtype="<f2").astype(np.float32)
        return (q.reshape(rows, cols).astype(np.float32) * scales[:, None]).reshape(shape)
    if kind != "five_value" or len(shape) != 2:
        raise ValueError(f"unsupported container record: {kind}, {shape}")
    row_bytes = (cols + 4) // 5
    sign_bytes = rows * row_bytes
    if len(blob) < sign_bytes + 4 * rows:
        raise ValueError("truncated five_value record")
    packed = np.frombuffer(blob[:sign_bytes], dtype=np.uint8).reshape(rows, row_bytes)
    if np.any(packed > 242):
        raise ValueError("invalid base-3 sign byte")
    codes = np.stack([(packed.astype(np.uint16) // (3 ** k)) % 3 for k in range(5)], axis=2)
    codes = codes.reshape(rows, row_bytes * 5)[:, :cols]
    nonzero = codes != 1
    count = int(nonzero.sum())
    magnitude_bytes = (count + 7) // 8
    off = sign_bytes + magnitude_bytes
    if len(blob) != off + 4 * rows:
        raise ValueError("invalid five_value record length")
    high = np.zeros(shape, dtype=bool)
    high[nonzero] = np.unpackbits(
        np.frombuffer(blob[sign_bytes:off], dtype=np.uint8), bitorder="little"
    )[:count].astype(bool)
    lo = np.frombuffer(blob[off:off + 2 * rows], dtype="<f2")
    hi = np.frombuffer(blob[off + 2 * rows:], dtype="<f2")
    # Magnitudes and their sign are exactly representable in F16; upcast
    # after the publisher's reconstruction, without another quantization.
    return ((codes.astype(np.int8) - 1).astype(np.float16)
            * np.where(high, hi[:, None], lo[:, None])).astype(np.float32)


def read_container(path: Path) -> dict[str, np.ndarray]:
    """Read dense HF-named F32 tensors, rejecting malformed records."""
    tensors = {}
    with Path(path).open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated container header")
        size = int.from_bytes(prefix, "little")
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError("invalid container header length")
        header_bytes = fh.read(size)
        if len(header_bytes) != size:
            raise ValueError("truncated container header")
        header = json.loads(header_bytes)
        if header.get("format") != "fermion-five-value-parakeet-v1":
            raise ValueError("unsupported Fermion container format")
        if not isinstance(header.get("index"), list) or not header["index"]:
            raise ValueError("missing container index")
        remaining = Path(path).stat().st_size - fh.tell()
        for record in header["index"]:
            name, kind = record["n"], record["k"]
            shape, length = record["shape"], record["b"]
            if not isinstance(name, str) or not name or not isinstance(kind, str):
                raise ValueError("invalid tensor identity")
            if (not isinstance(shape, list)
                    or any(type(d) is not int or d <= 0 for d in shape)
                    or type(length) is not int or not 0 <= length <= remaining):
                raise ValueError(f"invalid shape/length for {name}")
            if math.prod(shape) > 2**31:
                raise ValueError(f"oversized tensor: {name}")
            key = name + ".weight" if kind == "five_value" else name
            if key in tensors:
                raise ValueError(f"duplicate tensor: {key}")
            blob = fh.read(length)
            if len(blob) != length:
                raise ValueError(f"truncated tensor: {key}")
            remaining -= length
            value = _decode_record(kind, tuple(shape), blob)
            if key.endswith(".num_batches_tracked"):
                # Publisher stores an unused eval-only BN counter as F16,
                # which can overflow. Match its reference's int64 restore.
                if value.size != 1:
                    raise ValueError(f"invalid BatchNorm counter: {key}")
                value = np.array(int(value.item()) if np.isfinite(value.item()) else 0, dtype=np.int64)
            elif not np.isfinite(value).all():
                raise ValueError(f"non-finite weights: {key}")
            if value.ndim == 2 and key.endswith((".conv.pointwise_conv1.weight", ".conv.pointwise_conv2.weight")):
                value = value[:, :, None]
            tensors[key] = np.ascontiguousarray(value) if value.ndim else value
        if fh.read(1):
            raise ValueError("trailing container bytes")
    return tensors


def source_directory(model_spec: str, revision: str | None = None) -> Path:
    """Resolve a local bundle or extract the pinned public artifact safely."""
    local = Path(model_spec).expanduser()
    if local.is_dir() and (local / "model.fermion").is_file():
        verify_sha256(local / "model.fermion", CONTAINER_SHA256)
        return local
    if model_spec != REPO or revision not in (None, REVISION):
        raise ValueError(f"expected {REPO}@{REVISION} or a local extracted bundle")
    from huggingface_hub import hf_hub_download
    import zstandard

    archive = Path(hf_hub_download(REPO, ARCHIVE, revision=REVISION))
    verify_sha256(archive, ARCHIVE_SHA256)
    root = archive.parent / "phonon-2-extracted"
    root.mkdir(exist_ok=True)
    for filename in ("model.fermion", "config.json"):
        if not (root / filename).is_file():
            break
    else:
        verify_sha256(root / "model.fermion", CONTAINER_SHA256)
        return root
    allowed = {"model.fermion", "config.json", "packed_manifest.json", "bps_manifest.json"}
    seen = set()
    with archive.open("rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                if member.name not in allowed or not member.isfile() or member.name in seen:
                    raise ValueError(f"unexpected archive member: {member.name}")
                seen.add(member.name)
                src = tar.extractfile(member)
                target = root / member.name
                # Concurrent conversions share this cache. Publish only a
                # complete file, using a temporary path owned by this caller.
                out = tempfile.NamedTemporaryFile(dir=root, prefix=f".{member.name}.",
                                                  suffix=".tmp", delete=False)
                temporary = Path(out.name)
                try:
                    with out:
                        import shutil
                        with src:
                            shutil.copyfileobj(src, out)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
    if not {"config.json", "model.fermion"} <= seen:
        raise ValueError("incomplete Phonon bundle")
    verify_sha256(root / "model.fermion", CONTAINER_SHA256)
    return root


def base_directory() -> Path:
    """Fetch only pinned processor/config files, never the teacher weights."""
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(
        BASE_REPO, revision=BASE_REVISION,
        allow_patterns=["*.json", "*.model", "*.txt"],
    ))
