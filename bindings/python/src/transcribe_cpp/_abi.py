"""Load-time ABI verification of the generated ctypes layer.

Two independent checks, both required before the high-level API touches a struct:

1. **Offset self-check** — every generated Structure's ctypes size, alignment,
   and per-field offsets must equal the C-compiler layout libclang captured at
   generation time (``_generated.STRUCT_LAYOUT``). This catches a wrong type in
   the generated module on the running platform.

2. **Native agreement** — for every struct with a ``transcribe_abi_struct`` id,
   ctypes size/alignment must equal what the *loaded* native library reports
   (``transcribe_abi_struct_size``/``_align``). This catches a committed module
   that is stale relative to the actual ``.dylib``/``.so`` in this process.

A ctypes layout mismatch corrupts memory silently; turning it into an
ImportError here is the safety net behind the generated FFI.
"""

from __future__ import annotations

import ctypes

from . import _generated
from .errors import AbiError


def verify_layouts(lib: ctypes.CDLL) -> None:
    """Raise AbiError if the generated layer disagrees with itself or the lib."""
    mismatches: list[str] = []

    # 1. ctypes layout vs the captured C-compiler layout.
    for name, layout in _generated.STRUCT_LAYOUT.items():
        cls = getattr(_generated, name)
        if ctypes.sizeof(cls) != layout["size"]:
            mismatches.append(
                f"{name}: size {ctypes.sizeof(cls)} (ctypes) != {layout['size']} (generated)"
            )
        if ctypes.alignment(cls) != layout["align"]:
            mismatches.append(
                f"{name}: alignment {ctypes.alignment(cls)} (ctypes) != "
                f"{layout['align']} (generated)"
            )
        for field, offset in layout["offsets"].items():
            actual = getattr(cls, field).offset
            if actual != offset:
                mismatches.append(
                    f"{name}.{field}: offset {actual} (ctypes) != {offset} (generated)"
                )

    # 2. ctypes size/alignment vs the loaded native library.
    for name, abi_id in _generated.ABI_STRUCT_IDS.items():
        cls = getattr(_generated, name)
        native_size = lib.transcribe_abi_struct_size(abi_id)
        native_align = lib.transcribe_abi_struct_align(abi_id)
        if native_size == 0:
            mismatches.append(
                f"{name}: native library does not know abi id {abi_id} "
                "(library older than this binding)"
            )
            continue
        if ctypes.sizeof(cls) != native_size:
            mismatches.append(
                f"{name}: size {ctypes.sizeof(cls)} (binding) != {native_size} (native)"
            )
        if ctypes.alignment(cls) != native_align:
            mismatches.append(
                f"{name}: alignment {ctypes.alignment(cls)} (binding) != "
                f"{native_align} (native)"
            )

    if mismatches:
        raise AbiError(
            "transcribe_cpp ABI layout check failed — the generated binding and "
            "native library disagree on struct layout. This is a version/build "
            "skew or a stale generated layer; do not run against this library.\n  "
            + "\n  ".join(mismatches)
        )
