# FFI generator

Generates `src/transcribe_cpp/_generated.py` — the low-level ctypes layer — from
`include/transcribe/extensions.h` using libclang. The generated module is
**committed**; it is never hand-edited.

## Regenerate

```bash
cd bindings/python
uv run --no-project --with 'libclang==18.1.1' _generate/generate.py
```

Run this whenever the public C headers change. libclang is pinned so the output
is deterministic across machines; the freestanding headers (`stdbool.h`, …) come
from the host compiler's resource dir, discovered via `clang -print-resource-dir`
(macOS: `xcrun`).

## CI gate

```bash
uv run --no-project --with 'libclang==18.1.1' _generate/generate.py --check
```

Exit non-zero if the committed `_generated.py` is out of date. Because the
generator works from the parsed AST, the check is **semantic**: a comment- or
whitespace-only header edit produces no diff, while any real ABI change (a field,
type, enum value, or function signature) does — and then fails CI until the
binding is regenerated.

## What it emits

- ctypes `Structure` for every public struct, field-for-field.
- Enum values as module constants.
- `configure(lib)` — `restype`/`argtypes` for every public function.
- `ABI_STRUCT_IDS` and `STRUCT_LAYOUT` (sizes/aligns/offsets), used by
  `_abi.verify_layouts()` to check the layer against itself and the loaded
  native library at import.
