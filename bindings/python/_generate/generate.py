#!/usr/bin/env python3
"""Generate the low-level ctypes FFI layer from the public C headers.

This parses ``include/transcribe/extensions.h`` with libclang and emits
``src/transcribe_cpp/_generated.py`` — ctypes Structures (field-for-field from
the C structs, with the offsets the C compiler computed), enum constants,
function prototypes, and per-struct ABI metadata. The generated module is
committed; CI regenerates and runs ``git diff --exit-code`` so a header change
that is not reflected in the binding fails the build.

Because libclang works from the parsed AST (not header text), the output is
semantic: a comment-only or whitespace header edit produces no diff, while any
real ABI change (a field, a type, an enum value, a signature) does.

Usage:
    uv run --no-project --with 'libclang==18.1.1' _generate/generate.py            # write
    uv run --no-project --with 'libclang==18.1.1' _generate/generate.py --check    # CI gate
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import clang.cindex as ci
from clang.cindex import CursorKind, TypeKind

REPO = Path(__file__).resolve().parents[3]
INCLUDE = REPO / "include"
HEADER = INCLUDE / "transcribe" / "extensions.h"
OUTPUT = REPO / "bindings" / "python" / "src" / "transcribe_cpp" / "_generated.py"
# The binding-neutral home of the ABI digest (PUBLIC_HEADER_HASH). This
# generator is the oracle that computes it, but the checked-in file lives with
# the headers it digests so every consumer — CMake's provider-contract stamp,
# the Rust/TS/Swift drift gates — reads it WITHOUT depending on the Python
# binding's _generated.py. Same drift discipline as _generated.py: --check
# fails when either file is stale.
ABIHASH = INCLUDE / "transcribe.abihash"

# Fixed-width / size typedefs map to fixed-width ctypes so the generated layout
# is correct on every target (never c_long, whose width differs LP64 vs LLP64).
_SPELLING_INT = {
    "int8_t": "c_int8", "uint8_t": "c_uint8",
    "int16_t": "c_int16", "uint16_t": "c_uint16",
    "int32_t": "c_int32", "uint32_t": "c_uint32",
    "int64_t": "c_int64", "uint64_t": "c_uint64",
    "size_t": "c_size_t", "ssize_t": "c_ssize_t",
    "intptr_t": "c_ssize_t", "uintptr_t": "c_size_t",
}
_KIND_SCALAR = {
    TypeKind.VOID: None,
    TypeKind.BOOL: "c_bool",
    TypeKind.CHAR_S: "c_char", TypeKind.SCHAR: "c_char",
    TypeKind.CHAR_U: "c_ubyte", TypeKind.UCHAR: "c_ubyte",
    TypeKind.SHORT: "c_short", TypeKind.USHORT: "c_ushort",
    TypeKind.INT: "c_int", TypeKind.UINT: "c_uint",
    TypeKind.LONG: "c_long", TypeKind.ULONG: "c_ulong",
    TypeKind.LONGLONG: "c_longlong", TypeKind.ULONGLONG: "c_ulonglong",
    TypeKind.FLOAT: "c_float", TypeKind.DOUBLE: "c_double",
}


def clang_args() -> list[str]:
    args = ["-x", "c", "-std=c11", f"-I{INCLUDE}"]
    # Freestanding headers (stdbool/stdint/stddef) live in the compiler's
    # resource dir; the bundled libclang does not ship them.
    try:
        resdir = subprocess.check_output(
            ["xcrun", "clang", "-print-resource-dir"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        args.append(f"-isystem{resdir}/include")
        sdk = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        args.append(f"-isysroot{sdk}")
    except Exception:
        try:
            resdir = subprocess.check_output(
                ["clang", "-print-resource-dir"], text=True
            ).strip()
            args.append(f"-isystem{resdir}/include")
        except Exception:
            pass
    return args


def in_headers(cursor) -> bool:
    f = cursor.location.file
    return f is not None and str(INCLUDE) in str(f.name)


def int_macro_value(value_tokens):
    """Parse an object-like macro's value as an int, or None if it isn't one.

    Captures integer constants (EXT kind FourCCs, version components); skips
    string, attribute, and float/expression macros, which don't parse as int.
    """
    s = "".join(value_tokens).rstrip("uUlL")
    try:
        return int(s, 0)
    except ValueError:
        return None


class Surface:
    def __init__(self):
        self.enum_constants: list[tuple[str, int]] = []
        self.macros: list[tuple[str, int]] = []   # name -> int value
        self.structs: dict = {}        # name -> cursor
        self.struct_order: list[str] = []
        self.functions: dict = {}      # name -> cursor
        self.abi_ids: dict = {}        # struct name -> transcribe_abi_struct id


def collect(tu) -> Surface:
    s = Surface()
    seen_enum = set()
    seen_macro = set()
    for c in tu.cursor.walk_preorder():
        if not in_headers(c):
            continue
        if c.kind == CursorKind.ENUM_DECL and c.is_definition():
            for e in c.get_children():
                if e.kind == CursorKind.ENUM_CONSTANT_DECL and e.spelling not in seen_enum:
                    seen_enum.add(e.spelling)
                    s.enum_constants.append((e.spelling, e.enum_value))
        elif c.kind == CursorKind.MACRO_DEFINITION and c.spelling.startswith("TRANSCRIBE_"):
            if c.spelling in seen_macro:
                continue
            tokens = [t.spelling for t in c.get_tokens()]
            if len(tokens) < 2:
                continue  # bare guard macro, no value
            value = int_macro_value(tokens[1:])
            if value is not None:
                seen_macro.add(c.spelling)
                s.macros.append((c.spelling, value))
        elif c.kind == CursorKind.STRUCT_DECL and c.is_definition() and c.spelling:
            if c.spelling not in s.structs:
                s.structs[c.spelling] = c
                s.struct_order.append(c.spelling)
        elif c.kind == CursorKind.FUNCTION_DECL and c.spelling not in s.functions:
            s.functions[c.spelling] = c

    # Map transcribe_abi_struct enum members to struct names: the member
    # TRANSCRIBE_ABI_RUN_PARAMS corresponds to struct transcribe_run_params.
    for name, value in s.enum_constants:
        if name.startswith("TRANSCRIBE_ABI_"):
            struct = "transcribe_" + name[len("TRANSCRIBE_ABI_"):].lower()
            if struct in s.structs:
                s.abi_ids[struct] = value
    s.macros.sort(key=lambda kv: kv[0])  # name order for deterministic output
    return s


def map_type(t, structs: dict) -> str:
    """Return a ctypes expression (``_c.``-prefixed) for a clang Type."""
    spelling = t.spelling.replace("const ", "").replace("volatile ", "").strip()
    k = t.kind

    if k == TypeKind.POINTER:
        pointee = t.get_pointee()
        cp = pointee.get_canonical()
        if cp.kind == TypeKind.FUNCTIONPROTO:
            ret = map_type(cp.get_result(), structs) or "None"
            args = [map_type(a, structs) for a in cp.argument_types()]
            return f"_c.CFUNCTYPE({', '.join([ret] + args)})"
        if cp.kind in (TypeKind.CHAR_S, TypeKind.CHAR_U, TypeKind.SCHAR, TypeKind.UCHAR):
            return "_c.c_char_p"
        if cp.kind == TypeKind.VOID:
            return "_c.c_void_p"
        if cp.kind == TypeKind.RECORD:
            name = cp.get_declaration().spelling
            return f"_c.POINTER({name})" if name in structs else "_c.c_void_p"
        inner = map_type(pointee, structs)
        return f"_c.POINTER({inner})" if inner and inner != "None" else "_c.c_void_p"

    if spelling in _SPELLING_INT:
        return "_c." + _SPELLING_INT[spelling]
    if k == TypeKind.ENUM:
        return "_c.c_int"
    if k in (TypeKind.TYPEDEF, TypeKind.ELABORATED):
        return map_type(t.get_canonical(), structs)
    if k == TypeKind.RECORD:
        return t.get_declaration().spelling
    if k == TypeKind.CONSTANTARRAY:
        return f"({map_type(t.element_type, structs)} * {t.element_count})"
    if k in _KIND_SCALAR:
        v = _KIND_SCALAR[k]
        return "_c." + v if v else "None"
    raise SystemExit(f"unmapped type: {t.spelling!r} kind={k}")


def struct_fields(cursor):
    return [f for f in cursor.get_children() if f.kind == CursorKind.FIELD_DECL]


def order_by_value_deps(structs: dict, order: list[str]) -> list[str]:
    """Topologically order structs so a by-value member is defined first."""
    deps = {}
    for name in order:
        d = set()
        for f in struct_fields(structs[name]):
            ft = f.type.get_canonical()
            if ft.kind == TypeKind.RECORD:
                dep = ft.get_declaration().spelling
                if dep in structs:
                    d.add(dep)
            elif ft.kind == TypeKind.CONSTANTARRAY:
                et = ft.element_type.get_canonical()
                if et.kind == TypeKind.RECORD and et.get_declaration().spelling in structs:
                    d.add(et.get_declaration().spelling)
        deps[name] = d

    out, placed = [], set()
    while len(out) < len(order):
        progress = False
        for name in order:
            if name in placed:
                continue
            if deps[name] <= placed:
                out.append(name)
                placed.add(name)
                progress = True
        if not progress:  # cycle (shouldn't happen for this API)
            raise SystemExit(f"by-value struct cycle among {set(order) - placed}")
    return out


def render(s: Surface, libclang_version: str) -> tuple[str, str]:
    # The structural payload (everything below the imports) is built first so a
    # stable digest can be hashed over it and emitted as PUBLIC_HEADER_HASH. That
    # digest is the coarse cross-language ABI tag a native provider echoes back
    # at load time: it changes for any real ABI change (a field, type, enum
    # value, signature, layout) and stays put for a comment-only header edit,
    # exactly like the generator's own drift gate. It does NOT cover the libclang
    # version line (that lives in the docstring, above the hashed body).
    body: list[str] = []
    b = body.append

    b("# === enum constants ===")
    for name, value in s.enum_constants:
        b(f"{name} = {value}")
    b("")

    b("# === macro constants (integer object-like macros) ===")
    for name, value in s.macros:
        b(f"{name} = {value}")
    b("")

    struct_names = set(s.structs)
    b("# === structs ===")
    for name in s.struct_order:
        b(f"class {name}(_c.Structure):")
        b("    pass")
    b("")

    layout = {}
    for name in order_by_value_deps(s.structs, s.struct_order):
        cursor = s.structs[name]
        fields, offsets = [], {}
        for f in struct_fields(cursor):
            fields.append((f.spelling, map_type(f.type, struct_names)))
            offsets[f.spelling] = cursor.type.get_offset(f.spelling) // 8
        joined = ", ".join(f'("{fn}", {ct})' for fn, ct in fields)
        b(f"{name}._fields_ = [{joined}]")
        layout[name] = {
            "size": cursor.type.get_size(),
            "align": cursor.type.get_align(),
            "offsets": offsets,
        }
    b("")

    b("# === ABI metadata ===")
    b("# transcribe_abi_struct id per struct (for the native size/align check).")
    b("ABI_STRUCT_IDS = {")
    for name in s.struct_order:
        if name in s.abi_ids:
            b(f"    {name!r}: {s.abi_ids[name]},")
    b("}")
    b("")
    b("# C-compiler layout captured at generation (for offset self-check).")
    b("STRUCT_LAYOUT = {")
    for name in s.struct_order:
        lo = layout[name]
        b(f"    {name!r}: {{'size': {lo['size']}, 'align': {lo['align']}, "
          f"'offsets': {lo['offsets']!r}}},")
    b("}")
    b("")

    b("# === function prototypes ===")
    b("def configure(lib):")
    b('    """Stamp restype/argtypes onto a loaded CDLL."""')
    for name in sorted(s.functions):
        fn = s.functions[name]
        restype = map_type(fn.result_type, struct_names)
        argtypes = [map_type(a.type, struct_names) for a in fn.get_arguments()]
        b(f"    lib.{name}.restype = {restype}")
        b(f"    lib.{name}.argtypes = [{', '.join(argtypes)}]")
    b("")

    digest = hashlib.sha256("\n".join(body).encode("utf-8")).hexdigest()[:16]

    out = []
    w = out.append
    w('"""Low-level ctypes FFI for transcribe.cpp — AUTOGENERATED. DO NOT EDIT.')
    w("")
    w("Regenerate with:")
    w("    uv run --no-project --with 'libclang==18.1.1' \\")
    w("        bindings/python/_generate/generate.py")
    w("")
    w("Source: include/transcribe/extensions.h")
    w(f"libclang: {libclang_version}")
    w('"""')
    w("")
    w("import ctypes as _c")
    w("")
    w("# Stable digest of the ABI surface below (structs, enums, macros, layout,")
    w("# prototypes). A native provider package echoes this back so the API")
    w("# package can reject an ABI-mismatched provider before dlopen.")
    w(f'PUBLIC_HEADER_HASH = "{digest}"')
    w("")
    out.extend(body)
    return "\n".join(out), digest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed file is out of date")
    args = ap.parse_args()

    try:
        libclang_version = __import__("importlib.metadata", fromlist=["version"]).version(
            "libclang"
        )
    except Exception:
        libclang_version = "unknown"

    tu = ci.Index.create().parse(
        str(HEADER), args=clang_args(),
        options=ci.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
    )
    errors = [d for d in tu.diagnostics if d.severity >= ci.Diagnostic.Error]
    if errors:
        for d in errors:
            print(f"clang error: {d.spelling} at {d.location}", file=sys.stderr)
        return 2

    text, digest = render(collect(tu), libclang_version)
    hash_text = digest + "\n"

    if args.check:
        stale = []
        current = OUTPUT.read_text() if OUTPUT.exists() else ""
        if current != text:
            stale.append(str(OUTPUT))
        current_hash = ABIHASH.read_text() if ABIHASH.exists() else ""
        if current_hash != hash_text:
            stale.append(str(ABIHASH))
        if stale:
            for path in stale:
                print(f"{path} is out of date — regenerate with "
                      "_generate/generate.py", file=sys.stderr)
            return 1
        print(f"{OUTPUT.name} and {ABIHASH.name} are up to date")
        return 0

    OUTPUT.write_text(text)
    ABIHASH.write_text(hash_text)
    print(f"wrote {OUTPUT}")
    print(f"wrote {ABIHASH} ({digest})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
