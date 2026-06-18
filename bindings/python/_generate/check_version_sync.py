#!/usr/bin/env python3
"""Fail if the library version drifts across its three sources of truth.

The native library version is defined once, in
``include/transcribe.h`` (``TRANSCRIBE_VERSION_{MAJOR,MINOR,PATCH}``); CMake
parses it from there. The Python package repeats it in two more places —
``bindings/python/pyproject.toml`` (``project.version``) and the
``__version__`` in ``bindings/python/src/transcribe_cpp/__init__.py``. The
import-time gate enforces base-version match against the *loaded* library at
runtime; this script is the static, build-time counterpart so a forgotten bump
fails CI before anything is published.

Comparison is on the PEP 440 *release segment* (``MAJOR.MINOR.PATCH``): the
header is always a clean triple, while the Python side may legitimately carry a
``.postN`` packaging suffix that must still be accepted.

    uv run --no-project bindings/python/_generate/check_version_sync.py

Exit 0 when all three agree on the base version; 1 on drift; 2 if a version
could not be located (treated as a hard error, not a pass).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HEADER = REPO / "include" / "transcribe.h"
PYPROJECT = REPO / "bindings" / "python" / "pyproject.toml"
INIT = REPO / "bindings" / "python" / "src" / "transcribe_cpp" / "__init__.py"
TS_PACKAGE_JSON = REPO / "bindings" / "typescript" / "package.json"

# Binding package manifests (requirements doc §2: every manifest is derived
# from or gated against the header). Gated by the `active` flag: a 0.0.0
# name-reservation placeholder is NOT version-locked — flip its entry to True
# in the PR that lands the real binding. Inactive manifests are still parsed
# (file must exist and carry a readable version) so the mechanism itself
# stays exercised. Package.swift has no entry: SwiftPM versions via git tags,
# so its gate is the tag itself (release-workflow concern, not this script).
BINDING_MANIFESTS = [
    # (relative path, extractor name, active)
    # The Rust crates are real (0.0.1), so they're version-locked. The sys
    # crate's manifest is the repo-root Cargo.toml (it carries the whole C++
    # tree); the safe wrapper is the sibling member at
    # bindings/rust/transcribe-cpp/.
    ("Cargo.toml", "cargo", True),
    ("bindings/rust/transcribe-cpp/Cargo.toml", "cargo", True),
    ("bindings/typescript/package.json", "npm", True),
]


def base_version(version: str) -> str:
    """The leading dotted-numeric release segment (suffix stripped)."""
    m = re.match(r"\d+(?:\.\d+)*", version.strip())
    return m.group(0) if m else version.strip()


def header_version(text: str) -> str | None:
    parts = []
    for component in ("MAJOR", "MINOR", "PATCH"):
        m = re.search(rf"define\s+TRANSCRIBE_VERSION_{component}\s+(\d+)", text)
        if not m:
            return None
        parts.append(m.group(1))
    return ".".join(parts)


def pyproject_version(text: str) -> str | None:
    # project.version is a top-level string in [project]; match it directly
    # rather than pulling in a TOML parser (tomllib is 3.11+).
    m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


def init_version(text: str) -> str | None:
    m = re.search(r'(?m)^__version__\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


def cargo_version(text: str) -> str | None:
    # First `version = "..."` in the file: [package] leads a Cargo.toml by
    # convention, and dependency tables spell it `name = { version = ... }`.
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


def npm_version(text: str) -> str | None:
    m = re.search(r'"version"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None


_BINDING_EXTRACTORS = {"cargo": cargo_version, "npm": npm_version}


def native_pin_versions(text: str) -> "dict[str, str | None]":
    # Every native-provider pin (the hard dependency AND accelerator extras)
    # is the pre-1.0 base-version contract at resolver level:
    # transcribe-cpp-native[-suffix]==X.Y.Z.* — X.Y.Z must be the same base
    # as everything else. (The provider packages themselves can't drift:
    # their versions are parsed from the header at build time.)
    pins = re.findall(
        r'"(transcribe-cpp-native(?:-[a-z0-9]+)*)\s*==\s*([0-9.]+?)\.\*"', text
    )
    if not pins:
        return {"pyproject.toml (native pin)": None}
    return {f"pyproject.toml ({name} pin)": version for name, version in pins}


def npm_optional_pins(text: str) -> "dict[str, str | None]":
    # The npm analog of native_pin_versions: the API package
    # (bindings/typescript/package.json) pins each @transcribe-cpp/<platform>
    # provider in optionalDependencies at an exact version. Pre-1.0 they must
    # share the base version with everything else, exactly as the Python native
    # pins do. The release job (ts-release) re-syncs them to the published
    # version; this is the static counterpart so a forgotten bump fails CI.
    block = re.search(r'"optionalDependencies"\s*:\s*\{([^}]*)\}', text, re.S)
    pins = re.findall(r'"(@transcribe-cpp/[^"]+)"\s*:\s*"([^"]+)"', block.group(1)) if block else []
    if not pins:
        return {"package.json (optionalDependencies)": None}
    return {f"package.json ({name} pin)": version for name, version in pins}


def main() -> int:
    pyproject_text = PYPROJECT.read_text()
    sources = {
        "include/transcribe.h": header_version(HEADER.read_text()),
        "pyproject.toml": pyproject_version(pyproject_text),
        "__init__.__version__": init_version(INIT.read_text()),
    }
    sources.update(native_pin_versions(pyproject_text))
    if TS_PACKAGE_JSON.exists():
        sources.update(npm_optional_pins(TS_PACKAGE_JSON.read_text()))

    # Binding manifests: active ones join the equality set; inactive ones
    # must merely exist and parse (placeholder versions are reported, not
    # compared).
    inactive: dict[str, str] = {}
    for rel, kind, active in BINDING_MANIFESTS:
        path = REPO / rel
        version = (
            _BINDING_EXTRACTORS[kind](path.read_text()) if path.exists() else None
        )
        if active:
            sources[rel] = version
        elif version is None:
            sources[rel] = None  # missing/unparseable is an error either way
        else:
            inactive[rel] = version
    if inactive:
        detail = ", ".join(f"{name}={v}" for name, v in inactive.items())
        print(f"inactive binding manifests (parsed, not compared): {detail}")

    missing = [name for name, v in sources.items() if v is None]
    if missing:
        for name in missing:
            print(f"error: could not locate the version in {name}", file=sys.stderr)
        return 2

    bases = {name: base_version(v) for name, v in sources.items()}  # type: ignore[arg-type]
    distinct = set(bases.values())
    if len(distinct) != 1:
        print("version drift across sources (base MAJOR.MINOR.PATCH must agree):",
              file=sys.stderr)
        for name, v in sources.items():
            print(f"  {name}: {v}  (base {bases[name]})", file=sys.stderr)
        return 1

    base = distinct.pop()
    detail = ", ".join(f"{name}={v}" for name, v in sources.items())
    print(f"version sync ok: base {base} ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
