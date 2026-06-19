#!/usr/bin/env python3
"""Fail if the library version drifts across every place it is duplicated.

The native library version is defined once, in
``include/transcribe.h`` (``TRANSCRIBE_VERSION_{MAJOR,MINOR,PATCH}``); CMake
parses it from there. Every binding repeats it — in package manifests, lockfiles,
the Python ``__version__``, the cross-package dependency pins, and the Swift
``compiledVersion`` literal. The import-time gate enforces base-version match
against the *loaded* library at runtime; this script is the static, build-time
counterpart so a forgotten bump fails CI before anything is published.

This covers every §1b spot in ``notes/releasing.md`` — including the ones that
used to be §1c blind spots: the ``transcribe-cpp-sys`` dependency *pin*, both
``Cargo.lock`` entries, both ``package-lock.json`` spots, and Swift
``compiledVersion``. (Lockfile *internal* consistency — a stale lock silently
rewritten by an unlocked command — is still the job of the locked-command
checks, ``cargo metadata --locked`` / ``npm ci``, run in release-preflight.)

Comparison is on the PEP 440 *release segment* (``MAJOR.MINOR.PATCH``): the
header is always a clean triple, while a package side may legitimately carry a
``.postN`` packaging suffix that must still be accepted.

    uv run --no-project bindings/python/_generate/check_version_sync.py

Exit 0 when all agree on the base version; 1 on drift; 2 if a version could not
be located (treated as a hard error, not a pass).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HEADER = REPO / "include" / "transcribe.h"
PYPROJECT = REPO / "bindings" / "python" / "pyproject.toml"
INIT = REPO / "bindings" / "python" / "src" / "transcribe_cpp" / "__init__.py"
TS_PACKAGE_JSON = REPO / "bindings" / "typescript" / "package.json"
RUST_SAFE_CARGO = REPO / "bindings" / "rust" / "transcribe-cpp" / "Cargo.toml"
CARGO_LOCK = REPO / "Cargo.lock"
PACKAGE_LOCK = REPO / "bindings" / "typescript" / "package-lock.json"
SWIFT_SOURCE = REPO / "bindings" / "swift" / "Sources" / "TranscribeCpp" / "TranscribeCpp.swift"

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


def cargo_sys_pin(text: str) -> str | None:
    # The safe crate's dependency *pin* on the sys crate (a different field from
    # its own [package].version, which cargo_version() returns):
    #   transcribe-cpp-sys = { version = "X.Y.Z", path = "../../..", ... }
    m = re.search(
        r'transcribe-cpp-sys\s*=\s*\{[^}]*?\bversion\s*=\s*"([^"]+)"', text
    )
    return m.group(1) if m else None


def cargo_lock_versions(text: str) -> "dict[str, str | None]":
    # The two workspace crates pinned in Cargo.lock. cargo writes name then
    # version on consecutive lines within each [[package]] block; the closing
    # quote in the name match keeps "transcribe-cpp" from also matching
    # "transcribe-cpp-sys".
    out: dict[str, str | None] = {}
    for name in ("transcribe-cpp", "transcribe-cpp-sys"):
        m = re.search(rf'name = "{re.escape(name)}"\nversion = "([^"]+)"', text)
        out[f"Cargo.lock ({name})"] = m.group(1) if m else None
    return out


def package_lock_versions(text: str) -> "dict[str, str | None]":
    # The two spots npm keeps a root version in the lockfile: top-level
    # `.version` and `.packages[""].version` (the root package's own node).
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {"package-lock.json (root)": None, 'package-lock.json (packages[""])': None}
    return {
        "package-lock.json (root)": data.get("version"),
        'package-lock.json (packages[""])': (data.get("packages") or {}).get("", {}).get("version"),
    }


def swift_compiled_version(text: str) -> str | None:
    # The hand-maintained Swift literal `compiledVersion = "X.Y.Z"` that the
    # SwiftPM load gate (Transcribe.ensureCompatible) compares against the
    # linked library. (The Swift ABI pin is checked separately by
    # swift_abihash_check.py against include/transcribe.abihash.)
    m = re.search(r'compiledVersion\s*=\s*"([^"]+)"', text)
    return m.group(1) if m else None


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

    # Formerly §1c blind spots — now part of the equality set (releasing.md §8
    # P0 #2 slice B). Each file must exist; a missing one is a hard error below.
    sources["Cargo.toml (sys dep pin)"] = (
        cargo_sys_pin(RUST_SAFE_CARGO.read_text()) if RUST_SAFE_CARGO.exists() else None
    )
    if CARGO_LOCK.exists():
        sources.update(cargo_lock_versions(CARGO_LOCK.read_text()))
    else:
        sources["Cargo.lock"] = None
    if PACKAGE_LOCK.exists():
        sources.update(package_lock_versions(PACKAGE_LOCK.read_text()))
    else:
        sources["package-lock.json"] = None
    sources["TranscribeCpp.swift (compiledVersion)"] = (
        swift_compiled_version(SWIFT_SOURCE.read_text()) if SWIFT_SOURCE.exists() else None
    )

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
