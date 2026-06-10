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


def native_pin_version(text: str) -> str | None:
    # The hard dependency on the default provider is the pre-1.0 base-version
    # contract at resolver level: transcribe-cpp-native==X.Y.Z.* — X.Y.Z must
    # be the same base as everything else. (The provider package itself can't
    # drift: its version is parsed from the header at build time.)
    m = re.search(r'"transcribe-cpp-native\s*==\s*([0-9.]+?)\.\*"', text)
    return m.group(1) if m else None


def main() -> int:
    pyproject_text = PYPROJECT.read_text()
    sources = {
        "include/transcribe.h": header_version(HEADER.read_text()),
        "pyproject.toml": pyproject_version(pyproject_text),
        "pyproject.toml (native pin)": native_pin_version(pyproject_text),
        "__init__.__version__": init_version(INIT.read_text()),
    }

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
