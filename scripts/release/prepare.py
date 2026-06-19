#!/usr/bin/env python3
"""Author or verify a transcribe.cpp version bump — one command, every spot.

The version is authored once in ``include/transcribe.h``
(``TRANSCRIBE_VERSION_{MAJOR,MINOR,PATCH}``) and physically duplicated across
~14 files (see ``notes/releasing.md`` §1). This script is the executable form of
that bookkeeping and of the pre-tag checklist:

    uv run --no-project scripts/release/prepare.py X.Y.Z    # write the bump into
        # every §1b spot, regenerate the FFI, sync the lockfiles
    uv run --no-project scripts/release/prepare.py --check  # umbrella gate

``--check`` is the single source of truth for "is this tree release-consistent?"
— it runs the hard-tier drift/version assertions (version-sync across every
spot, the Python/TS + Rust FFI drift gates, the Swift ABI pin, and lockfile
freshness). ``release-preflight`` in ``publish.yml`` calls ``--check`` rather
than re-listing the individual gates.

Post-decoupling (releasing.md §8 P0 #1) a version-only bump produces NO change
to any generated file or the abihash; this script regenerates them anyway and
loudly surfaces a *real* ABI change (a moved ``include/transcribe.abihash``),
which additionally requires a conscious bump of the Swift ``pinnedHeaderHash``.

Exit 0 on success; non-zero if a write target could not be located or any
``--check`` gate failed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# --- §1b spots (hand-edited; kept in lockstep with the header) ---------------
HEADER = REPO / "include" / "transcribe.h"
ROOT_CARGO = REPO / "Cargo.toml"
SAFE_CARGO = REPO / "bindings" / "rust" / "transcribe-cpp" / "Cargo.toml"
CARGO_LOCK = REPO / "Cargo.lock"
PYPROJECT = REPO / "bindings" / "python" / "pyproject.toml"
INIT = REPO / "bindings" / "python" / "src" / "transcribe_cpp" / "__init__.py"
PACKAGE_JSON = REPO / "bindings" / "typescript" / "package.json"
TS_DIR = REPO / "bindings" / "typescript"
SWIFT_SOURCE = REPO / "bindings" / "swift" / "Sources" / "TranscribeCpp" / "TranscribeCpp.swift"
ABIHASH = REPO / "include" / "transcribe.abihash"

# --- tooling invoked for regeneration + the --check gate ---------------------
GENERATE_PY = REPO / "bindings" / "python" / "_generate" / "generate.py"
CHECK_SYNC = REPO / "bindings" / "python" / "_generate" / "check_version_sync.py"
SWIFT_ABIHASH = REPO / "scripts" / "ci" / "swift_abihash_check.py"
LIBCLANG = "libclang==18.1.1"

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


# --- small helpers -----------------------------------------------------------

def _edit(path: Path, pattern: str, repl: str, *, expected: int, flags: int = 0) -> None:
    """Apply a regex substitution in-place, asserting ``expected`` replacements.

    A silent no-match (a spot that moved or was reformatted) is a hard error: it
    is exactly the "forgot to bump X" failure this tool exists to prevent.
    """
    text = path.read_text()
    new, n = re.subn(pattern, repl, text, flags=flags)
    if n != expected:
        raise SystemExit(
            f"error: {path.relative_to(REPO)}: pattern {pattern!r} matched {n} "
            f"time(s), expected {expected} — the file shape changed; update "
            f"prepare.py."
        )
    if new != text:
        path.write_text(new)


def _run(cmd: list[str], *, cwd: Path | None = None, quiet: bool = False) -> bool:
    """Run a command; return True on exit 0. Streams output unless ``quiet``."""
    kw = {"cwd": str(cwd)} if cwd else {}
    if quiet:
        kw["stdout"] = subprocess.DEVNULL
    proc = subprocess.run(cmd, **kw)  # noqa: S603 — fixed argv, no shell
    return proc.returncode == 0


# --- write mode --------------------------------------------------------------

def write_version(version: str) -> int:
    if not VERSION_RE.match(version):
        raise SystemExit(
            f"error: version {version!r} must be a clean MAJOR.MINOR.PATCH triple "
            f"(the header macros cannot express a packaging suffix)."
        )
    major, minor, patch = version.split(".")

    # include/transcribe.h — the source of truth.
    _edit(HEADER, r"(#define TRANSCRIBE_VERSION_MAJOR )\d+", rf"\g<1>{major}", expected=1)
    _edit(HEADER, r"(#define TRANSCRIBE_VERSION_MINOR )\d+", rf"\g<1>{minor}", expected=1)
    _edit(HEADER, r"(#define TRANSCRIBE_VERSION_PATCH )\d+", rf"\g<1>{patch}", expected=1)

    # Rust manifests: sys [package].version, safe [package].version + sys pin.
    _edit(ROOT_CARGO, r'(?m)^version = "[^"]+"', f'version = "{version}"', expected=1)
    _edit(SAFE_CARGO, r'(?m)^version = "[^"]+"', f'version = "{version}"', expected=1)
    _edit(SAFE_CARGO, r'(transcribe-cpp-sys = \{ version = ")[^"]+(")',
          rf"\g<1>{version}\g<2>", expected=1)

    # Cargo.lock: both workspace-crate entries (edited directly so this stays
    # offline + deterministic; `cargo metadata --locked` in --check proves it).
    for name in ("transcribe-cpp", "transcribe-cpp-sys"):
        _edit(CARGO_LOCK, rf'(name = "{re.escape(name)}"\nversion = ")[^"]+(")',
              rf"\g<1>{version}\g<2>", expected=1)

    # Python: [project].version, __version__, and both native-provider pins.
    _edit(PYPROJECT, r'(?m)^version = "[^"]+"', f'version = "{version}"', expected=1)
    _edit(PYPROJECT, r"(transcribe-cpp-native(?:-[a-z0-9]+)*==)[0-9.]+(\.\*)",
          rf"\g<1>{version}\g<2>", expected=2)
    _edit(INIT, r'(?m)^(__version__ = ")[^"]+(")', rf"\g<1>{version}\g<2>", expected=1)

    # npm: top-level version + the 5 @transcribe-cpp/* optionalDependencies pins
    # (targeted edits preserve package.json's hand formatting; the lock is
    # regenerated below). The top-level "version" is the first such key.
    _edit(PACKAGE_JSON, r'("version"\s*:\s*")[^"]+(")', rf"\g<1>{version}\g<2>", expected=1)
    _edit(PACKAGE_JSON, r'("@transcribe-cpp/[^"]+"\s*:\s*")[^"]+(")',
          rf"\g<1>{version}\g<2>", expected=5)

    # Swift: the hand-maintained compiledVersion literal (the ABI pin is not
    # version-coupled post-§8-P0-#1 and is left to the regeneration check below).
    _edit(SWIFT_SOURCE, r'(compiledVersion = ")[^"]+(")', rf"\g<1>{version}\g<2>", expected=1)

    print(f"wrote version {version} into every §1b spot")

    # package-lock.json: regenerate metadata only (no install, no scripts), as
    # notes/releasing.md §3 prescribes. Updates root .version + .packages[""].
    print("syncing bindings/typescript/package-lock.json ...")
    if not _run(["npm", "install", "--package-lock-only", "--ignore-scripts"], cwd=TS_DIR):
        raise SystemExit(
            "error: `npm install --package-lock-only --ignore-scripts` failed — "
            "package-lock.json was not synced. Is npm on PATH?"
        )

    # Regenerate the FFI. Post-§8-P0-#1 a version-only bump yields no diff; a
    # moved abihash means a real ABI change slipped in and needs a Swift pin bump.
    before = ABIHASH.read_text() if ABIHASH.exists() else ""
    print("regenerating the FFI (generate.py, cargo xtask bindgen) ...")
    if not _run(["uv", "run", "--no-project", "--with", LIBCLANG, str(GENERATE_PY)]):
        raise SystemExit("error: generate.py failed")
    if not _run(["cargo", "xtask", "bindgen"], cwd=REPO):
        raise SystemExit("error: cargo xtask bindgen failed")
    after = ABIHASH.read_text() if ABIHASH.exists() else ""
    if before != after:
        print(
            "\n*** NOTE: include/transcribe.abihash MOVED — this is a REAL ABI "
            "change, not a version-only bump. Review what changed, audit the "
            "bindings, and update the Swift pinnedHeaderHash in\n"
            "    bindings/swift/Sources/TranscribeCpp/ABIHash.swift\n"
            f"to the new value ({after.strip()}) after a conscious review.\n",
            file=sys.stderr,
        )

    print(f"\nprepared {version}. Now run:  uv run --no-project scripts/release/prepare.py --check")
    return 0


# --- check mode --------------------------------------------------------------

def check() -> int:
    """The umbrella hard-tier release-consistency gate (releasing.md §1c)."""
    gates = [
        ("version sync (every §1b spot)",
         ["uv", "run", "--no-project", str(CHECK_SYNC)], None, False),
        ("python/ts FFI + abihash drift",
         ["uv", "run", "--no-project", "--with", LIBCLANG, str(GENERATE_PY), "--check"], None, False),
        ("rust FFI drift",
         ["cargo", "xtask", "bindgen", "--check"], REPO, False),
        ("swift ABI pin",
         ["uv", "run", "--no-project", str(SWIFT_ABIHASH)], None, False),
        ("Cargo.lock freshness (cargo metadata --locked)",
         ["cargo", "metadata", "--locked", "--format-version", "1"], REPO, True),
        ("package-lock.json freshness (npm ci)",
         ["npm", "ci", "--ignore-scripts"], TS_DIR, True),
    ]
    results: list[tuple[str, bool]] = []
    for label, cmd, cwd, quiet in gates:
        print(f"\n=== {label} ===")
        ok = _run(cmd, cwd=cwd, quiet=quiet)
        results.append((label, ok))

    print("\n--- prepare.py --check summary ---")
    failed = 0
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
        failed += not ok
    if failed:
        print(f"\n{failed} gate(s) failed — the tree is NOT release-consistent.", file=sys.stderr)
        return 1
    print("\nall gates passed — the tree is release-consistent.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("version", nargs="?", help="the X.Y.Z version to write everywhere")
    g.add_argument("--check", action="store_true",
                   help="verify the tree is release-consistent (no writes)")
    args = ap.parse_args()
    return check() if args.check else write_version(args.version)


if __name__ == "__main__":
    raise SystemExit(main())
