#!/usr/bin/env python3
"""Audit the `transcribe-cpp-sys` crate tarball before it can be published.

    python3 scripts/ci/rust_package_audit.py

The sys crate carries the whole C++ tree (it builds libtranscribe from source
via build.rs), so the same hazards as the Python sdist apply: ship too little
and a from-source build breaks; ship too much and multi-GB model/dump/report
trees leak into the registry. This is the Rust twin of the sdist content audit
(pyproject `sdist.exclude` + its CI assertion).

Three gates, all from `cargo package`:

  - REQUIRED prefixes present  (the vendored sources a build needs)
  - FORBIDDEN prefixes absent  (models/dumps/reports/etc. — never in a package)
  - compressed size under crates.io's 10 MB default cap

`cargo package` enumerates from the git index, so run it where the crate
sources are committed (CI checkout) or staged.

Stdlib only. Exit 0 on a clean audit, 1 on any violation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CRATE = "transcribe-cpp-sys"

# Sources a from-scratch build needs — at least one packaged file under each.
REQUIRED_PREFIXES = [
    "CMakeLists.txt",
    "include/",
    "src/",
    "cmake/",
    "ggml/",
    "bindings/rust/sys/build.rs",
    "bindings/rust/sys/src/lib.rs",
    "bindings/rust/sys/src/transcribe_sys.rs",
]

# Trees that must NEVER reach the registry. `tests/`, `dumps/`, and `reports/`
# are git-TRACKED, so only the manifest's `include` allowlist keeps them out —
# this gate is what proves the allowlist is doing its job.
FORBIDDEN_PREFIXES = [
    "models/",
    "dumps/",
    "reports/",
    "tests/",
    "samples/",
    "benchmarks/",
    "docs/",
    "notes/",
    "refs/",
    "build/",
    "dist/",
    "target/",
    ".github/",
    "ggml/examples/",
    "ggml/tests/",
]

MAX_COMPRESSED_BYTES = 10 * 1024 * 1024  # crates.io default cap


def package_file_list() -> list[str]:
    out = subprocess.run(
        ["cargo", "package", "--list", "--package", CRATE, "--allow-dirty"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def build_crate() -> Path:
    subprocess.run(
        ["cargo", "package", "--no-verify", "--package", CRATE, "--allow-dirty"],
        cwd=REPO,
        check=True,
    )
    crates = sorted((REPO / "target" / "package").glob(f"{CRATE}-*.crate"))
    if not crates:
        sys.exit("audit FAILED: no .crate produced")
    return crates[-1]


def main() -> int:
    files = package_file_list()
    print(f"cargo packaged {len(files)} files")
    failures: list[str] = []

    for prefix in REQUIRED_PREFIXES:
        if not any(f == prefix or f.startswith(prefix) for f in files):
            failures.append(f"MISSING required path: {prefix}")

    for prefix in FORBIDDEN_PREFIXES:
        hits = [f for f in files if f.startswith(prefix)]
        if hits:
            failures.append(
                f"FORBIDDEN path present ({prefix}): "
                f"{hits[0]}{' …' if len(hits) > 1 else ''} ({len(hits)} files)"
            )

    crate = build_crate()
    size = crate.stat().st_size
    print(f"crate: {crate.name} — {size / 1024 / 1024:.1f} MiB compressed")
    if size > MAX_COMPRESSED_BYTES:
        failures.append(
            f"crate is {size / 1024 / 1024:.1f} MiB, over the "
            f"{MAX_COMPRESSED_BYTES / 1024 / 1024:.0f} MiB crates.io cap"
        )

    if failures:
        print("\nrust package audit FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("rust package audit ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
