#!/usr/bin/env python3
"""Fail a tag release / rehearsal if the target version is ALREADY published.

`cargo publish --dry-run` does NOT fail when the version already exists — it
warns and exits 0 — and a real `cargo publish` then BURNS the attempt against an
immutable registry (crates.io versions are permanent). PyPI/npm filenames are
likewise write-once. This is the burn-prevention the dry-run can't give: a hard
check, run in release-preflight, that the version being released does not already
exist on ANY package publish.yml would upload it under.

  TAG (push):
    crates.io   transcribe-cpp-sys, transcribe-cpp
    PyPI        transcribe-cpp, transcribe-cpp-native, transcribe-cpp-native-cu12
                (cu12 only uploads when CU12_ON_PYPI, but a 200 there is always a
                 real burn, so it is always checked)
    npm         transcribe-cpp AND the five @transcribe-cpp/<platform> packages
                (the platform packages publish FIRST, so a reused platform version
                 burns before the API package is even attempted)

  REHEARSAL (dispatch):
    TestPyPI    transcribe-cpp, transcribe-cpp-native  (the dist-* set; cu12 is
                not rehearsed on TestPyPI, npm is packed locally — no registry)

Semantics: a definitive "already exists" is a HARD failure. "Available" passes.
A registry we cannot reach (network / 5xx) is a WARNING, not a block — we only
hard-fail when we are CERTAIN the version is taken, so a transient outage cannot
hold a release hostage.

    uv run --no-project scripts/release/check_registries.py <version> [--rehearsal]

Exit 0 if no registry definitively has the version; 1 if any does.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

UA = "transcribe-cpp-release-preflight (+https://github.com/handy-computer/transcribe.cpp)"

# The five @transcribe-cpp/<tuple> npm platform packages ts-release publishes
# before the API package (publish.yml ts-release / package.json optionalDeps).
NPM_PLATFORM_TUPLES = [
    "darwin-arm64-metal",
    "darwin-x64-cpu",
    "linux-x64-cpu-vulkan",
    "linux-arm64-cpu-vulkan",
    "win32-x64-cpu-vulkan",
]

TAKEN, AVAILABLE, UNKNOWN = "taken", "available", "unknown"


def _get(url: str) -> "tuple[int | None, bytes | None]":
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310 — fixed https hosts
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None, None


def _simple(url: str) -> str:
    """crates.io / PyPI per-version endpoints: 200 = taken, 404 = available."""
    st, _ = _get(url)
    return TAKEN if st == 200 else AVAILABLE if st == 404 else UNKNOWN


def _npm(name: str, version: str) -> str:
    """npm: fetch the packument and inspect its `versions` map.

    Per-version status codes are unreliable for SCOPED names, so we read the
    whole packument (the scope `/` is percent-encoded) and check membership. A
    404 means the package was never published at all — available.
    """
    enc = name.replace("/", "%2f")
    st, body = _get(f"https://registry.npmjs.org/{enc}")
    if st == 404:
        return AVAILABLE
    if st == 200 and body is not None:
        try:
            data = json.loads(body)
        except ValueError:
            return UNKNOWN
        return TAKEN if version in (data.get("versions") or {}) else AVAILABLE
    return UNKNOWN


def targets(version: str, rehearsal: bool) -> "list[tuple[str, str, object]]":
    """(registry label, package name, checker thunk) for each immutable upload."""
    if rehearsal:
        host = "test.pypi.org"
        return [
            ("TestPyPI", name, lambda n=name: _simple(f"https://{host}/pypi/{n}/{version}/json"))
            for name in ("transcribe-cpp", "transcribe-cpp-native")
        ]
    out: list[tuple[str, str, object]] = []
    for name in ("transcribe-cpp-sys", "transcribe-cpp"):
        out.append(("crates.io", name,
                    lambda n=name: _simple(f"https://crates.io/api/v1/crates/{n}/{version}")))
    for name in ("transcribe-cpp", "transcribe-cpp-native", "transcribe-cpp-native-cu12"):
        out.append(("PyPI", name,
                    lambda n=name: _simple(f"https://pypi.org/pypi/{n}/{version}/json")))
    npm_names = ["transcribe-cpp"] + [f"@transcribe-cpp/{t}" for t in NPM_PLATFORM_TUPLES]
    for name in npm_names:
        out.append(("npm", name, lambda n=name: _npm(n, version)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="the X.Y.Z version about to be published")
    ap.add_argument("--rehearsal", action="store_true",
                    help="check TestPyPI instead of the prod registries")
    args = ap.parse_args()

    taken: list[tuple[str, str]] = []
    unknown: list[tuple[str, str]] = []
    for reg, name, check in targets(args.version, args.rehearsal):
        result = check()
        if result == TAKEN:
            taken.append((reg, name))
            print(f"  TAKEN    {reg}: {name}=={args.version} already published")
        elif result == AVAILABLE:
            print(f"  ok       {reg}: {name}=={args.version} available")
        else:
            unknown.append((reg, name))
            print(f"  unknown  {reg}: {name} — could not determine, not blocking")

    if taken:
        joined = ", ".join(f"{r}/{n}" for r, n in taken)
        print(
            f"\n::error::version {args.version} is already published on: {joined}. "
            f"Bump the version (uv run --no-project scripts/release/prepare.py X.Y.Z) — "
            f"reusing it BURNS the immutable registry.",
            file=sys.stderr,
        )
        return 1
    if unknown:
        print(f"\n::warning::could not reach {len(unknown)} registry target(s); "
              f"availability there is unconfirmed.")
    print(f"\nversion {args.version} is available on all reachable target registries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
