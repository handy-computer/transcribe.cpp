#!/usr/bin/env python3
"""cargo semver-checks for transcribe-cpp-types and transcribe-cpp against
their latest release on crates.io.

    uv run --no-project python scripts/ci/rust_semver_check.py [--strict]

The tool judges a break against the version in the tree: on a PR (version =
last release) any break fails; at release, after prepare.py, a minor bump
(0.5 -> 0.6) allows breaks and a patch bump does not.

Runs informational in rust-ci and blocking (`--strict`) in publish.yml's
rust-verify / rust-rehearsal. Features: serde on (the derives are API), no
backend features (they don't change the Rust API and need SDKs to build).

The baseline is picked here, not by the tool: the newest non-yanked,
non-prerelease release no newer than the tree. A crate is skipped when it has
no such release, and transcribe-cpp is also skipped while its baseline
predates the types split (does not depend on transcribe-cpp-types): the tool
does not follow re-exports from another crate, so against such a baseline it
reports every type that moved to transcribe-cpp-types as removed. Keying the
skip on the baseline's own dependencies (rather than on whether
transcribe-cpp-types exists at all) keeps it right if the types crate is ever
published ahead of transcribe-cpp: a name placeholder, or a release that
failed after its first publish step.

Without --strict, an unreachable crates.io skips with a warning; with it, the
check fails rather than let a release through unchecked.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[2]
UA = "transcribe-cpp-ci (+https://github.com/handy-computer/transcribe.cpp)"
TYPES = "transcribe-cpp-types"
SAFE = "transcribe-cpp"
MANIFESTS = {
    TYPES: REPO / "bindings" / "rust" / "transcribe-cpp-types" / "Cargo.toml",
    SAFE: REPO / "bindings" / "rust" / "transcribe-cpp" / "Cargo.toml",
}


class Unreachable(Exception):
    pass


def get(path: str) -> dict | None:
    """GET a crates.io API path; None on 404, Unreachable on anything else."""
    req = urllib.request.Request(f"https://crates.io/api/v1/{path}", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise Unreachable(f"HTTP {e.code} for {path}") from e
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise Unreachable(f"{e} for {path}") from e


def parse(v: str) -> tuple[int, int, int] | None:
    """A plain X.Y.Z as a tuple; None for prereleases / build metadata."""
    parts = v.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]), int(parts[2])


def baseline(crate: str, current: str) -> str | None:
    """The newest non-yanked, non-prerelease release <= the tree's version."""
    d = get(f"crates/{crate}/versions?per_page=100")
    if d is None:
        return None
    if (d.get("meta") or {}).get("next_page"):
        # More than 100 versions: the newest are on the first page (crates.io
        # sorts newest first), which is all a baseline needs, but say so.
        print(f"note: {crate} has more than 100 versions; using the first page")
    cur = parse(current)
    if cur is None:
        raise SystemExit(f"error: {crate}'s version {current!r} is not X.Y.Z")
    best: tuple[int, int, int] | None = None
    for v in d["versions"]:
        t = parse(v["num"])
        if t is not None and not v["yanked"] and t <= cur and (best is None or t > best):
            best = t
    return None if best is None else ".".join(map(str, best))


def depends_on_types(version: str) -> bool:
    d = get(f"crates/{SAFE}/{version}/dependencies")
    if d is None:
        raise Unreachable(f"no dependency list for {SAFE} {version}")
    return any(dep["crate_id"] == TYPES and dep["kind"] == "normal" for dep in d["dependencies"])


def tree_version(crate: str) -> str:
    return tomllib.loads(MANIFESTS[crate].read_text())["package"]["version"]


def plan() -> list[tuple[str, str]]:
    """(crate, baseline) pairs to check; prints why anything is skipped."""
    out: list[tuple[str, str]] = []
    for crate in (TYPES, SAFE):
        current = tree_version(crate)
        base = baseline(crate, current)
        if base is None:
            print(f"::notice::{crate}: no release <= {current} on crates.io yet; skipped")
            continue
        if crate == SAFE and not depends_on_types(base):
            print(
                f"::notice::{crate}: baseline {base} predates {TYPES} (the types were "
                f"re-exported from it later); skipped, as cargo semver-checks would "
                f"report every moved type as removed"
            )
            continue
        out.append((crate, base))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--strict",
        action="store_true",
        help="fail (instead of skipping) when crates.io cannot be reached",
    )
    args = ap.parse_args()

    try:
        checks = plan()
    except Unreachable as e:
        if args.strict:
            print(f"::error::crates.io unreachable ({e}); semver check cannot run")
            return 1
        print(f"::warning::crates.io unreachable ({e}); semver checks skipped")
        return 0

    failed: list[str] = []
    for crate, base in checks:
        print(f"=== cargo semver-checks -p {crate} --baseline-version {base}", flush=True)
        rc = subprocess.run(
            [
                "cargo", "semver-checks", "-p", crate,
                "--baseline-version", base,
                "--only-explicit-features", "--features", "serde",
            ],
            cwd=REPO,
            check=False,
        ).returncode
        if rc != 0:
            failed.append(f"{crate} (vs {base})")
    if failed:
        print(f"::error::semver check failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
