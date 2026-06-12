# /// script
# requires-python = ">=3.9"
# ///
"""Generate a PEP 503 "simple repository" index from this repo's GitHub
release assets, for wheels too large or too specialized for PyPI.

The cu12 CUDA provider is ~197 MB — over PyPI's 100 MB cap — so it cannot live
on PyPI; future Windows-CUDA / cu13 / ROCm providers ride the same index.
GitHub hosts the wheels (as release assets); GitHub Pages hosts the few KB of
HTML this produces. pip/uv consume it as:

    pip install "transcribe-cpp[cu12]" \
        --extra-index-url https://<owner>.github.io/<repo>/whl/cu12

Modeled on llama-cpp-python's releases-to-pep-503.sh, with one simplification:
our flavors are distinguished by PACKAGE NAME (transcribe-cpp-native-cu12), not
by a build-tag suffix on the release, so a single release can carry every
flavor's wheels and we just filter assets by distribution name. No sha256 in
the URLs (would mean downloading every 197 MB wheel each run — the same
trade-off llama-cpp-python makes; assets are served over HTTPS from GitHub).

Usage (CI):
    GITHUB_REPOSITORY=owner/repo GITHUB_TOKEN=*** \
        python build_wheel_index.py --out index/whl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

#: flavor (index subdir) -> provider packages whose wheels live under it.
#: A wheel is matched to a flavor purely by its distribution name, so the same
#: flavor holds every platform tag of that package (linux + windows cu12).
FLAVORS: dict[str, list[str]] = {
    "cu12": ["transcribe-cpp-native-cu12"],
    # future: "cu13": ["transcribe-cpp-native-cu13"], "rocm": [...]
}

_API = "https://api.github.com"


def _get(url: str):
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as r:  # noqa: S310 (fixed github host)
        return json.load(r)


def normalize(name: str) -> str:
    """PEP 503 normalized name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_dist(filename: str) -> str:
    """Distribution name of a wheel file, PEP 503 normalized.

    A wheel filename is ``{distribution}-{version}-...whl`` and the
    distribution field never contains ``-`` (escaped to ``_``), so the first
    ``-`` field is the distribution.
    """
    return normalize(filename.split("-", 1)[0])


def all_release_wheels(repo: str):
    """Yield (filename, browser_download_url) for every .whl across releases."""
    page = 1
    while True:
        releases = _get(f"{_API}/repos/{repo}/releases?per_page=100&page={page}")
        if not releases:
            return
        for release in releases:
            for asset in release.get("assets", []):
                name = asset["name"]
                if name.endswith(".whl"):
                    yield name, asset["browser_download_url"]
        page += 1


def _page(body: str) -> str:
    return f"<!DOCTYPE html>\n<html><head><meta name=\"pypi:repository-version\" content=\"1.0\"></head><body>\n{body}\n</body></html>\n"


def build(repo: str, out: Path, wheels: list[tuple[str, str]]) -> None:
    for flavor, packages in FLAVORS.items():
        wanted = {normalize(p) for p in packages}
        by_pkg: dict[str, list[tuple[str, str]]] = {}
        for filename, url in wheels:
            dist = wheel_dist(filename)
            if dist in wanted:
                by_pkg.setdefault(dist, []).append((filename, url))

        flavor_dir = out / flavor
        flavor_dir.mkdir(parents=True, exist_ok=True)
        # Root index for the flavor: one link per package (always emit the
        # configured packages so the index is well-formed even before any wheel
        # has been released).
        pkgs = sorted(wanted)
        flavor_dir.joinpath("index.html").write_text(
            _page("\n".join(f'    <a href="{p}/">{p}</a>' for p in pkgs))
        )
        for pkg in pkgs:
            files = sorted(by_pkg.get(pkg, []))
            pkg_dir = flavor_dir / pkg
            pkg_dir.mkdir(parents=True, exist_ok=True)
            pkg_dir.joinpath("index.html").write_text(
                _page("\n".join(f'    <a href="{url}">{fn}</a>' for fn, url in files))
            )
            print(f"[{flavor}] {pkg}: {len(files)} wheel(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="index/whl", help="output directory")
    ap.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"),
                    help="owner/repo (default: $GITHUB_REPOSITORY)")
    args = ap.parse_args()
    if not args.repo:
        sys.exit("set $GITHUB_REPOSITORY or pass --repo owner/repo")
    wheels = list(all_release_wheels(args.repo))
    print(f"found {len(wheels)} wheel asset(s) across releases of {args.repo}")
    build(args.repo, Path(args.out), wheels)


if __name__ == "__main__":
    main()
