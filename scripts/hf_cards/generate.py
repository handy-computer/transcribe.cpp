#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "jinja2>=3.1",
#   "pyyaml>=6.0",
#   "huggingface-hub>=0.20",
# ]
# ///
"""Generate the HuggingFace README.md for a transcribe.cpp GGUF repo.

Reads a YAML spec (see parakeet-tdt-0.6b-v2.yaml for an example), fetches the
upstream model card at the pinned commit, and renders template.md.j2.

Default output is models/<spec-stem>/README.md alongside the GGUFs, so
`hf upload <repo> models/<spec-stem> .` picks it up in the same call.

Usage:
    uv run scripts/hf_cards/generate.py scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml
    uv run scripts/hf_cards/generate.py scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml -o other/README.md
    uv run scripts/hf_cards/generate.py scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml --stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent


def load_spec(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def fetch_upstream_card(repo_id: str, revision: str) -> str:
    """Download README.md from an HF repo at a specific commit.

    Strips the upstream YAML frontmatter so our emitted frontmatter is the only
    one in the final file.
    """
    path = hf_hub_download(repo_id=repo_id, filename="README.md", revision=revision)
    content = Path(path).read_text()
    if content.startswith("---\n"):
        end = content.find("\n---\n", 4)
        if end != -1:
            content = content[end + len("\n---\n"):]
    return content.strip()


def render(spec: dict, upstream_card: str) -> str:
    env = Environment(
        loader=FileSystemLoader(HERE),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template("template.md.j2")
    return template.render(upstream_card=upstream_card, **spec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", type=Path, help="Path to the YAML spec file")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write to this path. Defaults to models/<spec-stem>/README.md.",
    )
    ap.add_argument(
        "--stdout",
        action="store_true",
        help="Write to stdout instead of a file (overrides -o).",
    )
    ap.add_argument(
        "--skip-upstream",
        action="store_true",
        help="Skip fetching the upstream card (useful for offline template iteration)",
    )
    args = ap.parse_args()

    spec = load_spec(args.spec)
    upstream = (
        "_(upstream card not fetched — run without --skip-upstream to include it)_"
        if args.skip_upstream
        else fetch_upstream_card(spec["hf_repo"], spec["upstream_commit"])
    )
    out = render(spec, upstream)

    if args.stdout:
        sys.stdout.write(out)
        return 0

    output = args.output or (REPO_ROOT / "models" / args.spec.stem / "README.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(out)
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
