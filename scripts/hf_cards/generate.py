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

The spec is assembled from two sources. Everything measurable -- the upstream
and published repos, the pinned commit, licence, languages, the quant table
with its file sizes and headline error rates, the capability flags and the
per-rig speedups -- is DERIVED from the variant's catalog/<variant>.json
record. The YAML alongside this script carries only what a human writes: the
summary, tags, pipeline tag, validation pin, and prose notes. A key present in
the YAML still wins, so a card can narrow a derived value deliberately
(Breeze-ASR-25 advertises 2 of the 99 languages its tokenizer inherits).

Fetches the upstream model card at the pinned commit and renders
template.md.j2.

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

sys.path.insert(0, str(REPO_ROOT / "scripts" / "catalog"))
import cards  # noqa: E402

def load_spec(path: Path) -> dict:
    """The editorial YAML merged onto everything derived from the catalog."""
    editorial = yaml.safe_load(path.read_text()) or {}
    record = cards.common.load_record(editorial.get("variant", path.stem))
    spec = cards.merge(cards.derive_spec(record, editorial), editorial)
    spec["quants"] = cards.merge_quants(spec["quants"], editorial.get("quant_overrides", {}))
    for key in ("variant", "size", "quant_overrides"):
        spec.pop(key, None)
    return spec


def build_transcribe_cpp_block(spec: dict) -> str:
    """Serialize the `transcribe_cpp:` block (raw WER/RTF + capability flags).

    See docs/tools/hf-metadata-schema.md. Returns "" when a spec omits `perf`,
    opting out of the block.
    """
    if "perf" not in spec:
        return ""

    caps = spec.get("capabilities", {})
    wer = spec["wer"]
    dataset_key = wer.get("metadata_key", "librispeech_test_clean")
    block: dict = {}
    # Headline dataset: per-quant WER taken from the `quants:` column.
    headline = {
        q["name"].lower(): float(str(q["wer"]).rstrip("%"))
        for q in spec["quants"] if q.get("wer") is not None
    }
    if headline:
        block[f"wer_{dataset_key}"] = headline
    # Any additional per-quant WER maps listed inline under `wer:` (keyed by
    # dataset name, e.g. `librispeech_test_clean:`) are emitted as their own
    # `wer_<dataset>` blocks. Only dict values count as datasets; scalar keys
    # (metadata_key, source, notes) are skipped.
    for key, per_quant in wer.items():
        if isinstance(per_quant, dict):
            block[f"wer_{key}"] = {
                str(q).lower(): float(str(v).rstrip("%")) for q, v in per_quant.items()
            }
    for machine, backends in spec["perf"].items():
        block[f"rtf_{machine.replace('-', '_')}"] = backends
    # Optional non-WER task metrics (for example cpWER for
    # speaker-attributed ASR). Values are emitted verbatim so the spec keeps
    # the metric's natural shape and units.
    block.update(spec.get("metrics", {}))
    block["streaming"] = bool(caps.get("streaming", False))
    if "diarize" in caps:
        block["diarize"] = bool(caps["diarize"])
    block["translate"] = bool(caps.get("translate", False))
    block["lang_detect"] = bool(caps.get("lang_detect", False))
    block["timestamps"] = caps.get("timestamps", "none")
    dumped = yaml.safe_dump(
        {"transcribe_cpp": block}, sort_keys=False, default_flow_style=False
    )
    return dumped.rstrip("\n")


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
    return template.render(
        upstream_card=upstream_card,
        transcribe_cpp_yaml=build_transcribe_cpp_block(spec),
        **spec,
    )


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
    # Most families pin the upstream card to the same SHA as the ported
    # weights. Multi-branch upstream repos (gigaam) ship the family card
    # only on `main` while per-variant branches have empty README stubs;
    # `upstream_card_commit` lets a spec point the card-fetch at a
    # different revision than `upstream_commit`.
    card_commit = spec.get("upstream_card_commit", spec["upstream_commit"])
    upstream = (
        "_(upstream card not fetched — run without --skip-upstream to include it)_"
        if args.skip_upstream
        else fetch_upstream_card(spec["hf_repo"], card_commit)
    )
    out = render(spec, upstream)

    if args.stdout:
        sys.stdout.write(out)
        return 0

    # Default output path uses the upstream-cased model dir (slug from
    # hf_repo) so the README lands alongside the GGUFs in the same
    # directory `hf upload` will publish. The kebab-cased spec stem is
    # the internal handle; the filesystem dir mirrors upstream casing
    # (matches the converter's output dir convention).
    upstream_slug = spec["hf_repo"].rsplit("/", 1)[-1]
    output = args.output or (REPO_ROOT / "models" / upstream_slug / "README.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(out)
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
