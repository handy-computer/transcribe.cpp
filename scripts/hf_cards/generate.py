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

Two inputs, disjoint by construction:

  catalog/<variant>.json          identity, repos, licence, languages,
                                  capabilities, downloads, benchmark numbers
  scripts/hf_cards/<variant>.yaml editorial copy and release state only:
                                  summary, tags, pipeline tag, validation pin,
                                  prose notes, optional usage override

Nothing numeric or mechanical is read from the YAML; a number that belongs
on the card belongs in the catalog first. Fetches the upstream model card at
the pinned commit and renders template.md.j2.

Default output is models/<upstream-slug>/README.md alongside the GGUFs, so
`hf upload <repo> models/<upstream-slug> .` picks it up in the same call.

Usage:
    uv run scripts/hf_cards/generate.py scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml
    uv run scripts/hf_cards/generate.py scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml -o other/README.md
    uv run scripts/hf_cards/generate.py scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml --stdout
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download
from jinja2 import Environment, FileSystemLoader, StrictUndefined

HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "catalog"))
import common  # noqa: E402

# Keys the catalog owns. A spec that states one of these is stale by
# definition, so refuse it rather than let the two quietly diverge again.
CATALOG_OWNED = {
    "hf_repo", "target_repo", "upstream_commit", "license", "license_display",
    "license_name", "license_link", "languages", "capabilities", "perf",
    "quants", "metric", "catalog_sync",
}
CAP_FLAGS = ("streaming", "translate", "lang_detect")


def load_spec(path: Path) -> dict:
    """The editorial half of a card. Fails on any catalog-owned key."""
    with path.open() as f:
        spec = yaml.safe_load(f) or {}
    stale = sorted(CATALOG_OWNED & spec.keys())
    if stale:
        raise SystemExit(
            f"{path.name}: {', '.join(stale)} come from catalog/{path.stem}.json; "
            f"remove them from the spec")
    return spec


# --------------------------------------------------------------------------
# catalog -> card context


def derive_capabilities(record: dict) -> dict:
    """The boolean flags the `transcribe_cpp:` metadata block carries."""
    caps = record.get("capabilities", {})
    out = {flag: bool(caps.get(flag, {}).get("supported")) for flag in CAP_FLAGS}
    if caps.get("diarize", {}).get("supported"):
        out["diarize"] = True
    granularities = caps.get("timestamps", {}).get("granularities") or []
    # Advertise the finest granularity the port actually emits.
    out["timestamps"] = next((g for g in ("token", "word", "segment")
                              if g in granularities), "none")
    return out


def derive_perf(record: dict, default_quant: str | None) -> dict:
    """Speedup over realtime per rig/backend at the card's default quant,
    averaged over the benchmark samples."""
    cells: dict[tuple[str, str], list[float]] = {}
    for row in record.get("speed_benchmarks", []):
        if row["quant"] != default_quant:
            continue
        cells.setdefault((row["machine"], row["backend"]), []).append(row["xrt_compute"])
    perf: dict[str, dict[str, float]] = {}
    for (machine, backend), values in sorted(cells.items()):
        mean = round(statistics.fmean(values), 1)
        perf.setdefault(machine, {})[backend] = int(mean) if mean == int(mean) else mean
    return perf


def derive_quants(record: dict, secondary: dict | None) -> list[dict]:
    """One row per published GGUF: size, headline error rate, and the optional
    second metric column the spec supplies under `wer.<metadata_key2>`."""
    errors = common.headline_rows(record)
    quants = []
    for item in record.get("downloads", []):
        entry = {"name": item["quant"], "filename": item["filename"],
                 "size": common.fmt_size(item["size_bytes"])}
        row = errors.get(item["quant"])
        if row is not None:
            entry["wer"] = common.fmt_err(row)
        if secondary is not None:
            value = secondary.get(item["quant"].lower())
            if value is not None:
                entry["wer2"] = f"{float(value):.2f}%"
        quants.append(entry)
    return quants


def build_context(record: dict, spec: dict) -> dict:
    """Everything the template needs: catalog facts plus the editorial spec."""
    downloads = record.get("downloads", [])
    index = spec.get("default_quant_index", 0)
    default_quant = downloads[index]["quant"] if index < len(downloads) else None
    wer = dict(spec.get("wer") or {})
    if not wer.get("source"):
        wer["source"] = common.headline_label(record)
    secondary = None
    if "source2" in wer:
        key2 = wer.get("metadata_key2")
        if not key2 or not isinstance(wer.get(key2), dict):
            raise SystemExit("wer.source2 needs wer.metadata_key2 naming a "
                             "{quant: value} map under wer:")
        secondary = {str(q).lower(): v for q, v in wer[key2].items()}
    headline = common.headline(record) or {}
    ctx = {
        **spec,
        "hf_repo": record["upstream_repo"],
        "target_repo": record.get("published_repo"),
        "upstream_commit": record["upstream_commit"],
        "license": record["license"]["spdx"],
        "license_display": record["license"]["display"],
        "languages": list(record.get("languages", [])),
        "capabilities": derive_capabilities(record),
        "perf": derive_perf(record, default_quant),
        "quants": derive_quants(record, secondary),
        "wer": wer,
    }
    if headline.get("metric"):
        ctx["metric"] = headline["metric"].upper()
    for key in ("name", "link"):
        if record["license"].get(key):
            ctx[f"license_{key}"] = record["license"][key]
    if not ctx["target_repo"]:
        raise SystemExit(f"{record['variant']}: catalog has no published_repo")
    return ctx


# --------------------------------------------------------------------------
# rendering


def build_transcribe_cpp_block(ctx: dict) -> str:
    """Serialize the `transcribe_cpp:` block (raw WER/RTF + capability flags).

    See docs/tools/hf-metadata-schema.md. Returns "" when the catalog holds no
    speed rows for the default quant, opting out of the block.
    """
    if not ctx["perf"]:
        return ""

    caps = ctx["capabilities"]
    wer = ctx["wer"]
    dataset_key = wer.get("metadata_key", "librispeech_test_clean")
    block: dict = {}
    # Headline dataset: per-quant WER taken from the `quants:` column.
    headline = {
        q["name"].lower(): float(str(q["wer"]).rstrip("%"))
        for q in ctx["quants"] if q.get("wer") is not None
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
    for machine, backends in ctx["perf"].items():
        block[f"rtf_{machine.replace('-', '_')}"] = backends
    # Optional non-WER task metrics (for example cpWER for
    # speaker-attributed ASR). Values are emitted verbatim so the spec keeps
    # the metric's natural shape and units.
    block.update(ctx.get("metrics", {}))
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


def render(ctx: dict, upstream_card: str) -> str:
    env = Environment(
        loader=FileSystemLoader(HERE),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template("template.md.j2")
    return template.render(
        upstream_card=upstream_card,
        transcribe_cpp_yaml=build_transcribe_cpp_block(ctx),
        **ctx,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", type=Path, help="Path to the editorial YAML spec")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write to this path. Defaults to models/<upstream-slug>/README.md.",
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
    record = common.load_record(args.spec.stem)
    ctx = build_context(record, spec)
    # Most families pin the upstream card to the same SHA as the ported
    # weights. Multi-branch upstream repos (gigaam) ship the family card
    # only on `main` while per-variant branches have empty README stubs;
    # `upstream_card_commit` lets a spec point the card-fetch at a
    # different revision than the catalog's upstream_commit.
    card_commit = spec.get("upstream_card_commit", ctx["upstream_commit"])
    upstream = (
        "_(upstream card not fetched — run without --skip-upstream to include it)_"
        if args.skip_upstream
        else fetch_upstream_card(ctx["hf_repo"], card_commit)
    )
    out = render(ctx, upstream)

    if args.stdout:
        sys.stdout.write(out)
        return 0

    # Default output path uses the upstream-cased model dir (slug from
    # hf_repo) so the README lands alongside the GGUFs in the same
    # directory `hf upload` will publish. The kebab-cased spec stem is
    # the internal handle; the filesystem dir mirrors upstream casing
    # (matches the converter's output dir convention).
    upstream_slug = ctx["hf_repo"].rsplit("/", 1)[-1]
    output = args.output or (REPO_ROOT / "models" / upstream_slug / "README.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(out)
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
