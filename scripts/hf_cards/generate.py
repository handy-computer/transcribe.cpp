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
import re
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

# The editorial surface of a card spec, and the whole of it. Anything else is
# either owned by catalog/<variant>.json or a typo; both are refused, so the
# two can never quietly diverge again and a misspelled key cannot silently
# render an empty section. Adding a field here is a deliberate act.
SPEC_KEYS = {
    "pin_date",              # date the upstream revision was pinned
    "validation",            # reference framework, transcribe.cpp commit, date
    "pipeline_tag",          # HF Hub pipeline tag
    "tags",                  # HF Hub tags
    "summary",               # the card's opening paragraph
    "wer",                   # editorial caveats; see README.md
    "usage",                 # extra usage prose for an unusual model
    "upstream_card_commit",  # revision of the upstream card that was quoted
    "default_quant",         # override the Q8_0 default
}
CAP_FLAGS = ("streaming", "translate", "lang_detect")
DEFAULT_QUANT = "Q8_0"


def load_spec(path: Path) -> dict:
    """The editorial half of a card. Fails on any catalog-owned key."""
    with path.open() as f:
        spec = yaml.safe_load(f) or {}
    unknown = sorted(spec.keys() - SPEC_KEYS)
    if unknown:
        raise SystemExit(
            f"{path.name}: {', '.join(unknown)} is not an editorial field. It is "
            f"either derived from catalog/{path.stem}.json or misspelled; remove "
            f"it. Editorial fields: {', '.join(sorted(SPEC_KEYS))}")
    if "default_quant_index" in spec:
        raise SystemExit(f"{path.name}: default_quant_index is gone; the default is "
                         f"{DEFAULT_QUANT}, override with default_quant: <QUANT>")
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
        perf.setdefault(machine, {})[backend] = round(statistics.fmean(values), 2)
    return perf


def metric_key(row: dict) -> str:
    """`<metric>_<dataset>_<split|language>[_<scoring>][_<mode>]`, the name of
    the per-quant map this row belongs to in the metadata block."""
    key = f"{row['metric']}_{row['dataset']}_{common.dataset_tail(row)}"
    for extra in ("scoring", "mode"):
        if row.get(extra):
            key += f"_{row[extra]}"
    return re.sub(r"[^a-z0-9]+", "_", key.lower())


def derive_metric_blocks(record: dict) -> dict[str, dict[str, float]]:
    """Every per-quant error map the catalog holds, keyed by metric_key.
    Where a cell was measured under several recipes (batch size, timestamps)
    the profile-stamped row wins, else the first listed."""
    chosen: dict[tuple[str, str], dict] = {}
    for row in record.get("accuracy_benchmarks", []):
        cell = (metric_key(row), row["quant"].lower())
        if cell not in chosen or (row.get("engine_sha") and not chosen[cell].get("engine_sha")):
            chosen[cell] = row
    blocks: dict[str, dict[str, float]] = {}
    for (key, quant), row in chosen.items():
        blocks.setdefault(key, {})[quant] = row["err_pct"]
    return blocks


def derive_quants(record: dict, secondary: dict | None) -> list[dict]:
    """One row per published GGUF: size, headline error rate, and the optional
    second metric column named by `wer.secondary`."""
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



def hub_language_tags(languages) -> tuple[list[str], list[str]]:
    """Split catalog language tags the way the Hub's metadata validator wants.

    `language:` accepts ISO 639 codes only, so a model whose GGUF advertises
    locales (nemotron tags `en-US`) keeps the full tags in `language_bcp47`
    and contributes each primary subtag, deduped, to `language`.
    """
    tags = [str(lang) for lang in languages]
    base, seen = [], set()
    for tag in tags:
        primary = tag.split("-")[0].lower()
        if primary not in seen:
            seen.add(primary)
            base.append(primary)
    return base, [tag for tag in tags if "-" in tag]

DOCS_BASE = "https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models"


def docs_url(record: dict) -> str:
    """The model's page on GitHub: its own, or the family page it shares."""
    page = f"{record['variant']}.md"
    if not (common.DOCS_DIR / page).exists():
        page = record.get("docs_page") or page
    return f"{DOCS_BASE}/{page}"


def build_context(record: dict, spec: dict) -> dict:
    """Everything the template needs: catalog facts plus the editorial spec."""
    downloads = {item["quant"]: item for item in record.get("downloads", [])}
    default_quant = spec.get("default_quant", DEFAULT_QUANT)
    if default_quant not in downloads:
        raise SystemExit(f"{record['variant']}: default quant {default_quant!r} is not "
                         f"a published download ({', '.join(downloads) or 'none'})")
    wer = dict(spec.get("wer") or {})
    stale = [k for k, v in wer.items() if isinstance(v, dict)] + \
            [k for k in ("metadata_key", "metadata_key2") if k in wer]
    if stale or "metrics" in spec:
        raise SystemExit(f"{record['variant']}: per-quant numbers and metadata keys are "
                         f"derived from the catalog; remove wer.{'/'.join(stale)}"
                         + (" and metrics" if "metrics" in spec else ""))
    if not wer.get("source"):
        wer["source"] = common.headline_label(record)
    elif wer["source"] == common.headline_label(record):
        raise SystemExit(f"{record['variant']}: wer.source restates the catalog's "
                         f"headline label; remove it")
    wer["recipe"] = common.headline_recipe(record)
    blocks = derive_metric_blocks(record)
    secondary = None
    if "source2" in wer:
        key2 = wer.get("secondary")
        if key2 not in blocks:
            raise SystemExit(f"{record['variant']}: wer.secondary must name one of "
                             f"{sorted(blocks)}")
        secondary = blocks[key2]
    headline = common.headline(record) or {}
    hub_languages, languages_bcp47 = hub_language_tags(record.get("languages", []))
    ctx = {
        **spec,
        "hf_repo": record["upstream_repo"],
        "target_repo": record.get("published_repo"),
        "upstream_commit": record["upstream_commit"],
        "license": record["license"]["spdx"],
        "license_display": record["license"]["display"],
        "languages": hub_languages,
        "languages_bcp47": languages_bcp47,
        "capabilities": derive_capabilities(record),
        "perf": derive_perf(record, default_quant),
        "quants": derive_quants(record, secondary),
        "default_quant_filename": downloads[default_quant]["filename"],
        "wer": wer,
        "metric_blocks": blocks,
        "transcribe_docs_url": docs_url(record),
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
    """Serialize the `transcribe_cpp:` block (raw error rates, RTF, and
    capability flags), entirely from the catalog record.

    See docs/tools/hf-metadata-schema.md. Returns "" when the catalog holds no
    speed rows for the default quant, opting out of the block.
    """
    if not ctx["perf"]:
        return ""

    caps = ctx["capabilities"]
    # Bumped when key names or shapes change. 2: every result set the catalog
    # holds is emitted, keyed <metric>_<dataset>_<split|lang>[_scoring][_mode];
    # earlier cards emitted a hand-named headline map and up to one extra.
    block: dict = {"schema_version": 2}
    # Every per-quant error map the catalog holds, headline first.
    for key, per_quant in ctx["metric_blocks"].items():
        block[key] = dict(per_quant)
    for machine, backends in ctx["perf"].items():
        block[f"rtf_{machine.replace('-', '_')}"] = backends
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
