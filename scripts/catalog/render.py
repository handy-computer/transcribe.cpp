#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Render catalog-derived regions into docs/models/*.md.

The docs are hand-written pages with a few regions that restate what the
catalog or the HF card spec already owns: the download and perf tables, the
intro summary, and the WER methodology note. Rather than generate whole
files, this rewrites only the regions a doc explicitly delegates:

    <!-- catalog:downloads label="LibriSpeech test-clean" -->
    | Quantization | Download | Size | WER (LibriSpeech test-clean) |
    ...
    <!-- /catalog -->

Blocks: `downloads`, `perf machine=<slug>`, `accuracy` (one table per
dataset split beyond the headline), `intro` (upstream link plus the card
spec's `summary`), `prose field=wer.notes` (any `|` text field of the spec,
dotted path), and `family variants=a,b,c` (a roll-up row per variant, for
family pages). Everything outside a marker pair is untouched. The variant is the file stem
unless the marker overrides it with `variant=`, so family docs can pull a
table for a model they are not named after.

    uv run scripts/catalog/render.py                 # rewrite marked regions
    uv run scripts/catalog/render.py --check         # fail if any is stale
"""
from __future__ import annotations

import argparse
import difflib
import pathlib
import re
import shlex
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

OPEN = re.compile(r"^(\s*)<!--\s*catalog:([a-z-]+)\s*(.*?)\s*-->\s*$")
CLOSE = re.compile(r"^\s*<!--\s*/catalog\s*-->\s*$")


class RenderError(Exception):
    """A marker names something the catalog cannot currently supply."""


def parse_attrs(text: str) -> dict[str, str]:
    attrs = {}
    for token in shlex.split(text):
        key, _, value = token.partition("=")
        attrs[key] = value
    return attrs


def as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------
# blocks


def block_downloads(record: dict, attrs: dict[str, str]) -> list[str]:
    """The Download table: one row per published GGUF, plus the headline metric."""
    want_metric = as_bool(attrs.get("metric"), True)

    rows_by_quant = common.headline_rows(record) if want_metric else {}
    target = common.headline(record)
    if want_metric and not target:
        raise RenderError("metric column requested but headline_benchmark is null")

    header = ["Quantization", "Download", "Size"]
    aligns = ["l", "l", "r"]
    if want_metric:
        label = attrs.get("label") or common.headline_label(record)
        metric = attrs.get("metric_name") or target["metric"].upper()
        header.append(f"{metric} ({label})")
        aligns.append("r")

    body = []
    for item in record.get("downloads", []):
        url = common.download_url(record, item["filename"])
        if not url:
            raise RenderError("published_repo is null, so downloads have no URL")
        cells = [item["quant"], f"[{item['filename']}]({url})",
                 common.fmt_size(item["size_bytes"])]
        if want_metric:
            cells.append(common.fmt_err(rows_by_quant.get(item["quant"])))
        body.append(cells)
    if not body:
        raise RenderError("no downloads")
    return common.render_table(header, aligns, body)


def block_perf(record: dict, attrs: dict[str, str]) -> list[str]:
    """A per-machine latency grid: rows are (backend, sample), columns quants."""
    machine = attrs.get("machine")
    if not machine:
        raise RenderError("perf block needs machine=")
    rows = common.perf_rows(record, machine)
    if not rows:
        raise RenderError(f"no speed_benchmarks rows for machine {machine!r}")

    def ordered(index: int, override: str | None, rank) -> list[str]:
        if override:
            return override.split(",")
        return sorted({key[index] for key in rows}, key=rank)

    # GPU backends first, then CPU; samples shortest first; quants in the
    # record's download order (reference dtype down to the smallest quant).
    backend_rank = {"metal": 0, "cuda": 1, "vulkan": 2, "cpu": 9}
    duration = {key[1]: row["sample_duration_s"] for key, row in rows.items()}
    quant_rank = {item["quant"]: i for i, item in enumerate(record.get("downloads", []))}
    backends = ordered(0, attrs.get("backends"), lambda b: (backend_rank.get(b, 5), b))
    samples = ordered(1, attrs.get("samples"), lambda s: (duration.get(s, 0), s))
    quants = ordered(2, attrs.get("quants"), lambda q: (quant_rank.get(q, 99), q))
    dp_ms = int(attrs.get("dp_ms", 0))

    body, blocked = [], []
    for backend in backends:
        for sample in samples:
            present = [rows.get((backend, sample, q)) for q in quants]
            if not any(present):
                continue
            duration = next(r["sample_duration_s"] for r in present if r)
            cells = [backend.capitalize() if backend != "cpu" else "CPU",
                     f"{sample} ({duration:.1f}s)"]
            for quant, row in zip(quants, present):
                if row is None:
                    cells.append("-")
                    continue
                if row.get("total_ms") is None:
                    blocked.append(f"{backend}/{sample}/{quant}")
                    cells.append("-")
                    continue
                cells.append(f"{common.fmt_ms(row['total_ms'], dp_ms)} "
                             f"({common.fmt_xrt(row)})")
            body.append(cells)
    if blocked:
        raise RenderError(
            f"{len(blocked)} cell(s) on {machine} have no total_ms, so latency "
            f"cannot be rendered: {', '.join(blocked[:4])}"
            + (" ..." if len(blocked) > 4 else ""))
    if not body:
        raise RenderError(f"no rows matched on {machine}")
    return common.render_table(["Backend", "Sample"] + quants,
                               ["l", "l"] + ["r"] * len(quants), body,
                               rule_fill=True, max_pad=20)


_SPECS: dict[str, dict] = {}


def spec_for(record: dict) -> dict:
    """The editorial HF card spec, the prose source of truth for a variant."""
    variant = record["variant"]
    if variant not in _SPECS:
        path = common.CARDS_DIR / f"{variant}.yaml"
        if not path.exists():
            raise RenderError(f"no card spec at {path.relative_to(common.REPO)}")
        _SPECS[variant] = yaml.safe_load(path.read_text()) or {}
    return _SPECS[variant]


def prose_lines(text: object, what: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        raise RenderError(f"{what} is empty")
    return text.strip().split("\n")


def block_intro(record: dict, attrs: dict[str, str]) -> list[str]:
    """Upstream pointer from the catalog, then the card spec's summary."""
    repo = record["upstream_repo"]
    line = (f"Upstream: [`{repo}`](https://huggingface.co/{repo}) at "
            f"[`{record['upstream_commit']}`]"
            f"(https://huggingface.co/{repo}/commit/{record['upstream_commit']}).")
    return [line, ""] + prose_lines(spec_for(record).get("summary"), "summary")


def block_prose(record: dict, attrs: dict[str, str]) -> list[str]:
    """A text field of the card spec, named by dotted path (`wer.notes`)."""
    field = attrs.get("field")
    if not field:
        raise RenderError("prose block needs field=")
    value: object = spec_for(record)
    for part in field.split("."):
        value = value.get(part) if isinstance(value, dict) else None
    return prose_lines(value, f"spec field {field!r}")


def block_accuracy(record: dict, attrs: dict[str, str]) -> list[str]:
    """One table per dataset split: language rows, quant columns.

    The headline cell (dataset, split, language) is already the download
    table's last column, so it is left out unless `all=true`; the rest of its
    split still renders. `datasets=fleurs:test,librispeech:test-clean` narrows
    to named splits.
    """
    headline = common.headline(record) or {}
    wanted = None
    if attrs.get("datasets"):
        wanted = {tuple(item.split(":", 1)) for item in attrs["datasets"].split(",")}
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in record.get("accuracy_benchmarks", []):
        key = (row["dataset"], row["split"])
        if wanted is not None and key not in wanted:
            continue
        if wanted is None and not as_bool(attrs.get("all"), False) \
                and (*key, row["language"]) == (headline.get("dataset"), headline.get("split"),
                                               headline.get("language")):
            continue
        groups.setdefault(key, []).append(row)
    if not groups:
        raise RenderError("no accuracy rows beyond the headline benchmark")

    quant_rank = {item["quant"]: i for i, item in enumerate(record.get("downloads", []))}
    out: list[str] = []
    for (dataset, split), rows in sorted(groups.items()):
        quants = sorted({row["quant"] for row in rows}, key=lambda q: (quant_rank.get(q, 99), q))
        cells: dict[tuple[str, str], dict] = {}
        for row in rows:
            # Several recipes of one cell can coexist (batch size, timestamps);
            # the profile recipe wins, else the first listed.
            key = (row["language"], row["quant"])
            if key not in cells or row.get("engine_sha") and not cells[key].get("engine_sha"):
                cells[key] = row
        body = []
        for language in sorted({row["language"] for row in rows}):
            metric = next(cells[(language, q)]["metric"] for q in quants if (language, q) in cells)
            body.append([language, metric.upper()]
                        + [common.fmt_err(cells.get((language, q))) for q in quants])
        if out:
            out.append("")
        label = f"FLEURS {split}" if dataset == "fleurs" else common.dataset_label(dataset, split, "")
        out.extend([f"**{label}**", ""])
        out.extend(common.render_table(["Language", "Metric"] + quants,
                                       ["l", "l"] + ["r"] * len(quants), body))
    return out


def block_family(records: dict[str, dict], attrs: dict[str, str]) -> list[str]:
    """A family roll-up: one row per variant, headline number at one quant."""
    names = [name for name in attrs.get("variants", "").split(",") if name]
    if not names:
        raise RenderError("family block needs variants=")
    quant = attrs.get("quant", "Q8_0")
    body = []
    for name in names:
        record = records.get(name)
        if record is None:
            raise RenderError(f"no catalog record for {name!r}")
        download = next((d for d in record.get("downloads", []) if d["quant"] == quant), None)
        headline = common.headline(record) or {}
        doc = common.DOCS_DIR / f"{name}.md"
        link = (f"[{name}.md]({name}.md)" if doc.exists()
                else f"[{record['published_repo']}](https://huggingface.co/{record['published_repo']})")
        body.append([
            f"`{name}`", common.fmt_params(record["params"]), common.languages_summary(record),
            common.fmt_size(download["size_bytes"]) if download else "-",
            (f"{common.headline_label(record)} ({headline['metric'].upper()})"
             if headline else "-"),
            common.fmt_err(common.headline_rows(record).get(quant)),
            common.capabilities_summary(record), link])
    return common.render_table(
        ["Variant", "Params", "Languages", f"{quant} size", "Benchmark", quant,
         "Capabilities", "Doc"],
        ["l", "r", "l", "r", "l", "r", "l", "l"], body, max_pad=34)


BLOCKS = {"downloads": block_downloads, "perf": block_perf,
          "intro": block_intro, "prose": block_prose, "accuracy": block_accuracy}


# --------------------------------------------------------------------------
# file rewriting


def rewrite(path: pathlib.Path, records: dict[str, dict]) -> tuple[str, list[str]]:
    lines = path.read_text().splitlines()
    out, errors, index = [], [], 0
    while index < len(lines):
        match = OPEN.match(lines[index])
        if not match:
            out.append(lines[index])
            index += 1
            continue
        indent, name, raw = match.groups()
        close = next((j for j in range(index + 1, len(lines)) if CLOSE.match(lines[j])), None)
        if close is None:
            errors.append(f"{path.name}:{index + 1}: catalog:{name} has no <!-- /catalog -->")
            out.append(lines[index])
            index += 1
            continue
        out.append(lines[index])
        attrs = parse_attrs(raw)
        variant = attrs.get("variant", path.stem)
        try:
            if name == "family":
                rendered = block_family(records, attrs)
            elif name not in BLOCKS:
                raise RenderError(f"unknown block type {name!r}")
            elif variant not in records:
                raise RenderError(f"no catalog record for {variant!r}")
            else:
                rendered = BLOCKS[name](records[variant], attrs)
        except RenderError as exc:
            errors.append(f"{path.name}:{index + 1}: catalog:{name} {variant}: {exc}")
            out.extend(lines[index + 1:close])  # leave the region alone
        else:
            out.extend(indent + line for line in rendered)
        out.append(lines[close])
        index = close + 1
    return "\n".join(out) + "\n", errors


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="report stale regions and exit non-zero; write nothing")
    parser.add_argument("--docs", default=str(common.DOCS_DIR))
    parser.add_argument("paths", nargs="*", help="limit to these files")
    args = parser.parse_args()

    records = common.load_records()
    docs = ([pathlib.Path(p) for p in args.paths]
            or sorted(pathlib.Path(args.docs).glob("*.md")))

    stale, errors, rendered = [], [], 0
    for path in docs:
        current = path.read_text()
        text, file_errors = rewrite(path, records)
        errors.extend(file_errors)
        if not any(OPEN.match(line) for line in current.splitlines()):
            continue
        rendered += 1
        if text == current:
            continue
        stale.append(path)
        if args.check:
            diff = difflib.unified_diff(current.splitlines(), text.splitlines(),
                                        f"a/{path}", f"b/{path}", lineterm="", n=1)
            print("\n".join(diff))
        else:
            path.write_text(text)

    for error in errors:
        print(f"  error: {error}", file=sys.stderr)
    verb = "stale" if args.check else "rewritten"
    print(f"{rendered} doc(s) with markers; {len(stale)} {verb}; {len(errors)} error(s)")
    return 1 if (args.check and stale) or errors else 0


if __name__ == "__main__":
    sys.exit(main())
