#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Render catalog-derived tables into docs/models/*.md.

The docs are hand-written prose with a few tables that restate numbers the
catalog already owns. Rather than generate whole files, this rewrites only the
regions a doc explicitly delegates:

    <!-- catalog:downloads label="LibriSpeech test-clean" units=bin -->
    | Quantization | Download | Size | WER (LibriSpeech test-clean) |
    ...
    <!-- /catalog -->

Everything outside a marker pair is untouched. The variant is the file stem
unless the marker overrides it with `variant=`, so family docs can pull a
table for a model they are not named after.

    uv run scripts/catalog/render.py                 # rewrite marked regions
    uv run scripts/catalog/render.py --check         # fail if any is stale
    uv run scripts/catalog/render.py --adopt         # wrap existing tables

`--adopt` is a one-time migration: it finds a download table that is already
correct, wraps it in a marker, and records the units convention that table
uses so adoption changes no published string. Normalising conventions is then
a separate, deliberate edit to the marker.
"""
from __future__ import annotations

import argparse
import difflib
import pathlib
import re
import shlex
import sys

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


def fmt_attrs(attrs: dict[str, object]) -> str:
    out = []
    for key, value in attrs.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        value = str(value)
        out.append(f'{key}="{value}"' if " " in value or "," in value else f"{key}={value}")
    return " ".join(out)


# --------------------------------------------------------------------------
# blocks


def block_downloads(record: dict, attrs: dict[str, str]) -> list[str]:
    """The Download table: one row per published GGUF, plus the headline metric."""
    units = attrs.get("units", "dec")
    gb_dp = int(attrs.get("gb_dp", 2))
    mb_only = as_bool(attrs.get("mb_only"), False)
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
                 common.fmt_size(item["size_bytes"], units, gb_dp, mb_only)]
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

    def ordered(index: int, override: str | None) -> list[str]:
        if override:
            return override.split(",")
        seen = []
        for key in rows:
            if key[index] not in seen:
                seen.append(key[index])
        return seen

    backends = ordered(0, attrs.get("backends"))
    samples = ordered(1, attrs.get("samples"))
    quants = ordered(2, attrs.get("quants"))
    dp_ms = int(attrs.get("dp_ms", 0))
    dp_xrt = attrs.get("dp_xrt")
    dp_xrt = None if dp_xrt is None else int(dp_xrt)

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
                             f"({common.fmt_xrt(row['xrt_compute'], dp_xrt)})")
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
                               rule_fill=True)


BLOCKS = {"downloads": block_downloads, "perf": block_perf}


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
            if name not in BLOCKS:
                raise RenderError(f"unknown block type {name!r}")
            if variant not in records:
                raise RenderError(f"no catalog record for {variant!r}")
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
# adoption


DOWNLOAD_HEADER = re.compile(r"^\|\s*Quantization\s*\|\s*Download\s*\|\s*Size\s*\|(.*)$")
DOWNLOAD_ROW = re.compile(r"^\|\s*(\S+)\s*\|\s*\[[^\]]+\]\([^)]+\)\s*\|\s*([\d.]+\s*[GM]i?B)\s*\|")


def detect_units(record: dict, published: dict[str, str]) -> dict[str, object] | None:
    """Which size convention reproduces this table's existing strings."""
    sizes = common.downloads(record)
    best, score = None, -1
    for convention in common.size_conventions():
        hit = sum(1 for quant, text in published.items()
                  if quant in sizes
                  and common.fmt_size(sizes[quant]["size_bytes"], **convention) == text)
        if hit > score:
            best, score = convention, hit
    return best if score == len(published) else None


def adopt(path: pathlib.Path, records: dict[str, dict],
          units: dict | None = None) -> tuple[str, str]:
    """Wrap an existing Download table in a marker.

    With `units` unset the table must already be self-consistent, and adoption
    changes no published string. Passing a convention instead adopts on the
    house standard and lets the next render correct whatever was stale.
    """
    record = records.get(path.stem)
    if record is None:
        return "", "no catalog record"
    lines = path.read_text().splitlines()
    if any(OPEN.match(line) for line in lines):
        return "", "already has markers"

    start = next((i for i, line in enumerate(lines) if DOWNLOAD_HEADER.match(line)), None)
    if start is None:
        return "", "no Download table"
    tail = DOWNLOAD_HEADER.match(lines[start]).group(1)
    end = start + 2
    published = {}
    while end < len(lines) and lines[end].startswith("|"):
        row = DOWNLOAD_ROW.match(lines[end])
        if not row:
            return "", "download row this renderer cannot reproduce"
        published[row.group(1)] = re.sub(r"\s+", " ", row.group(2)).strip()
        end += 1

    if {d["quant"] for d in record.get("downloads", [])} != set(published):
        return "", "table and catalog list different quants"
    convention = detect_units(record, published) if units is None else dict(units)
    if convention is None:
        return "", "sizes match no single units convention (stale or hand-edited)"
    covered = set(common.headline_rows(record))
    if covered and not set(published) <= covered:
        return "", ("headline_benchmark covers only "
                    + ", ".join(sorted(covered)) + "; table publishes more")

    attrs: dict[str, object] = dict(convention)
    columns = [c.strip() for c in tail.split("|") if c.strip()]
    if not columns:
        attrs["metric"] = False
    else:
        head = columns[0]
        if len(columns) > 1:
            return "", "second metric column is not supported yet"
        label = re.match(r"^(\w+)\s*\((.+)\)$", head)
        if not label:
            return "", f"cannot parse metric header {head!r}"
        attrs["metric_name"], attrs["label"] = label.group(1), label.group(2)
        target = common.headline(record)
        if not target:
            return "", "headline_benchmark is null"
        if attrs["metric_name"] == target["metric"].upper():
            del attrs["metric_name"]
        if attrs["label"] == common.headline_label(record):
            del attrs["label"]

    body = lines[:start] + [f"<!-- catalog:downloads {fmt_attrs(attrs)} -->"] \
        + lines[start:end] + ["<!-- /catalog -->"] + lines[end:]
    return "\n".join(body) + "\n", ""


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="report stale regions and exit non-zero; write nothing")
    parser.add_argument("--adopt", action="store_true",
                        help="one-time: wrap existing correct tables in markers")
    parser.add_argument("--adopt-units", choices=("dec", "bin"),
                        help="adopt on this units convention instead of "
                             "detecting the table's own; stale sizes are then "
                             "corrected by the next render")
    parser.add_argument("--docs", default=str(common.DOCS_DIR))
    parser.add_argument("paths", nargs="*", help="limit to these files")
    args = parser.parse_args()

    records = common.load_records()
    docs = ([pathlib.Path(p) for p in args.paths]
            or sorted(pathlib.Path(args.docs).glob("*.md")))

    if args.adopt:
        units = {"units": args.adopt_units, "gb_dp": 2, "mb_only": False} \
            if args.adopt_units else None
        adopted, skipped = 0, []
        for path in docs:
            text, why = adopt(path, records, units)
            if why:
                skipped.append((path.name, why))
                continue
            path.write_text(text)
            adopted += 1
        print(f"adopted {adopted} download table(s)")
        for name, why in skipped:
            print(f"  skipped {name}: {why}")
        return 0

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
