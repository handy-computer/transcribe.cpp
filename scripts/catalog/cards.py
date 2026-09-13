"""Derive mechanical Hugging Face card fields from a catalog record.

Used by sync_hf_cards.py to populate the committed, standalone YAML specs.
Pure stdlib: it takes and returns plain dicts, and knows nothing about YAML or
templates.
"""
from __future__ import annotations

import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

CAP_FLAGS = ("streaming", "translate", "lang_detect")
DEFAULT_SIZE = {"units": "dec", "gb_dp": 2, "mb_only": False}


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
    """Speedup over realtime per rig/backend, at the card's default quant.

    One published figure per (rig, backend), averaged over the benchmark
    samples -- which is what the hand-written specs already did. Deriving it
    keeps the metadata block from drifting when a sweep is re-run, and picks
    up rigs a hand-written spec never got around to listing.
    """
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


def derive_quants(record: dict, size: dict) -> list[dict]:
    errors = common.headline_rows(record)
    quants = []
    for item in record.get("downloads", []):
        entry = {"name": item["quant"], "filename": item["filename"],
                 "size": common.fmt_size(item["size_bytes"], **size)}
        row = errors.get(item["quant"])
        if row is not None:
            entry["wer"] = common.fmt_err(row)
        quants.append(entry)
    return quants


def derive_spec(record: dict, editorial: dict) -> dict:
    """Everything the catalog can supply, before the editorial YAML lands."""
    downloads = record.get("downloads", [])
    index = editorial.get("default_quant_index", 0)
    default_quant = downloads[index]["quant"] if index < len(downloads) else None
    size = {**DEFAULT_SIZE, **(editorial.get("size") or {})}
    spec = {
        "hf_repo": record["upstream_repo"],
        "target_repo": record.get("published_repo"),
        "upstream_commit": record["upstream_commit"],
        "license": record["license"]["spdx"],
        "license_display": record["license"]["display"],
        "languages": list(record.get("languages", [])),
        "capabilities": derive_capabilities(record),
        "quants": derive_quants(record, size),
        "perf": derive_perf(record, default_quant),
    }
    label = common.headline_label(record)
    if label:
        spec["wer"] = {"source": label}
    return spec
