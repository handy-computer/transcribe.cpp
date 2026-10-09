"""Shared helpers for reading catalog records and rendering them.

Pure stdlib, so every consumer -- check.py, db.py, render.py and
scripts/hf_cards/generate.py -- can import it without a dependency block.

The catalog stores identity and exact numbers. Everything about how a number
LOOKS (units, decimal places, column padding, a dataset's display name) is a
rendering concern and lives here or in the marker that calls the renderer.
"""
from __future__ import annotations

import json
import pathlib
from urllib.parse import urlparse

REPO = pathlib.Path(__file__).resolve().parents[2]
CATALOG_DIR = REPO / "catalog"
DOCS_DIR = REPO / "docs" / "models"
CARDS_DIR = REPO / "scripts" / "hf_cards"

HEADLINE_KEYS = ("dataset", "split", "language", "metric", "batch_size", "timestamps")
# The recipe half of the pointer may be null, meaning "any"; the identity
# half never is. db.py's headline view encodes the same predicate in SQL.
HEADLINE_WILDCARD_KEYS = ("batch_size", "timestamps")


# --------------------------------------------------------------------------
# loading


def load_records(directory: pathlib.Path | None = None) -> dict[str, dict]:
    """Every catalog record, keyed by variant. `_`-prefixed files are tooling."""
    directory = directory or CATALOG_DIR
    return {path.stem: json.loads(path.read_text())
            for path in sorted(directory.glob("*.json"))
            if not path.name.startswith("_")}


def load_record(variant: str, directory: pathlib.Path | None = None) -> dict:
    path = (directory or CATALOG_DIR) / f"{variant}.json"
    if not path.exists():
        raise FileNotFoundError(f"no catalog record for {variant!r} at {path}")
    return json.loads(path.read_text())


# --------------------------------------------------------------------------
# sizes


def fmt_size(size_bytes: int) -> str:
    """Render a byte count the way a download table prints it: decimal MB
    below a gigabyte, decimal GB to two places above."""
    if size_bytes < 10**9:
        return f"{size_bytes / 10**6:.0f} MB"
    return f"{size_bytes / 10**9:.2f} GB"


# --------------------------------------------------------------------------
# dataset identity
#
# A benchmark row names its dataset with three fields (dataset, split,
# language), but every consumer wants a single string: a `<kind>:<value>`
# spec for the WER harness, a slug for a report filename, a key for the
# database. They differ in punctuation, not in meaning, so the rule that
# picks the value lives here once. scripts/wer/remote/dataset_specs.py owns
# the other direction (spec string -> manifest and volume paths) and
# `dataset_spec` below emits exactly what its parse_dataset_spec accepts.

# FLEURS publishes one split across many languages, so its language is what
# identifies a result; every other dataset varies by split instead. The WER
# harness agrees: `fleurs:zh` names a language and carries the split as a
# separate --split flag, while `librispeech:test-clean` names a split.
LANGUAGE_KEYED_DATASETS = ("fleurs", "eka-medical-asr")
# The split a language-keyed dataset is published at, which its spec leaves
# implicit.
PUBLISHED_SPLIT = {"fleurs": "test", "eka-medical-asr": "test"}
# Datasets that publish a single language, so a slug need not name one.
SINGLE_LANGUAGE_DATASETS = ("librispeech",)


def dataset_tail(row: dict) -> str:
    """The value half of a dataset spec: which FLEURS language, which
    LibriSpeech split."""
    if row["dataset"] in LANGUAGE_KEYED_DATASETS:
        return str(row["language"])
    return str(row["split"])


def dataset_slug(row: dict) -> str:
    """`fleurs-es`, `librispeech-test-clean`. The dataset half of a WER report
    filename, and the stem the catalog looks for when ingesting a score."""
    return f"{row['dataset']}-{dataset_tail(row)}"


def dataset_spec(row: dict) -> str:
    """`fleurs:es`, `librispeech:test-clean`. What run.py and the Modal sweep
    take as `--dataset`."""
    return f"{row['dataset']}:{dataset_tail(row)}"


# --------------------------------------------------------------------------
# accuracy

DATASET_LABELS = {
    ("librispeech", "test-clean"): "LibriSpeech test-clean",
    ("ami", "ihm-test"): "AMI IHM test",
}


def dataset_label(dataset: str, split: str, language: str) -> str:
    if dataset == "fleurs":
        # "mul": a multilingual pool (language ID scores several FLEURS
        # languages as one result set).
        return "FLEURS multilingual" if language == "mul" else f"FLEURS {language}"
    return DATASET_LABELS.get((dataset, split), f"{dataset} {split}")


def metric_label(metric: str) -> str:
    """A metric as a column header prints it: WER / CER / DER / CPWER, or
    "Top-1 accuracy" for language ID."""
    return "Top-1 accuracy" if metric == "accuracy" else metric.upper()


def headline(record: dict) -> dict | None:
    """The benchmark row-set a variant publishes in its download table.

    A variant can carry several runs of the same dataset that differ only in
    batch size or timestamp mode, so the pointer names the full identity
    tuple rather than just the dataset.
    """
    return record.get("headline_benchmark")


def headline_rows(record: dict) -> dict[str, dict]:
    """{quant: accuracy row} for the headline benchmark. Empty if unset."""
    target = headline(record)
    if not target:
        return {}
    rows = {}
    for row in record.get("accuracy_benchmarks", []):
        # A null batch_size or timestamps is an intentional wildcard for a
        # legacy table assembled before recipe metadata was standardized.
        if all((key in HEADLINE_WILDCARD_KEYS and target[key] is None)
               or row.get(key) == target[key]
               for key in HEADLINE_KEYS):
            rows[row["quant"]] = row
    return rows


def headline_label(record: dict) -> str:
    target = headline(record)
    if not target:
        return ""
    return dataset_label(target["dataset"], target["split"], target["language"])


def headline_recipe(record: dict) -> str:
    """The mechanical half of a WER note, from the headline rows themselves:
    dataset, size, batch, timestamps, backend, and which build measured it."""
    rows = list(headline_rows(record).values())
    target = headline(record)
    if not rows or not target:
        return ""
    measured = [row for row in rows if row.get("engine_sha")] or rows
    sample = measured[0]
    n_utts = max(row["n_utts"] for row in rows)
    unit = "meetings" if target["metric"] in ("der", "cpwer") else "utterances"
    # Language ID scores a pooled, cropped subset (the row's notes say which),
    # and has no batch size or timestamps.
    lid = target["metric"] == "accuracy"
    scope = "" if lid else "the full "
    parts = [f"{metric_label(target['metric'])} on {scope}{headline_label(record)} "
             f"{'' if lid else 'split '}({n_utts:,} {unit})"]
    batch_sizes = sorted({row["batch_size"] for row in rows
                          if row.get("batch_size") is not None and not lid})
    if len(batch_sizes) == 1:
        parts.append(f"batch size {batch_sizes[0]}")
    elif batch_sizes:
        parts.append("batch sizes " + " and ".join(str(size) for size in batch_sizes))
    if sample.get("timestamps") and not lid:
        parts.append(f"timestamps {sample['timestamps']}")
    if sample.get("language_hint"):
        parts.append(f"language hint `{sample['language_hint']}`")
    if sample.get("backend"):
        parts.append(f"{'scored' if lid else 'decoded'} on {sample['backend']}")
    text = ", ".join(parts) + "."
    shas = sorted({(row["engine_sha"], row.get("measured_on") or "")
                   for row in rows if row.get("engine_sha")})
    if shas:
        text += " Measured at " + "; ".join(
            f"transcribe.cpp `{sha}`" + (f" on {date}" if date else "") for sha, date in shas) + "."
    if any(not row.get("engine_sha") for row in rows):
        text += " Figures without a commit were published before provenance was recorded."
    return text


def row_pct(row: dict) -> float:
    """A row's percentage: the error rate, or the accuracy on metric=accuracy
    (language ID) rows, which carry acc_pct instead of err_pct."""
    return row["acc_pct"] if row.get("metric") == "accuracy" else row["err_pct"]


def fmt_err(row: dict | None, dp: int = 2) -> str:
    """An error rate (or, for accuracy rows, the accuracy) as a card prints
    it. `-` when the cell was not measured."""
    if row is None:
        return "-"
    return f"{row_pct(row):.{dp}f}%"


# --------------------------------------------------------------------------
# speed


def fmt_ms(total_ms: float, dp_ms: int = 0, dp_s: int = 2) -> str:
    if total_ms < 1000:
        return f"{total_ms:.{dp_ms}f} ms"
    return f"{total_ms / 1000:.{dp_s}f} s"


XRT_DP = 2


def fmt_xrt(row: dict) -> str:
    """Speedup over realtime, two decimals. A legacy row carries only the
    precision its doc published (an integer or one decimal), so it is printed
    as stored rather than padded to a fidelity it never had."""
    xrt = row["xrt_compute"]
    if row.get("engine_sha"):
        return f"{xrt:.{XRT_DP}f}×"
    text = repr(float(xrt))
    decimals = len(text.split(".")[1].rstrip("0"))
    return f"{xrt:.{min(decimals, XRT_DP)}f}×"


def perf_rows(record: dict, machine: str) -> dict[tuple[str, str, str], dict]:
    """{(backend, sample, quant): row} for one machine."""
    return {(row["backend"], row["sample"], row["quant"]): row
            for row in record.get("speed_benchmarks", [])
            if row["machine"] == machine}


# --------------------------------------------------------------------------
# summaries


def fmt_params(params: int) -> str:
    if params < 10**6:
        return f"{params // 10**3}K"
    if params >= 10**9:
        return f"{params / 10**9:.1f}B".replace(".0B", "B")
    return f"{round(params / 10**6):.0f}M"


def languages_summary(record: dict) -> str:
    """`en`, `en, de, fr`, or a count, plus a note when the model auto-detects."""
    langs = [str(lang) for lang in record.get("languages", [])]
    text = ", ".join(langs) if len(langs) <= 4 else f"{len(langs)} languages"
    if record.get("capabilities", {}).get("lang_detect", {}).get("supported"):
        text += " + auto-detect"
    return text or "-"


def capabilities_summary(record: dict) -> str:
    """The extras beyond plain transcription, as a short comma list."""
    caps = record.get("capabilities", {})
    out = []
    if record.get("role") == "langid":
        out.append(f"language ID ({len(record.get('languages', []))} languages)")
    if record.get("role") == "vad":
        out.append("voice activity detection")
    for name, label in (("translate", "translate"), ("streaming", "streaming"),
                        ("diarize", "diarize")):
        if caps.get(name, {}).get("supported"):
            out.append(label)
    grans = caps.get("timestamps", {}).get("granularities") or []
    if grans:
        out.append(f"{grans[0]} timestamps")
    return ", ".join(out) or "-"


# --------------------------------------------------------------------------
# source provenance


def upstream_is_hf(record: dict) -> bool:
    """An explicit non-HF URL overrides the legacy HF repo convention."""
    url = record.get("upstream_url")
    return not url or urlparse(url).hostname == "huggingface.co"


def upstream_url(record: dict) -> str:
    return record.get("upstream_url") or f"https://huggingface.co/{record['upstream_repo']}"


def upstream_commit_url(record: dict) -> str:
    url = upstream_url(record).rstrip("/")
    if upstream_is_hf(record) or urlparse(url).hostname == "github.com":
        return f"{url}/commit/{record['upstream_commit']}"
    return url


def source_provenance(record: dict) -> list[str]:
    """The exact packaged weights are distinct from the upstream code pin."""
    lines = []
    artifact = record.get("source_artifact")
    if artifact:
        lines.append(f"Source artifact: `{artifact['package']}=={artifact['version']}`, "
                     f"[`{artifact['filename']}`]({artifact['url']}). "
                     f"SHA256: `{artifact['sha256']}`.")
    return lines


def model_geometry(record: dict) -> list[str]:
    """Catalog-derived parameter count and native VAD frame geometry."""
    lines = []
    if record.get("role") == "vad":
        line = f"Model: {record['params']:,} parameters."
        info = record.get("vad_info")
        if info:
            line += (f" Input: {info['sample_rate']:,} Hz mono; "
                     f"{info['frame_samples']:,} samples/frame "
                     f"({info['frame_samples'] / info['sample_rate'] * 1000:g} ms).")
        lines.append(line)
    return lines


# --------------------------------------------------------------------------
# downloads


def download_url(record: dict, filename: str) -> str:
    repo = record.get("published_repo")
    if not repo:
        return ""
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


# --------------------------------------------------------------------------
# markdown tables


MAX_PAD = 14


def render_table(header: list[str], aligns: list[str], rows: list[list[str]],
                 rule_fill: bool = False, max_pad: int = MAX_PAD,
                 pad_header: bool = True) -> list[str]:
    """A GitHub markdown table, columns padded so the source reads as a grid.

    `aligns` is "l" or "r" per column. `rule_fill` draws the separator out to
    the column width (`| ------- |`) instead of the short form (`| --- |`);
    both are used in docs/models and neither renders differently.

    Columns wider than `max_pad` are left ragged: a download table's link
    column runs past 120 characters, and padding it buys nothing while making
    every other cell unreadable in the source.
    """
    source = [header] + rows if pad_header else rows
    widths = [max(len(row[i]) for row in source) for i in range(len(header))]
    widths = [0 if width > max_pad else width for width in widths]

    def line(cells: list[str], pad: bool = True) -> str:
        return "| " + " | ".join(
            (cell.rjust(width) if align == "r" else cell.ljust(width)) if pad else cell
            for cell, width, align in zip(cells, widths, aligns)).rstrip() + " |"

    head = line(header, pad_header)
    if rule_fill:
        rules = ["-" * max(width - 1, 2) + ":" if align == "r" else "-" * max(width, 3)
                 for width, align in zip(widths, aligns)]
        return [head, "| " + " | ".join(rules) + " |"] + [line(row) for row in rows]
    rules = ["---:" if align == "r" else "---" for align in aligns]
    return [head, "| " + " | ".join(rules) + " |"] + [line(row) for row in rows]


# --------------------------------------------------------------------------
# VAD measurements (shared by documentation and HF cards)


def measurement_provenance(rows: list[dict]) -> list[str]:
    builds = sorted({(row["engine_sha"], row.get("measured_on") or "",
                      row.get("publication_profile") or "")
                     for row in rows if row.get("engine_sha")})
    if not builds:
        return []
    return ["Measured at " + "; ".join(
        f"transcribe.cpp `{sha}`" + (f" on {date}" if date else "")
        + (f", profile `{profile}`" if profile else "")
        for sha, date, profile in builds) + "."]


def render_reference_parity(record: dict) -> list[str]:
    rows = record.get("reference_parity", [])
    if not rows:
        raise ValueError("no reference_parity rows")
    header = ["Dataset", "Language", "GGUF", "Identical files / files",
              "Segments", "Audio (s)", "Max abs probability delta"]
    aligns = ["l", "l", "l", "r", "r", "r", "r"]
    recipe, varying = [], []
    for key, label in (("backend", "Backend"), ("reference", "Reference"),
                       ("segmentation_params", "Segmentation parameters")):
        values = sorted({row[key] for row in rows})
        if len(values) == 1:
            recipe.append(f"{label}: `{values[0]}`.")
        else:
            varying.append(key)
            header.append(label)
            aligns.append("l")
    body = []
    for row in rows:
        delta = row.get("max_abs_prob_delta")
        label = (f"FLEURS {row['split']}" if row["dataset"] == "fleurs"
                 else dataset_label(row["dataset"], row["split"], row["language"]))
        body.append([
            label, row["language"], row["quant"],
            f"{row['n_identical']:,} / {row['n_files']:,}",
            f"{row['n_segments']:,}", f"{row['audio_duration_s']:.3f}",
            "-" if delta is None else f"{delta:.6g}"] + [row[key] for key in varying])
    return ["Reference parity compares speech segment boundaries with the reference "
            "using default segmentation parameters. It is not labeled VAD accuracy "
            "or WER.", " ".join(recipe), ""] + render_table(
                header, aligns, body
            ) + [""] + measurement_provenance(rows)


def stream_perf_rows(record: dict, machine: str) -> list[dict]:
    rank = {item["quant"]: i for i, item in enumerate(record.get("downloads", []))}
    return sorted((row for row in record.get("speed_benchmarks", [])
                   if row["machine"] == machine),
                  key=lambda row: (row["sample_duration_s"], row["backend"],
                                   rank.get(row["quant"], 99), row["quant"]))


def fmt_latency_ms(value: float) -> str:
    """Sub-ms call measurements must not round to the legacy `0 ms` cell."""
    return f"{value:.6g} ms"


def render_stream_perf(record: dict, machine: str,
                       rows: list[dict] | None = None) -> list[str]:
    rows = stream_perf_rows(record, machine) if rows is None else rows
    if not rows:
        raise ValueError(f"no stream speed_benchmarks rows for machine {machine!r}")
    required = ("feed_samples", "frame_samples", "threads", "n_calls",
                "warmup_calls", "total_ms", "median_ms", "p95_ms")
    body = []
    for row in rows:
        if any(row.get(key) is None for key in required):
            raise ValueError(f"stream row {row['sample']}/{row['quant']} needs "
                             + ", ".join(required))
        if row["frame_samples"] <= 0:
            raise ValueError("stream frame_samples must be positive")
        frames = row["feed_samples"] / row["frame_samples"]
        if frames <= 0:
            raise ValueError("stream frames/feed must be positive")
        body.append([
            row["backend"], row["quant"],
            f"{row['sample_duration_s'] * 1000:g} ms", f"{frames:g}",
            fmt_latency_ms(row["median_ms"]), fmt_latency_ms(row["p95_ms"]),
            fmt_latency_ms(row["median_ms"] / frames), f"{row['n_calls']:,}"])
    recipe = []
    for key, label in (("frame_samples", "Samples/frame"), ("threads", "Threads"),
                       ("warmup_calls", "Warmup calls")):
        values = sorted({row[key] for row in rows})
        recipe.append(label + ": " + ", ".join(f"{value:,}" for value in values) + ".")
    samples = list(dict.fromkeys(row["sample"] for row in rows))
    recipe.append("Source samples: " + ", ".join(f"`{sample}`" for sample in samples) + ".")
    intro = [
        "Streaming feed-call wall latency; the 32 ms chunk is the headline. "
        "128 ms and 512 ms chunks show feed-size amortization, not independent "
        "per-frame latency.",
        "Includes the Python ctypes/native API wall call; excludes model load "
        "and audio capture. One pass over the audio with preserved stream state. "
        "Warmup calls are excluded from the measured call statistics. "
        "Chunk duration is audio per feed, not the full clip duration.",
        "Median and p95 are measured per feed call. Amortized median/frame is "
        "median/feed divided by frames/feed, not a separately measured single-frame "
        "latency. Mean feed-call latency remains in `total_ms` metadata.",
        " ".join(recipe), ""]
    return intro + render_table(
        ["Backend", "GGUF", "Chunk", "Frames/feed", "Median/feed", "p95/feed",
         "Amortized median/frame", "Calls"],
        ["l", "l", "l"] + ["r"] * 5, body
    ) + ["", f"Machine: `{machine}`."] + measurement_provenance(rows)


# --------------------------------------------------------------------------
# writing records

WRAP_WIDTH = 79


def _compact(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _fill(items: list[str], pad: str, inner: str) -> str:
    """A scalar array too long for one line, filled to WRAP_WIDTH."""
    lines, current = [], ""
    for index, item in enumerate(items):
        piece = item + ("," if index < len(items) - 1 else "")
        candidate = (current + " " + piece) if current else inner + piece
        if current and len(candidate) > WRAP_WIDTH:
            lines.append(current)
            current = inner + piece
        else:
            current = candidate
    if current:
        lines.append(current)
    return "[\n" + "\n".join(lines) + "\n" + pad + "]"


def _holds_records(value) -> bool:
    """True when `value` contains a list of objects somewhere inside.

    That is the shape worth expanding: a streaming block's `presets` is a list
    of rows a reader scans, while an accuracy row's `errors` is three counts
    that belong on the row's own line. Both sit at the same depth, so depth
    alone cannot tell them apart.
    """
    if isinstance(value, list):
        return any(isinstance(item, dict) for item in value) or \
               any(_holds_records(item) for item in value)
    if isinstance(value, dict):
        return any(_holds_records(item) for item in value.values())
    return False


def _fmt(value, depth: int = 0, indent: int = 2) -> str:
    pad, inner = " " * (indent * depth), " " * (indent * (depth + 1))
    if isinstance(value, list):
        if not any(isinstance(item, (dict, list)) for item in value):
            one = _compact(value)
            if len(pad) + len(one) <= WRAP_WIDTH or not value:
                return one
            return _fill([json.dumps(v, ensure_ascii=False) for v in value], pad, inner)
        if depth >= 2 and not _holds_records(value):
            return _compact(value)
        if not value:
            return "[]"
        body = ",\n".join(inner + _fmt(item, depth + 1, indent) for item in value)
        return "[\n" + body + "\n" + pad + "]"
    if isinstance(value, dict):
        if depth >= 2 and not _holds_records(value):
            return _compact(value)
        if not value:
            return "{}"
        body = ",\n".join(f"{inner}{json.dumps(key, ensure_ascii=False)}: "
                          f"{_fmt(val, depth + 1, indent)}"
                          for key, val in value.items())
        return "{\n" + body + "\n" + pad + "}"
    return _compact(value)


def dumps_record(record: dict) -> str:
    """Serialize a record the way every checked-in record is written.

    Top level and depth-1 containers expand one entry per line; anything
    deeper, and any array of scalars, stays compact -- so a benchmark row is
    one greppable line and a 99-language list wraps instead of running 99
    lines. Plain `json.dumps(indent=2)` writes the same data as a file five to
    eight times longer, which turns a one-value correction into an
    unreviewable diff. Every writer here goes through this.
    """
    return _fmt(record) + "\n"


def write_record(path: pathlib.Path, record: dict) -> None:
    path.write_text(dumps_record(record))
