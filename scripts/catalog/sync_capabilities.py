#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf", "huggingface-hub", "requests"]
# ///
"""Read a record's `capabilities` block back out of its GGUF's KVs.

The GGUF is the truth and the catalog follows it. If the file is wrong, the
fix is a converter change plus a re-export, never an edit to the record:
absence is not falsity, since read_capability_bool() leaves a field alone
when its KV is missing and the family default then applies. The shared
writer factory (scripts/lib/gguf_common.py) fills any unset capability KV
with false, so every fresh export states all four.

Hand-writing this block is how moss-transcribe-diarize shipped as
diarize:false and how whisper-large-v3 came to claim translate:false while its
own GGUF says otherwise. This reads the truth back out instead.

Sources, and the reason each is what it is:

  translate / lang_detect / streaming / diarize
      `stt.capability.*` KVs, which is exactly what read_capability_kv() in
      src/transcribe-meta.cpp feeds to the public capability surface.
  timestamps.granularities
      NOT a KV. `max_timestamp_kind` is a family invariant applied in each
      family's load(), so the ceiling is scraped out of
      src/arch/<family>/capabilities.cpp and cannot drift from the C++.
      Granite scopes it per variant off stt.capability.word_timestamps.
  transcribe / batching
      Left alone: batching sits behind the transcribe_model_supports() probe
      rather than the capability struct, so reading it needs a loaded model.

A GGUF is read locally when present, otherwise its header is range-fetched
from the published repo (a few MB, not the weights).

    uv run scripts/catalog/sync_capabilities.py --dry-run
    uv run scripts/catalog/sync_capabilities.py --local-only
    uv run scripts/catalog/sync_capabilities.py
    uv run scripts/catalog/sync_capabilities.py --check   # exit 1 on any disagreement
    uv run scripts/catalog/sync_capabilities.py --check --models <variant>
                                # ship gate: every quant, every KV present, all agree
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

ARCH = common.REPO / "src" / "arch"

# catalog `family` -> src/arch directory, where the two spellings diverge.
FAMILY_DIRS = {
    "cohere_asr": "cohere",
    "granite_speech": "granite",
    "granite_speech_nar": "granite_nar",
}

KIND_RE = re.compile(r"max_timestamp_kind\s*=\s*TRANSCRIBE_TIMESTAMPS_(\w+)")

KV = {
    "translate": "stt.capability.translate",
    "lang_detect": "stt.capability.lang_detect",
    "streaming": "stt.capability.streaming",
    "diarize": "stt.capability.speaker_diarization",
}


def family_ceiling(family: str) -> str | None:
    """The family's max_timestamp_kind, read straight out of the C++."""
    path = ARCH / FAMILY_DIRS.get(family, family) / "capabilities.cpp"
    if not path.exists():
        return None
    match = KIND_RE.search(path.read_text())
    return match.group(1).lower() if match else None


def granularities(ceiling: str | None) -> list[str]:
    """Just the ceiling. max_timestamp_kind is the finest the family emits;
    whether it also assembles coarser output is a run() question, so listing
    the coarser kinds here would be a claim the C++ does not make."""
    return [] if not ceiling or ceiling == "none" else [ceiling]


def read_kvs(reader) -> dict:
    """Every stt.capability.* / stt.translation.* KV the file carries."""
    out = {}
    for key, field in reader.fields.items():
        if not (key.startswith("stt.capability.") or key.startswith("stt.translation.")):
            continue
        try:
            parts = field.parts
            if field.types and field.types[0].name == "ARRAY":
                out[key] = [bytes(parts[i]).decode("utf-8", "replace")
                            if parts[i].dtype.kind in "iu" and len(parts[i]) > 1
                            else parts[i].tolist() for i in field.data]
                out[key] = [v if isinstance(v, str) else
                            bytes(bytearray(v)).decode("utf-8", "replace")
                            for v in out[key]]
            else:
                value = parts[field.data[0]]
                out[key] = bool(value[0]) if field.types[0].name == "BOOL" else value.tolist()[0]
        except (IndexError, KeyError, AttributeError, UnicodeDecodeError):
            continue
    return out


def local_path(filename: str) -> pathlib.Path | None:
    for directory in sorted((common.REPO / "models").glob("*")):
        if directory.is_dir() and (directory / filename).exists():
            return directory / filename
    return None


def default_filename(record: dict) -> str:
    """The one file the sweep reads per record: a local one if any is on
    disk, else the smallest published quant (cheapest header fetch)."""
    for item in record["downloads"]:
        if local_path(item["filename"]):
            return item["filename"]
    return sorted(record["downloads"], key=lambda d: d["size_bytes"])[0]["filename"]


def open_gguf(record: dict, filename: str, local_only: bool):
    """A GGUFReader over the named file: local if present, else its
    range-fetched header from the published repo."""
    from gguf import GGUFReader

    path = local_path(filename)
    if path:
        return GGUFReader(str(path)), "local", None
    if local_only or not record.get("published_repo"):
        return None, None, f"{filename} not on disk"

    # Header-only read: pad the temp file out to the declared size so
    # GGUFReader's memmap of the tensor region stays in bounds and is never
    # touched. Same trick as scripts/audit_gguf_metadata.py.
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    try:
        url = hf_hub_url(record["published_repo"], filename)
        total = get_hf_file_metadata(url).size
        prefix = min(24 * 1024 * 1024, total)
        headers = build_hf_headers()
        headers["Range"] = f"bytes=0-{prefix - 1}"
        response = requests.get(url, headers=headers, timeout=120)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - any transport failure is just "unavailable"
        return None, None, f"{filename}: {type(exc).__name__}: {str(exc)[:70]}"
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
        tmp = pathlib.Path(handle.name)
        handle.write(response.content)
        handle.truncate(total)
    try:
        return GGUFReader(str(tmp)), "hub", None
    finally:
        tmp.unlink(missing_ok=True)


def build(record: dict, kvs: dict) -> dict:
    """The capabilities block this GGUF implies, keeping human-set flags."""
    previous = record.get("capabilities", {})

    def carry(name: str, supported: bool, extra: dict | None = None) -> dict:
        block = {"supported": supported}
        if supported:
            was = previous.get(name, {})
            block["verified"] = bool(was.get("verified")) if was.get("supported") else False
            block.update(extra or {})
            if was.get("note"):
                block["note"] = was["note"]
        return block

    caps = {"transcribe": previous.get("transcribe", {"supported": True, "verified": False})}

    targets = kvs.get("stt.translation.target_languages")
    pairs = kvs.get("stt.translation.pairs")
    translate = bool(kvs.get(KV["translate"], False))
    extra = {}
    if translate:
        extra = {"targets": targets or None, "pairs": pairs or None}
    caps["translate"] = carry("translate", translate, extra)
    caps["lang_detect"] = carry("lang_detect", bool(kvs.get(KV["lang_detect"], False)))

    ceiling = family_ceiling(record["family"])
    if record["family"].startswith("granite_speech") and not kvs.get(
            "stt.capability.word_timestamps", False):
        # Granite scopes the ceiling per variant, in arch/granite/model.cpp.
        ceiling = "none"
    grans = granularities(ceiling)
    caps["timestamps"] = carry("timestamps", bool(grans), {"granularities": grans} if grans else None)

    caps["streaming"] = carry("streaming", bool(kvs.get(KV["streaming"], False)))
    if caps["streaming"]["supported"]:
        for key in ("mode", "presets"):
            if key in previous.get("streaming", {}):
                caps["streaming"][key] = previous["streaming"][key]
    caps["diarize"] = carry("diarize", bool(kvs.get(KV["diarize"], False)))
    if caps["diarize"]["supported"]:
        for key in ("max_speakers", "granularity", "markup"):
            if key in previous.get("diarize", {}):
                caps["diarize"][key] = previous["diarize"][key]
    caps["batching"] = previous.get("batching", {"supported": False})
    for optional in ("punctuation", "casing", "itn"):
        if optional in previous:
            caps[optional] = previous[optional]
    return caps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-only", action="store_true",
                        help="skip the hub fallback")
    parser.add_argument("--models", default="",
                        help="comma-separated variants; every published quant "
                             "of each is inspected (default: all variants, one "
                             "file each)")
    parser.add_argument("--check", action="store_true",
                        help="write nothing; exit 1 if any record disagrees "
                             "with its GGUF. With --models, a missing "
                             "capability KV, an unreadable file, or quants that "
                             "disagree with each other also fail.")
    args = parser.parse_args()
    args.dry_run = args.dry_run or args.check

    records = common.load_records()
    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    unknown = selected - records.keys()
    if unknown:
        print(f"unknown catalog variant(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    strict = args.check and bool(selected)

    changed, unreachable, missing, disagree = [], [], [], []
    sources = collections.Counter()
    for variant, record in records.items():
        if selected and variant not in selected:
            continue
        filenames = ([item["filename"] for item in record["downloads"]] if selected
                     else [default_filename(record)])
        caps_by_file = {}
        for filename in filenames:
            reader, source, error = open_gguf(record, filename, args.local_only)
            if reader is None:
                unreachable.append((variant, error))
                continue
            sources[source] += 1
            kvs = read_kvs(reader)
            absent = [key for key in KV.values() if key not in kvs]
            if absent:
                missing.append((variant, filename, absent))
            caps_by_file[filename] = build(record, kvs)
        if not caps_by_file:
            continue
        distinct = {json.dumps(caps, sort_keys=True) for caps in caps_by_file.values()}
        if len(distinct) > 1:
            disagree.append((variant, sorted(caps_by_file)))
            continue
        caps = next(iter(caps_by_file.values()))
        before = record.get("capabilities", {})
        if caps == before:
            continue
        diff = [f"{name}: {before.get(name, {}).get('supported')} -> {block['supported']}"
                for name, block in caps.items()
                if before.get(name, {}).get("supported") != block.get("supported")]
        grain_before = (before.get("timestamps") or {}).get("granularities")
        grain_after = (caps.get("timestamps") or {}).get("granularities")
        if grain_before != grain_after and not any(d.startswith("timestamps") for d in diff):
            diff.append(f"timestamps: {grain_before} -> {grain_after}")
        changed.append((variant, diff))
        if not args.dry_run:
            record["capabilities"] = caps
            common.write_record(common.CATALOG_DIR / f"{variant}.json", record)

    print(f"read {sum(sources.values())} GGUF(s): "
          + ", ".join(f"{count} {where}" for where, count in sources.most_common()))
    print(f"{len(changed)} record(s) corrected\n")
    for variant, diff in changed:
        print(f"  {variant:42s} {'; '.join(diff) or 'payload only'}")
    if disagree:
        print(f"\n{len(disagree)} record(s) whose quants disagree with each other "
              f"(not updated):")
        for variant, files in disagree:
            print(f"  {variant:42s} {', '.join(files)}")
    if missing:
        # Every converter now writes every capability KV; a file without one
        # predates that and inherits the family default, which is exactly the
        # silence that let wrong flags ship. Fatal at ship time, a warning in
        # the sweep until the published files are re-exported.
        print(f"\n{len(missing)} file(s) missing capability KV(s)"
              + (":" if strict else " (warning; --check --models makes this fatal):"))
        for variant, filename, absent in missing:
            print(f"  {variant}/{filename}: {', '.join(absent)}")
    if unreachable:
        print(f"\n{len(unreachable)} file(s) not readable:")
        for variant, error in unreachable:
            print(f"  {variant:42s} {error}")
    if args.dry_run:
        print("\ndry run: nothing written")
    failed = bool(changed or disagree) if args.check else False
    if strict and (missing or unreachable):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
