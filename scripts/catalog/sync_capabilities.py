#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf", "huggingface-hub", "requests"]
# ///
"""Keep a record's `capabilities` block and its GGUF's KVs in agreement.

Two directions:

  (default)  GGUF -> catalog. Read the truth back out of the file.
  --repair   catalog -> GGUF. Rewrite a local GGUF so its stt.capability.*
             KVs declare what the record says, dropping any key spelled in a
             way no loader reads.

`--repair` exists because absence is not falsity. read_capability_bool()
returns OK and leaves the field ALONE when a key is missing, so a missing KV
inherits the family default -- and granite's default is deliberately
`supports_translate = true` so each variant's GGUF can lower it. The -plus
GGUF spelled that key `stt.capability.translation`, the lowering never
happened, and a model that does not translate has been advertising that it
does. Declaring every capability explicitly is what makes the file mean what
it says regardless of the loader's defaults.

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
    uv run scripts/catalog/sync_capabilities.py --repair <variant> [...]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
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


def open_gguf(record: dict, local_only: bool):
    """A GGUFReader over a local file, else over a range-fetched header."""
    from gguf import GGUFReader

    names = {item["filename"] for item in record["downloads"]}
    for directory in sorted((common.REPO / "models").glob("*")):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.gguf")):
            if path.name in names:
                return GGUFReader(str(path)), f"local {path.name}", None
    if local_only or not record.get("published_repo"):
        return None, None, "no local GGUF"

    # Header-only read: pad the temp file out to the declared size so
    # GGUFReader's memmap of the tensor region stays in bounds and is never
    # touched. Same trick as scripts/audit_gguf_metadata.py.
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    filename = sorted(record["downloads"], key=lambda d: d["size_bytes"])[0]["filename"]
    try:
        url = hf_hub_url(record["published_repo"], filename)
        total = get_hf_file_metadata(url).size
        prefix = min(24 * 1024 * 1024, total)
        headers = build_hf_headers()
        headers["Range"] = f"bytes=0-{prefix - 1}"
        response = requests.get(url, headers=headers, timeout=120)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - any transport failure is just "unavailable"
        return None, None, f"{type(exc).__name__}: {str(exc)[:70]}"
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
        tmp = pathlib.Path(handle.name)
        handle.write(response.content)
        handle.truncate(total)
    try:
        return GGUFReader(str(tmp)), f"hub {record['published_repo']}", None
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


# --------------------------------------------------------------------------
# repair: catalog -> GGUF


# Where each catalog capability lands as a KV. `timestamps` is deliberately
# absent: the ceiling is a family invariant in C++, not a per-file claim, and
# granite's per-variant lowering rides on word_timestamps instead.
REPAIR_KEYS = {
    "translate": "stt.capability.translate",
    "lang_detect": "stt.capability.lang_detect",
    "streaming": "stt.capability.streaming",
    "diarize": "stt.capability.speaker_diarization",
}
# Spellings that were shipped but that no loader reads. Dropped on repair so
# the file does not carry two contradictory answers.
DEAD_KEYS = {"stt.capability.translation"}


def identity_kvs(record: dict) -> dict[str, object]:
    """The general.* block a fresh conversion would emit for this record.

    Files converted before add_general_identity() landed in
    scripts/lib/gguf_common.py carry a bare slug instead of an identity, so
    an inspector cannot say who made the model or under what licence. Only
    name / basename / author / licence text come from the record's `identity`
    block; the rest is derived, because a second copy of a fact the catalog
    already holds is a second thing to drift.

    `general.size_label` is carried rather than computed from `params`: most
    converters bucket the parameter count, but the parakeet profiles hardcode
    a marketing label ("0.6B" for a 638M-parameter nemotron), so a derived
    value would not be what a fresh conversion writes.

    `general.languages` is deliberately NOT derived here: a file's advertised
    language list may narrow the record's on purpose (nemotron ships 40
    locales in its prompt table and advertises the 32 it can actually
    transcribe), so the record is the wrong source and the file already
    carries the right answer.
    """
    ident = record.get("identity")
    if not ident:
        return {}
    org, _, _ = record["upstream_repo"].partition("/")
    kvs: dict[str, object] = {
        "general.name": ident["name"],
        "general.basename": ident["basename"],
        "general.author": ident["author"],
        "general.organization": org,
        "general.license": record["license"]["spdx"],
        "general.repo_url": f"https://huggingface.co/{record['upstream_repo']}",
    }
    if ident.get("size_label"):
        kvs["general.size_label"] = ident["size_label"]
    if ident.get("license_name"):
        kvs["general.license.name"] = ident["license_name"]
    if ident.get("license_link"):
        kvs["general.license.link"] = ident["license_link"]
    return kvs


def repair_file(path: pathlib.Path, record: dict, out: pathlib.Path) -> dict:
    """Copy a GGUF, forcing its capability KVs to match the record.

    Tensor data is passed through untouched -- this changes what the file
    says about itself, never what it computes.

    Missing `general.*` identity keys are added too, written BEFORE the copy
    so they land ahead of the bulk tokenizer arrays: appending them would push
    a key past the trailer and cost a header range-read the very thing the
    trailer layout buys. An identity key the file already carries is left
    exactly as it is and reported, never overwritten -- the file was written
    by its converter and the catalog is the newcomer here.
    """
    from gguf import GGUFReader, GGUFValueType

    sys.path.insert(0, str(common.REPO / "scripts"))
    from lib.gguf_common import gguf_writer  # noqa: PLC0415

    reader = GGUFReader(str(path))
    arch = str(reader.fields["general.architecture"].contents())
    writer = gguf_writer(str(out), arch)

    want = {REPAIR_KEYS[name]: bool(block.get("supported"))
            for name, block in record["capabilities"].items()
            if name in REPAIR_KEYS}
    changed = {}

    for key, value in identity_kvs(record).items():
        if key not in reader.fields:
            writer.add_string(key, value)
            changed[key] = f"added = {value!r}"
        elif str(reader.fields[key].contents()) != str(value):
            changed[key] = (f"KEPT file value {reader.fields[key].contents()!r} "
                            f"(catalog says {value!r})")

    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        if key in DEAD_KEYS:
            changed[key] = "removed (read by no loader)"
            continue
        if key in want:
            if bool(field.contents()) != want[key]:
                changed[key] = f"{bool(field.contents())} -> {want[key]}"
            writer.add_bool(key, want[key])
            want.pop(key)
            continue
        vtype = field.types[0]
        sub = field.types[1] if len(field.types) > 1 else None
        writer.add_key_value(key, field.contents(),
                             vtype if vtype != GGUFValueType.ARRAY else vtype,
                             sub_type=sub)
    for key, value in want.items():          # capabilities the file never stated
        writer.add_bool(key, value)
        changed[key] = f"added = {value}"

    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return changed


def published_header(record: dict, filename: str):
    """A GGUFReader over the published file's header, range-fetched.

    Same trick as open_gguf(), but for one named file rather than the
    cheapest one, because the guard below has to compare like with like.
    """
    from gguf import GGUFReader
    import requests
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    url = hf_hub_url(record["published_repo"], filename)
    total = get_hf_file_metadata(url).size
    headers = build_hf_headers()
    headers["Range"] = f"bytes=0-{min(24 * 1024 * 1024, total) - 1}"
    response = requests.get(url, headers=headers, timeout=300)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as handle:
        tmp = pathlib.Path(handle.name)
        handle.write(response.content)
        handle.truncate(total)
    try:
        return GGUFReader(str(tmp))
    finally:
        tmp.unlink(missing_ok=True)


def divergence(local, published, intended: set[str]) -> list[str]:
    """Ways the local file differs from the published one beyond `intended`.

    A repair rewrites a local file and the result gets uploaded, so the local
    file is only a safe base if it IS what is published. A stale mirror looks
    identical to a repairable file -- same name, same quant, plausible KVs --
    and silently republishing it reverts whatever the published file gained
    since. That is not hypothetical: a stale granite-nar mirror here carried
    an older upstream snapshot (ctc_bpe 100353 vs the published 100352) and
    lacked stt.granite_nar.encoder.bpe_blank_id, so repairing and uploading it
    would have published different weights under an unchanged filename.

    Tensor shapes and dtypes are compared as well as KVs, since a different
    build is the case that actually matters and it shows up there first.
    """
    problems = []
    lt = {t.name: (tuple(int(x) for x in t.shape), t.tensor_type) for t in local.tensors}
    pt = {t.name: (tuple(int(x) for x in t.shape), t.tensor_type) for t in published.tensors}
    for name in sorted(pt.keys() - lt.keys()):
        problems.append(f"tensor {name} missing locally")
    for name in sorted(lt.keys() - pt.keys()):
        problems.append(f"tensor {name} not in the published file")
    for name in sorted(pt.keys() & lt.keys()):
        if pt[name] != lt[name]:
            problems.append(f"tensor {name}: published {pt[name][0]} "
                            f"{pt[name][1].name}, local {lt[name][0]} {lt[name][1].name}")
    for key in sorted(set(published.fields) | set(local.fields)):
        if key.startswith("GGUF.") or key in intended:
            continue
        here = local.fields[key].contents() if key in local.fields else None
        there = published.fields[key].contents() if key in published.fields else None
        if str(here)[:400] != str(there)[:400]:
            problems.append(f"{key}: published {str(there)[:60]!r}, local {str(here)[:60]!r}")
    return problems


def run_repair(variants: list[str], dry_run: bool, check_published: bool = True) -> int:
    records = common.load_records()
    names = variants or sorted(records)
    touched, refused = 0, 0
    for variant in names:
        record = records.get(variant)
        if record is None:
            print(f"  skip {variant}: no catalog record")
            continue
        for item in record["downloads"]:
            path = common.REPO / "models" / variant / item["filename"]
            if not path.exists():
                alt = next((p for p in (common.REPO / "models").glob(f"*/{item['filename']}")), None)
                path = alt if alt else path
            if not path.exists():
                print(f"  skip {variant}/{item['quant']}: {item['filename']} not on disk")
                continue
            intended = set(REPAIR_KEYS.values()) | DEAD_KEYS | set(identity_kvs(record))
            if check_published and record.get("published_repo"):
                try:
                    remote = published_header(record, item["filename"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  refuse {variant}/{item['quant']}: cannot read the "
                          f"published header to compare ({type(exc).__name__}: "
                          f"{str(exc)[:60]}); pass --skip-published-check to "
                          f"repair without the comparison")
                    refused += 1
                    continue
                from gguf import GGUFReader
                problems = divergence(GGUFReader(str(path)), remote, intended)
                if problems:
                    print(f"  REFUSE {variant}/{item['quant']}: local file is not "
                          f"what is published -- repairing it would republish a "
                          f"different build:")
                    for line in problems[:6]:
                        print(f"           {line}")
                    if len(problems) > 6:
                        print(f"           ... and {len(problems) - 6} more")
                    print(f"           re-download from {record['published_repo']} "
                          f"first, or pass --skip-published-check if the local "
                          f"file is deliberately newer")
                    refused += 1
                    continue
            out = path.with_suffix(".gguf.repaired")
            if dry_run:
                from gguf import GGUFReader
                kvs = read_kvs(GGUFReader(str(path)))
                want = {REPAIR_KEYS[n]: bool(b.get("supported"))
                        for n, b in record["capabilities"].items() if n in REPAIR_KEYS}
                diff = {k: f"{kvs.get(k)} -> {v}" for k, v in want.items() if kvs.get(k) != v}
                diff.update({k: "removed" for k in kvs if k in DEAD_KEYS})
                fields = GGUFReader(str(path)).fields
                for key, value in identity_kvs(record).items():
                    if key not in fields:
                        diff[key] = f"added = {value!r}"
                    elif str(fields[key].contents()) != str(value):
                        diff[key] = (f"KEPT file value {fields[key].contents()!r} "
                                     f"(catalog says {value!r})")
                print(f"  {variant}/{item['quant']}: {diff or 'already correct'}")
                continue
            changed = repair_file(path, record, out)
            os.replace(out, path)
            touched += 1
            print(f"  {variant}/{item['quant']}: {changed or 'no change'}")
    print(f"\n{touched} file(s) rewritten" + (" (dry run)" if dry_run else "")
          + (f", {refused} refused" if refused else ""))
    return 1 if refused else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-only", action="store_true",
                        help="skip the hub fallback")
    parser.add_argument("--repair", nargs="*", metavar="VARIANT",
                        help="rewrite local GGUFs so their capability KVs "
                             "declare what the catalog record says (all "
                             "variants when given no names)")
    parser.add_argument("--skip-published-check", action="store_true",
                        help="repair even when the local file diverges from "
                             "the published one. Only for a local file that is "
                             "deliberately newer than the Hub.")
    args = parser.parse_args()

    if args.repair is not None:
        return run_repair(args.repair, args.dry_run,
                          check_published=not args.skip_published_check)

    changed, unreachable, sources = [], [], collections.Counter()
    for variant, record in common.load_records().items():
        reader, source, error = open_gguf(record, args.local_only)
        if reader is None:
            unreachable.append((variant, error))
            continue
        sources[source.split()[0]] += 1
        caps = build(record, read_kvs(reader))
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
            (common.CATALOG_DIR / f"{variant}.json").write_text(
                common.dumps_record(record))

    print(f"read {sum(sources.values())} GGUF(s): "
          + ", ".join(f"{count} {where}" for where, count in sources.most_common()))
    print(f"{len(changed)} record(s) corrected\n")
    for variant, diff in changed:
        print(f"  {variant:42s} {'; '.join(diff) or 'payload only'}")
    if unreachable:
        print(f"\n{len(unreachable)} record(s) with no readable GGUF:")
        for variant, error in unreachable:
            print(f"  {variant:42s} {error}")
    if args.dry_run:
        print("\ndry run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
