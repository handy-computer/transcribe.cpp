#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["ruamel.yaml>=0.18"]
# ///
"""Populate committed Hugging Face YAML specs from the model catalog.

The YAML files remain complete, standalone inputs to hf_cards/generate.py.
This command fills mechanical fields that are absent from an editorial YAML
skeleton. Pass --refresh to deliberately replace those fields with current
catalog values; without it, already-published YAML values are preserved.
Fields listed under `catalog_sync.preserve` remain hand-maintained even during
a refresh.

    uv run scripts/catalog/sync_hf_cards.py --write
    uv run scripts/catalog/sync_hf_cards.py --write --refresh --models whisper-tiny
    uv run scripts/catalog/sync_hf_cards.py --check
    uv run scripts/catalog/sync_hf_cards.py --check-consistency
"""
from __future__ import annotations

import argparse
import copy
import pathlib
import sys

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cards  # noqa: E402

CARD_DIR = REPO / "scripts" / "hf_cards"
CATALOG_FIELDS = {
    "hf_repo", "target_repo", "upstream_commit", "license", "license_display",
    "languages", "capabilities", "perf", "quants",
}


def yaml_parser() -> YAML:
    parser = YAML()
    parser.preserve_quotes = True
    parser.width = 1000
    parser.indent(mapping=2, sequence=4, offset=2)
    return parser


def insert_before(doc: CommentedMap, before: str, key: str, value) -> None:
    keys = list(doc)
    index = keys.index(before) if before in keys else len(keys)
    doc.insert(index, key, copy.deepcopy(value))


def merge_quant_extras(derived: list[dict], existing: object) -> list[dict]:
    """Keep hand-written extra columns when refreshing catalog columns."""
    if not isinstance(existing, list):
        return derived
    by_name = {
        item.get("name"): item for item in existing
        if isinstance(item, dict) and item.get("name")
    }
    owned = {"name", "filename", "size", "wer"}
    out = []
    for item in derived:
        merged = dict(item)
        for key, value in by_name.get(item["name"], {}).items():
            if key not in owned:
                merged[key] = copy.deepcopy(value)
        out.append(merged)
    return out


def set_field(doc: CommentedMap, key: str, value, before: str, refresh: bool) -> bool:
    if key in doc and not refresh:
        return False
    if key in doc:
        if doc[key] == value:
            return False
        doc[key] = copy.deepcopy(value)
    else:
        insert_before(doc, before, key, value)
    return True


def preserved_fields(doc: CommentedMap) -> set[str]:
    """Catalog-owned fields this card deliberately keeps hand-maintained."""
    preserve = set((doc.get("catalog_sync") or {}).get("preserve", []))
    unknown = preserve - CATALOG_FIELDS
    if unknown:
        raise ValueError(f"unknown catalog_sync.preserve fields: {sorted(unknown)}")
    return preserve


def consistency_errors(doc: CommentedMap, record: dict) -> list[str]:
    """Catalog-owned fields whose committed value has drifted from the record.

    `--check` only proves every mechanical field is present. This proves the
    fields the card does NOT list under `catalog_sync.preserve` still agree
    with the catalog. Extra hand-written quant columns are presentation, not
    drift, so only the catalog-owned columns are compared.
    """
    preserve = preserved_fields(doc)
    derived = cards.derive_spec(record, doc)
    errors = []
    for key in sorted(CATALOG_FIELDS - {"quants"}):
        if key in preserve or key not in doc:
            continue
        if doc[key] != derived.get(key):
            errors.append(f"{key}: {doc[key]!r} != catalog {derived.get(key)!r}")
    if "quants" not in preserve:
        by_name = {item.get("name"): item for item in derived.get("quants", [])}
        for item in doc.get("quants") or []:
            current = by_name.get(item.get("name"))
            if current is None:
                errors.append(f"quants: {item.get('name')!r} is not in the catalog")
                continue
            for column in ("filename", "size", "wer"):
                if column in current and item.get(column) != current[column]:
                    errors.append(f"quants[{item.get('name')}].{column}: "
                                  f"{item.get(column)!r} != catalog {current[column]!r}")
    return errors


def sync_document(doc: CommentedMap, record: dict, refresh: bool) -> bool:
    """Fill or refresh catalog-owned fields; return whether values changed."""
    preserve = preserved_fields(doc)
    derived = cards.derive_spec(record, doc)
    derived["quants"] = merge_quant_extras(derived["quants"], doc.get("quants"))
    changed = False
    for key, before in (
        ("hf_repo", "transcribe_docs_url"),
        ("target_repo", "transcribe_docs_url"),
        ("upstream_commit", "pin_date"),
        ("license", "pipeline_tag"),
        ("license_display", "pipeline_tag"),
        ("languages", "tags"),
        ("capabilities", "wer"),
        ("perf", "wer"),
        ("quants", "__end__"),
    ):
        if key not in preserve:
            changed |= set_field(doc, key, derived[key], before, refresh)

    # The dataset label is catalog-derived only when the editorial spec has
    # not provided more precise display copy. Notes and extra datasets remain
    # hand-written in either mode.
    if derived.get("wer"):
        if "wer" not in doc:
            doc["wer"] = CommentedMap()
            changed = True
        if "source" not in doc["wer"]:
            doc["wer"].insert(0, "source", derived["wer"]["source"])
            changed = True
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="update YAML specs in place")
    mode.add_argument("--check", action="store_true", help="fail if required fields are absent")
    mode.add_argument("--check-consistency", action="store_true",
                      help="fail if a non-preserved catalog-owned field disagrees "
                           "with the catalog")
    ap.add_argument("--refresh", action="store_true",
                    help="replace existing mechanical fields (requires --write)")
    ap.add_argument("--models", default="", help="comma-separated variants (default: all)")
    args = ap.parse_args()
    if args.refresh and not args.write:
        ap.error("--refresh requires --write")

    selected = {item.strip() for item in args.models.split(",") if item.strip()}
    records = cards.common.load_records()
    unknown = selected - records.keys()
    if unknown:
        print(f"unknown catalog variant(s): {', '.join(sorted(unknown))}", file=sys.stderr)
        return 2

    parser = yaml_parser()
    changed_paths = []
    drifted = 0
    names = sorted(selected or records.keys())
    for name in names:
        path = CARD_DIR / f"{name}.yaml"
        if not path.exists():
            print(f"FAIL {name}: no editorial YAML skeleton at {path.relative_to(REPO)}",
                  file=sys.stderr)
            return 1
        doc = parser.load(path.read_text()) or CommentedMap()
        try:
            if args.check_consistency:
                errors = consistency_errors(doc, records[name])
            else:
                changed = sync_document(doc, records[name], args.refresh)
        except ValueError as exc:
            print(f"FAIL {name}: {exc}", file=sys.stderr)
            return 1

        if args.check_consistency:
            if errors:
                drifted += 1
                print(f"DRIFT {path.relative_to(REPO)}")
                for error in errors:
                    print(f"        {error}")
            continue

        if not changed:
            continue
        changed_paths.append(path)
        if args.write:
            with path.open("w") as stream:
                parser.dump(doc, stream)
            print(f"updated {path.relative_to(REPO)}")
        else:
            print(f"INCOMPLETE {path.relative_to(REPO)}")

    if args.check_consistency:
        print(f"HF card consistency: {len(names)} checked, "
              f"{len(names) - drifted} agree with the catalog, {drifted} drifted")
        return 1 if drifted else 0

    action = "updated" if args.write else "incomplete"
    print(f"HF card sync: {len(names)} checked, {len(changed_paths)} {action}")
    return 1 if changed_paths and args.check else 0


if __name__ == "__main__":
    sys.exit(main())
