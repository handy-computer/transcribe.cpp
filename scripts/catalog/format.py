#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Rewrite catalog records in the canonical layout, or check that they are.

The layout rules live in common.dumps_record, which every catalog writer
already uses; this is the same serializer applied to files edited by hand.

    uv run scripts/catalog/format.py            # rewrite every record
    uv run scripts/catalog/format.py --check    # exit 1 if any record differs
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("paths", nargs="*", type=pathlib.Path,
                        help="records to format (default: every catalog record)")
    args = parser.parse_args()
    paths = args.paths or sorted(
        path for path in common.CATALOG_DIR.glob("*.json") if not path.name.startswith("_"))
    stale = []
    for path in paths:
        current = path.read_text()
        rendered = common.dumps_record(json.loads(current))
        if current == rendered:
            continue
        stale.append(path)
        if not args.check:
            path.write_text(rendered)
    verb = "need formatting" if args.check else "rewritten"
    print(f"{len(paths)} record(s) checked; {len(stale)} {verb}")
    for path in stale:
        print(f"  {path.relative_to(common.REPO)}")
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
