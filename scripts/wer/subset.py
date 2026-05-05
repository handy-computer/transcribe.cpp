#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""subset.py — first-N-in-order subset of a WER acceptance manifest.

Takes the first N non-empty manifest rows in the order they appear in
the source file. If the manifest has fewer than N rows, the entire
manifest is copied through. The same source file always produces the
same output (manifest order is the only ordering); no hashing.

Used by porting-4-cpp Step 9 (subset WER sanity) so C++ and the
reference framework score against the same small subset.

Usage:
    uv run scripts/wer/subset.py \\
        --manifest samples/wer/librispeech-test-clean.manifest.jsonl \\
        --n 512 \\
        --out samples/wer/librispeech-test-clean.512.manifest.jsonl

Re-running with the same arguments is a no-op when the output already
exists; pass --force to regenerate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, type=Path,
                   help="Source manifest.jsonl path")
    p.add_argument("--n", type=int, default=512,
                   help="Subset size (default: 512)")
    p.add_argument("--out", required=True, type=Path,
                   help="Output manifest.jsonl path")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if --out already exists")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    if args.out.exists() and not args.force:
        print(f"OK already exists: {args.out} (pass --force to regenerate)")
        return 0

    entries: list[dict] = []
    with args.manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    if not entries:
        print(f"error: manifest is empty: {args.manifest}", file=sys.stderr)
        return 1

    if len(entries) <= args.n:
        subset = entries
        print(f"manifest has {len(entries)} entries, <= {args.n}; using full manifest")
    else:
        subset = entries[:args.n]
        print(f"selected first {len(subset)} of {len(entries)} entries (manifest order)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for e in subset:
            f.write(json.dumps(e) + "\n")
    print(f"wrote {args.out} ({len(subset)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
