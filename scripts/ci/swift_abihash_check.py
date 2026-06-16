#!/usr/bin/env python3
"""Fail if the Swift binding's pinned public-ABI hash drifts from the header.

The Swift binding does not generate an FFI layer (the Clang importer reads the C
headers directly), so the drift gate (notes/bindings-requirements.md §2) is a
PINNED constant — ``Transcribe.pinnedHeaderHash`` in
``bindings/swift/Sources/TranscribeCpp/ABIHash.swift`` — checked here against the
neutral ``include/transcribe.abihash`` emitted by the Python generator (the hash
oracle). When the header's ABI changes the neutral hash moves, this check goes
red, and a maintainer bumps the pinned constant after consciously reviewing the
change and auditing the wrapper.

    uv run --no-project scripts/ci/swift_abihash_check.py

Exit 0 when they agree; 1 on drift; 2 if either value could not be located.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ABIHASH_FILE = REPO / "include" / "transcribe.abihash"
PIN_FILE = REPO / "bindings" / "swift" / "Sources" / "TranscribeCpp" / "ABIHash.swift"


def main() -> int:
    if not ABIHASH_FILE.exists():
        print(f"error: missing {ABIHASH_FILE}", file=sys.stderr)
        return 2
    if not PIN_FILE.exists():
        print(f"error: missing {PIN_FILE}", file=sys.stderr)
        return 2

    neutral = ABIHASH_FILE.read_text().strip()
    m = re.search(r'pinnedHeaderHash\s*=\s*"([0-9a-fA-F]+)"', PIN_FILE.read_text())
    if not m:
        print(f"error: could not find pinnedHeaderHash in {PIN_FILE}", file=sys.stderr)
        return 2
    pinned = m.group(1)

    if pinned != neutral:
        print(
            "Swift ABI-hash drift: the public header ABI changed.\n"
            f"  include/transcribe.abihash : {neutral}\n"
            f"  ABIHash.swift (pinned)     : {pinned}\n"
            "Review the header change, audit the wrapper for new/changed structs,"
            " enums, or entry points, then update pinnedHeaderHash.",
            file=sys.stderr,
        )
        return 1

    print(f"swift abihash ok: {neutral}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
