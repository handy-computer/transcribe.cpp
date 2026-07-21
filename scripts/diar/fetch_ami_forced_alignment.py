#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
fetch_ami_forced_alignment.py - fetch NVIDIA's forced-alignment AMI RTTMs and
build a forced-alignment acceptance manifest.

NVIDIA's published AMI DER uses tight forced-alignment `only_words` RTTMs
(nttcslab-sp/diar-forced-alignment, ASRU 2025), NOT the denser manual
annotations shipped in diarizers-community/ami. Scoring against the manual
RTTMs inflates DER by ~13 pts (missed detection on within-utterance pauses);
scoring against these forced-alignment RTTMs reproduces the published number.

Reads samples/diar/ami-<config>-<split>.manifest.jsonl (from ingest_ami.py),
downloads the matching forced-alignment RTTM per meeting, and writes:

  samples/diar/ami-<config>-<split>-fa/<meeting>.rttm
  samples/diar/ami-<config>-<split>-fa.manifest.jsonl   (same audio, FA rttm)

Usage:
  uv run scripts/diar/fetch_ami_forced_alignment.py --config ihm --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
RAW = "https://raw.githubusercontent.com/nttcslab-sp/diar-forced-alignment/master/AMI/{split}/{meeting}.rttm"
# nttcslab uses the AMI train/dev/eval split; diarizers-community uses train/validation/test.
SPLIT_MAP = {"test": "test", "validation": "dev", "train": "train"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="ihm", choices=["ihm", "sdm"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    ds_id = f"ami-{args.config}-{args.split}"
    src_manifest = REPO / "samples" / "diar" / f"{ds_id}.manifest.jsonl"
    if not src_manifest.exists():
        print(f"error: {src_manifest} not found; run ingest_ami.py first.", file=sys.stderr)
        return 1
    fa_split = SPLIT_MAP[args.split]
    fa_dir = REPO / "samples" / "diar" / f"{ds_id}-fa"
    fa_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(src_manifest) if l.strip()]
    out = []
    for r in rows:
        meeting = r["id"].replace(".Mix-Headset", "").replace(".Array1-01", "")
        url = RAW.format(split=fa_split, meeting=meeting)
        dst = fa_dir / f"{meeting}.rttm"
        urllib.request.urlretrieve(url, dst)
        n = sum(1 for l in dst.read_text().splitlines() if l.startswith("SPEAKER"))
        out.append({**r, "rttm": str(dst.relative_to(REPO))})
        print(f"  {meeting}: {n} forced-alignment turns", flush=True)

    fa_manifest = REPO / "samples" / "diar" / f"{ds_id}-fa.manifest.jsonl"
    fa_manifest.write_text("\n".join(json.dumps(o) for o in out) + "\n")
    print(f"wrote {fa_manifest.relative_to(REPO)} ({len(out)} meetings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
