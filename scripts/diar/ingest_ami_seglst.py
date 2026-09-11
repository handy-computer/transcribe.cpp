#!/usr/bin/env python3
"""
ingest_ami_seglst.py - build SegLST reference transcripts for the AMI
acceptance manifest.

cpWER needs speaker-attributed reference TRANSCRIPTS; the
diarizers-community/ami dataset behind ingest_ami.py ships only RTTM
(speaker turns, no words). This pulls the per-segment manual transcripts
of edinburghcstr/ami via the HF datasets-server rows API (JSON, text
columns only — the parquet shards embed segmented audio and would cost
tens of GB of cache), groups them by meeting, and writes one SegLST JSON
per meeting, plus a manifest joining audio + rttm + seglst.

Reads  samples/diar/ami-<config>-<split>.manifest.jsonl   (ingest_ami.py)
Writes samples/diar/ami-<config>-<split>-seglst/<meeting>.seglst.json
       samples/diar/ami-<config>-<split>-mt.manifest.jsonl

Usage:
    uv run scripts/diar/ingest_ami_seglst.py --config ihm --split test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
API = "https://datasets-server.huggingface.co/rows"
PAGE = 100


def hf_token() -> str | None:
    import os

    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = Path.home() / ".cache" / "huggingface" / "token"
    return p.read_text().strip() if p.exists() else None


def fetch_page(config: str, split: str, offset: int, cache_dir: Path, token: str | None) -> dict:
    cache = cache_dir / f"{config}-{split}-{offset}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    q = urllib.parse.urlencode(
        {"dataset": "edinburghcstr/ami", "config": config, "split": split,
         "offset": offset, "length": PAGE}
    )
    req = urllib.request.Request(f"{API}?{q}")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for attempt in range(10):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = json.load(r)
            cache.write_text(json.dumps(payload))
            return payload
        except urllib.error.HTTPError as exc:
            if attempt == 9:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            wait = float(retry_after) if retry_after else min(60.0, 2.0 ** attempt)
            print(f"HTTP {exc.code} offset={offset}, waiting {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
        except Exception as exc:
            if attempt == 9:
                raise
            time.sleep(min(60.0, 2.0 ** attempt))
            print(f"retry offset={offset}: {exc}", file=sys.stderr)
    raise RuntimeError("unreachable")


def fetch_rows(config: str, split: str, cache_dir: Path):
    token = hf_token()
    offset = 0
    while True:
        payload = fetch_page(config, split, offset, cache_dir, token)
        rows = payload.get("rows", [])
        if not rows:
            return
        for r in rows:
            yield r["row"]
        offset += len(rows)
        if offset >= payload.get("num_rows_total", 0):
            return
        time.sleep(0.75)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="ihm", choices=["ihm", "sdm"])
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    base = REPO / "samples" / "diar"
    src_manifest = base / f"ami-{args.config}-{args.split}.manifest.jsonl"
    if not src_manifest.exists():
        print(f"error: {src_manifest} missing (run ingest_ami.py first)", file=sys.stderr)
        return 1
    entries = [json.loads(l) for l in src_manifest.read_text().splitlines() if l.strip()]
    meetings = {e["id"].split(".")[0]: e for e in entries}

    cache_dir = base / ".rows-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_meeting: dict[str, list[dict]] = defaultdict(list)
    n_rows = 0
    for row in fetch_rows(args.config, args.split, cache_dir):
        mid = row.get("meeting_id")
        text = (row.get("text") or "").strip()
        if mid not in meetings or not text:
            continue
        per_meeting[mid].append(
            {
                "session_id": mid,
                "speaker": row["speaker_id"],
                "start_time": float(row["begin_time"]),
                "end_time": float(row["end_time"]),
                "words": text,
            }
        )
        n_rows += 1
        if n_rows % 2000 == 0:
            print(f"...{n_rows} segments", file=sys.stderr)

    missing = sorted(set(meetings) - set(per_meeting))
    if missing:
        print(f"warning: no transcript rows for {missing}", file=sys.stderr)

    out_dir = base / f"ami-{args.config}-{args.split}-seglst"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = base / f"ami-{args.config}-{args.split}-mt.manifest.jsonl"
    with out_manifest.open("w") as mf:
        for mid, entry in sorted(meetings.items()):
            segs = sorted(per_meeting.get(mid, []), key=lambda s: s["start_time"])
            seglst_path = out_dir / f"{mid}.seglst.json"
            seglst_path.write_text(json.dumps(segs, indent=1) + "\n")
            row = dict(entry)
            row["seglst"] = str(seglst_path.relative_to(REPO))
            mf.write(json.dumps(row) + "\n")
            print(f"{mid}: {len(segs)} reference segments")

    print(f"wrote {out_manifest} ({len(meetings)} meetings, {n_rows} segments total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
