#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["meeteval>=0.4"]
# ///
"""
score_cpwer.py - cpWER for speaker-attributed multitalker hypotheses.

Scores hypothesis SegLST files against the reference SegLSTs from
ingest_ami_seglst.py using meeteval's cpwer (concatenated
minimum-permutation WER — the metric NVIDIA reports for the multitalker
parakeet models). Text is normalized with a whisper-style English
normalizer to match the repo's WER convention.

Inputs: a -mt manifest (audio + rttm + seglst columns) and a hypothesis
directory containing <meeting>.seglst.json files (from the C++ or NeMo
multitalker runners).

Usage:
    uv run scripts/diar/score_cpwer.py \
      --manifest samples/diar/ami-ihm-test-mt.manifest.jsonl \
      --hyp-dir reports/cpwer/<run>/ \
      --out reports/cpwer/<run>.score.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import meeteval.io
from meeteval.wer import cpwer

REPO = Path(__file__).resolve().parent.parent.parent

_norm_re = re.compile(r"[^a-z0-9' ]+")


def normalize(text: str) -> str:
    # Lowercase, strip punctuation, collapse whitespace. Matches the
    # de-PnC convention score.py applies before WER.
    text = text.lower().replace("-", " ")
    text = _norm_re.sub(" ", text)
    return " ".join(text.split())


def load_seglst(path: Path, session_id: str) -> meeteval.io.SegLST:
    rows = json.loads(path.read_text())
    cleaned = []
    for r in rows:
        words = normalize(r.get("words", ""))
        if not words:
            continue
        cleaned.append(
            {
                "session_id": session_id,
                "speaker": str(r["speaker"]),
                "start_time": float(r.get("start_time", 0.0)),
                "end_time": float(r.get("end_time", 0.0)),
                "words": words,
            }
        )
    return meeteval.io.SegLST.new(cleaned)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyp-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hyp_dir = Path(args.hyp_dir)
    per_meeting = {}
    tot_err = tot_len = 0
    for line in Path(args.manifest).read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        mid = e["id"].split(".")[0]
        ref_path = REPO / e["seglst"]
        hyp_path = hyp_dir / f"{mid}.seglst.json"
        if not hyp_path.exists():
            print(f"warning: missing hyp for {mid}, skipping", file=sys.stderr)
            continue
        ref = load_seglst(ref_path, mid)
        hyp = load_seglst(hyp_path, mid)
        r = cpwer(ref, hyp)[mid]
        per_meeting[mid] = {
            "errors": r.errors,
            "length": r.length,
            "cpwer_pct": round(100.0 * r.error_rate, 3) if r.error_rate is not None else None,
            "insertions": r.insertions,
            "deletions": r.deletions,
            "substitutions": r.substitutions,
            "missed_speaker": r.missed_speaker,
            "falarm_speaker": r.falarm_speaker,
            "scored_speaker": r.scored_speaker,
        }
        tot_err += r.errors
        tot_len += r.length

    if tot_len == 0:
        print("error: nothing scored", file=sys.stderr)
        return 1
    summary = {
        "metric": "cpWER",
        "cpwer_pct": round(100.0 * tot_err / tot_len, 3),
        "errors": tot_err,
        "length": tot_len,
        "n_meetings": len(per_meeting),
        "per_meeting": per_meeting,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"cpWER: {summary['cpwer_pct']}% over {len(per_meeting)} meetings ({tot_err}/{tot_len})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
