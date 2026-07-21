#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyannote.metrics>=3.2",
# ]
# ///
"""
score_der.py - Diarization Error Rate + Jaccard Error Rate scorer.

Diarization analog of scripts/wer/score.py. Compares predicted speaker
RTTMs (from scripts/diar/run_reference_sortformer_nemo.py, or later the C++
port) against the ground-truth RTTMs named in an AMI-style manifest.

DER and JER are computed with pyannote.metrics. Collar and overlap handling
mirror NVIDIA's AMI protocol: collar 0.0s, overlap INCLUDED (skip_overlap
False). The aggregate is pyannote's speech-time-weighted global figure, not
a mean of per-meeting rates.

Writes <input-stem>.score.json:
    {metric:"der", collar, skip_overlap, der, jer, der_pct, jer_pct,
     n_meetings, total_speech_sec, missed_pct, false_alarm_pct,
     confusion_pct, per_meeting:[{id, der, jer, ...}]}

Usage:
  uv run scripts/diar/score_der.py \
    --manifest samples/diar/ami-ihm-test.manifest.jsonl \
    --pred-dir reports/diar/pred/streaming-4spk-v2.1-ami-ihm-test \
    --out reports/diar/streaming-4spk-v2.1-REF.ami-ihm-test.score.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


def load_rttm(path: Path, uri: str) -> Annotation:
    ann = Annotation(uri=uri)
    if not path.exists():
        return ann
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 8 or p[0] != "SPEAKER":
            continue
        start, dur, spk = float(p[3]), float(p[4]), p[7]
        if dur <= 0:
            continue
        ann[Segment(start, start + dur)] = spk
    return ann


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--collar", type=float, default=0.0)
    ap.add_argument("--skip-overlap", action="store_true",
                    help="Exclude overlapped regions from scoring (default: include, matching NVIDIA AMI).")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    pred_dir = Path(args.pred_dir)
    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]

    der = DiarizationErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)
    jer = JaccardErrorRate(collar=args.collar, skip_overlap=args.skip_overlap)

    per_meeting = []
    for e in entries:
        uri = e["id"]
        ref = load_rttm(repo / e["rttm"], uri)
        hyp = load_rttm(pred_dir / f"{uri}.rttm", uri)
        comp = der(ref, hyp, detailed=True)
        d_i = comp["diarization error rate"]
        j_i = jer(ref, hyp)
        per_meeting.append({
            "id": uri,
            "der": round(d_i, 5),
            "jer": round(j_i, 5),
            "missed": round(comp["missed detection"], 3),
            "false_alarm": round(comp["false alarm"], 3),
            "confusion": round(comp["confusion"], 3),
            "total": round(comp["total"], 3),
            "num_speakers": e.get("num_speakers"),
        })
        print(f"  {uri}: DER {d_i*100:.2f}%  JER {j_i*100:.2f}%", flush=True)

    # Global (speech-time-weighted) aggregates.
    agg_der = abs(der)
    agg_jer = abs(jer)
    comp = der[:]  # accumulated components
    total = sum(pm["total"] for pm in per_meeting)
    missed = sum(pm["missed"] for pm in per_meeting)
    fa = sum(pm["false_alarm"] for pm in per_meeting)
    conf = sum(pm["confusion"] for pm in per_meeting)

    out = {
        "metric": "der",
        "collar": args.collar,
        "skip_overlap": bool(args.skip_overlap),
        "der": round(agg_der, 5),
        "jer": round(agg_jer, 5),
        "der_pct": round(agg_der * 100, 2),
        "jer_pct": round(agg_jer * 100, 2),
        "n_meetings": len(per_meeting),
        "total_speech_sec": round(total, 2),
        "missed_pct": round(missed / total * 100, 2) if total else None,
        "false_alarm_pct": round(fa / total * 100, 2) if total else None,
        "confusion_pct": round(conf / total * 100, 2) if total else None,
        "per_meeting": per_meeting,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nAGGREGATE DER {out['der_pct']}%  JER {out['jer_pct']}%  "
          f"(missed {out['missed_pct']}%, FA {out['false_alarm_pct']}%, conf {out['confusion_pct']}%) "
          f"over {out['n_meetings']} meetings")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
