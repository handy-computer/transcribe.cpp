#!/usr/bin/env python3
"""
run_reference_sortformer_nemo.py - measured Sortformer reference baseline
over a diarization manifest (diarization analog of
scripts/wer/run_reference_<family>_*.py).

Runs NeMo streaming Sortformer over every meeting in an AMI-style manifest
and writes one predicted RTTM per meeting plus a summary JSONL. Score with
scripts/diar/score_der.py. This is the ONE-TIME reference run: Stage 4 and
Stage 7 compare the C++ port's DER/JER against this measured baseline, not
against the publisher's number.

Streaming presets (frames are 80 ms) from the v2.1 model card:
  very_high_latency  30.4s  chunk_len=340 rc=40 fifo=40  update=300 spkcache=188  (fastest; card AMI-IHM DER 15.90)
  low_latency        1.04s  chunk_len=6   rc=7  fifo=188 update=144 spkcache=188  (card AMI-IHM DER 16.67)

Run through the Sortformer reference env:
  uv run --project scripts/envs/sortformer \
    scripts/diar/run_reference_sortformer_nemo.py \
    --manifest samples/diar/ami-ihm-test.manifest.jsonl \
    --model nvidia/diar_streaming_sortformer_4spk-v2.1 \
    --preset very_high_latency \
    --pred-dir reports/diar/pred/diar_streaming_sortformer_4spk-v2.1-ami-ihm-test \
    --out reports/diar/diar_streaming_sortformer_4spk-v2.1-REF.ami-ihm-test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PRESETS = {
    "very_high_latency": dict(chunk_len=340, chunk_right_context=40, fifo_len=40,
                              spkcache_update_period=300, spkcache_len=188),
    "high_latency": dict(chunk_len=124, chunk_right_context=1, fifo_len=124,
                         spkcache_update_period=124, spkcache_len=188),
    "low_latency": dict(chunk_len=6, chunk_right_context=7, fifo_len=188,
                        spkcache_update_period=144, spkcache_len=188),
}


def _apply_preset(m, preset: str) -> dict:
    cfg = PRESETS[preset]
    sm = m.sortformer_modules
    sm.chunk_len = cfg["chunk_len"]
    sm.chunk_right_context = cfg["chunk_right_context"]
    sm.fifo_len = cfg["fifo_len"]
    sm.spkcache_update_period = cfg["spkcache_update_period"]
    sm.spkcache_len = cfg["spkcache_len"]
    if hasattr(sm, "_check_streaming_parameters"):
        sm._check_streaming_parameters()
    return cfg


def _seg_to_rttm(uri: str, segments) -> list[str]:
    """NeMo diarize segments str() as 'start end speaker_k'."""
    lines = []
    for s in segments:
        parts = str(s).strip().split()
        if len(parts) < 3:
            continue
        start, end, spk = float(parts[0]), float(parts[1]), parts[2]
        dur = end - start
        if dur <= 0:
            continue
        lines.append(f"SPEAKER {uri} 1 {start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset", default="very_high_latency", choices=list(PRESETS))
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--postprocessing-yaml", default=None,
                    help="Optional NeMo diarization post-processing YAML (onset/offset/pad/min_duration).")
    ap.add_argument("--limit", type=int, default=0, help="Run only the first N meetings (smoke).")
    args = ap.parse_args()

    import torch
    from nemo.collections.asr.models import SortformerEncLabelModel

    repo = Path(__file__).resolve().parent.parent.parent
    entries = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        entries = entries[:args.limit]
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = SortformerEncLabelModel.from_pretrained(args.model, map_location=args.device)
    m.eval()
    cfg = _apply_preset(m, args.preset)
    print(f"preset={args.preset} {cfg}", flush=True)

    rows = []
    for i, e in enumerate(entries):
        wav = str(repo / e["audio"]) if not Path(e["audio"]).is_absolute() else e["audio"]
        uri = e["id"]
        t0 = time.time()
        kw = {}
        if args.postprocessing_yaml:
            kw["postprocessing_yaml"] = args.postprocessing_yaml
        segs = m.diarize(audio=[wav], batch_size=1, **kw)
        dt = time.time() - t0
        seglist = segs[0] if segs else []
        lines = _seg_to_rttm(uri, seglist)
        (pred_dir / f"{uri}.rttm").write_text("\n".join(lines) + "\n")
        rtf = dt / max(e.get("duration", 1.0), 1e-6)
        rows.append({"id": uri, "hyp_rttm": str(Path(args.pred_dir) / f"{uri}.rttm"),
                     "n_segments": len(lines), "duration": e.get("duration"),
                     "compute_sec": round(dt, 2), "rtf": round(rtf, 4)})
        print(f"  [{i}] {uri}: {len(lines)} segs, {dt:.1f}s (rtf {rtf:.3f})", flush=True)

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path} ({len(rows)} meetings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
