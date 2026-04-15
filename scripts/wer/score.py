#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jiwer>=3.0",
#     "whisper-normalizer",
# ]
# ///
"""
score.py — compute WER + bootstrap CI from a run.py report.

Usage:
    uv run scripts/wer/score.py \\
        reports/wer/parakeet-tdt-0.6b-v2-F32.test-clean.jsonl

Writes a .score.json alongside the input with:
    {wer, wer_ci_lo, wer_ci_hi, n, substitutions, deletions, insertions,
     per_utterance: [{id, ref, hyp, wer}, ...]}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import jiwer
from whisper_normalizer.english import EnglishTextNormalizer


def bootstrap_wer_ci(
    refs: list[str],
    hyps: list[str],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap 95% CI for WER. Returns (lo, hi)."""
    rng = random.Random(seed)
    n = len(refs)
    wers: list[float] = []
    for _ in range(n_boot):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        r = [refs[i] for i in indices]
        h = [hyps[i] for i in indices]
        w = jiwer.wer(r, h)
        wers.append(w)
    wers.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    return wers[lo_idx], wers[min(hi_idx, n_boot - 1)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", type=Path,
                   help="run.py output JSONL file")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iterations (default 1000)")
    args = p.parse_args()

    if not args.report.exists():
        print(f"error: {args.report} does not exist", file=sys.stderr)
        return 2

    # Load report. Skip the optional batch_header record (see
    # scripts/wer/run.py — first line carries load_ms, not a transcript).
    entries: list[dict] = []
    with open(args.report) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") == "batch_header":
                continue
            entries.append(rec)

    if not entries:
        print("error: report is empty", file=sys.stderr)
        return 2

    # Normalize refs and hyps.
    normalizer = EnglishTextNormalizer()
    refs_raw  = [e["ref_text"] for e in entries]
    hyps_raw  = [e["hyp_text"] for e in entries]
    refs_norm = [normalizer(r) for r in refs_raw]
    hyps_norm = [normalizer(h) for h in hyps_raw]

    # Skip empty refs (shouldn't happen in LibriSpeech but be safe).
    valid = [(r, h, e) for r, h, e in zip(refs_norm, hyps_norm, entries)
             if r.strip()]
    if not valid:
        print("error: no valid utterances after normalization", file=sys.stderr)
        return 2

    refs = [v[0] for v in valid]
    hyps = [v[1] for v in valid]
    ids  = [v[2]["id"] for v in valid]

    # Global WER.
    measures = jiwer.process_words(refs, hyps)
    global_wer = measures.wer
    substitutions = measures.substitutions
    deletions = measures.deletions
    insertions = measures.insertions

    # Per-utterance WER (for debugging / regression analysis).
    per_utt = []
    for uid, r, h in zip(ids, refs, hyps):
        u_wer = jiwer.process_words([r], [h]).wer if r.strip() else 0.0
        per_utt.append({"id": uid, "ref": r, "hyp": h, "wer": round(u_wer, 4)})

    # Bootstrap CI.
    ci_lo, ci_hi = bootstrap_wer_ci(refs, hyps, n_boot=args.n_boot)

    # Latency stats.
    latencies = [e["latency_ms"] for e in entries
                 if e.get("latency_ms", -1) > 0]
    lat_mean = sum(latencies) / len(latencies) if latencies else 0.0
    lat_p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    lat_p99 = sorted(latencies)[int(0.99 * len(latencies))] if latencies else 0.0

    n_errors = sum(1 for e in entries if e.get("error"))

    # Print summary.
    print(f"\n{'='*60}")
    print(f"  WER:   {global_wer*100:.2f}%  "
          f"(95% CI: [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%])")
    print(f"  N:     {len(refs)} utterances")
    print(f"  Sub:   {substitutions}  Del: {deletions}  Ins: {insertions}")
    print(f"  Errors (transcribe-cli failures): {n_errors}")
    print(f"  Latency: mean={lat_mean:.0f}ms "
          f"p50={lat_p50:.0f}ms p99={lat_p99:.0f}ms")
    print(f"{'='*60}\n")

    # Write score JSON.
    score_path = args.report.with_suffix(".score.json")
    score = {
        "wer": round(global_wer, 6),
        "wer_pct": round(global_wer * 100, 2),
        "wer_ci_lo": round(ci_lo, 6),
        "wer_ci_hi": round(ci_hi, 6),
        "n": len(refs),
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "n_errors": n_errors,
        "latency_mean_ms": round(lat_mean, 1),
        "latency_p50_ms": round(lat_p50, 1),
        "latency_p99_ms": round(lat_p99, 1),
        "report_file": str(args.report),
        "per_utterance": per_utt,
    }
    with open(score_path, "w") as f:
        json.dump(score, f, indent=2)
    print(f"score: {score_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
