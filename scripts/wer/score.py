#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jiwer>=3.0",
#     "whisper-normalizer",
# ]
# ///
"""
score.py — compute WER/CER + bootstrap CI from a run.py report.

Routes the metric and text normalizer by language:
    en              → WER + EnglishTextNormalizer
    zh/yue/ja/ko/th → CER + BasicTextNormalizer
    other           → WER + BasicTextNormalizer

Region suffixes (zh-tw, pt-br, ...) are stripped for routing, so
`--language zh-tw` and `--language zh` both pick CER on Mandarin.

Usage:
    uv run scripts/wer/score.py reports/wer/<...>.jsonl
    uv run scripts/wer/score.py reports/wer/<...>.jsonl --language vi
    uv run scripts/wer/score.py reports/wer/<...>.jsonl --language zh --metric cer

Writes a .score.json alongside the input with:
    {metric, language, error_rate, error_rate_pct, error_rate_ci_lo,
     error_rate_ci_hi, n, substitutions, deletions, insertions,
     per_utterance: [{id, ref, hyp, <metric>}, ...]}

For backward compatibility, WER reports also include `wer`, `wer_pct`,
`wer_ci_lo`, `wer_ci_hi` aliases so existing consumers (e.g.
porting-7-wer SKILL gate, compare.py) keep working.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

import jiwer
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer


# Languages where CER is the canonical metric. Region suffix is stripped
# before lookup so zh / zh-cn / zh-tw all resolve to the same set entry.
CER_LANGUAGES = {"zh", "yue", "ja", "ko", "th"}


def base_language(lang: str | None) -> str | None:
    if not lang:
        return None
    return lang.split("-", 1)[0].lower()


def resolve_metric(language: str | None, metric: str) -> str:
    """Resolve `auto` to `wer` or `cer` based on language."""
    if metric != "auto":
        return metric
    if base_language(language) in CER_LANGUAGES:
        return "cer"
    return "wer"


def resolve_normalizer(language: str | None):
    """Pick the text normalizer for the language.

    English (and a missing language hint, which we treat as English for
    backward compat with existing LibriSpeech reports) gets the full
    EnglishTextNormalizer: contraction expansion, English number words,
    English filler removal. Everything else gets BasicTextNormalizer:
    NFKC, punctuation strip, case fold, whitespace collapse.
    """
    base = base_language(language)
    if base in (None, "en"):
        return EnglishTextNormalizer()
    return BasicTextNormalizer()


def bootstrap_error_rate_ci(
    refs: list[str],
    hyps: list[str],
    metric_fn: Callable[[list[str], list[str]], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap CI for WER or CER. Returns (lo, hi)."""
    rng = random.Random(seed)
    n = len(refs)
    rates: list[float] = []
    for _ in range(n_boot):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        r = [refs[i] for i in indices]
        h = [hyps[i] for i in indices]
        rates.append(metric_fn(r, h))
    rates.sort()
    lo_idx = int((1 - ci) / 2 * n_boot)
    hi_idx = int((1 + ci) / 2 * n_boot)
    return rates[lo_idx], rates[min(hi_idx, n_boot - 1)]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", type=Path,
                   help="run.py output JSONL file")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Bootstrap iterations (default 1000)")
    p.add_argument("--language", default=None,
                   help="BCP-47 code (e.g. en, es, zh, zh-tw, vi). "
                        "Drives normalizer choice and CER auto-routing. "
                        "Default: en when omitted.")
    p.add_argument("--metric", choices=("wer", "cer", "auto"),
                   default="auto",
                   help="Error metric. 'auto' picks CER for "
                        "zh/yue/ja/ko/th and WER otherwise (default: auto)")
    args = p.parse_args()

    if not args.report.exists():
        print(f"error: {args.report} does not exist", file=sys.stderr)
        return 2

    language = args.language
    metric = resolve_metric(language, args.metric)
    normalizer = resolve_normalizer(language)

    # Load report. The batch_header record (first line, when present)
    # carries load_ms and the run-level language; the rest are
    # per-utterance results.
    entries: list[dict] = []
    header_language: str | None = None
    with open(args.report) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") == "batch_header":
                header_language = rec.get("language")
                continue
            entries.append(rec)

    # Inherit language from the report header when --language is unset.
    if language is None and header_language:
        language = header_language
        metric = resolve_metric(language, args.metric)
        normalizer = resolve_normalizer(language)
        print(f"language: inferred {language!r} from report header")

    if not entries:
        print("error: report is empty", file=sys.stderr)
        return 2

    # Normalize refs and hyps.
    refs_raw  = [e["ref_text"] for e in entries]
    hyps_raw  = [e["hyp_text"] for e in entries]
    refs_norm = [normalizer(r) for r in refs_raw]
    hyps_norm = [normalizer(h) for h in hyps_raw]

    # CER: strip all whitespace from both sides. Whitespace is not a
    # meaningful token for character error rate, and CJK conventions
    # vary widely (FLEURS spaces every character; Common Voice spaces
    # between phrases; model hypotheses depend on the family's
    # tokenizer). jiwer.process_characters otherwise treats space as a
    # token and inflates the score by 10x+ when the conventions differ.
    if metric == "cer":
        refs_norm = ["".join(r.split()) for r in refs_norm]
        hyps_norm = ["".join(h.split()) for h in hyps_norm]

    # Skip empty refs after normalization.
    valid = [(r, h, e) for r, h, e in zip(refs_norm, hyps_norm, entries)
             if r.strip()]
    if not valid:
        print("error: no valid utterances after normalization", file=sys.stderr)
        return 2

    refs = [v[0] for v in valid]
    hyps = [v[1] for v in valid]
    ids  = [v[2]["id"] for v in valid]

    # Global metric + sub/del/ins counts, plus per-utterance scorer.
    if metric == "wer":
        measures = jiwer.process_words(refs, hyps)
        global_rate = measures.wer
        metric_fn: Callable[[list[str], list[str]], float] = jiwer.wer
        per_utt_fn = lambda r, h: jiwer.process_words([r], [h]).wer
    else:
        measures = jiwer.process_characters(refs, hyps)
        global_rate = measures.cer
        metric_fn = jiwer.cer
        per_utt_fn = lambda r, h: jiwer.process_characters([r], [h]).cer
    substitutions = measures.substitutions
    deletions = measures.deletions
    insertions = measures.insertions

    # Per-utterance rate (for debugging / regression analysis).
    per_utt = []
    for uid, r, h in zip(ids, refs, hyps):
        u_rate = per_utt_fn(r, h) if r.strip() else 0.0
        per_utt.append({"id": uid, "ref": r, "hyp": h, metric: round(u_rate, 4)})

    # Bootstrap CI.
    ci_lo, ci_hi = bootstrap_error_rate_ci(
        refs, hyps, metric_fn, n_boot=args.n_boot
    )

    # Latency stats.
    latencies = [e["latency_ms"] for e in entries
                 if e.get("latency_ms", -1) > 0]
    lat_mean = sum(latencies) / len(latencies) if latencies else 0.0
    lat_p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    lat_p99 = sorted(latencies)[int(0.99 * len(latencies))] if latencies else 0.0

    n_errors = sum(1 for e in entries if e.get("error"))

    label = metric.upper()
    lang_display = language or "en (default)"
    print(f"\n{'='*60}")
    print(f"  {label}:   {global_rate*100:.2f}%  "
          f"(95% CI: [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%])")
    print(f"  language: {lang_display}")
    print(f"  N:     {len(refs)} utterances")
    print(f"  Sub:   {substitutions}  Del: {deletions}  Ins: {insertions}")
    print(f"  Errors (transcribe-cli failures): {n_errors}")
    print(f"  Latency: mean={lat_mean:.0f}ms "
          f"p50={lat_p50:.0f}ms p99={lat_p99:.0f}ms")
    print(f"{'='*60}\n")

    # Write score JSON.
    score_path = args.report.with_suffix(".score.json")
    score = {
        "metric": metric,
        "language": language,
        "error_rate": round(global_rate, 6),
        "error_rate_pct": round(global_rate * 100, 2),
        "error_rate_ci_lo": round(ci_lo, 6),
        "error_rate_ci_hi": round(ci_hi, 6),
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
    # Backward-compat aliases for WER reports. porting-7-wer SKILL.md
    # and compare.py read these field names directly.
    if metric == "wer":
        score["wer"] = score["error_rate"]
        score["wer_pct"] = score["error_rate_pct"]
        score["wer_ci_lo"] = score["error_rate_ci_lo"]
        score["wer_ci_hi"] = score["error_rate_ci_hi"]
    with open(score_path, "w") as f:
        json.dump(score, f, indent=2)
    print(f"score: {score_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
