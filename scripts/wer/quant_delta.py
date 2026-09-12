#!/usr/bin/env python3
"""quant_delta.py — paired Q8_0-vs-lower-quant degradation on FLEURS.

Why paired: a quant A/B runs both arms over the SAME utterances, so the two
error rates are highly correlated. Comparing their independent confidence
intervals throws that correlation away and is badly underpowered; the
repo's single-run noise floor (~0.3pp) then swamps deltas that are in fact
perfectly resolvable. Resampling the shared utterance set once and
recomputing both arms on that same resample keeps the pairing and measures
the delta directly.

Usage:
    uv run scripts/wer/quant_delta.py --quants Q5_K_M,Q4_K_M
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import random
import statistics
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
WER = REPO / "reports" / "wer"

# Architecture family, for the "does quant sensitivity track architecture"
# question. Conformer CTC/RNNT keep their conv stacks in F32/F16 (no
# quantized im2col in ggml), so they are expected to be the robust end.
FAMILY = {
    "whisper-large-v3": "enc-dec (tied embed)",
    "canary-1b-v2": "enc-dec",
    "parakeet-tdt-0.6b-v3": "RNNT conformer",
    "Qwen3-ASR-1.7B": "LLM decoder (tied embed)",
    "Fun-ASR-MLT-Nano-2512": "enc-dec (small)",
    "Voxtral-Mini-4B-Realtime-2602": "LLM decoder",
}


def per_utt(model: str, quant: str, lang: str) -> tuple[dict, str] | None:
    """{utt_id: (errors, ref_len)} plus the metric actually used."""
    hits = glob.glob(str(WER / f"{model}-{quant}.fleurs-{lang}.b8.score.json"))
    if not hits:
        return None
    d = json.loads(pathlib.Path(hits[0]).read_text())
    metric = "cer" if d.get("cer") is not None and d.get("wer") is None else \
        ("cer" if lang in {"zh", "yue", "ja", "ko", "th"} else "wer")
    out = {}
    empty = 0
    for u in d.get("per_utterance", []):
        rate = u.get(metric)
        if rate is None:
            return None
        ref = u["ref"]
        n = len(ref.split()) if metric == "wer" else len(ref.replace(" ", ""))
        if not u.get("hyp", "").strip():
            empty += 1
        if n:
            out[u["id"]] = (rate * n, n)
    return (out, metric, empty) if out else None


def agg(d: dict, ids: list[str]) -> float:
    e = sum(d[i][0] for i in ids)
    n = sum(d[i][1] for i in ids)
    return 100.0 * e / n if n else float("nan")


def paired_delta(base: dict, cand: dict, n_boot: int = 2000, seed: int = 42):
    """(base_err, cand_err, rel_delta_pct, lo, hi, n_utts) at 95%.

    rel_delta is (cand-base)/base in percent. CI is on that ratio, from a
    paired resample of the shared utterances."""
    ids = sorted(set(base) & set(cand))
    if len(ids) < 30:
        return None
    b0, c0 = agg(base, ids), agg(cand, ids)
    rng = random.Random(seed)
    rels = []
    for _ in range(n_boot):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        b, c = agg(base, s), agg(cand, s)
        if b > 0:
            rels.append(100.0 * (c - b) / b)
    rels.sort()
    lo = rels[int(0.025 * len(rels))]
    hi = rels[int(0.975 * len(rels))]
    return b0, c0, 100.0 * (c0 - b0) / b0, lo, hi, len(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quants", default="Q5_K_M,Q4_K_M")
    ap.add_argument("--base", default="Q8_0")
    ap.add_argument("--pairs-file", default="")
    args = ap.parse_args()
    quants = [q.strip() for q in args.quants.split(",") if q.strip()]

    # Discover cells from what actually scored, so a partially-complete
    # sweep still reports.
    cells = defaultdict(set)
    for p in glob.glob(str(WER / "*.fleurs-*.b8.score.json")):
        name = pathlib.Path(p).name
        for q in quants:
            if f"-{q}.fleurs-" in name:
                model = name.split(f"-{q}.fleurs-")[0]
                lang = name.split(".fleurs-")[1].split(".")[0]
                cells[model].add(lang)

    if not cells:
        print("no scored cells found for quants " + ",".join(quants))
        return 1

    rows = []
    print(f"{'model':30} {'lang':5} {'met':4} {'quant':8} "
          f"{'Q8_0':>7} {'cand':>7} {'rel%':>8} {'95% CI':>18} {'n':>5}")
    print("-" * 104)
    for model in sorted(cells):
        for lang in sorted(cells[model]):
            base = per_utt(model, args.base, lang)
            if not base:
                print(f"{model:30} {lang:5} -- no {args.base} baseline")
                continue
            for q in quants:
                cand = per_utt(model, q, lang)
                if not cand:
                    continue
                if cand[1] != base[1]:
                    print(f"{model:30} {lang:5} !! metric mismatch")
                    continue
                r = paired_delta(base[0], cand[0])
                if not r:
                    continue
                b0, c0, rel, lo, hi, n = r
                sig = "" if (lo <= 0 <= hi) else ("  *" if rel > 0 else "  +")
                # An empty hypothesis scores 100% and is a generation failure,
                # not transcription damage. If the empty count moves between
                # quants the WER delta is partly measuring that, so surface it
                # rather than let it sit inside the aggregate unlabelled.
                de = cand[2] - base[2]
                if base[2] or cand[2]:
                    sig += f"  [empty {base[2]}->{cand[2]}]"
                print(f"{model:30} {lang:5} {base[1]:4} {q:8} "
                      f"{b0:7.2f} {c0:7.2f} {rel:+8.2f} "
                      f"[{lo:+7.2f},{hi:+7.2f}] {n:5}{sig}")
                rows.append(dict(model=model, lang=lang, metric=base[1],
                                 quant=q, base=b0, cand=c0, rel=rel,
                                 lo=lo, hi=hi, n=n,
                                 family=FAMILY.get(model, "?"),
                                 is_en=(lang == "en"),
                                 empty_base=base[2], empty_cand=cand[2],
                                 empty_delta=de))
    print("\n  * = degradation excludes zero (real)   + = improvement excludes zero")

    out = WER / "quant_delta.json"
    out.write_text(json.dumps(rows, indent=1))

    # Aggregates: the two questions that drove the sweep.
    for q in quants:
        sub = [r for r in rows if r["quant"] == q]
        if not sub:
            continue
        en = [r["rel"] for r in sub if r["is_en"]]
        non = [r["rel"] for r in sub if not r["is_en"]]
        print(f"\n=== {q} ===")
        if en:
            print(f"  English      n={len(en):3}  mean rel {statistics.mean(en):+6.2f}%  "
                  f"max {max(en):+6.2f}%")
        if non:
            print(f"  non-English  n={len(non):3}  mean rel {statistics.mean(non):+6.2f}%  "
                  f"max {max(non):+6.2f}%")
        sigbad = [r for r in sub if r["lo"] > 0]
        print(f"  cells with a REAL (CI-excluding-zero) degradation: "
              f"{len(sigbad)}/{len(sub)}")
        byf = defaultdict(list)
        for r in sub:
            byf[r["family"]].append(r["rel"])
        print("  by architecture family:")
        for f, v in sorted(byf.items(), key=lambda x: -statistics.mean(x[1])):
            print(f"    {f:28} n={len(v):3}  mean {statistics.mean(v):+6.2f}%  "
                  f"max {max(v):+6.2f}%")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
