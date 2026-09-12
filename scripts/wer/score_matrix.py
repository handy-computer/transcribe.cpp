#!/usr/bin/env python3
"""score_matrix.py — score every FLEURS hyp file and assemble the model x
language matrix.

Scoring is driven from the filename's dataset id, NOT from the report
header. score.py infers the language from the header when --language is
omitted, and a header stamped "auto" (any model run without an explicit
language tag) silently falls back to the English normalizer and WER, which
would score Japanese with the wrong metric and look plausible. Passing
--language explicitly for every file removes that failure mode.

Usage:
    uv run scripts/wer/score_matrix.py                    # score + print matrix
    uv run scripts/wer/score_matrix.py --no-score         # re-read existing scores
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
WER_DIR = REPO / "reports" / "wer"

# Hyp files look like <slug>.<dataset-id>[.b8][.ts-..][...].jsonl
NAME = re.compile(r"^(?P<slug>.+?)\.fleurs-(?P<lang>[a-z-]+?)"
                  r"(?P<tags>(?:\.b\d+|\.ts-\w+|\.stream\d+ms|\.r\d+|"
                  r"\.filtered|-timestamps_\w+)*)\.jsonl$")

QUANT = re.compile(r"-(F32|BF16|F16|Q8_0|Q6_K|Q5_K_M|Q4_K_M|REF)$")

# Cells that need extra score.py arguments to be measured at all, with the
# reason. Breeze emits Traditional Chinese while FLEURS cmn_hans_cn refs are
# Simplified; folding both sides with OpenCC turns a 35% script mismatch into
# an 8.10% transcription measurement (which matches the number in its own
# hf_card). The fold is applied to reference and hypothesis alike, so it can
# only remove a script difference, never flatter the model on content.
CELL_SCORE_ARGS = {
    ("Breeze-ASR-25", "zh"): (
        ["--script-fold", "t2s"],
        "OpenCC t2s fold (Traditional model vs Simplified refs)",
    ),
    # Diarizing models emit `[start][Sxx]text[end]` around every turn. FLEURS
    # references are plain single-speaker text, so without --dediarize the
    # speaker and timestamp markup is scored as inserted words and the model
    # looks far worse than it transcribes. A lang of None applies to every
    # language the model is measured on, since this is a property of the
    # model's output format rather than of any one language.
    ("MOSS-Transcribe-Diarize", None): (
        ["--dediarize"],
        "strip [start][Sxx]...[end] turn markup (diarizing model)",
    ),
    ("multitalker-parakeet-streaming-0.6b-v1", None): (
        ["--dediarize"],
        "strip [start][Sxx]...[end] turn markup (diarizing model)",
    ),
}


def parse(path: pathlib.Path) -> tuple[str, str, str] | None:
    m = NAME.match(path.name)
    if not m:
        return None
    slug = m.group("slug")
    qm = QUANT.search(slug)
    model, quant = (slug[:qm.start()], qm.group(1)) if qm else (slug, "?")
    return model, quant, m.group("lang")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-score", action="store_true",
                    help="skip score.py; read existing .score.json only")
    ap.add_argument("--rescore", action="store_true",
                    help="re-run score.py even for hyps whose .score.json is "
                         "already newer than the hyp file")
    ap.add_argument("--quant", default="Q8_0")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    # The output path must carry the quant. --quant changes WHAT is scored;
    # without this the destination stayed "fleurs_matrix" for every quant, so
    # scoring a probe quant silently overwrote the Q8_0 matrix with a handful
    # of cells. Q8_0 keeps the bare name so existing callers and the DB
    # builder are unaffected. The separator is "_" not ".": with_suffix()
    # below would treat a ".Q5_K_M" tail as the suffix and replace it,
    # collapsing the path straight back onto fleurs_matrix.tsv.
    if not args.out:
        args.out = str(WER_DIR / ("fleurs_matrix" if args.quant == "Q8_0"
                                  else f"fleurs_matrix_{args.quant}"))

    files = sorted(p for p in WER_DIR.glob("*.fleurs-*.jsonl")
                   if not p.name.endswith(".score.json"))
    cells: dict[tuple[str, str], dict] = {}
    langs: set[str] = set()
    failures = []

    for p in files:
        got = parse(p)
        if not got:
            continue
        model, quant, lang = got
        if args.quant and quant != args.quant:
            continue
        score_path = p.with_suffix(".score.json")
        # Rescoring every file on every run costs minutes once the matrix is
        # a few hundred cells, and score.py is deterministic, so a score that
        # is newer than its hyp file is already current. --rescore forces it
        # (e.g. after a score.py or normalizer change).
        current = (score_path.exists()
                   and score_path.stat().st_mtime >= p.stat().st_mtime)
        extra, extra_why = CELL_SCORE_ARGS.get(
            (model, lang), CELL_SCORE_ARGS.get((model, None), ([], None)))
        if not args.no_score and not (current and not args.rescore):
            r = subprocess.run(
                ["uv", "run", "scripts/wer/score.py", str(p),
                 "--language", lang, *extra],
                cwd=REPO, capture_output=True, text=True)
            if r.returncode != 0:
                failures.append((p.name, r.stderr.strip().split("\n")[-1][:120]))
                continue
        if not score_path.exists():
            continue
        d = json.loads(score_path.read_text())
        langs.add(lang)
        cells[(model, lang)] = {
            "metric": d["metric"], "pct": d["error_rate_pct"],
            "note": extra_why,
            "ci_lo": round(d["error_rate_ci_lo"] * 100, 2),
            "ci_hi": round(d["error_rate_ci_hi"] * 100, 2), "n": d["n"],
        }

    models = sorted({m for m, _ in cells})
    cols = sorted(langs)
    out = pathlib.Path(args.out)
    with open(out.with_suffix(".tsv"), "w") as f:
        f.write("model\t" + "\t".join(cols) + "\n")
        for m in models:
            row = [f"{cells[(m, l)]['pct']:.2f}" if (m, l) in cells else ""
                   for l in cols]
            f.write(m + "\t" + "\t".join(row) + "\n")
    json.dump({"cells": {f"{m}|{l}": v for (m, l), v in cells.items()},
               "models": models, "languages": cols},
              open(out.with_suffix(".json"), "w"), indent=1)

    print(f"{len(cells)} cells, {len(models)} models, {len(cols)} languages "
          f"(quant={args.quant})")
    print(f"  {out.with_suffix('.tsv')}")

    # A cell scored on fewer utterances than its language's full split is a
    # leftover from a subset run (a --n-utts smoke test, or an interrupted
    # sweep). Its error rate is not comparable to the full-split cells beside
    # it, and nothing in the filename says so, so surface it loudly rather
    # than letting it sit in the matrix looking like a real measurement.
    full = defaultdict(int)
    for (m, l), v in cells.items():
        full[l] = max(full[l], v["n"])
    partial = [(m, l, v["n"], full[l]) for (m, l), v in sorted(cells.items())
               if v["n"] < full[l]]
    if partial:
        print(f"\n!! {len(partial)} PARTIAL cells (subset runs; not comparable):")
        for m, l, n, fn in partial:
            print(f"  {m} / {l}: n={n} vs {fn} for the full split")
    for (m, l), (_, why) in CELL_SCORE_ARGS.items():
        # l is None for model-wide rules, which cover every language the model
        # was measured on, so expand it rather than looking up a (m, None) key
        # that can never be in `cells`.
        for lang in ([l] if l is not None else
                     sorted(cl for (cm, cl) in cells if cm == m)):
            if (m, lang) in cells:
                print(f"  note: {m} / {lang} scored with {why} "
                      f"-> {cells[(m, lang)]['pct']:.2f}%")
    if failures:
        print(f"\n{len(failures)} scoring failures:")
        for n, e in failures[:10]:
            print(f"  {n}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
