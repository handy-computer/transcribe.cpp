#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
# ]
# ///
"""
score.py - open-set top-1 accuracy of a scripts/langid/run.py sweep and, with
--ref, its top-1 agreement with the reference sweep (the dataset gate).

    uv run scripts/langid/score.py reports/langid/cpp-f32-untrimmed.jsonl \\
        --ref reports/langid/ref-speechbrain-untrimmed.jsonl \\
        --json reports/langid/lang-id-voxlingua107-ecapa-F32.fleurs-mul.score.json

Accuracy is the macro mean over languages of strict code equality (`nn` is
not credited for `no`), per crop, with a 95% bootstrap CI that resamples
utterances within each language.

--json writes the score; with --ref it also writes the agreement next to it
(`.agreement.json`); scripts/catalog/ingest_accuracy.py reads both.

Agreement joins rows on (id, crop_s). Exit 1 on any top-1 disagreement or
if the sweeps do not cover the same rows; exit 2 if they differ in labels,
recipe or checkpoint revision. Only F32 is gated: --report-only (the quants)
exits 1 on a coverage mismatch only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
N_BOOT = 1000
BOOT_SEED = 42
CI = 0.95
RECIPE_FIELDS = ("crops", "max_audio_s", "n_utterances", "logit_kind", "sample_rate", "manifest_sha256")


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load(path: Path) -> tuple[dict, dict[tuple, dict]]:
    lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not lines or lines[0].get("type") != "header" or len(lines) < 2:
        die(f"{path} is not a run.py sweep (header line, then rows)")
    rows = {(r["id"], str(r["crop_s"])): r for r in lines[1:]}
    if len(rows) != len(lines) - 1:
        die(f"{path} has duplicate (id, crop_s) rows")
    return lines[0], rows


def bootstrap_ci(blocks: list[np.ndarray]) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    means = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        acc = 0.0
        for blk in blocks:
            acc += blk[rng.integers(0, blk.size, blk.size)].mean()
        means[b] = acc / len(blocks)
    means.sort()
    return (float(means[int((1 - CI) / 2 * N_BOOT)]),
            float(means[min(N_BOOT - 1, int((1 + CI) / 2 * N_BOOT))]))


def accuracy(header: dict, rows: list[dict]) -> tuple[list[str], dict]:
    """Per crop: n, macro-mean accuracy and its CI, in percent."""
    labels = np.array(header["labels"])
    languages = list(dict.fromkeys(r["language"] for r in rows))
    crops = {}
    for crop in dict.fromkeys(str(r["crop_s"]) for r in rows):
        sel = [r for r in rows if str(r["crop_s"]) == crop]
        lang = np.array([r["language"] for r in sel])
        logits = np.array([r["logits"] for r in sel], dtype=np.float64)
        correct = (labels[np.argmax(logits, axis=1)] == lang).astype(np.float64)
        blocks = [correct[lang == lg] for lg in languages if (lang == lg).any()]
        mean = sum(float(blk.mean()) for blk in blocks) / len(blocks)
        lo, hi = bootstrap_ci(blocks)
        crops[crop] = {"n": len(sel), "acc_pct": round(100.0 * mean, 2),
                       "ci95": [round(100.0 * lo, 2), round(100.0 * hi, 2)]}
    return languages, crops


def agreement(ref: Path, header: dict, rows: dict, run: Path) -> tuple[dict, list]:
    """The agreement JSON, and the first few disagreeing keys."""
    rh, rrows = load(ref)
    if rh["labels"] != header["labels"]:
        die("the sweeps disagree on the label set")
    differs = [f for f in RECIPE_FIELDS if rh["recipe"].get(f) != header["recipe"].get(f)]
    if differs:
        die(f"the sweeps' recipes differ in {differs}")
    revisions = {h["recipe"].get("revision") or h["recipe"].get("gguf_source_commit")
                 for h in (rh, header)}
    if len(revisions) != 1 or None in revisions:
        die(f"the sweeps do not trace back to one checkpoint revision: {revisions}")
    keys = sorted(set(rrows) & set(rows))
    if not keys:
        die("no shared (id, crop_s) rows")
    a = np.array([rrows[k]["logits"] for k in keys], dtype=np.float64)
    b = np.array([rows[k]["logits"] for k in keys], dtype=np.float64)
    if a.shape != b.shape:
        die(f"logit shapes differ: {a.shape} vs {b.shape}")
    agree = np.argmax(a, axis=1) == np.argmax(b, axis=1)
    return {
        "schema": "transcribe-langid-agreement-v1",
        "reference_run": rel(ref),
        "run": rel(run),
        "n": len(keys),
        "n_agree": int(agree.sum()),
        "max_abs_logit_delta": float(np.abs(a - b).max()),
        "disagreements": int((~agree).sum()),
        "same_rows": set(rrows) == set(rows),
    }, [keys[i] for i in np.flatnonzero(~agree)[:10]]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run.py JSONL sweep")
    p.add_argument("--ref", type=Path, help="reference sweep to measure agreement against")
    p.add_argument("--report-only", action="store_true",
                   help="record agreement without gating on it (the quants)")
    p.add_argument("--json", type=Path,
                   help="write the score here: <gguf stem>.fleurs-mul.score.json "
                        "under reports/langid/")
    args = p.parse_args(argv)
    if args.ref and args.json and not args.json.name.endswith(".score.json"):
        p.error("--json must end in .score.json (the agreement is written beside it)")

    header, rows = load(args.run)
    recipe = header["recipe"]
    languages, crops = accuracy(header, list(rows.values()))
    for crop, c in crops.items():
        print(f"crop {crop:>4}: {c['acc_pct']:6.2f}% "
              f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]  n={c['n']}")

    status = 0
    agr = None
    if args.ref:
        agr, disagreeing = agreement(args.ref, header, rows, args.run)
        print(f"top-1 agreement with {agr['reference_run']}: {agr['n_agree']}/{agr['n']}, "
              f"max |delta logit| {agr['max_abs_logit_delta']:.6g}")
        if not agr["same_rows"]:
            print("FAIL: the sweeps do not cover the same rows")
            status = 1
        elif agr["disagreements"] and not args.report_only:
            print(f"FAIL: {agr['disagreements']} disagreements, e.g. {disagreeing}")
            status = 1

    if args.json:
        names = [Path(m).name for m in recipe.get("manifests", [])]
        native = recipe.get("native_commit")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "schema": "transcribe-langid-score-v1",
            "run": rel(args.run),
            "engine": header["engine"],
            "model": header["model"],
            # The build the native library reports, not the checkout scoring it.
            "engine_sha": native if native != "unknown" else None,
            "backend": recipe.get("backend"),
            "dataset": "fleurs" if names and all(n.startswith("fleurs-") for n in names) else None,
            "split": "test",
            "language": "mul",
            "languages": languages,
            "created": header.get("created"),
            "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
            "crops": crops,
        }, indent=2) + "\n")
        print(f"wrote {args.json}")
        if agr:
            out = args.json.with_name(args.json.name.replace(".score.json", ".agreement.json"))
            out.write_text(json.dumps(agr, indent=2) + "\n")
            print(f"wrote {out}")
    return status


if __name__ == "__main__":
    sys.exit(main())
