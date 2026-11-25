#!/usr/bin/env python3
"""
score.py - the catalog score of one scripts/turn/eval_cpp.py run over
pipecat-ai/smart-turn-data-v3.2-test (profile turn-publication-v1).

  uv run --project scripts/envs/smart_turn scripts/turn/score.py \\
      reports/turn/smart-turn-v3.2-F32.smart-turn-data-v3.2-test.jsonl --ref-column p_fp32
  uv run --project scripts/envs/smart_turn scripts/turn/score.py \\
      reports/turn/sava-F32.smart-turn-data-v3.2-test.jsonl \
      --ref-jsonl reports/turn/sava-REF.smart-turn-data-v3.2-test.jsonl
  uv run scripts/catalog/ingest_accuracy.py --models smart-turn-v3.2,sava

A clip's score is the run's p_cpp (transcribe_turn_run's p over the whole
clip). At --threshold (default 0.5, the profile's) a clip is called complete
when p > threshold, as Pipecat's and Inworld's benchmarks do. Writes, next to
the run, <run stem>.score.json with:

  acc_pct, ci95   accuracy and its 95% bootstrap interval (1000 resamples of
                  the clips, seed 42)
  fp_pct, fn_pct  incomplete clips called complete and complete clips called
                  incomplete, in percent of ALL clips (Pipecat's convention)
  auc             ROC AUC of p against the label (ties count half)
  agreement       decisions that match the reference and the largest
                  |p - p_ref|: --ref-column names a column of the run
                  (p_fp32, Smart Turn's fp32 ONNX), --ref-jsonl a reference
                  run joined on clip id that holds every clip (SAVA:
                  run_reference_sava_torch.py). n_agree_at_0.9 is the same
                  count at 0.9.
  extra           not ingested: accuracy at 0.9, the best threshold, and the
                  real / synthetic split
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "transcribe-turn-score-v1"
RUN_SCHEMA = "transcribe-turn-cpp-v1"
# The Hugging Face dataset repo and its catalog identity.
DATASETS = {"pipecat-ai/smart-turn-data-v3.2-test": ("smart-turn-data-v3.2", "test", "mul")}
N_BOOT = 1000
BOOT_SEED = 42


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def auc(p: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney U / (n_pos * n_neg), with average ranks for ties."""
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(p.size, dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < p.size:
        j = i
        while j + 1 < p.size and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(label.sum())
    n_neg = int(label.size - n_pos)
    u = ranks[label].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def decisions(p: np.ndarray, label: np.ndarray, threshold: float) -> dict:
    called = p > threshold
    n = p.size
    fp = int((called & ~label).sum())
    fn = int((~called & label).sum())
    return {"acc_pct": round(100.0 * (n - fp - fn) / n, 2), "fp_pct": round(100.0 * fp / n, 2),
            "fn_pct": round(100.0 * fn / n, 2), "fp": fp, "fn": fn}


def bootstrap_ci(correct: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOT_SEED)
    means = np.sort([correct[rng.integers(0, correct.size, correct.size)].mean()
                     for _ in range(N_BOOT)])
    return [round(100.0 * float(means[int(0.025 * N_BOOT)]), 2),
            round(100.0 * float(means[min(N_BOOT - 1, int(0.975 * N_BOOT))]), 2)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="an eval_cpp.py output (.jsonl)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--ref-column", default=None,
                    help="a per-clip reference score column of the run (p_fp32: Smart Turn's fp32 ONNX)")
    ap.add_argument("--ref-jsonl", type=Path, default=None,
                    help="a reference run (rows with id and p_ref, e.g. run_reference_sava_torch.py's) "
                         "joined on clip id")
    ap.add_argument("--ref-name", default=None, help="what the reference column is (default: from the run)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    lines = args.run.read_text().splitlines()
    header = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:] if line.strip()]
    if header.get("schema") != RUN_SCHEMA:
        die(f"{args.run}: schema {header.get('schema')!r}, not {RUN_SCHEMA}")
    if header.get("dataset") not in DATASETS:
        die(f"{args.run}: unknown dataset {header.get('dataset')!r}")
    if not header.get("engine_sha"):
        die(f"{args.run}: no engine_sha (a run from before eval_cpp.py recorded its commit)")
    ref_path = REPO_ROOT / header["ref"]  # absolute stays absolute
    n_ref = sum(1 for line in ref_path.read_text().splitlines()[1:] if line.strip())
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        die(f"{args.run}: duplicate clip ids")
    full = header.get("limit", 0) == 0 and len(rows) == n_ref

    p = np.array([r["p_cpp"] for r in rows], dtype=np.float64)
    label = np.array([bool(r["label"]) for r in rows])
    at = decisions(p, label, args.threshold)
    correct = ((p > args.threshold) == label).astype(np.float64)
    dataset, split, language = DATASETS[header["dataset"]]
    score = {
        "schema": SCHEMA,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run": rel(args.run),
        "run_created": header.get("created"),
        "model": header["model"],
        "engine": "cpp",
        "engine_sha": header["engine_sha"],
        "backend": header.get("backend"),
        "threads": header.get("threads"),
        "dataset": dataset,
        "split": split,
        "language": language,
        "dataset_repo": header["dataset"],
        "dataset_revision": header.get("dataset_revision"),
        "full_split": full,
        "n": int(p.size),
        "n_complete": int(label.sum()),
        "threshold": args.threshold,
        "rule": "p > threshold",
        "acc_pct": at["acc_pct"],
        "ci95": bootstrap_ci(correct),
        "fp_pct": at["fp_pct"],
        "fn_pct": at["fn_pct"],
        "auc": round(auc(p, label), 5),
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
    }
    if args.ref_column and args.ref_jsonl:
        die("--ref-column and --ref-jsonl are exclusive")
    ref = ref_source = None
    if args.ref_column:
        ref = np.array([r[args.ref_column] for r in rows], dtype=np.float64)
        ref_source = {"ref_column": args.ref_column, "ref": args.ref_name or header["ref"]}
    elif args.ref_jsonl:
        ref_lines = args.ref_jsonl.read_text().splitlines()
        ref_header = json.loads(ref_lines[0])
        if ref_header.get("dataset") != header["dataset"]:
            die(f"{args.ref_jsonl}: dataset {ref_header.get('dataset')!r}, not {header['dataset']!r}")
        by_id = {}
        for line in ref_lines[1:]:
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]] = r
        missing = [i for i in ids if i not in by_id]
        if missing:
            die(f"{args.ref_jsonl}: {len(missing)} clips of the run missing (first {missing[0]})")
        for r in rows:
            if bool(by_id[r["id"]]["label"]) != bool(r["label"]):
                die(f"{r['id']}: label differs between the run and the reference")
        ref = np.array([by_id[i]["p_ref"] for i in ids], dtype=np.float64)
        ref_source = {"ref_jsonl": rel(args.ref_jsonl),
                      "ref": args.ref_name or " ".join(
                          str(x) for x in (ref_header.get("framework"),
                                           f"{ref_header.get('model_repo')}@{ref_header.get('model_revision')}",
                                           ref_header.get("precision")) if x)}
    if ref is not None:
        score["agreement"] = {
            **ref_source,
            "n_agree": int(((p > args.threshold) == (ref > args.threshold)).sum()),
            "n": int(p.size),
            "max_abs_p_delta": float(np.abs(p - ref).max()),
            "n_agree_at_0.9": int(((p > 0.9) == (ref > 0.9)).sum()),
        }
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    accs = [100.0 * float(((p > t) == label).mean()) for t in grid]
    best = int(np.argmax(accs))
    synthetic = np.array([bool(r.get("synthetic")) for r in rows])
    score["extra"] = {
        "acc_pct_at_0.9": decisions(p, label, 0.9)["acc_pct"],
        "best_threshold": float(grid[best]),
        "best_acc_pct": round(accs[best], 2),
        "acc_pct_real": round(100.0 * float(correct[~synthetic].mean()), 2) if (~synthetic).any() else None,
        "acc_pct_synthetic": round(100.0 * float(correct[synthetic].mean()), 2) if synthetic.any() else None,
    }
    out = args.out or args.run.with_name(args.run.name.removesuffix(".jsonl") + ".score.json")
    out.write_text(json.dumps(score, indent=2) + "\n")
    agree = score.get("agreement")
    print(f"{rel(args.run)}: acc {score['acc_pct']:.2f}% [{score['ci95'][0]:.2f}, {score['ci95'][1]:.2f}] "
          f"fp {score['fp_pct']:.2f}% fn {score['fn_pct']:.2f}% auc {score['auc']:.5f} n={score['n']}"
          + (f" agree {agree['n_agree']}/{agree['n']} max|dp| {agree['max_abs_p_delta']:.2g}" if agree else "")
          + ("" if full else "  (NOT the full split)"))
    print(f"wrote {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
