#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
compare.py - the dataset gate: does the C++ engine make the same decisions
as the SpeechBrain reference on the same audio?

The gate: the C++ F32 result makes the same top-1 decision as the SpeechBrain
reference on the same audio. Any disagreement fails, whatever the aggregate
agreement rate. The aggregate is also reported against --gate (99.9%).

Provenance is enforced, not just printed: both runs must share the label set,
the recipe (trim, crops, utterance count, crop dir, logit kind, sample rate)
and the manifest hashes, and a C++ run must score a GGUF converted from the
same checkpoint revision the reference run loaded.

Usage:
    uv run scripts/langid/compare.py \\
        reports/langid/ref-speechbrain-untrimmed.jsonl \\
        reports/langid/cpp-f32-untrimmed.jsonl \\
        --json reports/langid/lang-id-voxlingua107-ecapa-F32.fleurs-mul.agreement.json

Rows are joined on (id, crop_s, trim). Both runs must cover the same keys;
a key present in only one side is reported and counted as a failure, because
a silently dropped utterance would otherwise raise the agreement rate.

Exit status:
    0  no disagreement, agreement >= the gate, same keys
    1  a disagreement, agreement below the gate, or coverage differs
    2  usage / input / provenance error

Only F32 is held to the gate. For the quants pass --report-only: the
agreement is still measured, printed and written (--json, with "passed"),
but only a coverage mismatch exits 1.

Every disagreement is printed with both engines' top-1 and, for each engine,
that engine's own margin between the two codes in question.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

GATE = 0.999
MAX_LISTED_DISAGREEMENTS = 50


def load(path: Path) -> tuple[dict, dict[tuple, dict]]:
    if not path.exists():
        print(f"error: {path} does not exist", file=sys.stderr)
        raise SystemExit(2)
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if not lines:
        print(f"error: {path} is empty", file=sys.stderr)
        raise SystemExit(2)
    header = json.loads(lines[0])
    if header.get("type") != "header":
        print(f"error: {path} does not start with a header line", file=sys.stderr)
        raise SystemExit(2)
    rows = {}
    for line in lines[1:]:
        r = json.loads(line)
        key = (r["id"], str(r["crop_s"]), r["trim"])
        if key in rows:
            print(f"error: {path} has duplicate row {key}", file=sys.stderr)
            raise SystemExit(2)
        rows[key] = r
    return header, rows


def percentile(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile; no numpy so this stays a stdlib PEP-723 tool."""
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[k]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", type=Path, help="reference run JSONL (baseline)")
    p.add_argument("b", type=Path, help="run JSONL under test")
    p.add_argument("--gate", type=float, default=GATE,
                   help=f"minimum top-1 agreement rate (default {GATE})")
    p.add_argument("--max-listed", type=int, default=MAX_LISTED_DISAGREEMENTS)
    p.add_argument("--report-only", action="store_true",
                   help="measure and record agreement without gating on it (the quants); "
                        "a coverage mismatch still exits 1")
    p.add_argument("--json", type=Path, default=None,
                   help="also write the agreement for scripts/catalog/"
                        "ingest_accuracy.py; name it <gguf stem>.fleurs-mul."
                        "agreement.json under reports/langid/")
    args = p.parse_args(argv)

    ha, ra = load(args.a)
    hb, rb = load(args.b)

    labels_a = ha["labels"]
    labels_b = hb["labels"]
    if labels_a != labels_b:
        print("error: the two runs disagree on the label set; a shared label "
              "order is what makes the logit vectors comparable", file=sys.stderr)
        return 2

    print(f"A  {args.a}  engine={ha['engine']}  model={ha['model']}")
    print(f"B  {args.b}  engine={hb['engine']}  model={hb['model']}")
    provenance_ok = True
    for field in ("trim", "crops", "n_utterances", "crop_dir", "logit_kind",
                  "sample_rate", "manifest_sha256"):
        va, vb = ha["recipe"].get(field), hb["recipe"].get(field)
        provenance_ok &= va == vb
        if field == "manifest_sha256" and va == vb:
            print(f"   recipe.{field}: equal ({len(va or {})} manifests)")
            continue
        flag = "" if va == vb else "   <-- DIFFERS"
        print(f"   recipe.{field}: A={va!r} B={vb!r}{flag}")
    # A C++ run must score a GGUF converted from the reference's checkpoint.
    revisions = {h["engine"]: h["recipe"].get("revision") or h["recipe"].get("gguf_source_commit")
                 for h in (ha, hb)}
    print(f"   checkpoint revision: {revisions}")
    if len({r for r in revisions.values()}) != 1 or None in revisions.values():
        provenance_ok = False
        print("   <-- the runs do not trace back to one checkpoint revision")
    print()
    if not provenance_ok:
        print("error: provenance mismatch; refusing to compare", file=sys.stderr)
        return 2

    only_a = sorted(set(ra) - set(rb))
    only_b = sorted(set(rb) - set(ra))
    keys = sorted(set(ra) & set(rb))
    if only_a or only_b:
        print(f"coverage mismatch: {len(only_a)} rows only in A, "
              f"{len(only_b)} rows only in B")
        for k in (only_a + only_b)[:10]:
            print(f"  {k}")
        print()
    if not keys:
        print("error: no shared (id, crop_s, trim) keys", file=sys.stderr)
        return 2

    n_agree = 0
    abs_deltas: list[float] = []
    disagreements = []

    for key in keys:
        x, y = ra[key], rb[key]
        la, lb = x["logits"], y["logits"]
        if len(la) != len(lb):
            print(f"error: {key} has {len(la)} vs {len(lb)} logits", file=sys.stderr)
            return 2
        for u, v in zip(la, lb):
            abs_deltas.append(abs(u - v))

        ia = max(range(len(la)), key=lambda i: la[i])
        ib = max(range(len(lb)), key=lambda i: lb[i])
        if ia == ib:
            n_agree += 1
        else:
            # Each engine's own margin between the two contested labels.
            disagreements.append({
                "key": key,
                "language": x["language"],
                "a_top1": labels_a[ia],
                "b_top1": labels_a[ib],
                "a_gap": la[ia] - la[ib],
                "b_gap": lb[ib] - lb[ia],
                "max_abs_delta": max(abs(u - v) for u, v in zip(la, lb)),
            })

    n = len(keys)
    rate = n_agree / n
    abs_deltas.sort()

    print(f"joined rows: {n}")
    print(f"top-1 agreement: {n_agree}/{n} = {100 * rate:.4f}%  "
          f"(gate {100 * args.gate:.2f}%)")
    print(f"|delta logit| over {len(abs_deltas)} values: "
          f"max={abs_deltas[-1]:.6g}  p99={percentile(abs_deltas, 0.99):.6g}  "
          f"median={percentile(abs_deltas, 0.50):.6g}")
    print()

    if disagreements:
        disagreements.sort(key=lambda d: -max(abs(d["a_gap"]), abs(d["b_gap"])))
        print(f"{len(disagreements)} disagreements "
              f"(listed worst-margin first, up to {args.max_listed}):")
        print(f"{'id':<20} {'crop':>5} {'lang':>5} {'A top1':>7} {'B top1':>7} "
              f"{'A gap':>10} {'B gap':>10} {'max|dlogit|':>12}")
        for d in disagreements[: args.max_listed]:
            uid, crop, _trim = d["key"]
            print(f"{uid:<20} {crop:>5} {d['language']:>5} {d['a_top1']:>7} "
                  f"{d['b_top1']:>7} {d['a_gap']:10.5f} {d['b_gap']:10.5f} "
                  f"{d['max_abs_delta']:12.3g}")
        worst = max(max(abs(d["a_gap"]), abs(d["b_gap"])) for d in disagreements)
        print(f"\nlargest contested margin on either side: {worst:.5f}")
        print()

    if only_a or only_b:
        print("FAIL: the two runs do not cover the same rows")
        status = 1
    elif disagreements:
        print(f"FAIL: {len(disagreements)} disagreements: "
              f"{[d['key'] for d in disagreements[:10]]}")
        status = 1
    elif rate < args.gate:
        print(f"FAIL: agreement {100 * rate:.4f}% < {100 * args.gate:.2f}%")
        status = 1
    else:
        print(f"PASS: agreement {100 * rate:.4f}% >= {100 * args.gate:.2f}%")
        status = 0

    if args.json:
        # The catalog records every shipped GGUF's agreement, gate or not:
        # only F32 is held to the gate, the quants' counts are the evidence
        # for shipping them.
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "schema": "transcribe-langid-agreement-v1",
            "reference": ha["engine"],
            "reference_run": rel(args.a),
            "run": rel(args.b),
            "model": hb["model"],
            "n": n,
            "n_agree": n_agree,
            "max_abs_logit_delta": abs_deltas[-1],
            "disagreements": len(disagreements),
            "unlisted": len(disagreements),
            "same_rows": not (only_a or only_b),
            "gate": args.gate,
            "passed": status == 0,
        }, indent=2) + "\n")
        print(f"wrote {args.json}")
    if args.report_only and not (only_a or only_b):
        return 0
    return status


def rel(path: Path) -> str:
    """Repo-relative when possible, matching score.py's `run`."""
    try:
        return str(path.resolve().relative_to(Path(__file__).resolve().parents[2]))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
