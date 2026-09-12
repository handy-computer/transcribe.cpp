#!/usr/bin/env python3
"""by_language.py — pivot the FLEURS results into a language-centric view.

The database is keyed by (dataset, model, quant), which answers "how does
this model do everywhere". The recommendation question is the transpose:
"for someone who speaks X, what are the options".

Reads reports/wer/wer.db, so the numbers, the language names and the speed
figures all come from the same index the site queries. Run build_db.py
first; this script computes nothing that is not in the database.

Emits reports/wer/fleurs_by_language.json plus a readable markdown table.

Per language it records every measured model ranked by error rate, with the
95% CI on each cell. Rank 1 is the lowest measured error rate and nothing
more: two adjacent rows whose CIs overlap heavily are not distinguishable
by this table, and it makes no claim that they are.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3
import sys
from datetime import datetime, timezone

REPO = pathlib.Path(__file__).resolve().parents[2]
WER = REPO / "reports" / "wer"
DB = WER / "wer.db"

# The speed column: one number per machine, at the quant being tabulated,
# on the short sample, taking the fastest backend that machine publishes.
RIGS = ["m4-max", "ryzen-4750u"]
SAMPLE = "jfk"


def speed(con: sqlite3.Connection, quant: str) -> dict[str, dict[str, float]]:
    """{model: {rig: xrt}} at this quant, fastest backend, short sample.

    Falls back to whichever sample a model publishes when it has no jfk
    row, which is the GigaAM case (benched on a Russian clip)."""
    out: dict[str, dict[str, float]] = {}
    for model, rig, xrt in con.execute(
        "SELECT model, rig, max(xrt) FROM perf WHERE quant = ? AND sample = ? "
        "GROUP BY model, rig", (quant, SAMPLE)
    ):
        out.setdefault(model, {})[rig] = xrt
    for model, rig, xrt in con.execute(
        "SELECT model, rig, max(xrt) FROM perf WHERE quant = ? GROUP BY model, rig",
        (quant,)
    ):
        out.setdefault(model, {}).setdefault(rig, xrt)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--out", default=str(WER / "fleurs_by_language"))
    ap.add_argument("--quant", default="Q8_0")
    args = ap.parse_args()

    db = pathlib.Path(args.db)
    if not db.exists():
        print(f"error: {db} not found; run scripts/wer/build_db.py first",
              file=sys.stderr)
        return 2
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    meta = dict(con.execute("SELECT key, value FROM meta"))
    perf = speed(con, args.quant)

    langs: dict[str, dict] = {}
    for r in con.execute(
        "SELECT d.lang, l.name AS language, r.model, r.metric, r.err_pct, "
        "       r.ci_lo, r.ci_hi, r.n_utts, r.note "
        "FROM results r "
        "JOIN datasets d ON d.dataset = r.dataset "
        "JOIN languages l ON l.lang = d.lang "
        "WHERE d.source = 'fleurs' AND r.quant = ? "
        "ORDER BY d.lang, r.err_pct", (args.quant,)
    ):
        e = perf.get(r["model"], {})
        langs.setdefault(r["lang"], {"name": r["language"], "models": []})
        langs[r["lang"]]["models"].append({
            "model": r["model"], "name": r["language"], "pct": r["err_pct"],
            "ci": [r["ci_lo"], r["ci_hi"]], "metric": r["metric"],
            "n": r["n_utts"], "note": r["note"],
            "m4_max_rt": e.get("m4-max"), "r4750u_rt": e.get("ryzen-4750u"),
        })

    out: dict[str, dict] = {}
    for lang, cell in sorted(langs.items()):
        rows = cell["models"]
        for i, r in enumerate(rows, 1):
            r["rank"] = i
        out[lang] = {
            "name": cell["name"], "metric": rows[0]["metric"],
            "n_utts": rows[0]["n"], "n_models": len(rows),
            "best": rows[0]["model"], "best_pct": rows[0]["pct"],
            "models": rows,
        }

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": meta.get("dataset_scope", "google/fleurs test split"),
        "quant": args.quant,
        "batch_size": int(meta.get("recipe_batch_size", 8)),
        "timestamps": meta.get("recipe_timestamps", "none"),
        "n_languages": len(out),
        "n_cells": sum(v["n_models"] for v in out.values()),
        "languages": out,
    }
    jpath = pathlib.Path(args.out).with_suffix(".json")
    jpath.write_text(json.dumps(payload, indent=1, ensure_ascii=False))

    lines = [
        f"# FLEURS by language ({args.quant}, batch "
        f"{payload['batch_size']}, timestamps {payload['timestamps']})", "",
        f"{payload['n_languages']} languages, {payload['n_cells']} "
        "measurements. Rank is by measured error rate; where two CIs overlap "
        "the ordering between them is not meaningful.", "",
        f"Speed columns are the published realtime multiplier at "
        f"{args.quant} on the `{SAMPLE}` sample, fastest backend per machine, "
        "from the tables in `docs/models/*.md`. Blank means that model "
        "publishes no bench on that machine.", "",
        "Error rates use the metric named in each heading and are NOT "
        "comparable across languages, only within one.", "",
    ]
    for lang, v in out.items():
        lines.append(f"## {lang} - {v['name']} "
                     f"({v['metric'].upper()}, n={v['n_utts']}, "
                     f"{v['n_models']} models)")
        lines += ["", "| # | model | err% | 95% CI | M4 Max | 4750U |",
                  "|--:|---|--:|---|--:|--:|"]
        for r in v["models"]:
            note = f" ({r['note']})" if r["note"] else ""
            m4 = f"{r['m4_max_rt']:.0f}x" if r.get("m4_max_rt") else ""
            rz = f"{r['r4750u_rt']:.0f}x" if r.get("r4750u_rt") else ""
            lines.append(f"| {r['rank']} | {r['model']}{note} | "
                         f"{r['pct']:.2f} | {r['ci'][0]:.2f}-{r['ci'][1]:.2f} "
                         f"| {m4} | {rz} |")
        lines.append("")
    mpath = pathlib.Path(args.out).with_suffix(".md")
    mpath.write_text("\n".join(lines))

    print(f"{payload['n_languages']} languages, {payload['n_cells']} cells")
    print(f"  {jpath}")
    print(f"  {mpath}")
    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
