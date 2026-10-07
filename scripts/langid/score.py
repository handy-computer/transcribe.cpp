#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
# ]
# ///
"""
score.py - open-set top-1 accuracy of a run.py sweep.

Usage:
    uv run scripts/langid/score.py reports/langid/ref-speechbrain-untrimmed.jsonl \\
        --md reports/langid/ref-speechbrain-untrimmed.md

    # the catalog's accuracy row for one shipped GGUF (ingest_accuracy.py)
    uv run scripts/langid/score.py reports/langid/cpp-f32-untrimmed.jsonl \\
        --json reports/langid/lang-id-voxlingua107-ecapa-F32.fleurs-mul.score.json

The markdown report is the open-set top-1 accuracy per language and mean,
per crop, with a 95% bootstrap CI on the mean. Correctness is strict code
equality (`nn` is not credited for `no`, `be` not for `ru`, `jw` not for
`id`).

The mean is the MACRO mean over languages (each language weighs 1/15). The
bootstrap resamples utterances within each language and re-averages the
per-language accuracies, so the CI describes the same statistic the point
estimate does.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

CROP_ORDER_HINT = {"3": 0, "5": 1, "10": 2, "full": 99}
N_BOOT = 1000
BOOT_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class Run:
    def __init__(self, path: Path) -> None:
        self.path = path
        lines = [l for l in path.read_text().splitlines() if l.strip()]
        if not lines:
            raise SystemExit(f"error: {path} is empty")
        self.header = json.loads(lines[0])
        if self.header.get("type") != "header":
            raise SystemExit(f"error: {path} does not start with a header line")
        self.labels: list[str] = self.header["labels"]
        self.index_of = {c: i for i, c in enumerate(self.labels)}

        rows = [json.loads(l) for l in lines[1:]]
        if not rows:
            raise SystemExit(f"error: {path} has a header but no rows")
        self.crops = list(dict.fromkeys(str(r["crop_s"]) for r in rows))
        self.crops.sort(key=lambda c: CROP_ORDER_HINT.get(c, float(c) if c.replace(".", "").isdigit() else 50))
        self.languages = list(dict.fromkeys(r["language"] for r in rows))

        # Per crop: a [N, 107] logit matrix plus aligned id/language vectors.
        # One array per crop keeps every accuracy a vectorised argmax.
        self.by_crop: dict[str, dict] = {}
        for crop in self.crops:
            sel = [r for r in rows if str(r["crop_s"]) == crop]
            self.by_crop[crop] = {
                "ids": [r["id"] for r in sel],
                "lang": np.array([r["language"] for r in sel]),
                "logits": np.array([r["logits"] for r in sel], dtype=np.float64),
                "audio_s": np.array([r["audio_s"] for r in sel], dtype=np.float64),
            }
        self.n_rows = len(rows)

    @property
    def name(self) -> str:
        return self.path.stem


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def per_language_accuracy(pred_codes: np.ndarray, langs: np.ndarray,
                          order: list[str]) -> tuple[dict[str, float], float]:
    acc = {}
    for lg in order:
        m = langs == lg
        acc[lg] = float((pred_codes[m] == lg).mean()) if m.any() else float("nan")
    vals = [a for a in acc.values() if a == a]
    return acc, (sum(vals) / len(vals) if vals else float("nan"))


def bootstrap_macro_ci(correct: np.ndarray, langs: np.ndarray, order: list[str],
                       n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                       ci: float = 0.95) -> tuple[float, float]:
    """95% CI on the macro-mean accuracy, stratified by language.

    Utterances are resampled with replacement inside each language and the
    per-language accuracies re-averaged, so the resampled statistic is the
    same macro mean the point estimate reports.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for lg in order:
        m = langs == lg
        if m.any():
            blocks.append(correct[m].astype(np.float64))
    if not blocks:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        acc = 0.0
        for blk in blocks:
            idx = rng.integers(0, blk.size, blk.size)
            acc += blk[idx].mean()
        means[b] = acc / len(blocks)
    means.sort()
    lo = means[int((1 - ci) / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 + ci) / 2 * n_boot))]
    return float(lo), float(hi)


def rel(p: Path) -> str:
    """Repo-relative path when possible; reports are read in-tree."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def crop_head(c: str) -> str:
    return "full" if c == "full" else f"{c} s"


def crop_list(crops: list[str]) -> str:
    return ", ".join(crop_head(c) for c in crops)


def pct(x: float, digits: int = 1) -> str:
    if x != x:
        return "-"
    return f"{100.0 * x:.{digits}f}"


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    out.append("")
    return out


def section_setup(run: Run) -> list[str]:
    h = run.header
    r = h["recipe"]
    lines = [f"## Setup", ""]
    lines.append(f"- Engine: `{h['engine']}`, model `{h['model']}`")
    if "revision" in r:
        lines.append(f"- Revision: `{r['revision']}` "
                     f"(speechbrain {r.get('speechbrain')}, torch {r.get('torch')}, "
                     f"{r.get('device')}, {r.get('model_dtype')}, batch "
                     f"{r.get('batch_size')})")
    if "backend" in r:
        lines.append(f"- Backend `{r['backend']}`, threads {r.get('threads')}, "
                     f"max_audio_ms {r.get('max_audio_ms')}")
    lines.append(f"- Dataset: FLEURS `test`, {len(run.languages)} languages "
                 f"x {r['n_utterances'] // max(len(run.languages), 1)} utterances "
                 f"= {r['n_utterances']} clips, longest {r['longest_clip_s']:.1f} s")
    lines.append(f"- Crops: {crop_list(run.crops)}, from the start of the clip")
    lines.append(f"- Logits: `{r['logit_kind']}`; both engines read the same "
                 f"16-bit crops under `{r['crop_dir']}`")
    lines.append(f"- Bootstrap {N_BOOT} resamples, seed {BOOT_SEED}")
    lines.append(f"- Correctness is strict label-code equality "
                 f"(`nn` is not credited for `no`, `be` not for `ru`, `jw` not for `id`)")
    lines.append("")
    return lines


def openset(run: Run) -> tuple[dict[str, dict[str, float]], dict[str, tuple[float, float, float]]]:
    """Per crop: open-set accuracy per language, and the macro mean with its
    bootstrap CI as (mean, lo, hi)."""
    means: dict[str, tuple[float, float, float]] = {}
    per_crop_acc: dict[str, dict[str, float]] = {}
    for crop in run.crops:
        d = run.by_crop[crop]
        pred = np.array(run.labels)[np.argmax(d["logits"], axis=1)]
        acc, mean = per_language_accuracy(pred, d["lang"], run.languages)
        lo, hi = bootstrap_macro_ci(pred == d["lang"], d["lang"], run.languages)
        means[crop] = (mean, lo, hi)
        per_crop_acc[crop] = acc
    return per_crop_acc, means


def section_openset(run: Run) -> list[str]:
    langs_order = run.languages
    rows = []
    per_crop_acc, means = openset(run)

    for lg in langs_order:
        rows.append([lg] + [pct(per_crop_acc[c][lg]) for c in run.crops])
    rows.append(["**mean**"] + [f"**{pct(means[c][0])}**" for c in run.crops])
    rows.append(["95% CI"] + [f"{pct(means[c][1])}-{pct(means[c][2])}" for c in run.crops])

    lines = ["## Open-set top-1 accuracy (all 107 labels), %", ""]
    lines += md_table(["lang"] + [crop_head(c) for c in run.crops], rows)
    return lines


# ---------------------------------------------------------------------------
# Machine-readable headline (scripts/catalog/ingest_accuracy.py)
# ---------------------------------------------------------------------------

SCORE_SCHEMA = "transcribe-langid-score-v1"


def dataset_of(recipe: dict) -> str | None:
    """`fleurs` when every manifest is a scripts/langid/ingest.py FLEURS one
    (`fleurs-<code>.manifest.jsonl`, test split only), else None."""
    names = [Path(m).name for m in recipe.get("manifests", [])]
    return "fleurs" if names and all(n.startswith("fleurs-") for n in names) else None


def score_json(run: Run) -> dict:
    """The open-set macro mean per crop, with what produced it. The catalog
    takes one crop of this (the publication profile's) as a row; the run's
    header is carried so the importer can check the recipe itself."""
    h, r = run.header, run.header["recipe"]
    _, means = openset(run)
    return {
        "schema": SCORE_SCHEMA,
        "run": rel(run.path),
        "engine": h["engine"],
        "model": h["model"],
        # The build the native library reports, not the checkout scoring it.
        "engine_sha": r.get("native_commit") if r.get("native_commit") != "unknown" else None,
        "backend": r.get("backend"),
        "trim": r["trim"],
        "dataset": dataset_of(r),
        "split": "test",
        "language": "mul",
        "languages": run.languages,
        "decision_space": f"open ({len(run.labels)})",
        "created": h.get("created"),
        "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED},
        "crops": {
            crop: {
                "n": int(run.by_crop[crop]["lang"].size),
                "acc_pct": round(100.0 * mean, 2),
                "ci95": [round(100.0 * lo, 2), round(100.0 * hi, 2)],
            }
            for crop, (mean, lo, hi) in means.items()
        },
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def score_run(run: Run) -> list[str]:
    lines = [f"# Language-ID accuracy - `{run.name}`", ""]
    lines += section_setup(run)
    lines += section_openset(run)
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run.py JSONL report(s)")
    p.add_argument("--md", type=Path,
                   help="write the first run's markdown report here (later "
                        "runs go next to it as <run name>.md)")
    p.add_argument("--json", type=Path,
                   help="write the first run's headline (open-set accuracy "
                        "per crop, with provenance) here, for "
                        "scripts/catalog/ingest_accuracy.py; name it "
                        "<gguf stem>.fleurs-mul.score.json under reports/langid/")
    args = p.parse_args(argv)
    if args.json and len(args.runs) > 1:
        p.error("--json takes exactly one run")

    loaded = [Run(r) for r in args.runs]
    scored = [(run, score_run(run)) for run in loaded]

    text = "\n".join(scored[0][1]) + "\n"
    sys.stdout.write(text)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text)
        print(f"wrote {args.md}", file=sys.stderr)
        for run, lines in scored[1:]:
            alt = args.md.with_name(f"{run.name}.md")
            alt.write_text("\n".join(lines) + "\n")
            print(f"wrote {alt}", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(score_json(loaded[0]), indent=2) + "\n")
        print(f"wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
