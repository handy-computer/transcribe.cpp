#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy>=1.26",
# ]
# ///
"""
score.py - turn a run.py sweep into accuracy tables.

Everything below is computed from the stored 107-logit vectors, so a new
decision space costs a re-score, not a re-run.

Usage:
    uv run scripts/langid/score.py reports/langid/ref-speechbrain-untrimmed.jsonl \\
        --md reports/langid/ref-speechbrain-untrimmed.md

    # docs/accuracy.md is written from both reference recipes at once
    uv run scripts/langid/score.py \\
        reports/langid/ref-speechbrain-untrimmed.jsonl \\
        reports/langid/ref-speechbrain-trimmed.jsonl \\
        --docs docs/accuracy.md

Tables produced for each run:
  (a) open-set top-1 accuracy per language and mean, per crop, with a 95%
      bootstrap CI on the mean;
  (b) "handy40" restricted accuracy (logits masked to the 40-language set,
      softmax over the mask, argmax), per crop;
  (c) each restricted selection's accuracy over that selection's utterances;
  (d) confidence: median top-prob when right vs wrong, and coverage /
      precision at thresholds 0.5 / 0.7 / 0.9, on full clips, open-set and
      handy40;
  (e) the 10 confusion pairs with the most open-set errors.

Correctness is strict code equality. `nn` is not credited for `no`, `be` is
not credited for `ru`, `jw` is not credited for `id`: those label artefacts
are exactly what the restricted spaces are supposed to remove, and crediting
them would erase the effect the tables are measuring.

The mean is the MACRO mean over languages (each language weighs 1/15,
matching langid.cpp's original ONNX evaluation). The bootstrap resamples utterances within each
language and re-averages the per-language accuracies, so the CI describes
the same statistic the point estimate does.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

CROP_ORDER_HINT = {"3": 0, "5": 1, "10": 2, "full": 99}
N_BOOT = 1000
BOOT_SEED = 42
THRESHOLDS = (0.5, 0.7, 0.9)
TOP_CONFUSIONS = 10


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


def softmax_masked(logits: np.ndarray, mask_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Argmax and top-probability over a masked decision space.

    `logits` is [N, 107]; `mask_idx` selects the allowed columns. The softmax
    is taken over the allowed columns only, which is what
    langid_candidate.prob is.
    """
    z = logits[:, mask_idx]
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    arg = np.argmax(p, axis=1)
    return mask_idx[arg], p[np.arange(p.shape[0]), arg]


# ---------------------------------------------------------------------------
# Selections
# ---------------------------------------------------------------------------


def load_selections(path: Path, labels: list[str]) -> dict:
    spec = json.loads(path.read_text())
    aliases: dict[str, str] = spec.get("aliases", {})
    index_of = {c: i for i, c in enumerate(labels)}

    def resolve(codes: list[str], where: str) -> list[str]:
        out = []
        for c in codes:
            real = aliases.get(c, c)
            if real not in index_of:
                raise SystemExit(
                    f"error: {where}: code {c!r} (resolved {real!r}) is not in "
                    f"this model's label set")
            out.append(real)
        if len(set(out)) != len(out):
            dupes = [c for c, n in Counter(out).items() if n > 1]
            raise SystemExit(f"error: {where}: duplicate codes after aliasing: {dupes}")
        return out

    restricted = {}
    for name, entry in spec.get("restricted", {}).items():
        codes = resolve(entry["codes"], f"restricted.{name}")
        restricted[name] = {
            "label": entry.get("label", name),
            "placeholder": bool(entry.get("placeholder")),
            "codes": codes,
        }
    selections = []
    for entry in spec.get("selections", []):
        selections.append({
            "name": entry["name"],
            "codes": resolve(entry["codes"], f"selections.{entry['name']}"),
        })
    return {"aliases": aliases, "restricted": restricted, "selections": selections,
            "path": path}


def check_handy40(entry: dict, eval_languages: list[str]) -> list[str]:
    """Sanity assertions on the 40-language set, returned as warnings.

    An eval language missing from the set would score 0 by construction, and
    keeping `be` / `nn` / `jw` would defeat the whole point of the table
    (langid.cpp's original ONNX evaluation: restriction removes exactly those label artefacts).
    """
    warnings = []
    codes = entry["codes"]
    if len(codes) != 40:
        warnings.append(f"handy40 has {len(codes)} codes, not 40")
    missing = [c for c in eval_languages if c not in codes]
    if missing:
        warnings.append(f"handy40 omits evaluated languages {missing}; they "
                        f"score 0 by construction")
    artefacts = [c for c in ("be", "nn", "jw") if c in codes]
    if artefacts:
        warnings.append(f"handy40 contains the label artefacts {artefacts} that "
                        f"the restricted table exists to remove")
    return warnings


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


def section_setup(run: Run, sel: dict, warnings: list[str]) -> list[str]:
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
    lines.append(f"- Crops: {crop_list(run.crops)}, from the "
                 f"{'first speech frame' if r['trim'] == 'energy' else 'start of the clip'}"
                 f" (`--trim {r['trim']}`)")
    lines.append(f"- Trim rule: {r['trim_rule']}")
    lines.append(f"- Logits: `{r['logit_kind']}`; both engines read the same "
                 f"16-bit crops under `{r['crop_dir']}`")
    lines.append(f"- Decision spaces from `{rel(sel['path'])}`; "
                 f"bootstrap {N_BOOT} resamples, seed {BOOT_SEED}")
    lines.append(f"- Correctness is strict label-code equality "
                 f"(`nn` is not credited for `no`, `be` not for `ru`, `jw` not for `id`)")
    lines.append("")
    if warnings:
        lines.append("> **Warnings**")
        for w in warnings:
            lines.append(f"> - {w}")
        lines.append("")
    return lines


def section_openset(run: Run) -> tuple[list[str], dict]:
    langs_order = run.languages
    rows = []
    means: dict[str, tuple[float, float, float]] = {}
    per_crop_acc: dict[str, dict[str, float]] = {}
    for crop in run.crops:
        d = run.by_crop[crop]
        pred = np.array(run.labels)[np.argmax(d["logits"], axis=1)]
        acc, mean = per_language_accuracy(pred, d["lang"], langs_order)
        lo, hi = bootstrap_macro_ci(pred == d["lang"], d["lang"], langs_order)
        means[crop] = (mean, lo, hi)
        per_crop_acc[crop] = acc

    for lg in langs_order:
        rows.append([lg] + [pct(per_crop_acc[c][lg]) for c in run.crops])
    rows.append(["**mean**"] + [f"**{pct(means[c][0])}**" for c in run.crops])
    rows.append(["95% CI"] + [f"{pct(means[c][1])}-{pct(means[c][2])}" for c in run.crops])

    lines = ["## (a) Open-set top-1 accuracy (all 107 labels), %", ""]
    lines += md_table(["lang"] + [crop_head(c) for c in run.crops], rows)
    return lines, {"per_language": per_crop_acc, "mean": means}


def section_restricted(run: Run, sel: dict) -> tuple[list[str], dict]:
    lines = []
    out = {}
    for name, entry in sel["restricted"].items():
        idx = np.array([run.index_of[c] for c in entry["codes"]])
        rows = []
        per_crop_acc = {}
        means = {}
        for crop in run.crops:
            d = run.by_crop[crop]
            arg, _p = softmax_masked(d["logits"], idx)
            pred = np.array(run.labels)[arg]
            acc, mean = per_language_accuracy(pred, d["lang"], run.languages)
            lo, hi = bootstrap_macro_ci(pred == d["lang"], d["lang"], run.languages)
            per_crop_acc[crop] = acc
            means[crop] = (mean, lo, hi)
        for lg in run.languages:
            rows.append([lg] + [pct(per_crop_acc[c][lg]) for c in run.crops])
        rows.append(["**mean**"] + [f"**{pct(means[c][0])}**" for c in run.crops])
        rows.append(["95% CI"] + [f"{pct(means[c][1])}-{pct(means[c][2])}" for c in run.crops])
        tag = " (PLACEHOLDER list)" if entry["placeholder"] else ""
        lines += [f"## (b) Restricted to {entry['label']}"
                  f" - {len(entry['codes'])} labels{tag}, %", ""]
        lines += [f"`{' '.join(entry['codes'])}`", ""]
        lines += md_table(["lang"] + [crop_head(c) for c in run.crops],
                          rows)
        out[name] = {"per_language": per_crop_acc, "mean": means,
                     "codes": entry["codes"]}
    return lines, out


def section_selections(run: Run, sel: dict) -> tuple[list[str], dict]:
    rows = []
    out = {}
    for entry in sel["selections"]:
        codes = entry["codes"]
        idx = np.array([run.index_of[c] for c in codes])
        cells = []
        per_crop = {}
        for crop in run.crops:
            d = run.by_crop[crop]
            m = np.isin(d["lang"], codes)
            if not m.any():
                cells.append("-")
                per_crop[crop] = float("nan")
                continue
            arg, _p = softmax_masked(d["logits"][m], idx)
            pred = np.array(run.labels)[arg]
            acc = float((pred == d["lang"][m]).mean())
            per_crop[crop] = acc
            cells.append(pct(acc))
        n = int(np.isin(run.by_crop[run.crops[0]]["lang"], codes).sum())
        rows.append([entry["name"], str(n)] + cells)
        out[entry["name"]] = per_crop
    lines = ["## (c) Restricted to a user's selection "
             "(accuracy over that selection's utterances), %", ""]
    lines += md_table(["selection", "n utts"] +
                      [crop_head(c) for c in run.crops], rows)
    return lines, out


def section_confidence(run: Run, sel: dict) -> tuple[list[str], dict]:
    """Median top-prob right vs wrong, and coverage/precision at thresholds.

    Reported on the `full` crop (the realistic case: the caller has the whole
    utterance) in two decision spaces - open-set and handy40. `coverage` is
    the share of decisions kept, `precision` the share of kept decisions that
    are right; a threshold is only useful when it trades a little of the
    first for a lot of the second.
    """
    crop = "full" if "full" in run.crops else run.crops[-1]
    d = run.by_crop[crop]
    spaces: list[tuple[str, np.ndarray]] = [
        ("open (107)", np.arange(len(run.labels)))]
    for name, entry in sel["restricted"].items():
        spaces.append((f"{name} ({len(entry['codes'])})",
                       np.array([run.index_of[c] for c in entry["codes"]])))

    rows = []
    thr_rows = []
    out = {}
    for space_name, idx in spaces:
        arg, prob = softmax_masked(d["logits"], idx)
        pred = np.array(run.labels)[arg]
        right = pred == d["lang"]
        med_r = float(np.median(prob[right])) if right.any() else float("nan")
        med_w = float(np.median(prob[~right])) if (~right).any() else float("nan")
        rows.append([space_name, f"{med_r:.3f}", f"{med_w:.3f}",
                     f"{pct(float(right.mean()))}"])
        per_thr = {}
        for t in THRESHOLDS:
            keep = prob >= t
            cov = float(keep.mean())
            prec = float(right[keep].mean()) if keep.any() else float("nan")
            thr_rows.append([space_name, f"{t:.1f}", pct(cov), pct(prec)])
            per_thr[t] = (cov, prec)
        out[space_name] = {"median_right": med_r, "median_wrong": med_w,
                           "thresholds": per_thr}

    lines = [f"## (d) Confidence, `{crop}` clips", ""]
    lines += md_table(["decision space", "median top-prob right",
                       "median top-prob wrong", "accuracy %"], rows)
    lines += md_table(["decision space", "threshold", "coverage %",
                       "precision %"], thr_rows)
    return lines, out


def section_confusions(run: Run) -> tuple[list[str], list]:
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for crop in run.crops:
        d = run.by_crop[crop]
        pred = np.array(run.labels)[np.argmax(d["logits"], axis=1)]
        wrong = pred != d["lang"]
        for t, p in zip(d["lang"][wrong], pred[wrong]):
            counts[(str(t), str(p))][crop] += 1
    ranked = sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))[:TOP_CONFUSIONS]
    rows = []
    for (t, p), per_crop in ranked:
        rows.append([f"{t} -> {p}", str(sum(per_crop.values()))] +
                    [str(per_crop.get(c, 0)) for c in run.crops])
    lines = [f"## (e) Top {TOP_CONFUSIONS} open-set confusion pairs "
             f"(errors, all crops)", ""]
    lines += md_table(["true -> predicted", "total"] +
                      [crop_head(c) for c in run.crops], rows)
    return lines, [(k, dict(v)) for k, v in ranked]


# ---------------------------------------------------------------------------
# langid.cpp original-evaluation comparison
# ---------------------------------------------------------------------------

# langid.cpp's first evaluation, measured on an ONNX export of the same checkpoint,
# untrimmed crops from the start of the clip. Reproduced here so score.py can
# print the delta instead of a human eyeballing two documents.
PLAN_V1_OPEN = {"3": 0.67, "5": 0.85, "10": 0.91, "full": 0.91}
PLAN_V1_HANDY40 = {"3": 0.77, "5": 0.92, "10": 0.965, "full": 0.966}
PLAN_V1_SELECTIONS = {
    "en+zh":       {"3": 0.90, "5": 0.99,   "10": 1.000},
    "en+es+pt":    {"3": 0.98, "5": 0.995,  "10": 1.000},
    "es+pt":       {"3": 0.99, "5": 0.995,  "10": 1.000},
    "ja+ko":       {"3": 0.88, "5": 0.98,   "10": 1.000},
    "en+ru":       {"3": 0.99, "5": 1.00,   "10": 1.000},
    "no+da":       {"3": 0.96, "5": 0.998,  "10": 1.000},
    "en+de+ja":    {"3": 0.82, "5": 0.95,   "10": 0.998},
    "en+fr+de+es": {"3": 0.89, "5": 0.94,   "10": 0.990},
    "de+fr+ja":    {"3": 0.80, "5": 0.91,   "10": 0.985},
    "cs+sk":       {"3": 0.83, "5": 0.92,   "10": 0.970},
    "id+ms":       {"3": 0.88, "5": 0.895,  "10": 0.895},
}


def section_planv1(run: Run, openset: dict, restricted: dict,
                   selections: dict) -> list[str]:
    lines = ["## Delta vs langid.cpp's original ONNX evaluation (ONNX export of the same "
             "checkpoint, untrimmed)", ""]
    if run.header["recipe"]["trim"] != "none":
        lines.append("langid.cpp's original numbers are untrimmed; this run uses "
                     f"`--trim {run.header['recipe']['trim']}`, so the deltas "
                     "below are a recipe difference, not a discrepancy.")
        lines.append("")
    rows = []
    for crop in run.crops:
        ref = PLAN_V1_OPEN.get(crop)
        got = openset["mean"][crop][0]
        rows.append(["all 107", crop, pct(ref) if ref else "-", pct(got),
                     f"{100 * (got - ref):+.1f}" if ref else "-"])
    if "handy40" in restricted:
        for crop in run.crops:
            ref = PLAN_V1_HANDY40.get(crop)
            got = restricted["handy40"]["mean"][crop][0]
            rows.append(["Handy's 40*", crop, pct(ref) if ref else "-", pct(got),
                         f"{100 * (got - ref):+.1f}" if ref else "-"])
    lines += md_table(["decision space", "crop", "original %", "this run %", "delta pp"],
                      rows)
    if "handy40" in restricted:
        lines.append("\\* langid.cpp's original 40-language set is not recorded anywhere; "
                     "this row compares against the placeholder list in "
                     "`scripts/langid/selections.json`, so a delta of a point or "
                     "two is expected from the set alone.")
        lines.append("")

    rows = []
    crops_cmp = [c for c in run.crops if c in ("3", "5", "10")]
    for name, ref in PLAN_V1_SELECTIONS.items():
        got = selections.get(name)
        if got is None:
            continue
        cells = []
        for c in crops_cmp:
            if c in ref and got.get(c) == got.get(c):
                cells += [pct(ref[c]), pct(got[c]), f"{100 * (got[c] - ref[c]):+.1f}"]
            else:
                cells += ["-", "-", "-"]
        rows.append([name] + cells)
    head = ["selection"]
    for c in crops_cmp:
        head += [f"v1 {c}s", f"now {c}s", f"d {c}s"]
    lines += md_table(head, rows)
    return lines


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def score_run(run: Run, sel: dict) -> tuple[list[str], dict]:
    warnings = []
    if "handy40" in sel["restricted"]:
        warnings += check_handy40(sel["restricted"]["handy40"], run.languages)
        if sel["restricted"]["handy40"]["placeholder"]:
            warnings.append(
                "`handy40` is a PLACEHOLDER 40-language list, "
                "not the real Handy selection. Replace `restricted.handy40.codes` "
                "in scripts/langid/selections.json and re-run score.py; run.py does "
                "not need to run again.")

    lines = [f"# Language-ID accuracy - `{run.name}`", ""]
    lines += section_setup(run, sel, warnings)
    a, openset = section_openset(run)
    b, restricted = section_restricted(run, sel)
    c, selections = section_selections(run, sel)
    d, confidence = section_confidence(run, sel)
    e, confusions = section_confusions(run)
    lines += a + b + c + d + e
    lines += section_planv1(run, openset, restricted, selections)
    summary = {"open": openset, "restricted": restricted, "selections": selections,
               "confidence": confidence, "confusions": confusions,
               "warnings": warnings}
    return lines, summary


def docs_accuracy(runs: list[tuple[Run, dict, list[str]]], sel: dict,
                  gate_note: str | None = None) -> list[str]:
    """docs/accuracy.md: the two reference recipes side by side."""
    first = runs[0][0]
    r = first.header["recipe"]
    lines = [
        "# Accuracy",
        "",
        "> **These are the SpeechBrain reference numbers, not a separate "
        "measurement of langid.cpp.**",
        "> The C++ F32 result is required to agree with the reference on "
        ">= 99.9% of decisions over this same corpus.",
    ]
    if gate_note:
        lines.append(f"> {gate_note}")
    else:
        lines.append("> That gate has not been measured yet; until it is, treat "
                     "this page as the reference's numbers only.")
    lines += [
        "",
        "## Setup",
        "",
        f"- Model: `{first.header['model']}` @ `{r.get('revision')}` "
        f"(speechbrain {r.get('speechbrain')}, torch {r.get('torch')}, CPU, fp32, "
        f"batch {r.get('batch_size')}).",
        f"- Dataset: FLEURS `test`, {len(first.languages)} languages "
        f"(`{' '.join(first.languages)}`), the first "
        f"{r['n_utterances'] // max(len(first.languages), 1)} utterances of at "
        f"least 1 s in parquet order per language = {r['n_utterances']} clips, "
        f"mean ~12 s.",
        f"- Crops: {', '.join(c + ' s' for c in first.crops if c != 'full')} and "
        f"the full clip. A clip shorter than the crop is scored whole.",
        "- Two trim recipes:",
        "  - **untrimmed** - the crop starts at sample 0 of the raw clip. This "
        "is what langid.cpp's original ONNX evaluation measured; FLEURS clips open with silence, so "
        "\"3 s\" is roughly 2 s of speech.",
        "  - **trimmed** - leading audio is dropped up to the first 20 ms frame "
        "whose RMS exceeds 5% of the clip's loudest frame, then the crop is "
        "taken. This is the usage protocol langid.cpp asks callers to follow "
        "(feed speech, not the raw capture), so it is the honest number for a "
        "dictation app.",
        "- Correctness is strict label-code equality. `nn` is not credited for "
        "`no`, `be` not for `ru`, `jw` not for `id`.",
        "- The mean is the macro mean over languages; the CI is a "
        f"{N_BOOT}-resample stratified bootstrap, seed {BOOT_SEED}.",
        "- Generated by `scripts/langid/score.py` from "
        + ", ".join(f"`{rel(ru.path)}`" for ru, _s, _l in runs)
        + ".",
        "",
    ]

    crops = first.crops
    lines += ["## Top-1 accuracy, mean over the "
              f"{len(first.languages)} languages, %", ""]
    rows = []
    for ru, summary, label in runs:
        rows.append([f"all 107 ({label})"] +
                    [pct(summary["open"]["mean"][c][0]) for c in crops])
    for ru, summary, label in runs:
        if "handy40" in summary["restricted"]:
            rows.append([f"handy40* ({label})"] +
                        [pct(summary["restricted"]["handy40"]["mean"][c][0])
                         for c in crops])
    lines += md_table(["decision space"] +
                      [crop_head(c) for c in crops], rows)
    lines.append("\\* `handy40` is a **placeholder** 40-language dictation list "
                 ", not the real Handy selection; see "
                 "`scripts/langid/selections.json`.")
    lines.append("")

    lines += ["## Restricted to a user's selection, %", ""]
    head = ["selection"]
    for _ru, _s, label in runs:
        head += [f"{label} {crop_head(c)}" for c in crops]
    rows = []
    for entry in sel["selections"]:
        row = [entry["name"]]
        for _ru, summary, _label in runs:
            row += [pct(summary["selections"][entry["name"]].get(c, float("nan")))
                    for c in crops]
        rows.append(row)
    lines += md_table(head, rows)

    lines += ["## Per-run detail", ""]
    for ru, _s, label in runs:
        lines.append(f"- **{label}**: `reports/langid/{ru.name}.md`")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="run.py JSONL report(s)")
    p.add_argument("--selections", type=Path,
                   default=Path(__file__).resolve().parent / "selections.json")
    p.add_argument("--md", type=Path,
                   help="write the first run's markdown report here")
    p.add_argument("--docs", type=Path,
                   help="write a combined docs/accuracy.md from every run given")
    p.add_argument("--gate-note", default=None,
                   help="one sentence recording the measured dataset-gate "
                        "result; goes into --docs verbatim. Passing a claim "
                        "compare.py did not produce is how this page starts "
                        "lying, so quote its output.")
    args = p.parse_args(argv)

    loaded = [Run(r) for r in args.runs]
    sel = load_selections(args.selections, loaded[0].labels)

    scored = []
    for run in loaded:
        lines, summary = score_run(run, sel)
        label = run.header["recipe"]["trim"]
        label = "untrimmed" if label == "none" else "trimmed"
        scored.append((run, summary, label, lines))

    text = "\n".join(scored[0][3]) + "\n"
    sys.stdout.write(text)
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(text)
        print(f"wrote {args.md}", file=sys.stderr)
    for run, _summary, _label, lines in scored[1:]:
        if args.md:
            alt = args.md.with_name(f"{run.name}.md")
            alt.write_text("\n".join(lines) + "\n")
            print(f"wrote {alt}", file=sys.stderr)

    if args.docs:
        doc = docs_accuracy([(r, s, l) for r, s, l, _ln in scored], sel,
                            args.gate_note)
        args.docs.parent.mkdir(parents=True, exist_ok=True)
        args.docs.write_text("\n".join(doc) + "\n")
        print(f"wrote {args.docs}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
