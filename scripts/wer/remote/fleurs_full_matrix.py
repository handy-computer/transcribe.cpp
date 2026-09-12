#!/usr/bin/env python3
"""fleurs_full_matrix.py — every supported model over every language it claims.

`fleurs_matrix.py` answers "which model should a user pick for the languages
they speak", so its columns are a fixed union defined by the specialist
models and every model is measured over the same columns. This script answers
a different question: "what is each model's WER on each language it actually
advertises". The plan is therefore fully ragged — whisper contributes 82
columns that no other model can fill — and it covers the shipped quants
rather than one.

Two quants ship in the plan:
  Q8_0     every ASR model, always. It is the default shipped quant.
  Q5_K_M   only where the Q8_0 download is over 1 GB, where a smaller quant
           is worth shipping, plus explicit borderline additions.

What this does that `fleurs_matrix.py` does not:
  - canonicalizes alias language codes, so `no`/`nb`, `tl`/`fil` and
    `zh`/`zh-cn` resolve to ONE column and ONE ingest rather than two;
  - skips (model, quant, language) cells already scored at the matrix batch
    size, reading reports/wer/ directly rather than a cache;
  - carries a per-model GPU, because the 24B does not fit in 24 GB.

Usage:
    uv run scripts/wer/remote/fleurs_full_matrix.py --plan
    uv run scripts/wer/remote/fleurs_full_matrix.py --preflight   # ingest only
    uv run scripts/wer/remote/fleurs_full_matrix.py --run --cap 120
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from model_specs import resolve_model  # noqa: E402
from fleurs_matrix import (  # noqa: E402
    TAG_OVERRIDE_MODELS,
    card_langs,
    fleurs_langs,
    locale_for,
    spend_today,
    MAX_BLIND_DISPATCHES,
)

REPO = pathlib.Path(__file__).resolve().parents[3]
CARDS = REPO / "scripts" / "hf_cards"
REPORTS = REPO / "reports" / "wer"

# The batch size the matrix is measured at. Cells run at another batch size
# are NOT counted as generated: docs/tools/wer.md measures batching as WER-neutral
# to within ~0.08pp, but a matrix that silently mixes recipes cannot be
# defended later, and the affected cells are all sub-cent models.
MATRIX_BATCH = 8

QUANT_NAMES = ("BF16", "F16", "F32", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")

# Q5_K_M is swept where the Q8_0 download exceeds this, in bytes.
Q5_MIN_Q8_BYTES = 1_000_000_000

# Models under the threshold that still get a Q5_K_M sweep. Both sit just
# under 1 GB at Q8_0 and are cheap to add: turbo is a flagship users actually
# pick, and MOSS is the only diarizing model in the matrix.
Q5_EXTRA_MODELS = ("whisper-large-v3-turbo", "moss-transcribe-diarize")

# GPU placement. Sized on the quant file, but the file is NOT the working set:
# an L4 advertises 24 GB and reports 22563 MiB usable, and the runtime needs
# room for the KV cache, activations and any repacked weights on top of the
# tensors themselves. Voxtral-Small-24B Q5_K_M (17.14 GB on disk) OOMs on an L4
# asking for a further 8800 MiB for its packed gate/up buffer, so the headroom
# needed is at least ~1.5x the file. Anything over 13 GB goes to the 48 GB part.
BIG_GPU = "L40S"
BIG_GPU_MIN_BYTES = 13_000_000_000

# hf_cards that are not transcription models, so have no WER row.
NON_ASR_PIPELINES = {"voice-activity-detection"}

SIZE_CACHE = REPO / "reports" / "wer" / "_gguf_sizes.json"


# -------- card + Hub metadata ---------------------------------------------

def all_cards() -> list[str]:
    return sorted(p.stem for p in CARDS.glob("*.yaml"))


def card_field(model: str, key: str) -> str | None:
    txt = (CARDS / f"{model}.yaml").read_text()
    m = re.search(rf"^{re.escape(key)}:\s*(\S+)", txt, re.M)
    return m.group(1) if m else None


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    p = pathlib.Path("~/.cache/huggingface/token").expanduser()
    return p.read_text().strip() if p.exists() else None


def gguf_sizes(models: list[str], refresh: bool = False) -> dict[str, dict[str, int]]:
    """{model: {gguf filename: size in bytes}} from the Hub.

    The quant rule keys off the real download size, not the hf_card's `size:`
    string. The card is hand-maintained prose ("1.0 GB" for a 1.05 GB file),
    and rounding decides membership for models sitting on the threshold.
    Cached on disk because this is a plan-time input, not a run-time one."""
    cache: dict[str, dict[str, int]] = {}
    if SIZE_CACHE.exists() and not refresh:
        cache = json.loads(SIZE_CACHE.read_text())
    todo = [m for m in models if m not in cache]
    if todo:
        tok = hf_token()

        def one(model: str) -> tuple[str, dict[str, int]]:
            repo = card_field(model, "target_repo")
            if not repo:
                return model, {}
            req = urllib.request.Request(
                f"https://huggingface.co/api/models/{repo}?blobs=true")
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            try:
                d = json.load(urllib.request.urlopen(req, timeout=60))
            except Exception as e:
                print(f"[WARN] {model}: Hub query failed ({e})", file=sys.stderr)
                return model, {}
            return model, {s["rfilename"]: s.get("size") or 0
                           for s in d.get("siblings", [])
                           if s["rfilename"].endswith(".gguf")}

        with ThreadPoolExecutor(16) as ex:
            for model, files in ex.map(one, todo):
                cache[model] = files
        SIZE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        SIZE_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return cache


def quant_size(files: dict[str, int], quant: str) -> int:
    for fn, sz in files.items():
        stem = fn[:-5] if fn.endswith(".gguf") else fn
        if stem.endswith(f"-{quant}"):
            return sz or 0
    return 0


# -------- language canonicalization ---------------------------------------

def canonical_codes(FL: dict[str, str], measured: set[str]) -> dict[str, str]:
    """FLEURS config -> the ONE bare code used to ingest and name it.

    FLEURS_LANGS is many-to-one: `no` and `nb` both mean nb_no, `tl` and `fil`
    both mean fil_ph, `zh` and `zh-cn` both mean cmn_hans_cn. The dataset id
    (and therefore the manifest path, the Volume entry and every report
    filename) is derived from the bare code, so leaving both in the plan
    ingests the same audio twice and splits one language across two columns.
    Prefer a code already on disk so this run's reports file alongside the
    existing ones."""
    by_cfg: dict[str, list[str]] = defaultdict(list)
    for code, cfg in FL.items():
        by_cfg[cfg].append(code)
    out = {}
    for cfg, codes in by_cfg.items():
        prior = sorted(c for c in codes if c in measured)
        out[cfg] = prior[0] if prior else sorted(codes, key=lambda c: (len(c), c))[0]
    return out


def volume_manifest_langs() -> set[str] | None:
    """FLEURS languages whose manifest exists on the Modal /data Volume.

    This, not the local reports directory, is what "already ingested" means:
    `prefetch` writes manifests to the Volume the GPU cells read, and a
    language can be fully ingested there while nothing local mentions it.
    Returns None if the Volume cannot be listed, so the caller can tell
    "nothing ingested" apart from "could not check"."""
    try:
        out = subprocess.run(["modal", "volume", "ls", "transcribe-data", "/wer"],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return set(re.findall(r"fleurs-([a-z-]+)\.manifest\.jsonl", out.stdout))


# -------- what is already done --------------------------------------------

def generated_cells(FL: dict[str, str], canon: dict[str, str]) -> set[tuple[str, str, str]]:
    """{(gguf base, quant, canonical language)} already generated at MATRIX_BATCH.

    Keyed on the hyp .jsonl, NOT the .score.json. Generation is the expensive,
    remote, non-repeatable half; scoring is local, free and re-runnable, and
    normally lags a sweep by however long it takes to get round to it. Keying
    on the score would make every generated-but-unscored cell look unrun and
    re-dispatch GPU work that is already sitting on disk.

    Parsed from reports/wer/ rather than from a run cache: the cache keys on
    dispatch parameters and does not survive a recipe or filename change,
    while the hyp IS the artifact the matrix is built from.

    The language field has to be matched against the known FLEURS codes rather
    than split on a delimiter. Older reports carry a `-timestamps_none` suffix
    directly after the language, so a greedy split reads the language of
    `moonshine-base-ar-Q8_0.fleurs-ar-timestamps_none.score.json` as
    `ar-timestamps_none` and the cell looks unrun."""
    codes = sorted(FL, key=len, reverse=True)
    done: set[tuple[str, str, str]] = set()
    for p in REPORTS.glob("*.fleurs-*.jsonl"):
        m = re.match(r"(.+?)\.fleurs-(.+?)\.jsonl$", p.name)
        if not m:
            continue
        stem, tail = m.group(1), m.group(2)
        if stem.endswith("-REF"):          # framework reference, not our engine
            continue
        lang = next((c for c in codes
                     if tail == c or tail.startswith(c + ".") or tail.startswith(c + "-")),
                    None)
        if lang is None:
            continue
        rest = tail[len(lang):].lstrip(".-")
        # Derived runs, not full-split matrix cells: utterance subsets
        # (ru-512), filtered rescorings, and streaming-mode variants.
        if re.match(r"^\d", rest) or "filtered" in rest or "stream" in rest:
            continue
        if rest != f"b{MATRIX_BATCH}":
            continue
        quant = next((q for q in QUANT_NAMES if stem.endswith("-" + q)), None)
        if not quant:
            continue
        done.add((stem[:-(len(quant) + 1)], quant, canon.get(FL[lang], lang)))
    return done


# -------- plan -------------------------------------------------------------

def build_plan(only: set[str] | None, skip_done: bool,
               refresh_sizes: bool) -> tuple[list[dict], dict]:
    FL = fleurs_langs()
    models = [m for m in all_cards()
              if (CARDS / f"{m}.yaml").exists()
              and card_field(m, "pipeline_tag") not in NON_ASR_PIPELINES]
    if only:
        unknown = only - set(models)
        if unknown:
            sys.exit(f"--only names unknown or non-ASR models: {sorted(unknown)}")
        models = [m for m in models if m in only]

    sizes = gguf_sizes(models, refresh=refresh_sizes)
    measured = {p.name.split(".fleurs-")[1].split(".")[0]
                for p in REPORTS.glob("*.fleurs-*.jsonl")}
    canon = canonical_codes(FL, {c for c in measured if c in FL})
    done = generated_cells(FL, canon) if skip_done else set()

    plan: list[dict] = []
    stats = {"models": len(models), "columns": set(), "skipped": 0}
    for model in models:
        # Which files this model means is the CARD's business, not the repo
        # listing's: `sweep` resolves through resolve_model, and a repo can
        # ship more than one variant of the same quant. multitalker publishes
        # both a plain and a bundle/ tree, so picking off the listing can size
        # and name the wrong artifact from the one that actually runs.
        try:
            _repo, pinned = resolve_model(REPO, model)
        except SystemExit as e:
            print(f"[WARN] {model}: {e}, skipping", file=sys.stderr)
            continue
        listing = sizes.get(model, {})
        files = {fn: listing.get(fn, next(
            (s for f, s in listing.items() if f.rsplit("/", 1)[-1] == fn), 0))
            for fn in (pinned or [])}
        base = None
        for fn in files:
            stem = fn[:-5]
            q = next((x for x in QUANT_NAMES if stem.endswith("-" + x)), None)
            if q:
                base = stem.rsplit("/", 1)[-1][:-(len(q) + 1)]
                break
        if base is None:
            print(f"[WARN] {model}: no quantized GGUF pinned, skipping",
                  file=sys.stderr)
            continue

        q8 = quant_size(files, "Q8_0")
        quants = ["Q8_0"]
        if quant_size(files, "Q5_K_M") and (
                q8 > Q5_MIN_Q8_BYTES or model in Q5_EXTRA_MODELS):
            quants.append("Q5_K_M")

        # One column per FLEURS config the card claims, keeping the model's
        # own tag: the CLI validates --language against the GGUF's
        # general.languages, which is not always the canonical code.
        cols: dict[str, str] = {}
        for code in card_langs(model):
            cfg = FL.get(code)
            if cfg:
                cols.setdefault(canon[cfg], code)

        for quant in quants:
            gpu = BIG_GPU if quant_size(files, quant) > BIG_GPU_MIN_BYTES else "L4"
            for lang, tag in sorted(cols.items()):
                if model in TAG_OVERRIDE_MODELS:
                    tag = locale_for(tag)
                    if tag is None:
                        continue
                stats["columns"].add(lang)
                if (base, quant, lang) in done:
                    stats["skipped"] += 1
                    continue
                plan.append({"model": model, "quant": quant, "lang": lang,
                             "tag": tag, "gpu": gpu})
    stats["columns"] = sorted(stats["columns"])
    stats["measured_langs"] = sorted(c for c in measured if c in FL)
    return plan, stats


def group_invocations(plan: list[dict]) -> list[dict]:
    """One `sweep` call per (language, tag, quant, gpu).

    `sweep` takes a single --dataset, --language, --quants and --gpu, so those
    four fields are what a container fan-out can share; the models that agree
    on all four ride together."""
    groups: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for r in plan:
        groups[(r["lang"], r["tag"], r["quant"], r["gpu"])].append(r["model"])
    return [{"lang": l, "tag": t, "quant": q, "gpu": g, "models": sorted(ms)}
            for (l, t, q, g), ms in sorted(groups.items())]


# -------- dispatch ---------------------------------------------------------

def run_prefetch(langs: list[str], jobs: int) -> int:
    cmd = ["modal", "run", "scripts/wer/remote/modal_sweep.py::prefetch",
           "--datasets", ",".join(f"fleurs:{l}" for l in langs),
           "--jobs", str(jobs)]
    print("$ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=REPO).returncode


def run_invocation(inv: dict, batch: int, logdir: pathlib.Path) -> int:
    log = logdir / f"{inv['lang']}-{inv['tag']}.{inv['quant']}.{inv['gpu']}.log"
    cmd = [
        "modal", "run", "scripts/wer/remote/modal_sweep.py::sweep",
        "--models", ",".join(inv["models"]),
        "--dataset", f"fleurs:{inv['lang']}",
        "--quants", inv["quant"],
        "--batch-sizes", str(batch),
        "--gpu", inv["gpu"],
        "--language", inv["tag"],
    ]
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        return subprocess.run(cmd, cwd=REPO, stdout=f,
                              stderr=subprocess.STDOUT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--preflight", action="store_true",
                    help="ingest every manifest the plan needs and exit; CPU only")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated hf_card slugs")
    ap.add_argument("--skip", default="",
                    help="comma-separated hf_card slugs to EXCLUDE. Cells "
                         "already scored for a skipped model are left on disk; "
                         "this stops further spend on it, it does not retract "
                         "measurements already paid for.")
    ap.add_argument("--quants", default="", help="restrict to these quants")
    ap.add_argument("--langs", default="", help="restrict to these canonical languages")
    ap.add_argument("--cap", type=float, default=120.0,
                    help="stop DISPATCHING once THIS RUN has added this many "
                         "dollars. Incremental, not an absolute daily total: "
                         "Modal bills by UTC day, so a long sweep started in "
                         "the afternoon crosses midnight UTC and an absolute "
                         "cap silently resets to a fresh budget mid-run.")
    ap.add_argument("--batch", type=int, default=MATRIX_BATCH)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-skip-done", action="store_true",
                    help="re-run cells whose hyp already exists at the matrix "
                         "batch size (use when an engine change invalidates "
                         "them, e.g. the voxtral offline-delay default)")
    ap.add_argument("--refresh-sizes", action="store_true",
                    help="re-query GGUF sizes from the Hub instead of the cache")
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()

    only = {m.strip() for m in args.only.split(",") if m.strip()} or None
    plan, stats = build_plan(only, not args.no_skip_done, args.refresh_sizes)
    if args.skip:
        drop = {m.strip() for m in args.skip.split(",") if m.strip()}
        unknown = drop - {r["model"] for r in plan}
        plan = [r for r in plan if r["model"] not in drop]
        if unknown:
            print(f"note: --skip named {sorted(unknown)}, which the plan does "
                  f"not contain (already complete, or not an ASR card)")
    if args.quants:
        keep = {q.strip() for q in args.quants.split(",") if q.strip()}
        plan = [r for r in plan if r["quant"] in keep]
    if args.langs:
        keep = {l.strip() for l in args.langs.split(",") if l.strip()}
        plan = [r for r in plan if r["lang"] in keep]

    invs = group_invocations(plan)
    need = sorted({r["lang"] for r in plan})
    on_volume = volume_manifest_langs()
    # `measured_langs` says which languages this laptop has reports for, which
    # is the right tiebreak for naming but says nothing about what the cells
    # can read. Fall back to it only when the Volume cannot be listed.
    ingested = on_volume if on_volume is not None else set(stats["measured_langs"])
    new = [l for l in need if l not in ingested]

    print(f"models        : {stats['models']}")
    print(f"columns       : {len(stats['columns'])} FLEURS languages")
    print(f"cells to run  : {len(plan)}   (skipped {stats['skipped']} already "
          f"generated at b{MATRIX_BATCH})")
    print(f"invocations   : {len(invs)}")
    print(f"quants        : {sorted({r['quant'] for r in plan})}")
    print(f"gpus          : {sorted({r['gpu'] for r in plan})}")
    src = "Modal volume" if on_volume is not None else "local reports (volume unreachable)"
    print(f"manifests     : {len(need)} needed, {len(new)} not yet ingested "
          f"[per {src}]")
    if new:
        print(f"  to ingest   : {' '.join(new)}")

    if args.plan:
        by_model: dict[tuple[str, str], list[str]] = defaultdict(list)
        for r in plan:
            by_model[(r["model"], r["quant"])].append(r["lang"])
        print()
        for (m, q), ls in sorted(by_model.items()):
            print(f"  {m:<40} {q:<7} {len(ls):>3} langs")
        return 0

    if args.preflight:
        if not need:
            print("\nnothing to ingest")
            return 0
        print(f"\ningesting {len(need)} manifest(s) (CPU only, no GPU cost)")
        return run_prefetch(need, args.jobs)

    if not args.run:
        print("\n(nothing dispatched; pass --plan, --preflight or --run)")
        return 0

    if new:
        sys.exit(f"refusing to dispatch: {len(new)} manifest(s) have never been "
                 f"ingested. Run --preflight first so a Hub-side failure costs "
                 f"CPU seconds rather than a dispatched GPU sweep.")

    logdir = pathlib.Path(args.logdir) if args.logdir else \
        REPORTS / "_fleurs_full_matrix_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    start_spend = spend_today()
    if start_spend < 0:
        sys.exit("cannot read Modal billing; refusing to dispatch without a "
                 "spend baseline to measure the cap against")
    print(f"\nspend at start: ${start_spend:.2f} (UTC day)   "
          f"budget for this run: ${args.cap:.2f}")
    print(f"logs: {logdir}\n")

    state = {"done": 0, "held": [], "stop": False, "blind": 0,
             "baseline": start_spend}
    lock = threading.Lock()

    def dispatch(inv: dict) -> None:
        # Checked at dispatch time so in-flight invocations always finish.
        with lock:
            if state["stop"]:
                state["held"].append(inv)
                return
            spend = spend_today()
            # Measured against this run's own baseline. A negative delta means
            # the UTC day rolled over mid-sweep, so re-baseline rather than
            # treating the reset as free budget.
            if spend >= 0 and spend < start_spend:
                print(f"[note] UTC billing day rolled over "
                      f"(${start_spend:.2f} -> ${spend:.2f}); re-baselining")
                state["baseline"] = spend
            used = spend - state["baseline"] if spend >= 0 else -1.0
            if spend < 0:
                state["blind"] += 1
                print(f"[WARN] billing query failed "
                      f"({state['blind']}/{MAX_BLIND_DISPATCHES})")
                if state["blind"] >= MAX_BLIND_DISPATCHES:
                    state["stop"] = True
                    state["held"].append(inv)
                    print("[HOLD] no billing reading; refusing to continue blind.")
                    return
            else:
                state["blind"] = 0
                if used >= args.cap:
                    state["stop"] = True
                    state["held"].append(inv)
                    print(f"[HOLD] this run has added ${used:.2f} >= "
                          f"budget ${args.cap:.2f}")
                    return
        t0 = time.time()
        rc = run_invocation(inv, args.batch, logdir)
        with lock:
            state["done"] += 1
            print(f"[{state['done']}/{len(invs)}] fleurs:{inv['lang']} "
                  f"tag={inv['tag']} {inv['quant']} {inv['gpu']} "
                  f"n={len(inv['models'])} rc={rc} {time.time() - t0:.0f}s", flush=True)

    # First invocation alone: it warms the sm_XX CUDA build, and concurrent
    # no-op builds would race on one build dir.
    dispatch(invs[0])
    if len(invs) > 1 and not state["stop"]:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            list(ex.map(dispatch, invs[1:]))

    end_spend = spend_today()
    print(f"\ndispatched {state['done']}/{len(invs)}; this run added "
          f"${end_spend - state['baseline']:.2f} (UTC day now ${end_spend:.2f})")
    if state["held"]:
        print("held:", " ".join(f"{h['lang']}/{h['quant']}" for h in state["held"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
