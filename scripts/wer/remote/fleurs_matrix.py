#!/usr/bin/env python3
"""fleurs_matrix.py — drive the cross-model FLEURS WER matrix on Modal.

Builds the (model, language) run plan from the hf_cards' own `languages:`
lists, groups it into `modal_sweep.py::sweep` invocations, and dispatches
them under a spend cap read from Modal's billing API.

Why a driver instead of one sweep call: `sweep` applies a single
`--language` to every cell in the invocation and takes one `--dataset`, so
the matrix needs one invocation per (language, language-tag) pair. Models
mostly agree on bare BCP-47 codes, but a few disagree on aliases (whisper
says `tl`/`no` where fun-asr says `fil` and nemotron says `nb`), which is
why the plan carries a per-model tag rather than assuming the ingest code.

Usage:
    uv run scripts/wer/remote/fleurs_matrix.py --plan          # print, run nothing
    uv run scripts/wer/remote/fleurs_matrix.py --run --cap 60
"""
from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

REPO = pathlib.Path(__file__).resolve().parents[3]

CORE_MODELS = [
    "parakeet-unified-en-0.6b", "nemotron-3.5-asr-streaming-0.6b",
    "parakeet-tdt-0.6b-v3", "qwen3-asr-0.6b", "qwen3-asr-1.7b",
    "sensevoice-small", "gigaam-v3-e2e-rnnt", "canary-180m-flash",
    "canary-1b-v2", "cohere-transcribe-03-2026", "whisper-large-v3-turbo",
    "whisper-large-v3", "whisper-medium", "voxtral-mini-4b-realtime-2602",
    "breeze-asr-25",
]

# Phase 2: dispatched only if the cap still has room after the core matrix.
EXTRA_MODELS = [
    "fun-asr-mlt-nano-2512",   # 31 languages, no published numbers anywhere
    "whisper-small",           # weak-hardware floor across the union
    "parakeet-primeline",      # German specialist
]


def fleurs_langs() -> dict[str, str]:
    """BCP-47 -> FLEURS config, parsed out of ingest.py (which imports numpy,
    so this reads the literal rather than importing the module)."""
    src = (REPO / "scripts/wer/ingest.py").read_text()
    body = re.search(r"FLEURS_LANGS: dict\[str, str\] = \{(.*?)\n\}", src, re.S).group(1)
    return ast.literal_eval("{" + re.sub(r"#.*", "", body) + "}")


# Models whose accepted --language tag is NOT the bare code its hf_card
# advertises. The CLI validates the hint against the GGUF's general.languages
# and nemotron's converter writes locales there ("ar-AR"), while its card
# advertises the deduped bare codes ("ar"), so a bare tag is rejected with
# UNSUPPORTED_LANGUAGE before resolve_prompt_id ever runs.
TAG_OVERRIDE_MODELS = {"nemotron-3.5-asr-streaming-0.6b": "parakeet"}


def parakeet_prompt_locales() -> list[str]:
    """The nemotron locale list, read from the converter that writes it.

    Parsed rather than copied so the two can't drift: convert-parakeet.py is
    what stamps general.languages into the GGUF the CLI validates against.
    """
    src = (REPO / "scripts/convert-parakeet.py").read_text()
    i = src.index('"nemotron-3.5-asr-streaming-0.6b"')
    block = src[i:src.index('"lang_detect"', i)]
    m = re.search(r'"languages":\s*\[(.*?)\]', block, re.S)
    return re.findall(r'"([a-z]{2}-[A-Z]{2})"', m.group(1))


def locale_for(bare: str) -> str | None:
    """Bare code -> the first matching locale in converter order.

    Order matters and is deliberate: the list leads with the regional variant
    FLEURS itself ships (en-US, es-US for es_419, pt-BR for pt_br, fr-FR), so
    first-match also picks the closest regional match rather than an arbitrary
    one."""
    for loc in parakeet_prompt_locales():
        if loc.split("-")[0] == bare:
            return loc
    return None


def card_langs(model: str) -> list[str]:
    txt = (REPO / "scripts/hf_cards" / f"{model}.yaml").read_text()
    m = re.search(r"^languages:\n((?:\s+-\s+\S+\n)+)", txt, re.M)
    return [x.strip('"') for x in re.findall(r"-\s+(\S+)", m.group(1))] if m else []


def build_plan(models: list[str]) -> tuple[list[dict], dict[str, str]]:
    """Return (plan rows, {fleurs_config: canonical ingest code}).

    The union is defined by the CORE specialist models only, and deliberately
    not by `models`. Two reasons: whisper's 99-language list would otherwise
    pull in languages no candidate can be compared against, and a phase-2
    model must not widen the language set after the core matrix has already
    run, or the late-added rows would cover languages the core rows never
    measured. Extras therefore add rows, never columns."""
    FL = fleurs_langs()
    specialists = [m for m in CORE_MODELS if not m.startswith("whisper")]
    union: dict[str, str] = {}
    for m in specialists:
        for l in card_langs(m):
            if l in FL:
                union.setdefault(FL[l], l)
    plan = []
    for m in models:
        have = card_langs(m)
        for cfg, code in sorted(union.items()):
            tag = next((l for l in have if FL.get(l) == cfg), None)
            if not tag:
                continue
            if m in TAG_OVERRIDE_MODELS:
                tag = locale_for(tag)
                if tag is None:      # advertised bare code with no locale
                    continue
            plan.append({"model": m, "lang": code, "tag": tag})
    return plan, union


def plan_from_pairs(pairs: dict[str, list[str]]) -> list[dict]:
    """Ragged plan: an explicit {card_slug: [bare lang codes]} mapping.

    Unlike build_plan this does NOT compute a union. Each model carries its
    own language list, which is what a quantization probe wants: the
    comparison is within-model across quants, so models need a spread of
    easy-to-hard languages from their own supported set, not a shared column
    set. Tag resolution is identical to build_plan's so alias models
    (whisper's tl/no vs fil/nb) and locale models resolve the same way.
    """
    FL = fleurs_langs()
    plan = []
    for m, langs in sorted(pairs.items()):
        have = card_langs(m)
        for code in langs:
            cfg = FL.get(code)
            if cfg is None:
                sys.exit(f"{m}: {code!r} is not a FLEURS language")
            tag = next((l for l in have if FL.get(l) == cfg), None)
            if not tag:
                sys.exit(f"{m}: card does not advertise {code!r} "
                         f"(fleurs config {cfg})")
            if m in TAG_OVERRIDE_MODELS:
                tag = locale_for(tag)
                if tag is None:
                    sys.exit(f"{m}: no locale for {code!r}")
            plan.append({"model": m, "lang": code, "tag": tag})
    return plan


def group_invocations(plan: list[dict]) -> list[dict]:
    """One invocation per (ingest language, language tag). Models that agree
    on the tag share a container fan-out."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in plan:
        groups[(r["lang"], r["tag"])].append(r["model"])
    return [{"lang": lang, "tag": tag, "models": sorted(ms)}
            for (lang, tag), ms in sorted(groups.items())]


def spend_today(attempts: int = 3) -> float:
    """Actual workspace spend for today from Modal's billing API.

    Returns -1.0 only after `attempts` consecutive failures. The query is
    occasionally flaky (seen once mid-sweep), and a single blip must not be
    allowed to look like a budget reading."""
    for i in range(attempts):
        try:
            out = subprocess.run(
                ["modal", "billing", "report", "--for", "today", "--json"],
                capture_output=True, text=True, timeout=120)
            if out.returncode == 0:
                return sum(float(r["Cost"]) for r in json.loads(out.stdout))
        except Exception:
            pass
        if i + 1 < attempts:
            time.sleep(5 * (i + 1))
    return -1.0


# Consecutive failed billing reads tolerated before dispatch stops. A few
# blips are normal; flying blind through a whole sweep is not.
MAX_BLIND_DISPATCHES = 3


def run_invocation(inv: dict, gpu: str, batch: int, logdir: pathlib.Path,
                   quants: str = "Q8_0") -> int:
    qtag = quants.replace(',', '+')
    log = logdir / f"{inv['lang']}-{inv['tag']}.{qtag}.log"
    cmd = [
        "modal", "run", "scripts/wer/remote/modal_sweep.py::sweep",
        "--models", ",".join(inv["models"]),
        "--dataset", f"fleurs:{inv['lang']}",
        "--quants", quants,
        "--batch-sizes", str(batch),
        "--gpu", gpu,
        "--language", inv["tag"],
    ]
    with open(log, "w") as f:
        f.write(f"$ {' '.join(cmd)}\n\n")
        f.flush()
        return subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="print the plan and exit")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--extras", action="store_true", help="include phase-2 models")
    ap.add_argument("--only", default="",
                    help="comma-separated models to actually dispatch. The "
                         "language union is still computed from CORE_MODELS, "
                         "so this runs a subset of models over the same "
                         "columns the core matrix used.")
    ap.add_argument("--cap", type=float, default=60.0,
                    help="stop DISPATCHING once today's spend exceeds this "
                         "(in-flight invocations always finish)")
    ap.add_argument("--quants", default="Q8_0",
                    help="comma-separated quant substrings passed to "
                         "modal_sweep (e.g. Q5_K_M,Q4_K_M)")
    ap.add_argument("--pairs-file", default="",
                    help="JSON {card_slug: [lang,...]} for a ragged plan. "
                         "Bypasses the union; --only/--extras do not apply.")
    ap.add_argument("--gpu", default="L4")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=3,
                    help="invocations in flight at once. The first is always "
                         "run alone: it warms the CUDA build, and concurrent "
                         "no-op builds would race on the same build dir.")
    ap.add_argument("--logdir", default=None)
    args = ap.parse_args()

    if args.pairs_file:
        pairs = json.loads(pathlib.Path(args.pairs_file).read_text())
        plan = plan_from_pairs(pairs)
        models = sorted(pairs)
        union = {r["lang"]: r["lang"] for r in plan}
    else:
        models = CORE_MODELS + (EXTRA_MODELS if args.extras else [])
        plan, union = build_plan(models)
    if args.only and not args.pairs_file:
        keep = {m.strip() for m in args.only.split(",") if m.strip()}
        unknown = keep - set(models)
        if unknown:
            sys.exit(f"--only names models not in the plan: {sorted(unknown)}")
        plan = [r for r in plan if r["model"] in keep]
        models = [m for m in models if m in keep]
    invs = group_invocations(plan)

    print(f"models      : {len(models)}")
    print(f"languages   : {len(union)}")
    print(f"runs        : {len(plan)}")
    print(f"invocations : {len(invs)}")
    nq = len([q for q in args.quants.split(",") if q.strip()])
    print(f"quants      : {args.quants}  ({nq} per run)")
    print(f"cells       : {len(plan) * nq}")
    print(f"est audio-h : {len(plan) * nq * 2.4:.0f}")
    print(f"est cost    : ${len(plan) * nq * 2.4 * 0.051:.0f} (upper bound; "
          f"whisper-large-v3 rate applied to every model)")
    if args.plan:
        for inv in invs:
            print(f"  fleurs:{inv['lang']:<4} tag={inv['tag']:<4} "
                  f"n={len(inv['models']):>2}  {' '.join(inv['models'])}")
        return 0
    if not args.run:
        print("\n(nothing dispatched; pass --run)")
        return 0

    logdir = pathlib.Path(args.logdir) if args.logdir else \
        REPO / "reports" / "wer" / "_fleurs_matrix_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    start_spend = spend_today()
    print(f"\nspend at start: ${start_spend:.2f}   cap: ${args.cap:.2f}")
    print(f"logs: {logdir}\n")

    state = {"done": 0, "held": [], "stop": False, "blind": 0}
    lock = threading.Lock()

    def dispatch(i: int, inv: dict) -> None:
        # Cap is checked at DISPATCH time, so anything already running is
        # allowed to finish. That matches "don't kill in-flight work, just
        # stop starting new work".
        with lock:
            if state["stop"]:
                state["held"].append(inv)
                return
            spend = spend_today()
            if spend < 0:
                # Budget visibility lost. Tolerate a couple of blips, but do
                # NOT keep dispatching indefinitely against an unknown spend:
                # a guard that fails open forever is not a guard.
                state["blind"] += 1
                print(f"[WARN] billing query failed "
                      f"({state['blind']}/{MAX_BLIND_DISPATCHES}); "
                      f"dispatching fleurs:{inv['lang']}/{inv['tag']} "
                      f"without a spend reading")
                if state["blind"] >= MAX_BLIND_DISPATCHES:
                    state["stop"] = True
                    state["held"].append(inv)
                    print(f"[HOLD] no billing reading for "
                          f"{MAX_BLIND_DISPATCHES} dispatches; refusing to "
                          f"continue blind. Re-run once billing responds.")
                    return
            else:
                state["blind"] = 0
                if spend >= args.cap:
                    state["stop"] = True
                    state["held"].append(inv)
                    print(f"[HOLD] spend ${spend:.2f} >= cap ${args.cap:.2f}; "
                          f"holding fleurs:{inv['lang']}/{inv['tag']} and the rest")
                    return
        t0 = time.time()
        rc = run_invocation(inv, args.gpu, args.batch, logdir, args.quants)
        with lock:
            state["done"] += 1
            print(f"[{state['done']}/{len(invs)}] fleurs:{inv['lang']} "
                  f"tag={inv['tag']} models={len(inv['models'])} rc={rc} "
                  f"{time.time() - t0:.0f}s spend=${spend:.2f}", flush=True)

    # First invocation alone: warms the sm_XX build so the rest hit a cached
    # binary instead of racing concurrent cmake/ninja runs on one build dir.
    dispatch(1, invs[0])
    if len(invs) > 1 and not state["stop"]:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            list(ex.map(lambda t: dispatch(*t),
                        [(i, inv) for i, inv in enumerate(invs[1:], 2)]))
    done, held = state["done"], state["held"]

    final = spend_today()
    print(f"\ndispatched {done}/{len(invs)} invocations; spend today ${final:.2f}")
    if held:
        print("held:", " ".join(f"{h['lang']}/{h['tag']}" for h in held))
    return 0


if __name__ == "__main__":
    sys.exit(main())
