#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml", "requests"]
# ///
"""generate.py — build catalog/<variant>.json from artifacts.

The catalog is downstream of artifacts, always. This reads what exists and
writes a record; it never copies a number out of prose. A section with no
artifact behind it comes out empty, and the coverage report says so.

    uv run scripts/catalog/generate.py                 # every variant
    uv run scripts/catalog/generate.py --variant whisper-large-v3-turbo
    uv run scripts/catalog/generate.py --no-network    # skip Hub reads

Sources, in order of authority:
    the GGUF header      family, params, languages, capability surface
    reports/perf/        speed rows
    reports/wer/         accuracy rows (full splits only)
    the Hub API          published file names and exact sizes
    scripts/hf_cards/    licence and repo names ONLY, until those move
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import sys
import urllib.request

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_kv import hf_token, read_local, read_remote        # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CARDS, WER, PERF = REPO / "scripts/hf_cards", REPO / "reports/wer", REPO / "reports/perf"
OUT = REPO / "catalog"
CACHE = REPO / "build" / "catalog-cache.json"

QUANTS = ("F32", "BF16", "F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
# Derived runs, never publication rows: utterance subsets, streaming modes,
# rescorings, and the reference/bring-up presets.
DERIVED = re.compile(r"\.stream\d|\.r\d+$|\.filtered|notrimpad|trimflash|-timestamps_")
NOT_A_QUANT = ("REF", "CPP")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def cache_put(d: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, indent=1, sort_keys=True))


# ---------------------------------------------------------------- sources

def hub_files(repo: str, c: dict, network: bool) -> dict[str, int]:
    """{filename: size_bytes} for the GGUFs in a Hub repo."""
    if repo in c.get("hub", {}):
        return c["hub"][repo]
    if not network:
        return {}
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true")
    tok = hf_token()
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    try:
        d = json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e:                                   # noqa: BLE001
        print(f"  [warn] Hub query failed for {repo}: {e}", file=sys.stderr)
        return {}
    files = {s["rfilename"]: s.get("size") or 0 for s in d.get("siblings", [])
             if s["rfilename"].endswith(".gguf")}
    c.setdefault("hub", {})[repo] = files
    return files


def gguf_kv(slug: str, repo: str, files: dict, c: dict, network: bool) -> dict:
    """Capability surface + identity, from a local GGUF if present else the Hub."""
    key = f"{repo}|{slug}"
    if key in c.get("gguf", {}):
        return c["gguf"][key]
    local = sorted((REPO / "models").glob(f"*/{slug}-*.gguf"), key=lambda p: p.stat().st_size)
    kv = None
    if local:
        try:
            kv = read_local(local[0])
        except Exception as e:                               # noqa: BLE001
            print(f"  [warn] local GGUF unreadable ({e})", file=sys.stderr)
    if kv is None and network and files:
        smallest = min(files, key=lambda f: files[f] or 1 << 62)
        try:
            kv = read_remote(repo, smallest)
        except Exception as e:                               # noqa: BLE001
            print(f"  [warn] Hub GGUF read failed ({e})", file=sys.stderr)
    if kv is None:
        return {}
    c.setdefault("gguf", {})[key] = kv
    return kv


def recipe_label(score_path: pathlib.Path) -> str:
    """A deterministic label for the knobs run.py stamped into the hyp header.

    Read from the artifact, never asserted: a row's recipe is whatever actually
    ran. The label is a canonical rendering of the stamped knobs, so two runs
    with the same knobs always land on the same label.
    """
    hyp = pathlib.Path(str(score_path).replace(".score.json", ".jsonl"))
    knobs = {}
    if hyp.exists():
        try:
            with open(hyp) as f:
                first = json.loads(f.readline())
            knobs = first.get("recipe") or {}
        except Exception:                                    # noqa: BLE001
            knobs = {}
    ts = knobs.get("timestamps", "none")
    bs = knobs.get("batch_size", 1)
    if not knobs:                       # fall back to the filename's own tags
        name = score_path.name
        ts = "segment" if ".ts-segment" in name else "none"
        m = re.search(r"\.b(\d+)\.", name)
        bs = int(m.group(1)) if m else 1
    return f"{'ts-' + ts if ts != 'none' else 'standard'}" + (f".b{bs}" if bs != 1 else "")


def accuracy_rows(slug: str) -> list[dict]:
    rows = []
    for p in sorted(WER.glob(f"{slug}-*.score.json")):
        stem = p.name[:-len(".score.json")]
        model_part, _, rest = stem.partition(".")
        q = next((x for x in QUANTS + NOT_A_QUANT if model_part.endswith("-" + x)), None)
        if q is None or q in NOT_A_QUANT or model_part[:-(len(q) + 1)] != slug:
            continue
        if DERIVED.search(rest):
            continue
        dataset_id = rest.split(".")[0]
        if dataset_id.startswith("fleurs-"):
            lang = dataset_id[len("fleurs-"):]
            if not re.fullmatch(r"[a-z]{2,3}(-[a-z]{2,4})?", lang):
                continue                        # subset ids like fleurs-ru-508
            dataset, split = "fleurs", "test"
        elif dataset_id.startswith("librispeech-"):
            dataset, split, lang = "librispeech", dataset_id[len("librispeech-"):], "en"
        else:
            continue
        d = json.loads(p.read_text())
        pu = d.get("per_utterance") or []
        err = d.get("error_rate_pct", d.get("wer_pct"))
        if err is None:
            continue
        lo = d.get("error_rate_ci_lo", d.get("wer_ci_lo"))
        hi = d.get("error_rate_ci_hi", d.get("wer_ci_hi"))
        rows.append({
            "dataset": dataset, "split": split, "language": lang, "quant": q,
            "metric": d.get("metric", "wer"), "err_pct": err,
            "ci95": [round(lo * 100, 2) if lo is not None else None,
                     round(hi * 100, 2) if hi is not None else None],
            "n_utts": d["n"], "recipe": recipe_label(p),
            "engine_sha": None, "measured_on": None,
            "errors": {"sub": d["substitutions"], "del": d["deletions"], "ins": d["insertions"]},
            "empty_hyp": sum(1 for u in pu if not (u.get("hyp") or "").strip()) if pu else None,
            "utts_over_50pct": sum(1 for u in pu if (u.get(d.get("metric", "wer")) or 0) > 0.5)
            if pu else None,
        })
    return sorted(rows, key=lambda r: (r["dataset"], r["language"],
                                       QUANTS.index(r["quant"])))


def speed_rows(variant: str) -> list[dict]:
    rows = []
    for p in sorted(PERF.glob(f"*/*_{variant}_*.json")):
        if "_uncooled_backup" in str(p):
            continue
        d = json.loads(p.read_text())
        machine = d.get("machine", {}).get("slug")
        for r in d.get("runs", []):
            s = r.get("summary") or {}
            if "total_ms" not in s or "quant" not in r or "sample" not in r:
                continue        # bench reports predating the per-run quant/sample fields
            rows.append({
                "machine": machine, "backend": d["backend"], "quant": r["quant"].upper(),
                "sample": r["sample"], "sample_duration_s": round(r["sample_duration_s"], 3),
                "total_ms": round(s["total_ms"]["mean"], 1),
                "xrt_compute": r.get("rtf_compute_mean"),
                "load_ms": round(r["load_ms"], 1),
                "mel_ms": round(s["mel_ms"]["mean"], 1),
                "encode_ms": round(s["encode_ms"]["mean"], 1),
                "decode_ms": round(s["decode_ms"]["mean"], 1),
                "engine_sha": d.get("git_sha"), "measured_on": d["timestamp"][:10],
                "thermal_gated": None,
            })
    # One row per cell: a later bench of the same cell supersedes an earlier one.
    best: dict[tuple, dict] = {}
    for r in rows:
        best[(r["machine"], r["backend"], r["quant"], r["sample"])] = r
    return sorted(best.values(), key=lambda r: (r["machine"], r["backend"],
                                                QUANTS.index(r["quant"]),
                                                r["sample_duration_s"]))


def capabilities(kv: dict, card: dict) -> dict:
    """Derived from the GGUF's own capability surface; the card is not consulted.

    An absent KV means the loader's default, which is false -- the "information
    gap, not a claim" rule. verified is left false for every row: only a Stage 4
    Capability Validation observation may set it.
    """
    def cap(flag: bool, **extra) -> dict:
        return {"supported": True, **extra, "verified": False} if flag else {"supported": False}

    ts_kinds = []
    if kv.get("stt.capability.word_timestamps"):
        ts_kinds.append("word")
    if kv.get("stt.capability.timestamps"):
        ts_kinds.append("segment")
    out = {
        "transcribe": {"supported": True, "verified": False},
        "translate": cap(bool(kv.get("stt.capability.translate")),
                         targets=kv.get("stt.translation.target_languages"),
                         pairs=kv.get("stt.translation.pairs")),
        "lang_detect": cap(bool(kv.get("stt.capability.lang_detect"))),
        "timestamps": cap(bool(ts_kinds), granularities=ts_kinds or None),
        "streaming": cap(bool(kv.get("stt.capability.streaming"))),
        "diarize": cap(bool(kv.get("stt.capability.speaker_diarization")),
                       max_speakers=kv.get("stt.sortformer.max_speakers")),
        "batching": {"supported": True, "verified": False},
    }
    for k, v in out.items():                      # drop null payload keys
        out[k] = {kk: vv for kk, vv in v.items() if vv is not None}
    return out


# ---------------------------------------------------------------- assembly

def build(stem: str, c: dict, network: bool) -> tuple[dict, dict]:
    card = yaml.safe_load((CARDS / f"{stem}.yaml").read_text()) or {}
    quants = card.get("quants") or []
    slug = quants[0]["filename"].rsplit("-", 1)[0] if quants else stem
    repo = card.get("target_repo") or ""
    files = hub_files(repo, c, network) if repo else {}
    kv = gguf_kv(slug, repo, files, c, network)

    downloads = []
    for fn, size in files.items():
        q = next((x for x in QUANTS if fn.endswith(f"-{x}.gguf")), None)
        if q:
            downloads.append({"quant": q, "filename": fn.rsplit("/", 1)[-1], "size_bytes": size})
    downloads.sort(key=lambda d: QUANTS.index(d["quant"]))

    rec = {"schema": "transcribe-catalog-v1", "variant": stem,
           "family": kv.get("general.architecture") or card.get("family"),
           "display_name": card.get("display_name") or stem,
           "params": kv.get("_params"),
           "license": {"spdx": card.get("license"), "display": card.get("license_display")},
           "upstream_repo": card.get("hf_repo"), "published_repo": repo,
           "languages": kv.get("general.languages") or [str(x) for x in (card.get("languages") or [])],
           "language_tag_form": None,
           "long_form_strategy": None,
           "capabilities": capabilities(kv, card),
           "downloads": downloads,
           "accuracy_benchmarks": accuracy_rows(slug),
           "speed_benchmarks": speed_rows(stem)}
    rec = {k: v for k, v in rec.items() if v is not None}

    stamped = any(k.startswith("stt.capability.") for k in kv)
    cov = {"variant": stem, "gguf": bool(kv), "caps": stamped, "downloads": len(downloads),
           "accuracy": len(rec["accuracy_benchmarks"]), "speed": len(rec["speed_benchmarks"])}
    return rec, cov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="", help="one hf_card stem (default: all)")
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    stems = ([args.variant] if args.variant
             else sorted(p.stem for p in CARDS.glob("*.yaml")))
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    c, rows = cache(), []
    for stem in stems:
        try:
            rec, cov = build(stem, c, not args.no_network)
        except Exception as e:                               # noqa: BLE001
            print(f"  [FAIL] {stem}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        (out_dir / f"{stem}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n")
        rows.append(cov)
    cache_put(c)

    print(f"\n{'variant':42s} {'gguf':>5} {'caps':>5} {'files':>6} {'acc':>5} {'speed':>6}")
    for r in rows:
        print(f"{r['variant']:42s} {'yes' if r['gguf'] else 'NO':>5} "
              f"{'yes' if r['caps'] else 'NO':>5} {r['downloads']:>6} "
              f"{r['accuracy']:>5} {r['speed']:>6}")
    blind = [r["variant"] for r in rows if r["gguf"] and not r["caps"]]
    if blind:
        print(f"\n!! {len(blind)} variant(s) whose GGUF carries NO stt.capability.* key, so every "
              f"capability below transcribe reads as unsupported. Their converter never stamped "
              f"them and the family load() hardcodes the answer instead -- these must come from "
              f"transcribe_model_get_capabilities(), not the header:")
        for b in blind:
            print(f"     {b}")
    n = len(rows)
    print(f"\n{n} record(s). identity from GGUF: {sum(1 for r in rows if r['gguf'])}/{n}; "
          f"with accuracy rows: {sum(1 for r in rows if r['accuracy'])}/{n}; "
          f"with speed rows: {sum(1 for r in rows if r['speed'])}/{n}")
    print(f"wrote {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
