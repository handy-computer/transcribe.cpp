#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml", "requests"]
# ///
"""backfill_identity.py — write family + params into the hf_card specs.

The GGUF already knows both. `general.architecture` is the family string the
loader dispatches on, and the parameter count is the sum of the tensor
shapes. Neither was ever copied into the card, so nothing downstream could
group the Moonshine variants, or sort by model size.

Weights for most models are no longer on this disk (converted, uploaded,
deleted), so anything missing locally is read from the Hub over a range
request: a couple of MB per model rather than the whole file. From here on
the fields belong in the card when it is authored, in porting-5-quants or
porting-8-ship, where the GGUF is guaranteed to be at hand.

    uv run scripts/hf_cards/backfill_identity.py            # dry run
    uv run scripts/hf_cards/backfill_identity.py --write
    uv run scripts/hf_cards/backfill_identity.py --local-only
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

import requests
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]
CARDS = REPO / "scripts" / "hf_cards"
MODELS = REPO / "models"

sys.path.insert(0, str(REPO / "scripts" / "wer"))
from gguf_header import hf_token, identity, identity_remote   # noqa: E402

# Smallest first: the header is identical across quants, so fetch the file
# whose bulk we are least likely to touch if a header turns out to be large.
QUANT_ORDER = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16", "F32"]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def local_gguf(slug: str) -> pathlib.Path | None:
    dirs = {norm(p.name): p for p in MODELS.glob("*") if p.is_dir()}
    d = dirs.get(norm(slug))
    if not d:
        return None
    files = sorted(d.glob("*.gguf"), key=lambda p: p.stat().st_size)
    return files[0] if files else None


def remote_gguf(repo: str) -> str | None:
    """Pick one GGUF filename from a Hub repo listing."""
    h = {}
    tok = hf_token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    r = requests.get(f"https://huggingface.co/api/models/{repo}",
                     headers=h, timeout=30)
    if not r.ok:
        return None
    names = [s["rfilename"] for s in r.json().get("siblings", [])
             if s["rfilename"].endswith(".gguf")]
    for q in QUANT_ORDER:
        for n in names:
            if n.endswith(f"-{q}.gguf"):
                return n
    return names[0] if names else None


def insert_fields(text: str, family: str, params: str) -> str | None:
    """Add the two lines after the licence block, leaving the rest as is.

    A yaml round-trip would drop every comment in the file, and these specs
    are commented documents, so the edit is textual and anchored."""
    if re.search(r"^family:", text, re.M):
        return None
    m = None
    for pat in (r"^license_link:.*$", r"^license_name:.*$",
                r"^license_display:.*$", r"^license:.*$"):
        found = list(re.finditer(pat, text, re.M))
        if found:
            m = found[-1]
            break
    if not m:
        return None
    add = (f"\n\n# Identity, read from the GGUF: general.architecture, and the\n"
           f"# parameter count summed from the tensor shapes.\n"
           f"family: {family}\nparams: {params}")
    return text[:m.end()] + add + text[m.end():]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="edit the specs (default is a dry run)")
    ap.add_argument("--local-only", action="store_true",
                    help="skip models whose weights are not on this disk")
    ap.add_argument("--only", default="",
                    help="comma-separated card stems, for a targeted run")
    args = ap.parse_args()
    only = {norm(s) for s in args.only.split(",") if s.strip()}

    filled, failed, already = [], [], 0
    for card in sorted(CARDS.glob("*.yaml")):
        if only and norm(card.stem) not in only:
            continue
        spec = yaml.safe_load(card.read_text()) or {}
        if spec.get("family"):
            already += 1
            continue
        path, where = local_gguf(card.stem), "local"
        try:
            if path:
                info = identity(path)
                src = path.name
            elif args.local_only or not spec.get("target_repo"):
                failed.append((card.stem, "no local GGUF"))
                continue
            else:
                repo = spec["target_repo"]
                name = remote_gguf(repo)
                if not name:
                    failed.append((card.stem, f"no GGUF in {repo}"))
                    continue
                info = identity_remote(repo, name)
                where, src = "hub", f"{repo}/{name}"
        except Exception as e:                      # noqa: BLE001
            failed.append((card.stem, f"{type(e).__name__}: {e}"))
            continue
        if not info["family"]:
            failed.append((card.stem, "no general.architecture in the header"))
            continue
        params = f"{round(info['params'] / 1e6)}M"
        print(f"  {card.stem:38s} {where:6s} family={info['family']:20s} "
              f"params={params:>7s}  ({src})")
        filled.append((card, info["family"], params))

    if args.write:
        for card, family, params in filled:
            new = insert_fields(card.read_text(), family, params)
            if new:
                card.write_text(new)

    print(f"\n{len(filled)} card(s) {'written' if args.write else 'to write'}, "
          f"{already} already had it, {len(failed)} failed")
    for stem, why in failed:
        print(f"  FAIL {stem}: {why}")
    if not args.write and filled:
        print("\ndry run; pass --write to apply")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
