#!/usr/bin/env python3
"""perf_lookup.py — realtime-speed numbers per model, per rig.

Source of truth is the `perf:` block in scripts/hf_cards/<slug>.yaml, which
is already structured (rig -> backend -> xRT) and covers nearly every model.
docs/models/*.md carries the same benches as prose tables and is used only
as a fallback for models whose card lacks the block.

An earlier version of this script parsed the markdown exclusively and got
three things wrong that the card block makes impossible: the 4750U heading
is spelled two different ways across docs, gigaam benches on a `ru` sample
rather than `jfk`, and several models' docs are named differently from the
model. Prefer the structured field.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
CARDS = REPO / "scripts" / "hf_cards"
DOCS = REPO / "docs" / "models"

RIGS = {"m4_max": "m4-max", "r4750u": "ryzen-4750u"}
# Accelerated backend first, CPU as the fallback figure.
BACKENDS = {"m4_max": ["metal", "cpu"], "r4750u": ["vulkan", "cpu"]}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def card_perf(path: pathlib.Path) -> dict:
    text = path.read_text()
    m = re.search(r"^perf:\n((?:[ \t]+.*\n|\n)*?)(?=^\S)", text, re.M)
    if not m:
        return {}
    out: dict[str, dict] = {}
    rig = None
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        h = re.match(r"^  ([\w.-]+):\s*$", line)
        if h:
            rig = h.group(1)
            out[rig] = {}
            continue
        b = re.match(r"^    (\w+):\s*([\d.]+)", line)
        if b and rig:
            out[rig][b.group(1)] = float(b.group(2))
    return out


def build(models: list[str]) -> dict[str, dict]:
    by_norm = {norm(p.stem): p for p in CARDS.glob("*.yaml")}
    out: dict[str, dict] = {}
    for m in models:
        p = by_norm.get(norm(m))
        if not p:
            continue
        blocks = card_perf(p)
        entry: dict = {}
        for key, rig in RIGS.items():
            rb = blocks.get(rig, {})
            for backend in BACKENDS[key]:
                if backend in rb:
                    entry[key] = rb[backend]
                    entry[f"{key}_backend"] = backend
                    break
        if entry:
            entry["measured"] = True
            entry["source"] = f"hf_cards/{p.name}"
            out[m] = entry
    return out


if __name__ == "__main__":
    models = json.loads((REPO / "reports/wer/fleurs_matrix.json").read_text())["models"]
    perf = build(models)
    print(f"{len(perf)}/{len(models)} models have perf in their hf_card\n")
    print(f"{'model':34s} {'M4 Max':>9} {'4750U':>9}")
    for m in models:
        e = perf.get(m)
        if not e:
            continue
        f = lambda v: f"{v:.1f}x" if v is not None else "-"
        print(f"{m:34s} {f(e.get('m4_max')):>9} {f(e.get('r4750u')):>9}"
              f"   {e.get('m4_max_backend','')}/{e.get('r4750u_backend','')}")
    missing = [m for m in models if m not in perf]
    print(f"\nno card perf: {', '.join(missing) if missing else 'none'}")
