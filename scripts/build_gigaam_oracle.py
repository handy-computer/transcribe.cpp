#!/usr/bin/env python3
"""
Build the per-variant dump_coverage.json files and the family-level
tests/tolerances/gigaam.json from the on-disk tensor sidecars produced
by scripts/dump_reference_gigaam_author.py.

One-shot helper used at the end of Stage 2. Not invoked from the
runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "validate" / "gigaam"
TOL = ROOT / "tests" / "tolerances" / "gigaam.json"
FAMILY = "gigaam"

VARIANTS = [
    "gigaam-v3-e2e-rnnt",
    "gigaam-v3-e2e-ctc",
    "gigaam-v3-rnnt",
    "gigaam-v3-ctc",
]


def is_tensor_sidecar(meta: dict) -> bool:
    return all(k in meta for k in ("name", "shape", "dtype", "layout"))


def walk_variant(variant: str) -> list[dict]:
    """Return one entry per tensor sidecar under build/validate/gigaam/<variant>."""
    variant_dir = BUILD / variant
    if not variant_dir.exists():
        print(f"  warn: {variant_dir} missing")
        return []
    entries: list[dict] = []
    for json_path in sorted(variant_dir.rglob("*.json")):
        try:
            meta = json.loads(json_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  warn: bad JSON at {json_path}: {e}")
            continue
        if not is_tensor_sidecar(meta):
            continue
        rel = json_path.relative_to(variant_dir)
        parts = rel.parts
        # Layout: <case>/<stage_dir>/ref/<name>.json
        if len(parts) < 4 or parts[-2] != "ref":
            print(f"  warn: unexpected layout at {json_path}")
            continue
        case = parts[0]
        stage_dir = parts[1]
        entries.append({
            "case": case,
            "stage_dir": stage_dir,
            "stage": meta.get("stage", ""),
            "name": meta["name"],
            "shape": list(meta["shape"]),
            "dtype": meta["dtype"],
            "rel_path": str(rel),
        })
    return entries


def write_coverage(variant: str, entries: list[dict]) -> Path:
    out = BUILD / variant / "dump_coverage.json"
    payload = {
        "family": FAMILY,
        "variant": variant,
        "tensors": entries,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def aggregate_tolerances(per_variant: dict[str, list[dict]]) -> dict[str, dict]:
    """For each tensor name, take the max p99_abs and rms across variants
    where that tensor exists, and size the magnitude-aware budget against
    that worst-case magnitude.

    Returns: {name: {max_abs, mean_abs, _provisional, _seen_in}}
    """
    # Collect (name -> [(variant, rel_path, p99_abs, rms)])
    stats: dict[str, list[tuple[str, float, float]]] = {}
    for variant, entries in per_variant.items():
        for e in entries:
            json_path = BUILD / variant / e["rel_path"]
            meta = json.loads(json_path.read_text())
            p99 = float(meta.get("p99_abs", 0.0))
            rms = float(meta.get("rms", 0.0))
            stats.setdefault(e["name"], []).append((variant, p99, rms))

    out: dict[str, dict] = {}
    for name, rows in sorted(stats.items()):
        worst_p99 = max(r[1] for r in rows)
        worst_rms = max(r[2] for r in rows)
        max_abs = max(1e-4 * worst_p99, 1e-6)
        mean_abs = max(1e-5 * worst_rms, 1e-6)
        out[name] = {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "_provisional": True,
            "_seen_in": sorted({r[0] for r in rows}),
        }
    return out


def main() -> int:
    if not BUILD.exists():
        print(f"error: {BUILD} does not exist; run the dumper first")
        return 2

    per_variant: dict[str, list[dict]] = {}
    print("=== dump_coverage.json per variant ===")
    for v in VARIANTS:
        entries = walk_variant(v)
        if not entries:
            print(f"  {v}: 0 tensors (skipping coverage write)")
            continue
        cov_path = write_coverage(v, entries)
        per_variant[v] = entries
        print(f"  {v}: {len(entries)} tensors -> {cov_path.relative_to(ROOT)}")

    print()
    print("=== tolerances aggregation ===")
    tols = aggregate_tolerances(per_variant)
    payload: dict = {
        "_comment": [
            "GigaAM per-tensor tolerances for compare_tensors.py.",
            "",
            "CORRECTNESS REGIME",
            "- Reference: salute-developers/GigaAM @ 6e4b027c (author repo)",
            "  loaded via gigaam.load_model(..., fp16_encoder=False, device='cpu').",
            "  Dumper: scripts/dump_reference_gigaam_author.py.",
            "- C++: ggml CPU fp32 compute, weights stored as F32 in GGUF.",
            "- KV cache dtype: not applicable (RNN-T / CTC heads, no attention KV).",
            "- Mel frontend: C++ MelFrontend with center=false, htk mel, no slaney norm,",
            "  log(clamp(x, 1e-9, 1e9)) scaling.",
            "",
            "ENTRY SOURCING (provisional)",
            "- Per-tensor max_abs = max(1e-4 × p99_abs, 1e-6)",
            "- Per-tensor mean_abs = max(1e-5 × rms, 1e-6)",
            "- When the same tensor name appears in multiple variants, the worst-case",
            "  p99_abs and rms across variants drive the budget (so a single budget",
            "  works for all five variants).",
            "- All entries carry _provisional: true. Stage 4 finalizes against",
            "  observed C++ drift and removes the flag tensor-by-tensor.",
            "",
            "DO NOT SHIP a model while _provisional entries remain.",
        ],
        **tols,
    }
    TOL.parent.mkdir(parents=True, exist_ok=True)
    TOL.write_text(json.dumps(payload, indent=2) + "\n")

    tol_max_vals = sorted(v["max_abs"] for v in tols.values())
    n = len(tol_max_vals)
    print(f"  wrote {TOL.relative_to(ROOT)} with {n} tensor entries")
    if n:
        print(
            f"  max_abs distribution: "
            f"min={tol_max_vals[0]:.3e} "
            f"median={tol_max_vals[n // 2]:.3e} "
            f"max={tol_max_vals[-1]:.3e}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
