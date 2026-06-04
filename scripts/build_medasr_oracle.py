#!/usr/bin/env python3
"""
Build the per-variant dump_coverage.json file and the family-level
tests/tolerances/medasr.json from the on-disk tensor sidecars produced
by scripts/dump_reference_medasr_transformers.py.

One-shot helper used at the end of Stage 2. Not invoked from the runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "validate" / "medasr"
TOL = ROOT / "tests" / "tolerances" / "medasr.json"
FAMILY = "medasr"

VARIANTS = ["medasr"]


def is_tensor_sidecar(meta: dict) -> bool:
    return all(k in meta for k in ("name", "shape", "dtype", "layout"))


def walk_variant(variant: str) -> list[dict]:
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
            "MedASR per-tensor tolerances for compare_tensors.py.",
            "",
            "CORRECTNESS REGIME",
            "- Reference: google/medasr @ ae1e4845b4b07479735d93e1e591e566435b7104",
            "  loaded via transformers v5.0.0.dev0 (commit 65dc261512cbdb1ee72b88ae5b222f2605aad8e5,",
            "  AutoModelForCTC + AutoProcessor, attn_implementation=eager, dtype=float32).",
            "  Dumper: scripts/dump_reference_medasr_transformers.py.",
            "- C++: ggml CPU fp32 compute, weights stored as F32 in GGUF.",
            "- KV cache dtype: not applicable (CTC head, no autoregressive decoder).",
            "- Mel frontend: LasrFeatureExtractor — manual unfold + rfft, Hann symmetric,",
            "  no preemph, no dither, no CMVN; kaldi-mel weight matrix with HTK-style",
            "  DC-bin exclusion (125-7500 Hz, 128 bands); log(clamp(power, min=1e-5)).",
            "",
            "ENTRY SOURCING (provisional)",
            "- Per-tensor max_abs = max(1e-4 × p99_abs, 1e-6)",
            "- Per-tensor mean_abs = max(1e-5 × rms, 1e-6)",
            "- All entries carry _provisional: true. Stage 4 finalizes against",
            "  observed C++ drift and removes the flag tensor-by-tensor.",
            "",
            "NOTE ON MID-BLOCK SUB-STEP MAGNITUDES",
            "- enc.block.0.post_{ff1,attn,conv,ff2} carry huge activations (up to ~2e6)",
            "  because the macaron residual scalars (feed_forward_residual_weights=[1.5, 0.5],",
            "  conv_residual_weights=[2.0, 1.0]) amplify the running residual within a block;",
            "  the per-block out_norm collapses them back to ~O(1). The provisional max_abs",
            "  budget reflects the worst-magnitude raw value; do not be alarmed that",
            "  individual post-step budgets are ~1e2 in absolute units — they are scaled",
            "  to ~1e-4 × p99_abs as designed.",
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
