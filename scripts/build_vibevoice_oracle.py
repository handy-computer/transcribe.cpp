#!/usr/bin/env python3
"""
Build build/validate/vibevoice/<variant>/dump_coverage.json and the
family-level tests/tolerances/vibevoice.json from the on-disk tensor sidecars
produced by scripts/dump_reference_vibevoice_author.py (run on Modal).

One-shot helper used at the end of Stage 2. Not invoked from the runtime.
Layout: build/validate/vibevoice/<variant>/<case>/ref/<name>.{f32,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "validate" / "vibevoice"
TOL = ROOT / "tests" / "tolerances" / "vibevoice.json"
FAMILY = "vibevoice"
VARIANTS = ["vibevoice-asr"]

# non-tensor behavioral artifacts the dumper writes into the ref dir
NON_TENSOR = {"transcript.json", "transcript.parsed.json", "envelope.json",
              "dump_coverage.json"}


def is_tensor_sidecar(meta: dict) -> bool:
    return all(k in meta for k in ("name", "shape", "dtype", "layout"))


def walk_variant(variant: str) -> list[dict]:
    variant_dir = BUILD / variant
    if not variant_dir.exists():
        print(f"  warn: {variant_dir} missing")
        return []
    entries: list[dict] = []
    for json_path in sorted(variant_dir.rglob("*.json")):
        if json_path.name in NON_TENSOR:
            continue
        try:
            meta = json.loads(json_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  warn: bad JSON at {json_path}: {e}")
            continue
        if not is_tensor_sidecar(meta):
            continue
        rel = json_path.relative_to(variant_dir)
        parts = rel.parts
        # Layout: <case>/ref/<name>.json
        if len(parts) != 3 or parts[-2] != "ref":
            print(f"  warn: unexpected layout at {json_path}")
            continue
        entries.append({
            "case": parts[0],
            "stage_dir": "ref",
            "stage": meta.get("stage", ""),
            "name": meta["name"],
            "shape": list(meta["shape"]),
            "dtype": meta["dtype"],
            "rel_path": str(rel),
        })
    return entries


def write_coverage(variant: str, entries: list[dict]) -> Path:
    out = BUILD / variant / "dump_coverage.json"
    out.write_text(json.dumps(
        {"family": FAMILY, "variant": variant, "tensors": entries}, indent=2) + "\n")
    return out


def aggregate_tolerances(per_variant: dict[str, list[dict]]) -> dict[str, dict]:
    stats: dict[str, list[tuple[str, float, float]]] = {}
    for variant, entries in per_variant.items():
        for e in entries:
            meta = json.loads((BUILD / variant / e["rel_path"]).read_text())
            stats.setdefault(e["name"], []).append(
                (variant, float(meta.get("p99_abs", 0.0)), float(meta.get("rms", 0.0))))
    out: dict[str, dict] = {}
    for name, rows in sorted(stats.items()):
        worst_p99 = max(r[1] for r in rows)
        worst_rms = max(r[2] for r in rows)
        out[name] = {
            "max_abs": max(1e-4 * worst_p99, 1e-6),
            "mean_abs": max(1e-5 * worst_rms, 1e-6),
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
            print(f"  {v}: 0 tensors (skipping)")
            continue
        cov = write_coverage(v, entries)
        per_variant[v] = entries
        print(f"  {v}: {len(entries)} tensors -> {cov.relative_to(ROOT)}")

    tols = aggregate_tolerances(per_variant)
    payload: dict = {
        "_comment": [
            "VibeVoice-ASR per-tensor tolerances for compare_tensors.py.",
            "",
            "CORRECTNESS REGIME",
            "- Reference: microsoft/VibeVoice-ASR via the author `vibevoice` package",
            "  @ 303b2833e01cff4578ec278bbfe536da54bd19fe (HF rev",
            "  d0c9efdb8d614685062c04425d91e01b6f37d944), run on a Modal L40S GPU,",
            "  attn_implementation=eager. Dumper: scripts/dump_reference_vibevoice_author.py.",
            "- Audio-LLM: raw 24kHz -> acoustic+semantic causal-conv VAE tokenizers",
            "  (loaded fp32) -> SpeechConnector -> 3584 -> element-wise SUM -> scattered",
            "  into a Qwen2.5-7B LM (bf16) -> lm_head. C++: ggml, BF16 weights for the LM.",
            "",
            "DETERMINISM",
            "- The acoustic path samples a Gaussian VAE latent (fix_std=0.5); contract",
            "  tensors use the deterministic MEAN (mode()). Verified: transcription content",
            "  is identical mean-vs-sampled, and the lm_head argmax is stable across 8 noisy",
            "  passes (see <case>/ref/envelope.json for the sampled std band). The C++ port",
            "  must use the acoustic mean, not a sample.",
            "",
            "ENTRY SOURCING (provisional)",
            "- Per-tensor max_abs = max(1e-4 x p99_abs, 1e-6)",
            "- Per-tensor mean_abs = max(1e-5 x rms, 1e-6)",
            "- All entries carry _provisional: true. Stage 4 finalizes against observed",
            "  C++ drift and removes the flag tensor-by-tensor.",
            "- enc.* (VAE frontend) runs fp32 in the reference; enc.combined feeds the",
            "  bf16 LM, so expect the largest C++ drift to localize there.",
            "",
            "DO NOT SHIP a model while _provisional entries remain.",
        ],
        **tols,
    }
    TOL.parent.mkdir(parents=True, exist_ok=True)
    TOL.write_text(json.dumps(payload, indent=2) + "\n")
    vals = sorted(v["max_abs"] for v in tols.values())
    n = len(vals)
    print(f"\n  wrote {TOL.relative_to(ROOT)} with {n} tensor entries")
    if n:
        print(f"  max_abs distribution: min={vals[0]:.3e} "
              f"median={vals[n // 2]:.3e} max={vals[-1]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
