#!/usr/bin/env python3
"""
Build the per-variant dump_coverage.json files and the family-level
tests/tolerances/crisperwhisper.json from the on-disk tensor sidecars
produced by scripts/dump_reference_crisperwhisper_author.py.

One-shot helper used at the end of Stage 2. Not invoked from the runtime.

Layout note: validate.py's `cmd_ref` writes both the `encoder` and
`decode` subcommands into a single
`build/validate/<family>/<variant>/<case>/ref/` directory, and
compare_tensors.py diffs that directory against its `cpp/` sibling. So
the live layout is `<case>/ref/<name>.json` — two levels, not the
`<case>/<stage_dir>/ref/` shape some older coverage files still record.
`stage_dir` is emitted here for schema compatibility; the meaningful
grouping is the per-tensor `stage` field from the sidecar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAMILY = "crisperwhisper"
BUILD = ROOT / "build" / "validate" / FAMILY
TOL = ROOT / "tests" / "tolerances" / f"{FAMILY}.json"

VARIANTS = ["crisperwhisper-2.0-small"]


def is_tensor_sidecar(meta: dict) -> bool:
    """Discriminator for tensor sidecars.

    transcript.json and word_timestamps.json live in the same directory
    and are NOT tensors; iterating on the .json extension alone would
    pull them into the catalog and corrupt validate.py's tensor list.
    """
    return isinstance(meta, dict) and all(
        k in meta for k in ("name", "shape", "dtype", "layout")
    )


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
        # Live layout: <case>/ref/<name>.json
        if len(parts) < 3 or parts[-2] != "ref":
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
    out.write_text(json.dumps({
        "family": FAMILY,
        "variant": variant,
        "tensors": entries,
    }, indent=2) + "\n")
    return out


def aggregate_tolerances(per_variant: dict[str, list[dict]]) -> dict[str, dict]:
    """Magnitude-aware provisional budgets.

    max_abs  = max(1e-4 x p99_abs, 1e-6)
    mean_abs = max(1e-5 x rms,     1e-6)

    When a tensor name appears in several (case, variant) sidecars, the
    worst-magnitude instance sets the budget.
    """
    stats: dict[str, list[tuple[str, float, float]]] = {}
    for variant, entries in per_variant.items():
        for e in entries:
            meta = json.loads((BUILD / variant / e["rel_path"]).read_text())
            stats.setdefault(e["name"], []).append((
                variant,
                float(meta.get("p99_abs", 0.0)),
                float(meta.get("rms", 0.0)),
            ))

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


COMMENT = [
    "CrisperWhisper 2.0 per-tensor tolerances for compare_tensors.py.",
    "",
    "CORRECTNESS REGIME",
    "- Reference: nyralabs/CrisperWhisper2.0_small @ 57750c47fde52dc1b016ec2bd4bf4704944cf3df,",
    "  driven through the publisher's own package crisperwhisper==2.0.2",
    "  (MIT; https://github.com/nyrahealth/CrisperWhisper), transformers backend,",
    "  on transformers==4.57.6 with attn_implementation=eager.",
    "  Dumper: scripts/dump_reference_crisperwhisper_author.py.",
    "- Weights ship BF16; the reference runs compute_type=float32, i.e. BF16",
    "  storage upcast to F32 compute. That mirrors ggml's BF16-storage /",
    "  F32-compute regime rather than a tighter-than-reality F32-everything",
    "  baseline.",
    "- Decode contract: [verbatim_1..5] prompt tags BEFORE the Whisper prefix",
    "  <|sot|> <|en|> <|transcribe|> <|notimestamps|> (9 tokens total, no",
    "  <|startofprev|> wrapper). forced_decoder_ids and begin_suppress_tokens",
    "  are cleared by the engine; generation_config.suppress_tokens (88 ids) is",
    "  resolved to an explicit list per call.",
    "- Rewind features OFF: hallucination_mitigation, early_eot_recovery, and",
    "  temperature_fallback all default True in the publisher's transcribe() and",
    "  all three can rewind and re-decode. The dumper never enables them, so the",
    "  greedy trajectory is reproducible token-for-token.",
    "",
    "ENTRY SOURCING (provisional)",
    "- Per-tensor max_abs = max(1e-4 x p99_abs, 1e-6)",
    "- Per-tensor mean_abs = max(1e-5 x rms, 1e-6)",
    "- All entries carry _provisional: true. Stage 4 finalizes against observed",
    "  C++ drift and removes the flag tensor-by-tensor.",
    "",
    "NOTE ON MID-ENCODER ACTIVATION MAGNITUDE",
    "- enc.block.{6,8,11}.out carry |values| up to ~8e2 while enc.block.{0,3}.out",
    "  stay under ~5. This is Whisper's well-known outlier-feature behaviour in",
    "  the deeper encoder blocks, not a dump defect; enc.final collapses back to",
    "  ~3e1 after the final LayerNorm. The provisional budgets scale with p99_abs,",
    "  so the deep blocks legitimately get larger absolute budgets.",
    "",
    "NOTE ON FAMILY-SPECIFIC TENSORS",
    "- dec.intended.logits_raw is the prompt-pass logits under the sibling",
    "  [intended_1..5] prefix on the same audio and the same encoder output. Both",
    "  modes are MUST PASS for this port, so the mode-control surface gets its own",
    "  oracle rather than relying on the verbatim path alone.",
    "- dec.xattn.align is the [n_gen, F_enc] mean-over-alignment-heads",
    "  cross-attention matrix that the reference Viterbi word-timing aligner",
    "  consumes (heads from generation_config.alignment_heads). It is the only",
    "  tensor-level target the word-timestamps capability row has.",
    "",
    "DO NOT SHIP a model while _provisional entries remain.",
]


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

    if not per_variant:
        print("error: no tensors found under any variant")
        return 2

    print()
    print("=== tolerances aggregation ===")
    tols = aggregate_tolerances(per_variant)
    TOL.parent.mkdir(parents=True, exist_ok=True)
    TOL.write_text(json.dumps({"_comment": COMMENT, **tols}, indent=2) + "\n")

    vals = sorted(v["max_abs"] for v in tols.values())
    n = len(vals)
    print(f"  wrote {TOL.relative_to(ROOT)} with {n} tensor entries")
    print(
        f"  max_abs distribution: min={vals[0]:.3e} "
        f"median={vals[n // 2]:.3e} max={vals[-1]:.3e}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
