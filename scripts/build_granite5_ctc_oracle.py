#!/usr/bin/env python3
"""
Build the per-variant dump_coverage.json file and the family-level
tests/tolerances/granite5_ctc.json from the on-disk tensor sidecars produced
by scripts/dump_reference_granite5_ctc_transformers.py.

One-shot helper used at the end of Stage 2. Not invoked from the runtime.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build" / "validate" / "granite5_ctc"
TOL = ROOT / "tests" / "tolerances" / "granite5_ctc.json"
FAMILY = "granite5_ctc"

VARIANTS = ["granite-speech-5.0-470m-turboctc"]


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
        entries.append({
            "case": parts[0],
            "stage_dir": parts[1],
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
    stats: dict[str, list[tuple[str, str, float, float]]] = {}
    for variant, entries in per_variant.items():
        for e in entries:
            meta = json.loads((BUILD / variant / e["rel_path"]).read_text())
            stats.setdefault(e["name"], []).append(
                (variant, e["case"],
                 float(meta.get("p99_abs", 0.0)), float(meta.get("rms", 0.0))))

    out: dict[str, dict] = {}
    for name, rows in sorted(stats.items()):
        worst_p99 = max(r[2] for r in rows)
        worst_rms = max(r[3] for r in rows)
        out[name] = {
            "max_abs": max(1e-4 * worst_p99, 1e-6),
            "mean_abs": max(1e-5 * worst_rms, 1e-6),
            "_provisional": True,
            "_seen_in": sorted({r[0] for r in rows}),
            "_cases": sorted({r[1] for r in rows}),
        }
    return out


COMMENT = [
    "Granite Speech 5.0 TurboCTC per-tensor tolerances for compare_tensors.py.",
    "",
    "CORRECTNESS REGIME",
    "- Reference: ibm-granite/granite-speech-5.0-470m-turboctc @",
    "  18ca3c1de6cd092b5a30c39fb0f04550b38ed1a0, loaded via mainline",
    "  transformers 5.17.0 (AutoModelForCTC + AutoProcessor,",
    "  attn_implementation=eager). No trust_remote_code.",
    "  Dumper: scripts/dump_reference_granite5_ctc_transformers.py.",
    "- ORACLE DTYPE IS F32, not the BF16 the weights ship as. The BF16->F32",
    "  upcast is lossless, so these are the shipped weights unchanged, and",
    "  F32 activations over BF16 weights is exactly the transcribe.cpp",
    "  compute regime. Measured on jfk, a BF16 forward gives an IDENTICAL",
    "  transcript and identical token ids but tensors that differ from F32",
    "  by up to 2-3% of p99_abs (worst: enc.block.1.post_ff2, max |diff| =",
    "  0.44) -- roughly 250x the 1e-4 x p99_abs budget below. Dumping a BF16",
    "  oracle would have forced blind tolerance widening at Stage 4 and",
    "  hidden real implementation bugs. Do not re-dump at bf16.",
    "- C++ (expected at Stage 4): ggml F32 activations on BF16 GGUF weights.",
    "- KV cache dtype: not applicable (encoder-only; the 'decoder' is one",
    "  tied Linear plus an argmax).",
    "- Mel frontend: GraniteSpeech5FeatureExtractor -- torchaudio",
    "  MelSpectrogram (power=2.0, Hann periodic, center=True, reflect,",
    "  htk scale, norm=None), log10, per-utterance floor at max-8.0, x/4+1,",
    "  concat deltas(win=3), stack 2 frames -> 320-dim input.",
    "",
    "CASES",
    "- jfk  : mel_frames=1100 (EVEN) -> 550 stacked frames, 137 enc frames.",
    "- dots : mel_frames=3533 (ODD)  -> 1767 stacked frames, 441 enc frames.",
    "  The two parities take different feature-extractor paths: the odd case",
    "  right-pads the waveform and keeps the trailing center-pad frame, the",
    "  even case drops it. Per-tensor budgets take the MAX p99_abs / rms",
    "  across both cases, so they cover the worst-magnitude instance.",
    "- Between them the two cases also cover BOTH residual-pooling parities",
    "  at BOTH subsampling blocks, which one case cannot do:",
    "    jfk   550 -> 275 (even in, no frame dropped)",
    "                 -> 137 (odd in,  one frame dropped)",
    "    dots 1767 -> 883 (odd in,   one frame dropped)",
    "                 -> 441 (odd in,   one frame dropped)",
    "  The pooled residual is unfold(1,2,2).mean(-1), which silently drops a",
    "  trailing odd frame, and the conv output is then TRIMMED to that",
    "  length. A C++ implementation that rounds the other way passes jfk",
    "  block 0 and fails everywhere else.",
    "",
    "ENTRY SOURCING (provisional)",
    "- Per-tensor max_abs  = max(1e-4 x p99_abs, 1e-6)",
    "- Per-tensor mean_abs = max(1e-5 x rms,     1e-6)",
    "- All entries carry _provisional: true. Stage 4 finalizes against",
    "  observed C++ drift and removes the flag tensor-by-tensor.",
    "",
    "TENSORS THAT MATTER MOST AT STAGE 4",
    "- enc.block.{0,1}.post_conv is the subsampling observable: it sits",
    "  AFTER the mean-pooled residual plus the TRIMMED stride-2 conv output,",
    "  at the halved frame rate (jfk: 550->275 at block 0, 275->137 at",
    "  block 1). Getting the pool/trim order or the odd-frame drop wrong",
    "  shows up here first, as a shape or a whole-tensor mismatch.",
    "- enc.ctc.mid_logits / enc.ctc.mid_injection are the self-conditioned",
    "  CTC path after block index 7. mid_logits is a 16384-wide softmax",
    "  input; a BF16 softmax there is a plausible precision trap.",
    "- enc.block.7.out is captured BEFORE the mid-injection is added, so",
    "  block 8's input equals enc.block.7.out + enc.ctc.mid_injection.",
    "- enc.ctc_logits comes from ctc_head, which is weight-TIED to",
    "  encoder.out (the checkpoint has no ctc_head.* tensors).",
    "",
    "HOOK SEMANTICS VERIFIED AT STAGE 2 (bit-exact on jfk, max|diff| = 0.0):",
    "  encoder.out(enc.block.7.out)                 == enc.ctc.mid_logits",
    "  encoder.out_mid(softmax(enc.ctc.mid_logits)) == enc.ctc.mid_injection",
    "  ctc_head(enc.out)                            == enc.ctc_logits",
    "  ctc_head.weight/bias are the SAME torch storage as encoder.out",
    "  (identical data_ptr), confirming tie_word_embeddings=true. The",
    "  converter must emit that [16384, 1024] matrix once and point both",
    "  consumers at it. These identities are the Stage 4 contract: they let",
    "  the C++ be checked sub-step by sub-step without re-deriving which",
    "  projection feeds which consumer.",
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
        n_cases = len({e["case"] for e in entries})
        print(f"  {v}: {len(entries)} tensors across {n_cases} case(s) "
              f"-> {cov_path.relative_to(ROOT)}")

    print()
    print("=== tolerances aggregation ===")
    tols = aggregate_tolerances(per_variant)
    TOL.parent.mkdir(parents=True, exist_ok=True)
    TOL.write_text(json.dumps({"_comment": COMMENT, **tols}, indent=2) + "\n")

    vals = sorted(v["max_abs"] for v in tols.values())
    n = len(vals)
    print(f"  wrote {TOL.relative_to(ROOT)} with {n} tensor entries")
    if n:
        print(f"  max_abs distribution: min={vals[0]:.3e} "
              f"median={vals[n // 2]:.3e} max={vals[-1]:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
