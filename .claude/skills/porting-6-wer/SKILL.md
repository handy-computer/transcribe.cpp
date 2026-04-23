---
name: porting-6-wer
description: Scores WER for a ported model variant against the acceptance target declared in the intake's upstream_benchmarks (dataset + metric + score; LibriSpeech test-clean is only the default policy for English-capable models) and enforces the ref-dtype hard gate of ±0.01 absolute vs. that upstream score. Use after porting-5-bench. Input: full quant matrix, intake-selected acceptance target, acceptance manifest. Output: reports/wer/<variant>-<preset>.<dataset>.score.json per shipped preset, a markdown table summary, and a ship-gate decision. If the ref-dtype model misses by more than 0.01, the skill halts and hands back to porting-4-cpp.
---

# porting-6-wer

Stage 6 of the porting pipeline. The acceptance target is whatever the intake captured in `upstream_benchmarks[0]` — LibriSpeech test-clean for English models, FLEURS/Common Voice/custom dataset for others, as the publisher reports. Produces WER scores with bootstrap CIs for every shipped preset and enforces the ref-dtype hard gate. If the gate fails, the skill does not loosen tolerances or waive the test — it stops and sends the port back to Stage 4.

## Preconditions

- `models/<variant>/<variant>-<PRESET>.gguf` exists for every shipped preset.
- `reports/porting/<family>/<variant>/intake.json` has `upstream_benchmarks[0]` with a non-null `score` for the declared acceptance dataset.
- `build/bin/transcribe-cli` is built.
- Acceptance manifest is available (see Step 1).

## Workflow

```
WER progress:
- [ ] Step 1: Ensure acceptance manifest
- [ ] Step 2: Score the reference-dtype model
- [ ] Step 3: Enforce the ref-dtype hard gate
- [ ] Step 4: Score each shipped quant
- [ ] Step 5: Write the summary table
- [ ] Step 6: Sign-off review
```

### Step 1: Acceptance manifest (execute or ask-point)

**Read the acceptance target from intake first.** The skill does not decide which dataset to score against — the intake does, via `upstream_benchmarks[0].{dataset, metric, score, score_unit}`. Normalize to a slug: lowercase, spaces → hyphens. `"LibriSpeech test-clean"` → `librispeech-test-clean`. Use that `<dataset>` slug in every subsequent path.

Resolve the path to a shell variable `MANIFEST` — every later step uses `$MANIFEST`, never a reconstructed path.

**Case A: LibriSpeech test-clean.** Canonical is `samples/wer/librispeech-test-clean.manifest.jsonl`; accept `samples/wer/test-clean.manifest.jsonl` as a historical fallback (ingest.py's default output):

```bash
if   [ -f samples/wer/librispeech-test-clean.manifest.jsonl ]; then
    MANIFEST=samples/wer/librispeech-test-clean.manifest.jsonl
elif [ -f samples/wer/test-clean.manifest.jsonl ]; then
    MANIFEST=samples/wer/test-clean.manifest.jsonl
else
    uv run scripts/wer/ingest.py --manifest samples/wer/librispeech-test-clean.manifest.jsonl
    MANIFEST=samples/wer/librispeech-test-clean.manifest.jsonl
fi
```

`ingest.py` needs the extracted LibriSpeech test-clean split at `samples/wer/raw/LibriSpeech/test-clean/`. Ask the user to place the archive there first; download is out of scope.

**Case B: Any other dataset.** The skill does NOT assume LibriSpeech. Set `MANIFEST=samples/wer/<dataset>.manifest.jsonl` from the slugified intake name. If absent, ask the user to provide one or authorize an ingest adapter mirroring `scripts/wer/ingest.py`. Manifest shape: `{id, audio, ref_text}` per JSONL line (matches `scripts/wer/run.py`'s `entry["audio"]` read). Do not silently fall back to LibriSpeech.

Also confirm `upstream_benchmarks[0].metric == "wer"`. If the intake declared `cer`, `bleu`, or another metric, this skill does not apply — use the metric-specific scoring flow instead (not shipped yet).

### Step 2: Score the reference-dtype model (execute)

Read intake for the reference dtype and acceptance dataset:

```bash
REFDTYPE=$(uv run python -c "import json; d=json.load(open('reports/porting/<family>/<variant>/intake.json')); \
  m={'float32':'F32','float16':'F16','bfloat16':'BF16'}; print(m[d['dtype']['expected']])")
DATASET=$(uv run python -c "import json; d=json.load(open('reports/porting/<family>/<variant>/intake.json'))['upstream_benchmarks'][0]['dataset']; \
  print(d.replace(' ', '-').lower())")

# $MANIFEST was resolved in Step 1 — use it directly, never the ${DATASET} slug.
uv run scripts/wer/run.py \
  --model models/<variant>/<variant>-${REFDTYPE}.gguf \
  --manifest "$MANIFEST" \
  --out reports/wer/<variant>-${REFDTYPE}.${DATASET}.jsonl

uv run scripts/wer/score.py reports/wer/<variant>-${REFDTYPE}.${DATASET}.jsonl
```

`score.py` writes `reports/wer/<variant>-<REFDTYPE>.<DATASET>.score.json` alongside the JSONL with `wer`, `wer_pct`, `wer_ci_lo`, `wer_ci_hi`, `n`, and per-utterance detail.

### Step 3: Ref-dtype hard gate (execute)

```python
# uv run python -c '...'
import json, sys
family = "<family>"; variant = "<variant>"; refdtype = "<REFDTYPE>"; dataset = "<DATASET>"
intake = json.load(open(f"reports/porting/{family}/{variant}/intake.json"))
target = intake["upstream_benchmarks"][0]["score"]
# Upstream scores may be ratio (0.0169) or percent (1.69); normalize.
unit = intake["upstream_benchmarks"][0].get("score_unit", "ratio")
target_ratio = target / 100.0 if unit == "percent" else target
observed = json.load(open(f"reports/wer/{variant}-{refdtype}.{dataset}.score.json"))["wer"]
delta = observed - target_ratio
print(f"upstream={target_ratio:.4f} observed={observed:.4f} delta={delta:+.4f}")
ok = abs(delta) <= 0.01
sys.exit(0 if ok else 1)
```

**If the gate fails (|delta| > 0.01)**, halt. Do NOT score the quants, do NOT write the summary table, do NOT loosen anything. The failure almost always means a numerical issue in Stage 4 that the tolerance gates did not catch. Options to surface to the user:

1. Return to `porting-4-cpp` and investigate. Widening `dump_coverage.json` (the contract tensor set) is a common fix — the drift hid in a tensor the tolerances did not cover.
2. If the user is certain the upstream number is wrong (rare), this requires a maintainer decision to override and is captured as a note in the family doc.

**If the gate passes**, proceed.

### Step 4: Score each shipped quant (execute)

Loop over `F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M`, skipping whichever equals `REFDTYPE`:

```bash
for PRESET in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  [ "$PRESET" = "$REFDTYPE" ] && continue
  uv run scripts/wer/run.py \
    --model models/<variant>/<variant>-${PRESET}.gguf \
    --manifest "$MANIFEST" \
    --out reports/wer/<variant>-${PRESET}.${DATASET}.jsonl
  uv run scripts/wer/score.py reports/wer/<variant>-${PRESET}.${DATASET}.jsonl
done
```

Quant WER is reported, not gated in v1. A quant that regresses noticeably vs. the reference is surfaced in the summary but does not block ship — the user decides per-quant whether to ship.

### Step 5: Summary table (execute)

Write a markdown table at `reports/wer/<variant>.<dataset>.summary.md` with columns `| Preset | WER | 95% CI | n | Δ vs upstream |` and one row per scored preset. `porting-7-ship` consumes this into the model card.

### Step 6: Sign-off

Report:
- Ref-dtype gate status (pass/fail, observed, target, delta).
- Path to every produced `.score.json`.
- Path to the summary markdown.
- Any quant that regressed beyond ~0.5% absolute vs. the reference (flag for user attention).

**Do not commit.** WER outputs under `reports/wer/` are **local generated artifacts**, ignored by `.gitignore`. They exist so porting-7-ship can consume them into the docs/models/ table that gets committed. The reports themselves stay local; the summary tables and per-quant WER cells are what ships in-repo.

## Postconditions

- `reports/wer/<variant>-<PRESET>.<dataset>.score.json` for every shipped preset.
- `reports/wer/<variant>.<dataset>.summary.md` table.
- Ref-dtype hard gate status is known and reported.

## Pointers (read, not execute)

- `docs/porting/families/<family>.md` — acceptance dataset details
- `scripts/wer/ingest.py` — LibriSpeech-style manifest builder (shape reference for alternate datasets)
- `scripts/wer/run.py` — transcribe-cli driver for batch transcription
- `scripts/wer/score.py` — jiwer + bootstrap CI
- Existing summaries under `reports/wer/` — format reference
