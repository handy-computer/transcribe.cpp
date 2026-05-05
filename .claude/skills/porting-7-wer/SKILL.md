---
name: porting-7-wer
description: Full release WER sweep for a ported variant. Scores the full acceptance manifest declared in intake.upstream_benchmarks across the reference dtype and every shipped quant, enforces the ref-dtype hard gate of ±0.01 absolute vs. the upstream score, and surfaces quant WERs as documentation only (not gated). Use after porting-6-bench. Distinct from the Stage 4 subset WER sanity — Stage 7 is the user-facing quality artifact, not a parity check against the reference framework. Output: reports/wer/<variant>-<preset>.<dataset>.score.json per shipped preset, reports/wer/<variant>.<dataset>.summary.md, and a ship-gate decision.
---

# porting-7-wer

Stage 7 of the porting pipeline. The acceptance target is whatever the
intake captured in `upstream_benchmarks[0]` — LibriSpeech test-clean for
English models, FLEURS / Common Voice / custom dataset for others, as
the publisher reports. Produces WER scores with bootstrap CIs for every
shipped preset and enforces the ref-dtype hard gate. If the gate fails,
the skill does not loosen tolerances or waive the test — it stops and
sends the port back to Stage 4.

Quant WER is **reported, not gated**. The user decides per-quant whether
to ship based on the summary table. There is no automatic quant WER
threshold.

This is distinct from the Stage 4 subset WER sanity: Stage 4 compared
C++ against the reference framework on the first-512 manifest subset;
Stage 7 scores the full acceptance manifest against the publisher's
reported number for the user-facing model card.

## Preconditions

- `models/<variant>/<variant>-<PRESET>.gguf` exists for every shipped
  preset.
- `reports/porting/<family>/<variant>/intake.json` has
  `upstream_benchmarks[0]` with a non-null `score` for the declared
  acceptance dataset.
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

**Read the acceptance target from intake first.** The skill does not
decide which dataset to score against — the intake does, via
`upstream_benchmarks[0].{dataset, metric, score, score_unit}`. Normalize
to a slug: lowercase, spaces → hyphens. `"LibriSpeech test-clean"` →
`librispeech-test-clean`. Use that `<dataset>` slug in every subsequent
path.

Resolve the path to a shell variable `MANIFEST` — every later step uses
`$MANIFEST`, never a reconstructed path.

**Case A: LibriSpeech test-clean.** Canonical is
`samples/wer/librispeech-test-clean.manifest.jsonl`; accept
`samples/wer/test-clean.manifest.jsonl` as a historical fallback (ingest's
default output):

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

`ingest.py` needs the extracted LibriSpeech test-clean split at
`samples/wer/raw/LibriSpeech/test-clean/`. Ask the user to place the
archive there first; download is out of scope.

**Case B: Any other dataset.** The skill does NOT assume LibriSpeech.
Set `MANIFEST=samples/wer/<dataset>.manifest.jsonl` from the slugified
intake name. If absent, ask the user to provide one or authorize an
ingest adapter mirroring `scripts/wer/ingest.py`. Manifest shape:
`{id, audio, ref_text}` per JSONL line. Do not silently fall back to
LibriSpeech.

Also confirm `upstream_benchmarks[0].metric == "wer"`. If the intake
declared `cer`, `bleu`, or another metric, this skill does not apply.

### Step 2: Score the reference-dtype model (execute)

Read intake for the reference dtype and acceptance dataset:

```bash
REFDTYPE=$(uv run python -c "import json; d=json.load(open('reports/porting/<family>/<variant>/intake.json')); \
  m={'float32':'F32','float16':'F16','bfloat16':'BF16'}; print(m[d['dtype']['expected']])")
DATASET=$(uv run python -c "import json; d=json.load(open('reports/porting/<family>/<variant>/intake.json'))['upstream_benchmarks'][0]['dataset']; \
  print(d.replace(' ', '-').lower())")

# $MANIFEST was resolved in Step 1 — use it directly.
uv run scripts/wer/run.py \
  --model models/<variant>/<variant>-${REFDTYPE}.gguf \
  --manifest "$MANIFEST" \
  --out reports/wer/<variant>-${REFDTYPE}.${DATASET}.jsonl

uv run scripts/wer/score.py reports/wer/<variant>-${REFDTYPE}.${DATASET}.jsonl
```

`score.py` writes
`reports/wer/<variant>-<REFDTYPE>.<DATASET>.score.json` with `wer`,
`wer_pct`, `wer_ci_lo`, `wer_ci_hi`, `n`, and per-utterance detail.

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

**If the gate fails (|delta| > 0.01)**, halt. Do NOT score the quants,
do NOT write the summary table, do NOT loosen anything. The failure
almost always means a numerical issue in Stage 4 that the tolerance
gates did not catch. Options:

1. Return to `porting-4-cpp` and investigate. Widening
   `dump_coverage.json` (the contract tensor set) is a common fix — the
   drift hid in a tensor the tolerances did not cover.
2. If the user is certain the upstream number is wrong (rare), this
   requires a maintainer decision to override and is captured as a note
   in the family doc.

**If the gate passes**, proceed.

### Step 4: Score each shipped quant (execute)

Loop over `F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M`, skipping whichever equals
`REFDTYPE`:

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

Quant WER is reported, not gated. A quant that regresses noticeably vs.
the reference is surfaced in the summary but does not block ship — the
user decides per-quant whether to ship.

### Step 5: Summary table (execute)

Write a markdown table at `reports/wer/<variant>.<dataset>.summary.md`
with columns `| Preset | WER | 95% CI | n | Δ vs upstream |` and one row
per scored preset. Stage 8 (`porting-8-ship`) consumes this into the
model card.

### Step 6: Sign-off

Report:
- Manifest path and utterance count.
- Ref-dtype gate status (pass/fail, observed, target, delta).
- Path to every produced `.score.json`.
- Path to the summary markdown.
- Any quant that regressed beyond ~0.5% absolute vs. the reference (flag
  for user attention; not a gate).

**Do not commit.** WER outputs under `reports/wer/` are local generated
artifacts, ignored by `.gitignore`. The summary tables and per-quant
WER cells are what ships in-repo via Stage 8.

## Postconditions

- `reports/wer/<variant>-<PRESET>.<dataset>.score.json` for every
  shipped preset.
- `reports/wer/<variant>.<dataset>.summary.md` table.
- Ref-dtype hard gate status is known and reported.
- Sign-off names the manifest path and utterance count so consumers can
  verify which dataset was scored.
- Quant WER is documented but not gated; user decides per-quant.

## Pointers (read, not execute)

- `docs/porting/families/<family>.md` — acceptance dataset details
- `scripts/wer/ingest.py` — LibriSpeech-style manifest builder (shape
  reference for alternate datasets)
- `scripts/wer/run.py` — transcribe-cli driver for batch transcription
- `scripts/wer/score.py` — jiwer + bootstrap CI
- `scripts/wer/subset.py` — Stage 4 subset generator (different scope:
  Stage 7 uses the full manifest, not the 512-row subset)
- Existing summaries under `reports/wer/` — format reference
