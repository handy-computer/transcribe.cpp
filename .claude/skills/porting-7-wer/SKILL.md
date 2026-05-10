---
name: porting-7-wer
description: Full release WER sweep for a ported variant. Scores the full acceptance manifest declared in intake.upstream_benchmarks across the reference dtype and every shipped quant. Simple rule: if upstream WER is 3.59, the ref-dtype C++ WER must be 3.60 or lower. Anything higher blocks release until explained with extra testing. Quant WERs are documentation only, not gated. Use after porting-6-bench. Output: reports/wer/<variant>-<preset>.<dataset>.score.json per shipped preset, reports/wer/<variant>.<dataset>.summary.md, and a ship-gate decision.
---

# porting-7-wer

Stage 7 of the porting pipeline. The acceptance target is whatever the
intake captured in `upstream_benchmarks[0]` — LibriSpeech test-clean for
English models, FLEURS / Common Voice / custom dataset for others, as
the publisher reports. Produces WER scores with bootstrap CIs for every
shipped preset and applies one simple rule:

**C++ WER must be no more than upstream WER + 0.01 WER points.**

Example: if upstream WER is `3.59`, the highest allowed C++ WER is
`3.60`. `3.60` passes. `3.61` does not pass.

If C++ is higher than the allowed number, the skill does not loosen
tolerances or waive the test. It stops release sign-off until extra
testing explains the gap.

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
- [ ] Step 3: Check the ref-dtype WER limit
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

### Step 3: Ref-dtype WER limit (execute)

Use WER as the percent number humans read in reports.

Simple rule:

- Upstream WER: `3.59`
- Max allowed C++ WER: `3.60`
- `3.60` passes
- `3.61` is too high and must be investigated

```python
# uv run python -c '...'
import json, sys
from decimal import Decimal, ROUND_HALF_UP

def wer_number(x):
    return Decimal(str(x * 100.0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

family = "<family>"; variant = "<variant>"; refdtype = "<REFDTYPE>"; dataset = "<DATASET>"
intake = json.load(open(f"reports/porting/{family}/{variant}/intake.json"))
target = intake["upstream_benchmarks"][0]["score"]
# Upstream scores may be ratio (0.0169) or percent (1.69); normalize.
unit = intake["upstream_benchmarks"][0].get("score_unit", "ratio")
target_ratio = target / 100.0 if unit == "percent" else target
observed = json.load(open(f"reports/wer/{variant}-{refdtype}.{dataset}.score.json"))["wer"]
upstream_wer = wer_number(target_ratio)
cpp_wer = wer_number(observed)
max_allowed_wer = upstream_wer + Decimal("0.01")
over_by = cpp_wer - max_allowed_wer
print(f"upstream={upstream_wer} cpp={cpp_wer} max_allowed={max_allowed_wer} over_by={over_by:+}")
ok = cpp_wer <= max_allowed_wer
sys.exit(0 if ok else 1)
```

Round WER to two decimals before comparing. The number printed in the
report is the number used for the pass/fail decision.

**If C++ WER is higher than `upstream WER + 0.01`**, stop release
sign-off. Do NOT loosen anything. Do NOT accept it without a written
reason backed by extra testing.

The job is simple: prove whether the C++ implementation is wrong.

Required investigation:

1. Run the same audio list through C++ and the reference framework.
2. Compare WER and the worst transcript differences.
3. If C++ and the reference do not match, dump tensors around the likely
   bad layer/component and compare them.
4. Add missing tensors to `dump_coverage.json` if the current tensor
   checks missed the problem.
5. If C++ matches the reference but both are worse than upstream,
   document the dataset version, text normalization, decoding settings,
   and why the upstream number is not directly comparable.

Proceed only when the higher WER is explained, reviewed, and written in
the WER summary or family doc. Higher WER without evidence is a release
blocker.

**If C++ WER is at or below `upstream WER + 0.01`**, proceed.

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

Quant WER is reported, not gated. Still flag any quant that is more than
`0.01` worse than the ref-dtype model. Simple rule: if ref-dtype WER is
`3.59`, then quant WER `3.60` is okay and quant WER `3.61` must be
called out for review. The user decides per-quant whether to ship.

### Step 5: Summary table (execute)

Write a markdown table at `reports/wer/<variant>.<dataset>.summary.md`
with columns `| Preset | WER | Max allowed WER | 95% CI | n | Over allowed |` and one row
per scored preset. Stage 8 (`porting-8-ship`) consumes this into the
model card.

### Step 6: Sign-off

Report:
- Manifest path and utterance count.
- Ref-dtype status: upstream WER, C++ WER, max allowed WER, pass/blocked,
  and any required justification.
- Path to every produced `.score.json`.
- Path to the summary markdown.
- Any quant that is more than `0.01` worse than the ref-dtype WER. Example:
  if ref-dtype WER is `3.59`, then quant WER `3.61` must be flagged.

**Do not commit.** WER outputs under `reports/wer/` are local generated
artifacts, ignored by `.gitignore`. The summary tables and per-quant
WER cells are what ships in-repo via Stage 8.

## Postconditions

- `reports/wer/<variant>-<PRESET>.<dataset>.score.json` for every
  shipped preset.
- `reports/wer/<variant>.<dataset>.summary.md` table.
- Ref-dtype status is known and reported as plain WER numbers.
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
