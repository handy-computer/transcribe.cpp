---
name: porting-7-wer
description: Full release WER sweep for a ported variant. Scores the full acceptance manifest across the reference dtype and every shipped quant. Ref-dtype C++ is hard-gated against the measured Oracle reference baseline; quant WERs are human-reviewed, not auto-gated. Use after porting-6-bench. Output: reports/wer/<variant>-<preset>.<dataset>.score.json per shipped preset, reports/wer/<variant>.<dataset>.summary.md, and a ship-gate decision.
---

# porting-7-wer

Stage 7 of the porting pipeline. Scores the full acceptance manifest for
the ref dtype and every shipped quant. Step 3 checks the hard gate against
the measured Oracle reference baseline from Stage 2.

Quant WER is **reviewed and signed off** by the user, not auto-gated.

Stage 4 already gated ref dtype, and Stage 5 took a tentative quant read.
Stage 7 re-confirms after bench and records human review for every quant.

## Preconditions

- `models/<variant>/<variant>-<PRESET>.gguf` exists for every shipped
  preset.
- `reports/porting/<family>/<variant>/intake.json` has
  `upstream_benchmarks[0]` with a declared acceptance dataset and metric.
- `reports/wer/<variant>-REF.<dataset>.score.json` exists from
  `porting-2-oracle` Step 7. This measured reference score is the
  ref-dtype gate target.
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

Read the acceptance dataset and metric from `upstream_benchmarks[0].{dataset, metric}`. Slugify: lowercase, spaces → hyphens. `"LibriSpeech test-clean"` → `librispeech-test-clean`. Resolve to `$MANIFEST`; every later step uses `$MANIFEST`, never a reconstructed path. Confirm `metric` is `wer` or `cer`; anything else is out of scope. Do not use any publisher-reported score for pass/fail; the measured Oracle reference score is the gate target.

If the intake's dataset is not covered by `scripts/wer/ingest.py`, extend
that script before running this step.

**LibriSpeech.** Output is `samples/wer/librispeech-<split>.manifest.jsonl`; accept the legacy unprefixed `test-clean.manifest.jsonl` as a fallback:

```bash
if   [ -f samples/wer/librispeech-test-clean.manifest.jsonl ]; then
    MANIFEST=samples/wer/librispeech-test-clean.manifest.jsonl
elif [ -f samples/wer/test-clean.manifest.jsonl ]; then
    MANIFEST=samples/wer/test-clean.manifest.jsonl
else
    uv run scripts/wer/ingest.py librispeech
    MANIFEST=samples/wer/librispeech-test-clean.manifest.jsonl
fi
```

**FLEURS.** BCP-47 short code maps to the FLEURS config inside `ingest.py`:

```bash
LANG=vi
uv run scripts/wer/ingest.py fleurs --lang "$LANG"
MANIFEST=samples/wer/fleurs-${LANG}.manifest.jsonl
```

CER auto-routes for zh / yue / ja / ko / th via the manifest's `language` field. The score JSON's `error_rate_pct` is the canonical report metric.

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

`score.py` writes the `.score.json` consumed by the gate and summary.

### Step 3: Ref-dtype WER limit (execute)

Use the score JSON's canonical `error_rate` ratio, and print it as the
percent number humans read in reports.

Simple rule:

- Measured Oracle reference WER: `3.59`
- Max allowed C++ WER: `3.60`
- `3.60` passes
- `3.61` is too high and must be investigated

```python
# uv run python -c '...'
import json, sys
from decimal import Decimal, ROUND_HALF_UP

def rate_number(x):
    return Decimal(str(x * 100.0)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

family = "<family>"; variant = "<variant>"; refdtype = "<REFDTYPE>"; dataset = "<DATASET>"
reference = json.load(open(f"reports/wer/{variant}-REF.{dataset}.score.json"))["error_rate"]
observed = json.load(open(f"reports/wer/{variant}-{refdtype}.{dataset}.score.json"))["error_rate"]
reference_wer = rate_number(reference)
cpp_wer = rate_number(observed)
max_allowed_wer = reference_wer + Decimal("0.01")
over_by = cpp_wer - max_allowed_wer
print(f"reference={reference_wer} cpp={cpp_wer} max_allowed={max_allowed_wer} over_by={over_by:+}")
ok = cpp_wer <= max_allowed_wer
sys.exit(0 if ok else 1)
```

Round WER to two decimals before comparing. The number printed in the
report is the number used for the pass/fail decision.

If the gate exits nonzero, stop release sign-off. Do not loosen the limit
or accept the result without a written reason backed by extra testing. Use
the captured Oracle JSONL at `reports/wer/<variant>-REF.<dataset>.jsonl`
for one-for-one per-utterance diffing.

Required investigation when blocked:

1. Diff C++ against the captured reference per-utterance (`hyp_text` by
   `id`) from the Oracle JSONL; only re-run the reference framework if
   that baseline is missing or stale.
2. Compare WER and the worst transcript differences.
3. If C++ and reference differ, dump tensors around the likely
   bad layer/component and compare them.
4. Add missing tensors to `dump_coverage.json` if coverage missed the
   problem.
5. If C++ matches reference, document the evidence in the WER summary
   or family doc before proceeding.

Proceed only when the higher WER is explained, reviewed, and written in
the WER summary or family doc. Higher WER without evidence is a release
blocker.

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

Quant WER is reviewed and signed off by the user, not auto-gated.

Batch mode should be WER-neutral. If a `--batch-size > 1` sweep differs
from serial beyond dataset noise (~0.01), stop and report the numbers.

### Step 5: Summary table (execute)

Write a markdown table at `reports/wer/<variant>.<dataset>.summary.md`
with columns `| Preset | Error rate | Reference error rate | 95% CI | n | Review |`
and one row per scored preset. The ref-dtype row records the automatic
gate result (`PASS` or `BLOCKED`). Quant rows record human disposition
(`ACCEPTED`, `REJECTED`, or `PENDING REVIEW`) plus any short note the
user gives. Stage 8 (`porting-8-ship`) consumes this into the model card.

### Step 6: Sign-off

Report:
- Manifest path and utterance count.
- Ref-dtype status: measured Oracle reference WER, C++ WER, max allowed
  WER, pass/blocked, and any required justification.
- Path to every produced `.score.json`.
- Path to the summary markdown.
- Human disposition for every shipped quant. Quant WER has no automatic
  numeric gate; unresolved quant review means Stage 7 sign-off is pending.

**Do not commit.** WER outputs under `reports/wer/` are local generated
artifacts, ignored by `.gitignore`. The summary tables and per-quant
WER cells are what ships in-repo via Stage 8.

## Postconditions

- `reports/wer/<variant>-<PRESET>.<dataset>.score.json` for every
  shipped preset.
- `reports/wer/<variant>.<dataset>.summary.md` table.
- Ref-dtype status is known and reported as plain WER numbers against
  the measured Oracle reference baseline.
- Sign-off names the manifest path and utterance count so consumers can
  verify which dataset was scored.
- Quant WER is reviewed and signed off by the user, not auto-gated.

## Pointers (read, not execute)

- `docs/porting/families/<family>.md` — acceptance dataset details
- `scripts/wer/ingest.py` — LibriSpeech-style manifest builder (shape
  reference for alternate datasets)
- `scripts/wer/run.py` — transcribe-cli driver; runs `--batch` mode and
  accepts `--batch-size` (WER-neutral; correctness gated in Stage 4)
- `scripts/wer/score.py` — jiwer + bootstrap CI
- `scripts/wer/remote/modal_sweep.py` — GPU sweep (also used by Stages 4/5)
- Existing summaries under `reports/wer/` — format reference
