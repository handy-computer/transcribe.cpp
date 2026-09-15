---
name: porting-7-wer
description: Full release WER sweep for a ported variant. Scores the full acceptance manifest across the reference dtype and every shipped quant. Ref-dtype C++ is hard-gated against the measured Oracle reference baseline; quant WERs are human-reviewed, not auto-gated. Use after porting-6-bench. Output: reports/wer/<variant>-<preset>.<dataset>.score.json per shipped preset, reports/wer/<variant>.<dataset>.summary.md, and a ship-gate decision.
---

# porting-7-wer

Stage 7 of the porting pipeline. Scores the full acceptance manifest for
the ref dtype and every shipped quant. Step 3 checks the hard gate against
the measured Oracle reference baseline from Stage 2.

Quant WER is **reviewed and signed off** by the user, not auto-gated.

Stage 4 already gated ref dtype, and Stage 5 took a preliminary
512-utterance quant read. Stage 7 runs the authoritative full split after
bench and records human review for every quant.

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
- [ ] Step 2: Run the publication profile (every profile cell, every quant)
- [ ] Step 3: Check the ref-dtype WER limit
- [ ] Step 4: Score acceptance cells the profile does not cover
- [ ] Step 5: Write the summary table
- [ ] Step 6: Ingest into the catalog and render
- [ ] Step 7: Sign-off review
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

### Step 2: Run the publication profile (execute)

One sweep produces both the release numbers and the gate inputs. The
checked-in profile (`catalog/_benchmark_profiles.json`) selects the cells:
LibriSpeech test-clean at every downloaded quant for English-capable models,
and FLEURS test at Q8_0 for every supported FLEURS language. Run whatever is
missing:

```bash
modal run scripts/wer/remote/modal_sweep.py::publication_sweep \
  --models <variant>                 # --plan-only to inspect the expansion
for f in reports/wer/<variant>-*.jsonl; do uv run scripts/wer/score.py "$f"; done
```

The sweep writes `reports/wer/<variant>-<PRESET>.<dataset>[.bN].jsonl` and
`score.py` writes the matching `.score.json`. Every JSONL carries its decode
recipe, engine sha, and profile id in the batch header, and the score carries
them forward, which is what makes it ingestible in Step 6.

Do not also run the same cells locally with `run.py`: a second measurement of
one cell under a different backend is a second number to reconcile, not a
check.

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

### Step 4: Acceptance cells outside the profile (execute)

The profile covers the acceptance dataset for most ports. Two cases fall
outside it and are scored locally, exactly as the profile would, so the files
land under the same names:

- The acceptance dataset is FLEURS (a single-language port such as
  `parakeet-primeline` or `gigaam`): the profile ran Q8_0 only. Score the
  reference dtype and the remaining quants on `$MANIFEST`.
- The acceptance dataset is not a profile dataset at all (AMI for a
  diarizer): score every preset on `$MANIFEST`.

```bash
for PRESET in <REFDTYPE> F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  [ -f reports/wer/<variant>-${PRESET}.${DATASET}.score.json ] && continue
  uv run scripts/wer/run.py \
    --model models/<variant>/<variant>-${PRESET}.gguf \
    --manifest "$MANIFEST" \
    --out reports/wer/<variant>-${PRESET}.${DATASET}.jsonl
  uv run scripts/wer/score.py reports/wer/<variant>-${PRESET}.${DATASET}.jsonl
done
```

Quant WER is reviewed and signed off by the user, not auto-gated. Batch mode
should be WER-neutral; if a `--batch-size > 1` sweep differs from serial
beyond dataset noise (~0.01), stop and report the numbers.

### Step 5: Summary table (execute)

Write a markdown table at `reports/wer/<variant>.<dataset>.summary.md`
with columns `| Preset | Error rate | Reference error rate | 95% CI | n | Review |`
and one row per scored preset. The ref-dtype row records the automatic
gate result (`PASS` or `BLOCKED`). Quant rows record human disposition
(`ACCEPTED`, `REJECTED`, or `PENDING REVIEW`) plus any short note the
user gives. Stage 8 (`porting-8-ship`) consumes this into the model card.

### Step 6: Ingest into the catalog and render (execute)

Only profile-stamped scores are ingested; a score with no engine sha or a
different recipe is rejected by name, and the rejection is the finding.

```bash
uv run scripts/catalog/ingest_accuracy.py --models <variant>
```

Set `headline_benchmark` in `catalog/<variant>.json` to the cell the
download table features. A variant can carry several runs of one dataset
differing only in batch size or timestamp mode, so the pointer names the
whole identity:

```json
"headline_benchmark": {"dataset": "librispeech", "split": "test-clean",
                       "language": "en", "metric": "wer",
                       "batch_size": 1, "timestamps": "none"}
```

```bash
uv run scripts/catalog/check.py --publication-profile --models <variant>
uv run scripts/catalog/render.py
```

Never hand-edit a WER into a doc or an HF card spec: both are rendered from
the catalog, and CI fails when they drift.

### Step 7: Sign-off

Report:
- Manifest path and utterance count.
- Ref-dtype status: measured Oracle reference WER, C++ WER, max allowed
  WER, pass/blocked, and any required justification.
- Path to every produced `.score.json`, and which came from the profile
  sweep versus Step 4.
- Path to the summary markdown.
- The catalog check result for the variant.
- Human disposition for every shipped quant. Quant WER has no automatic
  numeric gate; unresolved quant review means Stage 7 sign-off is pending.

**Do not commit.** WER outputs under `reports/wer/` are local generated
artifacts, ignored by `.gitignore`. The catalog record and the rendered
tables are what ships in-repo.

## Postconditions

- `reports/wer/<variant>-<PRESET>.<dataset>.score.json` for every
  shipped preset, profile-stamped where the profile covers the cell.
- `reports/wer/<variant>.<dataset>.summary.md` table.
- Ref-dtype status is known and reported as plain WER numbers against
  the measured Oracle reference baseline.
- `catalog/<variant>.json` holds every profile accuracy cell and a
  `headline_benchmark`; the per-variant profile check passes.
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
