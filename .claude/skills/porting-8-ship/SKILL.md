---
name: porting-8-ship
description: Produces the user-facing and upload-ready documentation for a completed port. Checklist-driven — the skill asserts every preceding artifact exists in the expected location before drafting the family doc, model card, HF card YAML, and HF README. Use after porting-7-wer has produced the ref-dtype gate pass and the per-quant WER table. Output: docs/porting/families/<family>.md (filled), docs/models/<variant>.md, scripts/hf_cards/<variant>.yaml, and models/<variant>/README.md. Human review before HF upload is an ask-point. The actual upload is human-run.
---

# porting-8-ship

Stage 8 (final) of the porting pipeline. Checklist-driven local prep:
verify every preceding artifact exists, then draft the family doc, the
user-facing model card, the HF card YAML, and the HF README. Shipping
itself stays in the user's hands.

## Preconditions

- All earlier stages 1–7 complete.
- `reports/porting/<family>/<variant>/intake.json` complete.
- `reports/porting/<family>/forward-map.md` complete.
- `tests/golden/<family>/<variant>.manifest.json` complete.
- `tests/tolerances/<family>.json` reviewed and committed (no
  `_provisional` flags).
- `reports/convert/<variant>-<REFDTYPE>.json` (SHA of the reference GGUF).
- `reports/wer/<variant>-<PRESET>.<dataset>.score.json` for every shipped
  preset.
- `reports/perf/<machine>/*_<variant>_<backend>.json` for at least one
  reference machine.

## Workflow

```
Ship progress:
- [ ] Step 1: Pre-flight checklist (artifacts present)
- [ ] Step 2: Fill the family doc
- [ ] Step 3: Author the user-facing model card
- [ ] Step 4: Write the HF card YAML spec
- [ ] Step 5: Render the HF README
- [ ] Step 6: Pre-upload review
- [ ] Step 7: Sign-off review
```

### Step 1: Pre-flight checklist (execute)

Confirm every artifact exists. If any row is missing, halt and send the
user to the stage that owns the missing artifact — Stage 8 does not
fabricate inputs.

| Artifact | Expected path | Owning stage |
|---|---|---|
| Intake | `reports/porting/<family>/<variant>/intake.json` | Stage 1 |
| Manifest | `tests/golden/<family>/<variant>.manifest.json` | Stage 2 |
| Tolerances | `tests/tolerances/<family>.json` | Stage 4 |
| Forward map | `reports/porting/<family>/forward-map.md` | Stage 4 |
| Converter report | `reports/convert/<variant>-<REFDTYPE>.json` | Stage 3 |
| Quants | `models/<variant>/<variant>-*.gguf` | Stage 5 |
| Bench reports | `reports/perf/<machine>/*_<variant>_<backend>.json` | Stage 6 |
| WER score JSONs | `reports/wer/<variant>-*.<dataset>.score.json` | Stage 7 |
| WER summary | `reports/wer/<variant>.<dataset>.summary.md` | Stage 7 |

```bash
# Mechanical checklist runner
for path in \
  reports/porting/<family>/<variant>/intake.json \
  tests/golden/<family>/<variant>.manifest.json \
  tests/tolerances/<family>.json \
  reports/porting/<family>/forward-map.md \
  reports/convert/<variant>-<REFDTYPE>.json \
  reports/wer/<variant>.<dataset>.summary.md
do
  [ -f "$path" ] && echo "OK $path" || echo "MISSING $path"
done
ls models/<variant>/<variant>-*.gguf >/dev/null 2>&1 \
  && echo "OK quants" || echo "MISSING quants"
ls reports/perf/*/*<variant>*.json >/dev/null 2>&1 \
  && echo "OK bench" || echo "MISSING bench"
ls reports/wer/<variant>-*.<dataset>.score.json >/dev/null 2>&1 \
  && echo "OK wer-scores" || echo "MISSING wer-scores"
```

Any `MISSING` halts Stage 8.

### Step 2: Family doc (execute + ask-point)

Open `docs/porting/families/<family>.md`. If still the `_template.md`
shape, fill it section by section by pulling facts from the artifacts:

- **Identity** — from `intake.json`: `family`, `hf_repo`, `hf_revision`,
  `variants[]`, license from the HF model card.
- **References** — from `intake.reference_framework`,
  `intake.reference_rationale`, and `manifest.reference.entrypoint`.
- **Environment** — from `scripts/envs/<family>/pyproject.toml`.
- **Artifacts** — paths to manifest, tolerances, forward-map, converter
  report, validation-report bundle, bench reports, WER reports.
- **Commands** — concrete `uv run` invocations for reference dumps,
  conversion, validation, bench, WER. Match the existing shape of
  `docs/porting/families/parakeet.md` or `cohere.md`.
- **Notes** — anything the port surfaced that didn't fit elsewhere:
  tensor-name mapping decisions, reference-framework quirks, known drift
  sources paraphrased from the tolerances `_comment` block.

For a new family, draft the Known Limitations section from intake
capabilities (streaming flag, translation flag, language coverage,
timestamp granularity) plus any sharp edges the port surfaced. Do not
invent limitations the port didn't discover; do not omit limitations the
capabilities flags imply. Present the draft for human review in Step 6.

### Step 3: User-facing model card (execute)

Author `docs/models/<variant>.md`. The repo ships a Jinja template at
`docs/_templates/model-card.md.j2` and existing rendered cards (e.g.
`docs/models/parakeet-tdt-0.6b-v2.md`) are the shape reference.

Two acceptable approaches:

1. **Copy from the closest existing model card** and edit the fields by
   hand. Pull facts directly from artifacts — quants from
   `models/<variant>/`, WER from `reports/wer/<variant>-*.score.json`,
   bench from `reports/perf/<machine>/`, dataset / upstream score from
   `intake.upstream_benchmarks[0]`. Ask for `target_hf_repo` since it
   cannot be inferred. This is usually the right call.
2. **Render from the existing template** if the template already covers
   everything the variant needs and the variant has no rendered card
   yet. Build the context dict in a short ad-hoc `uv run python -c`
   and write the output. Do **not** extend the template with new
   context fields just for one family — handcraft those sections in
   the rendered markdown instead.

Either way, the result is a first draft. Subsequent regenerations must
respect human edits — do not blindly re-render after the user has
touched the file.

### Step 4: HF card YAML spec (execute)

Write `scripts/hf_cards/<variant>.yaml`, mirroring
`scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml`:

```yaml
hf_repo: <intake.hf_repo>
target_repo: <user-provided e.g. handy-computer/<variant>-gguf>
transcribe_docs_url: https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/<variant>.md

upstream_commit: <intake.hf_revision short sha>
pin_date: <today>

validation:
  reference: <intake.reference_framework>
  commit: <current repo HEAD short sha>
  date: <today>

license: <from upstream model card>
license_display: <human-facing form>
pipeline_tag: automatic-speech-recognition
languages: [<from intake.capabilities.languages>]
tags:
  - gguf
  - transcribe.cpp
  - asr
  - speech-to-text
  - <family>
  - <architecture-style>
```

### Step 5: Render the HF README (execute)

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/<variant>.yaml
```

Writes `models/<variant>/README.md` by default. This is what `hf upload`
will pick up alongside the GGUFs.

### Step 6: Pre-upload review (ask-point)

Drafts from Steps 2, 3, and 5 are now on disk. Present three paths for
human review:
- `docs/porting/families/<family>.md`
- `docs/models/<variant>.md`
- `models/<variant>/README.md`

Flag the sections most likely to over-promise so the user examines them
closely: the model card's `one_liner` and `capabilities_prose`, the
family doc's Known Limitations. Wait for explicit sign-off before Step 7.

### Step 7: Sign-off

Report:
- All four output paths (family doc, model card, HF YAML, HF README).
- Target HF repo.
- Pre-flight checklist outcome from Step 1.
- Next command for the user to run (actual upload is not the skill's
  job):
  ```bash
  hf upload <target_repo> models/<variant> .
  ```
- Remind the user to commit the docs/families/models/hf_cards changes
  manually before the upload.

**Do not commit. Do not upload.** The upload is a human action because
it publishes to a public registry.

## Postconditions

- Pre-flight checklist (Step 1) was green before any drafting.
- `docs/porting/families/<family>.md` filled and reviewed.
- `docs/models/<variant>.md` authored with a populated download / WER /
  bench table.
- `scripts/hf_cards/<variant>.yaml` committed-ready.
- `models/<variant>/README.md` rendered.

## Pointers (read, not execute)

- `docs/porting/families/_template.md` — family doc shape
- `docs/models/parakeet-tdt-0.6b-v2.md` — model card shape reference
- `scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml` — HF card YAML reference
- `scripts/hf_cards/template.md.j2` — Jinja template that generate.py
  renders
- `scripts/hf_cards/generate.py` — renderer (execute only via Step 5)
