---
name: porting-8-ship
description: Produces user-facing and upload-ready documentation for a completed port. Use after porting-7-wer has produced the ref-dtype gate pass and per-quant WER table. Output: filled family doc, model card, HF card YAML, rendered HF README, and private-repo docs upload.
---

# porting-8-ship

Stage 8 (final) verifies prior artifacts, drafts release docs, renders the
HF README, and pushes docs/README to the private HF repo. Public release
is out of scope.

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

State the **batch and streaming** posture from the Capability Validation
rows:
- Batch: whether the family ships an explicit `run_batch()` fast path
  (PASS) or runs the serial fallback (`ACCEPTED GAP — serial fallback`),
  and that batching is WER-neutral (byte-identical to single-stream).
- Streaming: if `capabilities.streaming` is true the row is PASS — say
  so and name the chunk/lookahead contract; it is never reported as an
  accepted gap for a natively-streaming model. If the model does not
  stream, omit the row.

### Step 3: User-facing model card (execute)

Author `docs/models/<variant>.md`. The repo ships a Jinja template at
`docs/_templates/model-card.md.j2` and existing rendered cards (e.g.
`docs/models/parakeet-tdt-0.6b-v2.md`) are the shape reference.

Two acceptable approaches:

1. **Copy from the closest existing model card** and edit by hand. Pull
   facts directly from artifacts — quants from
   `models/<variant>/`, WER and the measured reference baseline from
   `reports/wer/<variant>-*.score.json`, bench from
   `reports/perf/<machine>/`, and the acceptance dataset from
   `intake.upstream_benchmarks[0]`. Ask for `target_hf_repo` since it
   cannot be inferred.
2. **Render from the existing template** if the template already covers
   everything the variant needs and the variant has no rendered card
   yet. Build the context dict in a short ad-hoc `uv run python -c`
   and write the output. Do **not** extend the template with new
   context fields just for one family — handcraft those sections in
   the rendered markdown instead.

Subsequent regenerations must respect human edits.

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

Writes `models/<variant>/README.md` by default.

### Step 6: Pre-upload review (ask-point)

Drafts from Steps 2, 3, and 5 are now on disk. Present three paths for
human review:
- `docs/porting/families/<family>.md`
- `docs/models/<variant>.md`
- `models/<variant>/README.md`

Flag likely over-promising sections (`one_liner`, `capabilities_prose`,
Known Limitations) and wait for explicit sign-off before Step 7.

### Step 7: Sign-off

Report:
- All four output paths (family doc, model card, HF YAML, HF README).
- Target private HF repo.
- Pre-flight checklist outcome from Step 1.
- Push the rendered docs/README to the private repo:
  ```bash
  hf upload <target_repo> models/<variant> . --repo-type model
  ```
- Remind the user to commit the docs/families/models/hf_cards changes.
- If this port adds a new family (or new variants under an existing
  family), remind the user to update the supported-models table in the
  root `README.md` so the family/variants are listed.

**Do not commit.** Keep the repo private; flipping it public is a future
action, not part of this stage.

## Postconditions

- Pre-flight checklist (Step 1) was green before any drafting.
- `docs/porting/families/<family>.md` filled and reviewed.
- `docs/models/<variant>.md` authored with a populated download / WER /
  bench table.
- `scripts/hf_cards/<variant>.yaml` committed-ready.
- `models/<variant>/README.md` rendered.
- Docs/README pushed to the private HF repo; public flip deferred.

## Pointers (read, not execute)

- `docs/porting/families/_template.md` — family doc shape
- `docs/models/parakeet-tdt-0.6b-v2.md` — model card shape reference
- `scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml` — HF card YAML reference
- `scripts/hf_cards/template.md.j2` — Jinja template that generate.py
  renders
- `scripts/hf_cards/generate.py` — renderer (execute only via Step 5)
