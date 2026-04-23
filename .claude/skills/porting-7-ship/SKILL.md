---
name: porting-7-ship
description: Produces the user-facing and upload-ready documentation for a completed port — the family doc under docs/porting/families/, the user-facing model card under docs/models/, and the HuggingFace README rendered via scripts/hf_cards/generate.py. Use after porting-6-wer has produced the ref-dtype gate pass and the per-quant WER table. Input: intake, manifest, tolerances, bench reports, WER reports. Output: docs/porting/families/<family>.md (filled), docs/models/<variant>.md, scripts/hf_cards/<variant>.yaml, and models/<variant>/README.md. Human review before HF upload is an ask-point. The actual upload is human-run.
---

# porting-7-ship

Stage 7 (final) of the porting pipeline. Fills the family doc, authors the user-facing model card with auto-populated quant/WER/bench tables, writes the HF card yaml spec, renders the HF README, and prepares upload. Shipping itself stays in the user's hands.

## Preconditions

- `reports/porting/<family>/<variant>/intake.json` complete.
- `tests/golden/<family>/<variant>.manifest.json` complete.
- `tests/tolerances/<family>.json` reviewed and committed.
- `reports/wer/<variant>-<PRESET>.<dataset>.score.json` for every shipped preset.
- `reports/perf/<machine>/*_<variant>_<backend>.json` for at least one reference machine.
- `reports/convert/<variant>-<REFDTYPE>.json` (SHA of the reference GGUF).

## Workflow

```
Ship progress:
- [ ] Step 1: Fill the family doc (docs/porting/families/<family>.md)
- [ ] Step 2: Author the user-facing model card (docs/models/<variant>.md)
- [ ] Step 3: Write the HF card yaml spec
- [ ] Step 4: Render the HF README
- [ ] Step 5: Pre-upload review
- [ ] Step 6: Sign-off review
```

### Step 1: Family doc (execute + ask-point)

Open `docs/porting/families/<family>.md`. If it's still the `_template.md` shape, fill it section by section by pulling facts from the artifacts:

- **Identity** — from `intake.json`: `family`, `hf_repo`, `hf_revision`, `variants[]`, license from the HF model card.
- **References** — from `intake.json.reference_framework`, `intake.json.reference_rationale`, and `tests/golden/<family>/<variant>.manifest.json.reference.entrypoint`.
- **Environment** — from `scripts/envs/<family>/pyproject.toml`.
- **Artifacts** — paths to manifest, tolerances, converter report, validation-report bundle, bench reports, WER reports.
- **Commands** — the concrete `uv run` invocations for reference dumps, conversion, validation, bench, WER. Match the existing shape of `docs/porting/families/parakeet.md` or `cohere.md`.
- **Notes** — anything the port surfaced that didn't fit elsewhere: tensor-name mapping decisions, reference-framework quirks, known drift sources paraphrased from the tolerances `_comment` block.

For a new family, the skill drafts the Known Limitations section from intake capabilities (streaming flag, translation flag, language coverage, timestamp granularity) plus any sharp edges the port surfaced (e.g. tensor-name mapping quirks, reference-framework constraints). Do not invent limitations the port didn't discover; do not omit limitations the capabilities flags imply. Present the draft for human review in Step 5.

### Step 2: User-facing model card (execute)

Render `docs/models/<variant>.md` from `docs/_templates/model-card.md.j2` using Jinja2. **Draft every field from artifacts — the skill does not stop and ask the user to write prose.** Human review of the rendered file happens in Step 5.

Build the template context:

- `display_name` — derived from `intake.family` + variant (e.g. "Parakeet TDT 0.6B v2").
- `one_liner` — drafted from intake: architecture pattern + parameter count + positioning against siblings. Example for an encoder-transducer: "A 0.6B-parameter Conformer encoder with a TDT/RNNT transducer decoder."
- `capabilities_prose` — drafted from `intake.capabilities`: language list, streaming flag, translation flag, timestamp granularity. Include honest scope limits (e.g. "Offline English speech-to-text only. Not a streaming model and does not translate.").
- `hf_repo`, `upstream_commit_short`, `upstream_commit_url`, `pin_date`, `license_display` — from intake + upstream model card.
- `target_hf_repo` — **ask-point**: the user's chosen HF repo (e.g. `handy-computer/<variant>-gguf`). The skill cannot infer this.
- `presets` — enumerate `models/<variant>/<variant>-<PRESET>.gguf`; `stat` for size; read WER from `reports/wer/<variant>-<PRESET>.<dataset>.score.json`.
- `dataset_display`, `dataset_n`, `upstream_wer_pct` — from intake's `upstream_benchmarks[0]` + the score JSON.
- `validation_*`, `drift_source_prose` — paraphrased from the finalized tolerances `_comment` block and the latest validate run.
- `perf_machines` — from `reports/perf/<machine>/*_<variant>_<backend>.json`. Omit cleanly if bench reports are absent; do not block.

```bash
uv run --with jinja2 python -c "
from jinja2 import Environment, FileSystemLoader, StrictUndefined
env = Environment(loader=FileSystemLoader('docs/_templates'), undefined=StrictUndefined)
ctx = { ... }  # build from artifacts per the list above
open('docs/models/<variant>.md', 'w').write(env.get_template('model-card.md.j2').render(**ctx))
"
```

The rendered file is a **first draft**. Subsequent regenerations must respect human edits — do not blindly re-render after the user has touched it.

### Step 3: HF card yaml spec (execute)

Write `scripts/hf_cards/<variant>.yaml`, mirroring `scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml`:

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

### Step 4: Render the HF README (execute)

```bash
uv run scripts/hf_cards/generate.py scripts/hf_cards/<variant>.yaml
```

Writes `models/<variant>/README.md` by default. This is what `hf upload` will pick up alongside the GGUFs.

### Step 5: Pre-upload review (ask-point)

The drafts from Steps 1, 2, and 4 are now on disk. Present three paths for human review:
- `docs/porting/families/<family>.md`
- `docs/models/<variant>.md`
- `models/<variant>/README.md`

The user reviews and edits the drafts directly — the skill does not expect the user to author prose from scratch. Flag the sections most likely to over-promise so the user examines them closely: the model card's `one_liner` and `capabilities_prose`, the family doc's Known Limitations. Wait for explicit sign-off before Step 6.

### Step 6: Sign-off

Report:
- All four paths.
- Target HF repo.
- Next command for the user to run (actual upload is not the skill's job):
  ```bash
  hf upload <target_repo> models/<variant> .
  ```
- Remind the user to commit the docs/families/models/hf_cards changes manually before the upload.

**Do not commit. Do not upload.** The upload is a human action because it publishes to a public registry.

## Postconditions

- `docs/porting/families/<family>.md` filled and reviewed.
- `docs/models/<variant>.md` authored with a populated download/WER/bench table.
- `scripts/hf_cards/<variant>.yaml` committed-ready.
- `models/<variant>/README.md` rendered.

## Pointers (read, not execute)

- `docs/porting/families/_template.md` — family doc shape
- `docs/models/parakeet-tdt-0.6b-v2.md` — model card shape reference
- `scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml` — hf card yaml reference
- `scripts/hf_cards/template.md.j2` — Jinja template that generate.py renders
- `scripts/hf_cards/generate.py` — renderer (execute only via Step 4)
