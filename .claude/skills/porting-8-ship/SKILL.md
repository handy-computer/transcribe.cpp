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
- The catalog satisfies `asr-publication-v1` for this variant: complete
  accuracy and every speed quant/sample/machine/backend cell selected by the
  profile (currently Q8_0 and Q4_K_M on both `jfk` and `dots`, except an
  explicit supported-language sample override such as GigaAM's `ru`).

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
| Catalog publication profile | `catalog/_benchmark_profiles.json` + `catalog/<variant>.json` | Stages 6–7 |

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
uv run scripts/catalog/check.py --publication-profile --models <variant>
```

Any `MISSING` or publication-profile failure halts Stage 8. A
`legacy-published` provenance marker is honest migration provenance and may
satisfy the current gate; it is not permission to assign a guessed engine SHA.

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
- Batch: confirm the family ships an explicit `run_batch()` parallel fast
  path (PASS — batching `MUST PASS`, not optional) and that it is
  WER-neutral (byte-identical to single-stream). Serial fallback is
  reportable only as `ACCEPTED GAP — serial (benchmarked no faster)`, or as an
  explicit user-signed `BLOCKER`; it is never a silent default.
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
   facts directly from artifacts — published quants, accuracy, and speed from
   `catalog/<variant>.json`; measured reference context from
   `reports/wer/<variant>-*.score.json`; and the acceptance dataset from
   `intake.upstream_benchmarks[0]`. `published_repo` is catalog data, not an
   editorial value to ask for again.
2. **Render from the existing template** if the template already covers
   everything the variant needs and the variant has no rendered card
   yet. Build the context dict in a short ad-hoc `uv run python -c`
   and write the output. Do **not** extend the template with new
   context fields just for one family — handcraft those sections in
   the rendered markdown instead.

Subsequent regenerations must respect human edits.

### Step 4: HF card YAML spec (execute)

Write `scripts/hf_cards/<variant>.yaml`, mirroring a current nearby spec.
It contains editorial and release state only; identity, repositories, upstream
commit, license, language support, downloads, benchmark values, and capability
flags are derived from `catalog/<variant>.json`:

```yaml
transcribe_docs_url: https://github.com/handy-computer/transcribe.cpp/blob/main/docs/models/<variant>.md
pin_date: <ISO date when the upstream revision was pinned>

validation:
  reference: <intake.reference_framework>
  commit: <validated transcribe.cpp commit SHA>
  date: <today in UTC>

pipeline_tag: automatic-speech-recognition
tags:
  - gguf
  - transcribe.cpp
  - asr
  - speech-to-text
  - <family>
  - <architecture-style>
summary: |
  <reviewed editorial summary>
```

Before rendering, verify that the validation pin is a real commit and its date
is today's UTC ship date (while `pin_date` is not in the future):

```bash
uv run scripts/hf_cards/check_release.py <variant>
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

## Catalog (mandatory exit step)

**Never hand-write the `capabilities` block.** Hand-writing it is how
moss-transcribe-diarize shipped as `diarize:false`, how the granite GGUFs came
to carry `stt.capability.translation` where the loader reads
`stt.capability.translate`, and how nemotron-3.5 shipped with no streaming KV
at all. Read it back out of the file you are shipping:

```bash
uv run scripts/catalog/sync_capabilities.py
uv run scripts/catalog/render.py
uv run scripts/catalog/render.py --check
uv run scripts/catalog/check.py --publication-profile --models <variant>
uv run scripts/hf_cards/check_release.py <variant>
```

**Gate the upload, before `hf upload`, never after:**

```bash
uv run --project scripts/envs/moonshine scripts/audit_gguf_metadata.py models/<variant>
uv run scripts/catalog/sync_capabilities.py --repair <variant> --dry-run
```

`audit_gguf_metadata.py` exits non-zero on any metadata issue and was written
to gate exactly this. The `--repair --dry-run` pass must report `already
correct` for every quant: a capability KV that disagrees with the record means
the file and its own model card are about to contradict each other on the Hub.

**Audit the file you are about to upload, and know where it came from.** Both
tools read `models/<variant>/`, which for most variants is a symlink into
external storage holding whatever was built there last. That mirror can be
*older* than the Hub: a re-export or reconvert lands on the Hub and the local
copy is never refreshed. Auditing it then reports the mirror's gaps as if they
were the published file's, and repairing and uploading it republishes the older
build under an unchanged filename -- reverting whatever the published file had
gained. A stale `granite-speech-4.1-2b-nar` mirror here carried an older
upstream snapshot (`enc.ctc_bpe` 100353 vs the published 100352, no
`bpe_blank_id`) while looking like a perfectly ordinary repair target.

`sync_capabilities.py --repair` now range-reads the published header and
refuses any file whose tensor shapes, dtypes, or unrelated KVs differ from what
is published; `--skip-published-check` overrides it, and is only correct when
the local file is deliberately newer than the Hub. Nothing enforces this for a
plain `hf upload`, so before re-uploading a variant you did not just convert,
either re-download it from its published repo or confirm the divergence is
intended.

**Absence is not falsity.** `read_capability_bool()` returns OK and leaves the
field untouched when a key is missing, so a missing KV silently inherits the
family default. `granite/capabilities.cpp` sets `supports_translate = true` on
purpose so each variant's GGUF can lower it; `granite-speech-4.1-2b-plus`
spelled that key `stt.capability.translation`, the lowering never happened,
and a model that does not translate advertised that it does. Declare every
capability explicitly rather than relying on a default to be right.

If `sync_capabilities.py` disagrees with what the model actually does, the
GGUF is wrong and the fix is a converter change plus a re-export. Do not
paper over it with an override in the card spec.

The HF card spec under `scripts/hf_cards/` carries editorial copy only:
summary, tags, pipeline tag, validation pin, prose notes. Repos, commit,
licence, languages, quant table, capability flags and per-rig speedups are all
derived from the catalog record. `check.py` fails if a spec re-states one.

## Postconditions

- Pre-flight checklist and per-variant publication profile were green before
  any drafting.
- HF validation commit exists and `validation.date` equals the UTC ship date.
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
