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
- The catalog satisfies `asr-publication-v2` for this variant: complete
  accuracy and every speed quant/sample/machine/backend cell selected by the
  profile (currently Q8_0 and Q4_K_M on both `jfk` and `dots`, except an
  explicit supported-language sample override such as GigaAM's `ru`).

## Workflow

One ordered finalization. Each step gates the next; nothing is uploaded
until the human review in Step 7 has passed on files that already survived
every check.

```
Ship progress:
- [ ] Step 1: Verify inputs and GGUF metadata
- [ ] Step 2: Ingest measurements
- [ ] Step 3: Validate the publication profile
- [ ] Step 4: Author prose (family doc, model page, HF card spec)
- [ ] Step 5: Render docs and card
- [ ] Step 6: Verify no drift
- [ ] Step 7: Human review (ask-point)
- [ ] Step 8: Upload
```

### Step 1: Verify inputs and GGUF metadata (execute)

Confirm every artifact exists. If any row is missing, halt and send the
user to the stage that owns it. Stage 8 does not fabricate inputs.

| Artifact | Expected path | Owning stage |
|---|---|---|
| Intake | `reports/porting/<family>/<variant>/intake.json` | Stage 1 |
| Manifest | `tests/golden/<family>/<variant>.manifest.json` | Stage 2 |
| Tolerances | `tests/tolerances/<family>.json` | Stage 4 |
| Forward map | `reports/porting/<family>/forward-map.md` | Stage 4 |
| Converter report | `reports/convert/<variant>-<REFDTYPE>.json` | Stage 3 |
| Quants | `models/<variant>/<variant>-*.gguf` | Stage 5 |
| Bench reports, rig 1 | `### Apple M4 Max` section in `docs/models/<variant>.md` | Stage 6 |
| Bench reports, rig 2 | `### AMD Ryzen 7 PRO 4750U` section in `docs/models/<variant>.md` | Stage 6 |
| WER score JSONs | `reports/wer/<variant>-*.<dataset>.score.json` | Stage 7 |
| WER summary | `reports/wer/<variant>.<dataset>.summary.md` | Stage 7 |
| Catalog publication profile | `catalog/_benchmark_profiles.json` + `catalog/<variant>.json` | Stages 6–7 |

```bash
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
```

Then audit the files that will be uploaded:

```bash
uv run --project scripts/envs/moonshine scripts/audit_gguf_metadata.py models/<variant>
uv run scripts/catalog/sync_capabilities.py --check --models <variant>
```

`audit_gguf_metadata.py` exits non-zero on any metadata issue.
`sync_capabilities.py --check --models <variant>` reads every published quant
and exits non-zero if any file is unreadable, lacks a capability KV, disagrees
with another quant, or disagrees with the record. A file and its own model
card must not contradict each other on the Hub.

**Never hand-write the `capabilities` block.** Hand-writing it is how
moss-transcribe-diarize shipped as `diarize:false`, how the granite GGUFs came
to carry `stt.capability.translation` where the loader reads
`stt.capability.translate`, and how nemotron-3.5 shipped with no streaming KV
at all. `sync_capabilities.py` (without `--check`) reads the block back out of
the file. If it disagrees with what the model actually does, the GGUF is
wrong and the fix is a converter change plus a re-export, never an edit to the
record or the card spec.

**Absence is not falsity.** `read_capability_bool()` leaves a field untouched
when its KV is missing, so a missing KV silently inherits the family default.
`granite/capabilities.cpp` sets `supports_translate = true` on purpose so
each variant's GGUF can lower it; `granite-speech-4.1-2b-plus` spelled that
key `stt.capability.translation`, the lowering never happened, and a model
that does not translate advertised that it does. The shared writer factory in
`scripts/lib/gguf_common.py` writes `false` for any capability KV a converter
leaves unset, so every fresh export states all four.

**Know where the local files came from.** `models/<variant>/` is for most
variants a symlink into external storage holding whatever was built there
last. That mirror can be *older* than the Hub. Before re-uploading a variant
you did not just convert, either re-download it from its published repo or
confirm the divergence is intended.

### Step 2: Ingest measurements (execute)

Idempotent; Stages 6 and 7 normally did this already.

```bash
uv run scripts/catalog/ingest_perf.py
uv run scripts/catalog/ingest_accuracy.py --models <variant>
```

`ingest_perf.py` takes publication runs only and refuses a row whose xRT
would move more than 5% against the published value; re-bench at the current
sha rather than passing `--force`. `ingest_accuracy.py` takes profile-stamped
scores only. Confirm `headline_benchmark` is set in `catalog/<variant>.json`.

### Step 3: Validate the publication profile (execute)

```bash
uv run scripts/catalog/check.py --publication-profile --models <variant>
```

A failure halts Stage 8. A `legacy-published` provenance marker is honest
migration provenance and may satisfy the gate; it is not permission to assign
a guessed engine SHA.

### Step 4: Author prose (execute + ask-point)

Three files, prose only. Every number, repo, licence, language, capability,
and table is rendered from the catalog in Step 5.

### Step 4a: Family doc

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
capabilities flags imply. Present the draft for human review in Step 7.

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

### Step 4b: Model page

Author `docs/models/<variant>.md` by copying the closest existing model
page (e.g. `docs/models/parakeet-tdt-0.6b-v2.md`) and editing the prose.
There is one way to do this and this is it: a page is hand-written prose
around `<!-- catalog:… -->` regions, and the copy already carries the right
set of regions in the right order.

Replace the prose only. Every table, number, repo, licence, language and
capability comes from the `catalog:` regions, which Step 5 fills from
`catalog/<variant>.json`. Leave the region bodies as they are; the renderer
overwrites them. Reference context for the prose comes from
`reports/wer/<variant>-*.score.json` and the acceptance dataset from
`intake.upstream_benchmarks[0]`. `published_repo` is catalog data, not an
editorial value to ask for again.

Do not restate a catalogued number in the prose. The parameter count, the
download sizes, the error rates and the validation commit are all rendered a
few lines away, and a hand-typed copy of one is a second source of truth that
nothing checks. Say what the model is for; let the regions say how big and
how accurate it is.

Everything outside a `catalog:` region is yours, and re-running the renderer
never touches it.

### Step 4c: HF card spec

Write `scripts/hf_cards/<variant>.yaml`, mirroring a current nearby spec.
It contains editorial and release state only; identity, repositories, upstream
commit, license, language support, downloads, benchmark values, capability
flags, the metric column label and the link back to the model page are all
derived from `catalog/<variant>.json`. `generate.py` accepts only the keys
below and refuses anything else, so a field that belongs to the catalog
cannot creep back in and a misspelled key fails loudly:

```yaml
pin_date: <ISO date when the upstream revision was pinned>

validation:
  reference: <intake.reference_framework>
  commit: <the transcribe.cpp commit the validation actually ran at>
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

wer:
  notes: |
    <how the headline number was measured; anything a reader needs to compare it>
```

`validation.commit` is a real commit and is rendered as a link into the
tree, so it must be a SHA. Record the commit the validation ran at; the
reference framework's version belongs in `validation.reference`. Never
copy a plausible-looking SHA from a sibling spec: a pin that names no
real run is worse than no pin, because it cannot be told from one that does.

`wer.notes` holds editorial caveats only. The mechanical sentence
(dataset, size, batch, timestamps, build) is generated from the headline
rows, and the metric column label is the catalog's own; state a number here
only to compare against something the catalog does not hold, such as an
upstream self-reported figure.

### Step 5: Render docs and card (execute)

```bash
uv run scripts/hf_cards/check_release.py <variant>   # pin_date and validation pin
uv run scripts/catalog/render.py                      # docs/models/*.md and the root README
uv run scripts/hf_cards/generate.py scripts/hf_cards/<variant>.yaml
```

`render.py` fills every marked region: download and accuracy tables, perf
tables with their provenance, the intro and WER note from the spec, family
roll-ups, and the supported-models table in the root README. `generate.py`
writes `models/<variant>/README.md` from the spec and the catalog record, and
refuses a spec that states a catalog-owned field.

### Step 6: Verify no drift (execute)

```bash
uv run scripts/catalog/format.py --check
uv run scripts/catalog/check.py --publication-profile --models <variant>
uv run scripts/catalog/render.py --check
```

All three must be clean. This is what CI runs; a failure here is a failure
there.

### Step 7: Human review (ask-point)

Present for review:
- `docs/porting/families/<family>.md`
- `docs/models/<variant>.md`
- `models/<variant>/README.md`

Flag likely over-promising sections (the spec's `summary` and `wer.notes`,
which render into both the HF README and the model page; Known Limitations)
and wait for explicit sign-off before Step 8.

### Step 8: Upload (execute)

```bash
hf upload <target_repo> models/<variant> . --repo-type model
```

Report the output paths, the target private repo, and the Step 6 results.
Remind the user to commit the catalog, docs, and spec changes.

**Do not commit.** Keep the repo private; flipping it public is a future
action, not part of this stage.

## Postconditions

- Steps 1 to 3 were green before any prose was drafted.
- HF validation commit exists and `validation.date` equals the UTC ship date.
- `docs/porting/families/<family>.md` filled and reviewed.
- `docs/models/<variant>.md` authored; its marked regions render from the
  catalog.
- `scripts/hf_cards/<variant>.yaml` committed-ready.
- `models/<variant>/README.md` rendered and uploaded to the private repo;
  public flip deferred.
- The root README's supported-models table renders the new variant.

## Pointers (read, not execute)

- `docs/porting/families/_template.md` — family doc shape
- `docs/models/parakeet-tdt-0.6b-v2.md` — model page shape reference
- `scripts/hf_cards/parakeet-tdt-0.6b-v2.yaml` — HF card spec reference
- `scripts/hf_cards/README.md` — the render, review, upload loop
- `scripts/catalog/render.py` — marker blocks and what each renders
