---
name: porting-2-oracle
description: Builds the oracle packet Stage 4 implements against: tensor dumps, reference transcripts, measured reference WER, and provisional tolerances. Use after porting-1-intake clears Gate A and before porting-3-convert. Output: tests/golden/<family>/<variant>.manifest.json, build/validate/<family>/<variant>/, reports/wer/<variant>-REF.<dataset>.{jsonl,score.json}, and tests/tolerances/<family>.json.
---

# porting-2-oracle

Second stage of the porting pipeline. Produces the oracle packet Stage 4
implements against: tensor dumps, reference transcripts, measured
reference WER, and provisional tolerances.

## Preconditions

- `reports/porting/<family>/<variant>/intake.json` exists, is schema-valid,
  Preflight Gate A green.
- The per-family reference environment exists at
  `scripts/envs/<family>/pyproject.toml`. If not, this skill creates it.
- The per-family dump adapter exists at
  `scripts/dump_reference_<family>_<framework>.py`. If not, this skill
  creates a skeleton following the contract in `scripts/lib/ref_dump.py`
  (or in one of the existing dumpers).
- A committed reference checkpoint is available to the reference framework
  (HF download or `$TRANSCRIBE_MODELS_DIR/<slug>/`).
- `uv` is installed.

## Oracle packet contents

After this stage finishes, the oracle packet is exactly:

```text
tests/golden/<family>/<variant>.manifest.json
build/validate/<family>/<variant>/<case>/<stage>/ref/                  # tensor dumps + sidecars
build/validate/<family>/<variant>/dump_coverage.json                   # tensor catalog
build/validate/<family>/<variant>/<case>/<stage>/ref/transcript.json   # reference transcript per decode case
reports/wer/<variant>-REF.<dataset>.jsonl                              # per-utterance reference hyps (one-for-one debug oracle)
reports/wer/<variant>-REF.<dataset>.score.json                         # reference-measured WER baseline (the gate, not the published number)
tests/tolerances/<family>.json                                         # provisional, Stage 4 finalizes
```

`<case>` is the manifest case identifier (typically `jfk` for the default
sample; families may add additional audio cases). Per-stage tensors live
under `<case>/encoder/`, `<case>/decoder/`, etc., per the dumper's stage
names.

## Workflow

```
Oracle progress:
- [ ] Step 1: Confirm or write per-family env + dumper
- [ ] Step 2: Complete the golden manifest
- [ ] Step 3: Run the dumper for every manifest case
- [ ] Step 4: Verify reference transcripts
- [ ] Step 5: Generate dump_coverage.json
- [ ] Step 6: Write provisional tolerances from per-tensor magnitude
- [ ] Step 7: Reference WER baseline ran on the acceptance dataset
- [ ] Step 8: Sign-off review
```

### Step 1: Per-family env and dumper

Check for `scripts/envs/<family>/pyproject.toml` and
`scripts/dump_reference_<family>_<framework>.py`.

**If the family is already ported** (either script exists): use as-is.

**If the family is new**:
- Create `scripts/envs/<family>/pyproject.toml` from the closest existing
  reference env.
- Create `scripts/dump_reference_<family>_<framework>.py` from the closest
  dumper. It MUST emit tensors via `ref_dump.write_tensor` and transcripts
  via `write_transcript`; sidecars must include `rms` and `p99_abs`.
- Surface only unresolved technical decisions to the user.

### Step 2: Complete the golden manifest (execute)

`porting-1-intake` wrote a skeleton at
`tests/golden/<family>/<variant>.manifest.json`. Fill it in:

- `reference.{kind, entrypoint, source, revision}` — `kind` from
  `intake.reference_framework`, `entrypoint` pointing to the family
  dumper, `source` + `revision` to the pinned upstream URL + tag.
- `frontend`, `tokenizer_summary`, `capabilities` from intake.
- `cases: [...]` — the audio cases the dumper will run. Default is
  `[{audio: "samples/jfk.wav"}]`. Add additional rows only when the
  family has unusual audio characteristics that one sample cannot cover
  (non-English, tonal, music, long-form). Confirm with the user before
  adding cases beyond the default.
- Remove the `_skeleton` marker.

See `tests/golden/parakeet/parakeet-tdt-0.6b-v2.manifest.json` for the
complete shape.

### Step 3: Reference dumps (execute)

For every entry in `manifest.cases[]`, run the dumper's encoder + decode
subcommands. Default invocation for `samples/jfk.wav`:

```bash
uv run --project scripts/envs/<family> \
  scripts/dump_reference_<family>_<framework>.py encoder \
  --model <hf_repo> \
  --audio samples/jfk.wav \
  --out build/validate/<family>/<variant>/jfk/encoder/ref

uv run --project scripts/envs/<family> \
  scripts/dump_reference_<family>_<framework>.py decode \
  --model <hf_repo> \
  --audio samples/jfk.wav \
  --out build/validate/<family>/<variant>/jfk/decode/ref
```

Subcommand names vary per dumper — check the script's `--help` for the
exact set. For families with extra subcommands (e.g. `mel`, `prompt`)
run each as the family contract requires.

If the reference produces a transcript that is implausible (gibberish,
empty, language mismatch), STOP — the reference setup is broken, and
fixing it is a precondition for any C++ work. Do not proceed.

### Step 4: Reference transcripts (execute)

Every reference decode pass should produce a transcript when the dumper
exposes one. Use `scripts/lib/ref_dump.py::write_transcript` (most
existing dumpers already call it). Verify
`build/validate/<family>/<variant>/<case>/<stage>/ref/transcript.json`
exists for any decode pass that produces text or tokens.

If the reference framework does not expose a transcript helper for a
given decode path, skip the transcript write for that case rather than
inventing one — Stage 4's behavior validation runs against the C++
runtime via the family-doc Capability Validation table, not against
synthetic reference transcripts.

### Step 5: Generate dump_coverage.json (execute)

Walk `build/validate/<family>/<variant>/` and write `dump_coverage.json`
with `{family, variant, tensors: [{case, stage_dir, stage, name, shape, dtype}, ...]}`.

One entry per **tensor** sidecar — that means every `.json` file under
`<case>/<stage>/ref/` whose contents include the tensor metadata fields
written by `scripts/lib/ref_dump.py::write_tensor` (`name`, `shape`,
`dtype`, `layout`, plus `rms` / `p99_abs`). Skip any other JSON
files in the same directory: `transcript.json` (text/tokens, not a
tensor), and any future behavioral artifacts the dumper grows. The
discriminator is presence of `shape` + `dtype` + `layout` in the parsed
JSON; iterating purely on `.json` extension would silently include
transcripts and corrupt validate.py's tensor list.

This catalog is the contract Stage 4 consumes to know exactly which
tensors to emit and compare.

### Step 6: Provisional tolerances from per-tensor magnitude (execute)

Write `tests/tolerances/<family>.json` with one entry per tensor, derived
from the magnitude statistics already recorded in each sidecar
(`p99_abs` and `rms`):

```
max_abs  = max(1e-4 × p99_abs, 1e-6)   # 0.01% of typical large value
mean_abs = max(1e-5 × rms,     1e-6)   # 0.001% of RMS energy
```

When the same tensor name appears in multiple `(case, stage_dir)`
sidecars, take the **max** p99_abs and rms across them so the budget
covers the worst-magnitude instance.

Per-tensor entry: `{max_abs, mean_abs, _provisional: true}`. Top-of-file
`_comment` (array of strings) states:
- These tolerances are magnitude-aware: `1e-4 × p99_abs` for max,
  `1e-5 × rms` for mean, with a `1e-6` floor.
- They are provisional — Stage 4 finalizes them against observed C++ drift.
- Entries with `_provisional: true` make finalization-not-yet-run obvious.
- Do NOT ship a model while `_provisional` entries remain.

Tensors well above this budget in Stage 4 are signals to investigate.

### Step 7: Reference WER baseline (execute)

The Oracle WER run is the downstream accuracy baseline. Do not gate
against publisher scores: Stage 4 and Stage 7 compare C++ against this
measured reference run on our manifest and normalization. If intake
captured a publisher score, compare and report the delta for context only.

Keep every per-utterance reference hypothesis. Later WER drift should diff
C++ vs captured reference `hyp_text` by `id` instead of re-running the
reference.

The acceptance dataset is `intake.upstream_benchmarks[0].dataset`
(default LibriSpeech test-clean). Build the WER manifest once if it does
not exist, then run the per-family reference runner over the **full**
acceptance manifest and score it:

```bash
DATASET=$(uv run python -c "import json; \
  print(json.load(open('reports/porting/<family>/<variant>/intake.json'))['upstream_benchmarks'][0]['dataset'].replace(' ','-').lower())")

# Build the manifest if this dataset has not been ingested yet.
[ -f samples/wer/${DATASET}.manifest.jsonl ] || \
  uv run scripts/wer/ingest.py librispeech   # or: fleurs --lang <bcp47>

uv run --project scripts/envs/<family> \
  scripts/wer/run_reference_<family>_<framework>.py \
  --manifest samples/wer/${DATASET}.manifest.jsonl \
  --model <hf_repo> \
  --out reports/wer/<variant>-REF.${DATASET}.jsonl

uv run scripts/wer/score.py reports/wer/<variant>-REF.${DATASET}.jsonl
# writes reports/wer/<variant>-REF.${DATASET}.score.json
```

This is a **one-time** reference run. Stage 4 and Stage 7 gate C++ WER
against this same file; the reference is not re-run downstream.

**GPU option (Modal).** To run the reference on cloud GPUs instead of
locally, use the remote sweep — it installs the family venv on a GPU
container and runs the same runner, returning the JSONL to score locally:

```bash
modal run scripts/wer/remote/modal_sweep.py::reference_sweep \
  --variants <family>:<variant> --gpu L4
```

It writes the same `reports/wer/<variant>-REF.<dataset>.jsonl`. See
`scripts/wer/remote/REFERENCE_SWEEP_SPEC.md`. A new family needs a
`scripts/wer/run_reference_<family>_*.py` with the uniform
`--manifest/--model/--out/--device/--batch-size` contract (copy the
closest same-framework one).

### Step 8: Sign-off

Report:
- Number of manifest cases dumped.
- Total tensor count in `dump_coverage.json`.
- Path to the manifest.
- Tolerance summary: median and max `max_abs` across tensors so the
  user can eyeball whether the magnitude-aware numbers look sane (e.g.
  a logits-scale tensor should have a budget on the order of 1e-2; an
  embedding-scale tensor 1e-4).
- Per-case transcript paths so the user can sanity-check the reference
  output before C++ work.
- Reference WER baseline: measured reference WER on the acceptance
  dataset, the published score if present, the delta between them, the
  dataset + utterance count, and the
  `reports/wer/<variant>-REF.<dataset>.{jsonl,score.json}` paths.
- Confirmation that the user reviewed the provisional tolerances.

**Do not commit.**

## Postconditions

- Every path listed in "Oracle packet contents" exists and uses the same
  `<family>/<variant>/<case>/<stage>` conventions.
- Tensor sidecars include `rms` and `p99_abs`; `dump_coverage.json`
  catalogs tensors only.
- `tests/tolerances/<family>.json` is provisional, derived from
  per-tensor magnitude, every tensor carries `_provisional: true`, and the
  user reviewed it.
- The measured reference WER baseline exists for the acceptance dataset,
  and any measured-vs-published gap was reported to the user.

## Pointers (read, not execute)

- `docs/porting/2-artifacts-and-goldens.md` — manifest contract
- `docs/porting/4-numerical-validation.md` — where this file's outputs flow next
- `scripts/lib/ref_dump.py` — shared `write_tensor` / `write_transcript`
- Existing dumpers as template references:
  - `scripts/dump_reference_parakeet_nemo.py`
  - `scripts/dump_reference_cohere_transformers.py`
  - `scripts/dump_reference_qwen3_asr_author.py`
  - `scripts/dump_reference_whisper_transformers.py`
- Reference WER (Step 7):
  - `scripts/wer/run_reference_<family>_<framework>.py` — per-family
    reference WER runners (copy the closest same-framework one for a new
    family); local, GPU-accelerated if torch sees one
  - `scripts/wer/ingest.py` — builds the acceptance manifest (librispeech / fleurs)
  - `scripts/wer/score.py` — shared scorer; writes `<report>.score.json`
