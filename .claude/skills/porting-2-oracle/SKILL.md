---
name: porting-2-oracle
description: Builds the oracle packet — the reference answer key Stage 4 implements against. Runs the reference framework against the family's manifest cases once, captures tensor sidecars (with rms / p99_abs), reference transcripts (transcript.json), and a provisional tests/tolerances/<family>.json derived from per-tensor magnitude statistics (1e-4 × p99_abs for max, 1e-5 × rms for mean). Use after porting-1-intake clears Gate A and before porting-3-convert. Output: build/validate/<family>/<variant>/<case>/<stage>/ref/, dump_coverage.json, transcript.json per decode case, and a provisional tolerances file Stage 4 will finalize.
---

# porting-2-oracle

Second stage of the porting pipeline. Produces the oracle packet — the
collection of artifacts Stage 4 implements C++ against. The oracle is
the tensor dumps + transcripts + magnitude-aware provisional tolerances.

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
- [ ] Step 7: Sign-off review
```

### Step 1: Per-family env and dumper

Check for `scripts/envs/<family>/pyproject.toml` and
`scripts/dump_reference_<family>_<framework>.py`.

**If the family is already ported** (either script exists): use as-is.

**If the family is new**:
- Create `scripts/envs/<family>/pyproject.toml` listing the reference
  framework's package pins. Mirror the shape of `scripts/envs/parakeet/`
  (NeMo), `scripts/envs/cohere/` (Transformers), or `scripts/envs/qwen3_asr/`
  (author-repo style).
- Create `scripts/dump_reference_<family>_<framework>.py` using the
  argparse subcommand shape of the closest existing dumper. All dumpers
  MUST emit tensors via `scripts/lib/ref_dump.py::write_tensor` and
  transcripts via `write_transcript` so the on-disk contract is identical
  across families. The shared sidecar records `rms` and `p99_abs`
  alongside `min`/`max`/`mean` — Step 6 reads these to size tolerances.
- Surface to the user any unresolved technical decisions the closest
  existing dumper does not answer (novel reference framework wiring,
  unusual hook points, multi-stage decode quirks). Routine stubs do not
  need confirmation — just run them.

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

Why these multipliers: observed fp32-vs-fp32 drift between two correct
implementations (different BLAS, different reduction order, FMA vs
mul+add) is typically ~1e-5 relative per tensor — well below
Wilkinson worst-case (~1e-4 for stacked matmuls of width K across
L layers). The 1e-4 / 1e-5 prior is 10× looser than typical observed
drift: loose enough to admit benign cross-library variation, tight
enough that a real port issue (wrong layer norm epsilon, missing
residual, scale factor off, unintended dtype upcast) trips the
budget on first Stage 4 run. The 1e-6 floor catches near-zero
tensors (zero-init biases, all-zero pad regions) where the relative
budget would otherwise vanish.

Per-tensor entry: `{max_abs, mean_abs, _provisional: true}`. Top-of-file
`_comment` (array of strings) states:
- These tolerances are magnitude-aware: `1e-4 × p99_abs` for max,
  `1e-5 × rms` for mean, with a `1e-6` floor.
- They are provisional — Stage 4 finalizes them against observed C++ drift.
- Entries with `_provisional: true` make finalization-not-yet-run obvious.
- Do NOT ship a model while `_provisional` entries remain.

A correctly-ported tensor should land near or below this budget on
first run; tensors well above it are real signals to investigate, not
artifacts of a placeholder floor.

### Step 7: Sign-off

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
- Confirmation that the user reviewed the provisional tolerances.

**Do not commit.**

## Postconditions

- `.f32` + `.json` sidecar pairs exist under
  `build/validate/<family>/<variant>/<case>/<stage>/ref/` for every
  tensor the dumper emits.
- Sidecars include `rms` and `p99_abs` (provided by the shared
  `ref_dump.write_tensor`).
- `build/validate/<family>/<variant>/dump_coverage.json` catalogs every
  ref tensor (and only tensors — `transcript.json` and other behavioral
  files are excluded).
- `build/validate/<family>/<variant>/<case>/<stage>/ref/transcript.json`
  exists wherever the reference dumper exposes a transcript.
- `tests/golden/<family>/<variant>.manifest.json` lists the audio cases
  the dumper actually ran and names the reference entrypoint.
- `tests/tolerances/<family>.json` exists, populated from per-tensor
  magnitude (`max(1e-4 × p99_abs, 1e-6)` for max, `max(1e-5 × rms, 1e-6)`
  for mean), every tensor carries `_provisional: true`, top-of-file
  `_comment` states the derivation source.

## Pointers (read, not execute)

- `docs/porting/2-artifacts-and-goldens.md` — manifest contract
- `docs/porting/4-numerical-validation.md` — where this file's outputs flow next
- `scripts/lib/ref_dump.py` — shared `write_tensor` / `write_transcript`
- Existing dumpers as template references:
  - `scripts/dump_reference_parakeet_nemo.py`
  - `scripts/dump_reference_cohere_transformers.py`
  - `scripts/dump_reference_qwen3_asr_author.py`
  - `scripts/dump_reference_whisper_transformers.py`
