---
name: porting-2-refdump
description: Runs the reference implementation for a new model family and emits the contract tensor dumps, dump_coverage.json, and a provisional tests/tolerances/<family>.json skeleton. Use this after porting-1-intake has produced a schema-valid intake.json and before porting-3-convert. Input: intake.json and a per-family reference environment. Output: reference .f32 dumps under build/validate/<family>/<variant>/<case>/ref/, build/validate/<family>/<variant>/dump_coverage.json, and a provisional tolerances file that Stage 4 (porting-4-cpp) finalizes from observed C++ drift.
---

# porting-2-refdump

Second stage of the porting pipeline. Executes the family's reference implementation against the contract case(s), records what tensors were emitted, and writes a **provisional** `tests/tolerances/<family>.json` skeleton that Stage 4 will finalize with real drift values. The provisional file is deliberately loose so `porting-4-cpp`'s first `validate.py` pass runs end-to-end and produces observed drift, rather than short-circuiting on a missing file.

## Preconditions

- `reports/porting/<family>/<variant>/intake.json` exists, is schema-valid, and has Preflight Gate A green.
- The per-family reference environment exists at `scripts/envs/<family>/pyproject.toml`. If not, this skill creates it.
- The per-family dump adapter exists at `scripts/dump_reference_<family>_<framework>.py`. If not, this skill creates a skeleton following the contract in `scripts/lib/ref_dump.py` (or in one of the three existing dumpers).
- A committed reference checkpoint is available to the reference framework (HF download or `$TRANSCRIBE_MODELS_DIR/<slug>/`).
- `uv` is installed.

## Workflow

```
Refdump progress:
- [ ] Step 1: Confirm or write per-family env + dumper
- [ ] Step 2: Seed golden manifest from intake
- [ ] Step 3: Run the dump for each contract case (pass 1)
- [ ] Step 4: Repeat the dump (pass 2) for reference-stability noise floor
- [ ] Step 5: Generate dump_coverage.json
- [ ] Step 6: Write provisional tolerances from the noise floor
- [ ] Step 7: Human review of provisional tolerances
- [ ] Step 8: Sign-off review
```

### Step 1: Per-family env and dumper

Check for `scripts/envs/<family>/pyproject.toml` and `scripts/dump_reference_<family>_<framework>.py`.

**If the family is already ported** (either script exists): use as-is.

**If the family is new**:
- Create `scripts/envs/<family>/pyproject.toml` listing the reference framework's package pins. Mirror the shape of `scripts/envs/parakeet/pyproject.toml` (NeMo), `scripts/envs/cohere/pyproject.toml` (Transformers), or `scripts/envs/qwen3_asr/pyproject.toml` (author-repo style).
- Create `scripts/dump_reference_<family>_<framework>.py` using the argparse subcommand shape of the closest existing dumper:
  - NeMo → mirror `scripts/dump_reference_parakeet_nemo.py` (subcommands: `encoder`, `decode`).
  - Transformers → mirror `scripts/dump_reference_cohere_transformers.py` (subcommands: `mel`, `encoder`, `decode`).
  - Author repo → mirror `scripts/dump_reference_qwen3_asr_author.py` (subcommands: `encoder`, `decode`).
- All dumpers MUST emit tensors via the shared write-side (`scripts/lib/ref_dump.py`'s `write_tensor(name, array, stage, source)` helper) so the `.f32` + `.json` sidecar contract is identical across families.
- Ask the user to confirm the dumper stub before running anything against real weights.

### Step 2: Complete the golden manifest (execute)

`porting-1-intake` wrote a skeleton at `tests/golden/<family>/<variant>.manifest.json` with identity fields only (`_skeleton: true`). Fill it in:

- Add `reference.{kind, entrypoint}` — `kind` from `intake.reference_framework`, `entrypoint` pointing to `scripts/dump_reference_<family>_<framework>.py`.
- Add `frontend`, `tokenizer_summary`, `capabilities` from intake.
- Add `cases: [{audio: "samples/jfk.wav"}]` (or more if the user requests additional cases in Step 3).
- Remove the `_skeleton` marker.

See `tests/golden/parakeet/parakeet-tdt-0.6b-v2.manifest.json` for the complete shape.

Ask the user to fill in `reference.source` and `reference.revision` (the pinned upstream URL + version tag) — the script cannot infer these reliably.

### Step 3: Run the dump (execute)

Default contract case: `samples/jfk.wav`. Run the family's dumper for both the encoder and decode paths:

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

Subcommand names vary per dumper — check the script's `--help` for the exact set.

**Ask-point: contract case set.** Confirm with the user that `samples/jfk.wav` alone is sufficient for this family. If the family has unusual audio characteristics (non-English, tonal, music), ask whether to add a second case. Multiple cases go into `manifest.cases[]`.

### Step 4: Repeat dump for reference stability (execute)

Run each dumper invocation a second time with `--out .../ref2` instead of `--out .../ref`. Most reference frameworks are deterministic at inference, so ref vs ref2 typically diffs to fp32 round-off (~1e-7). If diffs are larger, the reference has a stochastic element (dropout not disabled, non-deterministic CUDA kernels, etc.); surface this to the user — a non-deterministic reference invalidates the tolerance contract.

Per tensor, compute `noise_max_abs` / `noise_mean_abs` = max/mean absolute difference between ref and ref2. This is the **noise floor**. A tolerance tighter than ~10× noise is physically impossible; Step 6 uses this to derive the provisional.

### Step 5: Generate dump_coverage.json (execute)

Walk `build/validate/<family>/<variant>/` and write `dump_coverage.json` with `{family, variant, tensors: [{case, stage, name, shape, dtype}, ...]}`. One entry per `.json` sidecar under any `<case>/<stage>/ref/`. Short `uv run python -c '...'` — glob the sidecars, read each, emit the list.

This is the contract Stage 4 consumes to know exactly which tensors to emit and compare.

### Step 6: Provisional tolerances from the noise floor (execute)

Write `tests/tolerances/<family>.json` with one entry per tensor, derived from the Step 4 ref-vs-ref diffs:

```
max_abs  = max(10 × noise_max_abs,  1e-6)
mean_abs = max(10 × noise_mean_abs, 1e-6)
```

Per-tensor entry: `{max_abs, mean_abs, _provisional: true}`. Top-of-file `_comment` (array of strings) states:
- These tolerances derive from 10× ref-vs-ref stability noise on `samples/jfk.wav`.
- They are provisional. `porting-4-cpp` finalizes them against observed C++ drift.
- Entries with `_provisional: true` flagged so it is obvious at a glance whether finalization ran.
- Do NOT ship a model while `_provisional` entries remain.

### Step 7: Human review of provisional tolerances (ask-point)

Present the provisional file to the user. Confirm:
- The noise floor looks reasonable (typical frameworks: max_abs ~1e-6 per-tensor; large values are a signal of stochastic reference behavior worth investigating).
- The `_comment` block clearly names ref-vs-ref stability as the source.
- The user is aware these will be rewritten (tightened or loosened per tensor) during Stage 4.

### Step 8: Sign-off

Report:
- Number of contract cases dumped (pass 1 + pass 2).
- Total tensor count in `dump_coverage.json`.
- Path to the manifest.
- Noise-floor summary: median and max ref-vs-ref noise across tensors.
- Confirmation that the user reviewed the provisional tolerances.

**Do not commit.**

## Postconditions

- `.f32` + `.json` sidecar pairs exist under `build/validate/<family>/<variant>/<case>/<stage>/{ref,ref2}/` for every tensor the dumper emits.
- `build/validate/<family>/<variant>/dump_coverage.json` catalogs them.
- `tests/golden/<family>/<variant>.manifest.json` exists and names the reference entrypoint.
- `tests/tolerances/<family>.json` exists, populated from 10× ref-vs-ref noise, reviewed by the user, with every tensor carrying `_provisional: true`.

## Pointers (read, not execute)

- `docs/porting/2-artifacts-and-goldens.md` — manifest contract
- `docs/porting/4-numerical-validation.md` — where this file's outputs flow next
- `scripts/lib/ref_dump.py` — the shared `write_tensor` helper (execute vs. read: read to understand, execute only via the per-family dumper)
- Existing dumpers as template references:
  - `scripts/dump_reference_parakeet_nemo.py`
  - `scripts/dump_reference_cohere_transformers.py`
  - `scripts/dump_reference_qwen3_asr_author.py`
