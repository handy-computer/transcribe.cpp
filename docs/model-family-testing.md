# Model Family Testing And Validation

This document defines the test and validation shape every model family
should converge on. Porting guides should reference this document rather
than re-describing the test matrix per family.

The full porting process starts earlier, with reference research,
artifact manifests, golden generation, conversion, and benchmark
baselines. See [`docs/porting/0-porting.md`](porting/0-porting.md) for
that linear workflow. This document is the test contract those porting
stages must eventually satisfy.

The goal is to make a new family cheap to bring up without weakening the
two things this repo depends on: strict numerical validation and
repeatable performance measurement.

## Family Test Contract

Each model family should have the same test contract, even when the
implementation details differ.

### 1. Fixture Loader Smoke

Purpose: catch loader, metadata, capability, tokenizer, and tensor-table
regressions without requiring a real model file.

Expected shape:

- Runs in the default test build when its synthetic fixture generator is
  available.
- Uses tiny generated GGUF fixtures under `tests/fixtures/`.
- Exercises the public C API for load and basic model metadata.
- May include test-only internal-header checks for hparams, tensor slots,
  representative tensor shapes, and representative tensor values.
- Returns `77` only for honest missing-fixture situations; stale fixture
  false-greens are not acceptable.

Naming:

- `transcribe_<family>_smoke`
- Source file: `tests/<family>_smoke.cpp`

Current examples:

- `transcribe_parakeet_smoke`
- Cohere does not yet have a synthetic fixture smoke.

### 2. Real Model Structural Smoke

Purpose: catch converter and real-checkpoint ingest drift.

Expected shape:

- Built only when `TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON`.
- Skips with return code `77` when the model path is not available.
- Uses one family-specific environment variable for the model path.
- Verifies architecture string, variant string if applicable,
  capabilities, hparams, language list, expected tensor count or
  tensor-table coverage, and canonical weight shapes.
- May use test-only internal-header checks via a private include of
  `src/`.

Naming:

- `transcribe_<family>_real_smoke`
- Source file: `tests/<family>_real_smoke.cpp`

Current examples:

- `transcribe_parakeet_real_smoke`
- `transcribe_cohere_real_smoke`

### 3. Real Model Public ABI / Transcript Smoke

Purpose: prove the family can run end-to-end through the public API and
produce an expected transcript on a stable sample.

Expected shape:

- Built only when `TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON`.
- Skips with return code `77` when the model path is not available.
- Loads the real model, runs `samples/jfk.wav` or an env-selected sample,
  and checks:
  - model load succeeds
  - context init succeeds
  - `transcribe_run` succeeds
  - `transcribe_full_text` is non-empty
  - transcript is within a family-specific edit-distance budget
  - result accessors relevant to the family are sane
  - timings are present and not obviously hung

Naming:

- Prefer `transcribe_<family>_e2e_smoke` for new families.
- Existing names may remain until there is a reason to rename.

Current examples:

- `transcribe_cohere_smoke`
- `transcribe_decoder_smoke` for Parakeet

### 4. Numerical Validation Gate

Purpose: localize tensor drift against a reference implementation.

The detailed tensor-by-tensor gate should be a script/target workflow,
not a new bespoke C++ test per family.

Expected shape:

```text
C++ dump -> reference dump -> compare_tensors.py --tolerances tests/tolerances/<family>.json
```

Current orchestration:

```bash
uv run scripts/validate.py all --family parakeet
uv run scripts/validate.py all --family cohere
```

Inputs:

- C++ dumps come from `TRANSCRIBE_DUMP_DIR`.
- Reference dumps come from `scripts/dump_reference.py` or the family
  reference dumper.
- Tolerances come from `tests/tolerances/<family>.json`.
- Transcript artifacts are `transcript.json`; when the reference writes
  one, validation requires exact C++ vs reference text equality.

Do not grow bespoke C++ tensor-comparison smoke tests per family. C++
smokes should cover load/run/API behavior; the dump/compare workflow
should cover numerical localization.

### 5. Benchmark Gate

Purpose: make performance work repeatable and reviewable.

Expected shape:

- Use `scripts/bench/run.py` to create reports.
- Use `scripts/bench/compare.py` to compare baseline and candidate
  report sets.
- Prefer wall-time RTF for user-visible speed decisions.
- Add per-block or per-op timing when optimizing encoder internals.
- Optionally record transcript/token hash or golden match status when
  running benchmark variants that might affect outputs.

## Tolerances

Tolerances are data, not C++ literals. The source of truth is:

```text
tests/tolerances/<family>.json
```

Use the current flat tensor-name mapping across families:

```json
{
  "_comment": ["why the tolerances are what they are"],
  "enc.final": {"max_abs": 1e-4, "mean_abs": 1e-6}
}
```

Generic fallback tolerances belong in the tool, for example
`compare_tensors.py --max-abs` and `--mean-abs`. Family-specific
tensor tolerances belong in `tests/tolerances/<family>.json`.

Quantization accuracy tolerances are separate for now because
`quant_accuracy.py` uses relative error bands by quant/family, while
dump comparison uses absolute per-tensor tolerances.

## C++ Smoke Tests And Tolerances

Avoid runtime JSON parsing in C++ smoke tests. It adds test-only runtime
machinery and makes the smoke tests harder to keep small.

If a C++ smoke test genuinely needs a tolerance from
`tests/tolerances/<family>.json`, prefer a generated header produced at
configure/build time. The generated header should include only the
constants that C++ tests need.

Example shape:

```cpp
#pragma once

#define TRANSCRIBE_TOL_PARAKEET_ENC_MEL_IN_MAX_ABS 5e-4
#define TRANSCRIBE_TOL_PARAKEET_ENC_MEL_IN_MEAN_ABS 5e-5
#define TRANSCRIBE_TOL_PARAKEET_ENC_FINAL_MAX_ABS 1e-4
#define TRANSCRIBE_TOL_PARAKEET_ENC_FINAL_MEAN_ABS 1e-6
```

Do not add this generator with a dependency that makes default C++
smoke builds require heavyweight reference environments. If the
generator is not trivial and dependency-light, keep the few C++ literals
temporarily and track the wiring as follow-up.

## New Family Checklist

For a new family, add or update:

- `src/arch/<family>/`
- registry entry in `src/transcribe-arch.cpp`
- converter script
- reference dump support
- `tests/tolerances/<family>.json`
- tiny fixture and `tests/<family>_smoke.cpp`
- `tests/<family>_real_smoke.cpp`
- `tests/<family>_e2e_smoke.cpp` or equivalent public ABI transcript gate
- bench matrix support
- `scripts/validate.py` manifest support
- `scripts/envs/<family>/pyproject.toml`

The checklist is the contract. The file names may vary for legacy
families, but family #3 should not need to invent a new testing shape.
