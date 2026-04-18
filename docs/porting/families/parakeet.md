# Parakeet

Status: supported with manifest-driven numerical validation against the
NeMo canonical reference. C++ CPU validation passes locally.

## Identity

- Family key: `parakeet`
- Upstream architecture string: `parakeet`
- HF source repo: `nvidia/parakeet-tdt-0.6b-v2`
- Variants: `tdt-0.6b-v2`, `tdt-0.6b-v3`

## References

- Canonical reference: **NeMo** (`nvidia/parakeet-tdt-0.6b-v2` and
  `nvidia/parakeet-tdt-0.6b-v3` via `ASRModel.from_pretrained`). NeMo is
  NVIDIA's own implementation and the authoritative source for Parakeet TDT
  weights and inference behavior.
  Script: `scripts/dump_reference_parakeet_nemo.py`.
- Instrumented reference: **NeMo** (same script, using forward hooks to
  capture per-stage intermediates without modifying NeMo internals).

Validation status:

- NeMo reference dumps and C++ CPU dumps pass via
  `uv run scripts/validate.py compare --family parakeet --variant <variant>`
  for both `parakeet-tdt-0.6b-v2` and `parakeet-tdt-0.6b-v3`.
- C++ and NeMo transcripts match exactly on `samples/jfk.wav` for both
  variants.

## Environment

```bash
# NeMo reference environment
uv run --project scripts/envs/parakeet ...
```

Python env: `scripts/envs/parakeet/pyproject.toml`
(nemo_toolkit[asr], torch, soundfile, numpy, sentencepiece).

## Golden Manifest

`tests/golden/parakeet/parakeet-tdt-0.6b-v2.manifest.json`

## Current Commands

Full validation:

```bash
uv run scripts/validate.py all --family parakeet
```

Or step by step:

```bash
uv run scripts/validate.py ref     --family parakeet
uv run scripts/validate.py cpp     --family parakeet
uv run scripts/validate.py compare --family parakeet
```

Conversion:

```bash
uv run --project scripts/envs/parakeet \
  scripts/convert-parakeet.py nvidia/parakeet-tdt-0.6b-v2
```

Real-model smokes:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_PARAKEET_GGUF=models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2-F32.gguf \
  ctest --test-dir build --output-on-failure -R 'parakeet|encoder|decoder'
```

## Gaps

- Manifests record `hf_revision` but not local artifact hashes.
- Default CTest no longer has Parakeet source-tree numerical golden
  payloads; use `validate.py` for numerical comparison.
