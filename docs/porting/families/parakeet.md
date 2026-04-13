# Parakeet

Status: supported with manifest-driven numerical validation against the
NeMo canonical reference. C++ CPU validation passes locally.

## Identity

- Family key: `parakeet`
- Upstream architecture string: `parakeet`
- HF source repo: `nvidia/parakeet-tdt-0.6b-v2`
- Variants: `tdt-0.6b-v2`, `tdt-0.6b-v3`

## References

- Canonical reference: **NeMo** (`nvidia/parakeet-tdt-0.6b-v2` via
  `ASRModel.from_pretrained`). NeMo is NVIDIA's own implementation and the
  authoritative source for Parakeet TDT weights and inference behavior.
  Script: `scripts/dump_reference_parakeet_nemo.py`.
- Instrumented reference: **NeMo** (same script, using forward hooks to
  capture per-stage intermediates without modifying NeMo internals).
- Legacy frontend reference: NeMo ONNX (`nemo128.onnx`) via
  `scripts/dump_reference.py mel-onnx`.

Validation status:

- NeMo reference dumps and C++ CPU dumps pass via
  `uv run scripts/validate.py compare --family parakeet`.
- C++ and NeMo transcripts match exactly on `samples/jfk.wav`.

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
uv run scripts/convert-parakeet.py \
  <path-to-exported-parakeet-safetensors-dir> \
  models/parakeet/parakeet-tdt-0.6b-v2.f32.gguf
```

Real-model smokes:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_REAL_PARAKEET_GGUF=models/parakeet/parakeet-tdt-0.6b-v2.f32.gguf \
  ctest --test-dir build --output-on-failure -R 'parakeet|encoder|decoder'
```

## Gaps

- Manifest exists, but the v2 schema does not yet record Hugging Face
  snapshot revision or local artifact hashes.
- Default CTest no longer has Parakeet source-tree numerical golden
  payloads; use `validate.py` for numerical comparison.
- Converter still reads the historical exported safetensors layout,
  while the canonical numerical reference is NeMo.
