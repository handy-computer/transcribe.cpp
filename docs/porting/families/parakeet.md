# Parakeet

Status: supported with historical validation shape; NeMo canonical reference
wired up, needs first run and bridge validation against MLX.

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
- Cross-check reference: **parakeet-mlx** via `scripts/dump_reference.py`.
  Historical MLX-based dumps used during initial port. Apple Silicon only.
- Legacy frontend reference: NeMo ONNX (`nemo128.onnx`) via
  `scripts/dump_reference.py mel-onnx`.

Bridge validation to establish:

- NeMo mel vs C++ mel (fp64 STFT in C++ vs NeMo's torch STFT).
- NeMo mel vs MLX mel (periodic-Hann vs symmetric-Hann known divergence).
- NeMo encoder vs MLX encoder (block-by-block comparison).
- NeMo decoder first-step vs MLX decoder first-step.
- NeMo full transcript vs C++ full transcript.

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

NeMo mel reference:

```bash
uv run --project scripts/envs/parakeet \
  scripts/dump_reference_parakeet_nemo.py mel \
  --model nvidia/parakeet-tdt-0.6b-v2 \
  --audio samples/jfk.wav \
  --out build/validate/parakeet/parakeet-tdt-0.6b-v2/jfk/mel/ref
```

NeMo encoder reference:

```bash
uv run --project scripts/envs/parakeet \
  scripts/dump_reference_parakeet_nemo.py encoder \
  --model nvidia/parakeet-tdt-0.6b-v2 \
  --audio samples/jfk.wav \
  --out build/validate/parakeet/parakeet-tdt-0.6b-v2/jfk/encoder/ref
```

NeMo decode reference (encoder + decoder + joint + transcript):

```bash
uv run --project scripts/envs/parakeet \
  scripts/dump_reference_parakeet_nemo.py decode \
  --model nvidia/parakeet-tdt-0.6b-v2 \
  --audio samples/jfk.wav \
  --out build/validate/parakeet/parakeet-tdt-0.6b-v2/jfk/decode/ref
```

MLX encoder reference (legacy, Apple Silicon only):

```bash
uv run scripts/dump_reference.py encoder \
  --model <path-to-parakeet-mlx-model> \
  --audio samples/jfk.wav \
  --out build/validate/parakeet/jfk/ref
```

Bridge validation (NeMo vs MLX):

```bash
uv run scripts/compare_tensors.py \
  build/validate/parakeet/parakeet-tdt-0.6b-v2/jfk/encoder/ref \
  build/validate/parakeet/jfk/ref \
  --tolerances tests/tolerances/parakeet.json
```

Conversion:

```bash
uv run scripts/convert-parakeet.py \
  <path-to-parakeet-tdt-0.6b-v2-mlx> \
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

- NeMo dump script written but not yet run — first execution will
  validate that NeMo installs cleanly and produces sane outputs.
- Bridge validation (NeMo vs MLX) not yet performed. Need to run both
  dump scripts on jfk.wav and compare with compare_tensors.py.
- HF snapshot revision not recorded in manifest.
- `frontend_smoke` currently assumes a source-tree `.f32` payload exists.
- `encoder_smoke` is a historical C++ numerical gate; future families
  should use validate-family dump comparison instead of cloning it.
- Converter reads MLX-format safetensors; if NeMo reference reveals
  weight layout differences, converter may need updating.
