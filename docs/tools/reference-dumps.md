# Reference dumps

Reference dump scripts run the upstream Python / MLX / NeMo reference
implementation on a sample audio file and write per-stage tensors to
disk in the shared `*.f32` + `*.json` format. These are compared
tensor-by-tensor against the C++ implementation's dumps to catch
numerical drift.

## Family scripts

| Script                                       | Reference backend              | Platform         |
|----------------------------------------------|--------------------------------|------------------|
| `scripts/dump_reference.py`                  | parakeet-mlx + ONNX preprocessor | macOS arm64      |
| `scripts/dump_reference_parakeet_nemo.py`    | NeMo (PyTorch)                 | Linux / macOS    |
| `scripts/dump_reference_cohere.py`           | Cohere MLX                     | macOS arm64      |
| `scripts/dump_reference_cohere_transformers.py` | HuggingFace Transformers    | Linux / macOS    |

`validate.py` picks the right one via the `_<reference>` suffix in the
filename; the manifest's `reference` field selects the backend.

## On-disk format

Every dumped tensor is two files:

```text
<name>.f32    raw little-endian float32, row-major (C order)
<name>.json   { "shape": [...], "dtype": "float32", "stage": "...", "source": "..." }
```

The C++ debug dumper in `src/transcribe-debug.{h,cpp}` (gated on
`TRANSCRIBE_DUMP_DIR`) writes the same format. This symmetry is what
lets `compare_tensors.py` diff the two dirs without translation.

## Typical invocation

```bash
uv run --directory scripts/envs/parakeet \
  ../dump_reference.py encoder \
    --model ~/sandboxes/transcribe/models/parakeet-tdt-0.6b-v2-mlx \
    --audio samples/jfk.wav \
    --out build/validate/parakeet/v2/jfk/ref
```

In practice you invoke `validate.py ref --family parakeet` and it
selects the env, script, and paths for you.

## Environments

Each reference backend has its own `uv` env under
`scripts/envs/<family>/`. NeMo pins Python <3.13 on macOS (kaldialign
wheel gap); MLX is arm64-only. The env boundaries exist because
installing all three reference stacks in one env doesn't resolve.

## Contract tensors

A tensor is "contract" when it has an entry in
`tests/tolerances/<family>.json`. Contract tensors must be present on
both sides and within tolerance for `validate compare` to pass. Any
other tensor dumped by the C++ side (debug-only) is reported if missing
from the reference but does not fail the run.

Add a tensor to the contract when you want to pin its numerical
behavior.
