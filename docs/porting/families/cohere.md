# Cohere ASR

Status: supported with manifest-driven numerical validation against
native Transformers. C++ CPU validation passes locally; Cohere still
needs a synthetic fixture smoke.

## Identity

- Family key: `cohere`
- Upstream architecture string: `cohere_asr`
- Current source directory shape: Hugging Face-style
  `cohere-transcribe-03-2026/`
- Variant: `cohere-transcribe-03-2026`

## References

- Canonical reference: native Hugging Face Transformers
  (`trust_remote_code=False`), either from `scripts/envs/cohere` or an
  optional local Transformers checkout.
- Instrumented reference: `scripts/dump_reference_cohere_transformers.py`
  run through `scripts/envs/cohere`; it must keep
  `trust_remote_code=False`.
- Initial manifest:
  `tests/golden/cohere/cohere-transcribe-03-2026.manifest.json`.

Do not use the Cohere remote-code path for goldens. Hugging Face
discussion #28 records garbage generations on that path and Cohere
recommends the native Transformers path going forward:
<https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/28>.

Validation status:

- Native Transformers and C++ both use the 10-token English punctuation
  prompt: `[13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]`.
- `uv run scripts/validate.py compare --family cohere` passes for CPU
  dumps generated from `models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf`.
- Backend-specific drift for CPU and accelerators.

## Current Commands

Full validation (reference dumps + C++ dumps + comparison):

```bash
uv run scripts/validate.py all --family cohere
```

Or step by step:

```bash
uv run scripts/validate.py ref     --family cohere
uv run scripts/validate.py cpp     --family cohere
uv run scripts/validate.py compare --family cohere
```

Conversion:

```bash
uv run scripts/convert-cohere.py \
  <path-to-cohere-transcribe-03-2026> \
  models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf
```

Real-model smokes:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_COHERE_MODEL=models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf \
  ctest --test-dir build --output-on-failure -R cohere
```

## Gaps

- No synthetic fixture smoke for Cohere.
- Manifest exists, but the v2 schema does not yet record Hugging Face
  snapshot revision or local artifact hashes.
- Reference hardware should still be captured for benchmark reports.
- `dump_reference_cohere.py` duplicates dump-writing logic from the
  Parakeet reference dumper.
