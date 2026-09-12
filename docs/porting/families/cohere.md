# Cohere ASR

Status: supported with manifest-driven numerical validation against
native Transformers. C++ CPU validation passes locally.

## Identity

- Family key: `cohere`
- Upstream architecture string: `cohere_asr`
- Current source directory shape: Hugging Face-style
  `cohere-transcribe-03-2026/`
- Variants: `cohere-transcribe-03-2026`,
  `cohere-transcribe-arabic-07-2026` (retrained Arabic-focused
  checkpoint, identical architecture and tensor layout; languages
  `[en, ar]`; config omits top-level `vocab_size` — the converter falls
  back to `head.num_classes`; upstream repo is gated)

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
uv run --project scripts/envs/cohere \
  scripts/convert-cohere.py CohereLabs/cohere-transcribe-03-2026
```

Real-model smokes:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_COHERE_GGUF=models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf \
  ctest --test-dir build --output-on-failure -R cohere
```

## Gaps

- Manifest records `hf_revision` but not local artifact hashes.
- Reference hardware should still be captured for benchmark reports.
- `cohere-transcribe-arabic-07-2026` has no per-variant golden tensor
  manifest, converter report, or bench run; it is validated end-to-end by
  WER parity against the native Transformers reference on FLEURS Arabic
  (C++ BF16 11.02% vs reference 11.00%, 428 utts, batch 8, L40S; see
  `reports/wer/cohere-transcribe-arabic-07-2026-*.fleurs-ar.b8.*`). The
  WER baseline runner is
  `scripts/wer/run_reference_cohere_transformers.py`.
