# Cohere ASR

Status: supported with historical validation shape; has a manifest-driven
native Transformers dump path for Cohere goldens; still needs tolerance
acceptance against C++ and synthetic fixture smoke.

## Identity

- Family key: `cohere`
- Upstream architecture string: `cohere_asr`
- Current source directory shape: Hugging Face-style
  `cohere-transcribe-03-2026/`
- Variant: `cohere-transcribe-03-2026`

## References

- Canonical reference: native Hugging Face Transformers
  (`trust_remote_code=False`) from a local Transformers checkout.
- Instrumented reference: `scripts/dump_reference_cohere_transformers.py`
  run through `scripts/envs/cohere`; it must keep
  `trust_remote_code=False`.
- Bridge reference: `mlx-audio`, still available via
  `scripts/dump_reference_cohere.py` for cross-checking.
- Initial manifest:
  `tests/golden/cohere/cohere-transcribe-03-2026.manifest.json`.

Do not use the Cohere remote-code path for goldens. Hugging Face
discussion #28 records garbage generations on that path and Cohere
recommends the native Transformers path going forward:
<https://huggingface.co/CohereLabs/cohere-transcribe-03-2026/discussions/28>.

Bridge validation to document:

- Transformers vs C++ frontend, encoder, encoder-decoder projection, and
  decoder prompt logits.
- Native Transformers currently emits a 10-token prompt for English
  punctuation mode: `[13764, 7, 4, 16, 62, 62, 5, 9, 11, 13]`.
  C++ currently dumps a 9-token prompt, so decoder tensor comparison is
  blocked until prompt policy is aligned or explicitly sliced.
- Transformers vs `mlx-audio` drift. The frontend currently differs in
  layout (`[n_mels, T]` for Transformers/C++, `[T, n_mels]` in the
  historical MLX dumper) and is not byte-identical after transposition.
- Backend-specific drift for CPU and accelerators.

## Current Commands

Reference encoder dump:

```bash
uv run scripts/golden.py plan \
  --manifest tests/golden/cohere/cohere-transcribe-03-2026.manifest.json
```

Generate reference dumps:

```bash
uv run scripts/golden.py generate \
  --manifest tests/golden/cohere/cohere-transcribe-03-2026.manifest.json
```

Direct Transformers encoder dump:

```bash
uv run --project scripts/envs/cohere \
  scripts/dump_reference_cohere_transformers.py encoder \
  --model ../models/cohere-transcribe-03-2026 \
  --transformers-ref ../refs/huggingface/transformers \
  --audio samples/jfk.wav \
  --out build/validate/cohere/cohere-transcribe-03-2026/jfk/encoder/ref \
  --model-dtype f32 \
  --torch-threads 1
```

Direct Transformers decode + transcript dump:

```bash
uv run --project scripts/envs/cohere \
  scripts/dump_reference_cohere_transformers.py decode \
  --model ../models/cohere-transcribe-03-2026 \
  --transformers-ref ../refs/huggingface/transformers \
  --audio samples/jfk.wav \
  --out build/validate/cohere/cohere-transcribe-03-2026/jfk/decode/ref \
  --language en \
  --max-new-tokens 256 \
  --model-dtype f32 \
  --torch-threads 1
```

Conversion:

```bash
uv run scripts/convert-cohere.py \
  <path-to-cohere-transcribe-03-2026> \
  models/cohere/cohere.f16.gguf \
  --quant f16
```

Real-model smokes:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

TRANSCRIBE_COHERE_MODEL=models/cohere/cohere.f16.gguf \
  ctest --test-dir build --output-on-failure -R cohere
```

## Gaps

- No synthetic fixture smoke for Cohere.
- Manifest exists, but its Hugging Face snapshot revision still needs to
  be filled in.
- C++ comparison against the native Transformers encoder dump is close
  but the early-layer mean tolerances need an explicit acceptance pass.
- Decoder tensor comparison against native Transformers is blocked by
  the prompt-length mismatch described above.
- Reference hardware should still be captured for benchmark reports.
- `dump_reference_cohere.py` duplicates dump-writing logic from
  `dump_reference.py`.
