# Workspace Setup

Porting work is easier when the repo, model snapshots, and reference
implementations have predictable locations. Contributors may use any
layout, but the recommended sandbox layout is:

```text
transcribe/
  transcribe.cpp/                 # this repo
  models/                         # Hugging Face or publisher model snapshots
    cohere-transcribe-03-2026/
    parakeet-tdt-0.6b-v2-mlx/
  refs/                           # cloned reference implementations
    huggingface/transformers/
    NVIDIA-NeMo/NeMo/
    models/parakeet/parakeet-mlx/
```

The committed manifests should not require this exact layout. Use env
vars to override local paths:

```bash
export TRANSCRIBE_COHERE_HF_MODEL=/path/to/cohere-transcribe-03-2026
export TRANSCRIBE_TRANSFORMERS_REF=/path/to/transformers
export TRANSCRIBE_COHERE_MODEL=/path/to/cohere.f16.gguf
export TRANSCRIBE_ARTIFACT_CACHE=~/Library/Caches/transcribe.cpp
```

The default manifest values may point at the recommended sandbox layout,
but every path must be overridable.

## Environment Directories

Each family should have a dependency environment description:

```text
scripts/envs/<family>/pyproject.toml
```

Use it for reference implementations and dump generation. Do not rely on
a developer's global Python environment.

## Local Payloads

Large files stay local or in an artifact cache:

- Hugging Face model snapshots
- GGUF conversions
- `.f32` tensor dumps
- WER datasets
- benchmark reports too large or machine-specific for review

The repo should commit only manifests, docs, fixture generators, tests,
and small source-controlled fixtures.

