# Workspace Setup

Porting work is easier when the repo, model snapshots, and reference
implementations have predictable locations. Contributors may use any
layout, but the recommended sandbox layout is:

```text
transcribe/
  transcribe.cpp/                 # this repo
  models/                         # Hugging Face or publisher model snapshots
    cohere-transcribe-03-2026/
    parakeet-tdt-0.6b-v2/
  refs/                           # cloned reference implementations
    huggingface/transformers/
    NVIDIA-NeMo/NeMo/
```

The committed manifests should not require this exact layout. Current
validation manifests use model ids by default; use CLI overrides for
local paths:

```bash
uv run scripts/validate.py ref --family cohere \
  --model /path/to/cohere-transcribe-03-2026

uv run scripts/validate.py cpp --family cohere \
  --gguf /path/to/cohere.bf16.gguf

export TRANSCRIBE_ARTIFACT_CACHE=~/Library/Caches/transcribe.cpp
```

Families may still use environment variables in their own tools or smoke
tests, but `validate.py` should keep the common path override surface to
`--model` and `--gguf`.

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
