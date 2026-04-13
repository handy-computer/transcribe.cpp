# Reference Research

Start with the Hugging Face repo name and produce a family note:

```text
docs/porting/families/<family>.md
```

The note is the audit trail for the port. It should be good enough that
another developer can reproduce the reference run, conversion, dumps, and
benchmarks after the port is complete.

## Research Packet

Record:

- Hugging Face repo, revision, license, and expected files.
- Model family key and upstream architecture string.
- First-party implementation and revision.
- Instrumentable implementation and revision.
- Third-party implementations checked.
- Tokenizer type and special token ids.
- Architecture pattern (see `0-porting.md`): encoder-decoder,
  audio-LLM, encoder+CTC, encoder+transducer.
- Encoder-decoder connection: cross-attention, token injection,
  joint network, or none (CTC).
- Frontend contract: sample rate, channel handling, and whatever
  preprocessing the model requires. Common items include window
  type, mel/filterbank parameters, normalization, dither, and
  pre-emphasis, but the implementer should document whatever their
  model actually does.
- Encoder and decoder architecture summary.
- Output contract: transcript only, segment/word/token timestamps,
  language selection, task selection, streaming support.
- Known unsupported features for the first port.
- Hardware and software used for reference runs.

### Decisions for Implementation

During research, identify decisions that will require human judgment
during implementation. Don't resolve them yet — just record them.
Examples:

- Semantic op mapping between reference and ggml (especially for
  custom attention patterns, positional encodings, or activation
  functions not directly available in ggml).
- Whether to simplify a component when the reference has multiple
  code paths (e.g. chunked vs full-sequence encoder processing,
  multimodal RoPE vs standard RoPE).
- When the decoder is architecturally a standard LLM (GQA, RoPE,
  SwiGLU), whether and how to leverage `llama.cpp` patterns.
- Prompt template and special token construction.
- What to share vs implement family-specific (e.g. Whisper mel
  frontend is shared across multiple model families).

Minimum hardware fields:

```text
host:
os:
cpu:
ram:
gpu:
accelerator_backend:
driver_or_runtime:
python:
reference_package_versions:
```

## Reference Roles

Use three labels, even if two labels point at the same implementation.

- **Canonical reference**: the implementation closest to the model
  publisher. This is the semantic source of truth.
- **Instrumented reference**: the implementation used to dump tensors.
  This may be MLX or another third-party implementation if it is easier
  to instrument.
- **Cross-check reference**: any additional implementation used to catch
  mistakes in the first two.

Policy:

- Prefer first-party or publisher-supported code as the canonical
  reference.
- It is acceptable to use MLX or another third-party implementation as
  the instrumented reference, but only after bridge validation.
- Do not treat "easy to instrument" as equivalent to "canonical".

## Bridge Validation

When canonical and instrumented references differ, prove the bridge.
Record the result in the family note.

At minimum:

- Run the same audio through both references.
- Compare normalized transcript text.
- Compare token ids if both expose them.
- Compare frontend output if both expose it.
- Compare logits or selected intermediate tensors if practical.
- Record any dtype or frontend differences that explain expected drift.

For existing families:

- Parakeet currently uses NeMo/ONNX for the frontend golden and
  `parakeet-mlx` for encoder/decoder dumps. The family note should make
  that split explicit.
- Cohere now uses native Transformers (`trust_remote_code=False`) for
  manifest-driven dumps and keeps `mlx-audio` as a bridge reference. The
  family note should record drift found between those references,
  including frontend layout differences.

## Reference Run Gate

Before writing converter or C++ code, create a reproducible reference run.
The run should produce:

- transcript text
- normalized transcript text
- token ids when available
- timing summary
- command line
- package versions
- model and audio hashes
- hardware summary

The target shape is a command like:

```bash
uv run scripts/reference/run.py \
  --family <family> \
  --model <reference-model-dir> \
  --audio samples/jfk.wav \
  --out reports/reference/<family>/jfk.json
```

If the unified runner does not exist yet, use the current family script
or a temporary script, but put the command and output path in the family
note.
