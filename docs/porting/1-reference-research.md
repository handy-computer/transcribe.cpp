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

## Choosing Your Reference Framework

**The model card is the decision.** Read it first. The publisher names the
canonical implementation; that's the reference. Don't reason
from "STT usually ships via X" or from what previous ports used — the
canonical framework for this family might be one you haven't seen before.

The intake produces one unambiguous answer: what the publisher says is
authoritative, at what revision. Lock it in the golden manifest's
`reference.kind` / `reference.source` / `reference.revision` fields and
explain the choice in the family note. The publisher may not specify a
revision, in this case use the latest release of the library.

Rules:

- Pick exactly one reference per family. Locking it in the manifest makes
  "what was validated against what" unambiguous for future reviewers.
- Use what the publisher points at, even if unfamiliar. A new framework on
  a new port is normal; defaulting to a familiar one without reading the
  card is a smell.
- Prefer publisher-supported code over reverse-engineered. A third-party
  implementation becomes the canonical reference only when no
  publisher-supported path can be instrumented.
- Don't switch frameworks mid-port. If the chosen framework lags on a
  specific variant, document the limitation in the family note.

Examples (illustrative, not prescriptive — your family's card may point
somewhere else entirely):

- Parakeet: the NVIDIA card points at NeMo, so the reference is NeMo
  (`nvidia/parakeet-tdt-0.6b-v2` via `ASRModel.from_pretrained`).
- Cohere: the Cohere card points at native Transformers with
  `trust_remote_code=False`. The remote-code path produces garbage outputs
  and is explicitly not the reference.

## Reference Roles

Use three labels, even if two labels point at the same implementation.

- **Canonical reference**: the implementation closest to the model
  publisher. This is the semantic source of truth.
- **Instrumented reference**: the implementation used to dump tensors.
  Prefer the canonical implementation here too. If that is not feasible,
  document the exception before using another implementation.
- **Cross-check reference**: any additional implementation used to catch
  mistakes in the first two.

Policy:

- Prefer first-party or publisher-supported code as the canonical
  reference.
- Prefer the same first-party or publisher-supported code for tensor
  dumps.
- Do not treat "easy to instrument" as equivalent to "canonical".
- Third-party implementations are useful for implementation research and
  performance context, but they should not be the default source of
  numerical goldens when a canonical reference can be instrumented.

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

- Parakeet now uses NeMo as both canonical and instrumented reference for
  manifest-driven validation.
- Cohere now uses native Transformers (`trust_remote_code=False`) as both
  canonical and instrumented reference for manifest-driven validation.

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

The current command shape is:

```bash
uv run scripts/validate.py ref --family <family>
```

Use `--model <reference-model-dir-or-id>` when the manifest's model value
is not the source you want to validate. The family note should record any
extra setup needed for the canonical reference environment.
