# Model Porting

This is the high-level procedure for adding a new model family to
`transcribe.cpp`, starting from only a Hugging Face repo name.

The process is mostly linear. Numerical bring-up and benchmark work loop
back into conversion and graph implementation, but a port should still
move through the same named stages and produce the same named artifacts.

## Architecture Patterns

Speech models fall into different architectural patterns. Identifying
the pattern early shapes every downstream decision (graph structure,
KV cache strategy, dump points, decoding loop).

Common patterns:

- **Encoder + transducer** (Parakeet): conformer encoder, RNNT/TDT
  joint network. Single-shot decode, no autoregressive loop.
- **Encoder-decoder with cross-attention** (Cohere, Whisper): encoder
  produces hidden states, decoder attends to them via cross-attention.
  Autoregressive decoding with KV cache.
- **Audio-LLM / token injection** (Qwen3-ASR, Qwen2-Audio): audio
  encoder produces embeddings that replace placeholder tokens in a
  causal LM's input sequence. No cross-attention — the LM processes
  the fused audio+text sequence. Autoregressive decoding with KV
  cache. The decoder is architecturally a standard LLM; `llama.cpp`
  is a useful reference for the decoder side.
- **Encoder + CTC**: encoder-only with a CTC head. No decoder, no
  autoregressive loop.

The implementer should identify which pattern applies during Step 1
(research) and record it in the family note. For multimodal models
with independent components (e.g. audio encoder + text LM), the
encoder and decoder bring-up can proceed in parallel.

## Stages

1. Research the model and choose references.
2. Establish a reproducible reference run.
3. Define the artifact and golden plan.
4. Generate reference dumps.
5. Build the first accuracy GGUF.
6. Add the C++ loader and synthetic fixture smoke.
7. Bring up numerical validation.
8. Add real-model structural and end-to-end smokes.
9. Add benchmark coverage.
10. Add quantized variants.
11. Promote the family to supported.

The detailed testing contract is in
[`docs/model-family-testing.md`](../model-family-testing.md). This
porting guide covers the earlier decisions and artifact flow that happen
before those tests exist.

## Required Artifacts

Every family should end with:

- `docs/porting/families/<family>.md`
- `scripts/envs/<family>/pyproject.toml`
- converter support, normally `scripts/convert-<family>.py`
- reference dump support, ideally through a unified reference dumper
- `tests/tolerances/<family>.json`
- synthetic fixture support under `tests/fixtures/`
- `tests/<family>_smoke.cpp`
- `tests/<family>_real_smoke.cpp`
- `tests/<family>_e2e_smoke.cpp` or an equivalent legacy-named test
- bench matrix support in `scripts/bench/run.py`
- validate-family support for dump -> reference -> compare

Recommended local workspace setup is described in
[`7-workspace-setup.md`](7-workspace-setup.md). Manifests may provide
defaults for that layout, but paths must remain overridable with
environment variables.

## Naming

Use a stable family key everywhere:

```text
parakeet
cohere
<new-family>
```

Use that key in paths, manifests, CTest names, report names, tolerance
files, and environment names. If the upstream architecture string differs
from the repo family key, record both in the family note.

## Current Gaps

The existing Parakeet and Cohere ports contain the pieces, but not all of
the process is centralized:

- Reference lineage is documented in script comments, not in family notes.
- Parakeet has synthetic fixtures; Cohere does not yet have a synthetic
  fixture smoke.
- Reference dump scripts are split by family.
- Goldens are not driven by committed manifests.
- Full dump/compare validation is not yet one command.
- Benchmark comparison exists as a script, but the porting process does
  not yet require reference baseline rows and accuracy hashes.

New families should follow this guide rather than copying the historical
shape verbatim.
