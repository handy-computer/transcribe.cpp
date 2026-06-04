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

The authoritative stage chain is the eight `porting-N` skills under
`.claude/skills/`:

```
1-intake → 2-oracle → 3-convert → 4-cpp → 5-quants → 6-bench → 7-wer → 8-ship
```

Each skill owns its postconditions and is the source of truth for what
must exist by the time you advance. Read the corresponding SKILL.md
before starting each stage.

The high-level intent of each stage:

1. **Intake** — research identity, capabilities; draft the family doc's
   `## Capability Validation` table; clear Preflight Gate A.
2. **Oracle** — run the reference framework on every manifest case,
   capture tensor dumps + transcripts and a provisional tolerances file
   sized from per-tensor magnitude (`1e-4 × p99_abs` / `1e-5 × rms`).
3. **Convert** — produce only the reference-dtype GGUF; clear Preflight
   Gate B.
4. **C++** — implement `src/arch/<family>/`, finalize tolerances, run
   the family-doc Capability Validation table, gate full ref-dtype WER
   vs the measured Oracle reference baseline.
5. **Quants** — generate the shipped quant matrix, CLI smoke each
   produced GGUF, and take a tentative quant WER read for human review.
6. **Bench** — performance matrix; every accepted iteration re-runs
   `validate.py all`.
7. **WER** — full release WER sweep; ref-dtype hard gate against the
   measured Oracle reference baseline, quants human-reviewed and not
   auto-gated.
8. **Ship** — checklist-driven local prep of the family doc, model card,
   HF YAML, HF README.

The detailed testing contract is in
[`docs/model-family-testing.md`](../model-family-testing.md).

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
- manifest support under `tests/golden/<family>/` so `scripts/validate.py`
  can run reference -> C++ -> compare

Numerical validation workflow is described in
[`4-numerical-validation.md`](4-numerical-validation.md). Common
numerical failure patterns and fixes are collected in
[`4a-numerical-troubleshooting.md`](4a-numerical-troubleshooting.md).

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

The existing Parakeet and Cohere ports now have the central numerical
validation path in place:

```bash
uv run scripts/validate.py all --family parakeet
uv run scripts/validate.py all --family cohere
```

Remaining gaps are narrower:

- Reference dump scripts are split by family.
- Golden manifests are intentionally minimal v2 manifests and do not yet
  record full snapshot hashes or artifact cache keys.
- Benchmark comparison exists as a script, but the porting process does
  not yet require reference baseline rows and accuracy hashes.
- Default `ctest` no longer depends on source-tree numerical golden
  payloads; numerical payloads are generated under `build/validate/`.

New families should follow this guide rather than copying the historical
shape verbatim.
