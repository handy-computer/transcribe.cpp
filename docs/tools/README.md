# Tools

Reference documentation for each script and binary used during porting,
validation, and release. This is the "what does this tool do and how do
I invoke it" index. For the sequenced porting flow (which tool runs
when), see [`docs/porting/0-porting.md`](../porting/0-porting.md).

## Pipeline at a glance

```text
  HF / NeMo checkpoint
            │
            ▼ scripts/convert-<family>.py       (Python — conversion only)
            │   emits source/reference-dtype GGUF
            ▼
  models/<slug>/<slug>-<REF_DTYPE>.gguf     ← "accuracy GGUF". Reference quant.
            │
            ▼ tools/transcribe-quantize         (C++ — quantization only)
            │   reads any GGUF, walks tensors, requantizes per preset
            ▼
  models/<slug>/<slug>-Q4_K_M.gguf (and siblings)
```

The two stages are **strictly separated**:

- Python never produces a lossy quantized GGUF. Converters do not have
  a `--quant` flag; they preserve the source/reference dtype.
- `transcribe-quantize` never reads safetensors, never touches tokenizers,
  never writes metadata from scratch. It only requantizes an existing
  GGUF.

This split mirrors `llama.cpp`'s `convert_hf_to_gguf.py` →
`llama-quantize` pipeline. See the per-tool docs for details.

## Conversion & quantization

- [**conversion.md**](conversion.md) — `scripts/convert-<family>.py`.
  Checkpoint → GGUF at full fp precision. Adding a new family entry point.
- [**quantization.md**](quantization.md) — `transcribe-quantize` C++
  binary. GGUF → quantized GGUF. Preset table, per-tensor policy,
  bucket rules, filename convention.

## Numerical validation

- [**reference-dumps.md**](reference-dumps.md) — `dump_reference*.py`.
  Generate per-stage tensor dumps from the upstream reference model.
- [**validate.md**](validate.md) — `validate.py`. Orchestrator that
  runs reference → C++ → compare end-to-end for a family.
- [**compare-tensors.md**](compare-tensors.md) — `compare_tensors.py`.
  Symmetric diff between a C++ dump dir and a reference dump dir with
  per-tensor tolerances.

## Performance & accuracy

- [**benchmarking.md**](benchmarking.md) — `scripts/bench/{run,compare}.py`.
  Perf matrices across (backend, family, quant, sample).
- [**wer.md**](wer.md) — `scripts/wer/{run,score,compare,ingest}.py`.
  Hypothesis generation + WER scoring against a reference corpus.
- **Quant diagnostics** — `scripts/quant_accuracy.py`. Optional
  activation-drift inspection between a baseline GGUF and a quantized
  sibling. Not a quant acceptance gate; use CLI output-validity plus WER.

## Other

- [**environment-variables.md**](../environment-variables.md) — the single
  reference for every env var the library, tests, and tooling recognize, split
  into Tier 1 runtime config, Tier 2 validation hooks (build-gated), and
  test/tooling.
- `scripts/envs/<family>/pyproject.toml` — per-family `uv` env. Each
  converter and reference dumper has its own env because NeMo and
  Transformers have conflicting dependency graphs.
