# Quantization

`transcribe-quantize` is the C++ binary that produces every lossy
quantized GGUF in this project. It reads an input GGUF (typically an
F32, F16, or BF16 conversion output), walks every tensor, and writes a
new GGUF where each tensor's dtype is chosen by a per-preset, per-bucket
policy.

Source: `tools/transcribe-quantize/main.cpp`. Built as `build/bin/transcribe-quantize`.

This is the only place lossy quantization happens in the project.
Python's `convert-<family>.py` never emits Q8_0, Q4_K_M, or any other
block-quantized type. See [`conversion.md`](conversion.md) for the
separation.

`transcribe-quantize` links `libggml` directly and calls
`ggml_quantize_chunk` — the same reference quantizers `llama-quantize`
uses. Everything the model might be quantized to in the future is
reachable without a Python gap.

## CLI

```bash
build/bin/transcribe-quantize INPUT.gguf OUTPUT.gguf --quant PRESET
```

Presets: `F16`, `Q8_0`, `Q6_K`, `Q5_K_M`, `Q4_K_M`. Names match
`llama.cpp`'s `llama-quantize` exactly.

Example:

```bash
build/bin/transcribe-quantize \
  models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3-F32.gguf \
  models/parakeet-tdt-0.6b-v3/parakeet-tdt-0.6b-v3-Q4_K_M.gguf \
  --quant Q4_K_M
```

## Output naming

`<slug>-<preset>.gguf` with the preset uppercase, matching `llama.cpp`:

```text
parakeet-tdt-0.6b-v3-Q4_K_M.gguf
parakeet-tdt-0.6b-v3-Q5_K_M.gguf
cohere-transcribe-03-2026-Q6_K.gguf
```

## How it works

```text
  input GGUF  ──▶  load into ggml_context (all tensors resident)
                     │
                     ▼
                   for each tensor:
                     classify(name) → Bucket
                     resolve(Bucket, preset) → target ggml_type
                     if src.type == target:   memcpy
                     else:                    dequant → fp32 → ggml_quantize_chunk
                     │
                     ▼
                   copy all KV (tokenizer, hparams, frontend) unchanged
                   override general.file_type with the new preset's tag
                   write output GGUF in one pass
```

Tensor classification lives here, not in the Python converters. The
converters only write source/reference-dtype GGUFs; this tool owns the
per-preset bucket policy for every canonical tensor name.

## Buckets

| Bucket   | What's in it                                                          | Quant behavior                                 |
|----------|-----------------------------------------------------------------------|------------------------------------------------|
| `Linear` | `ggml_mul_mat` operands: encoder FF, attention projections, predictor LSTM gates, joint projections, head | Block-quantized (Q4_K / Q5_K / Q6_K / Q8_0)    |
| `Embed`  | Tied token embedding (Cohere only; doubles as output projection)      | Bumped to Q6_K in `_M` presets                 |
| `ConvPw` | 1×1 pointwise conv kernels in conformer blocks                        | F16 (im2col+matmul supports F16 matmul)        |
| `Conv`   | Non-pointwise conv kernels: 2D pre-encode, depthwise                  | F32 or F16 (no quantized im2col in ggml)       |
| `Norm`   | Biases, LayerNorm/BatchNorm weight+bias, positional bias/encoding, frontend buffers | F32 (precision-sensitive, tiny)        |

Classification rules live in `classify_tensor()` in
`tools/transcribe-quantize/main.cpp`. They are substring-based on the
canonical tensor name (e.g. `norm_*`, `*.bias`, `*pointwise*.weight`,
`pred.embed.weight`).

## Preset table

From `tools/transcribe-quantize/main.cpp`:

| Preset    | Linear main | Linear fallback | attn.linear_out | Embed | Conv | ConvPw | Norm |
|-----------|-------------|-----------------|-----------------|-------|------|--------|------|
| `F16`     | F16         | F16             | F16             | —     | F32  | F16    | F32  |
| `Q8_0`    | Q8_0        | F16             | Q8_0            | —     | F32  | F16    | F32  |
| `Q6_K`    | Q6_K        | F16             | Q6_K            | —     | F32  | F16    | F32  |
| `Q5_K_M`  | Q5_K        | F16             | Q8_0            | Q6_K  | F32  | F16    | F32  |
| `Q4_K_M`  | Q4_K        | F16             | Q8_0            | Q6_K  | F32  | F16    | F32  |

**`linear_fallback`** is the type used when a tensor's inner dim doesn't
divide the target quant's block size. Always something smaller-block or
blockless (F16 here). Block mismatches can happen on oddly-shaped
projections; the fallback keeps the tool from crashing.

**`attn.linear_out`** gets bumped in `_M` presets (to Q8_0), matching
`llama.cpp`'s mixed-precision recipe for attention output projections.

**Embed slot** is `—` (no override) for uniform presets. Those fall
back to `linear_main`.

## Per-family policy overrides

Cohere has a tied token embedding (`dec.embed.token.weight` doubles as
the output projection). Under `Q4_K_M` / `Q5_K_M`, it is bumped to
Q6_K — without this, WER regresses measurably. This lives as a
family-scoped entry in the C++ policy table.

Parakeet does not have a tied embedding. `pred.embed.weight` is a
small predictor-only embedding and rides the `Linear` bucket.

Per-family overrides are deliberately kept in a Cohere section of the
policy table, **not** generalized into a `"*embed*" → bump` rule.
Generalizing from one family invites false-positive bumps on Parakeet's
predictor embedding and any future family's small auxiliary
embeddings. Revisit when we have 3+ families where the same tensor
pattern genuinely needs the same policy.

## Loader allowlist

Every family's loader (`src/arch/<family>/weights.cpp`) declares which
ggml types it will accept for each bucket. Post-unification, these are
shared constants:

- `transcribe::weights::kQuantLinearTypes` — F32, F16, BF16, Q4_0/1,
  Q5_0/1, Q8_0, Q4_K, Q5_K, Q6_K.
- `transcribe::weights::kQuantConvTypes` — F32, F16.

Every family must accept the full allowlist. A family that can't (e.g.
because a specific op is missing on a backend) is a bug, not a policy
difference.

## Presets roadmap

Today's preset table covers F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M. Legacy
quants (Q4_0/1, Q5_0/1) are accepted by the loader but not yet wired
into the preset table — Python used to emit them directly. Planned
additions:

- Q4_0, Q4_1, Q5_0, Q5_1 — declarative preset entries, no new code.
- Q3_K_S, IQ2_XXS, IQ4_XS — require imatrix support (not yet wired).

## Running in bulk

`scripts/quant_accuracy.py` drives a multi-preset comparison against
an F32 baseline. See [`validate.md`](validate.md) and
[`compare-tensors.md`](compare-tensors.md) for the underlying diff
harness.
