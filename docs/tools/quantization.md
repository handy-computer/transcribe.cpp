# Quantization

`transcribe-quantize` is the C++ binary that produces every lossy
quantized GGUF in this project. It reads an input GGUF (typically an
F32, F16, or BF16 conversion output), walks every tensor, and writes a
new GGUF where each tensor's dtype is chosen by a per-preset, per-bucket
policy.

Source: `tools/transcribe-quantize/`. CLI lives in `main.cpp`, the
classification / preset policy in `policy.cpp`. Built as
`build/bin/transcribe-quantize`.

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

Presets: `F16`, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q6_K`,
`Q5_K_M`, `Q4_K_M`. Names match `llama.cpp`'s `llama-quantize` exactly.

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
| `Embed`  | Tied token embedding (Cohere + Qwen3-ASR; doubles as output projection) | Bumped to Q6_K in `_M` presets                 |
| `ConvPw` | 1×1 pointwise conv kernels in conformer blocks                        | F16 (im2col+matmul supports F16 matmul)        |
| `Conv`   | Non-pointwise conv kernels: 2D pre-encode, depthwise                  | F32 or F16 (no quantized im2col in ggml)       |
| `Norm`   | Biases, LayerNorm/BatchNorm/RMSNorm weights, per-head q_norm/k_norm, positional bias/encoding, frontend buffers | F32 (precision-sensitive, tiny) |

Classification rules live in `classify_tensor()` in
`tools/transcribe-quantize/policy.cpp`. They are substring-based on the
canonical tensor name (e.g. `norm_*`, `*.bias`, `*pointwise*.weight`,
`*.q_norm.weight`, `*.ln_pre.weight`).

## Preset table

From `tools/transcribe-quantize/policy.cpp`:

| Preset    | Linear main | Linear fallback | attn output | Embed | Conv | ConvPw | Norm |
|-----------|-------------|-----------------|-------------|-------|------|--------|------|
| `F16`     | F16         | F16             | F16         | —     | F32  | F16    | F32  |
| `Q4_0`    | Q4_0        | F16             | Q4_0        | —     | F32  | F16    | F32  |
| `Q4_1`    | Q4_1        | F16             | Q4_1        | —     | F32  | F16    | F32  |
| `Q5_0`    | Q5_0        | F16             | Q5_0        | —     | F32  | F16    | F32  |
| `Q5_1`    | Q5_1        | F16             | Q5_1        | —     | F32  | F16    | F32  |
| `Q8_0`    | Q8_0        | F16             | Q8_0        | —     | F32  | F16    | F32  |
| `Q6_K`    | Q6_K        | F16             | Q6_K        | —     | F32  | F16    | F32  |
| `Q5_K_M`  | Q5_K        | F16             | Q8_0        | Q6_K  | F32  | F16    | F32  |
| `Q4_K_M`  | Q4_K        | F16             | Q8_0        | Q6_K  | F32  | F16    | F32  |

**`linear_fallback`** is the type used when a tensor's inner dim doesn't
divide the target quant's block size. Always something smaller-block or
blockless (F16 here). Block mismatches can happen on oddly-shaped
projections; the fallback keeps the tool from crashing.

**attn output** gets bumped in `_M` presets (to Q8_0), in the
spirit of `llama.cpp`'s `_M` presets — which also keep a few
precision-sensitive tensors above the base type. The specific tensor
choice differs: `llama.cpp`'s `Q4_K_M` / `Q5_K_M` bump `attn_v`,
`attn_qkv`, and `ffn_down` (to Q6_K, on select layers via
`use_more_bits`) and leave `attn_output` at the base type. Those
categories aren't named separately in transcribe's policy. The bump
matches three tensor suffixes — `attn.linear_out.weight` (Cohere,
Parakeet), `attn.out.weight` (Qwen3-ASR encoder), and `attn.o.weight`
(Qwen3-ASR decoder) — so all three families land in the same rule.

**Embed slot** is `—` (no override) for uniform presets. Those fall
back to `linear_main`. Legacy block quants (Q4_0/1, Q5_0/1) deliberately
leave both attn output and Embed at `linear_main` — they are uniform
accuracy/size tradeoffs, not mixed recipes.

## Per-family policy overrides

Two families tie the input embedding to the LM head and route it into
the `Embed` bucket so it bumps to Q6_K under the `_M` presets — without
the bump, WER regresses measurably:

- Cohere: `dec.embed.token.weight`
- Qwen3-ASR: `dec.token_embd.weight` (llama.cpp-style name)

Parakeet does not have a tied embedding. `pred.embed.weight` is a
small predictor-only embedding and rides the `Linear` bucket.

Each family override lives as an explicit name check in
`classify_tensor()`, **not** generalized into a `"*embed*" → bump`
rule. Generalizing invites false-positive bumps on Parakeet's predictor
embedding and any future family's small auxiliary embeddings. Revisit
when 3+ families share the same tensor pattern and the same policy.

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

Today's preset table covers F16, the legacy block quants (Q4_0, Q4_1,
Q5_0, Q5_1), Q8_0, Q6_K, Q5_K_M, and Q4_K_M. Planned additions:

- Q3_K_S, IQ2_XXS, IQ4_XS — require imatrix support (not yet wired).

## Running in bulk

`scripts/quant_accuracy.py` drives a multi-preset comparison against
an F32 baseline. See [`validate.md`](validate.md) and
[`compare-tensors.md`](compare-tensors.md) for the underlying diff
harness.
