# Conversion

The first GGUF for a family is an accuracy artifact. It should preserve
the reference tensor semantics before it optimizes size.

> **Tool reference**: for the CLI, preset names, output naming, and
> shared-library layout, see [`../tools/conversion.md`](../tools/conversion.md).
> For the quantization stage that runs *after* conversion, see
> [`../tools/quantization.md`](../tools/quantization.md).

## Separation of concerns

Conversion and quantization are **separate tools** with separate
responsibilities:

- **Python converter** (`scripts/convert-<family>.py`) — reads the
  upstream format, applies layout transforms, embeds the tokenizer,
  writes GGUF metadata, and emits tensors in the source/reference dtype.
  It has no `--quant` flag.
- **C++ quantizer** (`tools/transcribe-quantize`) — reads any GGUF,
  requantizes every tensor per a declarative preset table, writes a
  new GGUF. This is the only place block-quantized tensors are
  produced.

This mirrors `llama.cpp`'s `convert_hf_to_gguf.py` → `llama-quantize`
split. Python gets the ecosystem it's good at (safetensors, tokenizers,
format glue); C++ owns numerics end-to-end via `ggml_quantize_chunk`.

Quant preset names match `llama.cpp` exactly (`F16`, `Q8_0`, `Q6_K`,
`Q5_K_M`, `Q4_K_M`, …), and filenames follow the
`<slug>-<preset>.gguf` convention (e.g.
`parakeet-tdt-0.6b-v3-Q4_K_M.gguf`).

## Accuracy GGUF First

The reference dump script drives the numerics. The converter follows.

For the first bring-up:

- **Match the reference dtype exactly.** Run the reference dump script
  first and record what dtype it loaded (it writes `model_dtype` in the
  sidecar metadata). The first GGUF must use that dtype. If the
  reference loaded bf16, the first GGUF is bf16. If f32, then f32.
  A dtype mismatch between the reference and the GGUF means the C++
  inference path operates at a different precision than the reference,
  making tolerances absorb a hidden gap instead of genuine numerical
  drift.
- Use source-preserving dtype: do not upcast or downcast from the
  reference dtype. If the source dtype must be rounded (e.g. a bf16
  model converted to f16 because the loader does not support bf16),
  document it in the family note and run a second validation pass
  against that format.
- Keep tensor names, layouts, shapes, frontend metadata, tokenizer
  metadata, and hparams aligned with the reference dump conventions.

The first GGUF must be suitable for tensor-by-tensor numerical
comparison against the reference implementation. Do not start from a
quantized GGUF.

## Converter Manifest

Each conversion should write or print enough information to reconstruct:

- source model repo and revision
- source file paths and sha256 hashes
- converter command and converter revision
- output GGUF path and sha256
- family key, architecture string, and variant string
- source/reference dtype policy
- tensor count
- skipped tensors, tied tensors, and fused tensors

Target report path:

```text
reports/convert/<family>/<variant>-<dtype>.json
```

## Shared Code

Conversion is not always simple. The family-specific part is usually the
hparams and tensor catalog mapping. Shared logic lives in
`scripts/lib/` (plain importable module, not an installable package —
each per-family `uv` env adds `scripts/lib/` to `sys.path`):

- `gguf_common.py` — GGUF identity/KV helpers (`add_general_identity`),
  output-name derivation (`slug_from_repo_id`, `gguf_name`),
  reference-dtype routing + fp32/f16/bf16 `encode_for_gguf()`,
  special-token id helper (`safe_id`), and frontend-normalize
  canonicalization (`canonicalize_normalize`).
- `quant_policy.py` — preset name registry (names only; quantization
  math lives in `transcribe-quantize`).

Manifest writing, file hashing, HF snapshot resolution, sharded
safetensors reading, and tensor-name canonicalization are **currently
duplicated per-converter**, not shared — candidates for extraction into
`scripts/lib/`. The "Converter Manifest" section above describes the
target contract, not a shared implementation that exists today.

Family-specific converter code remains responsible for:

- reading the upstream config
- mapping upstream tensor names to canonical GGUF names
- performing required layout transforms
- identifying tied, skipped, or fused tensors

There is **no `Model` base class** and no auto-dispatch registry. Two
families of structurally different architectures (enc-dec vs
transducer) don't justify the inheritance cost. Revisit when we have
5+ families of the same shape. Until then, each `scripts/convert-<family>.py`
is a single readable top-to-bottom script that imports from
`scripts/lib/`.

## Tied Weights

When the model ties weights (e.g. `lm_head` shares `embed_tokens`),
store only one copy in the GGUF. The loader should try the output
tensor name first, and fall back to the embedding tensor if absent.
This is the same pattern `llama.cpp` uses: the converter omits
`output.weight` when tied, and the loader calls `create_tensor` with
`TENSOR_DUPLICATED` to reuse `token_embd.weight`.

For quantization, tied weights should use the output tensor's quant
type (typically higher quality, e.g. Q6_K in k-quant presets) since
the tensor serves double duty: embedding lookup (which needs row
access) and output projection (which needs matmul precision).

## Multi-Component Models

Models with architecturally distinct components (e.g. audio encoder
+ text decoder) may need different quantization bucket rules per
component. The audio encoder might use LayerNorm + bias while the
text decoder uses RMSNorm without bias. Conv2d weights need different
treatment than linear weights. Document per-component bucket rules in
the converter and family note.

## Quantization Policy

Required order:

1. Accuracy GGUF: `F32`, `BF16`, or source-preserving equivalent
   (produced by the Python converter).
2. `F16` if it is meaningful for the source family
   (`transcribe-quantize`).
3. `Q8_0` as the first quantized accuracy/perf tradeoff
   (`transcribe-quantize`).
4. Shipping candidates such as `Q5_K_M` and `Q4_K_M` only after tensor
   drift, WER, and benchmark data are recorded
   (`transcribe-quantize`).

Do not enable a quant preset just because ggml can represent it. Each
enabled preset needs:

- preset entry in `tools/transcribe-quantize/main.cpp`
- loader dtype support (via `transcribe::weights::kQuantLinearTypes` /
  `kQuantConvTypes` — every family accepts the full allowlist)
- numerical validation result (`scripts/validate.py`)
- WER or transcript accuracy result (`scripts/wer/`)
- benchmark result (`scripts/bench/run.py`)

Conv, norm, bias, frontend, positional, and embedding tensors may need
family-specific bucket rules. Per-family overrides live as scoped
entries in the C++ preset table, not as generalized pattern rules —
see the Cohere tied-embedding bump in
[`../tools/quantization.md`](../tools/quantization.md#per-family-policy-overrides).
