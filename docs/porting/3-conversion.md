# Conversion

The first GGUF for a family is an accuracy artifact. It should preserve
the reference tensor semantics before it optimizes size.

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
- quantization preset and dtype policy
- tensor count
- skipped tensors, tied tensors, and fused tensors

Target report path:

```text
reports/convert/<family>/<variant>.<quant>.json
```

## Shared Code

Conversion is not always simple. The family-specific part is usually the
hparams and tensor catalog mapping. The following should be shared over
time:

- GGUF metadata writing helpers
- tokenizer extraction helpers
- manifest writing
- tensor classification by bucket
- dtype and quant preset validation
- file hashing
- common layout assertions

Family-specific converter code should remain responsible for:

- reading the upstream config
- mapping upstream tensor names to canonical GGUF names
- performing required layout transforms
- identifying tied, skipped, or fused tensors

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

1. Accuracy GGUF: `f32`, `bf16`, or source-preserving equivalent.
2. `f16` if it is meaningful for the source family.
3. `q8_0` as the first quantized accuracy/perf tradeoff.
4. Shipping candidates such as `q5_k_m` and `q4_k_m` only after tensor
   drift, WER, and benchmark data are recorded.

Do not enable a quant preset just because ggml can represent it. Each
enabled preset needs:

- converter support
- loader dtype support
- numerical validation result
- WER or transcript accuracy result
- benchmark result

Conv, norm, bias, frontend, positional, and embedding tensors may need
family-specific bucket rules. Document those rules in the converter and
family note.

