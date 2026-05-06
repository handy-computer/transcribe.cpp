# Numerical Troubleshooting

This page collects common numerical validation failures seen while
porting speech and audio-language models to `transcribe.cpp`. The main
validation contract is in
[`4-numerical-validation.md`](4-numerical-validation.md); this page is
for diagnosis once a tensor exceeds tolerance or appears with the wrong
shape.

Start with the first failing tensor. Later tensors usually inherit the
earliest divergence, so large downstream errors are often symptoms, not
the root cause. Error magnitude alone is not diagnostic: read both the
reference implementation and the C++ graph at the divergent stage.

## Frontend And Mel/STFT

Mel spectrogram and raw-feature mismatches must be fixed before encoder
work continues. Every downstream tensor depends on frame count, frame
alignment, window values, and normalization.

### Mel off-by-one

**Symptoms:** mel or raw-feature output shape differs by one frame, or
the shapes match but values are shifted by one hop.

**Common causes:** `center=True` vs `center=False`, reflect vs zero vs
constant padding, final partial frame kept vs dropped, frame count using
floor vs ceil, sample-rate mismatch, or a different trim/pad policy.

**Fix:** trace the STFT frame calculation in both implementations:
input sample count, pad amount, window length, hop length, frame count,
and final-frame handling. Do not compensate with wider encoder
tolerances; fix the frontend contract.

### Window and filterbank mismatch

**Symptoms:** shapes match, but mel values differ consistently across
many frames or frequency bins.

**Common causes:** periodic vs symmetric Hann, Hamming vs Hann, Slaney
vs HTK mel scale, different filterbank normalization, missing
preemphasis, inference dither present on one side only, or biased vs
unbiased normalization variance.

**Fix:** inspect the reference feature extractor source, not just the
config. If the converter bakes windows or filterbanks into GGUF, bake
the exact values used by the reference inference path.

### Padding boundary artifacts

**Symptoms:** one or a few final conv-subsampled frames have much larger
diffs than the interior frames.

**Common causes:** boundary handling differences between framework conv
implementations, for example ggml im2col versus PyTorch native conv.

**Fix:** verify the interior frames are tight and document the boundary
profile in `tests/tolerances/<family>.json`. A known final-frame artifact
can be tolerated, but a frame-count or frame-shift bug cannot.

## Dtype And Accumulation

### Export/reference dtype mismatch

**Symptoms:** every stage is close but nonzero, often with error that
slowly accumulates through depth.

**Common causes:** comparing a bf16 or f16 GGUF against an f32 reference,
or accidentally promoting/demoting weights during conversion.

**Fix:** the first accuracy GGUF should use the reference's natural
inference dtype. If the production export intentionally uses lower
precision, document that precision target in the tolerance file and keep
output-adjacent gates tight enough to catch real bugs.

### Accumulation-order drift

**Symptoms:** mean error is low, max error may be higher in matmul or
conv-heavy regions, and normalized output gates become stable again.

**Common causes:** different GEMM, convolution, or attention kernels;
bf16/f16 accumulation order; fused kernels versus unfused kernels.

**Fix:** classify this only after ruling out layout and op mismatches.
Use observed CPU data to set tolerances and document where the drift
attenuates.

### Formula mismatch mistaken for precision

**Symptoms:** errors are small enough to look numerical but persist
systematically at a specific op.

**Common causes:** wrong norm epsilon, RMSNorm vs LayerNorm, biased vs
unbiased variance, wrong log base, wrong activation approximation, or an
unexpected fp32 upcast in the reference.

**Fix:** treat this as an implementation mismatch, not harmless precision
noise. Match constants, formulas, and intermediate dtypes exactly.

## Layout And Shape

### Reused scheduler buffers in C++ dumps

**Symptoms:** multiple unrelated C++ dump tensors are byte-exact equal,
early/middle layer tolerances are enormous while final output gates look
tight, or an intermediate's activation range looks like a later tensor.

**Common cause:** the family graph builder saved intermediate tensor
pointers for dumping after `graph_compute`, but did not mark those
tensors as graph outputs. The ggml scheduler reused their buffers for
later operations.

**Fix:** call `transcribe::debug::mark_tensor_for_dump()` on every
contract dump tensor while building the graph, before scheduler graph
allocation. Keep this gated through the helper rather than calling
`ggml_set_output()` unconditionally, so normal inference does not carry
extra live tensors.

### Layout mismatch

**Symptoms:** large errors with structure. Expected values may appear in
the output but at wrong positions; rows, columns, channels, or heads look
scrambled.

**Common causes:** transposed weight, wrong reshape axis, conv kernel
layout mismatch, head-splitting axis flipped, matmul operand order
wrong, or a converter-side permutation bug.

**Fix:** inspect the first tensor that consumes the affected weight. A
layout error may live in the converter even when it first appears during
C++ graph execution.

### Broadcast or stride mismatch

**Symptoms:** one dimension appears repeated, a bias is applied along the
wrong axis, or the flat comparison has a repeating error pattern.

**Common causes:** different row-major interpretation, implicit
broadcasting in PyTorch not mirrored in ggml, wrong view/contiguous
assumption, or a shape encoded in the wrong slow-to-fast order.

**Fix:** dump shape metadata on both sides and verify the semantic axes,
not only the element count.

## Graph And Op Mapping

### Op mismatch

**Symptoms:** upstream tensors are clean, then values become completely
wrong at one stage.

**Common causes:** wrong ggml op, missing bias, missing scale factor,
wrong activation, wrong division order, wrong residual source, or layer
order swapped.

**Fix:** identify the exact reference op and map it to the equivalent
ggml expression with matching parameters.

### Residual or skip connection

**Symptoms:** values are close but systematically offset, and error grows
with depth.

**Common causes:** missing residual add, wrong conformer macaron `0.5`
scale, residual applied before vs after norm, or the wrong tensor used as
the skip source.

**Fix:** trace every `+` in the reference block and verify the same
operands and scales in C++.

## Attention, Masks, And Positions

### Attention layout

**Symptoms:** attention output diverges with head-specific structure:
some heads may look clean while others are scrambled.

**Common causes:** Q/K/V projection layout mismatch, wrong head-splitting
axis, Q/K transpose order error, or wrong scale factor such as
`1/sqrt(d_model)` instead of `1/sqrt(d_head)`.

**Fix:** dump Q, K, V projections separately from attention scores and
attention output.

### RoPE rotation pattern (NORMAL vs NEOX)

**Symptoms:** encoder/decoder tensors upstream of MHSA match cleanly, but
the very first attention block output diverges by a large constant factor
(seen at ~16x on the first encoder block, growing exponentially with
depth). Disabling flash-attention, head-dim padding, or tweaking the
attention scale leaves the drift unchanged because the bug is orthogonal
to those.

**Common cause:** wrong `mode` argument to `ggml_rope_ext`. ggml has two
incompatible rotation pairings:

- `GGML_ROPE_TYPE_NORMAL` (=0) — interleaved / GPT-J style. Rotation
  pairs are `(0, 1), (2, 3), (4, 5), ...`. Reference's `rotate_half`
  slices `x[..., 0::2]` / `x[..., 1::2]`, and `apply_rotary_pos_emb`
  expands cos/sin via `repeat_interleave(2, dim=-1)`.
- `GGML_ROPE_TYPE_NEOX` (=2) — split-halves / GPT-NeoX style. Rotation
  pairs are `(0, D/2), (1, D/2+1), ...`. Reference's `rotate_half`
  slices `x[..., :D//2]` / `x[..., D//2:]`, and cos/sin are built via
  `cat((emb, emb), dim=-1)`.

The two modes produce numerically completely different attention even
though the rotation angles per position are identical.

**Fix:** before writing the C++ MHSA, open the reference's `rotate_half`
function (and the cos/sin assembly in `apply_rotary_pos_emb`) and pick
the matching mode. The rotate_half slicing pattern is the deciding
signal. `src/arch/moonshine/encoder.cpp` (interleaved → `NORMAL`) and
`src/arch/qwen3_asr/decoder.cpp` (split-halves → `NEOX`) are the
in-tree examples of the two modes. Different families inside the same
HF library make different choices — do not assume one based on another.

### Mask, length, or position mismatch

**Symptoms:** early decoder tensors are close, but attention output or
logits diverge sharply; errors may appear only at padded positions or
after a specific decode step.

**Common causes:** wrong causal mask, wrong cross-attention mask,
incorrect encoder lengths after subsampling, off-by-one position IDs,
RoPE offset error, relative-position bucket mismatch, or prompt length
not included in cache positions.

**Fix:** dump masks, position IDs, and effective lengths when possible.
For autoregressive models, compare prompt pass and first single-token
step separately.

### KV-cache mismatch

**Symptoms:** full prompt logits match, but incremental decoding diverges
after the first generated token.

**Common causes:** cache stored in the wrong layout, cache position
advanced incorrectly, prompt tokens reprocessed or skipped, cross-cache
recomputed differently, or attention mask not adjusted for cached tokens.

**Fix:** validate the no-cache prompt path first, then dump one-step
decode Q/K/V and cache slices.

## Tokenization And Prompts

### Tokenizer or prompt mismatch

**Symptoms:** embeddings are not near-exact, first decoder tensor is
wrong, or transcripts differ despite clean encoder tensors.

**Common causes:** different special token IDs, missing language/task
prompt, wrong BOS/EOS handling, tokenizer file drift, vocabulary order
changed during export, or a tied embedding/head weight mismatch.

**Fix:** dump token IDs and token embeddings. Embedding lookups should be
near-exact; if they are not, debug tokenizer and weight loading before
decoder layers.

### Audio token injection

**Symptoms:** audio-LLM decoder layers diverge immediately after the
fusion point.

**Common causes:** wrong placeholder token count, off-by-one audio length
calculation, injection at wrong positions, wrong mask positions, or audio
features inserted before the expected projection/norm.

**Fix:** dump the token embedding sequence before and after injection.
The post-injection tensor should contain audio features exactly at
placeholder positions and token embeddings everywhere else.

## Output Head

### log-softmax infinities

**Symptoms:** post-log-softmax tensors contain `-inf` or produce infinite
element-wise diffs, while raw logits are close.

**Common causes:** implementation computes `log(softmax(x))` and softmax
underflows, masked positions are carried through as infinities, or small
raw-logit differences move a very low-probability token across the
underflow boundary.

**Fix:** use pre-softmax logits as the numerical gate. Post-softmax or
log-prob tensors may be dumped for visibility, but if they use infinite
tolerance, they must not be the only output-head gate.

### Argmax or transcript mismatch with close logits

**Symptoms:** tensor diffs are within tolerance, but the selected token or
transcript differs.

**Common causes:** two logits are nearly tied, tie-breaking differs,
temperature/sampling is enabled in one path, timestamps or blank tokens
are handled differently, or beam/greedy settings differ.

**Fix:** compare raw logits at the decision step and record the top-k
tokens. Validation should use deterministic decoding.

## Semantic Unknowns

**Symptoms:** the C++ implementation faithfully mirrors the understood
reference path, but the reference uses a family-specific behavior not
covered by the current op mapping.

**Common causes:** custom positional encoding, non-standard masking,
streaming-state behavior, hidden prompt construction, special timestamp
rules, or framework hooks that alter inference behavior.

**Fix:** stop guessing. Capture the finding in the family note or porting
log, then decide whether to extend the graph, converter, or validation
contract.
