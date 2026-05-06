# GGML Reference Map

A curated index of *where to look* when you need to implement or debug
a ggml pattern during Stage 4 C++ bring-up of a new family.

This is **not** a ggml tutorial. It is not a replacement for the
per-family notes under [`families/`](families/) or for
[`4-numerical-validation.md`](4-numerical-validation.md). It answers
one question only: *"I need to do X in ggml — what is the closest
working example in this repo, or the closest upstream repo/file/search
token to inspect?"*

Workflow, stage exits, and bring-up procedure live in
[`agent-automation-plan.md`](agent-automation-plan.md) and the Stage 4
skills. This doc is only for ggml pattern lookup and debugging.

In-tree references use repo-relative `file:line` pointers. Upstream
references use portable `owner/repo :: file :: search token` pointers
so this doc does not depend on a particular local checkout layout. If a
function moved, re-grep the symbol or search token (e.g.
`rel_pos_mhsa`, `mha_self_cached`, `build_prefill_graph`) rather than
trusting an old range.

## Contents

- **[If implementing X, inspect Y](#if-implementing-x-inspect-y)** — lookup table: pattern name → closest in-tree example + upstream reference. Scan this first.
- **[Core patterns you will re-use](#core-patterns-you-will-re-use)** — prose explanations of the recurring patterns:
  - Mel / frontend
  - Conv / subsampling
  - Norms (LayerNorm, RMSNorm, fused BatchNorm)
  - Transformer self-attention + KV cache
  - GQA / RoPE
  - Encoder-decoder bridge vs. audio-token injection
  - Tied weights
  - Dump preservation / scheduler output marking
  - Graph build and compute driver
  - Weight loading + shape validation
  - Quantization policy touchpoints
- **[Common failure modes and where to look first](#common-failure-modes-and-where-to-look-first)** — symptom → most likely cause + where to dump next.

## "If implementing X, inspect Y"

| Need | Closest working example | Upstream reference |
|---|---|---|
| Log-mel frontend | `src/transcribe-mel.cpp:372` (`MelFrontend` ctor) + `src/transcribe-mel.cpp:409` (`compute`) | `ggml-org/whisper.cpp :: src/whisper.cpp :: search "log_mel_spectrogram"` |
| Conv subsampling (conformer stem) | `src/conformer/conformer.cpp:706` (`build_pre_encode`) + `src/arch/parakeet/encoder.cpp:247-256` | `ggml-org/whisper.cpp :: src/whisper.cpp :: search "ggml_conv_1d_ph"` |
| Conformer block (macaron FF / rel-pos MHSA / conv / FF / LN) | `src/conformer/conformer.cpp:607` (`build_conformer_block`) | — (transcribe-local) |
| LayerNorm (affine) | `src/conformer/conformer.cpp:49-60`; `src/arch/cohere/decoder.cpp:52-61` | `ggml-org/llama.cpp :: src/llama-graph.cpp :: search "build_norm"` |
| RMSNorm (+ per-head variant) | `src/arch/qwen3_asr/decoder.cpp:64-69`; `src/arch/qwen3_asr/decoder.cpp:238-239` | `ggml-org/llama.cpp :: src/models/llama.cpp :: search "ggml_rms_norm"` |
| Fused BatchNorm (folded scale+bias) | `src/conformer/conformer.cpp:268-279` | — |
| MHA self-attn, KV-cache append | `src/arch/cohere/decoder.cpp:162` (`mha_self_cached`) | `ggml-org/llama.cpp :: src/models/llama.cpp :: search "build_attn"` |
| Cross-attention (precomputed K/V) | `src/arch/cohere/decoder.cpp:276` (`mha_cross_cached`) | `ggml-org/whisper.cpp :: src/whisper.cpp :: search "Kcross"` |
| Encoder-decoder bridge (enc → K/V proj) | `src/arch/cohere/encoder.cpp:220-227`; `src/arch/cohere/decoder.cpp:513-572` (`build_cross_kv_graph`) | `ggml-org/whisper.cpp :: src/whisper.cpp :: search "Kcross"` |
| GQA (`n_head != n_head_kv`) | `src/arch/qwen3_asr/decoder.cpp:118-122`; `src/arch/qwen3_asr/decoder.cpp:233-235`; `src/arch/qwen3_asr/decoder.cpp:319-329` | `ggml-org/llama.cpp :: src/models/llama.cpp :: search "n_head_kv"` |
| RoPE (`ggml_rope_ext`, mode = NORMAL or NEOX — read reference's `rotate_half` first) | NEOX (split-halves): `src/arch/qwen3_asr/decoder.cpp:242-253`; `src/arch/qwen3_asr/decoder.cpp:525-536`. NORMAL (interleaved / GPT-J): `src/arch/moonshine/encoder.cpp` (`apply_partial_rope`) | `ggml-org/llama.cpp :: src/models/llama.cpp :: search "ggml_rope_ext"` |
| Flash attention (`ggml_flash_attn_ext`) + manual fallback | `src/arch/cohere/decoder.cpp:239-260`; `src/arch/qwen3_asr/decoder.cpp:302-340` | `ggml-org/llama.cpp :: src/llama-graph.cpp :: search "ggml_flash_attn_ext"` |
| Causal mask (built host-side, fed as input) | `src/arch/qwen3_asr/decoder.cpp:154-158` (F16 input mask) | `ggml-org/llama.cpp :: src/llama-graph.cpp :: search "build_attn_inp_kq_mask"` |
| Audio-token injection (splice enc output into LM embeddings) | `src/arch/qwen3_asr/decoder.cpp:168-212` | — |
| RNNT/TDT joint network | `src/arch/parakeet/decoder.cpp:416-450` (`joint_step`) | — |
| LSTM step (host-side, no ggml graph) | `src/arch/parakeet/decoder.cpp:300-340` | — |
| SwiGLU / SiLU-gated FFN | `src/conformer/conformer.cpp:73-90` (`feed_forward`) | `ggml-org/llama.cpp :: src/llama-graph.cpp :: search "ggml_swiglu_split"` |
| Learned positional embedding (view + add) | — | `ggml-org/whisper.cpp :: src/whisper.cpp :: search "positional_embedding"` |
| Tied input/output embedding | `src/arch/cohere/decoder.cpp:389`; `src/arch/cohere/decoder.cpp:483`; `src/arch/qwen3_asr/decoder.cpp:170`; `src/arch/qwen3_asr/decoder.cpp:407` | `ggml-org/llama.cpp :: src/llama-model.cpp :: search "LLM_TENSOR_TOKEN_EMBD.*TENSOR_DUPLICATED"` |

## Core patterns you will re-use

**Mel / frontend.** Keep the mel frontend on host; it runs once per
utterance and the numerical contract (preemphasis α, window length,
FFT size, reflect padding, Slaney filterbank, epsilon) must match the
reference exactly. See `src/transcribe-mel.cpp:437-684` for the whole
pipeline and `MelConfig` in `src/transcribe-mel.h:46-84` for what is
tunable. Verify with a dump against the reference before moving on —
every downstream tensor depends on it.

**Conv / subsampling.** Prefer `ggml_conv_1d_ph` / `ggml_conv_2d` on
backends where they are fast. When a backend lacks an F32 path for
depthwise / pointwise conv (historically Vulkan), route through
`conv_1d_f32` / `conv_2d_dw_f32` / `conv_1d_dw_f32` in
`src/conformer/conformer.cpp:155-261`, which pass the kernel dtype
through `ggml_im2col` instead of hardcoding F16. Families select the
path per-backend via `ConvPolicy` (`src/conformer/conformer.h:185-189`,
`src/conformer/conformer.cpp:299-312`, set in each encoder's builder).

**Norms.** For affine LayerNorm use `ggml_norm(x, eps) · γ + β`
(`src/conformer/conformer.cpp:49-60`). For RMSNorm use
`ggml_mul(ggml_rms_norm(x, eps), w)`
(`src/arch/qwen3_asr/decoder.cpp:64-69`).
BatchNorm is fused offline into a scale+bias pair and applied as two
ops (`src/conformer/conformer.cpp:268-279`); the converter emits the fused tensors.

**Transformer self-attention + KV cache.** The canonical in-tree
example is `mha_self_cached` in `src/arch/cohere/decoder.cpp:162-266`:

1. Q/K/V via `ggml_mul_mat` (+ optional bias), reshape to
   `[head_dim, heads, n_tokens]`.
2. Cache write: `ggml_view_1d` at `layer_offset + past_offset`, then
   `ggml_build_forward_expand(gf, ggml_cpy(ctx, Kcur, k_dst))`.
3. Cache read: `ggml_view_3d` spanning `[head_dim, n_past+n_tokens,
   heads]` at the same layer offset.
4. Attention: `ggml_flash_attn_ext(Q, K, V, mask, scale, 0, 0)` or
   fallback triple `ggml_mul_mat → ggml_soft_max_ext → ggml_mul_mat`
   (lines 240-260).
5. Output projection and optional bias.

**GQA / RoPE.** Qwen3-ASR is the in-tree GQA reference
(`src/arch/qwen3_asr/decoder.cpp`). Reshape Q to `n_heads` and K/V to
`n_kv_heads` (lines 233-235), apply per-head RMSNorm (238-239), then
`ggml_rope_ext` with `GGML_ROPE_TYPE_NEOX` (242-253). For the attention
op, flash-attn handles head-count broadcasting natively; the manual
path repeats K/V across `n_groups = n_heads / n_kv_heads` via
`ggml_repeat` onto a 4D template (319-329). The closest upstream mirror is
`ggml-org/llama.cpp :: src/models/llama.cpp :: search "n_head_kv"`.

**RoPE rotation mode is per-family.** `ggml_rope_ext` accepts two
incompatible pairings: `GGML_ROPE_TYPE_NEOX` (split-halves: pairs
`(0, D/2), (1, D/2+1), ...`) used by qwen3-asr above, and
`GGML_ROPE_TYPE_NORMAL` (interleaved / GPT-J: pairs `(0, 1), (2, 3),
...`) used by moonshine (`src/arch/moonshine/encoder.cpp`
`apply_partial_rope`). Pick the mode by reading the reference's
`rotate_half` and `apply_rotary_pos_emb`: slicing `[0::2]` /
`[1::2]` ⇒ NORMAL; slicing `[: D//2]` / `[D//2 :]` ⇒ NEOX. Picking
the wrong mode silently drifts ~16x at the first attention block —
see `docs/porting/4a-numerical-troubleshooting.md` "RoPE rotation
pattern".

**Encoder-decoder bridge vs audio-token injection.** Two different
models. Cohere (encoder-decoder): encoder output is projected once per
utterance into a layerwise cross-K/V cache
(`src/arch/cohere/encoder.cpp:220-227` plus
`src/arch/cohere/decoder.cpp:513-572`), and each decoder layer
cross-attends via `mha_cross_cached`
(`src/arch/cohere/decoder.cpp:276-345`).
Qwen3-ASR (audio-LLM / token injection): encoder output replaces a slice
of the LM's input embeddings with `ggml_concat` — see
`src/arch/qwen3_asr/decoder.cpp:168-212`. There is no cross-attention; the LM
sees a fused sequence.

**Tied weights.** When the lm_head is tied to the input embedding,
reuse the same tensor: `ggml_get_rows(ctx, token_w, ids)` for lookup
and `ggml_mul_mat(ctx, token_w, x)` for logits (examples in
`src/arch/cohere/decoder.cpp:389,483` and
`src/arch/qwen3_asr/decoder.cpp:170,407`).
Quantization policy then treats that tensor as `Embed`
(`tools/transcribe-quantize/policy.cpp:50-62`), so the preset picks a
wider type than generic linear weights.

**Dump preservation / scheduler output marking.** The ggml scheduler
may reuse intermediate buffers unless a tensor is a graph output. Any
intermediate whose pointer the builder stashes for a later
`dump_tensor` must go through
`transcribe::debug::mark_tensor_for_dump(t)` at build time
(declaration `src/transcribe-debug.h:77`, implementation
`src/transcribe-debug.cpp:129-133`, which is just `ggml_set_output`
behind the env gate). Canonical call sites: the block-0 observer in
`src/arch/parakeet/encoder.cpp:172-187`, the per-stage marks in
`src/arch/qwen3_asr/decoder.cpp:173, 212, 376, 390, 411`. Failing to
mark usually shows up as two dump files being byte-exact equal or as
early-stage tolerances that disappear at later gates — see
[`4-numerical-validation.md`](4-numerical-validation.md) ("Preserve C++
intermediates before scheduler allocation").

**Graph build and compute driver.** Allocate the cgraph with
`ggml_new_graph_custom(ctx, size, /*grads=*/false)` — the default
2048-node budget is too small for a full conformer encoder; 8192+
leaves headroom (`src/arch/parakeet/encoder.cpp:343`,
`src/arch/qwen3_asr/decoder.cpp:160` uses 16384 for a 28-layer decoder
prefill). The compute driver is the
same in every family: `ggml_backend_sched_reset` →
`ggml_backend_sched_alloc_graph` → set inputs via
`ggml_backend_tensor_set` → `ggml_backend_sched_graph_compute` (see
`src/arch/parakeet/model.cpp:663-764`).

**Weight loading + shape validation.** Every tensor read goes through
`transcribe::weights::find_tensor` in
`src/transcribe-weights-util.cpp:18-100`: pass `allowed_types` and
`expected_ne` (an initializer list of `int64_t` dims, fast-to-slow) and
it logs and returns null on mismatch. Wrap it with the `REQUIRE_TENSOR`
macro in the header so misses abort the load with the tensor name.

**Quantization policy touchpoints.** Per-tensor dtype is resolved by
`resolve_target_type(preset, name, ne0)` in
`tools/transcribe-quantize/policy.cpp:196-238`. It calls
`classify_tensor(name)` (lines 50-131) to bucket into
`{Linear, Embed, ConvPw, Conv, Norm}` and then maps bucket → preset
type. Two things to add for a new family: (a) family-specific overrides
for tied-embedding tensors at the top of `classify_tensor`, and (b)
smoke coverage in `tests/<family>_quantize_smoke.cpp`. Norms and biases
must stay F32; `Embed` and attention-output slots get wider types in
`_M` presets.

## Common failure modes and where to look first

- **Two dump files are byte-exact equal** or late tensors match but early
  ones don't → missing `mark_tensor_for_dump` while the graph is built.
  See the dump-preservation note above and
  [`4-numerical-validation.md`](4-numerical-validation.md).
- **Mel off by a small constant / first or last frame wrong** → window,
  reflect padding, or preemphasis α mismatch. Compare the first 400
  samples and the window slice against the reference; see the comments
  at the top of `src/transcribe-mel.cpp` and
  [`4a-numerical-troubleshooting.md`](4a-numerical-troubleshooting.md).
- **Pre-encode output magnitude way off** → BatchNorm fold orientation
  or conv-transpose mismatch. Dump `enc.pre_encode.out` and compare
  against the reference.
- **Embedding lookup diverges** → tokenizer, input-ids dtype, or
  row-major/col-major mismatch in the weight.
- **Logits have `-inf` in some positions** → post-softmax was dumped;
  switch to pre-softmax `dec.logits_raw`.
- **Flash-attn path wrong, fallback correct** → check K/V dtype
  (flash-attn expects F16), mask dtype (F16), and head-count
  broadcasting. Env override: `TRANSCRIBE_NO_FLASH=1`
  (`src/transcribe-flash-policy.cpp:11-23`).
- **Conv path wrong on Vulkan only** → pointwise or depthwise direct
  path on a backend that lacks the F16 kernel; force the im2col path
  with `TRANSCRIBE_CONV_NO_DIRECT_PW=1` or update `ConvPolicy`
  defaults.
- **Graph build fails with "too many nodes"** → bump the
  `ggml_new_graph_custom` size; each conformer block is ~90 ops.
- **KV-cache read returns zeros for past tokens** → layer offset or
  stride miscomputed in `ggml_view_3d`; verify against
  `src/arch/cohere/decoder.cpp:221-234` and confirm the flat buffer is sized
  `n_layer * n_ctx * hidden * element_size`.
