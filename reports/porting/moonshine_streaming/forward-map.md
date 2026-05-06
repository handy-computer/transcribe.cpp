# Forward map - moonshine_streaming

Reference: `transformers.models.moonshine_streaming.modeling_moonshine_streaming` @ HF transformers v5.7.0.
Closest in-tree analog: `src/arch/moonshine/` (encoder-decoder transformer with
partial RoPE on decoder self-attn) + a novel time-domain conv frontend.

A compact map from `MoonshineStreamingForConditionalGeneration` forward pass
to the C++ port. Architecture is shared across moonshine_streaming variants
(tiny / small / medium); per-variant differences are config-only
(n_layers, hidden_size, intermediate_size, n_heads, sliding_windows).

Block patterns are mapped once. Encoder dumps every block on tiny (6
layers ≤ 8). The decoder mid-generation dump (`dec.logits_raw.gen20`) is
required — `encoder-decoder` family with KV cache.

## Frontend

Time-domain "mel-free" stack. 50 Hz frame rate at 16 kHz (frame_len=80
samples = 5 ms). Runs INSIDE the encoder graph; no separate mel/STFT.

| Stage | Reference location | Output shape (ggml ne) | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|------------------------|-------------|--------------------|----------------|
| Read raw PCM | runtime audio loader | `[T_samples=176000]` f32 in [-1, 1] | `enc.audio.in` | `transcribe::audio::load_pcm` | shared |
| Reshape to frames | `embedder.forward` line 290 | `[frame_len=80, T_frames=2200]` | — | `ggml_reshape_2d(x, frame_len, T_frames)`; T_frames = T_samples/80 | new |
| Per-frame CMVN | `MoonshineStreamingFrameCMVN` | `[80, 2200]` | `enc.embedder.cmvn.out` | `ggml_norm(x, eps=1e-6)` (no affine — CMVN = (x-mean)/sqrt(var+eps)) | new (resembles layer-norm-without-affine) |
| asinh compression | `MoonshineStreamingAsinhCompression` (log_k learned scalar) | `[80, 2200]` | `enc.embedder.comp.out` | `asinh(exp(log_k) * x)` — express as `log(y + sqrt(y*y+1))` where `y = exp(log_k) * x` | new |
| Linear(frame_len → hidden) + SiLU | `embedder.linear` (no bias) | `[hidden=320, T_frames=2200]` | `enc.embedder.linear.out` | `ggml_silu(ggml_mul_mat(linear_w, x))` | new |
| Causal Conv1d (k=5, s=2, hidden→2·hidden) + bias + SiLU | `embedder.conv1` | `[2·hidden=640, T1=1100]` | `enc.embedder.conv1.out` | left-pad input by `k-1=4`, `ggml_conv_1d(stride=2, p=0)`, bias add, `ggml_silu` | new (causal conv stem) |
| Causal Conv1d (k=5, s=2, 2·hidden→hidden) + bias | `embedder.conv2` | `[hidden=320, T_enc=550]` | `enc.embedder.conv2.out` | left-pad by 4, conv1d stride 2, bias add — NO SiLU after conv2 | new |

Note: encoder hidden dim and "frame channels" shape both use ne[0] = C
(channels innermost), ne[1] = T. The Python dumper transposes to `[T, C]`
file order; that maps to ggml `ne=[C, T]`.

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Per-layer sliding-window mask | `MoonshineStreamingEncoder.forward` lines 405-411 (`create_bidirectional_mask + sliding_window_mask_function`) | `[T_enc, T_enc]` per layer | implicit (host-built) | Build host-side mask: `mask[q,k] = 0 if (q-k ∈ [0, L)) ∨ (k-q ∈ (0, R))` else `-inf`. Upload as input tensor per-layer. tiny: layers `[(16,4),(16,4),(16,0),(16,0),(16,4),(16,4)]`. | new |
| Encoder block ×6 (pre-LN MHSA + GELU MLP, no RoPE) | `MoonshineStreamingEncoderLayer.forward` | `[hidden, T_enc]` | `enc.block.{0..5}.out` | See "Encoder block pattern" | resembles `whisper` encoder block but with sliding-window mask in place of full bidirectional attn |
| Final LN (no bias, unit_offset PRE-FOLDED by converter) | `encoder.final_norm` | `[hidden, T_enc]` | `enc.final` | `ggml_norm(x, eps=1e-5)` then `* scale` (the GGUF tensor already includes the +1.0) | similar to whisper's final_layer_norm but without bias |

### Encoder block pattern

Pre-LN. attention_bias=False (q/k/v/o weight-only). LayerNorm scale is
the **pre-folded** gain (γ + 1.0 already baked in by converter).

```
x_in
↓ ln(no bias) = norm_attn       (scale already includes +1.0)
↓ q = x · Wq ; k = x · Wk ; v = x · Wv     (bias-less)
↓ scaled-dot-product attention with the per-layer sliding-window mask
   (NO RoPE on encoder — position info comes purely from the mask)
↓ o = attn · Wo                              (bias-less)
↓ residual: x = x_in + o
↓ ln(no bias) = norm_ffn         (pre-folded scale)
↓ y = fc1(x)                  (Linear, with bias)
   y = gelu(y)
   y = fc2(y)                 (Linear, with bias)
↓ residual: x = x + y
```

## Adapter

Sits between encoder output and decoder cross-attn. **Applied exactly
once** before the decoder layer loop (HF `decoder.forward` does this
in-place; the C++ port must apply outside the per-step graph to avoid
double application).

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| pos_emb slice | `decoder.pos_emb(arange(T_enc))` | `[enc_hidden, T_enc]` | `adapter.pos_emb` | `ggml_get_rows(adapter.pos_emb.weight, arange[T_enc])` | new |
| Add pos_emb to encoder output | `encoder_hidden_states += pos_emb` | `[enc_hidden, T_enc]` | (intermediate) | `ggml_add(encoder_out, pos_emb)` | new |
| Linear adapter.proj (only when enc_hidden ≠ dec_hidden) | `decoder.proj` | `[dec_hidden, T_enc]` | `adapter.out` | `ggml_mul_mat(adapter.proj.weight, x)` (Identity / skipped when adapter_has_proj=false) | new |

For tiny: enc_hidden == dec_hidden (320), `adapter_has_proj=false`,
`adapter.proj.weight` absent — adapter graph is just pos_emb + add.

## Decoder

Autoregressive, partial RoPE on self-attn, no RoPE on cross-attn,
SwiGLU MLP. ALL decoder LayerNorms are vanilla `nn.LayerNorm(bias=False)`
— **no unit_offset trick**.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Token embed lookup | `decoder.embed_tokens` | `[hidden, n_tokens]` | `dec.token_emb`, `dec.embed_sum` | `ggml_get_rows(dec.token_embd.weight, ids)` | moonshine |
| Decoder block ×6 | `MoonshineStreamingDecoderLayer.forward` | `[hidden, n_tokens]` | `dec.block.{0..5}.out` | See "Decoder block pattern" | moonshine + identical mask shape |
| Final LN (no bias, no unit_offset) | `decoder.norm` | `[hidden, n_tokens]` | `dec.out_before_head` | `ggml_norm(x, eps=1e-5) * scale` | moonshine |
| Untied lm_head | `proj_out` (separate from token_embd) | `[vocab=32768, n_tokens]` | `dec.logits_raw`, `dec.logits_raw.gen20` | `ggml_mul_mat(dec.lm_head.weight, x)` — **NOT tied** | new (moonshine has tied lm_head) |
| `log_softmax` | reference dumper only | `[vocab, n_tokens]` | `dec.logits` | C++ runtime applies log_softmax only when needed; tensor-level gate compares pre-softmax `logits_raw`. | shared |

### Decoder block pattern

Pre-LN throughout. attention_bias=False (q/k/v from `q_proj/k_proj/v_proj`
weight-only); o_proj is also bias=False.

```
x_in
↓ ln(no bias) = norm_self        (no offset; vanilla LN scale)
↓ self_q, self_k, self_v projections (bias-less)
↓ partial RoPE on self_q[..., :32] and self_k[..., :32]
   (positions = n_past + [0..n_tokens), partial_rotary_factor=0.8,
    head_dim=40, head_dim_rot=32)
   GPT-J interleaved (rotate_half slices [..., 0::2]/[..., 1::2]) →
   GGML_ROPE_TYPE_NORMAL.
↓ append (self_k, self_v) into per-layer self-attn cache at offset n_past
↓ scaled-dot-product attention with causal mask over the live cache
↓ self_out projection (bias-less)
↓ residual: x = x_in + self_out
↓ ln(no bias) = norm_cross
↓ cross_q projection (bias-less)
↓ cross_k = adapter_out · Wk_c    (computed once per session, cached)
↓ cross_v = adapter_out · Wv_c    (computed once per session, cached)
↓ scaled-dot-product attention (no causal mask; full T_enc)
↓ cross_out projection (bias-less)
↓ residual
↓ ln(no bias) = norm_ffn
↓ y = fc1(x)              (hidden → 2·intermediate, with bias)
   x_proj, gate = split y on last dim
   y = silu(gate) ⊙ x_proj
   y = fc2(y)              (intermediate → hidden, with bias)
↓ residual
```

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Adapter precompute | once per session | `[dec_hidden, T_enc]` | `adapter.out` | Apply pos_emb add (+ proj if present) on encoder output before any decoder graph runs | new |
| Cross-attn KV cache | once per session from `adapter.out` | `[d_model, T_enc] × 2 per layer` | implicit | Same dual-cache strategy as `MoonshineKvCache` | moonshine |
| Self-attn KV cache | grows per step | `[d_model, n_ctx_max] × 2 per layer` | implicit | Per-layer flat tensors; views over `[0..n_past+n_tokens)` | moonshine |
| Prompt pass | `[bos=1]` | `[vocab]` | `dec.logits_raw` | n_past=0, n_tokens=1 | moonshine |
| Step pass | each subsequent | `[vocab]` | `dec.logits_raw.gen20` | n_past>=1, n_tokens=1, position id = n_past | moonshine |
| Greedy step | argmax until eos=2 or `max_position_embeddings=4096` | seq of token IDs | transcript | Reference uses do_sample=False, num_beams=1 | moonshine |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (en, no language hint) | `model.generate(input_values, attention_mask)` | `transcribe-cli -m … audio.wav` | "Transcribe (auto / no hint)" → PASS expected once Stage 4 is green |
| Transcribe (en, explicit-language) | language hint ignored — model is en-only | `transcribe-cli ... -l en` | "Transcribe (explicit language=en)" → PASS expected (same observable as auto) |
| Streaming (incremental decode) | model architecture supports streaming session | one-shot only at Stage 4 | SKIP — not exposed by runtime; streaming session API is a post-port deliverable |
| Language detection | not advertised | n/a | SKIP — not exposed by runtime |
| Translation | not advertised | n/a | SKIP — not exposed by runtime |
| Timestamps | not advertised; vocab has no timestamp tokens | n/a | SKIP — not exposed by runtime |

## Deviations From Closest Analog (vs `src/arch/moonshine/`)

- **Frontend.** Moonshine has 3 conv layers on raw PCM (k=127/7/3,
  s=64/3/2, no GroupNorm in the middle is wrong — Moonshine has GN
  after conv0; streaming has none). Streaming reframes PCM into
  fixed-size 80-sample frames, applies parameter-free CMVN per frame,
  asinh-compresses, projects to hidden via a Linear layer, then runs
  two **causal** Conv1d layers (k=5, s=2 each).
- **Encoder attention.** Moonshine's encoder uses partial-RoPE 0.9
  bidirectional attention with no mask. Streaming has **no RoPE** on
  the encoder; position info is encoded by per-layer (left, right)
  sliding-window masks. The pattern across layers is fixed by the
  config and shipped as `stt.moonshine_streaming.encoder.sliding_windows`.
- **Encoder LayerNorm scale is pre-folded.** Moonshine encoder LNs are
  vanilla bias-less `nn.LayerNorm`. Streaming uses `MoonshineStreamingLayerNorm`
  with `unit_offset=True` (effective gain = γ + 1.0). The converter
  pre-folds the +1.0 into the GGUF tensor; C++ uses the gain tensor
  unchanged.
- **Adapter layer.** Moonshine has none — encoder output flows directly
  into decoder cross-attn. Streaming inserts an additive positional
  embedding (`adapter.pos_emb`) plus an optional projection
  (`adapter.proj`, present only when enc_hidden ≠ dec_hidden — absent
  for tiny). The HF reference does this in-place inside `decoder.forward`,
  which is a footgun for KV-cached step graphs. The C++ port applies the
  adapter ONCE before the layer loop, mirroring the dumper's
  `apply_adapter()` helper.
- **Untied lm_head.** Moonshine ties `proj_out.weight` to `dec.token_embd.weight`.
  Streaming has `tie_word_embeddings=false`; `proj_out.weight` is a
  separate tensor mapped to `dec.lm_head.weight` in the GGUF.
- **No conv-stem head_dim padding interactions.** Moonshine (tiny) needs
  `pad_head_dim_to_multiple_of=8` (head_dim 36 → 40). Streaming-tiny has
  head_dim=40 already, which is a multiple of 8 → padding is a no-op.
  The KV pad/unpad code path in mha_self/cross is still wired (matches
  HF's `pad_head_dim_to_multiple_of=8` config) for forward compatibility
  with small/medium variants.

## Variant Notes

- **moonshine-streaming-tiny**: family baseline above. Concrete dims:
  `enc_n_layers=dec_n_layers=6`, `hidden=320` (encoder == decoder), `n_heads=8`,
  `n_kv_heads=8` (no GQA), `head_dim=40`, `head_dim_rot=32` (partial 0.8),
  `intermediate=1280`, `vocab_size=32768`, `max_position_embeddings=4096`,
  F32 reference dtype, English-only. Sliding-window pattern
  `[(16,4),(16,4),(16,0),(16,0),(16,4),(16,4)]` — 4-frame right
  lookahead in 4 of 6 layers gives ~80 ms latency budget.
  `adapter_has_proj=false` (enc_hidden == dec_hidden).
- **RoPE rotation mode (decoder)**: same as moonshine — GPT-J /
  interleaved. `rotate_half` slices `[..., 0::2]` / `[..., 1::2]` →
  GGML_ROPE_TYPE_NORMAL, NOT NEOX.
- **`moonshine-streaming-small` / `-medium`**: deferred. Same architecture,
  larger dims, possibly enc_hidden ≠ dec_hidden (would activate the
  `adapter.proj.weight` path). Variant-level config plus an extra
  matmul per-session in the adapter precompute is the only delta.
