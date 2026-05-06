# Forward map - moonshine

Reference: `transformers.models.moonshine.modeling_moonshine` @ HF
revision `390624ed` (intake-pinned).
Closest in-tree analog: `src/arch/whisper/` (encoder-decoder transformer
with conv stem) plus partial-RoPE patterns from `src/arch/cohere/`.

A compact map from the `MoonshineForConditionalGeneration` forward pass
to the C++ port. The architecture is shared across moonshine variants
(tiny, base, ...); per-variant differences are config-only (n_layers,
hidden_size, intermediate_size, n_heads).

Block patterns (encoder block, decoder block) are mapped once. The
`enc.block.{0,1,2,4,5}.out` and `dec.block.{0,1,2,4,5}.out` gate tensors
in `dump_coverage.json` are the first/middle/last positions where the
shared pattern is checked. Block 3 is intentionally not gated.

## Frontend

Moonshine has no STFT, no mel filterbank, no per-utterance normalization.
The "frontend" is the encoder's three Conv1d layers operating on the
raw 16 kHz waveform. The Wav2Vec2FeatureExtractor on the HF side only
right-pads a batch with 0.0 and rescales int16 PCM to float32; both
behaviors live in the C++ runtime's audio-load path, not in a per-family
mel module.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Read raw PCM | runtime audio loader | `[T_samples]` f32 in `[-1, 1]` | `enc.audio.in` | existing `transcribe::audio::load_pcm` path | shared |
| Pad batch | `Wav2Vec2FeatureExtractor.__call__` (right-pad with 0.0) | `[T_samples_padded]` | — (single utterance) | n/a until batched inference lands | — |
| No mel / no STFT | — | — | — | — | — |

`stt.frontend.type="raw"`, `sample_rate=16000`, `num_mels=1`. The C++
encoder consumes the PCM directly via the conv stem below; the
reference dumper passes `input_values` (float32 PCM) into
`MoonshineEncoder.forward` unchanged.

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| `conv1` (k=127, s=64, no bias) → `tanh` | `MoonshineEncoder.forward` (modeling_moonshine.py:95) | `[B, d_model, T_conv1]` | `enc.conv1.out` (after tanh, **before** GroupNorm) | `ggml_conv_1d(stride=64)` then `ggml_tanh` | new — whisper conv1 has stride 1, this is novel |
| `groupnorm` (num_groups=1, eps=1e-5) | `MoonshineEncoder.forward` (modeling_moonshine.py:96) | same | `enc.groupnorm.out` | `ggml_group_norm(x, 1, eps)` then affine `* w + b` | new |
| `conv2` (k=7, s=3) → `gelu` | `MoonshineEncoder.forward` (modeling_moonshine.py:97) | `[B, 2·d_model, T_conv2]` | `enc.conv2.out` | `ggml_conv_1d(stride=3)` + `ggml_gelu` | new |
| `conv3` (k=3, s=2) → `gelu` → permute to `[B, T_enc, d_model]` | `MoonshineEncoder.forward` (modeling_moonshine.py:98–99) | `[B, T_enc, d_model]` | `enc.conv3.out` | `ggml_conv_1d(stride=2)` + `ggml_gelu` + transpose | new |
| RoPE cos/sin precompute (encoder) | `MoonshineEncoder.forward` rotary_emb | `[B, T_enc, head_dim_rot=32]` | `enc.rope.cos`, `enc.rope.sin` | head_dim_rot = `int(head_dim · 0.9)` rounded to even. Standard RoPE table init with `theta=10000`. | similar to cohere decoder partial RoPE; new for encoder |
| Encoder block ×6 (pre-LN MHSA + GELU MLP) | `MoonshineEncoderLayer.forward` | `[B, T_enc, d_model]` | `enc.block.{0,1,2,4,5}.out` | See "Encoder block pattern" below | whisper encoder block + partial RoPE on Q/K |
| Final LN (no bias) | `MoonshineEncoder.layer_norm` | `[B, T_enc, d_model]` | `enc.final` | `ggml_norm` + `ggml_mul(scale)` (no add — bias=False) | whisper has bias here; moonshine doesn't |

### Encoder block pattern

Pre-LN. attention_bias=False so q/k/v/o are weight-only.

```
x_in
↓ ln(no bias) = norm_attn
↓ q = x · Wq  ; k = x · Wk  ; v = x · Wv     (all bias-less)
↓ apply partial RoPE to q[..., :32], k[..., :32] using enc.rope.cos/sin
↓ scaled-dot-product attention (no causal mask — encoder is bidirectional)
↓ o = attn · Wo                              (bias-less)
↓ residual: x = x_in + o
↓ ln(no bias) = norm_ffn
↓ y = fc1(x)                  (Linear, with bias)
   y = gelu(y)
   y = fc2(y)                 (Linear, with bias)
↓ residual: x = x + y
```

## Decoder

Decoder is autoregressive with self-attn (causal, partial RoPE) +
cross-attn (no RoPE) + SwiGLU MLP.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Token embedding lookup | `MoonshineDecoder.embed_tokens` | `[B, T_tgt, d_model]` | `dec.token_emb`, `dec.embed_sum` | `ggml_get_rows(token_embd, ids)` | whisper minus pos_emb add |
| RoPE cos/sin precompute (decoder) | `MoonshineDecoder.forward` rotary_emb | `[B, T_tgt, head_dim_rot]` | implicit; reused step-to-step | Same builder as encoder; positions are absolute (n_past + step idx). | cohere decoder partial RoPE |
| Decoder block ×6 | `MoonshineDecoderLayer.forward` | `[B, T_tgt, d_model]` | `dec.block.{0,1,2,4,5}.out` | See "Decoder block pattern" below | whisper decoder + partial RoPE + SwiGLU |
| Final LN (no bias) | `MoonshineDecoder.norm` | `[B, T_tgt, d_model]` | `dec.out_before_head` | `ggml_norm` + `ggml_mul(scale)` (no bias add) | bias-less variant of whisper |
| Tied lm_head | `MoonshineForConditionalGeneration.proj_out` | `[B, T_tgt, vocab_size]` | `dec.logits_raw`, `dec.logits_raw.gen20` | `ggml_mul_mat(token_embd, x)` — same weight as embedding lookup | whisper |
| `log_softmax` | reference dumper only | `[B, T_tgt, vocab_size]` | `dec.logits` | C++ runtime applies `log_softmax` only when needed; tensor-level gate compares pre-softmax `logits_raw`. | shared |

### Decoder block pattern

Pre-LN throughout. attention_bias=False so q/k/v/o are weight-only on
both self-attn and cross-attn. Three layer-norms per block (no bias).

```
x_in
↓ ln(no bias) = norm_self
↓ self_q, self_k, self_v projections (bias-less)
↓ partial RoPE on self_q[..., :32] and self_k[..., :32]
   (positions = n_past + [0..T_tgt))
↓ append (self_k, self_v) into the per-layer self-attn cache at offset n_past
↓ scaled-dot-product attention with causal mask over the live cache
↓ self_out projection (bias-less)
↓ residual: x = x_in + self_out_residual
↓ ln(no bias) = norm_cross
↓ cross_q projection (bias-less)
↓ cross_k = encoder_out · Wk_c    (computed once at session start, cached)
↓ cross_v = encoder_out · Wv_c    (computed once at session start, cached)
↓ scaled-dot-product attention (no causal mask; full encoder length)
↓ cross_out projection (bias-less)
↓ residual: x = x + cross_out_residual
↓ ln(no bias) = norm_ffn
↓ y = fc1(x)              (hidden → 2·intermediate, with bias)
   x_proj, gate = split y on last dim
   y = silu(gate) ⊙ x_proj
   y = fc2(y)              (intermediate → hidden, with bias)
↓ residual: x = x + y
```

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Encoder cache (cross-attn KV) | computed once from final encoder hidden | `[T_enc, d_model]` × 2 per layer | implicit | Compute `cross_k_cache[i] = enc_out · Wk_c[i]` and `cross_v_cache[i] = enc_out · Wv_c[i]` once at session start; reuse every step. Same dual-cache strategy as `src/arch/cohere/cohere.h::CohereKvCache`. | cohere |
| Self-attn KV cache | grows per step | `[n_ctx_max, d_model]` × 2 per layer | implicit | Per-layer flat tensors; views over `[0..n_past + T_step)` for read, write at offset `n_past`. | cohere / whisper |
| Prompt pass | first decode call with `[bos]` | `[1, vocab_size]` | `dec.logits_raw` | n_past=0, T=1. | shared |
| Step pass | each subsequent decode | `[1, vocab_size]` | `dec.logits_raw.gen20` | n_past>=1, T=1. Position id = n_past. | whisper / cohere |
| Greedy step | argmax until eos=2 or `max_position_embeddings=194` | seq of token IDs | transcript | No suppress_tokens, no begin_suppress_tokens. | whisper minus suppression |
| Hallucination throttle | model card recommends ≤6.5 tok/s; HF leaves it to caller | — | — | Out-of-scope for Stage 4. Capability table marks SKIP. | — |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (en, no language hint) | `model.generate(input_values)` | `transcribe-cli -m … audio.wav` | "Transcribe (auto / no hint)" → PASS expected once Stage 4 is green |
| Language detection | not advertised | n/a | SKIP — not exposed by runtime |
| Translation | not advertised | n/a | SKIP — not exposed by runtime |
| Timestamps | not advertised; vocab has no timestamp tokens | n/a | SKIP — not exposed by runtime |

## Deviations From Closest Analog

- **Conv stem on raw waveform.** Whisper has 2 conv layers operating on
  pre-computed mel features. Moonshine has 3 conv layers on raw 16 kHz
  PCM (kernels 127/7/3, strides 64/3/2, no bias on conv1) plus a single
  GroupNorm right after `tanh(conv1)`. No pre-mel module; the conv stem
  IS the frontend.
- **No additive positional embedding.** Whisper carries learned
  `enc.pos_emb.weight` and `dec.pos_emb.weight`. Moonshine uses RoPE
  inside attention only. The dumper still emits `dec.embed_sum` (= the
  token embedding) for naming consistency with whisper/cohere goldens.
- **Partial RoPE 0.9 on Q/K only.** Encoder self-attn AND decoder
  self-attn rotate the leading `head_dim_rot = int(head_dim · 0.9)`
  rounded to even (`32` for tiny). Trailing dims pass through. Cross-attn
  is NOT rotated. Cohere uses partial RoPE in its decoder; that's the
  closest in-tree pattern to mirror.
- **All attention biases = False.** `config.attention_bias=False` so
  every q/k/v/o projection is weight-only on both encoder and decoder,
  on both self- and cross-attention. The weights struct collapses to
  weight-only slots.
- **Bias-less LayerNorms.** Every `nn.LayerNorm(...)` in
  `modeling_moonshine.py` is constructed with `bias=False`. The C++
  port emits `* w` only after `ggml_norm`, no `+ b`.
- **Decoder MLP is SwiGLU.** Encoder MLP is plain GELU (same as
  whisper). Decoder MLP fc1 outputs `2·intermediate`, splits into
  `[x_proj, gate]`, and computes `silu(gate) ⊙ x_proj`. fc2 takes
  `intermediate`.
- **Tied logits head with no bias.** `proj_out` reuses the token
  embedding tensor; no separate `lm_head.weight`, no `lm_head.bias`.
  The C++ graph uses `dec.token_embd.weight` for both the embedding
  lookup and the logits projection.
- **`pad_head_dim_to_multiple_of=8`** is set in HF config. For tiny:
  head_dim=288/8=36; padding to multiple of 8 → 40. The HF attention
  module pads q/k/v on the head dim for the matmul and slices off
  afterward. MLX does NOT pad. Whether this matters numerically depends
  on the backend; see Variant Notes.

## Variant Notes

- **moonshine-tiny**: family baseline above. Concrete dims:
  `enc_n_layers=dec_n_layers=6`, `hidden=288`, `n_heads=8`,
  `n_kv_heads=8` (no GQA), `head_dim=36`, `head_dim_rot=32` (partial
  0.9), `intermediate=1152`, `vocab_size=32768`,
  `max_position_embeddings=194`, F32 reference dtype, English-only.
- **`pad_head_dim_to_multiple_of` resolved at Stage 4**: HF moonshine
  pads `head_dim` from 36 to 40 (multiple of 8). C++ matches HF: q/k/v
  are zero-extended on dim 0 before flash-attention, then sliced back
  to `head_dim` before the merge-and-project. Attention scale uses the
  *unpadded* `head_dim` per HF (`self.scaling = self.head_dim**-0.5`).
  See `src/arch/moonshine/encoder.cpp::mha_encoder` and
  `src/arch/moonshine/decoder.cpp::mha_self_cached`. MLX's "skip
  padding" choice was not adopted.
- **RoPE rotation mode**: moonshine uses GPT-J / interleaved RoPE
  (`rotate_half` slices `[..., 0::2]` / `[..., 1::2]`); ggml mode is
  `GGML_ROPE_TYPE_NORMAL`, NOT `GGML_ROPE_TYPE_NEOX`. Picking NEOX
  silently drifts ~16x at the first attention block.
- **`moonshine-base`**: deferred. Same architecture, larger dims; the
  family-level map should hold. When porting base, verify `head_dim_rot`
  and that `pad_head_dim_to_multiple_of` resolves the same way.
