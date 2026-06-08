# Forward map - medasr

Reference: `transformers.models.lasr.modeling_lasr` @ `65dc261512cbdb1ee72b88ae5b222f2605aad8e5` (v5.0.0.dev0)
Closest in-tree analog: `src/arch/parakeet/` (Conformer + macaron) + `src/causal_lm/` (RoPE) + `src/arch/sensevoice/` (CTC head)

MedASR is the first in-tree `encoder-ctc` Conformer family. It does NOT
fit `build_conformer_block` from `src/conformer/conformer.h` because the
upstream block uses load-bearing scaled residuals on every sub-block
(macaron `[1.5, 0.5]`, conv `[2.0, 1.0]`) rather than the standard 0.5
macaron half-step, and the attention path is RoPE rather than
relative-position. The block is therefore hand-built in
`src/arch/medasr/encoder.cpp` using the lower-level shared helpers
(`layer_norm`, `feed_forward`, `conv_1d_dw_f32`, `fused_batch_norm`,
`add_conv_bias`) plus a per-family `rope_mhsa` written from
`src/causal_lm/`'s `ggml_rope_ext` pattern.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| log-mel | `feature_extraction_lasr.LasrFeatureExtractor.__call__` | `[T_mel=floor((n_pcm-400)/160)+1, 128]` f32 | `mel.in` | host-side `transcribe::MelFrontend`; LASR knobs require `center=false`, `pad_mode="zero"`, `pre_emphasis=0.0`, `normalize="none"`, `window=ckpt`, `filterbank=ckpt`, `log_compression="log_clamp_1e-5"`. Step 2 brings up the encoder with `TRANSCRIBE_MEDASR_MEL_FROM_REF=<ref_dir>` so Step 7 (frontend parity) is the first place the C++ mel actually runs. | `src/transcribe-mel.{h,cpp}` (extend with `center=false` knob and `log_compression` enum) |

For jfk.wav (176 400 samples): T_mel = (176 400 − 400)/160 + 1 = 1 099 frames — reference dump rounds to 1 098 (one trailing frame dropped because LASR's manual unfold does `len // win` floor on the residual; not centered). The forward-map records the formula; bit-exact frame count is a Step 7 parity check.

## Encoder

Input: `mel` ne `[T_mel, 128, 1, B]` f32 (matches MelFrontend row-major output).

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Subsampling: dense_0 | `LasrEncoderSubsampling.dense_0` | `[T_mel, 512]` | — | `mul_mat` + bias add | parakeet `feed_forward` w/o second linear |
| Subsampling: relu0 | `F.relu` | `[T_mel, 512]` | — | `ggml_relu` | n/a |
| Subsampling: conv_0 | `Conv1d(512, 512, k=5, s=2, p=0)` | `[T1=floor((T_mel-5)/2)+1, 512]` | — | `conv_1d_f32` (kernel `[KW=5, IC=512, OC=512]`, stride 2, pad 0) | conformer `conv_1d_f32` |
| Subsampling: conv_0 bias + relu | bias add + `F.relu` | `[T1, 512]` | — | `add_conv_bias` + `ggml_relu` | conformer |
| Subsampling: conv_1 | `Conv1d(512, 256, k=5, s=2, p=0)` | `[T_enc=floor((T1-5)/2)+1, 256]` | — | `conv_1d_f32` | conformer |
| Subsampling: conv_1 bias + relu | bias + `F.relu` | `[T_enc, 256]` | — | `add_conv_bias` + `ggml_relu` | n/a |
| Subsampling: dense_1 | `Linear(256, 512)` + bias | `[T_enc, 512]` | `enc.subsampling.out` | `mul_mat` + bias add | n/a |
| Block ×17: norm_ff1 | `LayerNorm(512, eps=1e-6, bias=False)` | `[T_enc, 512]` | (in `post_ff1`) | `layer_norm(x, w, nullptr)` with `eps=1e-6` | conformer `layer_norm` (eps hardcoded 1e-5 — see deviation) |
| Block ×17: ff1 | `LasrFeedForward` = `Linear → SiLU → Linear` (no biases) | `[T_enc, 512]` | — | `feed_forward(x, lin1_w, nullptr, lin2_w, nullptr)` | conformer `feed_forward` |
| Block ×17: scaled-residual ff1 | `x = x*1.5 + ff1*0.5` | `[T_enc, 512]` | `enc.block.0.post_ff1` (block 0 only) | `ggml_add(ggml_scale(x, 1.5), ggml_scale(ff1, 0.5))` | NOT `macaron_ff_residual` (deviation) |
| Block ×17: norm_attn | `LayerNorm` no bias | — | — | `layer_norm` w/ eps 1e-6 | conformer |
| Block ×17: self-attn Q/K/V | `Linear` no bias each → reshape `[d_model] → [head_dim, n_head]` | `[head_dim=64, n_head=8, T_enc, B]` | — | three `mul_mat` then `reshape_3d`+`permute` for head layout | causal_lm Q/K/V projection |
| Block ×17: RoPE on Q, K | `LasrEncoderRotaryEmbedding` (rope_theta=10 000, type=default) applied per-head | unchanged | — | `ggml_rope_ext(Q, positions, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, max_pos=10000, rope_theta=10000, 1,0,1,32,1)` and same for K | causal_lm RoPE call site (deviation: no MRoPE, no extrapolation) |
| Block ×17: scaled-dot attn | softmax(QK/√d_k)·V, no mask | `[head_dim, n_head, T_enc, B]` | — | `flash_attn_ext` or manual mul_mat+soft_max | causal_lm |
| Block ×17: o_proj | `Linear` no bias | `[T_enc, 512]` | — | reshape back + `mul_mat` | causal_lm |
| Block ×17: residual attn | `x = x + attn` (UNSCALED) | `[T_enc, 512]` | `enc.block.0.post_attn` | `ggml_add` | standard |
| Block ×17: norm_conv | `LayerNorm` no bias | — | — | `layer_norm` | conformer |
| Block ×17: conv module | `Conv1d pw1(512→1024,k=1)` → GLU → `Conv1d dw(512,k=32,p=15,g=512)` → `BatchNorm1d(512)` → `SiLU` → `Conv1d pw2(512→512,k=1)` | `[T_enc, 512]` | — | `mul_mat` + `ggml_swiglu` + `conv_1d_dw_f32` + `fused_batch_norm` + `silu` + `mul_mat`. Depthwise uses **non-centered** pad=15 (left=15, right=16 to keep T identical with k=32 even kernel) — verify against reference. | conformer `conv_module` (cannot reuse — that helper bakes in the unscaled `x + conv` residual; we need a custom scaled residual outside) |
| Block ×17: scaled-residual conv | `x = x*2.0 + conv*1.0` | `[T_enc, 512]` | `enc.block.0.post_conv` | `ggml_add(ggml_scale(x, 2.0), conv)` | deviation |
| Block ×17: norm_ff2 + ff2 | same shape as FF1 | `[T_enc, 512]` | — | mirrors FF1 | conformer |
| Block ×17: scaled-residual ff2 | `x = x*1.5 + ff2*0.5` | `[T_enc, 512]` | `enc.block.0.post_ff2` | `ggml_add(ggml_scale(x, 1.5), ggml_scale(ff2, 0.5))` | deviation |
| Block ×17: norm_out | `LayerNorm` no bias | `[T_enc, 512]` | `enc.block.{0,7,8,16}.out` | `layer_norm` w/ eps 1e-6 | conformer |
| enc.out_norm | top-level `LayerNorm` no bias | `[T_enc, 512]` | `enc.out_norm.out` | `layer_norm` | conformer |

## Decoder

n/a — non-autoregressive CTC head.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| CTC head | `LasrCtcHead.ctc_head = Conv1d(512, 512, k=1)` + bias | `[vocab=512, T_enc, 1, B]` (note `[vocab, T]` order) | `enc.ctc_logits` | `ggml_mul_mat(W_ctc, x) + b_ctc` (k=1 conv = mul_mat) | sensevoice `ctc_head` |
| Greedy decode | argmax + collapse-repeats + drop-blanks (blank=0) | `[N_tokens]` host ints | — | host-side loop on logits | adapt parakeet `decode_ctc_greedy` (blank=last-id → blank=0) |
| Detokenize | `LasrTokenizer.batch_decode` (SP) — keeps `</s>` id 2 in stream | string | `transcript.json` | shared `Tokenizer::decode_ctc` strips blanks pre-call | shared |

## Generation / KV Path

n/a — encoder-only CTC; no autoregressive state. `run_batch` fast path
batches utterances via the existing batch-axis (ne[3]) of the encoder
graph, identical to parakeet's CTC variant.

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| English transcribe | `generate` greedy CTC | same | "Transcribe / explicit en" + "Transcribe / auto" — both MUST PASS |
| Language detection | n/a (English-only) | reject `--language != en` with OK fallback to en | SKIP — not advertised |
| Translation | n/a | not exposed | SKIP — not advertised |
| Timestamps | n/a (CTC has none) | not exposed | SKIP — architecture has no timestamp head |
| Streaming | n/a (offline only) | not exposed | SKIP — `capabilities.streaming=false` |
| Batch (offline) | HF processor pads to common T_mel | implement `run_batch` via ne[3] batch axis on encoder graph (mirror parakeet CTC pattern); per-utterance valid-frame masks via `attn_pad_mask` / `conv_pad_mask` on each block | MUST PASS — text byte-identical, CPU tensor bit-exact at bs 2/4/8 |
| KenLM beam decode | shipped but not Stage-4 obligation | not exposed | SKIP — OUT OF SCOPE (first ship is greedy) |

## Deviations From Closest Analog

- **Macaron FF residual scalars are `[1.5, 0.5]`, NOT the standard `0.5`.** The shared `transcribe::conformer::macaron_ff_residual` helper hardcodes `x + 0.5 * FF(LN(x))` and is not reusable here. medasr hand-builds the FF1/FF2 residual as `ggml_add(ggml_scale(x, 1.5), ggml_scale(FF(LN(x)), 0.5))`.
- **Conv module residual scalars are `[2.0, 1.0]`.** `transcribe::conformer::conv_module` itself is callable for the inner pw1+GLU+dw+BN+SiLU+pw2 sub-block, but the surrounding scaled residual is hand-built (`ggml_add(ggml_scale(x, 2.0), conv)`).
- **Attention is RoPE, not relative-position.** `transcribe::conformer::rel_pos_mhsa` and `build_conformer_block` are not used. The Q/K projection + `ggml_rope_ext(..., GGML_ROPE_TYPE_NEOX, ...)` + softmax(QK/√d)·V + o_proj is implemented per-block in `src/arch/medasr/encoder.cpp` using the `causal_lm` pattern at `src/causal_lm/causal_lm.cpp:234-245`. `rope_theta=10 000`, `max_position_embeddings=10 000`.
- **`layer_norm` eps is `1e-6`, not the conformer-helper-hardcoded `1e-5`.** The shared helper uses `ggml_norm` with constant `kLayerNormEps = 1e-5f`. medasr loads `enc.layer_norm_eps` from the GGUF (`stt.medasr.encoder.layer_norm_eps`) and passes it through a local `medasr_layer_norm` shim that takes eps as a runtime argument.
- **All encoder LayerNorms are `bias=False`.** Every `_b` slot on the medasr block view stays nullptr; `layer_norm` is called with `beta=nullptr`. Same null-bias treatment as parakeet's FFN/attn linears; just applied to LN this time.
- **Subsampling is 1-D, not 2-D.** `LasrEncoderSubsampling` = `Linear(128→512) → ReLU → Conv1d(512,512,k=5,s=2) → ReLU → Conv1d(512,256,k=5,s=2) → ReLU → Linear(256→512)`. The leading `Linear(128→512)` reduces the mel dim to d_model BEFORE the convs, so the convs operate on `[T, 512]` channels (not `[F=128, T]` 2-D as in parakeet's DwStridingSubsampling). Two stride-2 convs → 4x effective downsampling. `transcribe::conformer::build_pre_encode` is not reusable; medasr writes its own `build_subsampling` in `encoder.cpp` using `conv_1d_f32`.
- **CTC head is `Conv1d(512→512, k=1)` with bias, not `Linear` no-bias.** Numerically equivalent to a linear projection but stored as `Conv1d`. The converter normalizes the tensor name to `ctc.proj.weight` (transposed if necessary to mul_mat-friendly layout); C++ runs `ggml_add(ggml_mul_mat(W, x), b)`.
- **CTC blank id = 0 (`<epsilon>`), NOT `vocab_size - 1`.** Parakeet/sensevoice convention is `blank = vocab_size - 1`. medasr reads `stt.medasr.ctc.blank_id` from the GGUF and uses it in greedy collapse. Hard-coded `last_id` assumptions in the shared CTC decoder are NOT reused — medasr ships its own greedy collapse in `decode_ctc.cpp` (or inlined in `model.cpp::run`).
- **Tokenizer specials match SentencePiece ids 0..3** (`<epsilon>`, `<s>`, `</s>`, `<unk>`). The reference WER scorer (`scripts/wer/run_reference_medasr_transformers.py`) uses `processor.batch_decode(token_ids, skip_special_tokens=True)`. The validate.py reference dumper does the same (changed from the README's non-skipping variant, see commit), so both reference paths now agree. The C++ greedy CTC collapse mirrors this by skipping ids 1 (`<s>`), 2 (`</s>`), and 3 (`<unk>`) in addition to the blank id 0. Without this skip, the CTC head's occasional last-frame `</s>` prediction leaks the literal string into the transcript and lifts dataset WER by ~0.2pp on LibriSpeech test-clean.
- **Conv-module depthwise dispatch is `direct_dw_in_block=true` (NOT the im2col path).** LASR's `conv_kernel=32` "same" padding is asymmetric `(15, 16)`; the conformer-helper's asymmetric-pad path uses `ggml_fill + ggml_concat` per side, which is NOT batch-stable in ggml CPU (drift ~1.4e-6 at the encoder output between B=1 and B>1). The direct depthwise op (`ggml_conv_2d_dw_direct`) handles the same kernel + symmetric inner-padding shape and is bit-exact across batch sizes. The trade-off is a Metal note: direct depthwise on k=32 has not been profiled on Apple GPU and may need a fallback there; CPU + Vulkan + CUDA are fine.

## Variant Notes

- `medasr`: single variant; no per-variant deviations. The `variants` list in `intake.json` has length 1. `config.varying_across_variants` is empty.
