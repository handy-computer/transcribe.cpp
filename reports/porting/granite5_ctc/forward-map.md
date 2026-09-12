# Forward map - granite5_ctc

Reference: `transformers==5.17.0`
`models/granite_speech5/{modeling,feature_extraction}_granite_speech5.py`
(generated from `modular_granite_speech5.py`) @ HF revision
`18ca3c1de6cd092b5a30c39fb0f04550b38ed1a0`
Closest in-tree analog: `src/arch/granite/` (same Granite Conformer block and the
shared `src/granite_conformer/shaw_attn.h`); CTC head + greedy collapse from
`src/arch/parakeet/`.

Architecture pattern `encoder-ctc`: one encoder forward, one tied linear head,
argmax + CTC collapse. No decoder, no KV cache, no autoregressive loop.

## Frontend

`GraniteSpeech5FeatureExtractor._extract_features`. All of it is host-side C++
(`granite5_ctc::compute_encoder_input`); the graph starts at the stacked
320-wide feature.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|---|---|---|---|---|---|
| Frame budget | `feature_extraction:_extract_features` L215-219 | scalar | — | `mel_frames = n/hop`; `num_frames = 2*ceil(mel_frames/2)`; right-pad PCM to `(num_frames-1)*hop+1` | none — new |
| log-mel | same, L220 (`mel_filters`) | `[80, num_frames]` | via `mel.in` | `MelFrontend` `normalize=per_utterance`, `window=hann_periodic`, `pad_mode=reflect`, htk fbank; baked `frontend.{mel_filterbank,window}` | `granite::compute_mel_encoder_input` |
| log10 + floor + affine | same, L221-223 | `[80, num_frames]` | via `mel.in` | inside `MelFrontend` per_utterance: `log10`, floor at `max-8`, `(v+4)/4` | `transcribe-mel.cpp` L741 |
| Frame-count override | same, L220 (`[..., :num_frames]`) | `[80, num_frames]` | via `mel.in` | new trailing `out_frames` arg on `MelFrontend::compute`; `num_frames` is `n_frames-1` for even mel_frames and `n_frames` for odd | none — new |
| Deltas | same, L224 (`compute_deltas`) | `[160, num_frames]` | via `mel.in` | host loop `d[t] = (x[min(t+1,T-1)] - x[max(t-1,0)])/2`; concat on the channel axis | none — new |
| 2-frame stack | same, L225-227 | `[T_enc, 320]`, `T_enc = num_frames/2` | `mel.in` | host interleave `[logmel(2t) ‖ delta(2t) ‖ logmel(2t+1) ‖ delta(2t+1)]` | `granite::compute_mel_encoder_input` (stack only) |

## Encoder

`GraniteSpeech5Encoder.forward`. 16 blocks; blocks 0 and 1 are
`GraniteSpeech5EncoderSubsamplingBlock`, 2-15 are `GraniteSpeech5EncoderBlock`.
`T0 = T_enc`, `T1 = T0/2`, `T2 = T1/2` (integer division); blocks 2-15 run at `T2`.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|---|---|---|---|---|---|
| input_linear | `modeling:GraniteSpeech5Encoder.forward` (`self.input_linear`) | `[1024, T0]` | `enc.input_linear.out` | `ggml_mul_mat` + bias | `granite/encoder.cpp` |
| pos_rows | `compute_attention_dists` | `[255]` i32 | — | host `rows[d] = clamp(ctx-1-d, ±ctx) + max_pos`, one row per relative offset; expanded by the skew path in `shaw_block_attn` | `granite::precompute_pos_rows` (same formula; see Deviations) |
| Block: FF1 macaron | `GraniteSpeech5EncoderBlock.forward` L1-3 | `[1024, T]` | `enc.block.{0,1,2}.post_ff1` | `conformer::macaron_ff_residual` (LN → lin1 → SiLU → lin2, `x + 0.5*ff`) | `granite_macaron` |
| Block: Shaw block-local attn | `GraniteSpeech5EncoderAttention.forward` | `[1024, T]` | `enc.block.{0,1,2}.post_attn` | `granite_conformer::shaw_block_attn` verbatim (fused `attn.kv`, pad-to-block, `-INF` on pad keys, slice before `o_proj`) | `granite/encoder.cpp` |
| Block: conv module | `GraniteSpeech5EncoderConvolutionModule.forward` | `[1024, T_out]` | `enc.block.{0,1,2}.post_conv` | LN → pointwise1 (1024→4096) → GLU → depthwise k=7 stride s → fused BN → SiLU → pointwise2 | `granite_conv_module` (stride 1 only) |
| Block: subsampling residual | `GraniteSpeech5EncoderSubsamplingBlock.forward` (`unfold(1,2,2).mean(-1)`) | `[1024, T/2]` | `enc.block.{0,1}.post_conv` | mean-pool pairs via `ggml_pool_1d` over a `[T, C]` view; trim conv output to the pooled length | none — new |
| Block: FF2 macaron | block L11-12 | `[1024, T_out]` | `enc.block.{0,1,2}.post_ff2` | `conformer::macaron_ff_residual` | `granite_macaron` |
| Block: norm_out | block L14 | `[1024, T_out]` | `enc.block.{0,1,2,7,8,15}.out` | `conformer::layer_norm` | `granite/encoder.cpp` |
| Self-conditioned CTC | encoder forward (`layer_idx + 1 == num_hidden_layers // 2`) | `[1024, T2]` | `enc.ctc.mid_logits`, `enc.ctc.mid_injection` | `ctc_proj` → `ggml_soft_max` over ne0 → `ctc_bypass` → add. Fires after block 7; `enc.block.7.out` is captured BEFORE the add | `granite/encoder.cpp` (bypass) |
| Encoder output | encoder forward return | `[1024, T2]` | `enc.out` | — | `granite/encoder.cpp` |

## Decoder

There is no decoder. `GraniteSpeech5ForCTC.forward` is one tied linear.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|---|---|---|---|---|---|
| CTC head | `GraniteSpeech5ForCTC.forward` (`self.ctc_head`) | `[16384, T2]` | `enc.ctc_logits` | `ggml_mul_mat(enc.ctc_proj.weight, enc.out)` + bias — the SAME tensor as the mid-layer head (`_tied_weights_keys`) | `granite/encoder.cpp` bypass proj |
| Greedy argmax | `GraniteSpeech5ForCTC.generate` (`logits.argmax(-1)`) | `[T2]` ids | `transcript.json` tokens | host argmax over ne0 | `parakeet/decoder.cpp` CTC path |
| CTC collapse | `ParakeetTokenizer._decode` (`itertools.groupby`, then drop `pad_token_id`) | token list | `transcript.json` | host: drop id equal to previous frame's id, then drop blank(0) | `parakeet/decoder.cpp` L1622 |
| Detokenize | `TokenizersBackend._decode` (ByteLevel) | text | `transcript.json` | `Tokenizer::decode`, `tokenizer.ggml.pre=granite` | shared tokenizer |

## Generation / KV Path

Not applicable: non-autoregressive. No `dec.logits_raw.gen<N>` coverage is
required (the Stage-4 mid-generation rule is scoped to KV-cache decoders).

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|---|---|---|---|---|---|
| — | — | — | — | single forward, no cache | — |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|---|---|---|---|
| Transcribe (explicit lang hint) | English-only; no language conditioning anywhere in the graph | `--language en` accepted and ignored | Transcribe (explicit lang hint) |
| Transcribe (auto/no hint) | identical forward | identical output to the hinted run | Transcribe (auto/no hint) |
| Batch (offline) | `processor(list_of_clips)` builds a padded batch + `attention_mask`; the mask halves after each subsampling block and masks pad key columns | `run_batch()` with per-utterance frame counts | Batch (offline) |
| Word timestamps | not advertised; CTC emission peaks are not word durations | unsupported | Word timestamps |
| Language detection | none | unsupported | Language detection |
| Translate | none | unsupported | Translate |
| Result segmentation | one decoded string | one untimed whole-clip text segment | Segment timestamps |
| Streaming | non-streaming; the frontend is centered and the per-utterance log-mel floor is global | unsupported | Streaming |
| Speaker diarization | none | unsupported | Speaker diarization |

## Deviations From Closest Analog

- **Shaw positional bias: NOT a deviation (corrected).** An earlier
  revision of this map claimed granite 5.0's `compute_attention_dists` has
  the opposite sign to granite 4.x. It does not. granite 4.x writes
  `dists[c*ctx + r] = clamp(c - r)` with `c` the OUTER (query) index and
  `r` the inner (key) index; granite 5.0 writes `dists[q*ctx + k] =
  clamp(q - k)`. Renaming c->q, r->k makes them the same code, and the two
  arrays are bit-identical for any `(context_size, max_pos_emb)`. Both are
  `query - key` at ne0 = key, ne1 = query, which is what `shaw_block_attn`
  expects. Both families now share one `precompute_pos_rows` formula.
  A genuinely mirrored fill would transpose the bias and is invisible on
  short clips (it only shows once a block is fully populated), so the
  guard against that remains `enc.block.2.post_attn` on `dots`.
- **In-block time subsampling.** granite 4.x has no subsampling block. Blocks 0
  and 1 run attention at the input rate, then halve inside the conv residual:
  `pooled = mean of frame pairs` (trailing odd frame dropped) and
  `x = pooled + conv_out[:, :len(pooled)]`. The depthwise conv runs at stride 2
  and emits `ceil(T/2)`, so for odd `T` the conv output is one frame longer than
  the pooled residual and must be trimmed, not the other way round.
- **Three attention geometries per forward.** Because `T` changes at blocks 0
  and 1, the block-local pad mask and zero-pad tile differ per stage. granite
  4.x builds one; granite5_ctc builds one per distinct `T` (3 total) and indexes
  them by stage.
- **Frontend.** granite 4.x is 80 mel × 2 stack = 160 with the trailing odd
  frame dropped (floor). granite5_ctc is (80 mel + 80 delta) × 2 stack = 320
  with the trailing group filled (ceil) by right-padding the waveform. The
  ceil/floor difference is exactly why `MelFrontend::compute` needs the explicit
  `out_frames` argument.
- **FFN bias.** `config.attention_bias=true` puts a bias on both FFN linears in
  every block (granite 4.x has them too) but NOT on q/k/v — the encoder
  attention hard-codes `bias=False` there. Only `o_proj` carries an attention
  bias.
- **Tied CTC head.** `ctc_head` and `encoder.out` are the same storage. The
  converter emits `enc.ctc_proj` once; the graph uses it at both sites.
- **Depthwise conv helper.** The block uses `conformer::conv_2d_dw_f32` on a
  (W, H=1, C, N) reshape rather than `conformer::conv_1d_dw_f32`. The 1-D
  helper switches algorithm on the batch size — im2col + mul_mat at B == 1,
  the direct depthwise op at B > 1 — which made a batched utterance differ
  from its single-shot self by ~7e-3 relative and failed the batch
  tensor-parity gate even though the text was byte-identical. The 2-D form
  takes the same im2col path at every batch size, so single-shot and batched
  are bit-identical.
- **CTC collapse lives in the tokenizer, not the model.** `generate()` only
  argmaxes; the run-length collapse is `ParakeetTokenizer._decode`
  (`itertools.groupby` then drop `pad_token_id`), which is why
  `tokenizer_config.json`'s `tokenizer_class` is load-bearing. Decoding with a
  plain byte-level BPE tokenizer emits every repeated frame.

## Variant Notes

- `granite-speech-5.0-470m-turboctc`: the family baseline; only variant at
  Stage 4. `subsample_layers=[0,1]`, `num_hidden_layers=16` (self-conditioning
  after block 7), `context_size=128`, `max_position_embeddings=512`,
  `conv_kernel_size=7`, `conv_expansion_factor=2`.
