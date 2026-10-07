# Forward map - sortformer

Reference: NeMo `SortformerEncLabelModel` (nemo.collections.asr.models.sortformer_diar_models) @ pin `6967f48fda2a`
Closest in-tree analog: src/arch/parakeet/ (shared NeMo ConformerEncoder for the FastConformer stage)

Compact map from the reference forward pass to the C++ port. Sortformer is an
`encoder-diarizer`: no tokenizer, no decoder, no text. The product is a `T x 4`
per-frame speaker-activity probability matrix (80 ms frames, arrival-order
columns, max 4 speakers). Two forward paths:

- **offline / full-context** (`forward` -> `forward_infer`): single-chunk,
  non-stateful. The C++ encoder must match this first. Gate tensors below.
- **streaming** (`forward_streaming`): AOSC speaker-cache + FIFO over 188-frame
  chunks. Needed for the streaming capability row and DER on long audio. See
  "Streaming path" section; only diverges from offline once T > chunk_len.

Dims (from GGUF KVs): conformer 17L d=512 h=8 dff=2048 k=9 sub=8 feat_in=128
BN; transformer 18L **post-LN** d=192 h=8 (head_dim=24) dff=768 ReLU, **no
final norm** (pre_ln=false); encoder_proj 512->192; diar fc1 192->192,
single_spk_head 192->4 (spk_head 384->4 is streaming-only).

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Mel log-spectrogram | NeMo `AudioToMelSpectrogramPreprocessor` (preprocessor.*, recomputed in C++) | [128, T_mel] (T_mel=1216 for 12s) | `enc.mel.in` | reuse parakeet MelFrontend: preemph 0.97, dither 1e-5, n_fft 512, win 400, hop 160, hann-periodic, 128 mel, `normalize=none` | parakeet `MelFrontend` |

Note: parakeet's NeMo preprocessor uses `normalize=per_feature`; Sortformer
uses `normalize=none`. Confirm the parakeet MelFrontend is parameterized on
normalize (KV `stt.frontend.normalize`) and does NOT hardcode per-feature.

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| FastConformer (pre_encode dw_striding x8 + 17 Conformer blocks) | `ConformerEncoder.forward` (encoder.*) | [T=150, 512] (ref emits [B,512,T], transposed to [T,512]) | `enc.fastconformer.out` | **verbatim reuse** of parakeet ConformerEncoder graph (macaron FF, rel-pos MHA w/ pos_bias_u/v, conv+BN, use_bias=true). Same tensor suffix map. | parakeet `encoder.cpp` |
| encoder_proj | `sortformer_modules.encoder_proj` (Linear 512->192) applied after `emb_seq.transpose(1,2)` | [150, 192] | `enc.encoder_proj.out` | `ggml_mul_mat(W, x) + b` | any linear |
| Transformer encoder (18x post-LN block) | `transformer/transformer_encoders.py::TransformerEncoder.forward` + `TransformerEncoderBlock.forward_postln` | [150, 192] | `enc.transformer.out` | per block below; no final norm | new (cohere decoder self-attn is closest MHA) |

### Transformer post-LN block (repeat x18)

Reference `forward_postln(x, mask)`:
```
a = MHA(q=x, k=x, v=x, mask)          # first_sub_layer
a = a + x                             # residual
a = LN1(a)                            # layer_norm_1  (eps 1e-5)
f = dense_out(relu(dense_in(a)))      # second_sub_layer (PositionWiseFF), 192->768->192
o = f + a                             # residual
o = LN2(o)                            # layer_norm_2  (eps 1e-5)
```
MHA (`transformer_modules.MultiHeadAttention`): q/k/v each `Linear(192,192)`
with bias (`attn.{q,k,v}`); `attn_scale = sqrt(sqrt(head_dim))`; q and k each
divided by attn_scale before `q@k^T` (=> scores/sqrt(head_dim)); `+ mask`
(0 attend / -10000 pad); softmax; `@v`; `out_projection` (`attn.out`, bias).
Gate tensors: verify block 0, 8, 17 output via TRANSCRIBE_DUMP_ALL_BLOCKS.

## Decoder

No decoder. The "head" is the diarization sigmoid stack.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Speaker sigmoids (offline) | `sortformer_modules.forward_speaker_sigmoids` | [150, 4] | `diar.preds_offline` | `relu(x)` -> `fc1`(192->192) -> `relu` -> `single_spk_head`(192->4) -> `sigmoid`; then `* mask` | new |

`forward_speaker_sigmoids(x)`:
```
h = relu(x)                 # dropout is identity at eval
h = fc1(h)                  # first_hidden_to_hidden, Linear(192,192)
h = relu(h)
s = single_spk_head(h)      # single_hidden_to_spks, Linear(192,4)
preds = sigmoid(s)
preds = preds * encoder_mask # zero padded frames
```

## Generation / KV Path

No autoregressive generation. N/A.

## Streaming path (forward_streaming) — IMPLEMENTED (Stage 4 streaming checkpoint)

Sync (`async_streaming=False`) stateful loop over chunks of `chunk_len`
diar-frames (chunk_len * subsampling_factor mel frames). Per chunk
(`forward_streaming_step`):

1. `encoder.pre_encode(chunk mel window incl. lc/rc context)` -> chunk_embs
   [T_diar, 512], **un-xscaled** (C++ Graph A: `conf::build_pre_encode`).
2. concat `[spkcache | fifo | chunk_embs]` (all 512-dim, host-owned state).
3. `frontend_encoder(bypass_pre_encode=True)`: xscale x sqrt(512) + rel pos_emb
   over T_concat + 17 conformer blocks (full attention) + `encoder_proj`
   512->192 (C++ Graph B: `conf::build_conformer_block` x17 + linear).
4. `forward_infer`: 18L post-LN transformer + `forward_speaker_sigmoids`
   (relu -> fc1 -> relu -> **single_spk_head** 192->4 -> sigmoid). NOTE: the
   inference path uses `single_hidden_to_spks` (192->4), NOT the 2*hidden
   `hidden_to_spks` (384->4) head, which is unused at inference. All frames are
   valid in sync mode so `apply_mask_to_preds` / encoder_mask is a no-op.
5. `streaming_update` (host): slice fifo/chunk preds, append the middle chunk
   to FIFO, pop `pop_out_len` frames FIFO->spkcache, update the silence
   profile, and `_compress_spkcache` when spkcache > spkcache_len.

C++ implementation: `src/arch/sortformer/model.cpp` (two-phase Graph A/B +
`run_streaming` driver) and `src/arch/sortformer/stream.cpp` (host
`streaming_update_sync` / `_compress_spkcache` / silence profile, exact ports of
`sortformer_modules.py`). The tensor-dump oracle runs on CPU, so `_compress_
spkcache`'s topk/sort tie-breaks target ATen CPU semantics (value desc,
flat-index asc; `-inf` picks mapped to `max_index` before the ascending sort).

Validation: single-chunk `diar.probs` == `diar.preds_offline` bit-close
(2.96e-4, oracle 150<chunk_len). Multi-chunk FIFO + AOSC compression exercised
via `VALIDATE_SORTFORMER_PRESET=small` on the oracle (uniform ~3.9e-4 F32
accumulation, no structural divergence) and end-to-end via AMI DER
(`scripts/diar/run_cpp_sortformer.py` -> dihard3 PP -> `score_der.py`).

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Streaming diarization <=4spk | `m.diarize()` streaming AOSC/FIFO | TODO: streaming hooks emit T x 4 | Streaming diarization (MUST PASS) |
| Offline diarization | `forward` single chunk | `run()` offline forward -> T x 4 | Offline diarization (MUST PASS) |
| Speaker-activity tensor | preds [T,4] | dump / result surface T x 4 | Speaker-activity tensor (MUST PASS) |
| Transcription/translation/timestamps | none (not a transcriber) | not exposed | OUT OF SCOPE rows |

## Deviations From Closest Analog

- No decoder, no tokenizer, no text result. Output is a float matrix; the
  transcript-oriented result/session API must carry it (see open decision:
  diar result surface vs tensor-dump-only for Stage 4).
- Transformer is **post-LN** (parakeet/cohere self-attn are pre-LN); LN sits
  AFTER each residual add, and there is no final norm.
- MHA scaling is split (q and k pre-divided by sqrt(sqrt(head_dim))) rather
  than scaling scores once; numerically equivalent, keep as-is for parity.
- Frontend `normalize=none` (parakeet uses per_feature).
- Encoder output is transposed [B,512,T]->[B,T,512] before encoder_proj.

## Variant Notes

- `diar_streaming_sortformer_4spk-v2.1`: family baseline (this port).
- `streaming-4spk-v2`: architecturally identical, CC-BY-4.0; add weights once
  v2.1 validated (sibling-variant shortcut, Step 1).
