# Forward map - parakeet

Reference: NVIDIA NeMo (`ASRModel.from_pretrained` -> `EncDecRNNTBPEModel` /
`EncDecCTCModelBPE` / `EncDecHybridRNNTCTCBPEModel`) @ NeMo v2.7.x
Closest in-tree analog: `src/arch/parakeet/` itself — this stage extends
the existing TDT v1 port to cover RNNT, CTC, the 1.1B FastConformer-XL
geometry, and `encoder.use_bias=true`. No new in-tree analog is needed.

A compact map from the reference forward pass to the C++ port.
Family-level because all 10 parakeet variants share a Conformer encoder;
per-head differences (TDT vs RNNT vs CTC) and per-variant geometry deltas
go in "Variant Notes" at the bottom.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Mel / log-mel filterbank | `model.preprocessor` (`AudioToMelSpectrogramPreprocessor` -> `FilterbankFeatures`, n_fft=512, win=400, hop=160, hann_periodic, preemph=0.97, per-feature CMVN) | `[T_mel, n_mels]` | `enc.mel.in` | `transcribe::MelFrontend` constructed once at load() from `stt.frontend.*` KV | unchanged from existing parakeet |

`n_mels` is **80** for every new variant **except** `unified-en-0.6b`
(128). Existing `tdt-0.6b-v2/v3` use 128. Frontend is otherwise identical
across the family.

## Encoder

Shared FastConformer encoder. The two existing geometry knobs are
`subsampling_factor` (always 8) and `n_mels` (80 or 128). The new knobs
introduced by Stage 4 are `n_layers` (17/24/42), `d_model` (512/1024),
`subsampling_channels` (256), and `use_bias` (true/false).

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Pre-encode subsampling (factor 8 = 3x stride-2 conv) | `model.encoder.pre_encode` (dw_striding) | `[T_enc, d_model]` | `enc.pre_encode.out` | `conf::build_pre_encode` over `PreEncodeView`. `pre_encode_freq = subsampling_channels * (n_mels / 8)` — 256·10=2560 (n_mels=80) or 256·16=4096 (n_mels=128). Final linear projects to d_model. | existing parakeet |
| Relative positional embedding | `RelPositionalEncoding` | `[2*T_enc-1, d_model]` | `enc.pos_emb` | host-built sin/cos table, uploaded as graph input | unchanged |
| Conformer block (FF1 → rel-pos MHSA → ConvModule(BN) → FF2 → LN) | `model.encoder.layers[i]` | `[T_enc, d_model]` | `enc.block.{0, n/2, n-1}.out` | `conf::build_conformer_block` over `BlockView`. Bias slots populated only when `enc_use_bias=true` (11 biases per block: ff1.linear1/linear2, attn.q/k/v/out, conv.pw1/dw/pw2, ff2.linear1/linear2). | existing parakeet — extended for biases |
| Final encoder LN | implicit in `layers[-1]` (norm_out) | `[T_enc, d_model]` | `enc.final` | `conf::named` on the final tensor | unchanged |

Block-spot dump indices vary by encoder depth: layers in {0, n/2, n-1}.
For n=17: {0, 8, 16}. For n=24: {0, 12, 23}. For n=42: {0, 21, 41}. The
`docs/porting/families/parakeet.md` Stage-2 entry already lists these.

## Decoder

Three head kinds dispatched on `stt.parakeet.head_kind` (default `tdt`
when KV missing, for the legacy v2/v3 GGUFs). All three host-side
implementations live in `src/arch/parakeet/decoder.cpp`.

### TDT (v2, v3, tdt-1.1b, tdt_ctc-110m, tdt_ctc-1.1b)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Predictor embed | `model.decoder.prediction.embed` | `[1, pred_hidden]` (per token) | `dec.embed.0` | `ggml_get_rows`-equivalent (host fp32 lookup) | existing parakeet |
| Predictor LSTM (1 or 2 layers) | `model.decoder.prediction.dec_rnn` | `[pred_hidden]` per layer (h, c) | `dec.lstm.{layer}.{h,c}.0` | hand-rolled fp32 LSTM step in decoder.cpp | existing parakeet — pred_n_layers=1 first exercised by tdt_ctc-110m |
| Joint enc/pred projections + activation + out | `model.joint` (TDTJoint) | `[vocab+1+num_extra]` per (t, u) | `dec.joint.0` | fp32 matmul + relu/sigmoid/tanh + matmul on host | existing parakeet |
| TDT decode loop (greedy) | `RNNTGreedyDecodeBatched` with TDT mode | per-emit token + duration | (tokens) | `decode_tdt_greedy` | existing parakeet |

### RNNT (rnnt-0.6b, rnnt-1.1b, unified-en-0.6b)

Predictor + joint identical in shape to TDT, but the joint emits
`vocab+1` logits (no duration extra-output channels). Decode loop is the
same shape minus duration:

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Predictor embed | `model.decoder.prediction.embed` | `[1, pred_hidden]` | `dec.embed.0` | reuse TDT predictor mirror | TDT path above |
| Predictor LSTM | `model.decoder.prediction.dec_rnn` | `[pred_hidden]` | `dec.lstm.{layer}.{h,c}.0` | reuse TDT path | TDT path above |
| Joint | `model.joint` (RNNTJoint, no extras) | `[vocab+1]` | `dec.joint.0` | reuse TDT path with `joint_num_extra_outputs=0`. Allowed by existing hp validation. | TDT path above |
| RNNT decode loop (greedy) | `RNNTGreedyDecodeBatched` (no TDT mode) | per-emit token | (tokens) | `decode_rnnt_greedy`: blank → advance 1 frame, non-blank → emit + stay (max_symbols cap, default 10) | new — sibling of `decode_tdt_greedy` |

### CTC (ctc-0.6b, ctc-1.1b)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| CTC head (1×1 Conv1d) | `model.decoder.decoder_layers.0` | `[T_enc, vocab+1]` | `dec.ctc.logprobs` (full), `dec.ctc.logprobs.0` (frame 0) | `head.ctc.weight @ enc[t] + head.ctc.bias` per frame, then log_softmax. Computed host-side from the encoder readback. | new — minimal host head |
| CTC greedy decode (per-frame argmax + run-length collapse + drop blank) | `model.decoding.greedy_search` (CTC) | token sequence | (tokens) | `decode_ctc_greedy` | new |

## Generation / KV Path

Parakeet has no autoregressive transformer KV cache. The TDT/RNNT
predictor LSTM has a step-pair `(h, c)` per layer that is committed on
every non-blank emit and rolled back on blank — host-managed in
`decoder.cpp`. CTC has no recurrent state. The Stage 4
mid-generation-gate-tensor requirement (decoder KV correctness past
n_past>0) does not apply to this family — there is no transformer KV
cache to mis-index.

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (en) | `transcribe()` returns greedy text | `transcribe-cli -m <gguf> --language en <wav>` produces non-empty English transcript | one row per variant |
| Punctuation / casing (tdt_ctc-1.1b, tdt_ctc-110m, unified-en-0.6b) | upstream training data preserves PnC | non-empty transcript with capital letters and `,.?!` | one row per PnC variant |
| Streaming (unified-en-0.6b) | shared offline+streaming weights | n/a — offline only in v1 | ACCEPTED GAP — streaming infra deferred |
| Word timestamps | NeMo word-aligner | derived host-side from emit-frame indices (TDT/RNNT) or per-frame argmax (CTC) | PASS — same code path as the existing v2/v3 word-timestamp gate, no per-variant differences |
| Translation, lang-detect, VAD | not advertised | n/a | n/a |

## Deviations From Closest Analog

- **head_kind dispatch**: same family handler covers TDT (predictor+joint+durations), RNNT (predictor+joint, no durations), and CTC (single 1×1 conv head). Loader reads `stt.parakeet.head_kind` (string KV); legacy v2/v3 GGUFs lack the KV and default to `"tdt"`. Predictor/joint/tdt-durations KV reads are conditional on the resolved head_kind so a CTC GGUF does not fail at "predictor.hidden missing".
- **`encoder.use_bias` per-variant**: existing v2/v3 ship with `use_bias=false` (every linear/conv biased); the 8 new variants ship with `use_bias=true`, contributing 11 biases per block. The shared `transcribe::conformer::BlockView` already exposes nullable bias slots; loader populates them only when the hparam is true.
- **n_mels=80 path**: every new variant except `unified-en-0.6b` uses 80-bin mels (vs 128 on v2/v3/unified-en). The pre-encode chain handles arbitrary even multiples of 8 mels — the existing "only 8/128 implemented" guard in `encoder.cpp` was a safety check from Phase 4 step 3a, now removed.
- **CTC head storage**: a single `(vocab+1, d_model, 1)` conv tensor flattened to `head.ctc.weight` + `(vocab+1,)` `head.ctc.bias` in the GGUF. Host decoder mirrors only these two tensors when head_kind=ctc; predictor and joint mirrors stay empty.
- **TDT-CTC hybrids (`tdt_ctc-110m`, `tdt_ctc-1.1b`)**: the upstream checkpoints carry both heads, but the converter emits TDT-only (the auxiliary CTC head is silently dropped — pure CTC variants cover that path). At Stage 4 these load through the standard TDT path with `head_kind="tdt"`.
- **Streaming attention (`att_context_style` = "chunked_limited")**: introduced by `nemotron-speech-streaming-en-0.6b`. NeMo's cache-aware streaming model trains with chunked attention but keeps the full `RelPositionalEncoding` (pos_emb buffer at 2T-1, not the shortened `LocalAttRelPositionalEncoding` size). The "regular" style is the existing path used by every other variant. Loader reads `stt.parakeet.encoder.att_context_style` as optional with default `"regular"` so legacy GGUFs are unaffected; only `att_context_style == "chunked_limited"` selects the new mask, computed as `chunk_size = att_context_right + 1`, `left_chunks = att_context_left / chunk_size` (every k-frame whose chunk index is in `[q_chunk - left_chunks, q_chunk]` is allowed). The mask is built host-side and threaded as a graph input that broadcasts across heads; added to `matrix_bd` before `flash_attn_ext`.
- **Causal depthwise conv (`conv_context_left`, `conv_context_right`)**: streaming-friendly Conformer convolution. NeMo's offline default is centered `(k-1)/2` on both sides; nemotron-streaming uses `[k-1, 0]` (left=8, right=0 for k=9) so the depthwise convolution does not look ahead. Loader reads both as optional with default `-1`; when `-1`, the conformer uses centered `(k-1)/2`. Otherwise depthwise conv padding becomes `(left, right)`.
- **LayerNorm in conv module (`conv_norm_type` = "layer_norm")**: streaming-stable replacement for BatchNorm. Same tensor names (`conv.bn.weight` / `conv.bn.bias`) but the `running_mean` / `running_var` tensors are absent and the per-batch statistics are computed online (it really is a LayerNorm over `d_model`, not a fused affine). Loader reads `stt.parakeet.encoder.conv_norm_type` as optional with default `"batch_norm"`; when `"layer_norm"`, the BN-fusion step is skipped and `conv_module` applies an unfused LayerNorm using the same scale/bias tensors.

## Variant Notes

- **`tdt-0.6b-v2`**, **`tdt-0.6b-v3`** (existing baseline): 24 layers, d_model=1024, n_mels=128, `use_bias=false`, TDT head, pred_n_layers=2. v3 multilingual (25 languages) — capability difference only; same forward graph.
- **`tdt-1.1b`**: 42 layers, d_model=1024, n_mels=80, `use_bias=true`, TDT head, pred_n_layers=2. **First exercise of the 1.1B FastConformer-XL geometry, the 80-mel pre-encode, and the encoder bias path.**
- **`tdt_ctc-1.1b`**: 42 layers, d_model=1024, n_mels=80, `use_bias=true`, TDT head, pred_n_layers=2. Same encoder geometry as `tdt-1.1b`, but **only variant in the family using local attention** (`att_context_size=[128,128]`, NeMo's `LocalAttRelPositionalEncoding`). Drives the pos_emb buffer length (257 instead of 2T-1) and adds a -INF band pad before `rel_shift` so out-of-window keys drop out of softmax.
- **`tdt_ctc-110m`**: 17 layers, d_model=512, n_mels=80, `use_bias=true`, TDT head, **pred_n_layers=1**. First family member with 1-layer predictor and the smaller "Medium" Conformer config (d_model=512, channels=256).
- **`rnnt-0.6b`**: 24 layers, d_model=1024, n_mels=80, `use_bias=true`, RNNT head, pred_n_layers=2. **First exercise of the RNNT decode loop (no durations).**
- **`rnnt-1.1b`**: 42 layers, d_model=1024, n_mels=80, `use_bias=true`, RNNT head, pred_n_layers=2. Sibling of rnnt-0.6b on the 1.1B encoder.
- **`unified-en-0.6b`**: 24 layers, d_model=1024, **n_mels=128** (the only new variant that keeps the v2/v3 mel geometry), `use_bias=true`, RNNT head, pred_n_layers=2. Streaming-shared weights but offline-only in v1.
- **`ctc-0.6b`**: 24 layers, d_model=1024, n_mels=80, `use_bias=true`, **CTC head**. **First exercise of CTC head loading and CTC greedy decode.**
- **`ctc-1.1b`**: 42 layers, d_model=1024, n_mels=80, `use_bias=true`, CTC head. Sibling of ctc-0.6b on the 1.1B encoder.
- **`nemotron-speech-streaming-en-0.6b`**: 24 layers, d_model=1024, **n_mels=128**, `use_bias=false`, RNNT head, pred_n_layers=2. **First variant exercising `att_context_style="chunked_limited"`, causal depthwise conv (`conv_context_left=8, conv_context_right=0` for kernel=9), `conv_norm_type="layer_norm"`, and `fe_normalize="none"` (NeMo's `normalize="NA"` no-op, canonicalised at convert time).** Frontend feature stats (per-feature mean/std) are baked into training rather than computed at inference, so the C++ mel frontend emits unnormalised log-mel and the loader accepts `normalize="none"` alongside `"per_feature"`. Streaming runtime knobs (lookahead, encoder chunk size, decoder context) are deferred to the streaming bring-up — offline-only in v1, same as `unified-en-0.6b`.
