# Forward map - gigaam

Reference: `salute-developers/GigaAM` package (commit 6e4b027c, pinned in `scripts/envs/gigaam/pyproject.toml`); per-variant HF SHA pinned in `scripts/dump_reference_gigaam_author.py::_VARIANT_REVISIONS`.
Closest in-tree analog: `src/arch/parakeet/` (Conformer + RNN-T / CTC), plus `src/arch/qwen3_asr/` for NEOX-mode rotary patterns.

Family baseline: 16-layer Conformer encoder, d_model=768, 16 heads, d_k=48, rotary positional embedding, LayerNorm in conv module, conv1d-based ×4 subsampling. RNN-T variants stack a 320-d 1-LSTM predictor + joint net (ReLU + Linear). CTC variants stack a single 1×1 Conv1d head. Upstream `v3_ssl` (encoder-only HuBERT-CTC pretraining checkpoint) is intentionally out of scope; transcribe.cpp has no encoder-output emission path.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Audio → log-mel | `gigaam.preprocess.FeatureExtractor` | `[B=1, n_mels=64, T_mel]` | `frontend.mel.out` | STFT (hann_periodic, center=false, win=320, hop=160, n_fft=320) → mag² → htk mel filterbank (baked `frontend.mel_filterbank` [64, 161]) → `log(clamp(x, 1e-9, 1e9))`. **Note `center=false`** (no reflect-pad either side), distinct from Whisper/qwen3. | `src/transcribe-mel.cpp` (new code path; existing frontend assumes center=true) |

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Pre-encode (StridingSubsampling) | `gigaam.encoder.StridingSubsampling` | `[B, T/4, 768]` | `enc.subsample.out` | conv1d(in=64, out=768, k=5, stride=2) → ReLU → conv1d(in=768, out=768, k=5, stride=2) → ReLU. Input is transposed [B, T, 64] → [B, 64, T] for conv1d, transposed back at exit. | `src/arch/parakeet/encoder.cpp::pre_encode` (parakeet uses 4-conv2d; gigaam is simpler 2-conv1d) |
| Rotary PE bank | `gigaam.encoder.RotaryPositionalEmbedding.create_pe` | `pe = [cos_bank; sin_bank]` shape `[2*L, 1, 1, d_k=48]` | `enc.pos_emb` | Build once at load time from `inv_freq = 1.0 / base^(2i/d_k)`, `freqs = positions ⊗ inv_freq`, `emb = cat(freqs, freqs)`. Store `[cos; sin]` stacked. Slice per call. **GigaAM applies rotary to x BEFORE Wq/Wk** (see deviation note). | `src/arch/qwen3_asr/decoder.cpp::apply_rope_neox` is the rotation primitive; the pre-projection placement is novel. |
| Block (×16; gate first/mid/last = 0/7/15) | `gigaam.encoder.ConformerLayer.forward` | `[B, T, 768]` | `enc.block.{0,7,15}.out` | Macaron FF1 (norm, FF, ×0.5 + residual) → norm-attn (LayerNorm, rotary self-attn, residual) → norm-conv (LayerNorm, ConformerConvolution, residual) → Macaron FF2 (norm, FF, ×0.5 + residual) → final LayerNorm `norm_out`. Final block output is the LN of the residual stream. | `src/arch/parakeet/encoder.cpp::conformer_block` |
| FeedForward | `gigaam.encoder.ConformerFeedForward` | `[B, T, 768]` | (inside block) | linear1 [D → 4D] → SiLU → linear2 [4D → D]. Both have bias. | `parakeet/encoder.cpp` FF (note: parakeet uses Swish, GigaAM uses `nn.SiLU` which is the same op) |
| Self-attention (rotary) | `gigaam.encoder.RotaryPositionMultiHeadAttention.forward` | `[B, T, 768]` | (inside block) | (1) reshape x to [T, B, H, d_k]; (2) apply rotary to x_pre on the d_k axis with NEOX split-halves: `rtt_half(x) = [-x[..., d_k/2:], x[..., :d_k/2]]`; rotated = `x*cos + rtt_half(x)*sin`; (3) reshape back to [B, T, D]; (4) Q = Wq @ x_rot, K = Wk @ x_rot, V = Wv @ x; (5) SDPA(Q, K, V, mask); (6) Wo. **No `linear_pos`, no `pos_bias_u/v`** (those are rel_pos-only). | `qwen3_asr/decoder.cpp` for `ggml_rope_ext(NEOX)`; pre-projection placement requires a new layout (see Deviations). |
| Conv module | `gigaam.encoder.ConformerConvolution.forward` | `[B, T, 768]` | (inside block) | x.transpose(1,2) → pointwise_conv1 [D → 2D] → GLU(dim=1) [2D → D] → depthwise_conv1d(k=5, padding=2, groups=D) → LayerNorm on channel axis (`x.transpose(1,2)` for LN, transpose back) → SiLU → pointwise_conv2 [D → D] → x.transpose(1,2). **conv_norm_type=layer_norm** (the source state_dict key is named `conv.batch_norm.*` for legacy reasons; converter renames to `conv.ln.*`). | parakeet `conv_norm_type=layer_norm` path (nemotron-streaming uses LN). |
| Encoder output | `gigaam.encoder.ConformerEncoder.forward` | `[B, 768, T/4]` | `enc.out` | After last block, transpose [B, T/4, 768] → [B, 768, T/4]. C++ port keeps either convention as long as gate tensor matches reference. | `parakeet/encoder.cpp` (final transpose handled at decoder boundary). |

## Decoder

### RNN-T variants (`gigaam-v3-e2e-rnnt`, `gigaam-v3-rnnt`)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Predictor LSTM | `gigaam.decoder.RNNTDecoder` (LSTM 1L, hidden=320, input=320) | `[B, U, 320]` | (per-step inside greedy loop) | Embed(num_classes, 320) → LSTM 1 layer. Bias is collapsed `bias_ih + bias_hh` in converter. Gate order PyTorch (i, f, g, o). Start state = embed of `<blank>` index. | `src/arch/parakeet/decoder.cpp` RNN-T predictor (single layer; parakeet has 1 or 2 layers). |
| Joint enc proj | `gigaam.decoder.JointNet.enc` | `[B, T_enc, 320]` | `rnnt.encoded` (input to joint) | Linear [768 → 320] applied to encoder output. | `parakeet/decoder.cpp` joint.enc. |
| Joint pred proj | `gigaam.decoder.JointNet.pred` | `[B, U, 320]` | (per-step) | Linear [320 → 320] applied to predictor output. | `parakeet/decoder.cpp` joint.pred. |
| Joint combine + output | `gigaam.decoder.JointNet.joint_net = Sequential(ReLU, Linear)` | `[B, T_enc, U, num_classes]` | (per-step) | Sum the two projections (broadcast), ReLU, Linear [320 → num_classes]. **Note GigaAM joint_net has only 2 elements (ReLU, Linear); parakeet TDT has 3 (Linear, ReLU, Linear).** Converter renames `joint.joint_net.1.*` → `joint.out.*`. | `parakeet/decoder.cpp` joint.out (drop the extra Linear). |
| Greedy decoding loop | `gigaam.decoding.RNNTGreedyDecoding.decode` | tokens, frames | `transcript.json` | Standard RNN-T greedy: for each encoder frame t in T_enc, emit symbols (up to `max_symbols=10`) until joint argmax = blank, then advance t. State carried across symbols within a frame; reset on blank. | `parakeet/decoder.cpp::rnnt_greedy_decode`. |

### CTC variants (`gigaam-v3-e2e-ctc`, `gigaam-v3-ctc`)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| CTC head | `gigaam.decoder.CTCHead` | `[B, T_enc, num_classes]` | `ctc.logits.raw` (pre-softmax), `ctc.log_probs` (post-softmax) | Single Conv1d (in=768, out=num_classes, kernel=1) on encoder output `[B, 768, T_enc]`, transposed to `[B, T_enc, num_classes]`. Then `log_softmax(dim=-1)`. | `parakeet/decoder.cpp::ctc_head`. |
| Greedy decode | `gigaam.decoding.CTCGreedyDecoding.decode` | tokens, frames | `transcript.json` | argmax over classes → collapse repeats → drop blanks. | `parakeet/decoder.cpp::ctc_greedy_decode`. |

## Generation / KV Path

RNN-T greedy is autoregressive over the symbol axis (within each encoder frame), but the predictor LSTM does not maintain a KV cache in the attention sense — it carries `(h, c)` state across symbols. No `dec.logits_raw.gen<N>` mid-generation tensor is needed for an RNN-T predictor; the LSTM state evolution is already gated by the predictor output at successive symbol indices. CTC has no generation step at all.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Symbol-loop state | RNN-T greedy decode | `(h, c)` per LSTM layer | (behavioral via `transcript.json`) | Standard LSTM state carry. No attention KV cache. | `parakeet/decoder.cpp` RNN-T loop. |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe Russian (explicit `--language ru`) | `gigaam.load_model(...).transcribe(wav)` returns cased+punct text for e2e_*, lowercased no-punct for non-e2e variants. | `transcribe-cli -m <gguf> --language ru <ru.wav>` should produce same. | "Transcribe (Russian)" (explicit) — must PASS. |
| Transcribe Russian (auto / no hint) | Monolingual; identical to explicit `ru`. | Same. | "Transcribe (Russian)" (auto) — must PASS. |
| Punctuation + casing | e2e variants emit cased+punct directly from SP vocab. | Inherent in tokenizer; no extra wiring. | "Punctuation + casing" — must PASS for e2e variants. |
| Translate | Not exposed (monolingual). | SKIP — not exposed by upstream. | SKIP row. |
| Language detection | Not exposed (monolingual). | SKIP — not exposed by upstream. | SKIP row. |
| Word timestamps | `gigaam` package exposes via `word_timestamps=True`; `modeling_gigaam.py` does not derive per-word timings. | ACCEPTED GAP — word-timestamp derivation lives in `gigaam` package code that we do not port. | ACCEPTED GAP row. |
| Segment timestamps (>25 s longform) | Upstream uses PyAnnote VAD. | ACCEPTED GAP — PyAnnote dependency. | ACCEPTED GAP row. |

## Deviations From Closest Analog (`src/arch/parakeet/`)

- **No relative-position artifacts in attention.** No `linear_pos`, `pos_bias_u`, `pos_bias_v`. Rotary PE replaces relative-pos shifting. Encoder block table has 34 tensors per layer vs parakeet's 41 (with biases). The `att_context_*`, `xscaling`, `conv_context_*` KV that parakeet emits are not applicable here.
- **Rotary mode and placement.** NEOX split-halves rotation (`rtt_half(x) = [-x[..., d_k/2:], x[..., :d_k/2]]`). **GigaAM applies rotary to pre-projection x rather than to post-projection Q/K** — Wq/Wk are applied to the rotated input, not to the original input then rotated. This is non-commutative (since Wq mixes across heads) so we must follow the reference order: reshape x to [T, B, H, d_k], rotate, reshape back to [B, T, D], project. Same rotated input feeds both Wq and Wk (since `query == key == x` at this point in self-attn); Wv is applied to unrotated x.
- **Conv module uses LayerNorm.** The source attribute is named `batch_norm` for legacy NeMo-naming reasons but is `nn.LayerNorm` at runtime (no `running_mean`/`running_var` in state_dict). Converter renames to `enc.blocks.{i}.conv.ln.*`. Parakeet has a similar code path via `nemotron-speech-streaming-en-0.6b` (`conv_norm_type=layer_norm`).
- **Pre-encode is 2-conv1d, not 4-conv2d.** Simpler subsampling: `Conv1d(64→768, k=5, s=2) → ReLU → Conv1d(768→768, k=5, s=2) → ReLU`. Total factor 4. Parakeet uses Conv2d on a [1, n_mels, T] grid; gigaam treats the n_mels axis as the channel axis directly.
- **Frontend differences vs parakeet:**
  - `center=false` (no reflect-pad on either side of the audio). Distinct from Whisper/qwen3/parakeet.
  - `mel_scale=htk` (HTK frequency formula) and `mel_norm=null` (no slaney area normalization). The mel filterbank is baked into the GGUF (`frontend.mel_filterbank`) so the C++ frontend memcpy's the buffer; the `stt.frontend.mel_norm = "htk"` KV is documentary.
  - SpecScaler is `log(clamp(x, 1e-9, 1e9))`, not `log10` and not `librosa.power_to_db`. Clamp constants emitted as `stt.frontend.log_clamp_{min,max}`.
- **Joint net is 2-element Sequential.** `Sequential(ReLU, Linear)` instead of parakeet TDT's `Sequential(Linear, ReLU, Linear)`. No TDT durations head; gigaam is pure RNN-T (or pure CTC).
- **Predictor hidden is 320 (smaller).** Parakeet uses 640.
- **Charwise tokenizer path** for `gigaam-v3-{rnnt,ctc}`. 33 character entries (Cyrillic + space) plus blank at index 33. C++ loader must accept `tokenizer.ggml.model = "char"` and detokenize by index → piece concatenation with no merge rules.

## Variant Notes

- `gigaam-v3-e2e-rnnt`: family baseline. RNN-T head, 1024-piece SP tokenizer with punctuation/casing (num_classes = 1024 + 1 blank = 1025). Acceptance gate: Stage 7 FLEURS ru.
- `gigaam-v3-e2e-ctc`: CTC head replacing RNN-T predictor + joint. 256-piece SP tokenizer with punctuation/casing (num_classes = 257). Encoder graph unchanged; head wiring swaps.
- `gigaam-v3-rnnt`: RNN-T head; charwise 33-entry vocab (num_classes = 34). Tokenizer path changes from SP to character-table lookup.
- `gigaam-v3-ctc`: CTC head + charwise 33-entry vocab (num_classes = 34).
- **Encoder weights are NOT shared across variants.** Stage 2 dumps showed different `enc.out` magnitudes per variant (e2e_rnnt: ±2.46, rnnt: -1.89/+2.94). Per-head fine-tuning drifts the encoder; convert each variant to its own GGUF rather than factoring out a shared-encoder GGUF.
