# Forward map - sensevoice

Reference: `funasr.models.sense_voice.model.SenseVoiceSmall` and
`funasr.models.sense_voice.model.SenseVoiceEncoderSmall` (FunASR 1.3.1
@ pin recorded in intake; class hierarchy lives at
`funasr/models/sense_voice/model.py` plus `funasr/models/sanm/{attention,
encoder}.py` for the SAN-M block, `funasr/frontends/wav_frontend.py`
for the kaldifeat fbank + LFR + per-feature CMVN).

Closest in-tree analog: encoder pattern resembles `src/arch/parakeet/`
(conformer-style transformer encoder + depthwise conv branch + CTC-ish
head), but the SAN-M block, two-tier depth, and prefix-embedding
prepend are all new. Frontend has no in-tree analog — kaldi HTK fbank
is unlike whisper/parakeet slaney mel paths and unlike moonshine's
PCM-into-conv stem.

This is the first FunASR-native port and the first `encoder-ctc`
architecture pattern in the repo.

## Frontend

WavFrontend: kaldi fbank + low-frame-rate stack + per-feature CMVN.
None of these stages exist in the existing whisper / parakeet / cohere
mel paths.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Read raw PCM | runtime audio loader | `[T_samples]` f32 in `[-1, 1]` | — | shared `transcribe::audio::load_pcm` | shared |
| Upscale (×32768) | `WavFrontend.upsacle_samples=True` (wav_frontend.py) | `[T_samples]` f32 | — | `ggml_scale(x, 32768.0)` before frame extraction. Required because kaldi fbank operates in int16 magnitude range. | new |
| Frame to 25 ms / 10 ms | `kaldi.fbank` | `[T_frames, 400]` | — | Sliding-window framer. `snip_edges=True` drops trailing partial frames. | new (parakeet/whisper centre and right-pad differently) |
| Hamming window (kaldi) | `kaldi.fbank(window_type="hamming")` | `[T_frames, 400]` | — | Kaldi hamming: `0.54 - 0.46·cos(2πn/(N-1))` (symmetric, N-1 denominator). Whisper Hann is periodic, so this is genuinely different. | new |
| FFT (pad to next pow2) | `kaldi.fbank` (round_to_power_of_two=True default) | `[T_frames, 257]` complex magnitude (n_fft=512) | — | `ggml_stft`-style real FFT padded to 512. Power spectrum (mag²). | new (whisper uses n_fft=400 directly) |
| Kaldi HTK mel filterbank | `kaldi.fbank(num_mel_bins=80)` | `[T_frames, 80]` | — | Triangular bins, HTK mel scale `2595·log10(1 + f/700)`, no slaney area normalization. Bin centers + slopes baked into a `[80, 257]` matrix. | new (whisper uses slaney mels; parakeet uses NeMo mels) |
| Log magnitude | `kaldi.fbank` | `[T_frames, 80]` | — | `log(max(x, energy_floor=0.0))` per bin. | new (whisper clamps differently) |
| LFR stack (m=7, n=6) | `apply_lfr` (wav_frontend.py) | `[T_lfr, 560]` | — | For each output frame i: concatenate input frames `[6i, 6i+6]` (m=7 frames). Last block reuses the final frame to fill if needed. | new |
| Per-feature CMVN | `apply_cmvn` (wav_frontend.py) | `[T_lfr, 560]` | `frontend.fbank.lfr.cmvn.out` | `(x + shift) * scale` with the 560-d shift/scale baked under `frontend.cmvn.{shift,scale}`. | new (parakeet's per-feature normalization stores `mean`/`std` not `shift`/`scale`) |

`stt.frontend.type="kaldi_fbank_lfr"`, `fbank_style="kaldi_htk"`,
`num_mels=80`, `n_fft=400` (the window length; the actual FFT size is
`next_pow2(400)=512` — kaldi semantics, the C++ frontend pads
internally), `hop_length=160`, `lfr_m=7`, `lfr_n=6`,
`upscale_samples=true`, `snip_edges=true`, `dither=0.0`.

## Encoder

Encoder is `SenseVoiceEncoderSmall`. Two depth tiers (`encoders0` +
`encoders` +
`tp_encoders`), all SAN-M blocks. The first block (`encoders0[0]`) is a
560 → 512 projection block; everything after is 512-d throughout. The
encoder's `forward` contains `xs_pad *= sqrt(d_model)` immediately
before the positional encoding — that is the d_model=512 scale, NOT the
input-dim 560 scale, and is applied to a tensor that is still 560-d at
that point.

### Pre-encoder prefix prepend

Implemented in `SenseVoiceSmall.inference`, NOT in the encoder. Mirrors
the dumper's reference flow.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| `language_query = embed[lid_dict[lang]]` | model.py inference() | `[1, 1, 560]` | `enc.prefix.lid_emb` | `ggml_get_rows(enc.embed.weight, [lid_idx])` | parakeet predictor token embed lookup pattern |
| `event_emo_query = embed[[1, 2]]` | model.py inference() | `[1, 2, 560]` | `enc.prefix.event_emo_emb` | `ggml_get_rows(enc.embed.weight, [1, 2])` | shared |
| `textnorm_query = embed[textnorm_dict[itn]]` | model.py inference() | `[1, 1, 560]` | `enc.prefix.textnorm_emb` | `ggml_get_rows(enc.embed.weight, [textnorm_idx])` | shared |
| `speech = cat(textnorm, fbank_lfr_cmvn) ; speech = cat(lid, event_emo, speech)` | model.py inference() | `[1, 4 + T_lfr, 560]` | `enc.input.with_prefix` | `ggml_concat` along time axis (axis=1) | new — no other family prepends learnable embeddings to the encoder input |

The `embed` table is a single `Embedding(16, 560)` keyed by 4 small
in-model dicts. **Index ranges (verified at intake/Stage 2):**
`lid_dict` = `{auto:0, zh:3, en:4, yue:7, ja:11, ko:15, nospeech:13}`,
`event_emo` always uses `[1, 2]` literals, `textnorm_dict` =
`{withitn:14, woitn:15}`. Stage 4 must read these dicts from the
upstream model.py at port time and bake them into the loader; they are
NOT part of the SentencePiece vocabulary (the CTC OUTPUT vocabulary
has separate language/event/emotion/itn token IDs at 24884+).

### Encoder body

```
x_pcm
↓ frontend → x [B, T_lfr, 560]
↓ prefix prepend → x [B, 4 + T_lfr, 560]
↓ x *= sqrt(512)               # d_model scale
↓ x += sinusoidal_pe(T, depth=560)   # depth = current width = 560
↓ encoders0[0]                 # SAN-M block: 560 → 512 projection
↓ encoders[0..48]              # 49 SAN-M blocks at 512
↓ after_norm                   # LayerNorm(512), eps=1e-12
↓ tp_encoders[0..19]           # 20 SAN-M blocks at 512
↓ tp_norm                      # LayerNorm(512), eps=1e-12
↓ ctc.head: Linear(512, 25055)
↓ log_softmax
```

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Scale by `sqrt(output_size())` | `SenseVoiceEncoderSmall.forward` | `[B, T, 560]` | implicit | `ggml_scale(x, sqrt(512.0))` | new (whisper / parakeet do not have this scale) |
| Sinusoidal positional encoding | `SinusoidalPositionEncoder.forward` (sense_voice/model.py) | `[B, T, 560]` | `enc.embed.out` | Build PE table once: `inv_timescales = exp(arange(d/2) * (-log(10000)/(d/2 - 1)))`, `scaled = positions[1..T] * inv_timescales`, `pe = concat(sin(scaled), cos(scaled))` along feature axis. **depth equals current width** (560 here, NOT 512). Position indices are 1-based. | new — whisper learns absolute pos_emb; parakeet uses relative-pos / RoPE |
| `encoders0[0]` | `EncoderLayerSANM.forward` with in_size=560, size=512 | `[B, T, 512]` | `enc.encoders0.0.out` | See "SAN-M block pattern (projection)" below | new |
| `encoders[0..48]` ×49 | `EncoderLayerSANM.forward` with in_size=size=512 | `[B, T, 512]` | `enc.encoders.{0,24,48}.out` | See "SAN-M block pattern (residual)" below | new |
| `after_norm` | `LayerNorm(512, eps=1e-12)` | `[B, T, 512]` | `enc.after_norm.out` | `ggml_norm(x, 1e-12)` then `* w + b` | shared |
| `tp_encoders[0..19]` ×20 | `EncoderLayerSANM.forward` (same as main tier) | `[B, T, 512]` | `enc.tp_encoders.{0,10,19}.out` | Same as main-tier blocks | shared with main tier |
| `tp_norm` | `LayerNorm(512, eps=1e-12)` | `[B, T, 512]` | `enc.tp_norm.out` | `ggml_norm(x, 1e-12)` then `* w + b` | shared |

### SAN-M block pattern (residual, encoders + tp_encoders)

Pre-norm. All linears have biases. ReLU activation in the FFN.

```
x_in
↓ ln1 = norm_attn(x_in)         # LayerNorm(d_model, eps=1e-12)
↓ q_k_v = ln1 · W_qkv + b_qkv   # Linear(d_model, 3·d_model)
↓ split q_k_v → q, k, v          # along last dim, each [B, T, d_model]
↓ q_h, k_h, v_h = reshape+transpose to [B, n_heads, T, d_k]
↓ FSMN branch (parallel to attention, runs on raw v BEFORE head split):
   - apply mask: v *= mask[:, :, None]    (zero pad positions)
   - transpose: [B, T, d_model] → [B, d_model, T]
   - constant-pad: left=(K-1)//2, right=K-1-left      (K=11 → 5/5)
   - depthwise Conv1d(d_model, d_model, K=11, stride=1, groups=d_model, bias=False)
   - transpose back: [B, d_model, T]
   - residual within FSMN: x_fsmn = x_fsmn + masked_v
   - apply mask again
↓ Attention:
   - q_h *= 1/sqrt(d_k)            # scale on Q only, BEFORE the matmul
   - scores = q_h @ k_h.T          # [B, n_heads, T, T]
   - mask: scores.masked_fill(mask_eq_0, -inf)
   - attn = softmax(scores).masked_fill(mask_eq_0, 0.0)
   - ctx = attn @ v_h              # [B, n_heads, T, d_k]
   - merge heads: [B, T, d_model]
   - att_out = ctx · W_out + b_out   # Linear(d_model, d_model)
↓ attn_branch = att_out + x_fsmn   # IMPORTANT: SAN-M attention output is the SUM of SDPA and FSMN
↓ x = x_in + attn_branch           # residual (in_size==size branch)
↓ ln2 = norm_ffn(x)               # LayerNorm(d_model, eps=1e-12)
↓ y = w_1(ln2) + b_1; y = ReLU(y); y = w_2(y) + b_2     # 512 → 2048 → 512
↓ x = x + y                        # residual
```

**SAN-M attention is the SUM of SDPA output and the FSMN memory branch,
not a residual sum.** The FSMN runs in parallel to the attention on the
unreshaped V tensor (post-mask), and its output is added to the
attention-output projection, not concatenated. This is the defining
feature of "Self-Attention Network with Memory."

### SAN-M block pattern (projection, encoders0[0] only)

Same internal structure but `in_size=560 != size=512`. The block has no
attention residual — the input-side residual is dropped because the
projection changes width. The FFN residual still adds the 512-d
post-attention tensor:

```
x_in [560]
↓ ln1 = norm_attn(x_in)         # LayerNorm(560, eps=1e-12) — see norm1 shape
↓ q_k_v = ln1 · W_qkv [1536, 560] + b_qkv [1536]
... (same QKV split + SDPA + FSMN as above; everything internal is now 512-d)
↓ attn_branch = att_out + x_fsmn   # [B, T, 512]
↓ x = attn_branch                  # NO residual add (in_size != size)
↓ ln2 = norm_ffn(x)               # LayerNorm(512, eps=1e-12)
↓ FFN as above (512 → 2048 → 512)
↓ x = x + y                        # residual on FFN only
```

The first-block `norm1` weight/bias are shape (560,) reflecting the
input-space LayerNorm; everything after `attn_branch` is 512-d.

## Decoder

There is no decoder. SenseVoiceSmall is non-autoregressive: the encoder
output goes straight into a CTC head and a greedy-CTC decode collapses
the time axis into output tokens.

## Generation / KV Path

There is no KV cache. CTC decoding has no autoregressive step-loop —
the entire output is read off the CTC log-probs in one pass.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| CTC linear | `CTC.ctc_lo` (Linear(512, 25055)) | `[B, T, 25055]` | `ctc.logits.raw` | `ggml_mul_mat(ctc.head.weight, x) + ctc.head.bias` | new (parakeet head is RNN-T joint, not CTC) |
| `log_softmax` | `CTC.log_softmax` | `[B, T, 25055]` | `ctc.log_probs` | The C++ runtime applies log_softmax in the post-processing path; the tensor-level gate compares pre-softmax logits, the behavior gate compares decoded text. | shared |
| Greedy CTC decode | `inference()` | seq of token IDs | transcript | `argmax` along feature axis → `unique_consecutive` (collapse repeats) → drop blanks (id=0) → SP detokenize. Tokens 4..end are speech; tokens 0..3 carry the language/event/emotion/itn labels. | new |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (explicit lang) | `inference(language="en")` selects `lid_dict["en"]=4` for the prefix embed | `transcribe-cli --language <code>` chooses the prefix embed slot | `Transcribe / explicit language hint (en|zh)` |
| Transcribe (auto / no hint) | `inference(language="auto")` selects `lid_dict["auto"]=0` | `transcribe-cli` without `--language` uses `auto` slot | `Transcribe / auto / no language hint` |
| Language detection | LID emerges in the OUTPUT at vocabulary IDs 24884–24992 (e.g. `<|en|>`); the auto prefix embed gives the model the freedom to predict any language | Runtime should expose either the raw token stream or a parsed `language` field. If the runtime strips control tokens by default, language detection is observable only with a raw-token flag. | `Language detection / LID emitted in output` |
| Translation | not advertised | n/a | SKIP — not advertised |
| Timestamps | non-AR CTC has no segment-timestamp head; upstream provides word timestamps via `ctc_forced_align` post-pass on the CTC log-probs | Out of scope for Stage 4 unless the runtime decides to expose forced alignment | SKIP — not exposed by runtime (initial port) |
| Streaming | not advertised | n/a | SKIP — not advertised |
| VAD | external `fsmn-vad` model, separate FunASR checkpoint | n/a | SKIP — not exposed by runtime |
| Speaker diarization | not advertised | n/a | SKIP — not advertised |
| Speech emotion (SER) | emerges in OUTPUT at 25001–25004 (`<|HAPPY|>` etc.) | Same observability constraint as language detection | `Speech emotion recognition (SER)` |
| Audio event (AED) | emerges in OUTPUT at 24993–25019 (`<|BGM|>`, `<|Speech|>`, …) | Same | `Audio event detection (AED)` |
| Inverse text normalization (ITN) | `use_itn=True` flips the textnorm prefix from `woitn` (15) to `withitn` (14); ITN labels emerge at 25016/25017 | Runtime needs an `--itn`/`use_itn` flag that maps to the textnorm prefix slot | `Inverse text normalization (ITN)` |

## Deviations From Closest Analog

- **Two-tier encoder depth.** No other ported family stacks `encoders` and `tp_encoders` with an inter-tier LayerNorm (`after_norm`) between them. The runtime walks both arrays sequentially.
- **Projection block at depth 0.** `encoders0[0]` widens the input-space residual stream from 560 to 512 by dropping the residual on the attention sub-layer. The skill's "no half-finished implementations" rule applies — the block must not silently pretend to add a 560 + 512 residual.
- **SAN-M attention = SDPA + FSMN.** The block adds the depthwise-FSMN branch to the SDPA output before residual. The FSMN consumes the unreshaped V tensor (single-head channel layout `[B, T, d_model]`) and runs a depthwise 1D conv with kernel 11 + symmetric pad (5/5). Internal residual `x_fsmn += masked_v` lives inside the FSMN, not the block.
- **Fused QKV.** `linear_q_k_v` is `nn.Linear(in_feat, 3·n_feat)` — converters store it as a single tensor `attn.qkv.weight`. The C++ port can either keep it fused (split via `ggml_view`) or split at load time. Fused is fewer ops but the split-views require careful stride math.
- **Mask is `[B, 1, T]`.** Encoder masks are sequence-length masks, NOT causal. They zero out the SOFTMAX scores AND the attention output (twice — once on scores via `masked_fill(-inf)`, again on attn via `masked_fill(0.0)` for the all-pad row case). They also zero the FSMN input V to prevent the depthwise conv from leaking padding into valid frames.
- **LayerNorm epsilon = 1e-12.** Whisper uses 1e-5. Forgetting this is the typical first-block drift bug in conformer ports.
- **Sinusoidal PE has 1-based positions.** `positions = arange(1, T+1)`, not `arange(0, T)`. Forgetting this rotates every PE table by one slot.
- **Sinusoidal PE depth follows current width, not d_model.** PE is built with `depth=560` for the first encoder pass. If the implementation treats PE as a precomputed `[max_T, d_model]` table sized by d_model=512, the encoders0 input vs PE shape mismatch will surface as a shape error, not a numerical drift.
- **Pre-encoder embedding scale uses `sqrt(d_model)=sqrt(512)` even on a 560-d tensor.** This is a Transformer-style embedding scale that ignores the projection-block widening; the constant is `sqrt(d_model_after)`, not `sqrt(d_input)`.
- **Frontend has no in-tree analog.** Kaldi HTK fbank, hamming window, no-slaney mels, LFR stack, per-feature CMVN baked under `frontend.cmvn.{shift,scale}`. The C++ port needs a new mel module — reusing whisper's slaney mels or NeMo's filterbank silently miscomputes the input.
- **Output language/event/emotion live in the CTC output vocabulary, NOT in the prefix-embed table.** The 16-row `enc.embed.weight` is input-only; the 25055-row CTC head is output-only. The two index spaces are unrelated.

## Variant Notes

- **sensevoice-small** (the only published variant): family baseline above. Concrete dims: `n_blocks=50` (= 1 encoders0 + 49 encoders), `tp_blocks=20`, `d_model=512`, `d_input=560`, `n_heads=4`, `d_k=128`, `d_ff=2048`, `kernel_size=11`, `sanm_shift=0`, `attention_type=sanm`, `vocab_size=25055`, F32 reference dtype.
