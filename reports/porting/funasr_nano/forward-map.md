# Forward map - funasr_nano

Reference: `funasr.models.fun_asr_nano.model.FunASRNano` @ FunASR 1.3.1
Closest in-tree analogs:
- Encoder + frontend: `src/arch/sensevoice/` (SAN-M encoder, kaldi LFR fbank)
- LLM decoder + audio-injection concat: `src/arch/qwen3_asr/` (Qwen3 causal LM)

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Load PCM 16k mono | `funasr.utils.load_utils.extract_fbank` | [N] float32 | (none — host) | host PCM read | sensevoice |
| Kaldi HTK fbank, 80 mels, 25/10 ms hamming | `funasr.frontends.wav_frontend.WavFrontend.forward` | [T_frames, 80] | (none) | reuse `KaldiFbankFrontend` | sensevoice |
| LFR (m=7, n=6) frame stack | `WavFrontend.apply_lfr` | [T_lfr, 560] | `frontend.fbank.lfr.cmvn.out` | host: stride-stack on n_mels axis | sensevoice |
| **No CMVN** (`apply_cmvn=false`) | n/a | [T_lfr, 560] | (same as above) | drop the CMVN add+mul | new (skip vs sensevoice) |

## Encoder (SenseVoiceEncoderSmall, frozen)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Sinusoidal PE add | `SinusoidalPositionEncoder` | [560, T_lfr] | `enc.embed.out` | host-built sin/cos table + ggml_add | sensevoice (without prefix prepend) |
| `encoders0[0]` projection block (560 → 512) | `EncoderLayerSANM` | [512, T_lfr] | `enc.encoders0.0.out` | `sanm_block_projection` (no attn residual) | sensevoice |
| `encoders[0..48]` (49× residual SAN-M) | same | [512, T_lfr] | `enc.encoders.{0,24,48}.out` | `sanm_block_residual` | sensevoice |
| `after_norm` LayerNorm eps=1e-12 | `funasr.models.transformer.layer_norm.LayerNorm` | [512, T_lfr] | `enc.after_norm.out` | `ggml_norm + mul + add` | sensevoice |
| `tp_encoders[0..19]` (20× residual SAN-M) | same | [512, T_lfr] | `enc.tp_encoders.{0,10,19}.out` | `sanm_block_residual` | sensevoice |
| `tp_norm` LayerNorm eps=1e-12 | same | [512, T_lfr] | `enc.tp_norm.out` | LayerNorm | sensevoice |
| **No CTC head** (dead in checkpoint) | n/a | n/a | (skipped) | skip; encoder output is `enc.tp_norm.out` | new (skip vs sensevoice) |

Key delta vs sensevoice encoder code: drop the prefix-embedding prepend
(`enc.input.with_prefix`), the `enc.embed.weight` get_rows lookups, and the CTC
head + log-softmax. The 70-block SAN-M body is identical.

## Adaptor (Transformer, n_layer=2, downsample_rate=1, use_low_frame_rate=true)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Identity reshape (k=1) | `Transformer.forward` | [512, T_lfr] | (no dump) | pass-through (k=1 means no fold) | new |
| `linear1` (512 → 2048) + ReLU | `nn.Linear + ReLU` | [2048, T_lfr] | `adaptor.linear1.out` (post-linear, pre-ReLU) | `mul_mat + add + relu` | new |
| `linear2` (2048 → 1024) | `nn.Linear` | [1024, T_lfr] | `adaptor.linear2.out` | `mul_mat + add` | new |
| Block 0/1 (LayerNorm 1e-12 + plain MHA + ffn) | `EncoderLayer` | [1024, T_lfr] | `adaptor.blocks.0.out`, `adaptor.out` | per-block: norm1 → q/k/v/out + biases → residual; norm2 → fc1 1024→256 + relu + fc2 256→1024 → residual | qwen3_asr enc block (closest, but no q_norm/k_norm here) |
| `fake_token_len` truncation | `data_load_speech.olens` | [1024, fake_token_len] | (host slice into LLM prompt) | `use_low_frame_rate` formula: `((((T-1)/2+1)-1)/2+1)/2+1`. T=183 → 23 frames | new |

Adaptor block deviations from qwen3_asr encoder block:
- LayerNorm (with bias) eps=1e-12, NOT RMSNorm.
- Plain Q/K/V projections with biases (no per-head Q/K-RMSNorm, no RoPE).
- FFN is bottleneck-shaped (`llm_dim // 4` = 256 hidden), ReLU activation, with biases.

## LLM decoder (Qwen3-0.6B, frozen, BF16)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Token embed | `embed_tokens` | [1024, T_prompt] | `dec.token_emb` | `ggml_get_rows(token_w, ids)` | qwen3_asr |
| Audio-row override at `[fbank_beg, fbank_beg + fake_token_len)` | `inputs_embeds[batch, fbank_beg:fbank_beg+L] = adaptor_out[:L]` | [1024, T_prompt] | `dec.inputs_embeds.with_audio` | concat(prefix \| adaptor_out[:fake_token_len] \| suffix) | qwen3_asr (same concat trick) |
| 28× Qwen3 block | `Qwen3DecoderLayer` | [1024, T_prompt] | `dec.block.{0,27}.out` | rms_norm + GQA 16/8 + per-head Q/K-RMSNorm + NeoX RoPE @ 1e6 + SwiGLU (packed gate_up) | qwen3_asr |
| Final RMSNorm | `model.norm` | [1024, T_prompt] | `dec.out_before_head` | `rms_norm * w` | qwen3_asr |
| Tied lm_head (slice last position) | `lm_head` | [vocab=151936] | `dec.logits_raw.prefill` | `mul_mat(token_w, last_x)` | qwen3_asr |

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Prefill, all positions | HF `generate` call 0 | [vocab] | `dec.logits_raw.prefill` | KV cache write [0, T_prompt) | qwen3_asr |
| Single-step decode @ n_past≥T_prompt | HF `generate` calls 1..N | [vocab] | `dec.logits_raw.gen8` | static-shape step graph (build once, reuse) | qwen3_asr |
| Stop on EOS (`151645 = <\|im_end\|>`) | HF `generate` | (token id) | (host) | host loop | qwen3_asr |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (explicit en) | `auto.generate(language="en", itn=False)` → prompt = `语音转写成en，不进行文本规整：` | `transcribe_run_params{language="en"}` builds the same prompt | row "Transcribe / explicit en" |
| Transcribe (explicit zh / ja) | same template, swap language code | same | rows "Transcribe / zh" and "Transcribe / ja" |
| Transcribe (auto / no hint) | `auto.generate(language=None)` → prompt = `语音转写：` | `language=nullptr` falls through to no-language template | row "Transcribe / auto" |
| ITN | `itn=True` removes the `，不进行文本规整` suffix from the prompt | flag through `params->use_itn` (mirrors `sensevoice` shape) | row "ITN" |
| Translation, timestamps, streaming, VAD, diarization | not exposed | SKIP rows in family doc | already SKIP at intake |

## Deviations From Closest Analog

- **Encoder + frontend duplicate sensevoice's SAN-M code.** Per `CLAUDE.md` "No refactoring during a port — note awkwardness, ship correctness first." The funasr_nano variant uses `KaldiFbankFrontend` without CMVN and the encoder graph without the prefix-embedding prepend / CTC head. Both are pruned forks of the sensevoice files. Follow-up: refactor SAN-M block + KaldiFbankFrontend into a shared module reused by `src/arch/sensevoice/` and `src/arch/funasr_nano/`. Tracked in family-doc Notes.
- **Adaptor uses `chunk_num = (seq_len - 1) // k + 1; pad_num = chunk_num*k - seq_len`** (FunASR-specific; identical to sensevoice's LFR but at adaptor input). With `downsample_rate=1` the formula is a no-op (chunk_num == seq_len, pad == 0); we keep the math symbolic for variant-portability.
- **`fake_token_len` truncation (`use_low_frame_rate=true`).** Adaptor produces 183 frames but only the first `((((T-1)/2+1)-1)/2+1)/2+1` of them (23 here) are spliced into the LLM prompt. C++ runtime computes `fake_token_len` host-side from the LFR frame count and slices `adaptor_out` accordingly.
- **Audio injection differs in surface from qwen3_asr.** qwen3_asr uses dedicated `<|audio_pad|>` / `<|audio_start|>` / `<|audio_end|>` special tokens; funasr_nano uses raw chat-template strings (`<|im_start|>`, `<|im_end|>`, `<|startofspeech|>`, `<|endofspeech|>`) where `<|startofspeech|>...<|endofspeech|>` is split out by the Python loader and replaced with `fake_token_len` zero-id tokens. The token-id 0 placeholder is overwritten with audio embeddings. In our graph the splice is identical to qwen3_asr's three-way concat (prefix \| audio \| suffix); only the token-id assembly differs.
- **No CTC head.** The checkpoint omits `ctc_decoder.blocks.{2,3,4}.*` and `ctc.ctc_lo.*`, so the path produces gibberish even when wired. Skip entirely; transcripts come from the LLM.

## Variant Notes

- `fun-asr-nano-2512`: same architecture as the family baseline. 50 SAN-M encoder blocks, 20 tp blocks, 28 Qwen3 LM layers, hidden=1024.
- `fun-asr-mlt-nano-2512` (sibling, not yet ported): same architecture, expanded language list (31 languages); will be a same-arch sibling-variant Stage 1 → 4 cycle.
