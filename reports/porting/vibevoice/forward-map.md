# Forward map - vibevoice

Reference: `models/_vendor/VibeVoice/vibevoice/` (author `vibevoice` package) @ HF rev
`d0c9efdb8d614685062c04425d91e01b6f37d944`. Canonical modules:
`modular/modeling_vibevoice_asr.py`, `modular/modular_vibevoice_tokenizer.py`,
`processor/vibevoice_asr_processor.py`.
Closest in-tree analog: `src/arch/qwen3_asr/` (audio-llm: audio frontend + Qwen causal LM
with audio-token injection). The VAE conv frontend has **no** in-tree analog.

Audio-LLM. Raw 24 kHz waveform -> two parallel causal-conv VAE encoders (acoustic vae_dim 64,
semantic vae_dim 128) -> SpeechConnector (-> 3584) each -> **element-wise SUM** -> scattered
into the Qwen2.5-7B LM embedding stream at `speech_pad` positions -> 28 Qwen2 layers -> lm_head
-> greedy decode of structured JSON. **Contract uses the acoustic VAE MEAN, never a sample.**

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Resample to 24 kHz mono | processor `__call__` (librosa load sr=24000) | [samples] | — | host-side resample (input PCM) | qwen3_asr frontend entry |
| AudioNormalizer | `processor/vibevoice_tokenizer_processor.py` AudioNormalizer (target_dB_FS=-25, eps 1e-6): `rms=sqrt(mean(x^2)); x*=10^(-25/20)/(rms+eps)`; then declip `if max(|x|)>1: x/=max(|x|)+eps` | [samples] | `enc.input_waveform` [264000] | host-side scalar pass over PCM (F32); **pin exact 0.0 drift** (env-inject for isolation) | none (new) |

## Encoder (acoustic + semantic VAE; same structure, different vae_dim / pad)

Reference: `modular_vibevoice_tokenizer.py` `TokenizerEncoder.forward` + `encode()`. Both encoders
share structure; acoustic vae_dim=64 (std_dist gaussian, fix_std 0.5 — irrelevant, we use mean),
semantic vae_dim=128 (std_dist none). `disable_last_norm=True` -> final norm is Identity.
encoder_ratios=[8,5,5,4,2,2], depths=[3,3,3,3,3,3,8], n_filters=32, layer_scale 1e-6.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Stem conv (downsample_layers.0) | SConv1d 1->32, k=7, s=1, causal | [32, S] | — | `ggml_conv_1d` w/ causal left-pad `(k-1)-(s-1)=6`, + stride-align right pad; bias add | none (new) |
| Block stack stage[i] (depth 3 or 8) | `Block1D.forward` | [C, S] | — | per-block: see Block1D pattern below | none (new) |
| Downsample conv (layers.1..6) | SConv1d C->2C, k=2·ratio, s=ratio, causal | [2C, S/ratio] | — | `ggml_conv_1d` stride=ratio, causal left-pad `(k-1)-(s-1)`, declip via `get_extra_padding` (ceil align) | none (new) |
| Final norm | `nn.Identity` (disable_last_norm) | [2048, T] | — | no-op | — |
| Head conv | SConv1d 2048->vae_dim, k=7, causal | [vae_dim, T] | — | `ggml_conv_1d` causal | none (new) |
| Permute -> mean | `encode()` returns `latents.permute(0,2,1)` | [T, vae_dim] | `enc.acoustic.mean` [83,64], `enc.semantic.mean` [83,128] | `ggml_cont(ggml_transpose())` | none (new) |

**Block1D pattern** (`Block1D.forward`, map once): operates on [C, S].
1. `r=x; x=ConvRMSNorm(x); x=DepthwiseConv(x); x*=gamma[C]; x=r+x`
   - ConvRMSNorm: RMS over channel axis per time step (transpose [C,S]->[S,C], RMSNorm dim=C, weight only, eps; transpose back). **eps: confirm 1e-5 vs 1e-6 in code before finalizing.**
   - DepthwiseConv (`mixer`, groups=C, k=7, causal, bias): `ggml_conv_1d_dw` (or grouped conv), causal left-pad 6.
   - `gamma` is per-channel layer-scale [C], broadcast over S.
2. `r=x; x=ConvRMSNorm_ffn(x); x=permute[C,S]->[S,C]; x=Linear2(GELU(Linear1(x))); permute back; x*=ffn_gamma[C]; x=r+x`
   - FFN linear1/linear2 with bias; activation GELU (`ACT2FN["gelu"]`).

T for jfk = ceil(264000/3200) = 83, matching `vae_tok_len` (count of speech_pad tokens).

## Decoder (Qwen2.5-7B LM)

Reference: `modeling_vibevoice_asr.py` forward + the Qwen2 `language_model`. 28 layers, hidden 3584,
heads 28 / kv 4, head_dim 128, intermediate 18944, rope_theta 1e6, rms_norm_eps 1e-6, vocab 152064.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Connector (×2) | `SpeechConnector`: fc1(vae_dim->3584)->RMSNorm(3584)->fc2(3584->3584) | [T, 3584] | `enc.acoustic.feat`, `enc.semantic.feat` [83,3584] | matmul+bias, RMSNorm, matmul+bias. **confirm no activation between norm and fc2** | qwen3_asr proj |
| Fusion | `combined = a_feat + s_feat` | [T, 3584] | `enc.combined` [83,3584] | `ggml_add` | none |
| Token embed | `embed_tokens(input_ids)` | [L, 3584] | `dec.token_emb` [143,3584]; pin exact 0.0 | `ggml_get_rows`; pure read | qwen3_asr embed |
| Speech scatter | `inputs_embeds[acoustic_input_mask] = combined` (replace speech_pad embeds) | [L, 3584] | `dec.inputs_embeds` [143,3584] | `ggml_set_rows` / masked copy at speech_pad positions | qwen3_asr audio inject |
| Qwen2 layer ×28 | standard Qwen2 block (pattern below) | [L, 3584] | `dec.block.{0,13,27}.out` | attn+rope+ffn (see pattern) | qwen3_asr decoder (adapt) |
| Final RMSNorm | `language_model.norm` | [L, 3584] | `dec.final_norm` [143,3584] | `ggml_rms_norm`×weight | qwen3_asr |
| lm_head | `lm_head` (untied, `dec.output.weight`) | [L, vocab] | `dec.logits_lastpos` [152064] (last pos) | matmul | qwen3_asr |

**Qwen2 layer pattern** (differs from qwen3_asr — call out in Deviations):
`r=x; x=RMSNorm_attn(x); q=Wq·x+bq; k=Wk·x+bk; v=Wv·x+bv` (GQA 28/4, head_dim 128, **q/k/v have
bias, no q/k-norm**); `rope(q,k, rotate_half, theta=1e6)`; causal SDPA; `x=r+Wo·attn` (Wo no bias);
`r=x; x=RMSNorm_ffn(x); x=r+Wdown(silu(Wgate·x)*Wup·x)`.

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Prompt build | processor `__call__` lines ~340-380 | token ids | — | host: `<\|im_start\|>system\n{SYSTEM_PROMPT}<\|im_end\|>\n<\|im_start\|>user\n` + speech_start + speech_pad×T + speech_end + `\nThis is a {dur:.2f} seconds audio, please transcribe it with these keys: Start time, End time, Speaker ID, Content<\|im_end\|>\n<\|im_start\|>assistant\n` | qwen3_asr prompt builder |
| Prefill | model forward, `use_cache` | [L, vocab] | `dec.logits_lastpos` | full causal prefill, KV fill | qwen3_asr prefill |
| Incremental decode | `generate` greedy; speech inputs only when cache_position[0]==0 | 1 tok/step | **`dec.logits_raw.gen<N>` (N≥8) — MUST ADD to dumper + C++** | KV-cache step, argmax, append | qwen3_asr decode |
| Stop | `eos_token_id` (151643) or `stop_strings=["}]"]`, max_new_tokens=448 | — | — | host: eos + JSON-end string match | qwen3_asr stop |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (auto lang) | greedy JSON | non-empty plausible transcript | Transcribe — MUST PASS |
| Language detect | implicit (model self-detects) | advertised via `stt.capability.lang_detect` | (covered by transcribe) |
| Segment timestamps | structured Start/End in JSON | parse JSON -> segment times | Segment timestamps — MUST PASS |
| Speaker diarization | Speaker ID in JSON | parse JSON -> speaker labels | Speaker diarization — MUST PASS |
| Translate / word-ts / streaming / hotwords | n/a | n/a | OUT OF SCOPE (per family doc) |
| Batch (offline) | n/a | `run_batch()` parity | Batch (offline) — MUST PASS |

## Deviations From Closest Analog (qwen3_asr)

- **Frontend is a learned causal-conv VAE, not a mel/STFT spectrogram.** Entirely new ggml
  subsystem: causal SConv1d (left-pad `(k-1)d-(s-1)` + ceil stride-align right pad), depthwise
  conv mixer, ConvRMSNorm (norm over channel axis), per-channel layer-scale gamma. No in-tree code.
- **Two parallel encoders + connectors fused by element-wise SUM**, then scattered into the LM at
  `speech_pad` positions. qwen3_asr injects a single audio stream.
- **LM is plain Qwen2.5, not Qwen3:** q/k/v carry biases; NO per-head q/k RMSNorm; **plain
  rotate_half RoPE (theta 1e6), NOT MRoPE/interleaved** (qwen3_asr uses mrope_section). Untied
  lm_head (`dec.output.weight`).
- **Determinism:** use the acoustic VAE mean (`mode`), never `sample` (fix_std=0.5 gaussian).
- **Structured output:** WER scoring needs a normalizer that strips Speaker/Start/End and keeps
  Content (Stage 7); segment-ts + diarization are in scope (parse the JSON).

## Variant Notes

- `vibevoice-asr`: family baseline (only variant). ~8.3 B params kept, BF16 reference. Long-form
  (60-min / 64K KV) deferred; first port targets short utterances (jfk / LibriSpeech segments).

## Resolved tensor/shape contract (from GGUF inspection)

- **Downsample strides come from REVERSED encoder_ratios.** Weight shapes (ggml ne
  `[k,in,out]`): `downsample.0`=[7,1,32] stem (s1); `downsample.1`=[4,32,64] s2; … ;
  `downsample.6`=[16,1024,2048] s8. **kernel = 2·stride for every strided conv**, so the port
  derives `stride = kernel/2` from the loaded weight (ordering-agnostic). Stage channels
  C[i]=32·2^i for i=0..6 (32,64,128,256,512,1024,2048); stage i runs AFTER downsample i.
- **VAE FFN = 4× expansion + GELU.** `ffn.linear1` C->4C, `ffn.linear2` 4C->C (e.g. 32->128,
  2048->8192). Both carry bias. `ffn_dim = 4*C`.
- Mixer depthwise weight `[7,1,C]` (groups=C, k=7, s1, causal); head `[7,2048,vae_dim]` (k7,s1).
- Connector: `fc1[vae_dim->3584]` -> `RMSNorm[3584]` -> `fc2[3584->3584]`; fc1/fc2 carry bias.

## Open items (resolve during Step 2, before sign-off)

- ConvRMSNorm eps (1e-5 config vs 1e-6 Block1D kwarg default) — read the constructed value.
- SpeechConnector: confirm exact op order and whether any activation sits between RMSNorm and fc2.
- `get_extra_padding_for_conv1d` exact arithmetic (right-pad for ceil stride alignment) — must
  match so encoder T == ceil(n_samples/3200).
- ggml causal conv: `conf::conv_1d_f32` takes symmetric padding only; left-pad manually
  (`ggml_pad`/view) by `(k-1)d-(s-1)` + ceil right-pad, then conv with padding=0.
- Compute location: the 9B LM will not run on this 16 GB machine; `validate.py all` block/logits
  gates and the Step 11 WER gate run on **Modal** (encoder-only gates may run locally). ← the
  local→Modal boundary; notify the user here.
