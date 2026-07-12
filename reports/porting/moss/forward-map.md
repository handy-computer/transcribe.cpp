# Forward map - moss

Reference: OpenMOSS custom `trust_remote_code` files (`modeling_moss_transcribe_diarize.py`,
`processing_moss_transcribe_diarize.py`) in `OpenMOSS-Team/MOSS-Transcribe-Diarize`
@ `d7231bbae2587a4af278735eb765b318c4f64edd`.
Closest in-tree analogs: `src/arch/whisper/` (encoder) + `src/arch/qwen3_asr/`
(Qwen3 decoder, audio-token injection, batch/step drivers, `src/causal_lm/`).

audio-llm: log-mel -> WhisperEncoder (24L) -> 4x time merge (1024->4096) ->
VQAdaptor (4096->1024) -> masked_scatter into Qwen3-0.6B embeds at `<|audio_pad|>`
(151671) positions -> Qwen3 decoder (28L) -> tied lm_head. Diarization/timestamps
are emergent plain text (`[start][Sxx]text[end]`), not tokens.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Chunk + pad audio to 30s | `processing._chunk_audio` (n_samples=480000; last chunk zero-padded to 30s) | list of [480000] chunks + per-chunk `token_len=(n-1)//1280+1` | (host) | host loop; one mel+encoder pass per chunk | qwen3_asr host chunking |
| Log-mel (WhisperFeatureExtractor) | `feature_extractor` (80 mel, n_fft 400, hop 160, hann_periodic, center/reflect, slaney, per-utterance log norm) | [80, 3000] per chunk | `enc.mel.in` | `transcribe::MelFrontend` w/ baked `frontend.mel_filterbank` + `frontend.window` | whisper / qwen3_asr MelFrontend |

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Conv1 (k3 s1 p1) + GELU-erf | `WhisperEncoder.conv1` | [1024, 3000] | — | `conf::conv_1d_f32` (F16 kernel, F32 in), `ggml_gelu_erf` | whisper `encoder.cpp` |
| Conv2 (k3 s2 p1) + GELU-erf | `WhisperEncoder.conv2` | [1024, 1500] | — | same, stride 2 | whisper |
| + learned pos emb | `embed_positions.weight` | [1024, 1500] | `enc.pos_add.out` | `ggml_add` (F32 `enc.pos_emb.weight`); no scale_embedding | whisper |
| 24x pre-LN block (LayerNorm, MHA gelu FFN; q/v/out have bias, k none) | `WhisperEncoderLayer` | [1024, 1500] | `enc.block.0.out`, `enc.block.23.out` | `conf::layer_norm` + `mha_encoder` (bidir, no mask) + gelu-erf FFN | whisper `build_block` |
| Final LayerNorm | `WhisperEncoder.layer_norm` | [1024, 1500] | `enc.ln_post.out` | `conf::layer_norm` | whisper |
| Per-chunk trim to `token_len*4` cols, concat chunks | `get_audio_features` (`whisper_features[:, :token_len*4]`) | [1024, sum(token_len)*4] | — | host trims cols before merge; single-chunk jfk -> [1024, 552] | new (moss) |
| 4x time merge | `time_merge`: `[:, :T_trim].reshape(B, T//4, D*4)` | [4096, T_enc] | `enc.merge.out` | `ggml_cont` then `ggml_reshape_2d(1024,T)->(4096,T/4)` (4 consec frames -> 4096) | new (moss) |
| VQAdaptor: fc1(4096->1024)+bias, SiLU, fc2(1024->1024)+bias, LayerNorm(eps 1e-6, bias) | `VQAdaptor.layers` | [1024, T_enc] | `enc.adaptor.out` | `mul_mat`+bias, `ggml_silu`, `mul_mat`+bias, `conf::layer_norm` (`adaptor.norm_out`) | new (moss) |

## Decoder

Qwen3-0.6B: 28 layers, GQA 16/8, head_dim 128, per-head q/k RMSNorm, NeoX RoPE
theta 1e6, SwiGLU (packed gate+up), RMSNorm eps ~1e-6, tied head. Identical block
math to `qwen3_asr` via `src/causal_lm/` (`BlockView`/`block_prefill`/`block_step`).

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Prompt build (chat wrapper + audio span + time markers) | `processor.__call__` / `_audio_span_ids` | [T_prompt] i32 (jfk: 227) | — | prefix ids + audio-span (host int math) + suffix ids; baked `stt.moss.prompt_*_tokens` / `digit_tokens` KV | qwen3_asr `build_prompt_tokens` |
| Token embed (all positions, pre-injection) | `Qwen3Model.embed_tokens` | [1024, T_prompt] | `dec.token_emb` | `ggml_get_rows(dec.token_embd, input_ids)` | qwen3_asr |
| Audio injection (masked_scatter at audio_pad) | `inject_audio_features` | [1024, T_prompt] | `dec.audio_injected` | elementwise blend `x*keep_mask + audio_dense` (audio positions non-contiguous due to markers) | qwen3_asr batched prefill blend |
| 28x Qwen3 block (prefill, causal) | `Qwen3Model.layers` | [1024, T_prompt] | `dec.block.0.out`, `dec.block.27.out` | `causal_lm::block_prefill` | qwen3_asr `decoder.cpp` |
| Final RMSNorm | `Qwen3Model.norm` | [1024, T_prompt] | `dec.out_before_head` | `ggml_rms_norm * dec.output_norm` | qwen3_asr |
| Tied lm_head (last position) | `lm_head` (tied to embed) | [151936] | `dec.logits_raw` | `ggml_mul_mat(dec.token_embd, last_x)` | qwen3_asr |

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Prefill writes KV [0,T_prompt) | `Qwen3Model` w/ cache | — | — | `block_prefill` into `causal_lm::KvCache` | qwen3_asr |
| Greedy step (argmax, one token) | `generate(do_sample=False)` | [1] i32/step | `dec.logits_raw.gen<N>` (N>=8, to add) | static `build_step_graph` reused per step | qwen3_asr `build_step_graph` |
| Stop on EOS 151645 / max_new | GenerationMixin | — | — | host loop, de-diarize before scoring | qwen3_asr run loop |
| Batched offline (B utts) | — | — | — | `kv_init_batched` + `block_prefill_batched` + `run_batched_step_loop` | qwen3_asr run_batch |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe en/zh (auto) | fixed Chinese diarize prompt for all audio | raw text -> de-diarize | Transcribe auto (en/zh) MUST PASS |
| Segment timestamps | emergent `[start][end]` text | present in raw transcript | Segment timestamps MUST PASS |
| Long-form >30s | 30s chunk + concat before merge | host chunk loop | Long-form MUST PASS |
| Diarization | emergent `[Sxx]` text | in raw transcript; no distinct API | OUT OF SCOPE (no API surface) |
| Explicit language / prompt override | prompt-text only, no token | not exposed (single baked prompt) | OUT OF SCOPE |
| Batch (offline) | — | `run_batch` parallel path | Batch MUST PASS |

## Deviations From Closest Analog

- Encoder is whisper (conv stem + learned PE + gelu LayerNorm blocks), NOT
  qwen3_asr's conv2d-subsample encoder. Reuse whisper's `mha_encoder` shape.
- Audio injection is a **blend** (`x*keep_mask + audio_dense`), not qwen3_asr's
  single-shot 3-way concat: MOSS audio-pad positions are non-contiguous because
  the processor interleaves time-marker digit tokens into the span. The blend is
  exactly qwen3_asr's *batched* prefill injection, reused for single prefill too.
- Prompt: fixed Chinese diarize instruction with a `system` block and
  `<|audio_start|>/<|audio_end|>` wrapper. Prefix/suffix token ids + digit ids are
  baked into the GGUF by the converter (from the reference tokenizer); C++ only
  reproduces the deterministic `_audio_span_ids` integer math. No C++ Chinese BPE,
  no Jinja rendering (language/prompt override is OUT OF SCOPE for this port).
- Merge is a pure `ggml_reshape` (row-major group-of-4), no learned op.

## Variant Notes

- `moss-transcribe-diarize`: family baseline (0.9B; Whisper-Medium enc + Qwen3-0.6B dec).
