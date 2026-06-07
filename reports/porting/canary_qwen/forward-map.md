# Forward map - canary_qwen

Reference: nvidia/canary-qwen-2.5b @ b1469e1bba1cfe140205529c79c434ca47180960
Reference framework: NeMo `>=2.5.0` `nemo.collections.speechlm2.models.SALM`
Closest in-tree analogs:
- Encoder + perception: src/arch/canary/ (FastConformer, byte-for-byte canary-1b-flash)
- Decoder + audio injection: src/arch/qwen3_asr/ (audio-LLM scatter pattern, Qwen3 LM blocks)
- Shared helpers: src/conformer/, src/causal_lm/

SALM is a perception-encoder + LM composition. The encoder half is identical
to canary-1b-flash (including the `enc.proj.weight` 1024->2048 projection,
which here is the AudioPerceptionModule's projection rather than canary's
optional encoder->decoder projection — the GGUF tensor names overlap on
purpose). The LM half is Qwen3-1.7B with LoRA merged at convert time.
Audio injection is a 3-way concat of [prefix_emb | enc_out | suffix_emb]
into the LM input embedding sequence.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| log-mel | nemo AudioToMelSpectrogramPreprocessor (n_mels=128, n_fft=512, win=400, hop=160, hann_symmetric, slaney filterbank, per_feature normalize, preemph=0.97, dither overridden 1e-5 -> 0.0 at inference) | [128, T_mel] | `enc.mel.in` (host-side dump) | `transcribe::MelFrontend::compute()`, baked filterbank + window from GGUF | identical to canary; window stays `hann_symmetric` despite GGUF KV reading `"hann"` (NeMo passes `periodic=False`) |

## Encoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| pre_encode (factor 8) | perception.encoder.pre_encode (DwStridingSubsampling: 1 std conv2d + 2 dw-pw pairs, ReLU, linear out) | [d_model=1024, T_enc=floor(T_mel/8)] | `enc.pre_encode.out` | `transcribe::conformer::build_pre_encode()` over PreEncodeView | canary/encoder.cpp:134-140 |
| pos_emb | perception.encoder.pos_enc (sinusoid, length 2T-1) | [1024, 2T_enc-1] | `enc.pos_emb` | graph input, host-built sinusoid; not loaded from GGUF | canary/encoder.cpp:147-151 |
| conformer block (×32) | perception.encoder.layers.{i} (macaron FF1 0.5x → rel-pos MHSA → conv module BN+SiLU → macaron FF2 0.5x → norm_out) | [1024, T_enc] | `enc.block.{0,16,31}.out` | `transcribe::conformer::build_conformer_block()` over BlockView, per-block bias on every linear (canary pattern, NOT parakeet) | canary/encoder.cpp:162-192 |
| final | identity (no extra projection on encoder side) | [1024, T_enc] | `enc.final` | named alias of last block output | canary/encoder.cpp:213-215 |
| perception proj | perception.proj (Linear 1024->2048 + bias) | [2048, T_enc] | `perception.proj.out` | `ggml_mul_mat(enc_proj.weight, x) + enc_proj.bias` — reuses canary's `enc.proj.*` GGUF tensor names | canary/encoder.cpp:205-211 (180m-flash projection slot) |

## Decoder

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| token embed | embed_tokens (top-level, tied to llm.lm_head; vocab 151936, hidden 2048) | [2048, T_prompt] | `dec.token_emb` | `ggml_get_rows(dec.token_embd.weight, input_ids)` | qwen3_asr/decoder.cpp:115-123 |
| audio inject | SALM scatter at audio_locator positions (single placeholder expanded to T_enc) | [2048, T_prompt - 1 + T_enc] | `dec.audio_injected` | 3-way `ggml_concat`: [prefix_emb \| perception.proj.out \| suffix_emb] over T_enc audio_locator slots | qwen3_asr/decoder.cpp:155-184 |
| qwen3 block (×28) | llm.model.layers.{i} (pre-LN RMSNorm, GQA 16Q/8KV head_dim=128 with per-head Q/K RMSNorm, NeoX RoPE θ=1e6, packed gate_up SwiGLU, intermediate=6144) | [2048, T_total] | `dec.block.{0,14,27}.out` | `transcribe::causal_lm::block_prefill()` over BlockView | qwen3_asr/decoder.cpp:185-220 |
| final RMSNorm | llm.model.norm | [2048, T_total] | `dec.out_before_head` | `ggml_rms_norm(x, eps) * dec.output_norm.weight` | qwen3_asr/decoder.cpp:225-235 |
| lm_head (tied) | llm.lm_head (tied to embed_tokens) | [151936] | `dec.logits_raw.gen0` | slice last position, `ggml_mul_mat(dec.token_embd.weight, x_last)` | qwen3_asr/decoder.cpp:240-254 |

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| KV cache layout | self-attn-only, [n_kv_heads · head_dim · n_ctx · n_layer] flat (8 · 128 · ≥1024 · 28) | per-layer K/V slices | n/a | `transcribe::causal_lm::kv_init()` + `kv_cache.{n,head}` | qwen3_asr/model.cpp:284-310 (kv_init) |
| prefill pass | LM forward over the audio-injected embedding sequence; argmax of last-position logits | [vocab] | `dec.logits_raw.gen0` (mid-prefill output) | one prefill graph build/alloc/compute, write KV at positions [0, T_total) | qwen3_asr/decoder.cpp prefill graph |
| step graph (autoregressive) | greedy decode loop, max_new_tokens=128, stop on EOS | [vocab] each step | `dec.logits_raw.gen8` ADDED for mid-gen coverage (skill requirement; reference dumper updated to capture step 8) | static-shape step graph reused per token, KV write via `set_rows` at index = current position | qwen3_asr/decoder.cpp step graph |
| stop / decode | EOS (151645) or max_new_tokens; `model.tokenizer.ids_to_text(ids)` | text | n/a (transcript.json) | greedy argmax loop reading int32 [1] back per step, decode via `Tokenizer::decode` | qwen3_asr/model.cpp:955-1125 |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe English (single-language model) | SALM.generate with `[{"role":"user","content":"Transcribe the following: <\|audioplaceholder\|>","audio":[wav_path]}]` | `transcribe_run()` with audio; no language hint | row: "Transcribe English (no hint)" → PASS expected |
| Transcribe with PnC (punctuation+capitalisation) | reference produces PnC text by default | C++ inherits whatever Qwen3 generates; no toggle | row: "Transcribe English (PnC by default)" → PASS expected |
| Translation | not advertised by canary-qwen-2.5b | n/a | row: "Translate" → SKIP — not advertised |
| Word/segment timestamps | not advertised | n/a | row: "Timestamps" → SKIP — not advertised |
| Streaming | not advertised | n/a | row: "Streaming" → SKIP — not advertised |
| Auto language detection | not advertised (English-only) | n/a | row: "Language detection" → SKIP — not advertised |
| VAD / diarization | not advertised | n/a | row: "VAD / diarization" → SKIP — not advertised |
| Forced language hint | not exposed by SALM.generate path | n/a | row: "Language hint" → SKIP — not exposed |

## Deviations From Closest Analog

- **Encoder weights load via `enc.proj.{weight,bias}` GGUF slot, but semantically it is the AudioPerceptionModule's 1024→2048 projection, not canary's 180m-flash encoder→decoder projection.** Numerically and structurally the same op (Linear out_features=dec_hidden=2048, in_features=enc_d_model=1024), but the loader must always require it (no `dec_has_encoder_decoder_proj` toggle — perception projection is mandatory for the audio-LLM coupling). Use a dedicated CanaryQwenWeights struct rather than reusing CanaryWeights to keep the semantics explicit.
- **No transducer / encoder-decoder cross-attention decoder.** canary's `CanaryDecBlock` (norm1+self / norm2+cross / norm3+ffn) is replaced wholesale by `causal_lm::block_prefill` / `block_step` over `causal_lm::BlockView`. Decoder hparams come from `stt.canary_qwen.decoder.*` (Qwen3 shape: GQA 16Q/8KV head_dim=128, intermediate=6144, rope_theta=1e6, RMSNorm eps=1e-6, tied embeddings).
- **Audio injection style.** qwen3_asr writes `audio_token_id × T_enc` directly into the prompt token list and concat-replaces those rows with encoder output. canary_qwen's reference puts a SINGLE `<|audioplaceholder|>` (id 151669) in the prompt and the SALM forward expands it into T_enc audio embeddings via scatter. Numerically equivalent — we adopt the qwen3_asr pattern (T_enc audio_locators in prompt + 3-way concat) because it keeps the prefill graph identical to qwen3_asr; the only knob is `audio_locator_id` from `stt.canary_qwen.perception.audio_locator_id`.
- **Standard 1D RoPE, NOT MRoPE.** qwen3_asr's RoPE handles MRoPE collapse for text-only; canary_qwen has no MRoPE at all. Pass `rope_mrope_section_*=0` and let `causal_lm::block_*` route through the NeoX RoPE path with single-stream positions. (causal_lm helper already handles this; the family loader just doesn't read MRoPE KVs.)
- **Mel window is `hann_symmetric`, not `hann_periodic`** despite the dumper sidecar declaring `hann_periodic` — NeMo's `AudioToMelSpectrogramPreprocessor` calls `torch.hann_window(win_length, periodic=False)`. Same call as canary; `MelConfig::window_type = "hann_symmetric"` regardless of the GGUF KV value (which is just the hann family, not the symmetric/periodic flag).
- **Conv-block linears all carry biases** (canary pattern, NOT parakeet), and all conv pointwise/depthwise carry biases too. Confirmed by the converter's ENCODER_BLOCK_TABLE (39 tensors per block). Use `BlockView` with every `_b` slot populated.
- **Mid-generation coverage gap.** Stage-2 dump_coverage only ships `dec.logits_raw.gen0`; the skill requires `n_past > 0` coverage. We add `dec.logits_raw.gen8` to BOTH the reference dumper and the C++ runner during this stage and add a tolerance entry; coverage gap to be filed back to porting-2-oracle.

## Variant Notes

- `canary-qwen-2.5b`: only variant. 32 encoder layers, 28 LM layers, vocab 151936, hidden_size=2048, head_dim=128, n_q_heads=16, n_kv_heads=8, intermediate=6144, max_position=40960, rope_theta=1e6.
