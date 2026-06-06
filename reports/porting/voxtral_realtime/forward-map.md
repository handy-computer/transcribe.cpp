# Forward map - voxtral_realtime

Reference: HuggingFace transformers `VoxtralRealtimeForConditionalGeneration`
(`transformers/models/voxtral_realtime/modeling_voxtral_realtime.py`, refs
checkout 5.x); streaming feature extractor + tekken tokenizer via
`mistral-common`. Closest in-tree analog: `src/arch/voxtral/` (sibling 2507
audio-LLM) for the load/run/projector/qwen3_lm-decoder scaffold, and
`src/arch/qwen3_asr/` for the audio-token injection pattern.

Voxtral Realtime (2602) is a **streaming audio-LLM**, architecturally DISTINCT
from the 2507 `voxtral` family (own arch dir). Shares only the projector shape,
the tekken tokenizer, and the family brand. Pipeline:
**streaming log-mel (fixed max) → causal-RoPE sliding-window(750) conv+transformer
encoder → 4× frame-group projector → Ministral causal LM with ADDITIVE audio
fusion and a per-layer delay-conditioned (ada-norm) FFN scale**.

Stage-4 scope (user-signed 2026-06-05): **true incremental streaming**. The
OFFLINE whole-clip forward is built first as the numerical baseline (the
`decode` oracle is the per-tensor contract); the `stream` oracle proves
offline↔streaming numerical equivalence; then the incremental scheduler
(conv padding cache + dual KV caches + windowed masks + 12.5 Hz chunk
scheduling + delay tokens + `--stream-chunk-ms` CLI / stream hooks) lands.

Reuse strategy: the **decoder block** is the shared `src/qwen3_lm/` module
(GQA, NEOX RoPE, SwiGLU packed gate+up, KV cache, batched paths) with per-head
Q/K-norm null (as voxtral 2507 does), PLUS a new optional per-layer FFN-branch
scale (`view.ffn_scale`) for the ada-norm. The **encoder** is a new graph
builder reusing `conf::conv_1d_f32` and the qwen3_lm RoPE/attention primitives
where possible (causal+sliding-window mask, NEOX RoPE, SwiGLU encoder MLP).

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Log-mel | `VoxtralRealtimeFeatureExtractor` (128 mel, n_fft 400, hop 160, win 400 periodic Hann, slaney fb 0–8000 Hz, drop last STFT frame, `log10(clamp(·,1e-10))`, **FIXED `global_log_mel_max=1.5`**, `max(log, 1.5−8)`, `(log+4)/4`) | `[128, T_mel]` (jfk 1496) | `enc.mel.in` | `transcribe::MelFrontend` (`normalize=global`, fixed max), filterbank+window baked into GGUF (`frontend.mel_filterbank [201,128]`, `frontend.window [400]`) | whisper / voxtral-2507 frontend (per-utterance max → **global fixed max here**) |
| Chunking | `center=True` for first/offline chunk, `center=False` for continuation chunks; conv padding cache carries left-pad across chunks | streamed | — | host chunk loop; offline path = single centered chunk | none (streaming-specific) |

## Encoder (VoxtralRealtimeEncoder — causal RoPE, sliding-window 750)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| conv1 | Conv1d 128→1280 k3 s1, **LEFT-pad causal (pad=2)** + bias, GELU | `[1280, T_mel]` | — | `conf::conv_1d_f32` (F16 kernel) with left-only pad (asymmetric pad host-side) + bias + `ggml_gelu_erf` | none (causal conv new) |
| conv2 | Conv1d 1280→1280 k3 s2, **LEFT-pad causal (pad=1)** + bias, GELU | `[1280, T_enc]` (jfk 748) | `enc.embedder.out` | same; stride 2 halves time | none |
| block ×32 | pre-norm RMSNorm: `norm_attn→attn→res`, `norm_ffn→SwiGLU→res`. Attn: 32 heads hd 64 (q-dim 2048), **full MHA (kv_heads=32)**, q/v/o bias, **k NO bias**, q×hd^-0.5, **NEOX RoPE θ=1e6**, **causal + sliding-window(750) mask**. FFN: SwiGLU/silu, gate/up no bias, **down WITH bias** | `[1280, T_enc]` | `enc.block.{0,16,31}.out` | new builder; RMSNorm (`ggml_rms_norm×w`), NEOX `ggml_rope_ext`, windowed-causal mask, packed gate+up + `ggml_silu` | qwen3_lm primitives (forked, not block_*) |
| final norm | RMSNorm `enc.final_norm.weight` | `[1280, T_enc]` | `enc.out` | `ggml_rms_norm × enc.final_norm.weight` | — |

## Projector (VoxtralRealtimeMultiModalProjector)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| frame group | reshape `[1280,T_enc]`→`[5120, T_enc/4]` (token i = concat enc frames 4i..4i+3, **truncate T_enc%4**) | `[5120, T_enc/4]` (jfk 187) | — | C-order reshape/view; same as voxtral-2507 projector | voxtral-2507 |
| linear_1 | Linear 5120→3072, no bias | `[3072, T_enc/4]` | — | `ggml_mul_mat(proj.linear_1.weight, x)` | voxtral-2507 |
| act | GELU (`projector_hidden_act=gelu`) | — | — | `ggml_gelu_erf` | — |
| linear_2 | Linear 3072→3072, no bias | `[3072, T_enc/4]` | `proj.out` | `ggml_mul_mat(proj.linear_2.weight, x)` | voxtral-2507 |

## Decoder (Ministral text backbone — 26 layers, ADDITIVE fusion, ada-norm)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| token embed | `embed_tokens(input_ids)` | `[3072, T_tok]` (jfk audio span 187) | `dec.token_emb` | `ggml_get_rows(dec.token_embd.weight, ids)` | voxtral-2507 |
| audio inject | **ADDITIVE**: `inputs_embeds += projector_out` at aligned audio positions (NOT masked_scatter) | `[3072, T_tok]` | `dec.audio_injected` | `ggml_add(token_emb_view, proj_out)` over the audio span | none (additive new) |
| time cond | `t_cond = [cos(d·inv_freq), sin(d·inv_freq)]` dim 3072, `d=num_delay_tokens` (default 6 = 480 ms); per-layer `ada(t)=linear2(gelu(linear1(t)))`; scale `s_l = 1 + ada(t)` is **constant per run** | per-layer `[3072]` | — | precompute `s_l` once at session setup from `dec.time_embed.inv_freq` + `dec.blocks.l.ada.linear_{1,2}` (tiny graph, host or device) | none (novel) |
| block ×26 | `residual+attn(input_layernorm(h))`; then `residual + mlp(post_attention_layernorm(h) · s_l)`. GQA 32q/8kv hd128, **NEOX RoPE θ=1e6**, sliding 8192, SwiGLU silu (interm 9216), no attn/ffn bias | `[3072, T_tok]` | `dec.block.{0,13,25}.out` | `qwen3_lm::block_*` with `attn_q/k_norm=null` + new `view.ffn_scale=s_l` | qwen3_lm shared (+ffn_scale hook) |
| final norm | `norm` RMSNorm eps 1e-5 | `[3072, T_tok]` | `dec.out_before_head` | `ggml_rms_norm × dec.output_norm.weight` | voxtral-2507 |
| lm_head | **TIED** (`lm_head.weight == embed_tokens.weight`) | `[131072]` | `dec.logits_raw` | `ggml_mul_mat(dec.token_embd.weight, last_x)` — tied | qwen3_asr (tied); voxtral-2507 is untied |

## Generation / Streaming Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| prompt | `[BOS]` + `[STREAMING_PAD=32]·(n_left_pad + num_delay_tokens)`; audio embeds added at aligned positions; output length clamped to `ceil(mel_frames / audio_length_per_tok=8)` | — | — | host token builder; additive audio fusion; greedy clamp | none (streaming) |
| prefill | offline whole-clip teacher-forced LM forward, `scores[0]` | `[131072]` | `dec.logits_raw` | `build_prefill_graph` (causal/sliding mask) | voxtral-2507 prefill |
| step | greedy `argmax`, KV-cached decode | per-step `[131072]` | `dec.logits_raw.gen8` (n_past=T_prompt+7) | `qwen3_lm::block_step` + argmax; dump gen8 | voxtral-2507 step |
| stream | incremental: conv padding cache + encoder StaticCache(750) + decoder sliding-KV(8192) re-run per chunk; downsample_factor=4 enc frames fed / decode step; 12.5 Hz | per-step `[131072]` | `stream.logits_raw`, `stream.logits_raw.gen8` | streaming scheduler + 5 stream hooks; gated equal to offline | none (novel) |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (lang hint) | streaming transcription template w/ language | `--language <l>` | Transcribe / explicit language (MUST PASS) |
| Transcribe (auto) | language auto-detected | no `--language` | Transcribe / auto (MUST PASS) |
| Streaming | incremental encode/decode interleave, configurable delay | `--stream-chunk-ms <N>` + stream hooks, incremental emit | Streaming (MUST PASS — forced; user-signed true streaming 2026-06-05) |
| Configurable delay | `num_delay_tokens` 1..(80 ms–2.4 s) via tekken | delay param (default 6 = 480 ms) | Configurable transcription delay (MUST PASS) |
| Language detection | auto-detect on transcription | not separately surfaced | `lang_detect` advertised in GGUF |
| Translate | not advertised (`translation=false`) | n/a | OUT OF SCOPE |
| Timestamps | none (no timestamp tokens) | n/a | OUT OF SCOPE |

## Deviations From Closest Analog (voxtral 2507)

- **Causal RoPE sliding-window encoder**, NOT Whisper bidirectional. Causal
  left-pad conv stem; RMSNorm pre-norm (not LayerNorm-with-bias); NEOX RoPE
  θ=1e6 hd 64; causal + sliding-window(750) mask; SwiGLU/silu MLP (down has
  bias). No stored sinusoidal `embed_positions`. Forked encoder builder.
- **ADDITIVE audio fusion** (`inputs_embeds += proj_out`), NOT the 3-way
  concat / masked_scatter of 2507. Audio embeds added onto the STREAMING_PAD
  placeholder positions.
- **Delay-conditioned ada-norm FFN scale**: per-layer `· (1 + ada(t_cond))`
  between `post_attention_layernorm` and the MLP. `ada` is `linear→gelu→linear`
  (NO RMSNorm inside despite the name); `t_cond` depends only on
  `num_delay_tokens`, so `s_l` is a per-layer constant precomputed once per run.
  Plan: add optional `view.ffn_scale` (broadcast `[hidden,1]`) to the shared
  qwen3_lm block; null for all existing callers (backward-compatible).
- **Tied lm_head** (`tie_word_embeddings=true`): reuse `dec.token_embd.weight`
  for logits. (voxtral-2507 is untied.)
- **True streaming**: conv padding cache + encoder StaticCache(750) + decoder
  sliding-KV(8192) re-run incrementally; two sliding-window mask widths
  (enc 750, dec 8192). Breaks the encode-once/decode-many structure. Built
  after the offline path is numerically green.

## Variant Notes

- `voxtral-mini-4b-realtime-2602`: the only realtime variant. Encoder 32 layers
  d1280 h32 hd64 sw750; decoder 26 layers d3072 GQA 32/8 hd128 sw8192 interm
  9216; projector 5120→3072→3072; delay default 6 (480 ms). For jfk
  (T_enc=748 < sw 750) the offline whole-clip path is exact ⇒ offline==stream.
