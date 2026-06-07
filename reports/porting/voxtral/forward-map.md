# Forward map - voxtral

Reference: HuggingFace transformers `VoxtralForConditionalGeneration`
(`transformers/models/voxtral/modeling_voxtral.py`) @ v4.57.6; tokenizer +
transcription template via `mistral-common` (`encode_transcription`).
Closest in-tree analog: `src/arch/qwen3_asr/` (audio-LLM: audio encoder +
causal LM with audio-token injection).

Voxtral 2507 is an **audio-LLM** (token injection, no cross-attention):
Whisper-large-v3 bidirectional encoder → 4× frame-group projector → Llama /
Ministral causal LM, with the projector output `masked_scatter`'d into the LM
input embeddings at `audio_token_id=24` placeholder positions.

Reuse strategy (see "Deviations" below): the **decoder block** is the shared
`src/causal_lm/` module (GQA, NEOX RoPE w/ configurable θ, SwiGLU packed gate+up,
KV cache, batched paths) with per-head Q/K-norm made optional. The **encoder**
is a thin graph builder over the already-shared `conf::conv_1d_f32` /
`conformer::layer_norm` / `ggml_gelu_erf` primitives (same math as
`src/arch/whisper/encoder.cpp`, which is file-local and not extracted per
project policy #8).

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Log-mel | `WhisperFeatureExtractor` (128 mel, n_fft 400, hop 160, win 400 periodic Hann, slaney fb 0–8000 Hz, drop last STFT frame, per-utterance log10: `max(log,log.max()-8)`, `(log+4)/4`) | `[128, 3000]` per 30 s chunk | `enc.mel.in` | `transcribe::MelFrontend` (`normalize=per_utterance`), filterbank+window baked into GGUF (`frontend.mel_filterbank`, `frontend.window`) | whisper (same frontend, 80 vs 128 mel) |
| Chunking | `VoxtralProcessor` pads PCM to mult of 480000, splits into N × 30 s, stacks on batch | N × `[128, 3000]` | — | host-side chunk loop; each chunk → 1500 enc frames → 375 audio tokens | whisper long-form |

## Encoder (VoxtralEncoder = Whisper-large-v3 encoder)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| conv1 | `modeling_voxtral.py:350` Conv1d 128→1280 k3 s1 p1 + GELU | `[1280, 3000]` | — | `conf::conv_1d_f32` (F16 kernels in GGUF) + bias add + `ggml_gelu_erf` | whisper conv stem |
| conv2 | `:351` Conv1d 1280→1280 k3 s2 p1 + GELU | `[1280, 1500]` | — | same; halves time | whisper conv stem |
| +pos_emb | `:352-355` permute, add fixed sinusoidal `embed_positions` (F32) | `[1280, 1500]` | — | load `enc.pos_emb.weight` (F32), `ggml_add` | whisper pos-emb add |
| block ×32 | `VoxtralEncoderLayer` pre-LN: LN→attn→res, LN→fc1→GELU→fc2→res. Attn: q/v/out have bias, **k_proj no bias**; q pre-scaled by `head_dim**-0.5`, `scaling=1.0`; bidirectional full | `[1280, 1500]` | `enc.block.{0,16,31}.out` | `conformer::layer_norm` (w+b); MHA d=1280 h=20 hd=64; flash/soft_max; FFN fc1 5120 GELU fc2 | whisper enc block |
| final LN | `:366` `layer_norm` | `[1280, 1500]` | `enc.out` | `conformer::layer_norm` (`enc.ln_post.{weight,bias}`) | whisper final LN |

## Projector (VoxtralMultiModalProjector)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| frame group | `get_audio_features:452` `reshape(-1, 5120)`: token i = concat(enc frames 4i..4i+3) | `[5120, 375]` | — | C-order reshape of `[1280,1500]` → `[5120,375]` (ggml `reshape`/view; verify row-major contiguity) | none (new) |
| linear_1 | `:390` Linear 5120→3072, no bias | `[3072, 375]` | — | `ggml_mul_mat(proj.linear_1.weight, x)` | none (new) |
| act | `:391` GELU | `[3072, 375]` | — | `ggml_gelu_erf` (projector_hidden_act=gelu) | — |
| linear_2 | `:392` Linear 3072→3072, no bias | `[3072, 375]` | `proj.out` | `ggml_mul_mat(proj.linear_2.weight, x)` | none (new) |

## Decoder (LlamaForCausalLM / Ministral text backbone)

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| token embed | `forward:512` `embed_tokens(input_ids)` | `[3072, 383]` | `dec.token_emb` | `ggml_get_rows(dec.token_embd.weight, ids)` | qwen3_asr |
| audio inject | `:518-521` `masked_scatter` audio embeds at `input_ids==24` | `[3072, 383]` | `dec.audio_injected` | placeholder run is contiguous → 3-way concat (prefix ⊕ proj.out ⊕ suffix) like qwen3_asr; batched: `ggml_set_rows` | qwen3_asr audio injection |
| block ×30 | Llama block: RMSNorm→GQA(32q/8kv, hd128)→res, RMSNorm→SwiGLU(silu, 8192)→res; **NEOX RoPE θ=1e8**; **no Q/K-norm**; no attn bias | `[3072, 383]` | `dec.block.{0,15,29}.out` | `causal_lm::block_prefill/step` with `attn_q_norm=attn_k_norm=nullptr` (Q/K-norm made optional) | causal_lm shared |
| final norm | `LlamaModel.norm` RMSNorm eps 1e-5 | `[3072, 383]` | `dec.out_before_head` | `ggml_rms_norm` × `dec.output_norm.weight` | qwen3_asr |
| lm_head | **UNTIED** `lm_head.weight` (separate `dec.output.weight`) | `[131072]` | `dec.logits_raw` | `ggml_mul_mat(dec.output.weight, last_x)` — NOT tied to token_embd | (qwen3_asr ties; voxtral does not) |

## Generation / KV Path

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| prompt (transcribe) | mistral-common `_encode_instruct_transcription`: `[BOS][INST][BEGIN_AUDIO]`+`[AUDIO]×375`+`[/INST]`(+`lang:<l>` if hint)+`[TRANSCRIBE]` | `[383]` (en) / `[378]` (auto) | — | host token builder; audio placeholders → enc positions | qwen3_asr build_prompt_tokens |
| prompt (instruct) | `apply_chat_template`: `[BOS][INST][BEGIN_AUDIO]`+`[AUDIO]×375`+`BPE(instruction)`+`[/INST]` (no `[TRANSCRIBE]`) | `[378+k]` | — | host token builder; tekken BPE for instruction text; same audio injection | (new — translate/prompt path) |
| prefill | `language_model(inputs_embeds=...)` greedy, `scores[0]` | `[131072]` | `dec.logits_raw` (`scores[0]`) | `build_prefill_graph` (causal mask, positions 0..382) | qwen3_asr prefill |
| step | greedy `argmax`, KV-cached decode until EOS=2 | per-step `[131072]` | `dec.logits_raw.gen<N>` (N≥8) | `causal_lm::block_step` + on-device argmax; dump a gen>0 step | qwen3_asr step |

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (lang hint) | `apply_transcription_request(language=en)` → `lang:en` + `[TRANSCRIBE]` | `--language en` → append `lang:<l>` tokens before `[TRANSCRIBE]` | Transcribe / explicit language hint (MUST PASS) |
| Transcribe (auto) | `language=None` → omit `lang:` tokens (mistral-common allows; HF wrapper does not) | no `--language` → omit `lang:` tokens | Transcribe / auto (MUST PASS) |
| Translate (speech→text) | **chat instruction**, not a token: audio + text via `apply_chat_template`; there is NO `[TRANSLATE]` token | `--translate`(+`--target-language`) synthesizes instruction `"Translate this to {Language}."` (English default) → instruction template | Translate (MUST PASS 2507) — resolved: instruction path |
| Prompt / audio understanding | `apply_chat_template` user msg = audio + free-text instruction | `--prompt "<text>"` (wired to `TRANSCRIBE_FEATURE_INITIAL_PROMPT`) → instruction template; greedy decode | exposed-but-ungated (English-only acceptance covers transcribe+translate) |
| Language detection | default transcription auto-detects; no explicit detect token | detected language not separately surfaced by transcription template | `lang_detect` advertised in GGUF |

## Deviations From Closest Analog

- **Untied lm_head.** qwen3_asr ties lm_head to token_embd; Voxtral ships a
  separate `dec.output.weight`. Use it for the logits mul_mat; do not tie.
- **No per-head Q/K RMSNorm.** Qwen3 blocks apply `attn_q_norm`/`attn_k_norm`
  unconditionally (`causal_lm.cpp:229-230,396-397,547-548`). Voxtral's Llama has
  none. Plan: make those `ggml_rms_norm×norm` steps conditional on a non-null
  `view.attn_q_norm` (backward-compatible; existing callers pass non-null). Then
  the shared block serves Voxtral unchanged otherwise.
- **Encoder is Whisper, not the qwen3_asr conformer-style encoder.** Different
  conv stem (2× Conv1d vs 3× Conv2d), LayerNorm-with-bias (not RMSNorm),
  attention WITH biases (k_proj excepted), GELU FFN, learned-stored sinusoidal
  pos-emb. Built thin over shared `conf::conv_1d_f32` + `conformer::layer_norm`;
  same math as `src/arch/whisper/encoder.cpp` (file-local; not reused to avoid a
  mid-port refactor — noted as future "extract shared whisper encoder").
- **Projector frame-grouping reshape** (`[1280,1500]`→`[5120,375]`, token i =
  concat of 4 consecutive encoder frames) has no in-tree analog; must preserve
  C-order so frame 4i..4i+3 land contiguous in the 5120 axis.
- **Translate is a chat instruction, not a task token** (RESOLVED, user-signed
  2026-06-04). Voxtral has no `[TRANSLATE]` token; translation, Q&A and
  summarization all go through `apply_chat_template` (audio + free-text
  instruction). Implemented as ONE instruction-template path:
  `[BOS][INST][BEGIN_AUDIO]`+audio+`BPE(instruction)`+`[/INST]`, distinct from
  the transcription-request template. Exposed via `--translate`/`--target-language`
  (synthesizes `"Translate this to {Language}."`, English default) and a general
  `--prompt "<text>"` wired to `TRANSCRIBE_FEATURE_INITIAL_PROMPT`. Transcribe +
  translate are the gated MUST-PASS rows; general prompting is exposed-but-ungated.

## Variant Notes

- `voxtral-mini-3b-2507`: family baseline (this port). Ministral-3B decoder
  (30 layers, hidden 3072, GQA 32/8, hd 128), 32-layer Whisper-large-v3 encoder.
- `voxtral-small-24b-2507`: same architecture, larger decoder
  (`hidden_size`, `intermediate_size`, `num_hidden_layers` vary — listed in
  intake `varying_across_variants`). Sibling-variant shortcut should apply.
- `voxtral-mini-4b-realtime-2602`: architecturally distinct streaming variant
  (`voxtral_realtime`), DEFERRED; not covered here.
- **Translate MUST-PASS scope** — RESOLVED (user-signed 2026-06-04): implement
  the chat-instruction path this port and expose a general `--prompt`. See
  Deviations.
