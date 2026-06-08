# Forward map - granite

Reference: HuggingFace `transformers.models.granite_speech` (mainline,
v5.8.1+) — `modeling_granite_speech.py` is the source of truth.
Closest in-tree analog: `src/arch/qwen3_asr/` for the audio-LLM
end-to-end shape (encoder + projector + scatter into LM embed + KV-cache
greedy decode); `src/conformer/` for the macaron FFN and the GLU conv
module with BatchNorm. The Shaw block-local attention has no analog in
tree and is implemented inline in `src/arch/granite/encoder.cpp`.

This stage targets 3 AR variants: `granite-4.0-1b-speech`,
`granite-speech-4.1-2b`, `granite-speech-4.1-2b-plus`. The NAR variant
lives in family `granite_nar` and is out of scope.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| torchaudio MelSpectrogram (n_fft=512, win=400 zero-padded to 512, hop=160, hann_periodic, **htk mel scale, no mel norm, power=2.0, no log, no per-utterance norm**) + **2-frame stack** so the encoder input is 160 dims per stacked frame | `GraniteSpeechFeatureExtractor.melspec` then `_get_mel_filter_bank` then 2-frame reshape | `[T_mel/2, 160]` | `enc.mel.in` | `transcribe::MelFrontend` configured for htk + no-norm + no-log + power=2; followed by a host 2-frame stack producing the 160-dim input. The 2-frame stack is the same pattern as parakeet pre-encode-into-channels, but here it is a pure reshape (no conv). | extend `MelFrontend` to support `mel_norm="htk"` and a new `log=false, power=2` mode; no closer in-tree analog |

The frontend differs from Whisper's log-mel and from parakeet's
per-feature CMVN. The Stage 2 oracle's `enc.mel.in` is the bit-identical
target.

## Encoder

Single Conformer geometry across all 3 variants: 16 layers, hidden_dim=1024,
8 heads, dim_head=128, context_size=200, max_pos_emb=512, conv_kernel=15,
conv_expansion=2, ffn_mult=4, input_dim=160, output_dim=348 (CTC head).

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Pre-encode (linear lift 160 → 1024) | `GraniteSpeechCTCEncoder.input_linear` | `[T, 1024]` | `enc.input_linear.out` | `linear(input_linear.w, x) + input_linear.b` | none — granite has no conv subsampling |
| Conformer block (FF1 → block-local Shaw self-attn → conv module (BN) → FF2 → post LN) | `GraniteSpeechConformerBlock` | `[T, 1024]` | `enc.block.{0, n/2, n-1}.out` | hand-built `granite::build_encoder_block`. FF1/FF2 use `transcribe::conformer::macaron_ff_residual`. Conv module uses `transcribe::conformer::conv_module` with `conv_norm_type=BatchNorm`, `conv_kernel=15`, asymmetric padding `(7, 7)`. Attention is a new `granite::shaw_block_attn` helper (see Deviations). | `transcribe::conformer::*` for FFN + conv; `qwen3_asr` for block iteration shape |
| Self-conditioned CTC bypass at layer N/2 (=8) | inline in `GraniteSpeechCTCEncoder.forward` | `[T, 1024]` | (residual is captured indirectly via subsequent block outs) | After block 8 emits `h`, host or graph computes `h_mid = softmax(linear(ctc_proj, h))`, then `h := h + linear(ctc_bypass, h_mid)` before block 9. `ctc_proj` is the 1024→348 linear; `ctc_bypass` is 348→1024. The softmax is over the channel dim. | none — novel pattern in tree |

Block dump indices: layers in {0, n/2, n-1} = {0, 8, 15}. Stage 2 dumps
`enc.block.0.out`, `enc.block.8.out` (where the bypass tap is), and
`enc.block.15.out` plus `enc.out`.

**Variant note**: `granite-speech-4.1-2b-plus` sets
`encoder_config.cat_hidden_layers=[3]`. The encoder concatenates
`block[3].out` with the final `block[15].out` along the channel axis,
producing a `[T, 2048]` tensor for the projector's cross-attention K/V.
For 1b/2b this list is empty and the projector reads `[T, 1024]`. Loader
reads `stt.granite.encoder.cat_hidden_layers` (uint32 array, possibly
empty) and either holds onto the `block[3]` output for later concat or
skips the bookkeeping entirely.

## Projector (Q-Former)

BLIP-2 Q-Former, 2 layers. Same dimensions across all 3 variants except
that the cross-attention K/V input dim is `encoder_hidden_size` (1024
for 1b/2b, 2048 for plus).

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Window-slicer: chunk encoder output into `nblocks = T / window_size` blocks of `window_size=15` frames each | `GraniteSpeechEncoderProjector.forward` | `[nblocks, 15, 1024 or 2048]` | (implicit) | host-side reshape — encoder output `[T, C]` → `[nblocks, 15, C]`. Trailing partial window is dropped by the upstream processor (audio length is padded to a multiple of `window_size * downsample_rate * mel_per_audio_token`). | none |
| Query broadcast: 3 learned queries replicated across nblocks | `query` ([1, 3, 1024]) | `[nblocks, 3, 1024]` | (implicit) | `ggml_repeat`-style broadcast of the query buffer | none |
| 2-layer Q-Former (self-attn → cross-attn → FFN; per-layer LN-after-residual) | `Blip2QFormerLayer` | `[nblocks, 3, 1024]` | `proj.qformer.out` | hand-built `granite::build_qformer_layer`. Each layer: query → self-attn → LN → cross-attn(query, kv=encoder window) → LN → FFN → LN. K/V projected from the per-window encoder slice. | none — Q-Former is new in tree |
| Final Q-Former LN | `Blip2QFormerEncoder.layernorm` | `[nblocks, 3, 1024]` | (implicit) | `layer_norm(final_norm, q)` | none |
| Linear lift to LM hidden (1024 → 2048) | `GraniteSpeechEncoderProjector.linear` | `[nblocks * 3, 2048]` | `proj.out` | `linear(proj.linear, q.reshape(nblocks*3, 1024))` | none |

The projector yields `nblocks * 3` audio tokens, each in LM hidden
space. These tokens are scattered into the LM input embedding at
positions where `input_ids == audio_token_id (100352)`.

## Decoder (Granite-4 LM)

40 layers, hidden=2048, GQA(16 query / 4 KV) head_dim=128, RMSNorm,
SwiGLU, RoPE theta=10000, max_position=4096, vocab=100353.

**Granite-4 scalar multipliers** are applied inside the forward pass:
- `embedding_multiplier=12.0` — multiplied into `embed(tokens)` before
  the first block.
- `attention_multiplier=1/128=0.0078125` — replaces the standard
  `1/sqrt(head_dim)=1/sqrt(128)=0.08838834...` softmax scale. Crucially
  this is a much SMALLER scale (~11x).
- `residual_multiplier=0.22` — every residual add is `x = x +
  0.22 * sublayer_out` rather than `x + sublayer_out`.
- `logits_scaling=8.0` — final logits divided by 8 before sampling.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Token embed + multiplier | `embed_tokens(input_ids) * embedding_multiplier` | `[seq, 2048]` | `dec.token_emb` | `ggml_get_rows(token_embd, ids); ggml_scale(emb, 12.0)` | `causal_lm` (without scale) |
| Audio injection (scatter projector output at `audio_token_id` positions) | `GraniteSpeechForConditionalGeneration.get_audio_embeddings_for_replace` | `[seq, 2048]` | `dec.audio_injected` | host-side scatter into the embed buffer before upload. The audio token positions are known at prompt-build time. Same pattern as qwen3_asr's audio-token scatter. | qwen3_asr |
| Granite-4 layer (RMSNorm pre-attn → GQA attn with attention_multiplier scale + RoPE → residual×0.22 → RMSNorm pre-MLP → SwiGLU MLP → residual×0.22) | `GraniteModel.layers[i]` | `[seq, 2048]` | `dec.block.{0, n/2, n-1}.out` = `dec.block.{0,20,39}.out` | new `transcribe::granite_lm::build_layer`. Mirrors `causal_lm` minus q_norm/k_norm and with the residual_multiplier baked in. | `causal_lm` |
| Final RMSNorm | `GraniteModel.norm` | `[seq, 2048]` | `dec.out_before_head` | `rms_norm(out_norm)` | `causal_lm` |
| LM head (tied for -plus, separate for 1b/2b) + logits_scaling | `lm_head(out) / logits_scaling` | `[seq, 100353]` | `dec.logits_raw` | `linear(output.w or token_embd.w, x); ggml_scale(logits, 1/8)` | `causal_lm` plus the scale |

The forward graph is built once with `n_past=0, n_tokens=seq_len` for
the prompt pass and once with `n_past=K, n_tokens=1` for each step in
the autoregressive loop, identical to qwen3_asr's shape.

## Generation / KV Path

Greedy decode loop. After the prompt pass:
1. Read final-token logits, argmax → next token id.
2. Append to KV cache via step-graph (`n_past = prev_len, n_tokens = 1`).
3. Stop on `<|end_of_text|>` (id 100257) or `max_new_tokens`.

Mid-generation gate tensor: `dec.logits_raw.gen8` (after 8 step-loop
iterations) — REQUIRED per the Stage 4 skill for autoregressive
decoders. ACCEPTED GAP at Stage 4 sign-off: the Stage 2 dumper captures
only the prompt-pass `dec.logits_raw`; the C++ step graph reuses the
prefill block_step path which is exercised every time `transcribe-cli`
emits more than one token (every WER utterance), and the LibriSpeech
test-clean 512 subset WER (1b: 1.26%; 2b: 1.35%) covers thousands of
step iterations across the corpus. A dedicated `dec.logits_raw.gen8`
dump on both reference and C++ sides is queued for the next granite
iteration (it ships alongside the `-plus` `cat_hidden_layers` work).

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (en) | `model.generate()` produces non-empty English transcript via chat-template `USER: <audio> can you transcribe the speech into a written format?\n ASSISTANT:` | `transcribe-cli -m <gguf> --language en <wav>` → non-empty English transcript | one row per variant, all 3 PASS expected |
| Transcribe (fr/de/es/pt[/ja]) | per-language prompt template | per-variant lang flag → non-empty native transcript | one row per (variant, lang); PASS where audio sample is available, otherwise ACCEPTED GAP |
| Translate (1b / 2b) | `Translate this speech to {target}` chat-template variant | new `--translate-to <bcp47>` (or equivalent) CLI flag selects the translate prompt; output is a non-empty native-language transcript in the target language | one row per direction we have audio for; PASS expected for at least one X→En pair |
| Language detection | not advertised by granite (the chat template selects language explicitly) | n/a | n/a |
| Word timestamps (plus only) | upstream emits inline `<\|n.nn\|> word` markers under the timestamps prompt template | new `--word-timestamps` flag selects the timestamps prompt; runtime parses inline markers into `transcribe_word_t`-equivalent records and surfaces them via the existing word-timestamp output channel (same channel parakeet uses) | PASS expected on plus variant |
| Speaker diarization (plus only) | upstream supports diarized output via prompt template | not exposed by the v1 C++ CLI | SKIP — not exposed by runtime (revisit post-v1) |
| Streaming, VAD | not advertised | n/a | n/a |

## Deviations From Closest Analog

- **Shaw block-local attention**. Each Conformer block reshapes the
  sequence into `num_blocks = ceil(T / context_size)` blocks of
  `context_size = 200` frames each (last block right-padded with zeros
  and the pad rows masked to `-inf`). Within each block, attention is
  full bidirectional with a Shaw-style position bias matrix `pos_attn[i,
  j] = einsum_d(q[i], rel_pos_emb[j-i+max_pos_emb])`, where
  `rel_pos_emb` is a learned table of shape `[2*max_pos_emb+1=1025,
  dim_head=128]`. The bias is ADDED to `QK^T` before softmax, then
  scaled with `attention_multiplier=1/sqrt(head_dim)`. The shared
  `transcribe::conformer::rel_pos_mhsa` is Transformer-XL style (sin/cos
  pos table + `rel_shift` + `pos_bias_u/v`) which is structurally
  different, so granite implements its own attention helper in
  `src/arch/granite/encoder.cpp`. The block-major reshape can be done as
  a graph view + permute when `T` is a multiple of `context_size`;
  otherwise host-side right-pad applied to the encoder input before
  graph build (the pad amount is a function of mel length, known at
  encode time).
- **Self-conditioned CTC bypass**. At layer N/2 (=8 for 16-layer
  encoder) the post-block hidden is detached, projected through the
  CTC head (1024 → 348), softmaxed over the channel axis, then
  projected back via `out_mid` (348 → 1024) and added as a residual to
  the running hidden state before block 9. We bake this into the
  encoder graph; the auxiliary CTC logits are not surfaced to the
  caller. No in-tree analog.
- **`encoder.out_mid` is loaded but `encoder.out` is consumed only by the
  bypass**, not as a CTC head for inference. We load both tensors
  because the bypass uses both. The tensors are quantizable per the
  standard linear bucket; the bias goes to F32.
- **Q-Former projector** (BLIP-2). 2 layers of {self-attn, cross-attn,
  FFN with LN-after-residual}. No in-tree precedent. Build inline in
  `granite::projector.cpp`. The `intermediate_query/output_query`
  upstream names collapse to `ffn.up/ffn.down` in our GGUF since
  granite always runs the Q-Former in query-only mode.
- **Granite-4 LM**. Distinct from Llama / Qwen3 by the four scalar
  multipliers (embedding, logits, attention, residual) — every other
  scalar in the forward is parameterised by these. A missing multiplier
  silently degrades accuracy without crashing. New
  `src/granite_lm/granite_lm.{h,cpp}` mirrors `src/causal_lm/` but bakes
  the multipliers into the graph and drops q_norm/k_norm.
- **Audio-token scatter** is unchanged from qwen3_asr: build
  `inputs_embeds = embed(input_ids) * embedding_multiplier`, then host-
  side overwrite the rows where `input_ids == audio_token_id (100352)`
  with the projector output (linear-lifted to 2048). The result is
  uploaded as a graph input to the LM block stack. No `audio_start` /
  `audio_end` sentinel tokens — granite uses pure positional
  injection.

## Variant Notes

- **`granite-4.0-1b-speech`**: baseline. `cat_hidden_layers=[]`,
  `tie_word_embeddings=false` (separate `dec.output.weight`). Chat
  template: `USER: <content>\n ASSISTANT:`. Translation in 5
  directions (X→En for fr/de/es/pt/ja; En→X likewise) advertised but
  reaches the C++ CLI only through the language prompt — translation
  capability is `SKIP — not exposed by runtime` in v1.
- **`granite-speech-4.1-2b`**: same architecture as 1b. Same chat
  template, same capabilities. No code change vs 1b.
- **`granite-speech-4.1-2b-plus`**: `cat_hidden_layers=[3]` (encoder
  concatenates layer-3 hidden with final hidden, doubling the
  projector's cross-attention K/V input from 1024 to 2048),
  `tie_word_embeddings=true` (no separate `dec.output.weight`; lm_head
  reuses `dec.token_embd.weight`). Chat template is the full Granite-4
  `<|start_of_role|>system...` form, not the bare USER:/ASSISTANT:
  pair. Word-timestamps and speaker-diarization capabilities advertised
  but `SKIP — not exposed by runtime` in v1.
