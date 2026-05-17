# Forward map - granite_nar

Reference: HuggingFace `transformers` v5.5.3 with `trust_remote_code=True`
on `ibm-granite/granite-speech-4.1-2b-nar` @ `7d20732d` — the repo's own
`modeling_nle.py` (`NLENARDecoder`, `NLENARForConditionalGeneration`,
`EncoderProjectorQFormer`) is the source of truth.
Closest in-tree analog: `src/arch/granite/` (AR sibling) for the
Conformer encoder geometry, the Shaw block-local attention helper, the
self-conditioned CTC bypass, the mel frontend (htk + power=2 + no log +
2-frame stack), and the Granite-4 LM block math. The bidirectional-LLM
forward and the NLE-style query+pool projector are new in-tree.

Single-variant family at intake time:
`granite-speech-4.1-2b-nar`. Future NAR releases would slot in here.

## Frontend

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| torchaudio MelSpectrogram (n_fft=512, win=400 zero-padded to 512, hop=160, hann_periodic, htk mel scale, no mel norm, power=2.0, no log, no per-utterance norm) + 2-frame stack → 160-dim encoder input | `NLEFeatureExtractor` (flat config; numerically identical to the AR `GraniteSpeechFeatureExtractor`) | `[T_mel/2, 160]` | `enc.mel_in` | `transcribe::MelFrontend` configured for `mel_norm="htk"`, `log=false`, `power=2`; followed by a host-side 2-frame stack. Same code path as the AR granite family. | `src/arch/granite/` mel frontend; identical preprocessor |

The frontend is bit-identical with the AR sibling. Stage 2's
`enc.mel.in` sidecar (the C++ runner reuses `enc.mel_in` for the same
tensor) is the canonical target.

## Encoder

Single Conformer geometry (same as AR granite, with NAR-specific extras):
16 layers, `hidden_dim=1024`, 8 heads × `dim_head=128`, `context_size=200`,
`max_pos_emb=512`, `conv_kernel=15`, `conv_expansion=2`, `ffn_mult=4`,
`input_dim=160`, `output_dim=348` (char-CTC head),
`bpe_output_dim=100353` (BPE-CTC head), `bpe_pooling_window=4`,
`self_conditioning_layer=8`, `attn_type=block`, `loss_lambda=0.2`.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Pre-encode (linear lift 160 → 1024) | `NLEConformerEncoder.input_linear` | `[T, 1024]` | `enc.input_linear.out` | `linear(input_linear.w, x) + input_linear.b` | `src/arch/granite/encoder.cpp` (identical) |
| Conformer block (FF1 → block-local Shaw self-attn → conv module (BN) → FF2 → post LN) | `NLEConformerBlock` | `[T, 1024]` | `enc.block.{0, n/2-1, n/2, n-1}.out` = `enc.block.{0, 7, 8, 15}.out` | hand-built `granite_nar::build_encoder_block`. FF1/FF2 use `transcribe::conformer::macaron_ff_residual`. Conv module uses `transcribe::conformer::conv_module` with BatchNorm + asymmetric pad. Block-local Shaw self-attn implemented inline in `src/arch/granite_nar/encoder.cpp`. | granite's `shaw_block_attn` helper |
| Self-conditioned CTC bypass at layer N/2 (=8) | inline in `NLEConformerEncoder.forward` | `[T, 1024]` | mid-CTC blank probs surface as `enc.mid_blank_probs` (debug-only) | After block 7 emits `h`, compute `h_mid = softmax(char_ctc_proj(h))`, then `h := h + ctc_bypass(h_mid)` before block 8. | granite (identical pattern) |
| Per-layer hidden capture for the projector | `NLENARForConditionalGeneration.encoder.forward` returns `(hidden, all_layer_outputs)` and the projector indexes `encoder_layer_indices=[4, 8, 12, -1]` | 4 captured tensors of `[T, 1024]` each, concatenated along channel → `[T, 4096]` | `enc.cat_out` | host-side capture of `block_out[3]`, `block_out[7]`, `block_out[11]`, `block_out[15]` (the `[4,8,12,-1]` indices are 1-based in the reference); 4-way `ggml_concat` along ne[0]. | none — NAR-specific multi-layer fanout |
| Char-CTC head | `NLEConformerEncoder.linear_decoder` (1024 → 348) | `[T, 348]` | `enc.ctc_logits` | `linear(ctc_proj.w, x) + ctc_proj.b` | granite (identical) |
| BPE-CTC head | `NLEConformerEncoder.bpe_linear_decoder` (1024 → 100353) with 4-window mean pooling on hidden first | `[T/4, 100353]` | `enc.ctc_bpe_logits` | mean-pool hidden over a 4-frame window, then linear. Output is the initial CTC hypothesis fed to the bidirectional LLM editor. | none |

Block dump indices: `{0, n/2 - 1, n/2, n - 1} = {0, 7, 8, 15}` — the
pre-bypass and post-bypass blocks are both dumped so a regression in
the CTC-bypass path is localized between `block.7.out` and `block.8.out`.

## Projector (NLE simplified Q-Former)

`EncoderProjectorQFormer` from `modeling_nle.py`. Strictly simpler than
the AR BLIP-2 Q-Former: no self-attention sublayer, no cross-attn
concat trick. 2 layers; each layer is **pre-LN → cross-attn → residual
→ pre-LN → FFN (SiLU) → residual**. The Q-side is a `mean-pool(K/V) +
learned query` initialization.

`block_size=15`, `downsample_rate=5` → `n_query=3`. Projector hidden
`prj_hidden=2048`, `n_heads=32` → `head_dim=64`. K/V input dim
`enc_layers × enc_hidden = 4 × 1024 = 4096`. LLM hidden 2048.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Per-encoder-layer LayerNorm on the 4-way cat (`[T, 4096]`) | `EncoderProjectorQFormer.layer_norms[i]` | `[T, 4096]` | (implicit) | host-side slice into 4×`[T, 1024]`, per-slice LN with its own gamma/beta, concat back along channel | none — NAR-specific |
| Linear lift to projector hidden (4096 → 2048) + GELU | `EncoderProjectorQFormer.layer_projector` | `[T, 2048]` | (implicit) | `gelu(linear(layer_projector.w, x))` | none |
| Right-pad to `nblocks × block_size` along T, reshape to `[2048, 15, nblocks]` | `EncoderProjectorQFormer.forward` window-slicer | `[2048, 15, nblocks]` | (implicit) | `ggml_concat(proj_lp, zero_pad, dim=1)` then `ggml_reshape_3d` | none |
| K/V: `windowed + window_positions` (broadcast position embed) | `EncoderProjectorQFormer.window_positions` | `[2048, 15, nblocks]` | (implicit) | `ggml_add(windowed, window_positions_broadcast)` (ne[2]=1 broadcast over nblocks) | none |
| Q: `mean_pool(windowed, downsample=5) + learned_query[2048,3]` | `EncoderProjectorQFormer.query` | `[2048, 3, nblocks]` | (implicit) | reshape windowed to `[h, downsample, n_query, nblocks]`, permute to put downsample on ne[0], `ggml_mean`, broadcast-add `query[h,n_query,1]` | none |
| 2-layer Q-Former (cross-attn-only + FFN, both pre-LN, no self-attn) | `Blip2QFormerLayer`-style but no self-attn block | `[2048, 3, nblocks]` | `proj.qformer.out` | `granite_nar::build_projector_graph` — per layer: `q_norm = LN(query); attn = cross_attn(q_norm, kv, kv)` then add residual; `f_norm = LN(query); ff = down(silu(up(f_norm)))` then add residual | granite's Q-Former minus self-attn |
| Final out_norm + linear (2048 → 2048 LLM-space) | `EncoderProjectorQFormer.out_norm` + `out_linear` | `[2048, nblocks × 3]` | `proj.out` | LN → reshape to `[2048, n_query × nblocks]` → linear | none |

The projector yields `nblocks × 3` audio tokens, each in LM hidden
space. Unlike the AR family there is **no audio-token scatter into the
LM input** — the projector output flows directly into the bidirectional
LLM as a contiguous prefix (see Generation / KV Path).

## Decoder (bidirectional Granite-4 LM)

Same dims as the AR granite-4 LM (40 layers, hidden=2048,
GQA(16Q/4KV), `head_dim=128`, RMSNorm, SwiGLU, RoPE θ=10000,
`max_position=4096`) but with:
- `tie_word_embeddings=true` (no separate `dec.output.weight`).
- `vocab_size=100352` (vs 100353 in AR).
- **Causal mask disabled** (`is_causal=False` on every layer). Forward
  is a single bidirectional pass; no KV cache.

Granite-4 scalar multipliers (same as AR):
- `embedding_multiplier=12.0` — applied to the flat input embeds.
- `attention_multiplier=1/128` — replaces the standard `1/sqrt(head_dim)`
  softmax scale.
- `residual_multiplier=0.22` — every residual add is `x = x +
  0.22 × sublayer_out`.
- `logits_scaling`: applied by `GraniteForCausalLM.forward` but the NLE
  reference **bypasses that wrapper** and calls `llm.lm_head(text_x)`
  directly. We mirror that: raw lm_head logits, no `/8`. Argmax is
  scale-invariant so the transcript is unaffected, and the
  `dec.text_logits` dump is value-aligned with the reference.

| Stage | Reference location | Output shape | Gate tensor | ggml / C++ pattern | In-tree analog |
|-------|--------------------|--------------|-------------|--------------------|----------------|
| Flat input embeds = `cat(audio_in/embedding_multiplier, embed_tokens(text_ids_with_eos_slots))` along the sequence axis | `NLENARForConditionalGeneration.forward` | `[2048, T_total]` where `T_total = n_audio + n_text` | `dec.flat_embeds` | host-side: `audio_in` arrives pre-divided by `embedding_multiplier`; `text_emb = ggml_get_rows(token_w, text_ids)` then `ggml_cast` to F32; `ggml_concat(audio_in, text_emb, dim=1)`. Audio rows round-trip identity through the post-multiplier. | granite AR token-embed + audio-scatter, but flipped to a concat instead of a scatter |
| Apply `embedding_multiplier=12.0` to the whole flat sequence | `× embedding_multiplier` | `[2048, T_total]` | (implicit) | `ggml_scale(flat, 12.0)` | granite |
| Bidirectional Granite-4 layer (RMSNorm pre-attn → GQA attn with `1/128` scale + RoPE + **no causal mask** → residual×0.22 → RMSNorm pre-MLP → SwiGLU MLP → residual×0.22) | `NLENARDecoder.layers[i]` (built from `transformers.models.granite` with `is_causal=False`) | `[2048, T_total]` | (no per-layer dump in current coverage; see Deviations) | `granite_nar::block_bidi`. Mirrors AR `granite_lm` minus the KV cache, plus `ggml_soft_max_ext(kq, mask=nullptr, scale=1/128, max_bias=0)`. RoPE positions span `[0, T_total)`. | AR granite `granite_lm::build_layer` |
| Final RMSNorm | `NLENARDecoder.norm` | `[2048, T_total]` | (implicit) | `rms_norm(out_norm, x)` | granite |
| Slice text portion `text_x = x[:, n_audio_tokens:]` | inline in `NLENARForConditionalGeneration.forward` | `[2048, n_text]` | (implicit) | `ggml_view_2d` + `ggml_cont` | granite (slice of audio-aligned positions in AR) |
| Raw lm_head over text positions | `llm.lm_head(text_x)` (no `/logits_scaling`) | `[100352, n_text]` | `dec.text_logits` | `ggml_mul_mat(token_w, text_x)` (tied embeddings) | granite (with `/logits_scaling`) |

## Generation / KV Path

**No autoregressive generation, no KV cache.** A single bidirectional
forward pass produces raw lm_head logits over the text-slot portion.
The post-graph CTC decode (argmax over each text position, drop EOS,
collapse repeats) yields the final transcript.

Hypothesis seeding (host-side, pre-LLM forward):
1. Run encoder → char-CTC and BPE-CTC heads.
2. CTC-decode `enc.ctc_bpe_logits` (argmax → drop blanks → collapse
   repeats) to an initial BPE hypothesis of length `n` tokens.
3. Build the text-slot template via
   `granite_nar::add_insertion_slots(hyp_ids, eos_id, out)`: place EOS
   at every even index, the hypothesis tokens at odd indices, length
   `max(2n + 1, 8)`. The bidirectional LLM "edits" this template in one
   shot.
4. Concatenate the projector output as the audio prefix; the LLM sees
   `[audio_tokens, text_template]`, attends bidirectionally, and emits
   per-position logits over the BPE vocab.
5. `granite_nar::argmax_collapse_drop_eos` on the text-slice logits
   yields the final token sequence.

There is no `dec.logits_raw.gen<N>` analog — the architecture is
single-pass non-autoregressive, so the `n_past > 0` step-graph code
path the Stage 4 skill calls out for AR decoders does not exist here.

## Capabilities And Language Controls

| Capability | Reference behavior | C++ API behavior | Family-doc Capability Validation row |
|------------|--------------------|------------------|--------------------------------------|
| Transcribe (en/fr/de/es/pt) | `NLENARForConditionalGeneration.forward(audio)` produces logits → CTC decode → text; language is implicit in the model weights (no language flag) | `transcribe-cli -m <gguf> <wav>` and the `--language` form are equivalent — language is not used by NAR | one row per (variant); PASS expected |
| Translation | NOT advertised by NAR (single-task ASR) | n/a | `SKIP — not exposed by NAR family` |
| Word/segment timestamps | NAR encoder pools to per-window output; no per-token timing recovery | not exposed | `SKIP — not exposed by runtime` |
| Streaming, VAD | not advertised | n/a | n/a |

## Deviations From Closest Analog

- **Bidirectional LLM forward** is the central deviation from the AR
  granite family. Every attention block runs `ggml_soft_max_ext(kq,
  mask=nullptr, scale=attention_multiplier, max_bias=0)`. Cross-checked
  via `validate.py all`: the `dec.text_logits` drift (max 2.86 / mean
  0.26 against a patched-bidirectional reference) sits at the bf16
  reduction-order noise floor for a 40-layer stack, not at the orders
  of magnitude that would indicate a missed/applied causal mask.
- **Reference fidelity for the bidirectional forward** required
  monkey-patching
  `transformers.models.granite.modeling_granite.create_causal_mask` to
  `None` in the Stage 2 dumper and Stage 7 REF runner. Without that
  patch, eager + transformers' auto-built causal mask produces causal
  attention even when `layer.is_causal=False`; only `flash-attn-2`
  bypasses it natively (Mac dev does not have flash-attn-2). With the
  patch, eager matches flash-attn-2 semantics. Verified by
  reproducing the model card's LibriSpeech test-clean WER to two
  decimals (1.29%).
- **No `dec.logits_raw.gen<N>` mid-generation tensor**. The Stage 4
  skill mandates this for autoregressive decoders to exercise the
  `n_past > 0` step-graph code path. NAR has no step graph (single
  bidirectional forward), so there is no `n_past > 0` regime to gate.
  This is an architectural property of the family, not a coverage gap.
- **Raw lm_head logits**, not `lm_head / logits_scaling`. The reference
  NLE bypasses `GraniteForCausalLM.forward` and calls `llm.lm_head`
  directly. We mirror that so the `dec.text_logits` dump is value-
  aligned with the reference. CTC decode is argmax, which is scale-
  invariant, so the transcript is unaffected either way.
- **NLE simplified Q-Former projector**: no self-attention sublayer,
  no cross-attn-of-encoder-frames-as-K trick from the AR Q-Former.
  Initial query is `mean-pool(windowed K/V along the downsample=5 axis)
  + learned 3-query`. 2 cross-attn+FFN layers, both pre-LN. The K/V
  side already has `window_positions` broadcast added so the
  cross-attn sees positional information without a separate pos-embed
  module.
- **4-layer multi-hidden encoder fanout**. The projector consumes the
  concatenated outputs of encoder layers `{4, 8, 12, -1}` (1-based in
  the reference; 0-based in the C++ runner: `block_out[3]`,
  `block_out[7]`, `block_out[11]`, `block_out[15]`). Per-layer LN is
  applied inside the projector; the C++ encoder dumps the
  concatenated `enc.cat_out` for cross-check.
- **No audio-token scatter into the LM**. AR granite scatters
  projector tokens at `audio_token_id (100352)` positions; NAR
  concatenates audio tokens as a prefix to the text-slot embeds. There
  is no `audio_token_index` in the NAR config.
- **Tied embeddings.** `tie_word_embeddings=true`, so the loader does
  not look for a separate `dec.output.weight`; `lm_head` reuses
  `dec.token_embd.weight`. Different from AR 1b/2b (untied) and
  matching AR `granite-speech-4.1-2b-plus` (tied).
- **Shape-stability caveat on `dec.*` tensors**. The NAR forward is
  shape-dependent on the BPE-CTC initial hypothesis
  (`n_text = max(2n + 1, 8)`). On marginal inputs, tiny bf16 noise at
  the CTC head's argmax can flip frames between blank and non-blank,
  producing different `n_text` between reference and C++ and making
  point-wise tensor comparison impossible. The jfk sample is robust;
  other audio cases must be verified before being added to the dump
  coverage.

## Variant Notes

- **`granite-speech-4.1-2b-nar`** (only variant at intake time): the
  baseline described in this map. Future NAR releases from IBM would
  declare any architectural deltas here.
