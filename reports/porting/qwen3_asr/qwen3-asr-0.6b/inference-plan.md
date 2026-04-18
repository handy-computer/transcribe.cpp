# Qwen3-ASR-0.6B Inference Bring-Up Plan

Precondition: scaffolding is in (intake, family note, converter, reference
dumper, C++ loader through `build_qwen3_asr_weights`). `transcribe-cli`
loads the BF16 GGUF end-to-end and reports `not implemented` on run.
Everything below is graph construction and the decode loop.

Structure mirrors the audio-llm pattern in `docs/porting/0-porting.md`:
encoder first (validates independently), then LM prefill with audio
injection (validates against reference with real encoder output), then
the generation loop (validates against reference transcript). Every
step has a dump-point parity gate before moving on — no "hope it
composes" at the end.

## Phase 0 — Reference dumps exist

Unblocks everything that follows. If the author repo can't be
instrumented on this machine, we discover it now instead of after
writing 3k lines of C++.

1. Install the env: `uv sync --project scripts/envs/qwen3_asr` (pinned
   to Python >=3.11,<3.13 for macOS NeMo compat).
2. Run the reference dumper against `samples/jfk.wav` with no language
   hint (let it auto-detect).
3. Verify all expected `.f32` + `.json` dump points land in
   `build/validate/qwen3_asr/qwen3-asr-0.6b/jfk/ref/` and the transcript
   is intelligible English.
4. Fill real tolerances in `tests/tolerances/qwen3_asr.json` (replace
   the `"TODO"` markers with starting budgets ~matching Cohere's BF16
   scale; tighten during bring-up).
5. Update `scripts/validate.py` family registry so
   `validate.py all --family qwen3_asr` knows about us (small edit,
   same shape as Cohere).

**Gate:** `validate.py ref --family qwen3_asr` produces all dumps and a
transcript.

## Phase 1 — Audio encoder graph

New file: `src/arch/qwen3_asr/encoder.{h,cpp}`. One
`build_encoder_graph()` function returning an `EncoderBuild` struct
with graph, mel input, encoder output, and dump-point tensor handles.

1. **Mel input** → `enc.mel.in` dump (host-side, uploaded each run).
2. **Subsample:** three `ggml_conv_2d` (stride 2, pad 1) with GELU
   between, then `ggml_reshape_3d` to `[B, T', C*F]`, then `mul_mat`
   with `enc.conv_out.weight` → `enc.subsample.out`.
3. **Sinusoidal PE:** precompute host-side (matches
   `SinusoidsPositionEmbedding`, `max_source_positions=1500`), upload
   to a graph input, add → `enc.pos_add.out`.
4. **Chunked `cu_seqlens` mask:** host-side compute from
   `feature_lens` and `n_window_infer // (n_window*2)`, feed as
   additive attention bias (same pattern Cohere uses for
   padding-aware masks). First port assumes single-utterance input
   ≤30 s, so the chunking math is deterministic on `mel_n_frames`.
5. **18 encoder blocks** (pre-LN, bias-carrying linear projections,
   bidirectional): `layer_norm` → Q/K/V linear → `mul_mat` + softmax
   + `mul_mat` attention with the additive mask → residual →
   `layer_norm` → fc1 (GELU) → fc2 → residual. Dump
   `enc.block.0.out`, `enc.block.17.out`, plus any extras the
   tolerance file requests.
6. **Post head:** `ln_post` → `proj1` (GELU) → `proj2` → `enc.proj.out`
   (shape `[output_dim=1024, T_enc]`).

Run the encoder-only path, add a C++ debug dump hook matching the
reference contract. `validate.py compare` against Phase-0 dumps for
`enc.*` names. Fix drift before advancing.

**Gate:** encoder dumps match reference within tolerance.

## Phase 2 — LM prefill with audio injection

New files: `src/arch/qwen3_asr/decoder.{h,cpp}` and
`src/arch/qwen3_asr/prompt.{h,cpp}`.

1. **Prompt construction** (`prompt.{h,cpp}`): render the chat template
   to a token sequence. For the first port, implement a narrow
   renderer — no full Jinja engine — that handles exactly the template
   shipped in `chat_template.json`:
   `<|im_start|>system\n<system_text><|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>*N<|audio_end|><|im_end|>\n<|im_start|>assistant\n`.
   `N` = number of audio-frame placeholders, computed from encoder
   output length. If this turns out to need real Jinja later, revisit.
   Template is short and fixed across 0.6B and 1.7B.
2. **Token embedding lookup** → `dec.token_emb` dump.
3. **Audio injection scatter:** for every position where
   `input_id == audio_token_id (151676)`, replace the embedding row
   with the corresponding row of encoder output. Implementation:
   host-side gather of target positions, `ggml_set_rows` (or
   equivalent scatter) into the embedding tensor. Dump
   `dec.audio_injected`.
4. **Rotary embedding table:** precompute interleaved MRoPE cos/sin
   host-side from `mrope_section`, `rope_theta`,
   `rope_mrope_interleaved`. The first port assumes single-modality
   positions (all positions are "temporal" for text-only ASR), which
   reduces the 3D section rotation to flat RoPE — verify this
   reduction against the reference at `dec.block.0.out` before
   relying on it. If the reduction fails, fall back to materializing
   the full `[3, B, T, head_dim/2]` grid and interleaving per section.
5. **Causal mask** for prefill: standard lower-triangular. Audio
   positions can attend to each other and to preceding text; the
   chat-template layout is causal w.r.t. token order, so no custom
   masking beyond lower-triangular.
6. **28 LM blocks** (pre-LN RMSNorm, GQA 16/8 with per-head Q/K
   RMSNorm inside attention, SwiGLU MLP, RoPE on Q/K). Dump
   `dec.block.0.out` and `dec.block.27.out`.
7. **Final RMSNorm** → `dec.out_before_head` → `mul_mat` with tied
   `dec.token_embd.weight` transposed → `dec.logits_raw`.

Flash-attention use: head_dim=128 is supported by every ggml backend
we ship, so default `decoder_use_flash = true`. Not on the critical
path for correctness; drop if it complicates bring-up.

**Gate:** prefill dumps (`dec.token_emb`, `dec.audio_injected`, both
block outputs, `dec.out_before_head`, `dec.logits_raw`) match reference
within tolerance. First-token argmax on the logits matches the
reference's first generated token.

## Phase 3 — Generation loop with KV cache

Wire `QwenAsrContext::run()`:

1. Mel compute (already on the model via `MelFrontend`).
2. Encoder graph compute, read encoder output to host.
3. KV cache init sized for `max_new_tokens + prompt_len` (don't size
   for 65k — it's 2 GB; cap at something like 2048 by default with a
   param override).
4. Prefill graph: upload prompt tokens, upload encoder output, run,
   read argmax-on-GPU result (same pattern as Cohere — don't pull full
   vocab logits to host every step).
5. Step loop: single-token graph, reserve sched with a worst-case
   graph on first iteration, KV head advances, stop on
   `eos_token_id (151645, <|im_end|>)`.
6. Parse the generated text. Qwen3-ASR emits the transcript as
   "language X …"; for the first port, pass the raw string through to
   `full_text` and stuff it into a single `SegmentEntry` with zero
   timings (matches the advertised `TIMESTAMPS_NONE` capability).
   Extraction of the language tag into `caps.languages` can come later
   if we want to expose it.

**Gate:** `transcribe-cli -m qwen3-asr-0.6b-BF16.gguf samples/jfk.wav`
produces a transcript that normalizes equal to the reference's
transcript. `validate.py all --family qwen3_asr` passes.

## Phase 4 — Smoke tests

New files, following the Cohere layout:

- `tests/qwen3_asr_smoke.cpp` — fixture-backed loader smoke
  (shape/hparam checks only, no real weights). Needs a tiny synthetic
  fixture generator under `tests/fixtures/`.
- `tests/qwen3_asr_real_smoke.cpp` — load + generate a short
  transcript behind `TRANSCRIBE_BUILD_REAL_MODEL_TESTS`. Skips with
  RC 77 when `TRANSCRIBE_REAL_QWEN3_ASR_GGUF` is unset.
- `tests/qwen3_asr_e2e_smoke.cpp` — public-ABI round trip matching
  the Cohere e2e shape.

Wire into `tests/CMakeLists.txt`.

## Out of scope for this port

- **Streaming** (`stream_transcribe` / chunk rollback): sibling task,
  documented in family note.
- **Forced aligner**: separate checkpoint and family
  (`qwen3_forced_aligner`), different head contract.
- **Quantization** (Q8_0, Q5_K_M, …): comes after BF16 accuracy lands
  per `docs/porting/3-conversion.md`. `tools/transcribe-quantize` may
  need per-family bucket overrides for per-head norms; handle then.
- **Metal/Vulkan validation**: CPU is source of truth. Primary
  accelerator parity is a follow-on gate.
- **1.7B variant**: same architecture, different dims — should be a
  config-only port once 0.6B is validated.

## Risks and escape hatches

1. **MRoPE reduction assumption wrong** → fall back to the full 3D
   interleaved cos/sin table. Costs an extra scatter op per prefill,
   no semantic change. Gate is at end of Phase 2.
2. **Chunked `cu_seqlens` mask** edge cases with very short
   (<1 window) or boundary-aligned inputs → host-side mask
   construction is easy to debug; add a unit test for the mask builder
   if we hit trouble.
3. **Chat-template renderer too narrow** → first port ships with the
   0.6B template hard-coded. If 1.7B ships a different template, we
   either add a second renderer or pull in a real Jinja lib; decide
   when we see it.
4. **Tokenizer decode of Qwen BPE** — the existing tokenizer handles
   `gpt2` byte-level BPE. If it doesn't round-trip Qwen3's CJK-heavy
   output correctly, fix the tokenizer (shared across future Qwen-
   family ports); don't paper over it here.

## Size estimate

~1500–2000 lines of new C++ across encoder / decoder / prompt, plus
tolerance-file updates and test wiring. Each phase is independently
verifiable via `validate.py compare`, so failures localize.
