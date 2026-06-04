# Moonshine Streaming

Status: supported (one-shot decode + real-time streaming with
mid-stream partial transcripts) with manifest-driven numerical
validation against the HF Transformers reference, `validate.py all`
passing on CPU. Each `stream_feed` runs the encoder over a sliding
PCM window, applies the adapter on the new emit slice, projects
cross-attention K/V per layer, then re-decodes the transcript from
BOS — so callers see a live tentative transcript that grows and
self-corrects as audio arrives. The longest token-id prefix that
re-appeared identically across the last `stable_prefix_agreement_n`
feeds (default 3) is marked committed; tokens past the divergence
stay tentative until later feeds confirm them or `stream_finalize`
commits everything. Final transcript parity with `transcribe_run` is
validated by `transcribe_moonshine_streaming_stream_parity` across
chunk sizes 1, 20, 40, 80, 160, 500, 1000 ms.

Because the decoder re-attends over a growing cross-attention context,
a token that was committed on an earlier feed can still change in the
raw hypothesis later. `committed_text` is append-only and is **not**
rolled back, so when `full_text` revises a committed byte the
committed/tentative seam is transiently incoherent (committed_text +
tentative_text no longer reconstruct full_text). `committed_text` is
therefore best-effort for this family; `full_text` is the authoritative
raw hypothesis. Raising `stable_prefix_agreement_n` reduces the chance
of a wrong commit at the cost of committing later.

## Identity

- Family key: `moonshine_streaming`
- Upstream architecture string: `moonshine_streaming` (`MoonshineStreamingForConditionalGeneration`)
- Hugging Face repo: `UsefulSensors/moonshine-streaming-tiny` (first port; per-variant
  intakes pin each variant's repo + revision under
  `reports/porting/moonshine_streaming/<variant>/intake.json`)
- Hugging Face revision: `f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a`
- License: MIT (see model card)
- Variants:
  - `moonshine-streaming-tiny` (34M params, 6 enc / 6 dec layers, hidden 320, F32 reference dtype, English-only).
  - `moonshine-streaming-small` (123M params, 10 enc / 10 dec layers, encoder 620 / decoder 512, encoder attn-dim 512 = 8×64, F32, English-only).
  - `moonshine-streaming-medium` (245M params, 14 enc / 14 dec layers, encoder 768 / decoder 640, encoder attn-dim 640 = 10×64, F32, English-only).

Note: small/medium have **non-square encoder attention** —
`enc_d_model > enc_n_heads × enc_head_dim`. Encoder Q/K/V projections
take residual_dim → attn_dim and O projects attn_dim → residual_dim.
Tiny coincidentally has them equal (320 = 8 × 40) and so the original
arch port does not separate the two dims; small/medium force the
distinction.

## Relationship to the moonshine family

`moonshine_streaming` is a **separate family**, not a moonshine variant.
The decision is recorded in
`reports/porting/moonshine_streaming/moonshine-streaming-tiny/intake.json::known_risks[0]`.
Genuinely shared genes (BPE-32768 tokenizer, partial-RoPE attention
topology in the decoder, SwiGLU FFN, encoder-decoder pattern, MIT
license) are narrow; sharing should go through `scripts/lib/` helpers
and shared C++ utilities, not through co-location in `src/arch/moonshine/`.

Independent of moonshine:
- HF `model_type` is `moonshine_streaming` with a separate
  `MoonshineStreamingForConditionalGeneration` class.
- Encoder is "ergodic" (no positional embeddings on encoder self-attn) +
  sliding-window attention (per-layer (L,R) windows) — moonshine uses
  partial RoPE + full self-attention.
- Frontend is 50 Hz time-domain features + CMVN + 2 causal stride-2
  convs — moonshine uses a 3-conv stack on raw PCM at ~42 Hz.
- A new `adapter` layer between encoder and decoder injects learned
  positional embeddings.
- `tie_word_embeddings=false` (moonshine: true), `pad_token_id=0`
  (moonshine: 2), and the tokenizer `vocab_sha256` differs.
- Streaming runtime contract (chunked encoder feeding) — moonshine is
  one-shot.

## References

- Canonical reference: **transformers** —
  `transformers.models.moonshine_streaming.{MoonshineStreamingForConditionalGeneration, MoonshineStreamingConfig}`
  paired with `AutoProcessor` (which composes `Wav2Vec2FeatureExtractor`
  + `PreTrainedTokenizerFast`). The processor is
  `MoonshineStreamingProcessor` (distinct from the non-streaming
  `MoonshineProcessor`). No `trust_remote_code` required. The streaming
  class ships in mainline transformers (verified at v5.7.0).
- Instrumented reference: **transformers** (same class set, with
  forward hooks for tensor dumps in Stage 2).
- Cross-check references:
  - Upstream paper: `download.moonshine.ai/docs/moonshine_streaming_paper.pdf`
    (cite for the ergodic encoder + sliding-window + adapter
    architecture).
  - `refs/mlx/mlx-audio/mlx_audio/stt/models/` — does NOT yet contain a
    `moonshine_streaming/` entry as of intake date. Verify when porting;
    if absent, the upstream paper is the only architectural cross-read.

## Commands

The variant key is one of `moonshine-streaming-tiny`,
`moonshine-streaming-small`, or `moonshine-streaming-medium`. Examples
below use `moonshine-streaming-tiny`; substitute as needed.

Full validation (reference dump → C++ dump → tensor compare):

```bash
uv run scripts/validate.py all --family moonshine_streaming --variant moonshine-streaming-tiny
```

Reference dumps (encoder intermediates + decode prompt-pass + transcript):

```bash
uv run --project scripts/envs/moonshine_streaming \
  scripts/dump_reference_moonshine_streaming_transformers.py decode \
  --model UsefulSensors/moonshine-streaming-tiny \
  --audio samples/jfk.wav \
  --out build/validate/moonshine_streaming/moonshine-streaming-tiny/jfk/decode/ref \
  --torch-threads 1
```

Conversion:

```bash
uv run --project scripts/envs/moonshine_streaming \
  scripts/convert-moonshine_streaming.py UsefulSensors/moonshine-streaming-tiny
```

Quantize (after conversion produces the F32 reference GGUF):

```bash
build/bin/transcribe-quantize \
  models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf \
  models/moonshine-streaming-tiny/moonshine-streaming-tiny-Q8_0.gguf \
  --quant Q8_0
```

Performance benchmark:

```bash
uv run scripts/bench/run.py \
  --models moonshine-streaming-tiny \
  --quants f16,q8_0 \
  --samples jfk \
  --backends metal,cpu \
  --iters 5 --warmup 2 \
  --name moonshine-streaming-tiny-publication
```

WER sweep (LibriSpeech test-clean, 2620 utterances; one run per shipped quant):

```bash
uv run scripts/wer/run.py \
  --model models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-streaming-tiny-F32.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/moonshine-streaming-tiny-F32.test-clean.jsonl
```

HF reference WER on the same manifest (sanity check; CPU-only, ~4 min for tiny):

```bash
uv run --project scripts/envs/moonshine_streaming \
  scripts/dump_reference_moonshine_streaming_transformers.py wer \
  --model UsefulSensors/moonshine-streaming-tiny \
  --manifest samples/wer/test-clean.manifest.jsonl \
  --out reports/wer/moonshine-streaming-tiny-REF.test-clean.jsonl
uv run scripts/wer/score.py reports/wer/moonshine-streaming-tiny-REF.test-clean.jsonl
```

## Architecture summary

- Pattern: `encoder-decoder` (audio encoder + autoregressive text decoder
  with cross-attention) — but with an "ergodic encoder + adapter +
  decoder" three-stage shape rather than the conventional
  encoder-decoder.
- Frontend: **50 Hz time-domain features** with CMVN, followed by two
  causal stride-2 convolutions. `frame_ms=5.0` in `encoder_config`.
  Sample rate 16 kHz, `feature_size=1` (the conv input channel after the
  time-domain features). Whether CMVN statistics are stored as model
  weights or computed online is a Stage 2 question.
- Encoder: 6 transformer blocks, hidden 320, 8 heads (= 8 KV heads),
  GELU FFN (intermediate 1280), pre-norm, **no positional embeddings on
  self-attn (ergodic)**, **sliding-window self-attention** with per-layer
  windows `[(16,4),(16,4),(16,0),(16,0),(16,4),(16,4)]` — 16 left-context
  frames everywhere; 4 right-context (lookahead) on the first two and
  last two layers, 0 right-context (strictly causal) on the middle two.
  Total of ~80 ms lookahead from the four lookahead layers at 50 Hz.
- Adapter: aligns encoder→decoder dimensions and adds learned positional
  embeddings before decoder cross-attention. Exact op set is a Stage 2
  question; transformers source is the authoritative answer.
- Decoder: token embedding (vocab 32768), 6 transformer blocks with
  self-attn (causal, partial RoPE with `partial_rotary_factor=0.8`) +
  cross-attn (no RoPE) + SwiGLU FFN (`hidden_act=silu`,
  intermediate 1280 = 4 × hidden), pre-norm, final LN, **separate**
  logits head (NOT tied — `tie_word_embeddings=false`).
- Tokenizer: byte-level BPE, vocab 32768. Special tokens: bos=1, eos=2,
  pad=0, decoder_start=1. No language or task tokens. **Vocab SHA differs
  from non-streaming moonshine** — do not assume tokenizer parity.
- Generation contract: decoder seeded with `decoder_start_token_id=1`;
  greedy or temperature sampling until eos=2 or `max_length` (the model
  card recommends bounding `max_length` per audio duration via
  `seq_lens * 6.5 / 16000` to avoid hallucination loops).

## Capabilities (from intake)

- Languages: English only (`en`).
- Language detection: no.
- Translation: no.
- Timestamps: none (no timestamp tokens in vocab; model card does not
  advertise).
- **Streaming: yes** (advertised). Phase 4b-full of the streaming
  API proposal is implemented. Per `stream_feed`: incremental encoder
  over a sliding PCM window, adapter applied per-slice with absolute
  pos_ids, cross-attention K/V projected per-slice and accumulated in
  host buffers, then a fresh AR decode from BOS over the current
  cross-KV produces a live partial transcript. The committed prefix
  is the longest token-id prefix that re-appeared identically in two
  consecutive feeds; the tentative suffix may be revised by later
  feeds. `stream_finalize` commits everything. The streaming latency
  profile — ≈240 ms cumulative encoder right-context (the
  `(L=16, R∈{4,0}) × 6` sliding-window stack at 50 Hz, 20 ms/frame),
  natural 20 ms emit unit, family-recommended 80 ms feed cadence — is
  documented here rather than advertised as capability fields; the
  generic `supports_streaming` gate is the only streaming flag on
  `transcribe_capabilities`. See "Streaming validation strategy" below.
- VAD / diarization: no.

## Upstream benchmarks (from model card, Open ASR results table)

- LibriSpeech test-clean (en): 4.49 WER %
- LibriSpeech test-other (en): 12.09 WER %
- TED-LIUM (en): 6.12 WER %
- GigaSpeech (en): 13.90 WER %
- AMI (en): 19.03 WER %
- Earnings-22 (en): 20.27 WER %
- SPGISpeech (en): 6.16 WER %
- VoxPopuli (en): 14.02 WER %

Acceptance dataset for Stage 7 WER gate: **LibriSpeech test-clean**
(upstream-reported 4.49 % for tiny).

## Known risks

See `reports/porting/moonshine_streaming/moonshine-streaming-tiny/intake.json::known_risks`.
Highlights:

1. Family-placement decision (this is a separate family, not a moonshine
   variant). Cross-port code-sharing should go through `scripts/lib/`
   and shared C++ utilities, not co-location.
2. Novel encoder: ergodic (no pos-enc on self-attn) + sliding-window
   attention. Existing causal mask helpers will not cover the
   (L=16, R=4) lookahead pattern — need a new ggml mask shape.
3. Streaming runtime contract: Phase 4b-encoder runs the encoder
   incrementally over a sliding PCM window per feed and runs adapter +
   decoder once at finalize. Numerical parity with `transcribe_run` is
   asserted by the `transcribe_moonshine_streaming_stream_parity` test
   across a range of chunk sizes. Mid-stream partial transcripts
   (`result_changed=true` per feed) remain a Phase 4b-full follow-up.
   The HF Transformers reference itself is one-shot today (model card:
   "the current Transformers code path does not yet implement fully
   efficient streaming"); our streaming path is independent of that
   absence because the architecture is mechanically streaming-capable.
4. Adapter layer is new (not present in moonshine). Need to enumerate
   its ops at Stage 2 from the transformers source.
5. Frontend is novel (50 Hz time-domain + CMVN + 2 causal stride-2 convs).
   CMVN statistics location (model weights vs online) is a Stage 2
   question.
6. Config schema diverges from moonshine: encoder hyperparameters in a
   nested `encoder_config` sub-object. Converter and C++ loader must
   descend into it.
7. `tie_word_embeddings=false` — converter must NOT tie, GGUF must carry
   an explicit `lm_head` tensor.
8. `pad_token_id=0` (vs moonshine's 2). Tokenizer `vocab_sha256` differs.
9. `max_position_embeddings=4096` (vs moonshine's 194). Decoder KV cache
   sizing must accommodate the longer max length.

## Capability Validation

One row per advertised capability. Stage 1 drafts the rows with
`Status: TODO`; Stage 4 fills in the observed `Status` after running
each command. Allowed statuses:

- `PASS` — command ran and the observable matched.
- `SKIP — not exposed by runtime` — capability is advertised upstream
  but the public CLI/API does not surface a way to verify it. The row
  stays here so readers see the gap honestly.
- `ACCEPTED GAP — <reason>` — capability is exposed but intentionally
  not exercised here; reason names what would unblock the row.

All three variants (`tiny`, `small`, `medium`) share one implementation
and frontend, so the `Transcribe` rows below — shown with `tiny` as the
representative — hold identically for `small` and `medium` (substitute
the model path). Streaming is the per-variant-audited capability and gets
an explicit row each; all three were verified streaming-final == one-shot
on 2026-06-04.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf samples/jfk.wav` | non-empty plausible transcript (English-only model — auto path is the same as explicit `en`) | PASS |
| Streaming (tiny) | chunked feed with live partial transcripts (Phase 4b-full) | `build/bin/transcribe-cli -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf --stream-chunk-ms 500 samples/jfk.wav` | per-feed lines show a growing `partial="..."` tentative transcript as audio arrives ("And so" → "And so my fellow" → ...), with `result_changed=true` and `revision` bumping whenever the text advances. `buffered_ms` settles at the published 240 ms right-context. `finalize:` commits the final transcript, which matches one-shot. `transcribe_moonshine_streaming_stream_parity` asserts: (a) final text matches one-shot across chunk sizes 1..1000 ms, (b) revision monotonic, (c) `n_committed_tokens` monotonic, (d) `result_changed=true` implies text changed, (e) `n_committed_tokens == n_tokens` after finalize. | PASS (Phase 4b-full) |
| Streaming (small) | chunked feed (Phase 4b-full) | `build/bin/transcribe-cli -m models/moonshine-streaming-small/moonshine-streaming-small-F32.gguf --backend cpu --threads 1 --stream-chunk-ms 500 samples/jfk.wav` | final streamed transcript byte-equal to one-shot; same partial/revision/commit invariants as the `tiny` row | PASS (Phase 4b-full) |
| Streaming (medium) | chunked feed (Phase 4b-full) | `build/bin/transcribe-cli -m models/moonshine-streaming-medium/moonshine-streaming-medium-F32.gguf --backend cpu --threads 1 --stream-chunk-ms 500 samples/jfk.wav` | final streamed transcript byte-equal to one-shot; same partial/revision/commit invariants as the `tiny` row | PASS (Phase 4b-full) |

## Streaming validation strategy

Phase 4b-full runs the encoder, adapter, cross-attention K/V
projection, and AR decoder all incrementally as audio arrives. The
property that makes the streaming path equivalent to one-shot is the
encoder's *ergodicity*: there is no positional encoding on encoder
self-attn, every layer uses a sliding-window mask with bounded `(L, R)`
context, and the frontend convs are causal. Output frame `t` is a
function only of conv-stack output frames in `[t - L_total, t + R_total]`
where `L_total = sum_i max(0, L_i - 1) = 90` and
`R_total = sum_i max(0, R_i - 1) = 12` for the tiny variant. So
re-running the encoder over a sliding window containing those
neighbors produces frame `t`'s value bit-identically.

The adapter (`pos_emb` get_rows + add + optional `proj`) is indexed by
absolute frame so it can run on a slice with pos_ids
`[abs_frame_offset, abs_frame_offset + n_frames)`. The cross-attention
K/V projections are per-frame linear, so projecting a slice and
concatenating across feeds produces the same values a one-shot batched
projection would.

`stream_feed`:

1. Append new PCM, bump `input_received_ms`.
2. Compute `available_T = floor(total_pcm / samples_per_enc_frame)`
   in absolute encoder-frame units (the PCM buffer is sliding;
   `pcm_start_sample` tracks the absolute sample index of buffer[0]
   so the absolute frame count keeps growing across trims).
   `stable_T = available_T - R_total` is the frame index past which
   right-context isn't yet available.
3. For new stable frames `[T_emitted, stable_T)`, build a PCM window
   covering `[T_emitted - L_total - frontend_pad, stable_T + R_total)`
   in encoder-frame units (with `frontend_pad = 4` enc frames of
   conv-stack history beyond the L_total mask context), encode, then
   on the emit slice `[T_emitted, stable_T)`:
   - apply the adapter with absolute pos_ids → append to
     `stream_adapter_committed`;
   - run the cross-KV projection graph → append per-layer K and V to
     `stream_cross_k_committed[il]` / `stream_cross_v_committed[il]`.
4. `T_emitted = stable_T`. Bump `audio_committed_ms` to match.
   `result_changed = false` (no tokens emitted yet).
5. **PCM trimming**: drop samples in `[pcm_start_sample,
   (T_emitted - L_total - frontend_pad) * samples_per_enc_frame)`
   from the front of `stream_pcm_buffer`. Bumps `pcm_start_sample`
   to the new boundary. The retained suffix covers every PCM
   sample any future feed (or the finalize tail) can possibly
   need — at the tiny variant that's ~1.8 s of audio.
6. **Partial decode**: if any new encoder frames committed since
   the previous decode, allocate / reuse the `kv_cache` for the
   current `T_emitted`, upload the accumulated per-layer K/V buffers
   via `build_cross_kv_commit_graph`, and run the AR greedy decoder
   from BOS over the cross-attended cache. Result text fields
   (`tokens`, `segments`, `full_text`) are overwritten from scratch.
   Commit prefix: compute longest token-id prefix shared with the
   previous decode → `n_committed_tokens`. Bump `stream_revision`
   and set `update.result_changed = true` iff `full_text` actually
   changed.

`stream_finalize`:

1. Right-pad the trailing PCM to a multiple of `enc_frame_len`
   (matches one-shot's `right_pad_pcm`). `pcm_start_sample` stays
   frame-aligned.
2. If frames remain past `T_emitted`, run one more
   `flush_stable_frames` over `[T_emitted, T_total)` with no
   right-context margin. Adapter + cross-KV K/V projections for the
   tail append to the host buffers as in feed.
3. If `T_emitted` advanced past `stream_last_decoded_T` (or no
   result yet), re-run `decode_partial` so the final transcript
   reflects the just-added tail frames. Otherwise the last feed's
   decode is already final and we keep it as-is. Note finalize
   ignores the decode-interval throttle — the final decode always
   runs when there's new audio to fold in.
4. Commit everything: `n_committed_tokens / words / segments` are
   set to the full counts. Bump `stream_revision` to mark the
   state transition.

### Decode throttle

The per-feed AR decode is the bulk of the streaming compute cost (it
re-runs from BOS over the full cross-KV, so cost scales with the
emitted token count). Decoupling decode cadence from feed cadence is
critical: callers feeding 10–20 ms chunks from a microphone would
otherwise trigger a decode every 20 ms of audio, which is overkill
for any real UI and quickly slower than real-time on the smaller
variants.

The decode throttle lives in
`transcribe_moonshine_streaming_stream_ext::min_decode_interval_ms`,
passed through `transcribe_stream_params::family`:

- **`-1`** (sentinel, the default of
  `TRANSCRIBE_MOONSHINE_STREAMING_STREAM_EXT_INIT`): resolved
  to the family default of **240 ms** = one cumulative-right-context
  window = ~4 partial-transcript updates per second after warmup.
- **`0`**: decode on every encoder-frame advance (no throttle).
  Useful for tests / benchmarks; not recommended for end-user UIs.
- **`>0`**: minimum gap in milliseconds of audio between successive
  partial decodes.

Examples on jfk (11 s, 20 ms feed chunks):

| `min_decode_interval_ms` | Partial updates |
|---|---|
| 0 (no throttle)        | ~75 |
| 240 (default)          | ~29 |
| 500                    | ~17 |

In all cases the **final** transcript at `stream_finalize` matches
`transcribe_run` byte-for-byte. Only the intermediate update count
and CPU cost change.

This knob exclusively affects how often per-feed partial transcripts
fire; it does not change the ≈240 ms right-context floor on when audio
can first contribute to a stable encoder frame.

Validation tiers (proposal terminology):

1. **Upstream streaming reference**: the HF transformers code path is
   not fully streaming today (see the model card note and intake risk
   #3), so there is no upstream streaming reference to compare against.
2. **Streaming whole vs upstream one-shot**: the existing WER smoke
   for moonshine_streaming runs the C++ one-shot path against the
   upstream HF one-shot transcripts. Because streaming feed/finalize
   share `decode_from_committed_enc`, that WER number certifies the
   streaming path equally.
3. **Optimized one-shot vs stream-path parity**: covered by
   `tests/moonshine_streaming_stream_parity.cpp`. The test loads jfk
   audio, runs `transcribe_run`, then runs `stream_begin/feed/finalize`
   with chunk sizes `{1, 20, 40, 80, 160, 500, 1000} ms` and asserts
   identical final transcript bytes vs the one-shot reference. Per-feed
   invariants: `revision` monotonic, `n_committed_tokens` monotonic,
   `result_changed=true` implies `full_text` differs from the prior
   snapshot, `n_committed_tokens == n_tokens` after finalize, and
   post-stream `transcribe_run` still produces the reference text.
   Decode throttle: separate cases with `min_decode_interval_ms`
   `{0, -1=default, 500}` assert that each higher throttle strictly
   reduces the partial-update count, that all three produce identical
   final transcripts, and that the default-throttle update count
   falls in a reasonable range (20..80 for 11 s of audio).

Follow-ups beyond Phase 4b-full:

- **KV cache resize amortization**: today the `kv_cache` is
  free + re-allocated when `T_enc` changes between decodes. For
  small `T_enc` deltas (one extra frame), the realloc is wasteful.
  Doubling-grow with a separate `T_active` field would amortize
  this to amortized-constant per feed at the cost of a small
  refactor in `build_decoder_graph_kv` (stride vs length).
- **Alignment-aware commit**: today the commit prefix is the
  longest token-id match across two consecutive decodes. A
  cross-attention-alignment-aware version could commit a token as
  soon as its dominant encoder-frame anchor is safely before the
  right-context-pending tail, which is a stronger guarantee and
  would let callers display committed words sooner.

## Notes

- Moonshine Streaming is English-only and emits transcript-only output
  (no language tokens, no task tokens, no timestamp tokens). Translate /
  language-detect / timestamps rows are intentionally absent from the
  capability table because the model does not advertise them.
- First port targets greedy argmax single-utterance transcription. The
  initial Stage 4 work shipped the one-shot path (matches HF reference);
  Phase 4b-encoder layered real streaming on top by reusing the same
  encoder + decoder helpers.
- The model card warns about hallucination loops on short / noisy
  segments; Stage 7 WER work should bound `max_new_tokens` per audio
  duration following the model-card guidance (`seq_lens * 6.5 / 16000`).
