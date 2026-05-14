# Moonshine Streaming

Status: supported (one-shot decode) with manifest-driven numerical
validation against the HF Transformers reference, `validate.py all`
passing on CPU. Streaming session API not yet wired into
`transcribe-cli` — the reference itself is one-shot at the time of port,
so the C++ scope mirrors that and the chunked-feed surface is a
follow-up.

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
- **Streaming: yes** (advertised). Phase 4a of the streaming API
  proposal exposes a *stream-of-whole* implementation: `stream_begin`
  buffers PCM, `stream_finalize` runs the existing one-shot inference
  path on the accumulated buffer. The streaming dispatcher, lifecycle
  state machine, and result-snapshot accessors all work; only the
  inference itself is non-incremental. Real incremental encoder /
  decoder is Phase 4b+ and keeps the same `supports_streaming` flag.
  See "Streaming validation strategy" below.
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
3. Streaming runtime contract: Phase 4a exposes the public streaming
   session API and CLI smoke path as stream-of-whole. Real chunked
   encoder feeding with persistent encoder state is still Phase 4b+.
   The reference itself is one-shot today (model card: "the current
   Transformers code path does not yet implement fully efficient
   streaming"), so Phase 4a validates the stream-of-whole path against
   the existing one-shot numerics.
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

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | PASS |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf samples/jfk.wav` | non-empty plausible transcript (English-only model — auto path is the same as explicit `en`) | PASS |
| Streaming | chunked feed, finalize at end (Phase 4a stream-of-whole) | `build/bin/transcribe-cli -m models/moonshine-streaming-tiny/moonshine-streaming-tiny-F32.gguf --stream-chunk-ms 500 samples/jfk.wav` | per-feed progress lines (`feed[i]: input=... ms buffered=... ms`), then a single `finalize:` line, then the final transcript. result_changed flips only at finalize for stream-of-whole — partial transcripts mid-stream are a Phase 4b feature. | PASS (Phase 4a) |

## Streaming validation strategy

Phase 4a wires the streaming session API to moonshine_streaming as
*stream-of-whole*: `stream_begin` snapshots the run params and clears
the family's audio scratch; `stream_feed` appends PCM to the scratch
buffer; `stream_finalize` invokes the same internal helper that
`transcribe_run` invokes (`run_one_shot_inner` in
`src/arch/moonshine_streaming/model.cpp`). Both entry points are
siblings of that helper — neither owns the inference. See the
"run/stream parity" comment block in model.cpp for why we chose
sibling-sharing over `run()` literally delegating through the public
stream hooks.

This gives the streaming-vs-batch parity the proposal asks for
*by construction* for Phase 4a:

1. **Upstream streaming reference**: the HF transformers code path is
   not fully streaming today (see the model card note and intake risk
   #3), so there is no upstream streaming reference to compare against.
   The streaming proposal's first validation tier (streaming whole vs
   upstream streaming) does not apply.
2. **Streaming whole vs upstream one-shot**: the existing WER smoke
   for moonshine_streaming runs the C++ one-shot path against the
   upstream HF one-shot transcripts. Because the streaming-of-whole
   path shares the inference helper, that same WER number certifies
   stream-of-whole equally — anyone with the checkpoint downloaded
   can confirm with `--stream-chunk-ms 500` on the same sample.
3. **Optimized one-shot vs stream-path parity**: not applicable at
   Phase 4a — there is only one inference path. This tier becomes
   meaningful when Phase 4b introduces a real incremental encoder /
   decoder; at that point an empirical parity test belongs alongside
   the WER smoke.

Phase 4b items to revisit when real incremental streaming lands:

- Per-feed `result_changed = true` partial transcripts (today only
  finalize flips this).
- `streaming_lookahead_ms` / `streaming_chunk_ms` capability hints
  (today both 0 — "unsupported or unknown").
- Empirical stream-vs-batch WER parity smoke (matched-text or
  matched-token-id assertion) using the same fixture audio the
  current run() smoke uses.

## Notes

- Moonshine Streaming is English-only and emits transcript-only output
  (no language tokens, no task tokens, no timestamp tokens). Translate /
  language-detect / timestamps rows are intentionally absent from the
  capability table because the model does not advertise them.
- First port targets greedy argmax single-utterance transcription as a
  one-shot pass through the encoder (matches the reference's current
  behavior; streaming session API is a post-port follow-up).
- The model card warns about hallucination loops on short / noisy
  segments; Stage 7 WER work should bound `max_new_tokens` per audio
  duration following the model-card guidance (`seq_lens * 6.5 / 16000`).
