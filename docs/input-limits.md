# Audio Input Limits

How transcribe.cpp bounds audio input length, what a caller observes when a
limit is reached, and how to discover the limit before calling. This is both
the design reference for the library and the contract a consumer can rely on.

## The one-paragraph version

There is no single global length limit. Each model family falls into one of
three buckets. Some families chunk long audio internally and have no practical
limit; some have a hard context-window limit and reject over-length input
*before* running; some accept any length but lose accuracy past the window they
were trained on and warn when you cross it. Whatever the bucket, the library
never truncates silently: an over-length input is rejected up front with
`TRANSCRIBE_ERR_INPUT_TOO_LONG`; a transcript that runs into the context or
generation budget mid-decode returns the hard status
`TRANSCRIBE_ERR_OUTPUT_TRUNCATED` (with the partial transcript still readable
and `transcribe_was_truncated()` set); and a soft-window family logs a `WARN`
and proceeds. Every model reports its usable limit through
`transcribe_capabilities::max_audio_ms` (or, per-session,
`transcribe_session_get_limits()`) so you can check before you call. (Streaming
is the one exception to the non-OK truncation rule — see below.)

## Discovering the limit (consumer-facing)

Read `transcribe_capabilities::max_audio_ms` after loading a model:

```c
struct transcribe_capabilities caps;
transcribe_capabilities_init(&caps);
transcribe_model_get_capabilities(model, &caps);

if (caps.max_audio_ms == 0) {
    /* No practical limit. See the family notes for any absolute safety cap. */
} else {
    /* Usable ceiling in milliseconds of 16 kHz audio. Longer input is
       either rejected (hard-cap families) or warned-and-degraded
       (soft-window families) — see the family's bucket below. */
}
```

`max_audio_ms` is a single honest number derived from the model's real
metadata, not a hardcoded guess. For hard-cap families it is computed from the
decoder context window minus a **representative** prompt and a generation
reserve — the same shape as the upfront gate, but the gate uses the *exact*
prompt for each call (which shifts a little with the language hint and other
options), so `max_audio_ms` is advisory and slightly conservative, not an
exact per-call bound. For soft-window families it is the advisory window the
model was trained on. `0` means "no practical limit." It is a model-level
value at the default context; for the effective limit under a lowered
`n_ctx`, use `transcribe_session_get_limits()`.

## The three buckets

### 1. Chunked / unbounded — "just works"

| Families | Limit | Behavior |
| --- | --- | --- |
| whisper, parakeet (all variants), voxtral_realtime | none (`max_audio_ms = 0`) | Long audio is windowed internally and stitched (whisper), processed by an unbounded/streaming encoder (parakeet, voxtral_realtime), or padded if short. No practical limit; all three **ignore `n_ctx`** (it cannot lower a limit they don't have). whisper and parakeet never error on length. voxtral_realtime has one exception — its absolute `dec_max_position` cap (~2.9 h, see below): a clip past it returns `INPUT_TOO_LONG` (one-shot/batch), and a stream that reaches it flags `was_truncated`. |

Whisper slices audio into 30 s windows with prev-context stitching; parakeet's
conformer is effectively unbounded (the encoder positional table is recomputed
per run, not a fixed wall). This holds for every parakeet variant, including the
cache-aware streaming RNN-T member (`nemotron-speech-streaming-en-0.6b`): the
RNN-T transducer has no decoder context window, and streaming carries
constant-memory caches (`cache_last_channel` / `cache_last_time` + the decoder
LSTM state) rather than a growing KV, so it is unbounded in both offline and
streaming modes. These families do not need and do not have a length gate.

### 2. Hard context cap — reject up front

| Families | Limit source | Behavior |
| --- | --- | --- |
| qwen3_asr, canary_qwen, funasr_nano, granite, granite_nar, voxtral, cohere, canary | decoder context window (`dec_max_position_embeddings` / `dec_max_seq`), or the encoder positional table (`enc_pos_emb_max_len`, for cohere/canary) — all from GGUF | KV cache grows to fit, clamped to the model's true max. Over-length input is **rejected before the decode** (or before the encoder, where the encoder table is the binding limit) with `TRANSCRIBE_ERR_INPUT_TOO_LONG`. |

These families wrap an LLM-style decoder whose context window
(`audio_tokens + prompt + generation`) is the binding constraint. The number of
tokens a clip consumes is a deterministic function of its sample count
(`n_samples → mel frames → fixed subsampling → audio tokens`), so the library
computes the prefill size *before* running the encoder and rejects an
over-length clip immediately — the caller never pays for a compute pass that
cannot fit. The rejection goes through the log callback, not raw stderr.

The one case that cannot be predicted up front is the transcript itself running
long enough to exhaust the remaining budget mid-decode (rare — the output would
have to be very large for the audio length). There, the run returns the hard
status `TRANSCRIBE_ERR_OUTPUT_TRUNCATED` while keeping the partial transcript
readable (exactly like an aborted run); `transcribe_was_truncated(session)` is
also set, and a `WARN` is logged. A truncated transcript is never returned as
`TRANSCRIBE_OK` — a caller cannot mistake it for complete — and the partial
output is never discarded. In `transcribe_run_batch` this is a per-utterance
status (the whole-batch call still returns `TRANSCRIBE_OK`).

### 3. Soft window — warn and proceed

| Families | Window | Behavior |
| --- | --- | --- |
| gigaam (~25 s), sensevoice (~30 s), medasr (~400 s), moonshine (output-bound, ~48 s), moonshine_streaming (output-bound, ~17 min) | training / positional window | Any length is accepted; past the window the library emits a `WARN` (degraded accuracy is possible) and proceeds. `max_audio_ms` reports the window as advisory. |

These families have no hard architectural wall but were trained on a bounded
window; beyond it, accuracy degrades rather than failing. The library does not
reject (a caller may knowingly accept degradation) but it does warn so the
degradation is never silent. Upstream `gigaam` rejects over-length audio
outright; transcribe.cpp deliberately warns-and-proceeds instead so callers
keep control.

Moonshine is the honest edge case in this bucket: its cap is on *output*
(`max_length = 194` decode tokens, ≈ 48 s), not input, so it cannot be gated on
audio length — a dense short clip can hit it too. It is reported via
`transcribe_was_truncated()` and a `WARN` (and, offline, the hard
`TRANSCRIBE_ERR_OUTPUT_TRUNCATED` status) when the cap is reached.
`moonshine_streaming` has the same output-bound shape with a much larger window
(`dec_max_position_embeddings = 4096`, ≈ 17 min); because it also streams, its
truncation follows the streaming rule below — `stream_finalize` still returns
`TRANSCRIBE_OK` and the truncation surfaces only through
`transcribe_was_truncated()`.

## Context sizing and the `n_ctx` knob

For bucket-2 families the decoder KV cache **grows to fit the actual input**,
bucketed to avoid reallocation churn, and is **clamped to the model's true
maximum** read from GGUF metadata. This replaced earlier per-family hardcoded
walls (e.g. a fixed 2048 that ignored a much larger real context window) so the
GGUF metadata is the single source of truth for the ceiling.

`transcribe_session_params::n_ctx` lets a caller **lower** the ceiling to bound
memory:

- `0` (default): use the model's true maximum from GGUF.
- `> 0`: cap the context at this many tokens. Values above the model maximum are
  clamped down to it — the knob can only narrow, never extend past what the
  model supports.

For decoder-context-bound families, lowering `n_ctx` lowers the effective audio
limit and the upfront gate enforces the lowered value. Note that `max_audio_ms`
is a *model-level* capability, queried before any session exists, so it always
reports the model's default-context ceiling (`n_ctx == 0`); it is not re-derived
for a session that narrows `n_ctx`. A session that lowers `n_ctx` may therefore
reject audio shorter than the advertised `max_audio_ms`.

Encoder-bound families are different. For cohere and canary, the input-audio
limit is the encoder positional table, while `n_ctx` only bounds the decoder
self-KV / output budget. In those families `transcribe_session_get_limits()`
reports a smaller `effective_n_ctx` and `max_kv_bytes` when `n_ctx` is lowered,
but `effective_max_audio_ms` stays pinned to the encoder input bound.

**Chunked / unbounded families (bucket 1) ignore `n_ctx` entirely.** whisper,
parakeet, and voxtral_realtime have no lowerable context ceiling, so a non-zero
`n_ctx` is a documented no-op and `transcribe_session_get_limits()` keeps
reporting them unbounded. voxtral_realtime is the subtle case: it *does* have an
absolute decoder position cap (`dec_max_position`, ~2.9 h of audio), but that is
the model's true RoPE wall — not a memory ceiling a caller can lower — so
`n_ctx` does not narrow it (its decoder KV is a constant-memory sliding ring;
there is nothing for `n_ctx` to bound). A clip past the absolute cap is rejected
with `TRANSCRIBE_ERR_INPUT_TOO_LONG` (one-shot and batch) or surfaced via
`transcribe_was_truncated()` (streaming) regardless of `n_ctx`.

## What a caller observes — summary

| Situation | Status | Log | Result |
| --- | --- | --- | --- |
| Input within limit | `TRANSCRIBE_OK` | — | full transcript |
| Over-length, hard-cap family | `TRANSCRIBE_ERR_INPUT_TOO_LONG` | `ERROR` via callback | no transcript (rejected before the decode) |
| Generation ran long mid-decode | `TRANSCRIBE_ERR_OUTPUT_TRUNCATED` | `WARN` via callback | partial transcript readable; `transcribe_was_truncated() == true` |
| Over-window, soft-window family | `TRANSCRIBE_OK` | `WARN` via callback | full transcript (accuracy may be degraded) |
| Chunked / unbounded family | `TRANSCRIBE_OK` | — | full transcript |
| Cache/graph allocation failed | `TRANSCRIBE_ERR_OOM` | `ERROR` via callback | no transcript (no silent context shrink) |

In `transcribe_run_batch`, `INPUT_TOO_LONG` and `OUTPUT_TRUNCATED` are
per-utterance statuses (`transcribe_batch_status(session, i)`); the whole-batch
call returns `TRANSCRIBE_OK`.

`transcribe_was_truncated(session)` is reset at the top of every
`transcribe_run`, `transcribe_run_batch`, and `transcribe_stream_begin` (the same
lifecycle as `transcribe_was_aborted`).

## Streaming is the exception

`TRANSCRIBE_ERR_OUTPUT_TRUNCATED` is an **offline-only** status
(`transcribe_run` / `transcribe_run_batch`). An active stream is incremental
and has its own terminal-state machine (`transcribe_stream_*`,
IDLE/ACTIVE/FINISHED/FAILED), and `stream_feed` / `stream_finalize` return the
status of *that step*, not a verdict on the whole transcript. So when a
streaming decode reaches its context cap (e.g. `voxtral_realtime` at its
absolute position limit — hours of continuous audio, or `moonshine_streaming`
at its output window), the stream does **not** fail and `stream_finalize`
returns `TRANSCRIBE_OK`; the truncation is surfaced through
`transcribe_was_truncated(session)` and a `WARN`. This is deliberate:
forcing a stream into a failed terminal state on truncation would discard the
committed text the caller has been consuming. A streaming caller that needs to
detect truncation should check `transcribe_was_truncated()` after finalize.

A streaming family with no context cap at all — parakeet's cache-aware RNN-T
variant (`nemotron-speech-streaming-en-0.6b`) — never reaches this case: its
caches are constant-memory and the RNN-T decoder has no position limit, so a
stream of any length runs unbounded and `was_truncated` stays false.

## Design notes (for maintainers)

- The upfront gate and `max_audio_ms` share a shape for decoder-context-bound
  families but differ in precision:
  `max_audio_ms ≈ (ceiling − representative_prompt − generation_reserve) / tokens_per_ms`,
  where `ceiling` is the GGUF max. The gate uses the *exact* per-call prompt, so
  `max_audio_ms` is advisory/slightly-conservative, not the exact bound. The
  per-model rate constants live in `transcribe_model::LimitsBasis`, which
  `transcribe_session_get_limits()` reads to recompute the effective limit at a
  lowered `n_ctx`. For encoder-bound families, the session query keeps the audio
  bound from `max_audio_ms` while still reporting the lowered decoder KV budget.
- `n_ctx` is a session context/KV cap, never a way to extend the model:
  `effective_n_ctx = n_ctx == 0 ? model_max : min(n_ctx, model_max)`.
- Allocation failures (KV cache, compute graph) return `TRANSCRIBE_ERR_OOM`
  with a logged diagnostic, never `GGUF`/`BACKEND`, and the library does not
  silently shrink the context and retry.
- Length/context messages must go through the shared `transcribe_log` helpers,
  never `fprintf(stderr)` — a consumer that installed a log sink must see them.
- New families pick a bucket at port time and fill `max_audio_ms` accordingly
  (`0` for chunked/unbounded). The model-card template carries an "Input limits"
  block that must be populated with the real number.
