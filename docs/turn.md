# End-of-turn detection: API design (DRAFT, not locked)

Status: **draft**. The owner locks the API; until then this file is the single
source of truth for every session working on turn detection. Only the design
session edits the API below; other sessions propose changes to it (see
[Change log](#change-log)).

Branch: `turn-detection`. Models: Smart Turn v3.2 (Pipecat) and SAVA (Inworld).
Nothing is released until both models work and the owner locks the API.

## Decisions (owner, settled)

| # | Decision |
|---|---|
| 1 | A new role, `TRANSCRIBE_ROLE_TURN`, in a new header `include/transcribe/turn.h`. SAVA's per-frame turn scores come out of a TURN stream, not the VAD stream. |
| 2 | VAD docs allow a model to judge a frame late (lookahead). Silero is unchanged. |
| 3 | Whole recordings: the library makes decisions (copying upstream rules exactly), as `transcribe_vad_run` does for Silero. Live audio: scores only; the caller decides (option C1). |
| 4 | Each model's recommended cut-off is the default; callers can override it. Results give the score and a yes/no. |
| 5 | Out of scope: turn models that read text, and two-channel models. Text context would come later as a new `transcribe_turn_params` field. |
| 6 | VAD's default threshold comes from the model (Silero keeps 0.5). |
| 7 | Smart Turn: v3.2 only. SAVA ships if it passes our own tests. |
| 8 | Bindings (Rust, TypeScript) are done after the API is locked. |
| 9 | onnx / onnxruntime are allowed in `scripts/envs/` (reference tooling only). |

## Settled details

- Several live users on one streaming model: `transcribe_turn_stream_feed_batch`
  feeds many sessions in one compute; each session ends exactly as after its
  own feed (owner decision, from `tc-turn-perf/reports/perf/turn-batching.md`:
  for 8 SAVA users a batched 40 ms tick takes ~15 ms on one core, while
  serial feeds miss the 40 ms budget). No batch call for clip models: batching
  8 Smart Turn clips saves only 4-7% of CPU.
- SAVA keeps stream == run bit for bit (the 4-column padding stays; no ggml
  patch). Ship F32, F16 and Q8_0 for both models.

- SAVA's recommended cut-off is 0.9 (Inworld's policy), also for a whole clip.
  Inworld's clip benchmark reports accuracy at 0.5; the AUC is threshold-free.
- `ended` is `p >= threshold`, like VAD. Pipecat uses `p > 0.5`.
- Turn segmentation exposes Inworld's whole policy, like `transcribe_vad_params`
  exposes Silero's.

## How it fits together

| | Smart Turn v3.2 | SAVA |
|---|---|---|
| roles | TURN | VAD, TURN |
| `transcribe_turn_run` (a clip) | score of the last 8 s; `p`, `ended` | runs the clip from a fresh state; `p`, `ended` of the last frame; per-frame scores; turns found in the clip |
| `transcribe_turn_stream_*` | NOT_IMPLEMENTED (`frame_samples` 0) | per-frame turn scores, and speech scores from the same pass |
| `transcribe_vad_*` | UNSUPPORTED_ROLE | speech scores only (`lookahead_frames` 3, `frame_samples` 640) |
| cut-off (`threshold` default) | 0.5 | 0.9 |

Typical live use:

- Silero + Smart Turn: VAD stream to hear speech and pauses; on each pause,
  `transcribe_turn_run` on the current turn's audio; `ended` says whether to answer.
- SAVA: one TURN stream; the caller applies its own rule to the speech and turn
  tracks (Inworld's reference rule is in this doc and in the CLI).

## Usage (clip)

```c
#include "transcribe/turn.h"

struct transcribe_model * m = NULL;
transcribe_model_load_file("smart-turn-v3.2-F32.gguf", NULL, &m);
struct transcribe_turn_session * t = NULL;
transcribe_turn_session_init(m, NULL, &t);

/* On each pause the VAD hears: the current turn's audio, 16 kHz mono f32. */
transcribe_turn_run(t, turn_pcm, n_samples, NULL);
struct transcribe_turn_result r;
transcribe_turn_result_init(&r);
transcribe_turn_get_result(t, &r);
if (r.ended) { /* answer */ }               /* r.p: the probability */

transcribe_turn_session_free(t);
transcribe_model_free(m);
```

## Status and testing

| Area | Tests | State |
|---|---|---|
| Role plumbing | `tests/turn_dispatch_unit.cpp`: fake clip and streaming archs (crop, threshold, NOT_IMPLEMENTED stream, chunk-size independence, speech-track alignment, flush, reset, abort, invalid input, segmenter worked by hand); `role_resolve_unit`, `api_smoke` (ABI ids) | done |
| VAD lookahead | `tests/vad_lookahead_unit.cpp`: fake lookahead model (judged frames, flush, run, model threshold); Silero output unchanged on `samples/fleurs-en.wav` | done |
| Smart Turn | `validate.py all --family smart_turn` (16 tensors x 8 cases, `p` within 1e-4, decision); `smart_turn_smoke` (toy fixture); `smart_turn_real_smoke`; the full Pipecat test set through `scripts/turn/eval_cpp.py`: 31,527 clips, decisions identical to the fp32 ONNX, max abs dp 2.4e-5 | done (F32) |
| SAVA | `validate.py all --family sava` (32 tensors x 8 cases incl. a 306 s stream; `p`, decision and turn segments identical to `sava.segment`); `sava_smoke` (stream == run bit for bit, any chunking); `sava_real_smoke` | done (F32) |
| Service-shaped loop | `examples/turn_live` (`transcribe-turn-live`): 20 ms chunks, `speech_start` / `turn` events; Silero + Smart Turn (Pipecat's rule) and SAVA alone (Inworld's rule on the stream tracks). With SAVA the live events equal `transcribe_turn_run`'s segments on `samples/dots.wav` | done |
| Bindings | Rust, TypeScript, Python, Swift | after the API is locked |

## Header

The header is `include/transcribe/turn.h` on this branch; it is the draft.
Changes to it are recorded in the change log below.

## Draft changes to existing headers

`include/transcribe.h` (append-only enums):

```c
    TRANSCRIBE_ROLE_TURN = 1u << 4,              /* transcribe_role */
    TRANSCRIBE_EXT_SLOT_TURN_RUN = 3,            /* transcribe_ext_slot: transcribe_turn_params::family */
```

`include/transcribe/vad.h`:

```c
struct transcribe_vad_info {
    uint64_t struct_size;
    int32_t  sample_rate;
    int32_t  frame_samples;    /* Silero 512, SAVA 640 */
    /* NEW. Frames a score lags its audio: frame t is judged once frame
     * t + lookahead_frames has been fed. Silero 0, SAVA 3. */
    int32_t  lookahead_frames;
    /* NEW. The model's recommended speech threshold: the default of
     * transcribe_vad_params::threshold. Silero 0.5. */
    double   threshold;
};

struct transcribe_vad_params {
    ...
    /* CHANGED. Default: -1, which uses transcribe_vad_info::threshold
     * (Silero: 0.5, as before). Any negative value uses the model's. */
    double   threshold;
    ...
};
```

Wording changes in `vad.h` and `docs/vad.md` (behaviour for Silero unchanged):

- feed "returns a probability for each completed frame" becomes "for each
  frame the model has judged; with lookahead, frame t is judged once frame
  t + lookahead_frames is complete".
- flush "returns one probability, or none" becomes "returns the probabilities
  of every frame not yet judged".
- `first_frame` keeps its meaning: the first returned (judged) frame.
- `transcribe_vad_params_init` sets `threshold` to -1 instead of 0.5. A
  caller that reads the field after init sees -1; the effective value for
  Silero is still 0.5.

Docs: `docs/roles.md` gets a TURN row; `docs/extension-kinds.md` gets the
`TURN_RUN` slot; `docs/models/smart-turn-v3.2.md`, `docs/models/sava.md`.

## Segment details (from Inworld's reference)

- Turn segments reproduce `sava.segment` exactly, so they can **overlap**: a
  turn closed by the model or by silence ends `lag + 1` frames after the frame
  judged silent, and the next turn starts `pre_roll_ms` before its first
  speech with no clamp to the previous end. They are ordered by start.
- Turns the reference reports as dropped (too little speech) are not returned,
  as VAD drops short segments.
- The last `lag` frames of a run are judged with the turn scores of the flush's
  silent steps, as in the reference. A stream's flush does not return those
  scores (the turn track covers audio frames only), so a caller's own live rule
  sees a turn still open at the end of a stream and closes it itself.
- Stream and run share one code path in C++ and agree bit for bit on CPU. The
  reference's stream and run differ by up to about 2e-6 (chunked scan vs
  recurrence), so parity with the reference is a tolerance, not equality.

## Semantics to pin down in tests

- Smart Turn `run`: left zero-pad to 8 s, or keep the last 8 s; waveform
  normalised over the whole 8 s window including the padding; Whisper 80-bin
  log-mel, 800 frames; `p` matches onnxruntime on the fp32 ONNX.
- Audio before the last 8 s does not change Smart Turn's `p`.
- SAVA `run` equals feed of the whole clip plus flush, from a fresh state;
  `p` is the last real frame's turn score; the turn scores of flush's silent
  padding steps are not returned.
- SAVA streams are independent of how the audio is chunked, and equal `run`
  on CPU.
- SAVA speech track: `speech_first_frame + n_speech_probs` never exceeds
  `first_frame + n_probs - speech_lookahead_frames` before flush.
- SAVA segments reproduce Inworld's `sava.segment` frame for frame, mapped to
  samples (`frame * 640`, end clipped to the input).
- SAVA through `transcribe_vad_*`: the same speech scores as the TURN speech
  track; `vad_run` segments use Silero's algorithm on 40 ms frames.

## Change log

- Draft 1: initial draft from owner decisions 1-9.
- Draft 2: open questions settled (0.9, `>=`, full policy); header moved to `include/transcribe/turn.h`.
- Draft 3: VAD `threshold` / `lookahead_frames` implemented; segment details from the SAVA intake recorded.
- Draft 4: `transcribe_turn_stream_feed_batch` added (owner decision); `docs/roles.md` notes that a batched call is one compute.
