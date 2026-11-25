# End-of-turn detection

Status: **draft 6, not locked.** The owner locks the API. Until then this file
records the design, every decision and the change log.

A voice assistant has to know when the user has finished talking, so it can
answer. Waiting for a pause is not enough: people pause in the middle of a
sentence. A turn model listens to how the speaker sounds and gives a
probability that the turn is over.

API: `include/transcribe/turn.h` (role `TRANSCRIBE_ROLE_TURN`). Models:
Smart Turn v3.2 (Pipecat) and SAVA (Inworld). Branch: `turn-detection`.

## The two models

| | Smart Turn v3.2 | SAVA |
|---|---|---|
| Roles | TURN | VAD and TURN |
| How it listens | one clip at a time | a live stream, 40 ms frames |
| `transcribe_turn_run` | scores the last 8 s of the clip | runs the whole clip; `p` is the last frame's score; also every frame's turn and speech scores |
| `transcribe_turn_stream_*` | NOT_IMPLEMENTED (`frame_samples` is 0) | a turn score per frame, and a speech score per frame from the same pass |
| `transcribe_vad_*` | UNSUPPORTED_ROLE | speech scores only |
| Default cut-off (`threshold`) | 0.5 | 0.9 |

## How to use it

### Clip model (Silero VAD + Smart Turn)

Run a VAD stream to hear when the user speaks and pauses. At each pause, score
the audio of the current turn:

```c
#include "transcribe/turn.h"

struct transcribe_model * m = NULL;
transcribe_model_load_file("smart-turn-v3.2-F32.gguf", NULL, &m);
struct transcribe_turn_session * t = NULL;
transcribe_turn_session_init(m, NULL, &t);

/* At a pause: the audio since the user started talking, 16 kHz mono. */
transcribe_turn_run(t, turn_pcm, n_samples, NULL);
struct transcribe_turn_result r;
transcribe_turn_result_init(&r);
transcribe_turn_get_result(t, &r);
if (r.ended) { /* answer */ }   /* r.p is the probability */

transcribe_turn_session_free(t);
transcribe_model_free(m);
```

### Streaming model (SAVA alone)

Feed audio as it arrives (`transcribe_turn_stream_feed`). Each call returns a
turn score for every new frame and a speech score for every frame the model
has judged. SAVA judges speech 3 frames (120 ms) late, so the speech scores
trail the turn scores; `first_frame` and `speech_first_frame` say which frames
each call returns. Your code decides when the turn is over, for example
Inworld's rule: a turn opens at the first speech frame; while it is open, a
silent frame ends it if its turn score reaches the cut-off, or if the pause has
lasted 1.6 s. `examples/turn_live` shows both setups with 20 ms chunks.

For many users on one model, `transcribe_turn_stream_feed_batch` processes
every user's audio in one call, much faster than one feed per user.

## Decisions (owner)

| # | Decision |
|---|---|
| 1 | A new role, `TRANSCRIBE_ROLE_TURN`, in a new header `include/transcribe/turn.h`. SAVA's turn scores come from a TURN stream, not the VAD stream. |
| 2 | VAD docs allow a model to judge a frame late (lookahead). Silero is unchanged. `vad.h` gets no new fields, only reworded comments. |
| 3 | For a clip, the library gives the score and a yes/no (`p`, `ended`). For live audio it gives scores only; the caller decides. The library does not cut recordings into turns (draft 6). |
| 4 | Each model's recommended cut-off is the default; callers can override it. |
| 5 | Out of scope: turn models that read text, and two-channel models. |
| 6 | Smart Turn: v3.2 only. SAVA ships because it passes our tests. |
| 7 | Bindings (Rust, Python, TypeScript, Swift) come after the API is locked. |
| 8 | onnx / onnxruntime are allowed in `scripts/envs/` (reference tooling only). |
| 9 | Keep the API small: model-specific settings stay out of the core until a second model needs them. |

## Settled details

- `ended` is `p >= threshold`, like VAD. (Pipecat uses `p > 0.5`.)
- SAVA's cut-off is 0.9 (Inworld's), also for a clip. Inworld's clip benchmark
  reports accuracy at 0.5; the AUC does not depend on the cut-off.
- `transcribe_turn_stream_feed_batch`: each session ends exactly as after its
  own feed. For 8 SAVA users a batched 40 ms tick takes ~15 ms on one core;
  separate feeds miss the 40 ms budget. No batch call for clip models:
  batching 8 Smart Turn clips saves only 4-7% of CPU.
- SAVA: a stream and a run of the same audio give the same scores, bit for
  bit, on the CPU. No ggml patches for speed. F32, F16 and Q8_0 ship for both
  models; F16 is slower than F32 on ARM CPUs.
- The SAVA GGUF still records Inworld's whole turn rule (`stt.sava.policy.*`);
  the library reads only the cut-off.

## Tests

| Area | Tests |
|---|---|
| Role plumbing | `tests/turn_dispatch_unit.cpp`: fake clip and streaming models (cropping, threshold, NOT_IMPLEMENTED stream, any chunk size, speech-track alignment, flush, reset, abort, invalid input, batch); `role_resolve_unit`; `api_smoke` (ABI ids) |
| VAD lookahead | `tests/vad_lookahead_unit.cpp`; Silero output unchanged on `samples/fleurs-en.wav` |
| Smart Turn | `validate.py all --family smart_turn` (16 tensors x 8 cases, `p` within 1e-4, same decision); `smart_turn_smoke`; `smart_turn_real_smoke`; the Pipecat test set (31,527 clips) through `scripts/turn/eval_cpp.py`: decisions identical to the fp32 ONNX |
| SAVA | `validate.py all --family sava` (32 tensors x 8 cases incl. a 306 s stream, among them the per-frame turn and speech scores; `p` within 1e-4, same decision); `sava_smoke` (stream == run bit for bit, any chunking, batch); `sava_real_smoke` |
| Live loop | `examples/turn_live`: Silero + Smart Turn (Pipecat's rule) and SAVA alone (Inworld's rule on the stream) |
| Bindings | after the API is locked |

## Changes to existing headers

- `include/transcribe.h`: `TRANSCRIBE_ROLE_TURN = 1u << 4` and the ABI ids
  `TRANSCRIBE_ABI_TURN_*` (append-only enums).
- `include/transcribe/vad.h`: comments only. Feed returns "a probability for
  each frame the model judged"; flush returns "every frame not yet judged".
  `first_frame` is the first returned frame, as before.
- Docs: `docs/roles.md` (TURN row), `docs/models/smart-turn-v3.2.md`,
  `docs/models/sava.md`.

## Change log

- Draft 1: initial draft from owner decisions.
- Draft 2: cut-off 0.9 for SAVA, `>=`; header moved to `include/transcribe/turn.h`.
- Draft 3: VAD `threshold` / `lookahead_frames` added; SAVA turn segments recorded.
- Draft 4: `transcribe_turn_stream_feed_batch` added.
- Draft 5 (owner review): `vad.h` drops `lookahead_frames` and `threshold`;
  SAVA's segment settings move to a SAVA extension (`include/transcribe/sava.h`);
  core params `{struct_size, threshold, family}`.
- Draft 6 (owner review): turn segments removed (the segment struct, the
  end-reason enum, `transcribe_turn_get_segment`, `n_segments`), and with them
  `include/transcribe/sava.h`, the `SVTN` kind, `TRANSCRIBE_EXT_SLOT_TURN_RUN`
  and `transcribe_turn_params::family`. `transcribe_turn_info` drops
  `speech_lookahead_frames` (`speech_first_frame` gives it). `max_audio_ms`
  and `sample_rate` stay, as in `langid.h`. Header comments rewritten in the
  style of `vad.h`.
