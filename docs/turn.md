# End-of-turn detection

A voice assistant has to know when the user has finished talking, so it can
answer. A pause alone is a poor signal: people pause in the middle of a
sentence. A turn model listens to how the speaker sounds and returns a
probability that the turn is over: near 0, they will keep talking; near 1,
their turn has ended.

A turn model has `TRANSCRIBE_ROLE_TURN` in `transcribe_model_roles()`. API:
`include/transcribe/turn.h`. Audio is 16 kHz mono float32.

| | [Smart Turn v3.2](models/smart-turn-v3.2.md) (Pipecat) | [SAVA](models/sava.md) (Inworld) |
|---|---|---|
| How it listens | one clip at a time (the last 8 s) | a live stream of 40 ms frames |
| `transcribe_turn_run` | yes | yes |
| `transcribe_turn_stream_*` | no (`frame_samples` is 0) | yes |
| Also scores speech | no | yes (`has_speech`) |
| Default cut-off (`threshold`) | 0.5 | 0.9 |
| Other roles | none | VAD |

## Score a clip

Use a VAD (for example [Silero](vad.md)) to hear when the user pauses. At each
pause, score the audio of the current turn, from where the user started
talking:

```c
#include "transcribe/turn.h"

struct transcribe_model * m = NULL;
transcribe_model_load_file("smart-turn-v3.2-F32.gguf", NULL, &m);

struct transcribe_turn_session * t = NULL;
transcribe_turn_session_init(m, NULL, &t);

transcribe_turn_run(t, turn_pcm, n_samples, NULL);   /* NULL: default cut-off */

struct transcribe_turn_result r;
transcribe_turn_result_init(&r);
transcribe_turn_get_result(t, &r);
if (r.ended) {
    /* the user is done: answer */
}                                                    /* r.p is the probability */

transcribe_turn_session_free(t);
transcribe_model_free(m);
```

`r.ended` is `r.p >= threshold`. To use another cut-off, pass params:

```c
struct transcribe_turn_params tp;
transcribe_turn_params_init(&tp);
tp.threshold = 0.7;
transcribe_turn_run(t, turn_pcm, n_samples, &tp);
```

Smart Turn looks at the last 8 s of the audio (`transcribe_turn_info::max_audio_ms`);
longer audio is fine. On SAVA, `transcribe_turn_run` scores the clip from a
fresh state; `p` is the turn probability of its last frame, and the result also
holds every frame's turn and speech probabilities.

From the CLI: `transcribe-cli -m smart-turn-v3.2-F32.gguf turn.wav` prints
`turn: p=... ended=... threshold=...`.

## Stream (SAVA)

Feed audio as it arrives, in pieces of any size. Each feed returns:

- a turn probability for every frame it completed: frames `first_frame` to
  `first_frame + n_probs - 1`;
- a speech probability for every frame the model judged: frames
  `speech_first_frame` to `speech_first_frame + n_speech_probs - 1`.

SAVA judges speech 3 frames (120 ms) late, so the speech probabilities trail
the turn probabilities. The speech probability of frame `t` arrives together
with the turn probability of frame `t + 3`; that pair is what Inworld's rule
looks at. You can read the lag from any feed that returned speech
probabilities: `(first_frame + n_probs) - (speech_first_frame + n_speech_probs)`.

```c
struct transcribe_turn_session * t = NULL;
transcribe_turn_session_init(m, NULL, &t);             /* a SAVA model */

for (;;) {
    int n = read_microphone(buf, 320);                 /* 20 ms */
    transcribe_turn_stream_feed(t, buf, n);
    transcribe_turn_get_result(t, &r);
    const float * turn   = transcribe_turn_probs(t);        /* r.n_probs values */
    const float * speech = transcribe_turn_speech_probs(t); /* r.n_speech_probs values */
    /* your rule: see below */
}
transcribe_turn_stream_flush(t);   /* scores the rest; the next feed starts at frame 0 */
```

The library returns probabilities only; your code decides when a turn ends.
Inworld's rule:

1. A turn opens on the first frame with speech probability `>= 0.5` (keep the
   800 ms of audio before it).
2. While it is open, each silent frame looks at its paired turn probability:
   `>= 0.9` ends the turn.
3. 1.6 s of silence ends it anyway.

`examples/turn_live` runs this rule (and Silero + Smart Turn, Pipecat's
setup) on a WAV fed in 20 ms pieces:

```bash
transcribe-turn-live --turn sava-Q8_0.gguf in.wav
transcribe-turn-live --vad silero-vad-v6.2-F32.gguf --turn smart-turn-v3.2-F32.gguf in.wav
```

A stream and `transcribe_turn_run` over the same audio give the same
probabilities, bit for bit, on the CPU.

### Many users on one model

A server with several live users can feed them all in one call:

```c
struct transcribe_turn_session * sessions[8];   /* one per user, same model */
const float * pcm[8];
int           n_samples[8];
transcribe_turn_stream_feed_batch(sessions, pcm, n_samples, 8);
/* read each user's result from sessions[i], as after its own feed */
```

The model reads its weights once for all users, which is much faster than
eight feeds: on an Apple M1, 8 SAVA users take about 7-10 ms per 40 ms of
audio. Each user's result is exactly what its own feed would give.

### Speech only

SAVA also serves the VAD role: `transcribe_vad_*` returns the same speech
probabilities (see [vad.md](vad.md)). Use the turn session if you want both;
it computes them in one pass.

## Threads and backend

`TRANSCRIBE_BACKEND_AUTO` picks the CPU for both models; it is the validated
path. The default thread count is the library's (the usable CPUs, at most 8).
A SAVA frame is a small amount of work: on a machine busy with other jobs, 1
thread per session (`transcribe_turn_session_params::n_threads`) or the batched
feed is faster and steadier than many threads.

Threading and lifetime rules are shared with the other roles:
[roles.md](roles.md). The bindings (Rust, Python, TypeScript, Swift) expose the
same API as `TurnSession`: [bindings.md](bindings.md).
