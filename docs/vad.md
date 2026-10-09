# Voice activity detection

A VAD model (`transcribe_model_roles()` has `TRANSCRIBE_ROLE_VAD`) scores
fixed-size frames of 16 kHz audio with a speech probability and turns them
into speech segments. API: `include/transcribe/vad.h`. The shipped model is
[Silero VAD v6.2](models/silero-vad-v6.2.md): 512-sample (32 ms) frames.
`transcribe_model_load_file` also accepts whisper.cpp's
`ggml-silero-v5.1.2.bin` and `ggml-silero-v6.2.0.bin`.

## Offline

```c
#include "transcribe/vad.h"

struct transcribe_model * m = NULL;
transcribe_model_load_file("silero-vad-v6.2-F32.gguf", NULL, &m);

struct transcribe_vad_session * vad = NULL;
transcribe_vad_session_init(m, NULL, &vad);

struct transcribe_vad_params vp;
transcribe_vad_params_init(&vp);
vp.threshold = 0.6;                              /* optional */
transcribe_vad_run(vad, pcm, n_samples, &vp);    /* 16 kHz mono float32 */

struct transcribe_vad_result r;
transcribe_vad_result_init(&r);
transcribe_vad_get_result(vad, &r);
for (int i = 0; i < r.n_segments; ++i) {
    struct transcribe_vad_segment s;
    transcribe_vad_segment_init(&s);
    transcribe_vad_get_segment(vad, i, &s);      /* [s.start_sample, s.end_sample) */
}
const float * p = transcribe_vad_probs(vad);     /* r.n_probs per-frame probabilities */

transcribe_vad_session_free(vad);
transcribe_model_free(m);
```

From the CLI: `transcribe-cli -m silero-vad-v6.2-F32.gguf --vad-threshold 0.6 clip.wav`.

`transcribe_vad_run` reproduces Silero's `get_speech_timestamps`: with the same
parameters the segments match the reference sample for sample. The parameters,
with the reference's defaults:

| field | meaning | default |
|---|---|---|
| `threshold` | a frame with `p >= threshold` starts speech | 0.5 |
| `neg_threshold` | inside speech, `p < neg_threshold` is silence (< 0 = default) | `max(threshold - 0.15, 0.01)` |
| `min_speech_ms` | shorter segments are dropped | 250 |
| `min_silence_ms` | silence this long ends a segment | 100 |
| `speech_pad_ms` | padding added to both ends of a segment | 30 |
| `max_speech_ms` | longer segments are split at a pause (0 = no limit) | 0 |

## Streaming

`transcribe_vad_stream_feed` takes audio in pieces of any size and returns the
probabilities of the frames each call completes; `first_frame` is the stream
position of the first one. `transcribe_vad_stream_flush` scores the buffered
remainder zero padded and ends the stream; `transcribe_vad_stream_reset` ends
it without scoring. On CPU, stream probabilities are bit-identical to an
offline run over the same audio.

```c
for (;;) {
    int n = read_microphone(buf, 320);           /* 20 ms */
    transcribe_vad_stream_feed(vad, buf, n);
    transcribe_vad_get_result(vad, &r);
    const float * p = transcribe_vad_probs(vad); /* frames r.first_frame .. + r.n_probs */
}
transcribe_vad_stream_flush(vad);
```

To turn stream probabilities into live START/END events, feed them to a
`transcribe_vad_iterator` (Silero's `VADIterator`). It owns no model and takes
`threshold`, `neg_threshold` (default `threshold - 0.15`, no 0.01 floor),
`min_silence_ms` and `speech_pad_ms`, with the same meanings as above:

```c
struct transcribe_vad_iterator * it = NULL;
transcribe_vad_iterator_init(512, NULL, &it);    /* info.frame_samples */

/* after each stream_feed / stream_flush: */
transcribe_vad_iterator_feed(it, transcribe_vad_probs(vad), r.n_probs);
struct transcribe_vad_iterator_result ir;
transcribe_vad_iterator_result_init(&ir);
transcribe_vad_iterator_get_result(it, &ir);
for (int i = 0; i < ir.n_events; ++i) {
    struct transcribe_vad_event e;
    transcribe_vad_event_init(&e);
    transcribe_vad_iterator_get_event(it, i, &e); /* e.type START/END, e.sample */
}
transcribe_vad_iterator_free(it);
```

As upstream, the iterator does not emit END at end of input: close any open
speech yourself at the real audio length.

## Backend

`TRANSCRIBE_BACKEND_AUTO` resolves to the CPU for VAD models; a GPU backend can
be requested explicitly but is less accurate (see the model page). Threading
and lifetime rules are shared with the other roles: [roles.md](roles.md).
