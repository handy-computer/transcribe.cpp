# Voice activity detection

A VAD model (`transcribe_model_roles()` has `TRANSCRIBE_ROLE_VAD`) scores
fixed-size frames of 16 kHz audio with a speech probability and turns them
into speech segments. API: `include/transcribe/vad.h`. The shipped model is
[Silero VAD v6.2](models/silero-vad-v6.2.md): 512-sample (32 ms) frames,
1.2 MB.

## Offline

```c
#include "transcribe/vad.h"

struct transcribe_model * m = NULL;
transcribe_model_load_file("silero-vad-v6.2-F32.gguf", NULL, &m);

struct transcribe_vad_session * vad = NULL;
transcribe_vad_session_init(m, NULL, &vad);

struct transcribe_vad_params vp;
transcribe_vad_params_init(&vp);                 /* Silero's defaults */
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

From the CLI: `transcribe-cli -m silero-vad-v6.2-F32.gguf clip.wav` prints
one `segment:` line per segment (`--vad-threshold P` overrides the threshold).

`transcribe_vad_run` reproduces Silero's `get_speech_timestamps(audio, model,
...)`: the clip is scored from a fresh model state, the last frame zero
padded, and the probabilities are segmented by a line-for-line port of
`get_speech_timestamps_from_probs`. With the same parameters the segments are
the reference's, sample for sample (see the model page for the evidence).

`transcribe_vad_params` mirrors the reference's keyword arguments:

| field | reference argument | default |
|---|---|---|
| `threshold` | `threshold` | 0.5 |
| `neg_threshold` (< 0 = default) | `neg_threshold` | `max(threshold - 0.15, 0.01)` |
| `min_speech_ms` | `min_speech_duration_ms` | 250 |
| `min_silence_ms` | `min_silence_duration_ms` | 100 |
| `speech_pad_ms` | `speech_pad_ms` | 30 |
| `max_speech_ms` (0 = no limit) | `max_speech_duration_s` | no limit |
| `min_silence_at_max_speech_ms` | `min_silence_at_max_speech` | 98 |
| `use_max_possible_silence` | `use_max_poss_sil_at_max_speech` | true |

## Streaming

`transcribe_vad_stream_feed` takes audio in pieces of any size (including
less than a frame) and scores every frame it completes, carrying the model's
recurrent state and audio context across calls. The result holds the
probabilities of the frames this call completed; `first_frame` is the stream
position of the first one. `transcribe_vad_stream_flush` scores the buffered
remainder zero padded and ends the stream; `transcribe_vad_stream_reset` ends
it without scoring.

```c
for (;;) {
    int n = read_microphone(buf, 320);           /* 20 ms */
    transcribe_vad_stream_feed(vad, buf, n);
    transcribe_vad_get_result(vad, &r);
    const float * p = transcribe_vad_probs(vad);
    for (int i = 0; i < r.n_probs; ++i) {
        /* frame r.first_frame + i covers samples
           [(first_frame + i) * 512, (first_frame + i + 1) * 512) */
    }
}
transcribe_vad_stream_flush(vad);
```

On CPU, the probabilities of a stream are bit-identical to those of one
`transcribe_vad_run` over the same audio with the same thread count, for any
chunking (tested with 1 and 4 threads).
Streaming returns probabilities only; segmenting a live stream (hysteresis,
end-of-speech hangover) is the caller's policy. Segmenting the collected
probabilities with the offline rules is what `transcribe_vad_run` does.

## Performance

The front end and the convolutional encoder, which do not depend on the
recurrent state, run as one batched ggml graph over up to 256 frames; the
LSTM recurrence and the output head run on the host. CPU is the default:
`TRANSCRIBE_BACKEND_AUTO` resolves to the CPU for VAD models to preserve
reference accuracy: the measured CPU probability error is about 7e-6,
versus about 2e-3 on Metal. A GPU backend can still be requested
explicitly. Measured numbers are on the model page.

Threading and lifetime rules are shared with the other roles:
[roles.md](roles.md).
