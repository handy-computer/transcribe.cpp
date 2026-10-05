# Language ID

A LANGID model (`transcribe_model_roles()` has `TRANSCRIBE_ROLE_LANGID`)
answers "which language is this?" API: `include/transcribe/langid.h`. The one
shipped family is ecapa_tdnn, SpeechBrain's VoxLingua107 ECAPA-TDNN
(`docs/models/lang-id-voxlingua107-ecapa.md`): a classifier over 107 labels.

Almost all of the difference between a good and a bad integration is in what
you feed it and what you let it choose from. This page is the protocol, with
the measured numbers behind each rule. They come from FLEURS `test`, 15
languages x 200 utterances (`scripts/langid/`), scored with the SpeechBrain
reference; the C++ engine makes the same decisions on the same audio.

## Minimal use

```c
#include "transcribe/langid.h"

struct transcribe_model * m = NULL;
transcribe_model_load_file("lang-id-voxlingua107-ecapa-Q8_0.gguf", NULL, &m);

struct transcribe_langid_session * lid = NULL;
transcribe_langid_session_init(m, NULL, &lid);

const char * allowed[] = { "en", "de", "fr" };
struct transcribe_langid_params lp;
transcribe_langid_params_init(&lp);
lp.allowed   = allowed;
lp.n_allowed = 3;
transcribe_langid_run(lid, pcm, n_samples, &lp);   /* 16 kHz mono float32 */

struct transcribe_langid_result r;
transcribe_langid_result_init(&r);
transcribe_langid_get_result(lid, &r);
struct transcribe_langid_candidate c;
transcribe_langid_candidate_init(&c);
transcribe_langid_get_candidate(lid, 0, &c);       /* c.code, c.p, r.allowed_mass */

transcribe_langid_session_free(lid);
transcribe_model_free(m);
```

From the CLI: `transcribe-cli -m lang-id-voxlingua107-ecapa-Q8_0.gguf --allow en,de,fr --top 3 clip.wav`.

## 1. Feed speech, not the raw capture

Trim leading silence before you crop. Any VAD or energy gate will do; the
evaluation's "trimmed" recipe drops audio up to the first 20 ms frame whose
RMS exceeds 5% of the clip's loudest frame.

Open-set top-1 accuracy, mean over the 15 languages:

| | 3 s | 5 s | 10 s | full |
|---|---|---|---|---|
| crop from sample 0 | 67.0 | 85.2 | 91.1 | 91.4 |
| crop from first speech | 83.4 | 89.1 | 91.7 | 91.5 |

At 3 s that is +16.4 points for a change that costs nothing. FLEURS clips open
with silence, so an untrimmed "3 s" is roughly 2 s of speech. By 10 s the gap
has closed: the amount of speech is what matters, not the amount of audio.
Give it at least 3 s of speech and prefer 5 s.

## 2. Always pass `allowed`

A dictation app knows which languages its user has enabled. Passing them in
`transcribe_langid_params::allowed` masks every other logit before the
argmax. This is the largest accuracy lever, and it is free.

Accuracy over a user's selection, trimmed crops:

| selection | 3 s | 5 s | 10 s | full |
|---|---|---|---|---|
| en+zh | 99.8 | 100.0 | 100.0 | 100.0 |
| en+ru | 99.5 | 100.0 | 100.0 | 100.0 |
| es+pt | 100.0 | 99.8 | 100.0 | 100.0 |
| ja+ko | 96.0 | 99.8 | 100.0 | 100.0 |
| no+da | 99.8 | 100.0 | 100.0 | 100.0 |
| en+fr+de+es | 96.9 | 97.9 | 99.8 | 99.5 |
| de+fr+ja | 94.2 | 96.8 | 99.7 | 99.3 |
| cs+sk | 92.8 | 93.8 | 96.5 | 96.2 |
| id+ms | 88.2 | 90.2 | 89.8 | 89.0 |

`en+ru` is 99.5% at 3 s while open-set Russian is 54.0%: open-set errors go
to neighbouring languages (`ru` -> `be`), and a selection without the
neighbour removes them. A wider but bounded 40-language list lifts the
untrimmed open-set mean from 67.0 / 85.2 / 91.1 to 77.4 / 92.3 / 96.6 at
3 / 5 / 10 s.

If the user has exactly one language enabled, don't call language ID at all.

Only NULL (with `n_allowed == 0`) means "every label". A non-NULL list with
`n_allowed <= 0`, a NULL list with `n_allowed > 0`, or a NULL element is
`TRANSCRIBE_ERR_INVALID_ARG`; an unknown code or alias is
`TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`; duplicates count once.

## 3. Decide at 5 s, re-decide at 10 s if you are not confident

Identify on the first 5 s of speech; if the top `p` is below 0.9, identify
again at 10 s and use that. What 0.9 buys you, measured on full clips (mean
~12 s), so this is the confidence-to-precision relation rather than the 5 s
operating point:

| decision space | threshold | coverage % | precision % |
|---|---|---|---|
| open (107) | 0.5 | 97.7 | 92.6 |
| open (107) | 0.7 | 88.8 | 95.9 |
| open (107) | 0.9 | 77.1 | 99.0 |
| 40-language list | 0.5 | 98.7 | 97.3 |
| 40-language list | 0.7 | 95.9 | 98.6 |
| 40-language list | 0.9 | 92.0 | 99.4 |

The median top probability is 0.999 when the model is right and 0.650 when it
is wrong, so a 0.9 gate is a real signal.

## 4. `allowed_mass` is the "not in your selection" signal

`transcribe_langid_result::allowed_mass` is the unrestricted probability the
allowed set captured (1.0 when unrestricted). When a user with `en+de` starts
speaking Japanese, `p` still confidently picks `en` or `de` (the mask cannot
express "neither"), but `allowed_mass` collapses. Use it to prompt or to hold
off an automatic switch, not as a hard reject: it also drops on noise and on
very short audio.

This is a closed-set classifier: there is no "no speech" or "unknown
language" label, and silence, music or noise still produce a confident-looking
top candidate. Gate the input with VAD; there is no rejection threshold you
can rely on from the scores alone.

## 5. Long dictations: re-identify on a rolling window

Re-identify every 20-30 s on recent speech rather than on the whole recording.
`transcribe_langid_session_params::max_audio_ms` (default 30000) caps the
scored span, and longer input is scored on its last `max_audio_ms`, so handing
the growing buffer to `transcribe_langid_run` is a trailing-window classifier.
`transcribe_langid_result::audio_ms` reports what was scored.

The log-mel is mean-normalised over the whole scored span, so a 5 s window's
result is not the prefix of the 10 s result: overlapping calls are independent
decisions. Cost scales with the scored span, not the buffer.

## 6. Labels are the model's own codes

Codes are mostly ISO 639-1 with VoxLingua107's legacy spellings (`iw` Hebrew,
`jw` Javanese, `tl` Filipino, `no` Norwegian Bokmal). The GGUF stores aliases,
so `he`, `jv`, `fil` and `nb` are accepted in `allowed` and by
`transcribe_langid_label_index`, but results always report the model's code.

There is no canonicalization across models. To feed the answer to an ASR
model, match `code` against that model's `transcribe_capabilities::languages`
yourself (and map `iw` / `he` etc. where they differ).

## 7. Known weak spots

Top open-set confusions (untrimmed, summed over all crops):

| true -> predicted | errors |
|---|---|
| ru -> be (Belarusian) | 271 |
| no -> nn (Norwegian Nynorsk) | 229 |
| id -> jw (Javanese) | 200 |
| ms -> id | 157 |
| cs -> sk | 59 |

- **`no` / `nn`** are not reliably separable. If you mean "Norwegian", put
  `no` in the selection and leave `nn` out.
- **`ru` / `be`**: excluding `be` fixes it.
- **`id` / `ms`** stay at 88-90% even restricted to the pair, at every crop
  length; distinguish them downstream if you must.
- **`cs` / `sk`** improve with audio but stay around 96%.
- **No Cantonese label**: Cantonese is classified as `zh`.
- **`de`** is weak open-set (68.5% at 3 s trimmed), mostly to `ro`, `yi`, `nn`.

## 8. Input, threading and placement

- 16 kHz mono float32; NaN / Inf is `TRANSCRIBE_ERR_INVALID_ARG`.
- Scored audio under 500 ms (`transcribe_langid_info::min_audio_ms`) is
  `TRANSCRIBE_ERR_INPUT_TOO_SHORT`. Between 500 ms and 3 s the answer is weak;
  treat it as provisional.
- `top_k` trims the returned list only; the mask and softmax are unaffected.
- Sessions, model lifetime and the one-compute-per-model rule are those of
  every role (`docs/roles.md`). Budget `n_threads` explicitly when language ID
  and ASR run on the CPU at the same time.
- Placement is per model (`transcribe_model_load_params`); a language ID model
  can sit on the CPU while ASR uses the GPU.
