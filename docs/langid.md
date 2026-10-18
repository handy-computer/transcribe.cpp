# Language ID

A LANGID model (`transcribe_model_roles()` has `TRANSCRIBE_ROLE_LANGID`)
identifies the spoken language of a clip and returns its labels ranked by
probability. API: `include/transcribe/langid.h`. The shipped model is
[VoxLingua107 ECAPA-TDNN](models/lang-id-voxlingua107-ecapa.md), 107 labels.

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
transcribe_langid_get_result(lid, &r);             /* r.n_candidates == 3 */
struct transcribe_langid_candidate c;
transcribe_langid_candidate_init(&c);
transcribe_langid_get_candidate(lid, 0, &c);       /* c.code, c.p, r.allowed_mass */

transcribe_langid_session_free(lid);
transcribe_model_free(m);
```

From the CLI: `transcribe-cli -m lang-id-voxlingua107-ecapa-Q8_0.gguf --allow en,de,fr --top 3 clip.wav`
(`--top N` only limits how many ranked candidates the CLI prints; 0 = all).

## Parameters and results

- **`allowed`** restricts the decision to the listed labels: candidates are
  ranked by `p`, the softmax over the allowed set. The network still scores
  every label. NULL with
  `n_allowed == 0` means every label. A non-NULL list with `n_allowed <= 0`,
  a NULL list with `n_allowed > 0`, or a NULL element is
  `TRANSCRIBE_ERR_INVALID_ARG`; an unknown code is
  `TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE`; duplicates count once.
- **`allowed_mass`** (`transcribe_langid_result`) is the share of the softmax
  over every label that falls in the allowed set, 1 when unrestricted. A low
  value means the unrestricted model puts most of its probability outside the
  allowed set.
- **Candidates.** Every run returns every allowed label, ranked by `p`
  (descending; ties keep label order). `n_candidates`
  (`transcribe_langid_result`) is the allowed-set size after duplicates
  count once: `n_labels` when unrestricted.
- **Length.** Input longer than `transcribe_langid_info::max_audio_ms`
  (30000 for VoxLingua107) is scored on its first `max_audio_ms`, with no
  error; the window is a fixed fact of the model, not a session setting.
  Scored audio shorter than `transcribe_langid_info::min_audio_ms` (500 ms)
  is `TRANSCRIBE_ERR_INPUT_TOO_SHORT`.
- **Labels** are the model's own codes. VoxLingua107 uses `iw` (Hebrew),
  `jw` (Javanese), `tl` (Filipino) and `no` (Norwegian Bokmal); the GGUF
  stores `he`, `jv`, `fil` and `nb` as aliases, accepted in `allowed` and by
  `transcribe_langid_label_index`. Results always report the model's code.
  Codes are not canonicalized across models: match `code` against an ASR
  model's `transcribe_capabilities::languages` yourself.

The model is a closed-set classifier with no "unknown" or "no speech" label.

Sessions, threading and model lifetime follow the rules shared by every role
(`docs/roles.md`).
