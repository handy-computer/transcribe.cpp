# Roles

A loaded model serves one or more **roles**, the kinds of work it can do.
`transcribe_model_roles(model)` returns them as a bitmask of
`transcribe_role`; an entry point that needs a role the model lacks returns
`TRANSCRIBE_ERR_UNSUPPORTED_ROLE`.

| Role | API | Session | Product |
|---|---|---|---|
| `TRANSCRIBE_ROLE_ASR` | `include/transcribe.h` | `transcribe_session` | text, timestamps, speaker-attributed segments |
| `TRANSCRIBE_ROLE_DIARIZE` | `include/transcribe/diarize.h` | `transcribe_diarize_session` | who spoke when |

Every role follows the same rules:

- **One load per model**, with its own backend (`transcribe_model_load_params`).
- **Sessions.** Each role has its own session type created from a model.
  A session is used by one thread at a time; the model outlives its
  sessions; at most one compute is in flight per model across all sessions
  of all roles. Different models compute in parallel.
- **Input.** 16 kHz mono float32 PCM; NaN / Inf is `INVALID_ARG`.
- **Results** are copied out and replaced by the next run; malformed input
  is rejected before the previous result is touched.

ASR models that attribute speakers inside a transcript (granite, moss,
multitalker parakeet) keep `transcribe_run_params::diarize` and stay ASR.
The DIARIZE role is for models whose product is speaker turns
(Sortformer).

Family extensions name the entry point they apply to through their slot;
see `docs/extension-kinds.md`.
