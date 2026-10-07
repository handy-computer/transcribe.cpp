# Migrating to transcribe.cpp 0.4

Version 0.4 introduces **roles** (`docs/roles.md`): a loaded model serves
one or more kinds of work, each with its own API and session type. Rebuild
native consumers against the 0.4 headers and upgrade each language package
and native provider together.

## Diarizers moved to the DIARIZE role

Models whose product is speaker turns (Sortformer) no longer run through the
ASR API. `transcribe_model_roles()` reports `TRANSCRIBE_ROLE_DIARIZE` for
them, and `transcribe_session_init` / `transcribe_open` return
`TRANSCRIBE_ERR_UNSUPPORTED_ROLE`.

| 0.3 | 0.4 |
| --- | --- |
| `transcribe_session_init(model, ...)` | `transcribe_diarize_session_init(model, ...)` |
| `transcribe_run(session, pcm, n, &run_params)` | `transcribe_diarize_run(session, pcm, n, &diarize_params)` |
| `transcribe_n_speaker_segments` / `transcribe_get_speaker_segment` | `transcribe_diarize_n_segments` / `transcribe_diarize_get_segment` (same `transcribe_speaker_segment` rows) |
| `transcribe_sortformer_stream_ext` (`SFST`, RUN slot) | `transcribe_sortformer_diarize_ext` (`SFDR`, DIARIZE_RUN slot); same preset enum |
| `transcribe_model_supports(sortformer, TRANSCRIBE_FEATURE_DIARIZATION)` was true | false; check `transcribe_model_roles()` instead |
| `transcribe-cli --batch ... --batch-jsonl` "speakers" for Sortformer | `transcribe-cli -m sortformer.gguf audio.wav` prints the segments (`-o PATH` writes them); `--batch` is ASR-only |

Segments are byte-identical to 0.3 for the same audio, preset and backend.
Speaker-attributing ASR models (granite, moss, multitalker parakeet) are
unchanged; see `docs/roles.md`.

## New: the LANGID role

Language identification is a new role (`include/transcribe/langid.h`,
`docs/langid.md`) with a new status, `TRANSCRIBE_ERR_INPUT_TOO_SHORT` (21).
It is new API, so nothing migrates.

## Capabilities are ASR-only

`transcribe_model_get_capabilities` returns `TRANSCRIBE_ERR_UNSUPPORTED_ROLE`
for a model without the ASR role, instead of a zeroed struct. Role facts come
from the role's own query (`transcribe_diarize_get_info`).

## Behavior changes on every model

- **Non-finite audio is rejected.** `transcribe_run`, `transcribe_run_batch`
  and `transcribe_stream_feed` return `TRANSCRIBE_ERR_INVALID_ARG` when any
  sample is NaN or +-Inf, before the previous result is touched. In a batch,
  one such utterance rejects the whole batch. A rejected feed leaves the
  stream ACTIVE. Silence is still valid.
- **A stream whose family hook throws now ends.** begin / feed / finalize
  mark it FAILED with `last_status` equal to the returned status (OOM or
  BACKEND); `transcribe_stream_reset` always ends IDLE.
- **Host thread-pool failures return an error** instead of terminating the
  process.

## Language bindings

- New `Model.roles`, `DiarizeSession`, `LangIdSession`, an `UnsupportedRole`
  error for status 20 (`.unsupportedRole` in Swift) and an `InputTooShort`
  error for status 21; see `docs/bindings.md`. The
  Sortformer ASR-path extension (`SortformerStreamOptions` and equivalents)
  is removed. Sortformer runs through a diarize session instead:

  | Binding | 0.3 | 0.4 |
  | --- | --- | --- |
  | Python | `model.session()` + `SortformerStreamOptions` | `model.diarize_session()` + `SortformerDiarizeOptions` |
  | Rust | `RunExtension::Sortformer(SortformerStreamOptions)` | `model.diarize_session()` + `DiarizeOptions` with `DiarizeExtension::Sortformer(SortformerDiarizeOptions)` |
  | Swift | `RunExtension.sortformer(SortformerStreamOptions)` | `model.diarizeSession()` + `DiarizeOptions(family: .sortformer(SortformerDiarizeOptions(...)))` |
  | TypeScript | `{ kind: "sortformer" }` on the `run` slot | `model.createDiarizeSession()` + `{ kind: "sortformer_diarize" }` on the new `"diarize_run"` slot |

  In every binding, `capabilities` on a model without ASR (Sortformer)
  raises / throws / returns `UnsupportedRole` instead of reporting a zeroed
  struct, and `supports(diarization)` is false for Sortformer; use
  `Model.roles`.
- A stream that ends FAILED inside a feed (abort, or a native error)
  releases the model's stream lease in every binding, so other sessions on
  the model can run without `Busy`. A feed rejected before the native call
  (e.g. NaN) keeps the lease and the stream stays usable.
- **Rust:** `Model::capabilities()` returns `Result<Capabilities>`
  (`Err(Error::UnsupportedRole)` on a model without ASR). `ExtSlot` gains
  `DiarizeRun` and `AbiStruct` gains the diarize and langid structs; both are now
  `#[non_exhaustive]`, so later roles add variants without another break.
- **Swift:** `Model.capabilities` is `get throws`.
- **Python:** calls on one model now serialize (they used to race), and a
  run / run_batch / second stream on any session of a model with an active
  stream raises `transcribe_cpp.Busy`, matching the other bindings. On the
  stream's own session these raised `InvalidArgument` in 0.3.
- **TypeScript:** an unknown Sortformer preset string is rejected instead of
  silently using the default.
