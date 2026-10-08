# Binding Author Notes

Use `include/transcribe/extensions.h` as the default header for generated
bindings. It includes `transcribe.h` plus every family extension header shipped
by the install, so generators see the full public surface in one translation
unit.

Use `include/transcribe.h` directly only for bindings that intentionally expose
the stable generic ABI and omit family-specific extension structs and telemetry.

The header split is source organization, not a separate library boundary:
family headers include `transcribe.h`, do not depend on each other, and are
flattened normally by C preprocessors, bindgen, CFFI API-mode builds, cgo
preambles, and Swift/ObjC module maps.

## ABI digest, contracts, and the link manifest

Three binding-neutral artifacts exist so no binding depends on another
binding's generated files:

- **`include/transcribe.abihash`** — the public-ABI digest (sha256/16 over
  the normalized FFI surface: structs, enums, macros, layout, prototypes).
  Emitted by `bindings/python/_generate/generate.py` (the hash oracle) and
  drift-gated in CI alongside `_generated.py`. Every first-class binding
  pins this value: when the header's ABI changes, the hash moves and the
  binding's CI goes red until its FFI layer is regenerated or consciously
  reviewed. Comment-only header edits do not move it.
- **`contract.json`** — stamped into every native artifact directory
  (`_native/` in provider wheels; the root of extracted
  `transcribe-native-<tuple>` bundles): `version`, `header_hash`,
  `backends`, `lane`. A binding validates `version` (pre-1.0: exact base
  match) and `header_hash` (must equal the hash its FFI layer was generated
  against) BEFORE dlopen. This is the same contract the Python provider
  enforces via `_contract.py`.
- **`lib/transcribe-link.json`** — installed by `cmake --install` (the
  `TRANSCRIBE_INSTALL` rules): the machine-readable link interface for
  non-CMake consumers building from source (the Rust `-sys` crate's
  `build.rs`). Archive order, system libs, frameworks, flags, and — for
  `GGML_BACKEND_DL` installs — `module_dir`, the directory to hand to
  `transcribe_init_backends()`. Proven per push by the link-smoke CI lane,
  which compiles a toy C consumer from nothing but this manifest in both
  static and shared postures. The Rust `-sys` crate consumes it from two
  sources: its own vendored source build (the default), or — with the
  `TRANSCRIBE_DIR` env var pointing at an installed prefix — a prebuilt
  tree, skipping the source build entirely.

## Result text pointers: copy at the FFI boundary

Every accessor that returns a `const char *` (`transcribe_full_text`,
`transcribe_detected_language`, and the `text` field of the segment / word /
token row structs) returns a *borrowed* pointer into session-owned storage. See
the "Result text-pointer lifetime" block at the top of `include/transcribe.h`
for the full contract. The binding-relevant rule:

- **Offline path:** the pointer is valid until the next `transcribe_run` /
  `transcribe_stream_begin` / `transcribe_stream_reset` / `transcribe_session_free`
  on the same session.
- **Streaming path:** raw result pointers, including `transcribe_full_text`
  and row `text` fields, may be replaced by every
  `transcribe_stream_feed` / `transcribe_stream_finalize` call. Bindings that
  need UI-stable streaming text should use `transcribe_stream_get_text()` and
  expose owned copies of `committed_text` and `tentative_text`.

The simplest safe rule for a binding is to **copy the bytes at the FFI
boundary**, which the marshal-and-copy idioms each language already uses make
automatic: Python `ctypes.c_char_p` / `.decode()`, Go `C.GoString`, Rust
`CStr::to_str().to_owned()`, Swift `String(cString:)`. A binding that instead
hands out a zero-copy view must scope it to the current callback/update turn
and document that it dies at the next stream mutation.

First-class bindings expose both streaming result shapes as owned values:

- the UI-facing `transcribe_stream_get_text()` view (full, committed, and
  tentative text), and
- a full structured snapshot built from the ordinary current-result accessors
  (clean/raw text, detected language, timestamp kind, segments, speaker turns,
  words, tokens, and timings).

The structured snapshot must be copied completely before the next feed or
finalize call; bindings must not return the session-owned pointers directly.

## Diarization result contract

First-class bindings expose the generic diarization surface rather than only
the generated FFI:

- a run option with `default` / `off` / `on` values (`default` is globally
  off),
- the `diarization` feature probe,
- `speaker_id` / `speakerId` on every segment (1-based, 0 = unattributed), and
- speaker-turn rows with start/end milliseconds, speaker id, and confidence.

Speaker-turn times may both be zero when a model attributes text but does not
produce timing (Granite SAA). Timestamp selection is an independent axis:
explicit unsupported combinations return `UNSUPPORTED_TIMESTAMPS`; `AUTO`
chooses the richest granularity compatible with the selected task. Result
objects own copies of every row and remain valid after the next run.

That contract is for ASR models that attribute speakers inside a transcript.

## Compute lock and stream lease

The C library allows one compute in flight per model and leaves enforcing it
to the caller. Every first-class binding enforces it with a model-wide lock
shared by every session of every role: a call waits behind any other compute
on the same model.

The bindings are also stricter than C about streams. `transcribe.h` lets
several sessions of one model each hold an active stream and interleave
their feeds; a binding allows one active stream per model. Starting a stream
takes the model's stream lease, and ending it (finalize, reset, a feed that
leaves the stream FAILED, or dropping / closing the stream) releases it.
While the lease is held, `run`, `run_batch`, a new stream, a diarize run and
a langid run on any session of that model raise `Busy` instead of waiting. A feed
rejected before the native call (e.g. NaN input) keeps the lease.

## Roles and the DIARIZE session

Each binding exposes the role mask as `Model.roles` (Python `frozenset[Role]`,
TypeScript `readonly ('asr' | 'diarize' | 'langid')[]`, Rust `Roles`, Swift
`Roles` option set). Status 20 surfaces as `UnsupportedRole`
(`.unsupportedRole` in Swift), including from capabilities on a model without
ASR. Status 21 surfaces as `InputTooShort` (`.inputTooShort` in Swift).

A DIARIZE model (`docs/roles.md`) gets its own session type: `DiarizeSession`
from `model.diarize_session()` (Python, Rust), `model.diarizeSession()`
(Swift) or `model.createDiarizeSession()` (TypeScript), plus `diarize_info` /
`diarizeInfo` (sample rate, max speakers). `run(pcm, …)` returns copied-out
speaker-turn rows (the same row type as above) and takes the family's
diarize extension (Sortformer: the preset on the DIARIZE_RUN slot).

A diarize run follows the same execution rules as an ASR run: it holds the
model-wide compute lock, waits behind other compute on the model, raises
`Busy` while a stream on any session of the model holds the stream lease,
keeps its model alive, honours cancellation, and defers native frees that
race an in-flight call.

## The LANGID session

A LANGID model gets `LangIdSession` from `model.langid_session()` (Python,
Rust), `model.langIdSession()` (Swift) or `model.createLangIdSession()`
(TypeScript), plus `langid_info` / `langIdInfo` / `langidInfo` (sample
rate, label count, minimum audio, and the fixed maximum: longer input is
scored on its first `max_audio_ms`) and the label table (codes, names, and
alias lookup). `run(pcm, allowed=…)` returns a copied-out result: every
allowed label ranked by `p` (index, code, name, `p`, `logit`) and
`allowed_mass`.

`allowed` is the first caller-owned `const char * const *` input. Every
binding keeps the array and each encoded string alive until the native call
returns (TypeScript frees them only after its async worker call settles).
Omitted / `None` / `nil` / `null` passes NULL, meaning every label; an empty
list raises `InvalidArgument`.

A langid run follows the same execution rules as an ASR or diarize run:
model-wide compute lock, `Busy` under a stream lease, results copied out
under the lock, cancellation, and deferred frees.

## Raw text

Every first-class binding exposes `raw_text` / `rawText` on the materialized
result alongside the clean `text`: the model's decoded output before family
post-processing (diarization markers, timestamp/special tokens, tag filtering,
whitespace trims). It equals the clean text modulo whitespace for families
that emit clean text natively, and is the recommended replacement for
`keep_special_tags` when the goal is recovering what the model emitted —
unlike the flag it works for every family, covers plain-text markers, and does
not give up the clean transcript. It is present on single, batch, and full
structured stream snapshots (and may be empty before a stream has produced a
successful hypothesis).

## Generic parameter parity

First-class high-level bindings expose every field of the generic model-load,
session, run, and stream parameter structs. Generated/raw FFI coverage is not
sufficient: an application must not need private binding internals to set a
public generic option. In particular, `transcribe_run_params::pnc` and `itn`
are typed three-state controls (`default`, `off`, `on`) and must flow through
single-run, batch, streaming, and one-shot convenience surfaces wherever those
surfaces exist.

Every generic option needs model-free enum/type coverage (plus direct
materialization coverage where the binding architecture permits it) and, when a
supporting model exists, a model-gated behavior test. A new field added to a
generic parameter struct must update all first-class bindings and this
conformance coverage in the same change.

When adding a new family extension, update:

- `include/transcribe/<family>.h` with the typed struct, kind constant, and
  `transcribe_<family>_<name>_ext_init()` function declaration.
- `include/transcribe/extensions.h` with one new include.
- `docs/extension-kinds.md` with the registered FourCC value and the slot
  (`RUN` or `STREAM`) the kind is legal on.
