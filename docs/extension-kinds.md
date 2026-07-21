# Extension Kinds Registry

`transcribe.h` defines an opaque, kind-tagged family extension surface
(`struct transcribe_ext`). Each family declares its own typed extension
structs in `include/transcribe/<family>.h` with `struct transcribe_ext`
embedded at field 0. The kind tag is a 32-bit value; this file is the
registry that prevents collisions in the FourCC value space.

New families reserve a value here in the same PR that introduces the
family header. Kinds are fixed once shipped: an additive layout change
keeps the same kind and grows `size`; an incompatible layout change
gets a new kind.

## Allocation rules

- Prefer readable FourCC-style values (`'P','K','S','T'` little-endian
  packs to `0x54534B50`).
- Use upper-case ASCII for the four bytes when possible — case-folding
  collisions are easier to spot during review.
- One kind per schema. A family may own multiple kinds when it ships
  configuration surfaces that apply to different model variants
  (e.g. cache-aware vs chunked-attention parakeet streaming).
- Document the accepting model variants in the family header next to
  the kind constant, not here. This file is just the value registry.

## Slot

Each kind is legal in exactly one API slot. The slot enumerator names the
entry-point family whose `family` field the typed extension is pointed at:

- `TRANSCRIBE_EXT_SLOT_RUN` — `transcribe_run_params::family`, validated
  by `transcribe_run`.
- `TRANSCRIBE_EXT_SLOT_STREAM` — `transcribe_stream_params::family`,
  validated by `transcribe_stream_begin`.

Mismatched-slot extensions are rejected with `TRANSCRIBE_ERR_INVALID_ARG`
at the entry point. A future slot (e.g. `_SESSION`, `_MODEL_LOAD`,
`_TOKENIZE`) would be one new enum value alongside an entry-point
parameter; existing kinds keep their slot for life.

## Registry

| Kind value     | FourCC | Slot     | Symbol                                          | Family       | Header                          |
| -------------- | ------ | -------- | ----------------------------------------------- | ------------ | ------------------------------- |
| `0x54534B50`   | `PKST` | `STREAM` | `TRANSCRIBE_EXT_KIND_PARAKEET_STREAM`           | parakeet     | `include/transcribe/parakeet.h` |
| `0x53424B50`   | `PKBS` | `STREAM` | `TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM`  | parakeet     | `include/transcribe/parakeet.h` |
| `0x5453534D`   | `MSST` | `STREAM` | `TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM` | moonshine_streaming | `include/transcribe/moonshine_streaming.h` |
| `0x4E524857`   | `WHRN` | `RUN`    | `TRANSCRIBE_EXT_KIND_WHISPER_RUN`               | whisper      | `include/transcribe/whisper.h`  |
| `0x54535256`   | `VRST` | `STREAM` | `TRANSCRIBE_EXT_KIND_VOXTRAL_REALTIME_STREAM`   | voxtral_realtime | `include/transcribe/voxtral_realtime.h` |
| `0x54534653`   | `SFST` | `RUN`    | `TRANSCRIBE_EXT_KIND_SORTFORMER_STREAM`         | sortformer   | `include/transcribe/sortformer.h` |

Empty rows reserved for future allocations:

| Kind value | FourCC | Slot | Symbol | Family | Header |
| ---------- | ------ | ---- | ------ | ------ | ------ |
| _next_     |        |      |        |        |        |
