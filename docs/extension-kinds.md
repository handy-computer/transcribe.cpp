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

## Registry

| Kind value     | FourCC | Symbol                                          | Family       | Header                          |
| -------------- | ------ | ----------------------------------------------- | ------------ | ------------------------------- |
| `0x54534B50`   | `PKST` | `TRANSCRIBE_EXT_KIND_PARAKEET_STREAM`           | parakeet     | `include/transcribe/parakeet.h` |
| `0x53424B50`   | `PKBS` | `TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM`  | parakeet     | `include/transcribe/parakeet.h` |
| `0x5453534D`   | `MSST` | `TRANSCRIBE_EXT_KIND_MOONSHINE_STREAMING_STREAM` | moonshine_streaming | `include/transcribe/moonshine_streaming.h` |
| `0x4E524857`   | `WHRN` | `TRANSCRIBE_EXT_KIND_WHISPER_RUN`               | whisper      | `include/transcribe/whisper.h`  |

Empty rows reserved for future allocations:

| Kind value | FourCC | Symbol | Family | Header |
| ---------- | ------ | ------ | ------ | ------ |
| _next_     |        |        |        |        |
