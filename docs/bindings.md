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

When adding a new family extension, update:

- `include/transcribe/<family>.h` with the typed struct, kind constant, and
  `transcribe_<family>_<name>_ext_init()` function declaration.
- `include/transcribe/extensions.h` with one new include.
- `docs/extension-kinds.md` with the registered FourCC value and the slot
  (`RUN` or `STREAM`) the kind is legal on.
