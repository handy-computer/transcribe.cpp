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

When adding a new family extension, update:

- `include/transcribe/<family>.h` with the typed struct, kind constant, and
  INIT macro.
- `include/transcribe/extensions.h` with one new include.
- `docs/extension-kinds.md` with the registered FourCC value.
