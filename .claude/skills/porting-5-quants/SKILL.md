---
name: porting-5-quants
description: Produces the shipped quant matrix from the reference-dtype GGUF and runs CLI output-validity smoke per produced GGUF. Use after porting-4-cpp has finalized tolerances and passed validate.py + the subset WER sanity. Input: models/<variant>/<variant>-<REFDTYPE>.gguf. Output: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M alongside the reference-dtype GGUF; a CLI smoke pass per file. No tensor-level numerical comparison is required for quant acceptance — that is intentional.
---

# porting-5-quants

Stage 5 of the porting pipeline. Mechanical: builds the quantizer, runs
`scripts/quantize-all.py`, and confirms every produced GGUF is loadable by
the C++ runtime. WER for quants is Stage 7's release sweep, not here.

## Preconditions

- `porting-4-cpp` complete: `validate.py all` green at ref dtype, the
  family-doc Capability Validation table is fully resolved (no TODO
  rows), and subset WER vs reference is within 0.005.
- `models/<variant>/<variant>-<REFDTYPE>.gguf` exists, where REFDTYPE is
  the intake's `dtype.expected` mapped to the GGUF preset suffix.
- `build/bin/transcribe-cli` and `build/bin/transcribe-quantize` are
  buildable.

## Workflow

```
Quants progress:
- [ ] Step 1: Build transcribe-quantize
- [ ] Step 2: Run quantize-all
- [ ] Step 3: CLI output-validity smoke per produced GGUF
- [ ] Step 4: Sign-off review
```

### Step 1: Build the quantizer (execute)

```bash
cmake --build build --target transcribe-quantize
```

### Step 2: Run quantize-all (execute)

```bash
uv run scripts/quantize-all.py models/<variant>/<variant>-<REFDTYPE>.gguf
```

Produces `<variant>-F16.gguf`, `<variant>-Q8_0.gguf`, `<variant>-Q6_K.gguf`,
`<variant>-Q5_K_M.gguf`, `<variant>-Q4_K_M.gguf` alongside the reference-
dtype GGUF. The script skips the tier suffix that duplicates the source.

### Step 3: CLI output-validity smoke (execute)

For every produced GGUF, confirm the C++ runtime can load it and produce
a valid transcript on the primary sample for the family. Do **not** run
tensor/numeric comparisons on quantized GGUFs — quantization is
intentionally lossy, activation drift is not a stable acceptance signal,
and Stage 7 WER is the user-facing quant quality report.

```bash
SAMPLE=samples/jfk.wav  # primary sample for the family
for gguf in models/<variant>/<variant>-*.gguf; do
  out="$(build/bin/transcribe-cli -m "$gguf" "$SAMPLE")" && \
    printf '%s\n' "$out" | grep -q '^text: .\+' && echo "OK $gguf" \
    || echo "FAIL $gguf"
done
```

Any FAIL is a real bug — investigate before sign-off. Common causes:
loader missing a quant tier in `weights.cpp`, fused-norm code path not
handling the q-tier dequant, missing tensor in the per-quant tier.

### Step 4: Sign-off

Report:
- Listing of every produced GGUF with file size.
- Any GGUF that failed the CLI smoke (with the failing output).

**Do not commit.**

## Postconditions

- All five derived presets (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M) exist under
  `models/<variant>/`, with the source tier suffix skipped where it
  duplicates one of them.
- Every produced GGUF loads via `build/bin/transcribe-cli` and emits a
  non-empty `text:` line on the primary sample.
- No tensor-level numerical comparison is required (or expected) for
  quant acceptance.

## Pointers (read, not execute)

- `docs/tools/quantization.md` — `transcribe-quantize` invocation
- `scripts/quantize-all.py` — quant matrix driver
- `scripts/lib/quant_policy.py` — per-family tier policy if a family needs
  to override the default DERIVED_PRESETS
