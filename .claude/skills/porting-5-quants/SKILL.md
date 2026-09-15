---
name: porting-5-quants
description: Produces the shipped quant matrix from the reference-dtype GGUF, smoke-tests each, publishes the matrix to a private HF repo, and takes a preliminary 512-utterance WER read (Modal or local) for human review. Use after porting-4-cpp has finalized tolerances and passed validate.py + the full ref-dtype WER gate. Input: models/<variant>/<variant>-<REFDTYPE>.gguf. Output: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M alongside the reference-dtype GGUF; a CLI smoke pass per file; quants pushed to a private HF repo; preliminary per-quant WER. Authoritative full-split quant WER is Stage 7. No tensor-level numerical comparison is required for quant acceptance — that is intentional.
---

# porting-5-quants

Stage 5 of the porting pipeline. Builds the quantizer, runs
`scripts/quantize-all.py`, smoke-tests each GGUF, publishes the matrix to a
private HF repo, and takes a preliminary 512-utterance WER read for human
review. Authoritative full-split quant WER is Stage 7.

## Preconditions

- `porting-4-cpp` complete: `validate.py all` green at ref dtype, the
  family-doc Capability Validation table is fully resolved (no TODO
  rows), and the full ref-dtype C++ WER is no more than the Oracle
  reference WER + 0.01.
- `models/<variant>/<variant>-<REFDTYPE>.gguf` exists, where REFDTYPE is
  the intake's `dtype.expected` mapped to the GGUF preset suffix.
- `build/bin/transcribe-cli` and `build/bin/transcribe-quantize` are
  buildable.
- `hf` authenticated for the target org (private upload). Modal optional
  for the preliminary 512-utterance WER sweep.

## Workflow

```
Quants progress:
- [ ] Step 1: Build transcribe-quantize
- [ ] Step 2: Run quantize-all
- [ ] Step 3: CLI output-validity smoke per produced GGUF
- [ ] Step 4: Publish quants to a private HF repo
- [ ] Step 5: Preliminary 512-utterance WER sweep (Modal or local)
- [ ] Step 6: Sign-off review
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
intentionally lossy, and Stage 7 WER is the user-facing quant quality
report.

```bash
SAMPLE=samples/jfk.wav  # primary sample for the family
for gguf in models/<variant>/<variant>-*.gguf; do
  out="$(build/bin/transcribe-cli -m "$gguf" "$SAMPLE")" && \
    printf '%s\n' "$out" | grep -q '^text: .\+' && echo "OK $gguf" \
    || echo "FAIL $gguf"
done
```

Any FAIL is a real bug; investigate before sign-off.

### Step 4: Publish quants to a private HF repo (execute)

Push the full matrix to a **private** repo (convention
`<org>/<variant>-gguf`). Everything stays private for now; flipping it
public is a future Stage 8 action, not done here.

```bash
hf repo create <org>/<variant>-gguf --repo-type model --private  # if absent
hf upload <org>/<variant>-gguf models/<variant> . --repo-type model
```

### Step 5: Preliminary 512-utterance WER sweep (execute)

Run each quant on the first 512 utterances of the acceptance manifest. This is
a bring-up signal only: it catches a clearly bad quant before benchmarking,
while Stage 7 remains the authoritative full-split publication run. Subset
artifacts carry `.n512` in their names and are never ingested into the catalog.
Use Modal if credentials are available; otherwise run locally.

```bash
# Modal: the output paths include .n512.
modal run scripts/wer/remote/modal_sweep.py::sweep \
  --models <org>/<variant>-gguf --quants "" --n-utts 512

# Local: materialize a separately named subset manifest and output.
mkdir -p build/wer
head -n 512 "$MANIFEST" > build/wer/<variant>.<dataset>.n512.manifest.jsonl
for q in F16 Q8_0 Q6_K Q5_K_M Q4_K_M; do
  uv run scripts/wer/run.py --model models/<variant>/<variant>-$q.gguf \
    --manifest build/wer/<variant>.<dataset>.n512.manifest.jsonl \
    --out reports/wer/<variant>-$q.<dataset>.n512.jsonl
  uv run scripts/wer/score.py reports/wer/<variant>-$q.<dataset>.n512.jsonl
done
```

Report the preliminary per-quant table for user review before Stage 6. Do not
copy these rows into `catalog/<variant>.json`; Stage 7 owns published accuracy.

### Step 6: Sign-off

Report:
- Every produced GGUF with file size.
- Any GGUF that failed the CLI smoke (with the failing output).
- The private HF repo the matrix was pushed to.
- Preliminary 512-utterance per-quant WER table (Stage 7 authoritative).

**Do not commit.**

## Postconditions

- All five derived presets (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M) exist under
  `models/<variant>/`, with the source tier suffix skipped where it
  duplicates one of them.
- Every produced GGUF loads via `build/bin/transcribe-cli` and emits a
  non-empty `text:` line on the primary sample.
- No tensor-level numerical comparison is required (or expected) for
  quant acceptance.
- Quant matrix pushed to a private HF repo (`<org>/<variant>-gguf`).
- Preliminary 512-utterance per-quant WER produced and reviewed; authoritative
  full-split WER is Stage 7, and only Stage 7 results enter the catalog.
- The GGUFs are now the input Stage 6 seeds `catalog/<variant>.json` from
  (`scripts/catalog/new_record.py`), so a wrong capability KV or licence
  read propagates into the catalog. Fix it here, not in the record.

## Pointers (read, not execute)

- `docs/tools/quantization.md` — `transcribe-quantize` invocation
- `scripts/quantize-all.py` — quant matrix driver
- `scripts/lib/quant_policy.py` — per-family tier policy if a family needs
  to override the default DERIVED_PRESETS
