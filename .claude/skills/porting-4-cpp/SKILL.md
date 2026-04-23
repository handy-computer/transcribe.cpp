---
name: porting-4-cpp
description: Brings up the C++ implementation for a new model family, finalizes the per-tensor tolerances file from observed drift (replacing the provisional skeleton Stage 2 wrote), and generates the full shipped quant matrix. Use after porting-3-convert has produced the reference-dtype GGUF. Input: intake.json, manifest, dump_coverage.json, provisional tolerances, reference-dtype GGUF, ggml-reference-map.md. Output: src/arch/<family>/* source, finalized tests/tolerances/<family>.json (LLM-authored, human-reviewed, with a _comment block explaining drift sources; every tensor entry no longer carries _provisional), validate.py all green at ref dtype, and models/<slug>/<slug>-<PRESET>.gguf for F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M. This is numerical-validation-critical; treat it that way.
---

# porting-4-cpp

Stage 4 of the porting pipeline. Implements `src/arch/<family>/*`, authors `tests/tolerances/<family>.json` from the first honest C++ drift, and ends with a full shipped quant matrix. The outer safety net is Stage 6's ref-dtype WER gate; the inner safety net is this skill's tolerance review.

## Preconditions

- `reports/porting/<family>/<variant>/intake.json` schema-valid, Preflight Gate A green.
- `build/validate/<family>/<variant>/dump_coverage.json` exists.
- `models/<variant>/<variant>-<REFDTYPE>.gguf` exists, Preflight Gate B green.
- `tests/golden/<family>/<variant>.manifest.json` populated.
- `build/bin/transcribe-cli` and `build/bin/transcribe-quantize` are buildable.

## Workflow

```
CPP progress:
- [ ] Step 1: Check sibling-variant shortcut
- [ ] Step 2: Implement src/arch/<family>/ (open-ended)
- [ ] Step 3: First validate.py run (against provisional tolerances) — capture observed drift
- [ ] Step 4: LLM-finalize tolerances (recipe below, replaces provisional)
- [ ] Step 5: Human review of finalized tolerances
- [ ] Step 6: validate.py all green at ref dtype
- [ ] Step 7: Generate full quant matrix
- [ ] Step 8: Structural loader check on every quant
- [ ] Step 9: Sign-off review
```

### Step 1: Sibling-variant shortcut (execute)

If `src/arch/<family>/` already exists (a prior variant was ported) and the new variant declares the same `architecture_pattern`:

```bash
uv run scripts/validate.py all --family <family> --variant <variant>
```

**If it exits 0**, Stage 4 is done. Skip to Step 7 (quant matrix). Existing tolerances hold.

**If it fails**, the new variant exposes an assumption that didn't hold (often a config field in `config.varying_across_variants`). Continue to Step 2 — fix the `src/arch/<family>/` code that assumed the old shape.

### Step 2: Implement src/arch/<family>/ (open-ended)

For a new family, create `src/arch/<family>/{weights,encoder,decoder,model,capabilities}.{h,cpp}` following the shape of the closest existing family:
- `encoder-transducer` → `src/arch/parakeet/`
- `encoder-decoder` → `src/arch/cohere/`
- `audio-llm` → `src/arch/qwen3_asr/`
- `encoder-ctc` → no in-tree reference yet; combine parakeet's encoder with a minimal CTC head.

**Use `docs/porting/ggml-reference-map.md` as the primary ggml-pattern lookup** (TOC at top of file). Read the relevant subsections only, not the whole file. Load the closest family's `src/arch/<family>/` as a working example.

Wire the new family into the arch dispatch in `src/` per the existing family wiring. Build iteratively:

```bash
cmake --build build --target transcribe-cli
```

This step is open-ended engineering work. The skill's job is to frame the loop, not to write C++. Surface to the user any ggml-pattern decisions they should make explicitly rather than silently (see the "Human Decisions" list in the parent CLAUDE.md).

**Family-specific requirements.** Some families ship implementation work beyond the arch pattern itself — e.g. whisper must support loading whisper.cpp-compatible `.bin` files on the loader side. These extras do NOT flow through intake → convert → validate (our GGUF remains the canonical numerical reference), but they are real C++ work for this stage. Capture each one in `docs/porting/families/<family>.md` under a "Family-specific implementation notes" section, and confirm with the user at the start of Stage 4 that all such items are listed. Skip none silently.

### Step 3: First validate.py run (execute)

```bash
uv run scripts/validate.py all --family <family> --variant <variant>
```

This runs against the **provisional tolerances skeleton** that `porting-2-refdump` wrote. Because the provisional tolerances are deliberately loose, the comparison completes end-to-end rather than short-circuiting, and the report lists per-tensor observed `max_abs` / `mean_abs` between C++ and reference. **This is the raw material for Step 4.**

If validate.py fails for reasons other than loose tolerances (C++ crashes, shape mismatches, missing tensor dumps), those are real bugs — fix them before finalizing tolerances. Finalizing tolerances on top of broken C++ papers over the problem.

### Step 4: LLM-finalize tolerances (execute)

Load Stage 2 provisional `tests/tolerances/<family>.json` and for each tensor compare the provisional (10× ref-vs-ref noise floor) against observed C++ drift. Do not rewrite from scratch — **tighten, keep, or widen each entry with explicit justification**. Recipe: `finalized = max(1.5 × observed, provisional_noise_floor, 1e-6)`. Per-tensor decision:

- `1.5 × observed < provisional`: C++ drift is at or below noise — rare, suspicious. Keep the noise floor; tighten only with user confirmation.
- `1.5 × observed ≈ provisional`: C++ is within noise. Preserve provisional.
- `1.5 × observed > provisional`: C++ introduced drift. Record the widening factor in `_comment` and name the mechanism.

Remove every `_provisional: true` flag during finalization. Presence of `_provisional: true` at sign-off is a blocking error. File shape mirrors `tests/tolerances/parakeet.json` — top-of-file `_comment` array then per-tensor entries.

**The `_comment` block is the numerical-conception part.** Write it BY INSPECTING THE DRIFT PROFILE. It must name: (a) reference framework + mode (fp32 NeMo? fp16 Transformers?), (b) C++ compute dtype (CPU fp32? Metal fp16?), (c) dominant drift source (STFT precision, attention op-order, LSTM init, etc.), (d) attenuation pattern through the network — flat or decaying? Flat is suspect and must be named, (e) which entries were tightened/kept/widened vs. Stage 2 provisional and why for the widenings, (f) a `_suspects` sub-list naming every tensor whose finalized tolerance exceeds 5% of signal magnitude. Suspects MUST be debugged in Step 5, not silently accepted. See `tests/tolerances/parakeet.json` for the canonical shape.

### Step 5: Human review of tolerances

Hand the file to the user. The user reads the `_comment` block (in particular the `_suspects` list) and either:
- Accepts the tolerances as-is (signs off).
- Pushes back — the skill returns to Step 2 to debug the suspect tensors, not to Step 4 to loosen their tolerances.

**Silent acceptance is how bugs hide.** Loose tolerances without a named mechanism must be debugged before they are accepted.

### Step 6: Ref-dtype validation green (execute)

```bash
uv run scripts/validate.py all --family <family> --variant <variant>
```

Must exit 0. If it fails after tolerances were accepted, either the tolerances are genuinely too tight (rare if Step 4's recipe was applied correctly) or C++ drift is flaky across runs (investigate — it shouldn't be).

### Step 7: Full quant matrix (execute)

```bash
cmake --build build --target transcribe-quantize
uv run scripts/quantize-all.py models/<variant>/<variant>-<REFDTYPE>.gguf
```

This produces `<variant>-F16.gguf`, `<variant>-Q8_0.gguf`, `<variant>-Q6_K.gguf`, `<variant>-Q5_K_M.gguf`, `<variant>-Q4_K_M.gguf` alongside the reference-dtype GGUF.

### Step 8: Structural loader check (execute)

For each produced GGUF, confirm it loads via the C++ loader without error. Minimally, the existing `transcribe_loader_smoke` CTest covers synthetic error paths; for the real GGUFs run a tiny transcribe-cli invocation:

```bash
for gguf in models/<variant>/<variant>-*.gguf; do
  build/bin/transcribe-cli -m "$gguf" samples/jfk.wav >/dev/null && echo "OK $gguf" || echo "FAIL $gguf"
done
```

### Step 9: Sign-off

Report:
- Path to the tolerances file + confirmation the user reviewed it.
- `validate.py all` exit code at ref dtype.
- Full quant matrix listing with file sizes.
- Any quant that failed structural load.

**Do not commit.**

## Postconditions

- `src/arch/<family>/` exists with the five-file shape (`weights`, `encoder`, `decoder`, `model`, `capabilities` — each as a `.h` / `.cpp` pair).
- `tests/tolerances/<family>.json` exists, has a `_comment` block, no `_provisional` flags remain, and was reviewed by the user.
- `validate.py all --family <family> --variant <variant>` exits 0 at reference dtype.
- All five derived presets present under `models/<variant>/`.
- Every produced GGUF loads structurally.

## Pointers (read, not execute)

- `docs/porting/ggml-reference-map.md` — ggml-pattern lookup (TOC at top). Primary Stage 4 aid.
- `docs/porting/4-numerical-validation.md` — validate.py contract and common failure modes.
- `docs/porting/4a-numerical-troubleshooting.md` — drift patterns and their usual causes.
- `docs/tools/quantization.md` — transcribe-quantize invocation.
- `src/arch/parakeet/`, `src/arch/cohere/`, `src/arch/qwen3_asr/` — working arch implementations. Use as structure references, not code to copy wholesale.
- `tests/tolerances/parakeet.json` — canonical `_comment` block shape reference.
