---
name: porting-4-cpp
description: Brings up the C++ implementation for a new model family, finalizes the per-tensor tolerances file from observed drift in the reference-dtype + matching-KV-dtype regime (replacing the provisional skeleton Stage 2 wrote), and generates the full shipped quant matrix. Use after porting-3-convert has produced the reference-dtype GGUF. Input: intake.json, manifest, dump_coverage.json, provisional tolerances, reference-dtype GGUF, ggml-reference-map.md. Output: src/arch/<family>/* source, finalized tests/tolerances/<family>.json (LLM-authored, human-reviewed, with a _comment block explaining drift sources; every tensor entry no longer carries _provisional), validate.py all green at ref dtype, and models/<slug>/<slug>-<PRESET>.gguf for F16/Q8_0/Q6_K/Q5_K_M/Q4_K_M. This is numerical-validation-critical; treat it that way.
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
- [ ] Step 7: Verify frontend parity for production inference
- [ ] Step 8: Verify advertised capabilities are implemented
- [ ] Step 9: Generate full quant matrix
- [ ] Step 10: CLI output-validity check on every quant
- [ ] Step 11: Sign-off review
```

### Step 1: Sibling-variant shortcut (execute)

If `src/arch/<family>/` already exists (a prior variant was ported) and the new variant declares the same `architecture_pattern`:

```bash
uv run scripts/validate.py all --family <family> --variant <variant>
```

**If it exits 0**, Stage 4 is done. Skip to Step 7 (quant matrix). Existing tolerances hold.

**If it fails**, the new variant exposes an assumption that didn't hold (often a config field in `config.varying_across_variants`). Continue to Step 2 — fix the `src/arch/<family>/` code that assumed the old shape.

### Step 2: Implement src/arch/<family>/ (open-ended)

For a new family, create `src/arch/<family>/` with these files, following the shape of the closest existing family:
- `weights.{h,cpp}`, `encoder.{h,cpp}`, `decoder.{h,cpp}` — one `.h`/`.cpp` pair each.
- `model.cpp` and `capabilities.cpp` — implementation only. The corresponding types and declarations live in a single family-level header `<family>.h` (see `parakeet/parakeet.h`, `cohere/cohere.h`, `qwen3_asr/qwen3_asr.h`), which declares the concrete `*Model` / `*Context` subclasses and the `apply_family_invariants` entry point. No separate `model.h` or `capabilities.h`.

Pick the closest existing family by architecture pattern:
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

**Mid-generation tensor coverage (autoregressive decoders).** Families whose decoder maintains a KV cache (`encoder-decoder`, `audio-llm`) MUST include at least one dump point that exercises the `n_past > 0` step-graph code path. The prompt pass (`n_past=0, n_tokens=seq_len`) does not cover cache write/read offsets, causal-mask indexing for a single new token, or position-id handling past the prompt — a bug in any of those can pass every prompt-pass tolerance and still corrupt the model mid-transcription. Convention: a tensor named `dec.logits_raw.gen<N>` (or equivalent for the family's logit shape), captured after N ≥ 8 completed step-loop iterations on BOTH sides (reference dumper and C++ runner), with both sides running matching greedy rules (same suppress/begin-suppress policy, same argmax) so the cached prefix agrees. If `build/validate/<family>/<variant>/dump_coverage.json` doesn't list a mid-generation tensor, add it here in both the reference dumper and the C++ runner, and file the missing coverage back to `porting-2-refdump` for the next family.

### Step 3: First validate.py run (execute)

```bash
uv run scripts/validate.py all --family <family> --variant <variant>
```

This runs against the **provisional tolerances skeleton** that `porting-2-refdump` wrote. Because the provisional tolerances are deliberately loose, the comparison completes end-to-end rather than short-circuiting, and the report lists per-tensor observed `max_abs` / `mean_abs` between C++ and reference. **This is the raw material for Step 4.**

If validate.py fails for reasons other than loose tolerances (C++ crashes, shape mismatches, missing tensor dumps), those are real bugs — fix them before finalizing tolerances. Finalizing tolerances on top of broken C++ papers over the problem.

**Measurement hygiene.** Every drift number that lands in the tolerance file MUST come from a `validate.py all` run with the default flags. Do not copy numbers from direct `transcribe-cli` invocations — they may be running a different compute path (C++ mel frontend vs. reference mel injection, default KV dtype vs. an explicit override, a different quant preset), and silently mixing measurements from different paths into one tolerance file gives loose tolerances that mask real regressions. If you need to probe a specific configuration (e.g. `--kv-type f32` vs. `--kv-type f16` to isolate a cache drift component), measure it via `validate.py` and label the measurement explicitly — never fold it into the reference-regime numbers.

### Step 4: LLM-finalize tolerances (execute)

**The correctness regime.** Stage 4 validates C++ against reference in a single fixed regime:

- reference-dtype GGUF (F32 / F16 / BF16 per the intake);
- KV cache dtype matching the weight dtype (AUTO should resolve this automatically; if the family's AUTO policy defaults to F16 on an F32 model, fix the policy — AUTO means "match the weights" — or run validation with explicit `--kv-type f32`);
- reference mel (injected via the family's mel-override env var if the C++ mel frontend hasn't been implemented yet);
- backend/threads chosen for determinism (`--backend cpu --threads 1` is the conservative default).

**Every tolerance number in this file is measured in that regime.** Production variants — quantized GGUFs, F16 cache on an F32 model, the C++ mel frontend, GPU backends — produce different (usually wider) drift. Those measurements belong in Stage 5/6/7 reports, NOT in this file. Mixing regimes gives loose tolerances that mask real regressions.

Load Stage 2 provisional `tests/tolerances/<family>.json` and for each tensor compare the provisional (10× ref-vs-ref noise floor) against observed C++ drift. Do not rewrite from scratch — **tighten, keep, or widen each entry with explicit justification**. Recipe: `finalized = max(1.5 × observed, provisional_noise_floor, 1e-6)`. Per-tensor decision:

- `1.5 × observed < provisional`: C++ drift is at or below noise — rare, suspicious. Keep the noise floor; tighten only with user confirmation.
- `1.5 × observed ≈ provisional`: C++ is within noise. Preserve provisional.
- `1.5 × observed > provisional`: C++ introduced drift. Record the widening factor in `_comment` and name the mechanism.

**Zero-drift exception.** For tensors that are pure GGUF reads (embedding lookups, positional embedding views) or pure adds of GGUF-baked F32 weights (e.g. `embed_tokens + pos_emb`), pin both `max_abs` and `mean_abs` at **exact `0.0`** — not the `1e-6` provisional floor. Nonzero drift on such tensors indicates an unintended dtype conversion somewhere in the lookup/add path, and it IS a bug signal that must be investigated, not absorbed by a noise floor. This also applies to frontend inputs read via an env-var injection path (e.g. `enc.mel.in` when the reference mel is loaded from disk).

**Regression-detection sanity check.** After applying the 1.5× recipe, audit each entry: would a plausible 5× blowup in observed drift still pass this tolerance? If yes, the tolerance is too loose regardless of observed-justification. Tolerances MUST fail a realistic implementation regression, not just pass the current implementation. In practice: if you rounded generously to the next power-of-ten, revise down; if 1.5× already sits near the 5× blow-up threshold (observed drift was close to a power-of-ten boundary), this is fine. See `docs/porting/4a-numerical-troubleshooting.md` for the "how loose is too loose" pattern.

Remove every `_provisional: true` flag during finalization. Presence of `_provisional: true` at sign-off is a blocking error. File shape mirrors `tests/tolerances/parakeet.json` — top-of-file `_comment` array then per-tensor entries.

**The `_comment` block is the numerical-conception part.** Write it BY INSPECTING THE DRIFT PROFILE. It must name: (a) the correctness regime (dtype + KV dtype + mel source + backend, as declared above), (b) reference framework + mode (fp32 NeMo? fp16 Transformers?), (c) C++ compute dtype (CPU fp32? Metal fp16?), (d) dominant drift source (STFT precision, attention op-order, LSTM init, F16 KV round-trip, etc.), (e) attenuation pattern through the network — flat or decaying? Flat is suspect and must be named, (f) which entries were tightened/kept/widened vs. Stage 2 provisional and why for the widenings, (g) a `_suspects` sub-list naming every tensor whose finalized tolerance exceeds 5% of signal magnitude. Suspects MUST be debugged in Step 5, not silently accepted. See `tests/tolerances/parakeet.json` for the canonical shape.

**Suspects must be localized, not just flagged.** For each tensor entering the `_suspects` list, inspect the per-position/per-head drift distribution before accepting. A spread drift (every element roughly equally off) is the expected shape for fp32 matmul reduction-order noise — accept with a named mechanism. A drift concentrated on one position (e.g. pos=1 but not others) or one feature dimension (e.g. feat=73 across multiple inputs) points at a structural bug — a missed causal-mask row, a wrong position-id offset, a shape permutation. Use `np.argmax(abs(ref - cpp))` plus per-position p99 to localize before accepting. The `_comment` block MUST name the feature/position where drift concentrates for every accepted suspect. A concentrated drift pattern that cannot be attributed to a specific outlier activation is a bug signal, not a tolerance-widening signal.

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

### Step 7: Frontend parity (execute)

If validation injects reference frontend tensors to isolate later graph drift (for example `TRANSCRIBE_WHISPER_MEL_FROM_REF`), that is only an isolation tool. It does **not** validate inference-time preprocessing.

Run the family frontend on real PCM in C++ and compare the produced frontend input tensor against the reference framework's tensor. For Whisper this is:

```bash
uv run scripts/validate.py mel --family <family> --variant <variant>
```

The check must fail if production C++ mel (`enc.mel.in`) does not match the reference frontend within an explicit, measured tolerance. Do not mark Stage 4 complete if the encoder/decoder only pass by bypassing the C++ frontend with injected tensors.

### Step 8: Capability implementation (execute)

Public `transcribe_capabilities` must describe implemented runtime behavior, not just upstream model potential and not a reduced subset to dodge work. If intake/GGUF says the model supports language detection, translation, or segment timestamps, the decode path must implement those features or the port is not Stage-4 complete. The only acceptable exception is a documented upstream capability that is deliberately out of project scope for the family and has been removed from the converter/intake contract before Stage 4.

For each advertised capability, add or run a real-model smoke/API test that exercises the public ABI. Examples: no-language-hint run exercises language detection; `TRANSCRIBE_TASK_TRANSLATE` exercises translation; `TRANSCRIBE_TIMESTAMPS_AUTO`/`SEGMENT` must return segment timestamps if `max_timestamp_kind` is `SEGMENT`.

### Step 9: Full quant matrix (execute)

```bash
cmake --build build --target transcribe-quantize
uv run scripts/quantize-all.py models/<variant>/<variant>-<REFDTYPE>.gguf
```

This produces `<variant>-F16.gguf`, `<variant>-Q8_0.gguf`, `<variant>-Q6_K.gguf`, `<variant>-Q5_K_M.gguf`, `<variant>-Q4_K_M.gguf` alongside the reference-dtype GGUF.

### Step 10: Quant CLI output-validity check (execute)

For each produced GGUF, confirm the C++ runtime can load it and produce a valid transcript through `transcribe-cli`. Do **not** run tensor/numeric comparisons on quantized GGUFs for Stage 4 acceptance; quantization is intentionally lossy and activation-level drift is not a stable acceptance signal. Use ref-dtype `validate.py` for numerical parity, CLI smoke for quant runtime validity, and Stage 6 WER for user-facing quant quality.

```bash
for gguf in models/<variant>/<variant>-*.gguf; do
  out="$(build/bin/transcribe-cli -m "$gguf" samples/jfk.wav)" && \
    printf '%s\n' "$out" | grep -q '^text: .\+' && echo "OK $gguf" || echo "FAIL $gguf"
done
```

### Step 11: Sign-off

Report:
- Path to the tolerances file + confirmation the user reviewed it.
- `validate.py all` exit code at ref dtype.
- Frontend parity command + exit code (or an explicit reason no separate frontend exists).
- Capability smoke/API test command + exit code.
- Full quant matrix listing with file sizes.
- Any quant that failed CLI output-validity.

**Do not commit.**

## Postconditions

- `src/arch/<family>/` exists with the in-tree shape: `weights.{h,cpp}`, `encoder.{h,cpp}`, `decoder.{h,cpp}` as `.h`/`.cpp` pairs, plus `model.cpp` and `capabilities.cpp` backed by a single family-level header `<family>.h`.
- `tests/tolerances/<family>.json` exists, has a `_comment` block naming the correctness regime (dtype + KV dtype + mel source + backend), no `_provisional` flags remain, pure-lookup/pure-add tensors are pinned at exact `0.0`, and was reviewed by the user.
- `validate.py all --family <family> --variant <variant>` exits 0 at reference dtype in the declared regime.
- For autoregressive-decoder families: at least one `dec.logits_raw.gen<N>` (or equivalent mid-generation tensor, N ≥ 8) is dumped by both the reference and the C++ runner and gated in the tolerance file.
- Production frontend output is validated against the reference frontend when the family has an inference-time frontend.
- Every public capability advertised by the GGUF/intake is implemented by the C++ runtime and covered by a smoke/API test.
- All five derived presets present under `models/<variant>/`.
- Every produced GGUF loads and produces valid CLI output.

## Pointers (read, not execute)

- `docs/porting/ggml-reference-map.md` — ggml-pattern lookup (TOC at top). Primary Stage 4 aid.
- `docs/porting/4-numerical-validation.md` — validate.py contract and common failure modes.
- `docs/porting/4a-numerical-troubleshooting.md` — drift patterns and their usual causes.
- `docs/tools/quantization.md` — transcribe-quantize invocation.
- `src/arch/parakeet/`, `src/arch/cohere/`, `src/arch/qwen3_asr/` — working arch implementations. Use as structure references, not code to copy wholesale.
- `tests/tolerances/parakeet.json` — canonical `_comment` block shape reference.
