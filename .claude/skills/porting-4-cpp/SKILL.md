---
name: porting-4-cpp
description: Brings up the C++ implementation for a new model family at the reference dtype, finalizes per-tensor tolerances from observed drift, runs the family-doc Capability Validation table commands, and ends with a 512-utterance subset WER sanity vs the reference framework. Use after porting-3-convert has produced the reference-dtype GGUF. Input: intake.json, manifest, dump_coverage.json, provisional tolerances, reference-dtype GGUF, family forward-map.md, family doc Capability Validation table. Output: src/arch/<family>/* source, family-level reports/porting/<family>/forward-map.md, finalized tests/tolerances/<family>.json (no _provisional flags, _comment block explaining drift sources), validate.py all green at ref dtype, Capability Validation table updated with PASS/SKIP/ACCEPTED-GAP per row, and Stage 4 subset WER within 0.005 of reference. Quant generation is Stage 5, not here.
---

# porting-4-cpp

Stage 4 of the porting pipeline. Implements `src/arch/<family>/*`, authors
`tests/tolerances/<family>.json` from the first honest C++ drift, runs
the family-doc Capability Validation table, and ends with a deterministic
subset WER sanity. Quant generation moved to Stage 5; this stage stays
ref-dtype only.

First principle: the reference implementation is the source of truth. Open
it before writing C++. Re-open it when debugging drift.

## Preconditions

- `reports/porting/<family>/<variant>/intake.json` schema-valid, Preflight
  Gate A green.
- `docs/porting/families/<family>.md` has a `## Capability Validation`
  table drafted at intake time (one row per advertised capability).
- `build/validate/<family>/<variant>/dump_coverage.json` exists.
- `models/<variant>/<variant>-<REFDTYPE>.gguf` exists, Preflight Gate B
  green.
- `tests/golden/<family>/<variant>.manifest.json` populated.
- `build/bin/transcribe-cli` is buildable.

## Workflow

```
CPP progress:
- [ ] Step 0: Family-level forward-map.md
- [ ] Step 1: Sibling-variant shortcut
- [ ] Step 2: Implement src/arch/<family>/ (open-ended)
- [ ] Step 3: First validate.py run (against provisional tolerances)
- [ ] Step 4: Finalize tolerances (recipe below, replaces provisional)
- [ ] Step 5: Human review of finalized tolerances
- [ ] Step 6: validate.py all green at ref dtype
- [ ] Step 7: Verify frontend parity for production inference
- [ ] Step 8: Run the family-doc Capability Validation table
- [ ] Step 9: Stage 4 subset WER sanity
- [ ] Step 10: Sign-off review
```

### Step 0: Family-level forward-map.md (execute)

Before writing C++, create or update `reports/porting/<family>/forward-map.md`
from `docs/porting/families/_forward-map-template.md`. This is a planning
artifact — short, family-level, one page-ish. Most variants share the
same forward structure. Variant-specific differences go into the
"Variant Notes" section only when they affect graph shape, control flow,
capabilities, or validation coverage.

For a new family, write the map before C++ work, but it can have `TODO`
or `UNKNOWN` rows during bring-up — open questions are how the map earns
its keep. For a new variant in an existing family, read the existing map
and update it only if the variant changes the architecture.

```bash
[ -f reports/porting/<family>/forward-map.md ] || \
  cp docs/porting/families/_forward-map-template.md \
     reports/porting/<family>/forward-map.md
```

The map is short. Repeated layers map the block pattern once and list
the first/middle/last gate tensors — not one row per layer. The
"no unresolved rows" gate applies only at Stage 4 sign-off (Step 10),
not while you are working through the implementation.

### Step 1: Sibling-variant shortcut (execute)

If `src/arch/<family>/` already exists (a prior variant was ported) and the
new variant declares the same `architecture_pattern`:

```bash
uv run scripts/validate.py all --family <family> --variant <variant>
```

**If it exits 0**, jump to Step 7 (frontend parity) and continue from
there — Steps 0/2/3/4/5/6 are already paid for. Existing tolerances hold.

**If it fails**, the new variant exposes an assumption that didn't hold
(often a config field in `config.varying_across_variants`). Continue to
Step 2 — fix the `src/arch/<family>/` code that assumed the old shape, and
update the forward-map's "Variant Notes" section.

### Step 2: Implement src/arch/<family>/ (open-ended)

For a new family, create `src/arch/<family>/` with these files, following
the shape of the closest existing family:
- `weights.{h,cpp}`, `encoder.{h,cpp}`, `decoder.{h,cpp}` — one `.h`/`.cpp`
  pair each.
- `model.cpp` and `capabilities.cpp` — implementation only. The
  corresponding types and declarations live in a single family-level
  header `<family>.h` (see `parakeet/parakeet.h`, `cohere/cohere.h`,
  `qwen3_asr/qwen3_asr.h`), which declares the concrete `*Model` /
  `*Context` subclasses and the `apply_family_invariants` entry point. No
  separate `model.h` or `capabilities.h`.

Pick the closest existing family by architecture pattern:
- `encoder-transducer` → `src/arch/parakeet/`
- `encoder-decoder` → `src/arch/cohere/`
- `audio-llm` → `src/arch/qwen3_asr/`
- `encoder-ctc` → no in-tree reference yet; combine parakeet's encoder
  with a minimal CTC head.

**Use `docs/porting/ggml-reference-map.md` as the primary ggml-pattern
lookup** (TOC at top of file). Read the relevant subsections only, not the
whole file. Load the closest family's `src/arch/<family>/` as a working
example.

Wire the new family into the arch dispatch in `src/` per the existing
family wiring. Build iteratively:

```bash
cmake --build build --target transcribe-cli
```

Surface to the user any ggml-pattern decisions they should make explicitly
rather than silently (see "Human Decisions" in the parent CLAUDE.md).

**Family-specific requirements.** Some families ship implementation work
beyond the arch pattern itself — e.g. whisper must support loading
whisper.cpp-compatible `.bin` files on the loader side. These extras do
NOT flow through intake → convert → validate (our GGUF remains the
canonical numerical reference), but they are real C++ work for this stage.
Capture each one in `docs/porting/families/<family>.md` under a
"Family-specific implementation notes" section, and confirm with the user
at the start of Stage 4 that all such items are listed.

**Mid-generation tensor coverage (autoregressive decoders).** Families
whose decoder maintains a KV cache (`encoder-decoder`, `audio-llm`) MUST
include at least one dump point that exercises the `n_past > 0` step-graph
code path. The prompt pass (`n_past=0, n_tokens=seq_len`) does not cover
cache write/read offsets, causal-mask indexing for a single new token, or
position-id handling past the prompt — a bug in any of those can pass
every prompt-pass tolerance and still corrupt the model mid-transcription.
Convention: a tensor named `dec.logits_raw.gen<N>` (or equivalent for the
family's logit shape), captured after N ≥ 8 completed step-loop iterations
on BOTH sides, with both sides running matching greedy rules. If
`dump_coverage.json` doesn't list a mid-generation tensor, add it here in
both the reference dumper and the C++ runner, and file the missing
coverage back to `porting-2-oracle` for the next family.

### Step 3: First validate.py run (execute)

```bash
uv run scripts/validate.py all --family <family> --variant <variant>
```

Runs against the **provisional tolerances** that `porting-2-oracle` wrote.
Because the provisional tolerances are loose, the comparison completes
end-to-end and the report lists per-tensor observed `max_abs` / `mean_abs`
between C++ and reference. **This is the raw material for Step 4.**

If validate.py fails for reasons other than loose tolerances (C++ crashes,
shape mismatches, missing tensor dumps), those are real bugs — fix them
before finalizing tolerances. Finalizing on top of broken C++ papers over
the problem.

**Measurement hygiene.** Every drift number that lands in the tolerance
file MUST come from a `validate.py all` run with the default flags. Do not
copy numbers from direct `transcribe-cli` invocations — they may be
running a different compute path. If you need to probe a specific
configuration (e.g. `--kv-type f32` vs `--kv-type f16`), measure it via
`validate.py` and label the measurement explicitly — never fold it into
the reference-regime numbers.

### Step 4: Finalize tolerances (execute)

**The correctness regime.** Stage 4 validates C++ against reference in a
single fixed regime:

- reference-dtype GGUF (F32 / F16 / BF16 per the intake);
- KV cache dtype matching the weight dtype (AUTO should resolve this; if
  the family's AUTO policy defaults to F16 on an F32 model, fix the policy
  or run validation with explicit `--kv-type f32`);
- reference mel (injected via the family's mel-override env var if the C++
  mel frontend hasn't been implemented yet);
- backend/threads chosen for determinism (`--backend cpu --threads 1` is
  the conservative default).

**Every tolerance number in this file is measured in that regime.**
Production variants — quantized GGUFs, F16 cache on an F32 model, the C++
mel frontend, GPU backends — produce different (usually wider) drift.
Those measurements belong in Stage 6/7 reports, NOT in this file.

Load Stage 2 provisional `tests/tolerances/<family>.json` and for each
tensor compare the provisional (magnitude-aware: `1e-4 × p99_abs` /
`1e-5 × rms`) against observed C++ drift. **Tighten, keep, or widen
each entry with explicit justification.** Recipe:
`finalized = max(1.5 × observed, provisional_magnitude_budget, 1e-6)`.

- `1.5 × observed < provisional`: C++ drift is well below the magnitude
  budget — common on simple tensors. Keep the magnitude budget;
  tighten only with user confirmation.
- `1.5 × observed ≈ provisional`: C++ within budget. Preserve provisional.
- `1.5 × observed > provisional`: C++ drift exceeds the magnitude
  budget. Record the widening factor in `_comment` and name the
  mechanism (BLAS accumulation order, frontend FFT precision,
  bf16 vs fp32 storage, etc.).

**Zero-drift exception.** For tensors that are pure GGUF reads (embedding
lookups, positional embedding views) or pure adds of GGUF-baked F32
weights (e.g. `embed_tokens + pos_emb`), pin both `max_abs` and `mean_abs`
at **exact `0.0`** — not the `1e-6` provisional floor. Nonzero drift on
such tensors indicates an unintended dtype conversion somewhere in the
lookup/add path; investigate, do not absorb. Same applies to frontend
inputs read via env-var injection (e.g. `enc.mel.in` when reference mel
loaded from disk).

Remove every `_provisional: true` flag during finalization. Any remaining
`_provisional` at sign-off is a blocking error. File shape mirrors
`tests/tolerances/parakeet.json`.

**The `_comment` block is the numerical-conception part.** Write it BY
INSPECTING THE DRIFT PROFILE. It must name: (a) the correctness regime
(dtype + KV dtype + mel source + backend), (b) reference framework + mode,
(c) C++ compute dtype, (d) dominant drift source (STFT precision,
attention op-order, LSTM init, F16 KV round-trip), (e) which entries
were tightened/kept/widened vs Stage 2 provisional and why for each
widening.

**Suspicious localized drift.** If a tensor's drift is concentrated on
one position or feature dimension (rather than spread across all
elements like fp32 reduction-order noise typically is), that is a
structural-bug signal — a missed causal-mask row, a wrong position-id
offset, a shape permutation. Inspect it before accepting; do not absorb
it into a wider tolerance. Spread drift can be widened with a named
mechanism in `_comment`.

### Step 5: Human review of tolerances

Hand the file to the user. The user reads the `_comment` block and
either:
- Accepts as-is (signs off).
- Pushes back — return to Step 2 to debug, NOT to Step 4 to loosen
  tolerances.

**Silent acceptance is how bugs hide.** Loose tolerances without a named
mechanism must be debugged before they are accepted.

### Step 6: Ref-dtype validation green (execute)

```bash
uv run scripts/validate.py all --family <family> --variant <variant>
```

Must exit 0. If it fails after tolerances were accepted, either the
tolerances are genuinely too tight or C++ drift is flaky across runs —
investigate.

### Step 7: Frontend parity (execute)

If validation injects reference frontend tensors to isolate later graph
drift (for example `TRANSCRIBE_WHISPER_MEL_FROM_REF`), that is only an
isolation tool. It does **not** validate inference-time preprocessing.

Run the family frontend on real PCM in C++ and compare the produced
frontend input tensor against the reference framework's tensor. For
Whisper:

```bash
uv run scripts/validate.py mel --family <family> --variant <variant>
```

The check must fail if production C++ mel does not match the reference
frontend within an explicit, measured tolerance. Do not mark Stage 4
complete if the encoder/decoder only pass by bypassing the C++ frontend
with injected tensors.

### Step 8: Run the family-doc Capability Validation table (execute)

Open `docs/porting/families/<family>.md` and find the
`## Capability Validation` table. Each row names an advertised
capability, the mode (explicit-language, auto/no-hint, etc.), the actual
CLI/API command that exercises it, the expected observable, and a
status. At intake time the status is `TODO`; this step fills it in.

For each row:

1. Run the listed command against the ref-dtype GGUF.
2. Compare the runtime output against the expected observable. The
   observables are deliberately loose ("non-empty plausible transcript,"
   "timestamp output present") because the runtime can only observe what
   the public CLI/API exposes — do not invent assertions the runtime
   cannot actually surface.
3. Update the row's `Status` cell to one of:
   - `PASS` — command ran and the observable matched.
   - `SKIP — not exposed by runtime` — the capability is advertised but
     the C++ CLI/API does not expose a way to observe it. The row stays
     in the table; users reading the family doc see the gap honestly.
   - `ACCEPTED GAP — <reason>` — the capability is exposed but
     intentionally not exercised in this port (e.g. requires audio
     samples we don't ship). The reason must name what unblocks the
     row later.

Every advertised capability must end up with PASS, SKIP, or ACCEPTED
GAP — not TODO. If a row that should be PASS instead fails the runtime
check, that is a real regression: fix the C++ before signing off Stage
4, do not downgrade the row.

This step replaces the older intake-resident behavior cases. The signal
"we implemented tensor parity but forgot user-visible behavior" is
caught by the table being incomplete or full of SKIP rows that the user
expected to be PASS. Do not introduce a `tests/contract/<family>/`
framework unless a family genuinely needs a bespoke harness.

### Step 9: Stage 4 subset WER sanity (execute)

Run a small subset of the acceptance manifest on the C++ ref-dtype GGUF
and the reference framework, then compare WERs.

```bash
# Take the first 512 manifest rows in order. Same source manifest →
# same subset every time; re-runs are no-ops once the file exists.
uv run scripts/wer/subset.py \
  --manifest samples/wer/<dataset>.manifest.jsonl \
  --n 512 \
  --out samples/wer/<dataset>.512.manifest.jsonl

uv run scripts/wer/run.py \
  --model models/<variant>/<variant>-<REFDTYPE>.gguf \
  --manifest samples/wer/<dataset>.512.manifest.jsonl \
  --out reports/wer/<variant>-<REFDTYPE>.<dataset>-512.jsonl
uv run scripts/wer/score.py reports/wer/<variant>-<REFDTYPE>.<dataset>-512.jsonl
```

Run the reference framework on the same subset file (per-family driver)
and record the resulting WER alongside the C++ score.

**Hard gate**: `abs(cpp_wer - ref_wer) <= 0.005` on the same subset
file. Both runners must score against the exact same `.512.manifest.jsonl`
path; reporting the path + utterance count in sign-off is enough — no
SHA enforcement.

If the dataset has fewer than 512 utterances, the subset is the full
manifest. If the reference runner is not yet wired up for this dataset,
record an accepted gap in the family doc and rely on tensor validation
+ the Capability Validation table for Stage 4 acceptance. Do **not**
compare the 512-utterance subset against a public full-dataset upstream
number — the subsets are not the same.

Also surface the worst per-utterance transcript differences for human
review.

### Step 10: Sign-off

Report:
- Family forward-map path.
- Path to the tolerances file + confirmation the user reviewed it.
- `validate.py all` exit code at ref dtype.
- Frontend parity command + exit code (or explicit reason no separate
  frontend exists).
- Capability Validation table outcome: per-row PASS / SKIP /
  ACCEPTED-GAP, with the family-doc path so the user can read it.
- Subset WER C++ vs reference and the |delta|.

**Do not commit.** Quant generation and CLI smokes are Stage 5
(`porting-5-quants`).

## Postconditions

- `reports/porting/<family>/forward-map.md` exists with no unresolved
  `TODO` / `UNKNOWN` / `accepted_gap` rows.
- `src/arch/<family>/` follows the in-tree shape: `weights.{h,cpp}`,
  `encoder.{h,cpp}`, `decoder.{h,cpp}` as `.h`/`.cpp` pairs, plus
  `model.cpp` and `capabilities.cpp` backed by a single family-level
  header `<family>.h`.
- `tests/tolerances/<family>.json` has a `_comment` block naming the
  correctness regime, no `_provisional` flags remain, pure-lookup/pure-add
  tensors are pinned at exact `0.0`, and was reviewed by the user.
- `validate.py all --family <family> --variant <variant>` exits 0 at
  reference dtype in the declared regime.
- For autoregressive-decoder families: at least one
  `dec.logits_raw.gen<N>` (N ≥ 8) is dumped and gated.
- Production frontend output is validated against the reference frontend
  when the family has an inference-time frontend.
- The family-doc Capability Validation table has every row updated to
  PASS, SKIP — not exposed by runtime, or ACCEPTED GAP — `<reason>`.
  No row remains TODO.
- Stage 4 subset WER vs reference on the same 512-row subset file is
  within 0.005 absolute. Sign-off names the subset path and utterance
  count.
- No quantized GGUFs are produced here (Stage 5 owns that).

## Pointers (read, not execute)

- `docs/porting/ggml-reference-map.md` — ggml-pattern lookup (TOC at top).
- `docs/porting/4-numerical-validation.md` — validate.py contract and
  common failure modes.
- `docs/porting/4a-numerical-troubleshooting.md` — drift patterns and
  their usual causes.
- `docs/porting/families/_forward-map-template.md` — Step 0 template.
- `src/arch/parakeet/`, `src/arch/cohere/`, `src/arch/qwen3_asr/` —
  working arch implementations.
- `tests/tolerances/parakeet.json` — canonical `_comment` block shape
  reference.
