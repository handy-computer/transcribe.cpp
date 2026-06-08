---
name: porting-4-cpp
description: Brings up the C++ implementation for a new model family at the reference dtype. Produces src/arch/<family>/*, finalized tolerances, resolved capability checks, batch parity, and full ref-dtype WER against the Oracle reference baseline. Use after porting-3-convert. Quant generation is Stage 5, not here.
---

# porting-4-cpp

Stage 4 of the porting pipeline. Implements `src/arch/<family>/*`, authors
`tests/tolerances/<family>.json` from the first honest C++ drift, runs
the family-doc Capability Validation table, and ends with a full
ref-dtype WER gate (batch 1 + batch 8). Quant generation moved to Stage 5;
this stage stays ref-dtype only.

First principle: the reference implementation is the source of truth. Open
it before writing C++. Re-open it when debugging drift.

## Preconditions

- `reports/porting/<family>/<variant>/intake.json` schema-valid, Preflight
  Gate A green.
- `docs/porting/families/<family>.md` has a `## Capability Validation`
  table drafted at intake time (one row per advertised capability), with
  every row's `Target` filled and user-signed (`MUST PASS` /
  `OUT OF SCOPE — <reason>`) and `Status: TODO`. If any `Target` is still
  blank, send the port back to `porting-1-intake` for the scope sign-off
  before implementing.
- `build/validate/<family>/<variant>/dump_coverage.json` exists.
- `models/<variant>/<variant>-<REFDTYPE>.gguf` exists, Preflight Gate B
  green.
- `tests/golden/<family>/<variant>.manifest.json` populated.
- `build/bin/transcribe-cli` is buildable.
- `hf` authenticated for the target org (Step 10 pushes the ref-dtype
  GGUF to a private repo so Step 11's WER gate can run on Modal). Local-
  only WER runs do not need this.

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
- [ ] Step 9: Batch parity (offline run_batch vs serial)
- [ ] Step 10: Publish ref-dtype to private HF repo (enables remote WER)
- [ ] Step 11: Full ref-dtype WER gate (batch 1 + batch 8)
- [ ] Step 12: Sign-off review
```

### Step 0: Family-level forward-map.md (execute)

Before writing C++, create or update `reports/porting/<family>/forward-map.md`
from `docs/porting/families/_forward-map-template.md`. This is a planning
artifact — short, family-level, one page-ish. Most variants share the
same forward structure. Variant-specific differences go into the
"Variant Notes" section only when they affect graph shape, control flow,
capabilities, or validation coverage.

For a new family, write the map before C++ work; `TODO` / `UNKNOWN` rows
are allowed during bring-up. For a new variant in an existing family,
update it only when architecture or validation coverage changes.

```bash
[ -f reports/porting/<family>/forward-map.md ] || \
  cp docs/porting/families/_forward-map-template.md \
     reports/porting/<family>/forward-map.md
```

Keep the map short. Repeated layers map the block pattern once. The
"no unresolved rows" gate applies only at Stage 4 sign-off.

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

**Family-specific requirements.** Capture any implementation work that
does not flow through convert/validate in
`docs/porting/families/<family>.md`, and confirm the list with the user
at Stage 4 start.

**Mid-generation tensor coverage (autoregressive decoders).** Families
with KV-cache decoders (`encoder-decoder`, `audio-llm`) MUST dump at least
one `n_past > 0` step tensor on both sides, conventionally
`dec.logits_raw.gen<N>` after N ≥ 8 greedy steps. If coverage is missing,
add it to the reference dumper and C++ runner.

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

The `_comment` block must name the correctness regime, reference mode,
C++ compute dtype, dominant drift source, and any entries widened vs
Stage 2 provisional.

Inspect localized drift before accepting it; do not absorb likely mask,
position, or shape bugs into wider tolerances.

### Step 5: Human review of tolerances

Hand the file to the user. The user reads the `_comment` block and
either:
- Accepts as-is (signs off).
- Pushes back — return to Step 2 to debug, NOT to Step 4 to loosen
  tolerances.

Loose tolerances without a named mechanism must be debugged before
acceptance.

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
CLI/API command that exercises it, the expected observable, a `Target`
(the user-signed scope decision from Stage 1), and a `Status`. At intake
time the `Status` is `TODO`; this step fills it in. Do not edit the
`Target` column here.

For each row:

1. Run the listed command against the ref-dtype GGUF.
2. Compare the runtime output against the expected observable.
3. Update the row's `Status` cell, honoring its `Target`:
   - `PASS` — command ran and the observable matched.
   - `SKIP — not exposed by runtime` — the capability is advertised but
     the C++ CLI/API does not expose a way to observe it. The row stays
     in the table; users reading the family doc see the gap honestly.
     **Only legal when `Target` is `OUT OF SCOPE`.**
   - `ACCEPTED GAP — <reason>` — the capability is exposed but
     intentionally not exercised in this port (e.g. requires audio
     samples we don't ship). The reason must name what unblocks the
     row later. **Only legal when `Target` is `OUT OF SCOPE`** (plus the
     batch serial-fallback exception in Step 9).

A `Target: MUST PASS` row must resolve to `PASS` unless the user re-signs
it as `OUT OF SCOPE`. `SKIP` / `ACCEPTED GAP` are legal only on
`OUT OF SCOPE` rows, plus the conditional batch serial-fallback exception in
Step 9 (only when a real `run_batch()` benchmarked no faster than serial).
No row may remain `TODO`. If `capabilities.streaming: true`, the
streaming row is forced `MUST PASS`; a blocked streaming implementation
requires explicit user sign-off as a blocker. The `Batch (offline)` row is
likewise forced `MUST PASS` for every family: a real `run_batch()` parallel
path ships (Step 9), and serial fallback is legal only with Stage-6 bench
evidence that parallel is no faster than serial after optimization.

### Step 9: Batch parity (execute)

Offline batching is a throughput feature. In the Stage 4 CPU regime,
batched output must match the single-utterance path.

Text parity (byte-identical hypotheses, serial vs batched), and freeze
the golden baseline that later changes gate against:

```bash
uv run scripts/batch_parity.py \
  --model models/<variant>/<variant>-<REFDTYPE>.gguf \
  --samples-dir samples/wer/<dataset> \
  --batch-sizes 2,4,8 --backend cpu \
  --golden-out tests/golden/batch/<variant>.cpu.json
```

Tensor parity on the per-utterance encoder output (bit-exact on CPU; the
GPU flash-attention path is expected to drift ~1e-3 and is not the Stage
4 regime):

```bash
uv run scripts/batch_tensor_parity.py \
  --model models/<variant>/<variant>-<REFDTYPE>.gguf \
  --samples-dir samples/wer/<dataset> --batch-size 4 --backend cpu
```

Batching **`MUST PASS`** - Every family MUST implement an explicit
`run_batch()` parallel path. Add/update the `Batch (offline)` Capability Validation
row:
- `PASS` — the family implements an explicit `run_batch()` fast path AND its
  output is within 0.01% of WER of batch size 1, over a large set
- `ACCEPTED GAP — serial (benchmarked no faster)` — legal ONLY when a real
  `run_batch()` WAS implemented and the Stage-6 batch-throughput sweep
  (`transcribe-batch-bench`) measured it at **no faster than the serial
  loop** on the target backend(s).
- A missing `run_batch()` (serial fallback with no parallel path written) is
  **NOT** an accepted gap — it is a `BLOCKER` requiring explicit user
  sign-off, the same bar as a blocked streaming implementation.

A text or tensor parity mismatch is a batching bug, never an accepted gap.

### Step 10: Publish ref-dtype to private HF repo (execute)

Push the ref-dtype GGUF to the private HF repo so the Step 11 WER gate
can run on Modal (and so Stage 5 can extend the same repo with the full
quant matrix). Local-only WER runs do not need this, but Modal does.

```bash
hf repo create <org>/<variant>-gguf --repo-type model --private  # if absent
hf upload <org>/<variant>-gguf models/<variant>/<variant>-<REFDTYPE>.gguf --repo-type model
```

### Step 11: Full ref-dtype WER gate (execute)

Score the **full** acceptance manifest on the C++ ref-dtype GGUF at batch
1 and batch 8, and gate against the Stage-2 Oracle reference WER
(`reports/wer/<variant>-REF.<dataset>.{jsonl,score.json}`) — our own
reference run, not a published number. Run it on Modal if credentials are
available; otherwise run locally:

```bash
# local
for B in 1 8; do
  uv run scripts/wer/run.py --model models/<variant>/<variant>-<REFDTYPE>.gguf \
    --manifest "$MANIFEST" --batch-size $B \
    --out reports/wer/<variant>-<REFDTYPE>.<dataset>.b$B.jsonl
  uv run scripts/wer/score.py reports/wer/<variant>-<REFDTYPE>.<dataset>.b$B.jsonl
done
# or Modal: modal run scripts/wer/remote/modal_sweep.py::sweep \
#   --models <repo-or-card> --quants <REFDTYPE>
```

Gate (batch 1): `C++ ref-dtype WER ≤ Oracle reference WER + 0.01pp`
(percentage points, NOT fraction). The Oracle WER is reported in percent
form (e.g. `error_rate_pct: 17.88`).

Scores deviating more than this in either direction is a is a blocker.
You will want to diff utterances versus the reference. 
Oracle JSONL with `scripts/wer/compare.py` and investigate.

Batch 1 vs batch 8 is human-reviewed. A WER delta beyond dataset noise
(~0.01pp) is a potetnail batching bug and needs to be flagged and 
explcitly signed off by a human.

If the Oracle reference baseline is missing, that is a `porting-2-oracle`
gap. Send it back rather than gating against a published number.

### Step 12: Sign-off

Report:
- Family forward-map path.
- Path to the tolerances file + confirmation the user reviewed it.
- `validate.py all` exit code at ref dtype.
- Frontend parity command + exit code (or explicit reason no separate
  frontend exists).
- Capability Validation table outcome: per-row `Target` → `Status`
  (PASS / SKIP / ACCEPTED-GAP), with the family-doc path so the user can
  read it. Confirm every `MUST PASS` row resolved to `PASS`; flag any
  that did not as a blocker (or an explicit user-signed scope change to
  `OUT OF SCOPE`). If the model streams natively, the streaming row is
  PASS or an explicit user-signed BLOCKER — never a silent gap.
- Batch parity: text + tensor parity result, the golden fixture path
  (`tests/golden/batch/<variant>.cpu.json`), and the `Batch (offline)`
  row status (PASS or ACCEPTED GAP — serial fallback).
- Private HF repo URL where the ref-dtype GGUF was pushed (Step 10).
- Full ref-dtype WER: Oracle reference WER, C++ batch-1 WER, max allowed,
  pass/blocked; plus the batch-8 WER and the user's sign-off that batching
  is WER-neutral.

**Do not commit.** Quant generation and CLI smokes are Stage 5
(`porting-5-quants`).

## Postconditions

- `reports/porting/<family>/forward-map.md` has no unresolved `TODO` /
  `UNKNOWN` / `accepted_gap` rows.
- `src/arch/<family>/` follows the in-tree family shape and is wired into
  the arch dispatch.
- `tests/tolerances/<family>.json` is finalized: `_comment` names the
  correctness regime, no `_provisional` flags remain, pure-read/pure-add
  tensors are exact `0.0`, and the user reviewed it.
- `validate.py all --family <family> --variant <variant>` exits 0 at
  reference dtype in the declared regime.
- Autoregressive-decoder families dump and gate at least one
  `dec.logits_raw.gen<N>` with `N >= 8`.
- Production frontend output is validated against the reference frontend
  when the family has an inference-time frontend.
- Capability Validation has no `TODO`; every `MUST PASS` row is `PASS`
  unless the user explicitly re-signed scope.
- If `capabilities.streaming` is true, the streaming row is `PASS` or an
  explicit user-signed `BLOCKER`.
- Batch parity is gated: an explicit `run_batch()` parallel path ships, text
  + CPU tensor checks pass at 2/4/8, the golden fixture exists, and the
  family doc records `Batch (offline)` as `PASS`.
- Full ref-dtype WER ran on the complete acceptance manifest at batch 1
  and batch 8; batch 1 passes the `Oracle reference WER + 0.01pp`
  (percentage points, NOT fraction) gate, and batch 8 is user-reviewed as
  WER-neutral.
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
