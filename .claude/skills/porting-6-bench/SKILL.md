---
name: porting-6-bench
description: Runs the publication performance benchmark for a ported model variant and scripts the hypothesis → change → bench → accept-or-revert loop. Use after porting-5-quants has produced the full shipped quant matrix. Input: full quant matrix at models/<variant>/, and the two publication rigs (Apple M4 Max, AMD Ryzen 7 PRO 4750U) that every shipped model card carries. Output: reports/perf/<machine>/<name>_<variant>_<backend>.json per bench run on EACH rig, scoped to the cells that ship in docs/models/<variant>.md. Every accepted performance iteration is followed by a validate.py all gate so a perf change cannot land while breaking ref-dtype numerics.
---

# porting-6-bench

Stage 6 of the porting pipeline. Runs publication-scope benchmarks,
validates report schema, and gates accepted performance changes with
`validate.py all`.

Anything beyond the publication scope (more presets, more samples,
longer iter counts) is good-to-know exploration, not a sign-off
requirement.

## Preconditions

- `models/<variant>/<variant>-<PRESET>.gguf` exists for every shipped
  preset (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M) — i.e. Stage 5 complete.
- `catalog/<variant>.json` exists, or Step 0 creates it. Nothing this stage
  measures can be published without a record to hold it.
- `build/bin/transcribe-bench` and `build/bin/transcribe-cli` are built
  under `build/`.
- `scripts/bench/run.py` is runnable.

## Build directory

Default build directory is `build/`. New ports should not introduce
additional build directories.

## Reference machine matrix (REQUIRED: both rigs)

Stage 6 is **not** complete on one machine. Every shipped model card in
`docs/models/` carries two rig sections, and they are the two rigs Stage 8
checks for:

| Rig | Card heading | Backends | `perf:` key in the HF yaml | Metadata key |
|---|---|---|---|---|
| Apple M4 Max (macOS) | `### Apple M4 Max` | `metal`, `cpu` | `m4-max` | `rtf_m4_max` |
| AMD Ryzen 7 PRO 4750U (Fedora, Vega 8 / RADV RENOIR) | `### AMD Ryzen 7 PRO 4750U` | `vulkan`, `cpu` | `ryzen-4750u` | `rtf_ryzen_4750u` |

Rules:

- **Both rigs are required for sign-off.** A single-rig bench is an
  incomplete Stage 6, not a complete one. Say so explicitly in the Step 7
  report rather than letting the missing rig pass unremarked.
- The bench must run **on** the rig. `scripts/bench/run.py` derives
  `<machine>` from the CPU brand string, so the reports land under that
  rig's own `reports/perf/<slug>/`. Vulkan cells only exist on the Ryzen
  box; Metal cells only exist on the Mac.
- `reports/` is gitignored (`.gitignore:66`), so the per-rig JSON never
  travels with the repo. The durable artifact is the speed rows in
  **`catalog/<variant>.json`**, ingested from those reports; the doc table
  and the HF card are rendered from them. Never transcribe a number by hand.
- A dev box that is neither rig (for example a base `apple-m4`) is
  iteration data. It may be added as an extra card section, but it does
  **not** substitute for either required rig.
- If a rig is unreachable, Stage 6 stops at `INCOMPLETE — <rig> pending`
  and the variant carries that state into Stage 8. Only the user may sign
  off shipping with a rig missing, and the card must then say which rig
  the numbers come from.

## Standardized bench schema

`scripts/bench/run.py` refuses to write a report under
`reports/perf/<machine>/<name>_<variant>_<backend>.json` that lacks any field
the catalog ingester or `compare.py` reads (`report_gaps()` in the driver). A
report on disk is complete by construction; a refusal is a bench-harness
regression and halts Stage 6.

## Workflow

```
Bench progress:
- [ ] Step 0: Catalog record exists (create it on a first port)
- [ ] Step 1: Confirm full quant matrix present
- [ ] Step 2: Rebuild transcribe-bench
- [ ] Step 3: Confirm bench scope (publication default, optional widening)
- [ ] Step 4: Capture publication baseline
- [ ] Step 4b: Batch throughput sweep (good-to-know, non-gating)
- [ ] Step 5: Iteration loop (human-driven, with validate gate per accept)
- [ ] Step 6: Sign-off review
```

### Step 0: Catalog record (execute, first port only)

Stage 6 is the first stage that writes to the catalog, so the record has to
exist before a report can be ingested. A model that was ported before the
catalog existed already has one; a new port does not.

```bash
ls catalog/<variant>.json || uv run scripts/catalog/new_record.py <variant> \
    --long-form <chunked-unbounded|hard-cap|soft-window> --docs-page <family>.md
```

Everything mechanical is read from artifacts Stage 5 already produced: the
intake supplies family, upstream repo and revision, and languages; the GGUFs
under `models/<variant>/` supply the download table, byte sizes, parameter
count, capability KVs, licence and display name. The two required flags are
the facts no artifact carries: which `docs/input-limits.md` bucket the family
falls into, and which page under `docs/models/` documents it. Benchmark rows
are left empty for this stage and Stage 7 to fill.

Check the seeded record before benching. `--published-repo`,
`--display-name`, `--license`, `--license-display` and `--language-tag-form`
override a wrong guess; a capability KV that the GGUF states wrongly is a
Stage 5 export bug, so fix it there and re-run rather than editing the record.

### Step 1: Matrix presence (execute)

```bash
ls models/<variant>/<variant>-*.gguf
```

Expect one file per preset in `REFERENCE_TIERS ∪ DERIVED_PRESETS` that
the variant declared. Missing files → return to Stage 5.

### Step 2: Rebuild transcribe-bench (execute)

```bash
cmake --build build --target transcribe-bench
```

`transcribe-bench` is a separate cmake target; always rebuild before
capturing baseline.

### Step 3: Bench scope (ask-point)

**Publication scope (default, required for sign-off).** This is the
matrix that ends up rendered in `docs/models/<variant>.md`:

The checked-in profile `asr-publication-v2` is the source of truth. It measures
the publication quants Q8_0 and Q4_K_M when downloaded, on both `jfk` and
`dots`, with three iterations after one warmup, using the backends assigned to
the detected publication machine (M4 Max CPU/Metal or Ryzen 4750U CPU/Vulkan).
Every selected quant/sample/target cell is required. The Russian-only GigaAM
variants currently override the defaults with their published short `ru`
fixture; add a long Russian fixture to those overrides when one is available.
Add every future mandatory sample to the profile rather than restating the
matrix as command-line flags.

Narrowed or widened sweeps are allowed for iteration, but sign-off is decided
by the profile.

### Step 4: Baseline capture (execute)

**Bench runs must be strictly serial** — never spawn concurrent
`transcribe-bench` processes. Concurrent runs contend for CPU/GPU and
pollute timings.

Publication-scope baseline (default):

```bash
uv run scripts/bench/run.py --profile --models <variant>
```

Writes one report per (variant, backend) pair to `reports/perf/<machine>/`.
Record the file paths. The final on-doc bench at sign-off should re-run
this command with `--name <variant>-publication` (matching the
reproduction command rendered in `docs/models/<variant>.md`).

### Step 4b: Batch throughput sweep (execute, good-to-know)

Optional batch throughput sweep. Correctness is already gated in Stage 4;
this step measures batch-size scaling:

```bash
cmake --build build --target transcribe-batch-bench
build/bin/transcribe-batch-bench \
  -m models/<variant>/<variant>-F16.gguf \
  samples/jfk.wav \
  --batch-sizes 1,2,4,8,16,32 --iters 3
```

Emits per-batch `{batch_size, per_utt_ms, wall_ms}` to
`reports/perf/<machine>/<name>_<variant>_batch_<backend>.json`. This is
exploratory and does not gate sign-off. Batch runs stay strictly serial.

### Step 5: Iteration loop (human-driven)

For each optimization hypothesis:

1. User states the hypothesis and expected timing/hash effect.
2. User makes the code change.
3. Skill rebuilds:
   ```bash
   cmake --build build --target transcribe-cli transcribe-bench
   ```
4. Skill re-runs the bench at the same scope used for the baseline:
   ```bash
   uv run scripts/bench/run.py \
     --models <variant> \
     --quants q8_0,q4_k_m \
     --samples jfk,dots \
     --backends metal,cpu,vulkan \
     --iters 3 --warmup 1 \
     --name "<hypothesis-slug>-$(date -u +%Y%m%dT%H%M%SZ)"
   ```
5. Skill compares baseline ↔ candidate using `scripts/bench/compare.py`:
   ```bash
   uv run scripts/bench/compare.py \
     --baseline reports/perf/<machine>/baseline-*_<variant>_<backend>.json \
     --candidate reports/perf/<machine>/<hypothesis-slug>-*_<variant>_<backend>.json
   ```
6. Skill reports timing deltas and whether `transcript_sha256` /
   `token_ids_sha256` changed. An unexpected hash change is a revert
   signal.
7. **Accepted-change validation gate (execute on every accept).** Before
   the user commits, the skill re-runs:
   ```bash
   uv run scripts/validate.py all --family <family> --variant <variant>
   ```
   If validate.py fails, revert or fix the perf change. If it passes, the
   candidate becomes the new baseline.
8. User reverts if the timing or hash signals are wrong.

Repeat until the user is satisfied.

### Step 6: Sign-off

Report:
- Baseline reports and machine matrix covered. Name **each of the two
  required rigs** and its state: covered, or `INCOMPLETE — pending`. Do
  not report Stage 6 as complete while either rig is missing.
- Total iterations run, net timing improvement, and that every accepted
  iteration passed `validate.py all`.

**Do not commit.** Bench reports under `reports/perf/` may or may not be
committed at the user's discretion.

## Catalog (mandatory exit step)

`reports/` is gitignored, so a bench report exists only on the machine that
produced it. The stage is not finished until the numbers are in the catalog.

1. Run the profile-selected publishable measurement, not a hand-written matrix:

   ```bash
   uv run scripts/bench/run.py --profile --models <variant>
   ```

   `--profile` stamps both `publication: true` and the profile id in the
   report. Hypothesis-loop runs (the optimization iterations above) MUST NOT
   carry it: on CPU they differ from the shipped figure by tens of percent,
   and the whole point is that they never reach a doc.

2. Fold them in and regenerate the tables:

   ```bash
   uv run scripts/catalog/ingest_perf.py
   uv run scripts/catalog/render.py
   uv run scripts/catalog/check.py --publication-profile --models <variant>
   ```

   `ingest_perf.py` ingests publication runs only. It refuses a row whose xRT
   would move more than 5% against what the doc already published, because
   that is a different build rather than a better reading of the same one;
   re-bench at the current sha instead of passing `--force`.

3. `check.py` must report every speed row for this variant as measured and
   sourced. A row carrying a bare xRT with no `engine_sha` is not publishable.

## Postconditions

- `catalog/<variant>.json` exists and is schema-valid.
- A sourced speed measurement for Q8_0 and Q4_K_M when downloaded, on both
  `jfk` and `dots`, for every profile machine/backend target. Legacy xRT-only
  rows may satisfy a cell but are explicitly marked and should be replaced
  during the long-form/memory sweep.
- Optimization iteration loop scripted end-to-end (user drives
  hypotheses; skill runs the loop).
- Every accepted performance iteration was followed by a passing
  `validate.py all` run.
- Wider sweeps (more presets, more samples, larger iter counts) are
  optional. They are good-to-know context and may inform follow-up work
  but do not gate Stage 6 sign-off.
- Optionally, a batch throughput sweep at
  `reports/perf/<machine>/<name>_<variant>_batch_<backend>.json` via
  `transcribe-batch-bench`. Good-to-know; informs the batched-vs-serial
  fast-path decision but does not gate sign-off.

## Pointers (read, not execute)

- `docs/porting/5-benchmarks.md` — bench procedure context
- `scripts/catalog/new_record.py` — seeds the record from intake + GGUFs
- `catalog/_schema.json` — what a record is allowed to hold
- `scripts/bench/run.py` — driver, already discovers `build/bin/` first
- `scripts/bench/compare.py` — baseline-vs-candidate delta table
- `tools/transcribe-bench/main.cpp` — bench binary source if the schema
  needs extension
- `examples/bench/batch_bench.cpp` — `transcribe-batch-bench` source
  (offline batch throughput sweep)
- `scripts/batch_parity.py` — Stage 4 batch correctness gate (referenced
  here only to confirm correctness was already proven)
- Existing reports under `reports/perf/<machine>/` — shape references
