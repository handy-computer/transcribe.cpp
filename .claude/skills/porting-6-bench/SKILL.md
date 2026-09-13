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
  travels with the repo. The durable artifact is the **rendered table in
  `docs/models/<variant>.md`** plus the `perf:` block in
  `scripts/hf_cards/<variant>.yaml`. Transcribe the numbers into both.
- A dev box that is neither rig (for example a base `apple-m4`) is
  iteration data. It may be added as an extra card section, but it does
  **not** substitute for either required rig.
- If a rig is unreachable, Stage 6 stops at `INCOMPLETE — <rig> pending`
  and the variant carries that state into Stage 8. Only the user may sign
  off shipping with a rig missing, and the card must then say which rig
  the numbers come from.

## Standardized bench schema

Every per-cell report under
`reports/perf/<machine>/<name>_<variant>_<backend>.json` is checked by the
`required_*` sets in Step 5. Missing required fields block Stage 6;
optional gaps are surfaced but do not gate.

## Workflow

```
Bench progress:
- [ ] Step 1: Confirm full quant matrix present
- [ ] Step 2: Rebuild transcribe-bench
- [ ] Step 3: Confirm bench scope (publication default, optional widening)
- [ ] Step 4: Capture publication baseline
- [ ] Step 4b: Batch throughput sweep (good-to-know, non-gating)
- [ ] Step 5: Validate schema completeness
- [ ] Step 6: Iteration loop (human-driven, with validate gate per accept)
- [ ] Step 7: Sign-off review
```

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

The checked-in profile `asr-publication-v1` is the source of truth. It measures
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
uv run scripts/bench/run.py --profile \
  --models <variant> \
  --name <variant>-publication-baseline-$(date -u +%Y%m%dT%H%M%SZ)
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

### Step 5: Schema validation (execute)

For each report file, confirm every required field is present and
surface any optional gaps:

```python
# uv run python -c '...'
import json, pathlib, sys
required_top = {"git_sha", "machine"}
required_run = {"schema", "model_path", "backend", "per_iter",
                "sample_duration_s", "rtf_wall_mean", "transcript_sha256"}
required_iter = {"mel_ms", "encode_ms", "decode_ms", "total_ms", "wall_ms"}
optional_top = {"git_dirty"}
optional_run = {"rtf_compute_mean", "token_ids_sha256"}
for p in sys.argv[1:]:
    d = json.loads(pathlib.Path(p).read_text())
    missing_top = required_top - d.keys()
    missing_run = any(required_run - r.keys() for r in d["runs"])
    missing_iter = any(required_iter - it.keys() for r in d["runs"] for it in r["per_iter"])
    absent_opt_top = optional_top - d.keys()
    absent_opt_run = any(optional_run - r.keys() for r in d["runs"])
    print(f"{p}: required_top_missing={missing_top} required_run_missing={missing_run} "
          f"required_iter_missing={missing_iter} optional_absent_top={absent_opt_top} "
          f"optional_absent_run={absent_opt_run}")
```

Any missing **required** field is a bench-harness regression — halt Stage
6 sign-off. Absent **optional** fields are surfaced but do not block.

### Step 6: Iteration loop (human-driven)

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

### Step 7: Sign-off

Report:
- Baseline reports and machine matrix covered. Name **each of the two
  required rigs** and its state: covered, or `INCOMPLETE — pending`. Do
  not report Stage 6 as complete while either rig is missing.
- Any schema gaps observed.
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

- A sourced speed measurement for Q8_0 and Q4_K_M when downloaded, on both
  `jfk` and `dots`, for every profile machine/backend target. Legacy xRT-only
  rows may satisfy a cell but are explicitly marked and should be replaced
  during the long-form/memory sweep.
- Schema completeness reported to the user; any gap is a known bench-
  harness task, not a porting task.
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
- `scripts/bench/run.py` — driver, already discovers `build/bin/` first
- `scripts/bench/compare.py` — baseline-vs-candidate delta table
- `tools/transcribe-bench/main.cpp` — bench binary source if the schema
  needs extension
- `examples/bench/batch_bench.cpp` — `transcribe-batch-bench` source
  (offline batch throughput sweep)
- `scripts/batch_parity.py` — Stage 4 batch correctness gate (referenced
  here only to confirm correctness was already proven)
- Existing reports under `reports/perf/<machine>/` — shape references
