---
name: porting-review
description: Reviews one stage of a port against its postconditions and produces a structured review report. Invoked as porting-review with stage=N (1-8), family, and variant arguments. Use at the end of any porting stage to double-check artifacts before committing, or out-of-band to re-review a stage without re-running it. Input: the stage number plus family and variant. Output: a markdown review report to stdout listing per-check pass/fail status, any artifacts that violate postconditions, and concrete remediation steps.
---

# porting-review

Cross-cutting review skill. Reads the artifacts produced by a given
porting stage, checks them against that stage's postconditions, and
emits a pass/fail/warn report. Does not modify any artifact — strictly
read-only.

## Preconditions

- `family` and `variant` identify an in-progress or completed port.
- `stage` is an integer 1–8 identifying which stage to review.
- The relevant artifacts for that stage are expected to exist; their
  absence is itself a review finding.

## Workflow

```
Review progress:
- [ ] Step 1: Parse arguments (stage, family, variant)
- [ ] Step 2: Load the stage's checklist
- [ ] Step 3: Run every check (read-only)
- [ ] Step 4: Emit the markdown review report
```

### Step 1: Arguments

Required: `stage` (1–8), `family`, `variant`. Fail fast if any is
missing or the stage is out of range.

### Step 2: Stage checklist

Each stage has a fixed checklist. The skill applies the one for the
requested stage.

**Stage 1 — porting-1-intake**
- [ ] `reports/porting/<family>/<variant>/intake.json` exists.
- [ ] Validates against `docs/porting/families/_intake-schema.json`.
- [ ] Human-judgment fields (`reference_framework`,
      `reference_rationale`, `architecture_pattern`, `known_risks`) are
      all non-null.
- [ ] `upstream_benchmarks[0]` has a non-null `score` and a `dataset`
      (LibriSpeech test-clean, or an explicit alternate if the model
      isn't English).
- [ ] `docs/porting/families/<family>.md` exists and contains a
      `## Capability Validation` table with one row per advertised
      capability. Rows for capabilities the runtime cannot observe stay
      in the table and resolve to SKIP at Stage 4. Stage 1 drafts the
      rows; at Stage 1 review, every row's status may still be `TODO`.
- [ ] `intake_gaps` is either empty or every entry has an explanation.
- [ ] Preflight Gate A passes:
      `uv run scripts/preflight.py --family <family> --variant <variant> --gate A`
      exits 0.
- [ ] `docs/porting/families/<family>.md` exists (may still be a draft).

**Stage 2 — porting-2-oracle**
- [ ] `build/validate/<family>/<variant>/dump_coverage.json` exists and
      `tensors[]` is non-empty.
- [ ] Every entry in `tensors[]` has a matching `.f32` + `.json` pair
      under `build/validate/<family>/<variant>/<case>/<stage>/ref/`.
- [ ] A parallel `ref2/` directory exists alongside each `ref/` with
      the same tensor set — the ref-vs-ref stability pass. Missing
      `ref2/` means the noise floor was not measured and the
      tolerances will be arbitrary.
- [ ] Every `.json` sidecar carries `rms` and `p99_abs` alongside
      `min`/`max`/`mean` (the shared `ref_dump.write_tensor` helper
      writes them; their absence indicates a stale dumper).
- [ ] For every manifest case whose decode pass produces text or
      tokens, `build/validate/<family>/<variant>/<case>/<stage>/ref/transcript.json`
      exists. (Optional when the dumper does not expose a transcript
      helper for that decode path.)
- [ ] `tests/golden/<family>/<variant>.manifest.json` exists, names
      `reference.entrypoint`, and its `cases[]` lists the audio cases
      the dumper actually ran (typically `samples/jfk.wav` plus any
      family-specific extras).
- [ ] `tests/tolerances/<family>.json` exists. Every tensor entry
      carries `_provisional: true`. Top-of-file `_comment` array names
      ref-vs-ref stability (noise floor) as the derivation source.
      Provisional numbers derive from `max(10 × noise, 1e-6)` per
      tensor.

**Stage 3 — porting-3-convert**
- [ ] `models/<variant>/<variant>-<REFDTYPE>.gguf` exists, where
      REFDTYPE matches intake's `dtype.expected`.
- [ ] `reports/convert/<variant>-<REFDTYPE>.json` exists and records
      `gguf_sha256`, `source_hf_repo`, `source_hf_revision`.
- [ ] Preflight Gate B passes:
      `uv run scripts/preflight.py --family <family> --variant <variant> --gate B`
      exits 0.
- [ ] Loader smoke (split by case):
      - If `src/arch/<family>/` does NOT exist yet (new family): Gate B
        + parseable GGUF metadata is sufficient. The loader returns
        `UNSUPPORTED_ARCH`; record as pass-with-note.
      - If `src/arch/<family>/` already exists: `build/bin/transcribe-cli
        -m <gguf> samples/jfk.wav` must exit 0. Missing is a **FAIL**.
- [ ] No quantized GGUFs (F16/Q8_0/...) exist yet unless the source was
      already F16 or BF16 — quants are Stage 5 territory.

**Stage 4 — porting-4-cpp**
- [ ] `reports/porting/<family>/forward-map.md` exists with no
      remaining `TODO`, `UNKNOWN`, or `accepted_gap` rows that the user
      did not explicitly accept.
- [ ] `src/arch/<family>/` follows the in-tree shape: `weights.{h,cpp}`,
      `encoder.{h,cpp}`, `decoder.{h,cpp}` as paired `.h`/`.cpp`, plus
      `model.cpp` and `capabilities.cpp` backed by a single family-
      level header `<family>.h`. No separate `model.h` or
      `capabilities.h`.
- [ ] `tests/tolerances/<family>.json` has a `_comment` block (JSON
      array of strings) explaining drift sources. No tensor entry
      carries `_provisional: true`.
- [ ] `uv run scripts/validate.py all --family <family> --variant <variant>`
      exits 0 at reference dtype.
- [ ] If the family has an inference-time frontend, production frontend
      parity is validated:
      `uv run scripts/validate.py mel --family <family> --variant <variant>`
      exits 0.
- [ ] The family doc's `## Capability Validation` table is up to date:
      every advertised capability has a row with `Status` set to PASS,
      SKIP — not exposed by runtime, or ACCEPTED GAP — `<reason>`. No
      row remains TODO. Any advertised capability with no row at all is
      a FAIL — add it or downgrade the capability claim.
- [ ] Stage 4 subset WER sanity passes: |cpp_wer - ref_wer| ≤ 0.005 on
      the same 512-row subset file
      (`samples/wer/<dataset>.512.manifest.jsonl`). Sign-off names the
      subset path and utterance count — no SHA enforcement.
- [ ] No quantized GGUFs were produced in Stage 4. They are a Stage 5
      concern.

**Stage 5 — porting-5-quants**
- [ ] All five derived presets (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M) exist
      under `models/<variant>/`, with the source tier suffix skipped
      where it duplicates one of them.
- [ ] Every produced GGUF loads via
      `build/bin/transcribe-cli -m <gguf> samples/jfk.wav` (or the
      family's primary sample) and emits a non-empty `text:` line.
- [ ] No tensor-level numerical comparisons were attempted on
      quantized GGUFs — that is intentional. Per-quant quality lives in
      the Stage 7 WER summary table for user inspection; it is not an
      automated gate.

**Stage 6 — porting-6-bench**
- [ ] At least one baseline report under
      `reports/perf/<machine>/*_<variant>_<backend>.json`.
- [ ] Each report has all required schema fields:
      `machine.{slug,cpu_model,os,hostname}`, `git_sha`, per-run
      `model_path` / `backend` / `per_iter[]` (with mel/encode/decode/
      total/wall), `rtf_wall_mean`, `transcript_sha256`. Missing any of
      these is a **FAIL**.
- [ ] Optional fields (`token_ids_sha256`, `rtf_compute_mean`,
      `git_dirty`) surfaced if absent but do not block — **WARN only**.
- [ ] If any iteration was accepted during this stage, `validate.py
      all` was re-run and exited 0 for the accepted change.

**Stage 7 — porting-7-wer**
- [ ] `reports/wer/<variant>-<REFDTYPE>.<dataset>.score.json` exists.
- [ ] Ref-dtype hard gate: `|observed_wer - upstream_target_ratio| <=
      0.01`. Fail is a HARD finding that sends the port back to Stage 4.
- [ ] One `.score.json` per shipped quant preset.
- [ ] Every `.score.json` has `wer_ci_lo`, `wer_ci_hi`, `n` populated.
- [ ] `reports/wer/<variant>.<dataset>.summary.md` exists.
- [ ] No automatic gate is enforced on quant WER; the summary surfaces
      large regressions for user attention only.

**Stage 8 — porting-8-ship**
- [ ] Pre-flight checklist (Stage 8 Step 1) green: every prior-stage
      artifact present in its expected location.
- [ ] `docs/porting/families/<family>.md` exists with no remaining
      `TODO` placeholders from `_template.md`.
- [ ] `docs/models/<variant>.md` exists with a download table that has
      one row per shipped preset and populated WER + size columns.
- [ ] `scripts/hf_cards/<variant>.yaml` exists with `target_repo` set.
- [ ] `models/<variant>/README.md` exists (rendered by
      `scripts/hf_cards/generate.py`).

### Step 3: Run checks (read-only)

Execute each check. Never modify artifacts — reading, parsing, and
running preflight/validate in read-only mode is allowed.

For checks that shell out (preflight, validate, loader smoke), capture
exit codes but do not re-run things that are expensive (e.g. do NOT
re-run WER scoring). If an artifact that proves the check is absent,
flag as FAIL with "artifact missing" rather than re-running to produce
it.

### Step 4: Emit the report

Produce a markdown report to stdout in this shape:

```markdown
# Review — Stage <N>: <stage-name>
family=<family> variant=<variant>
generated <ISO timestamp>

| Check | Status | Detail |
|-------|--------|--------|
| <check description> | PASS/FAIL/WARN | <path / delta / reason> |
...

## Summary
- <N> passed, <M> failed, <K> warnings.
- Overall: <READY TO ADVANCE | NEEDS ATTENTION | BLOCKED>.

## Remediation
<For each FAIL, a concrete next action. E.g.:
- "FAIL: tests/tolerances/<family>.json has no _comment block.
  → Return to porting-4-cpp Step 4 and add the drift-source comment.">
```

## Postconditions

- A markdown review report is emitted to stdout.
- No artifacts were modified.
- The report names every failing check with a concrete remediation
  action (not "investigate further").

## Pointers (read, not execute)

- The stage skills themselves
  (`porting-1-intake/SKILL.md` through `porting-8-ship/SKILL.md`) define
  the authoritative postconditions; this skill's checklists mirror them
  but the stage SKILL.md is the source of truth.
- `scripts/preflight.py` — read-only invocation for Gate A and Gate B
  checks.
- `scripts/validate.py` — exit-code-only invocation for Stage 4's
  numerical gate.
