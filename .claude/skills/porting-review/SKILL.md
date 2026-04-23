---
name: porting-review
description: Reviews one stage of a port against its postconditions and produces a structured review report. Invoked as porting-review with stage=N, family, and variant arguments. Use at the end of any porting stage to double-check artifacts before committing, or out-of-band to re-review a stage without re-running it. Input: the stage number (1-7) plus family and variant. Output: a markdown review report to stdout listing per-check pass/fail status, any artifacts that violate postconditions, and concrete remediation steps.
---

# porting-review

Cross-cutting review skill. Reads the artifacts produced by a given porting stage, checks them against that stage's postconditions, and emits a pass/fail/warn report. Does not modify any artifact — strictly read-only.

## Preconditions

- `family` and `variant` identify an in-progress or completed port.
- `stage` is an integer 1–7 identifying which stage to review.
- The relevant artifacts for that stage are expected to exist; their absence is itself a review finding.

## Workflow

```
Review progress:
- [ ] Step 1: Parse arguments (stage, family, variant)
- [ ] Step 2: Load the stage's checklist
- [ ] Step 3: Run every check (read-only)
- [ ] Step 4: Emit the markdown review report
```

### Step 1: Arguments

Required: `stage` (1–7), `family`, `variant`. Fail fast if any is missing or the stage is out of range.

### Step 2: Stage checklist

Each stage has a fixed checklist. The skill applies the one for the requested stage.

**Stage 1 — porting-1-intake**
- [ ] `reports/porting/<family>/<variant>/intake.json` exists.
- [ ] Validates against `docs/porting/families/_intake-schema.json`.
- [ ] Human-judgment fields (`reference_framework`, `reference_rationale`, `architecture_pattern`, `known_risks`) are all non-null.
- [ ] `upstream_benchmarks[0]` has a non-null `score` and a `dataset` (LibriSpeech test-clean, or an explicit alternate if the model isn't English).
- [ ] `intake_gaps` is either empty or every entry has an explanation.
- [ ] Preflight Gate A passes: `uv run scripts/preflight.py --family <family> --variant <variant> --gate A` exits 0.
- [ ] `docs/porting/families/<family>.md` exists (may still be a draft).

**Stage 2 — porting-2-refdump**
- [ ] `build/validate/<family>/<variant>/dump_coverage.json` exists.
- [ ] `dump_coverage.json.tensors[]` is non-empty.
- [ ] Every entry in `tensors[]` has a matching `.f32` + `.json` pair under `build/validate/<family>/<variant>/<case>/<stage>/ref/`.
- [ ] A parallel `ref2/` directory exists alongside each `ref/` with the same tensor set — the ref-vs-ref stability pass porting-2-refdump runs to seed tolerances. Missing `ref2/` means the noise floor was not measured and the tolerances below will be arbitrary.
- [ ] `tests/golden/<family>/<variant>.manifest.json` exists and names `reference.entrypoint`.
- [ ] `tests/tolerances/<family>.json` EXISTS. Every tensor entry carries `_provisional: true`. Top-of-file `_comment` array names ref-vs-ref stability (noise floor) as the derivation source. Provisional numbers derive from `max(10 × noise, 1e-6)` per tensor.

**Stage 3 — porting-3-convert**
- [ ] `models/<variant>/<variant>-<REFDTYPE>.gguf` exists, where REFDTYPE matches intake's `dtype.expected`.
- [ ] `reports/convert/<variant>-<REFDTYPE>.json` exists and records `gguf_sha256`, `source_hf_repo`, `source_hf_revision`.
- [ ] Preflight Gate B passes: `uv run scripts/preflight.py --family <family> --variant <variant> --gate B` exits 0.
- [ ] Loader smoke (split by case):
  - If `src/arch/<family>/` does NOT exist yet (new family): Gate B + parseable GGUF metadata is sufficient. The loader returns `UNSUPPORTED_ARCH`; that is acceptable at this stage — record it as a pass-with-note, not a FAIL.
  - If `src/arch/<family>/` already exists (new variant in a ported family): `build/bin/transcribe-cli -m <gguf> samples/jfk.wav` must exit 0. Missing is a **FAIL**.
- [ ] No quantized GGUFs (F16/Q8_0/...) exist yet unless the source was already F16 or BF16 — quants are Stage 4.

**Stage 4 — porting-4-cpp**
- [ ] `src/arch/<family>/` exists with `weights.{h,cpp}`, `encoder.{h,cpp}`, `decoder.{h,cpp}`, `model.{h,cpp}`, `capabilities.{h,cpp}`.
- [ ] `tests/tolerances/<family>.json` exists and has a `_comment` block (JSON array of strings, multi-line) explaining drift sources.
- [ ] `uv run scripts/validate.py all --family <family> --variant <variant>` exits 0 at reference dtype.
- [ ] All five derived presets (F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M) exist under `models/<variant>/`, with the source tier suffix skipped where it duplicates one of them.
- [ ] Every produced GGUF loads via `build/bin/transcribe-cli -m <gguf> samples/jfk.wav` without error.

**Stage 5 — porting-5-bench**
- [ ] At least one baseline report under `reports/perf/<machine>/*_<variant>_<backend>.json`.
- [ ] Each report has all **required** schema fields: `machine.{slug,cpu_model,os,hostname}`, `git_sha`, per-run `model_path` / `backend` / `per_iter[]` (with mel/encode/decode/total/wall), `rtf_wall_mean`, `transcript_sha256`. Missing any of these is a **FAIL**.
- [ ] Optional fields (`token_ids_sha256`, `rtf_compute_mean`, `git_dirty`) surfaced if absent but do not block — **WARN only**.

**Stage 6 — porting-6-wer**
- [ ] `reports/wer/<variant>-<REFDTYPE>.<dataset>.score.json` exists.
- [ ] Ref-dtype hard gate: `|observed_wer - upstream_target_ratio| <= 0.01`. Fail is a HARD finding that sends the port back to Stage 4.
- [ ] One `.score.json` per shipped quant preset.
- [ ] Every `.score.json` has `wer_ci_lo`, `wer_ci_hi`, and `n` populated.
- [ ] `reports/wer/<variant>.<dataset>.summary.md` exists.

**Stage 7 — porting-7-ship**
- [ ] `docs/porting/families/<family>.md` exists with no remaining `TODO` placeholders from `_template.md`.
- [ ] `docs/models/<variant>.md` exists with a download table that has one row per shipped preset and populated WER + size columns.
- [ ] `scripts/hf_cards/<variant>.yaml` exists with `target_repo` set.
- [ ] `models/<variant>/README.md` exists (rendered by `scripts/hf_cards/generate.py`).

### Step 3: Run checks (read-only)

Execute each check. Never modify artifacts — reading, parsing, and running preflight/validate in read-only mode is allowed.

For checks that shell out (preflight, validate, loader smoke), capture exit codes but do not re-run things that are expensive (e.g. do NOT re-run WER scoring). If an artifact that proves the check is absent, flag as FAIL with "artifact missing" rather than re-running to produce it.

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
- The report names every failing check with a concrete remediation action (not "investigate further").

## Pointers (read, not execute)

- The stage skills themselves (`porting-1-intake/SKILL.md` through `porting-7-ship/SKILL.md`) define the authoritative postconditions; this skill's checklists mirror them but the stage SKILL.md is the source of truth.
- `scripts/preflight.py` — read-only invocation for Gate A and Gate B checks.
- `scripts/validate.py` — exit-code-only invocation for Stage 4's numerical gate.
