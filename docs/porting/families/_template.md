# <Family>

Status: research | bring-up | validation | supported

## Identity

- Family key:
- Upstream architecture string:
- Hugging Face repo:
- Hugging Face revision:
- License:
- Variants:

## References

- Canonical reference:
- Instrumented reference:
- Cross-check references:

<!-- Optional: only add a "Bridge validation" subsection here if the
canonical and instrumented references differ. Per-machine env details
live in `reports/perf/<machine>/*.json`; reference package pins live in
`scripts/envs/<family>/pyproject.toml`; per-port artifact paths follow
the layout policy in CLAUDE.md and don't need to be enumerated here. -->

## Commands

Reference run:

```bash
TODO
```

Reference dumps:

```bash
TODO
```

Conversion:

```bash
TODO
```

Validation:

```bash
TODO
```

Benchmarks:

```bash
TODO
```

## Capability Validation

One row per advertised capability. Each row carries two human/agent
columns that are filled at different stages:

- **`Target`** — the *scope decision*, set at Stage 1 and signed off by
  the user. It declares whether this port is obligated to deliver the
  capability. This is the contract Stage 4 implements against.
- **`Status`** — the *observed outcome*, filled at Stage 4 after running
  the command. Stage 1 leaves it `TODO`.

Allowed `Target` values (Stage 1):

- `MUST PASS` — in scope for this port. Stage 4 must resolve `Status` to
  `PASS`; it may not be downgraded to SKIP/ACCEPTED GAP without the user
  re-signing the scope change.
- `OUT OF SCOPE — <reason>` — explicitly deferred by the user at intake.
  Stage 4 may resolve it to SKIP or ACCEPTED GAP. The reason names what
  would bring it back in scope.

Allowed `Status` values (Stage 4):

- `PASS` — command ran and the observable matched.
- `SKIP — not exposed by runtime` — capability is advertised upstream
  but the public CLI/API does not surface a way to verify it. The row
  stays here so readers see the gap honestly. Only legal when `Target`
  is `OUT OF SCOPE`.
- `ACCEPTED GAP — <reason>` — capability is exposed but intentionally
  not exercised here; reason names what would unblock the row. Only
  legal when `Target` is `OUT OF SCOPE` (or the batch serial-fallback
  case below).

Do not invent observables the runtime cannot actually produce (e.g.
do not write "detected language equals X" if the CLI does not print
the detected language) — those rows resolve to SKIP, not invented
checks.

Two rows have their `Target` forced rather than left to the user:
- **Streaming**, when `capabilities.streaming: true`. The runtime exposes
  a streaming path (`--stream-chunk-ms`), so a natively-streaming model
  is always `Target: MUST PASS` and MUST resolve its streaming row to
  `PASS` (byte-equal vs the reference streaming runner). It may not be
  `SKIP — not exposed by runtime` or `ACCEPTED GAP`; if blocked it is an
  explicit user-signed BLOCKER.
- **Batch (offline)**. Every family inherits `transcribe_run_batch` via
  the serial fallback, so this row is always present with `Target:
  MUST PASS`: `PASS` when the family ships an explicit `run_batch()` fast
  path (text byte-identical and CPU tensor parity bit-exact), or the one
  sanctioned exception `ACCEPTED GAP — serial fallback` when it runs the
  fallback loop (correct, just un-accelerated).

| Capability | Mode | Command / test | Expected observable | Target | Status |
|------------|------|----------------|---------------------|--------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/<variant>/<variant>-<REFDTYPE>.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | MUST PASS | TODO |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/<variant>/<variant>-<REFDTYPE>.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | MUST PASS | TODO |
| Translate | only if exposed | `<actual supported command>` | non-empty English transcript on non-English audio | <MUST PASS \| OUT OF SCOPE — reason> | TODO |
| Segment timestamps | only if exposed | `<actual supported command>` | timestamp output present at segment granularity | <MUST PASS \| OUT OF SCOPE — reason> | TODO |
| Word timestamps | only if exposed | `<actual supported command>` | timestamp output present at word granularity | <MUST PASS \| OUT OF SCOPE — reason> | TODO |
| Streaming | only if `capabilities.streaming` | `build/bin/transcribe-cli -m models/<variant>/<variant>-<REFDTYPE>.gguf --stream-chunk-ms <N> --backend cpu --threads 1 samples/jfk.wav` | byte-equal transcript vs reference streaming runner | MUST PASS | TODO |
| Batch (offline) | run_batch vs serial | `uv run scripts/batch_parity.py --model models/<variant>/<variant>-<REFDTYPE>.gguf --samples-dir samples/wer/<dataset> --batch-sizes 2,4,8 --backend cpu` | byte-identical hypotheses + CPU tensor parity | MUST PASS | TODO |

## Notes

- TODO

