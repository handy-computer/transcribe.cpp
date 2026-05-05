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

One row per advertised capability. Stage 1 drafts the rows with
`Status: TODO`; Stage 4 fills in the observed `Status` after running
each command. Allowed statuses:

- `PASS` — command ran and the observable matched.
- `SKIP — not exposed by runtime` — capability is advertised upstream
  but the public CLI/API does not surface a way to verify it. The row
  stays here so readers see the gap honestly.
- `ACCEPTED GAP — <reason>` — capability is exposed but intentionally
  not exercised here; reason names what would unblock the row.

Do not invent observables the runtime cannot actually produce (e.g.
do not write "detected language equals X" if the CLI does not print
the detected language) — those rows resolve to SKIP, not invented
checks.

| Capability | Mode | Command / test | Expected observable | Status |
|------------|------|----------------|---------------------|--------|
| Transcribe | explicit language hint | `build/bin/transcribe-cli -m models/<variant>/<variant>-<REFDTYPE>.gguf --language en samples/jfk.wav` | non-empty plausible English transcript | TODO |
| Transcribe | auto / no language hint | `build/bin/transcribe-cli -m models/<variant>/<variant>-<REFDTYPE>.gguf samples/jfk.wav` | non-empty plausible transcript on the auto-detect path | TODO |
| Translate | only if exposed | `<actual supported command>` | non-empty English transcript on non-English audio | TODO |
| Segment timestamps | only if exposed | `<actual supported command>` | timestamp output present at segment granularity | TODO |
| Word timestamps | only if exposed | `<actual supported command>` | timestamp output present at word granularity | TODO |

## Notes

- TODO

