# Validate

`scripts/validate.py` is the convention-driven numerical validation
orchestrator. It runs the three stages of the accuracy loop — reference
dump, C++ dump, compare — for a family, with every path derived from
convention.

## What it does

```text
  validate.py ref     → runs dump_reference_<family>_*.py → writes build/validate/<family>/<variant>/<case>/ref/
  validate.py cpp     → runs transcribe-cli with TRANSCRIBE_DUMP_DIR → writes build/validate/.../cpp/
  validate.py compare → runs compare_tensors.py with tests/tolerances/<family>.json; compares transcript text and, when the reference dump carries word rows, word timestamps
  validate.py all     → ref + cpp + compare in sequence
```

One required input: `--family <name>`. Everything else is discovered:

| Input                  | Path                                                                 |
|------------------------|----------------------------------------------------------------------|
| Manifest               | `tests/golden/<family>/*.manifest.json`                              |
| Dump script            | `scripts/dump_reference_<family>_<reference>.py`                     |
| Python env             | `scripts/envs/<family>/`                                             |
| Tolerances             | `tests/tolerances/<family>.json`                                     |
| Audio                  | `samples/jfk.wav` (manifest can override)                            |
| GGUF                   | `models/<family>/` (first `*.gguf`, preferring `*.bf16` / `*.f32`)   |
| Reference output       | `build/validate/<family>/<variant>/<case>/ref/`                      |
| C++ output             | `build/validate/<family>/<variant>/<case>/cpp/`                      |

## CLI

```bash
uv run scripts/validate.py ref     --family cohere
uv run scripts/validate.py cpp     --family cohere
uv run scripts/validate.py compare --family cohere
uv run scripts/validate.py all     --family cohere

# Override model or GGUF
uv run scripts/validate.py all --family cohere --model /path/to/checkpoint
uv run scripts/validate.py cpp --family cohere --gguf models/cohere-transcribe-03-2026/cohere-transcribe-03-2026-BF16.gguf
```

## Exit codes

- `0` — all contract tensors are within tolerance and transcript text matches;
  when the reference dump carries word rows, all word timestamps are within the
  family's `timestamps` budget.
- `1` — any tensor is out of tolerance, missing, or shape-mismatched; any
  required transcript text or word timestamp is missing or mismatched.

Exit-code driven, no interactive prompts (per project policy).

## When to run

- After any loader change in `src/arch/<family>/weights.cpp`.
- After any graph change in `src/arch/<family>/encoder.cpp` or
  `decoder.cpp`.
- After regenerating a GGUF with a new converter.
- As the dev gate before running WER (which is slower).

WER is the user-facing gate. `validate` is the per-tensor dev gate that
catches numerical regressions in seconds.
