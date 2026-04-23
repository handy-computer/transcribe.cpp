# transcribe.cpp Agent Conventions

## Python (critical)
- ALWAYS use `uv run` for every Python invocation. Never bare `python`, `python3`, or `pip`.
- Use `uv pip` for packages, `uv sync` for envs, `uv run` for scripts.
- Per-family reference environments live under `scripts/envs/<family>/` and are invoked as `uv run --project scripts/envs/<family> scripts/<script>.py ...`.

## Build
- `cmake --build build --target transcribe-cli` after C++ changes.

## Verification
- `uv run scripts/validate.py all --family <f> [--variant <v>]` for end-to-end numerical checks. `--variant` is required when the family has multiple manifests.
- `uv run scripts/compare_tensors.py` for manual debugging of a single tensor pair.
- `uv run scripts/preflight.py --family <f> [--variant <v>]` for cheap metadata/config gates before expensive numerical work.
- Never suppress test failures without root cause analysis.

## Porting a new model
Use the `porting-*` skills in `.claude/skills/`. Stage skills are independent and run in order:
`porting-1-intake` → `porting-2-refdump` → `porting-3-convert` → `porting-4-cpp` → `porting-5-bench` → `porting-6-wer` → `porting-7-ship`.
Each stage ends with a manual human-authored commit. See `docs/porting/agent-automation-plan.md` for the full plan.
