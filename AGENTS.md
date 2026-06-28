# transcribe.cpp Agent Conventions

Read `CONTRIBUTING.md` for contribution policy, review gates, and coding
style. This file is only the local command/automation convention sheet for
coding agents.

## Python

- ALWAYS use `uv run` for every Python invocation. Never bare `python`,
  `python3`, or `pip`.
- Use `uv pip` for packages, `uv sync` for envs, and `uv run` for scripts.
- Per-family reference environments live under `scripts/envs/<family>/` and are
  invoked as:

```bash
uv run --project scripts/envs/<family> scripts/<script>.py ...
```

## Build

- After C++ changes, run:

```bash
cmake --build build --target transcribe-cli
```

## Formatting

- Format our C/C++ before committing. The formatter is pinned and fetched via
  `uvx`, so do not rely on a system clang-format:

```bash
scripts/ci/clang-format.sh            # format our tree in place (default)
scripts/ci/clang-format.sh --check    # verify, no changes
```

- Scope is our code only. Vendored trees (`ggml/`, `src/third_party/`) and
  verbatim upstream copies (`src/transcribe-unicode-data.cpp`) are never
  formatted. CI gates our C/C++ in
  `.github/workflows/clang-format.yml`.

## Verification

- End-to-end numerical checks:

```bash
uv run scripts/validate.py all --family <f> [--variant <v>]
```

`--variant` is required when the family has multiple manifests.

- Manual tensor debugging:

```bash
uv run scripts/compare_tensors.py ...
```

- Cheap metadata/config gates before expensive numerical work:

```bash
uv run scripts/preflight.py --family <f> [--variant <v>]
```

- Never suppress test failures without root cause analysis.

## Porting a New Model

Use the `porting-*` skills in `.claude/skills/`. Stage skills are independent
and run in order:

```text
porting-1-intake -> porting-2-oracle -> porting-3-convert -> porting-4-cpp
-> porting-5-quants -> porting-6-bench -> porting-7-wer -> porting-8-ship
```

## Git Hygiene

- Do not commit, push, create pull requests, or comment on pull requests unless
  the user explicitly asks for that action.
- Do not reformat unrelated code in the same change as a behavior fix.
