## Summary

<!-- What changed, and why? Keep this concise. -->

## Scope

<!-- List the main areas touched, for example src/arch/<family>, converter, validation, bindings, packaging, docs. -->

## AI Assistance

<!-- State whether AI meaningfully assisted with code or docs. The human author is expected to understand and own the change. -->

## Validation

<!-- Paste command output or concise summaries for the gates that apply. -->

- [ ] Default tests: `ctest --test-dir build`
- [ ] Preflight A: `uv run scripts/preflight.py --family <f> [--variant <v>] --gate A`
- [ ] Preflight B: `uv run scripts/preflight.py --family <f> [--variant <v>] --gate B`
- [ ] Numerical validation: `uv run scripts/validate.py all --family <f> [--variant <v>] --report`
- [ ] Real-model smoke: `ctest --test-dir build -R <family>`
- [ ] WER/benchmark impact recorded, if this PR claims performance or publication readiness

## Model Artifacts

<!-- For model-family or converter changes: source revision, converter command, output GGUF filename, SHA256, and where reviewers can get the artifact. Do not commit GGUFs. -->

## ggml Impact

<!-- Choose one and explain if needed. -->

- [ ] No ggml changes
- [ ] Vendored ggml sync only
- [ ] Local ggml patch, with upstream issue/PR link or plan
- [ ] New ggml op/type/backend behavior, with CPU validation and maintenance rationale

## Maintenance Notes

<!-- Known limitations, follow-up work, expected ownership, or review risks. -->
