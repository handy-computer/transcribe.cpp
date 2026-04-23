# Contributing to transcribe.cpp

This repo ports speech-to-text model families to a C/C++ ggml runtime. Most
substantial contributions add a new model family; smaller contributions fix
bugs, improve tooling, or add backend coverage.

The porting workflow lives in `docs/porting/`. The docs define the canonical
stages, artifacts, and gates. When project-local porting skills exist (for
example under `.claude/skills/porting-*`), they are the preferred UX for
agents working in this repo. Contributors using Codex or another harness can
follow the docs directly and still produce the same artifacts.

## Proposing a new model port

1. **Open an issue first.** Include the upstream model repo, proposed family
   key, variant name, and architecture pattern: encoder-transducer,
   encoder-decoder, audio-LLM/token-injection, or encoder-CTC.
2. **Use the project-local stage skills if they exist; otherwise start with
   `docs/porting/0-porting.md` and `docs/porting/agent-automation-plan.md`.**
   The docs remain the canonical workflow; the skills are the preferred UX
   when present.
3. **Follow the stage order.** Each stage produces inputs consumed by the next
   one. In particular, do not start the C++ implementation before the intake,
   reference choice, golden manifest, reference dumper, and first accuracy
   GGUF plan are clear.

## Merge vs publication

An **accepted family** is ready to merge when the C++ implementation,
conversion path, validation, and smoke tests are reviewable and green. It does
not need an official Hugging Face upload.

A **published variant** is ready for users after the accepted family also has
per-quant WER/benchmark results, rendered model-card content, canonical GGUFs
uploaded under the `handy-computer` Hugging Face organization, and a downloaded
GGUF roundtrip check.

Contributors normally submit accepted-family PRs. Maintainers publish canonical
variants separately.

## PR package

For a new family PR, commit the source-controlled contract and implementation:

- family note: `docs/porting/families/<family>.md`
- intake: `reports/porting/<family>/<variant>/intake.json`
- porting log: `reports/porting/<family>/<variant>/_porting-log.md`
- reference environment and dumper: `scripts/envs/<family>/pyproject.toml`,
  `scripts/dump_reference_<family>_<reference>.py`
- converter: normally `scripts/convert-<family>.py`
- golden manifest: `tests/golden/<family>/<variant>.manifest.json`
- tolerances: `tests/tolerances/<family>.json`
- C++ implementation and registry/build wiring: `src/arch/<family>/`,
  `src/transcribe-arch.cpp`, `CMakeLists.txt` / `tests/CMakeLists.txt` as needed
- smoke tests: synthetic fixture smoke, real-model structural smoke, and
  public ABI / transcript smoke, following `docs/model-family-testing.md`
- benchmark/validation wiring needed for `scripts/validate.py` and
  `scripts/bench/run.py`
- when automation changes are part of the PR: project-local skill updates
  under `.claude/skills/porting-*`

Paste review evidence into the PR description, not into generated report files:

- Preflight Gate A stdout:
  `uv run scripts/preflight.py --family <f> [--variant <v>] --gate A`
- Preflight Gate B stdout after conversion:
  `uv run scripts/preflight.py --family <f> [--variant <v>] --gate B`
- `uv run scripts/validate.py all --family <f> [--variant <v>] --report`
  summary: tensor compare output plus transcript match line
- default `ctest --test-dir build` result
- real-model smoke command/output, using the family-specific model env var
  documented in the family note
- conversion summary: source model revision, converter command, output GGUF
  filename, and SHA256
- WER and benchmark numbers, only if the PR is also claiming publication
  readiness

Do not commit:

- GGUF/model binaries
- heavyweight tensor dumps under `build/validate/`
- per-run preflight outputs or validate report bundles
- HF credentials or upload logs
- personal/global agent-harness config outside the project-local automation
  shipped with the repo, such as private Codex skills, local sandbox-only
  prompt files, or untracked harness state

`--variant <v>` is required whenever a family has multiple manifests.

## Review gates

Required before merge:

| Gate | Command / owner | Requirement |
| --- | --- | --- |
| Intake signoff | human review | Schema-valid intake; `reference_framework`, `architecture_pattern`, and `known_risks` reviewed; dtype/frontend/tokenizer gaps resolved or explicitly accepted |
| Preflight A | `uv run scripts/preflight.py --family <f> [--variant <v>] --gate A` | Pass, or warnings tied to accepted intake gaps |
| Preflight B | `uv run scripts/preflight.py --family <f> [--variant <v>] --gate B` | Pass after converter exists |
| Numerical validation | `uv run scripts/validate.py all --family <f> [--variant <v>]` | All tolerance-file tensors within bounds; transcript matches reference when present |
| Default tests | `ctest --test-dir build` | All enabled default tests pass |
| Real-model smokes | `ctest --test-dir build -R <family>` with `TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON` | Pass on a representative accuracy GGUF |

Required before canonical publication:

| Gate | Owner | Requirement |
| --- | --- | --- |
| WER / benchmarks | maintainer or contributor | Per-quant numbers recorded in `docs/models/<family>.md` |
| HF card | maintainer | `scripts/hf_cards/<variant>.yaml` updated and README rendered |
| Canonical upload | maintainer | GGUFs uploaded to the `handy-computer` Hugging Face organization |
| Download roundtrip | maintainer | Downloaded canonical GGUF reloads and validates cleanly; validation commit recorded in docs/HF card |
| Preflight D | maintainer | Planned post-quantization gate once implemented |

Preflight Gate C is a runtime bring-up aid, not a separate PR gate. Gate D and
the upload roundtrip are maintainer-side publication gates.

## GGUF artifact policy

Large GGUFs never live in git.

Contributors may produce and share candidate GGUFs to support review. For any
candidate GGUF, include the URL or path, SHA256, source model revision,
converter command, and quantization preset if applicable. Candidate GGUFs are
review artifacts only; they are not the canonical distribution.

The canonical GGUF source for this project is the `handy-computer` Hugging Face
organization, using repos such as `handy-computer/<variant>-gguf`. A GGUF is a
project release artifact only after a maintainer publishes it there and records
the validation commit in both:

- `docs/models/<family>.md`
- `scripts/hf_cards/<variant>.yaml`, rendered into the HF README

Golden manifests are not mutated for uploads. They pin validation provenance
for the port. Release state lives in the model card and rendered HF README; HF
revision history preserves the card and files for each upload.

## Reviewer checklist

Reviewers should verify:

1. Intake is signed off and matches the implementation.
2. Preflight A+B evidence is in the PR and warnings are understood.
3. The validation summary shows every tolerance-file tensor within bounds and
   an exact transcript match when the reference emits a transcript.
4. Default tests and real-model smokes pass locally or on reviewer hardware.
5. `tests/tolerances/<family>.json` has a `_comment` explaining the reference
   framework, dtype, observed divergence profile, and any widened tolerance.
6. The family note explains bridge validation when canonical and instrumented
   references differ.
7. The porting log captures surprises that should feed future doc/tool fixes.
8. Shared-code refactors are minimal and explicitly justified.

## Local commands

The detailed workflow lives in `docs/porting/`; this is the short gate runner:

```bash
cmake -B build -DTRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON
cmake --build build

uv run scripts/intake.py inspect --repo <org/model> --family <f> --variant <v> \
    --out reports/porting/<f>/<v>/intake.json

uv run scripts/preflight.py --family <f> --variant <v> --gate A
uv run --project scripts/envs/<f> scripts/convert-<family>.py <org/model>
uv run scripts/preflight.py --family <f> --variant <v> --gate B

uv run scripts/validate.py all --family <f> --variant <v>
uv run scripts/validate.py all --family <f> --variant <v> --report

# Use the family-specific env var from the family note.
TRANSCRIBE_<FAMILY>_MODEL=models/<variant>/<variant>-<quant>.gguf \
    ctest --test-dir build -R <family>
```

## Policies

- CPU validation is required for every accepted family.
- Metal and Vulkan coverage are optional for the initial merge unless the PR
  claims backend support.
- Tolerances are data. Keep them in `tests/tolerances/<family>.json`, justify
  every wide value in `_comment`, and do not duplicate per-family tolerance
  profiles in process docs.
- Personal agent shortcuts are local configuration. Do not commit harness
  skills, sandbox files, or prompt state.

## Getting help

- Process question: open an issue tagged `process`.
- Validation divergence: read `docs/porting/4-numerical-validation.md` and
  `docs/porting/4a-numerical-troubleshooting.md`, then open an issue with
  preflight and tensor-compare output.
- License or semantic op mapping: escalate to the maintainer; do not guess.
