# Contributing to transcribe.cpp

This repo is an inference engine for transcription (ASR) models, built on top
of GGML.

The project is intentionally conservative. It is a library surface, a model
runtime, and a set of package artifacts that users embed in larger processes.
Changes should be easy to review, portable across supported platforms, and
maintainable after merge. Contributors who submit substantial features, model
families, packaging changes, or backend work may be asked to help maintain those
areas over time.

The porting workflow lives in `docs/porting/`. The docs define the canonical
stages, artifacts, and gates. When project-local porting skills exist (for
example under `.claude/skills/porting-*`), they are the preferred UX for
agents working in this repo. Contributors using Codex or another harness can
follow the docs directly and still produce the same artifacts.

## AI-assisted contributions

AI tools may be used when a human contributor is driving the design, reviewing
the output, and prepared to debug and maintain the result.

If AI meaningfully assisted with code or documentation:

1. Disclose that usage in the PR.
2. Manually review the generated or assisted changes before submission.

Do not use AI to write PR descriptions, issue reports, commit messages, project
discussions, or replies to reviewers. Do not submit automated commits or pull
requests. Obviously AI written PR descriptions will almost certainly be rejected.

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

## Coding style

The style below is adapted from the current llama.cpp and whisper.cpp
contribution guidelines and made local so contributors do not need to chase
external documents before writing code.

General rules:

- Avoid adding third-party dependencies, extra files, extra headers, or new
  build-time requirements unless they are necessary and justified.
- Always consider compatibility with supported operating systems, compilers,
  CPU architectures, accelerators, and package formats.
- Prefer plain, C-like C++ for runtime code. Use basic `for` loops and simple
  helper functions over clever STL, template, or lambda-heavy constructs.
- Keep abstractions narrow. Add one only when it removes real duplication,
  centralizes ownership/lifetime, or matches an established local pattern.
- New model families and backend work should bring up CPU support first unless
  there is a specific reason to do otherwise.
- Do not add new ggml operators, `ggml_type` values, or backend behavior without
  CPU validation, benchmark or accuracy evidence where relevant, and an upstream
  plan for ggml/llama.cpp when appropriate.
- Do not mix functional changes with broad reformatting or unrelated cleanup.
- Keep comments concise. Explain non-obvious invariants, ABI contracts, tensor
  layout, numerical choices, and portability traps. Do not preserve local task
  history or design backstory in source comments.
- Use ASCII in source files and comments unless a file or data format requires
  otherwise.

Formatting rules:

- Use 4 spaces for indentation.
- Put braces on the same line.
- Use `void * ptr` and `int & a` pointer/reference spacing.
- Use vertical alignment when it improves readability and batch editing.
- Clean up trailing whitespace.
- Format with the pinned formatter rather than a system clang-format:

  ```bash
  scripts/ci/clang-format.sh            # format our tree in place
  scripts/ci/clang-format.sh --check    # verify only, no changes
  ```

  It wraps clang-format `22.1.5` (fetched via `uvx`) against the root
  `.clang-format`, a profile based on llama.cpp's. The version is pinned in the
  script so local output matches CI byte-for-byte; bump it there and reformat
  the tree in the same commit.
- Formatting scope is our C/C++ only. Vendored trees (`ggml/`,
  `src/third_party/`) and verbatim upstream copies
  (`src/transcribe-unicode-data.cpp`) are never reformatted.
- Do not reformat unrelated code as part of a behavior change. CI enforces this
  through the `clang-format` workflow (`.github/workflows/clang-format.yml`),
  which gates our C/C++ against the pinned formatter.

Naming rules:

- Use `snake_case` for functions, variables, and type names.
- Prefer names with the longest common prefix when grouping related values:
  `number_small`, `number_big` rather than `small_number`, `big_number`.
- Public symbols are prefixed with `transcribe_` or `TRANSCRIBE_`.
- Enum values are uppercase and prefixed by the enum name:

```c
enum transcribe_task {
    TRANSCRIBE_TASK_TRANSCRIBE = 0,
    TRANSCRIBE_TASK_TRANSLATE  = 1,
};
```

- In public C/C++ headers, use sized integer types such as `int32_t` where ABI
  size matters. `size_t` is appropriate for allocation sizes and byte offsets.
- In C++ code, omit optional `struct` and `enum` keywords when they are not
  needed:

```cpp
// OK
transcribe_model * model;
transcribe_task task;

// Not OK in C++ implementation code
struct transcribe_model * model;
enum transcribe_task task;
```

- C/C++ filenames are lowercase. Prefer dashes for ordinary source files.
  Family directories and family public headers may use the canonical family key,
  including underscores, so API names and file paths stay aligned. Headers use
  `.h`; source files use `.c` or `.cpp`.
- Python library/module filenames are lowercase with underscores. Command-line
  scripts may use dashes when following the existing converter/tool convention.

API and tensor rules:

- Keep `include/transcribe.h` and family public headers free of ggml includes.
  Callers should not need to include `<ggml.h>`.
- Public structs crossing the ABI use the documented `struct_size` convention.
- Tensors store data in row-major order. Dimension 0 is columns, dimension 1 is
  rows, and dimension 2 is matrices.
- `ggml_mul_mat(ctx, A, B)` follows ggml's convention, not ordinary source-code
  reading order: treat existing ggml/llama.cpp graph patterns as the reference
  when adding graph code.

When this repo intentionally differs from llama.cpp/whisper.cpp, keep the
difference local and documented near the API or subsystem that needs it. The
main accepted internal exception is that the multi-family runtime may use C++
ownership helpers or narrow virtual bases where they centralize lifetime and
avoid duplicating model/session teardown logic.

## Review gates

Required before merge:

| Gate | Command / owner | Requirement |
| --- | --- | --- |
| Formatting | `scripts/ci/clang-format.sh --check` (CI: clang-format workflow) | All our C/C++ matches the pinned clang-format |
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
