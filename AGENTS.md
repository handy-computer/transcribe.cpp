# Instructions for transcribe.cpp

This project ports speech-to-text model families to a C/C++ ggml runtime. It
prioritizes stability, portability, reviewable changes, and long-term
maintenance.

## Contributor Ownership

Submitting code means taking responsibility for it. Contributors may be asked to
help maintain the areas they change, especially for new model families, runtime
features, packaging surfaces, or backend-specific behavior.

Contributions are welcome, including small fixes and focused improvements, but
the review bar is intentionally high:

- Understand every line you submit.
- Keep changes scoped to one feature, fix, model family, or maintenance task.
- Provide the validation evidence requested by `CONTRIBUTING.md`.
- Avoid speculative rewrites, broad refactors, and unrelated cleanup.
- Prefer established local patterns over introducing new infrastructure.

## AI-Assisted Work

AI tools may be used when a human contributor is driving the design, reviewing
the result, and prepared to debug and maintain the change.

Requirements:

- Disclose meaningful AI assistance in the pull request.
- Manually review generated or AI-assisted code before submission.
- Be ready to explain the code without relying on an AI tool.
- Do not use AI to write pull request descriptions, commit messages, issue
  reports, reviewer replies, or project discussions.
- Do not submit fully automated commits or pull requests.

Agents must not commit, push, create pull requests, or comment on pull requests
without explicit human approval for that exact action.

## Code Style

Use the coding style in `CONTRIBUTING.md`. The short version:

- Follow the ggml ecosystem style used by llama.cpp and whisper.cpp.
- Keep runtime C++ plain and C-like, especially around ggml graph construction.
- Avoid new dependencies unless there is a strong portability and maintenance
  argument.
- Prefer CPU-first bring-up for new model families and backend work.
- Keep comments concise. Explain non-obvious invariants, not local task history.
- Use ASCII in source files and comments unless a file or data format requires
  otherwise.
- Do not reformat unrelated code in a functional change.

Before editing, read the nearby implementation and tests. If a change introduces
a new architectural pattern or expands the public ABI, pause for explicit design
review.

## Local Commands

Use the commands documented in `CONTRIBUTING.md` and `CLAUDE.md`. For Python
scripts in this repository, use `uv run`.
