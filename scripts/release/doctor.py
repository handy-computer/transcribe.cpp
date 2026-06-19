#!/usr/bin/env python3
"""Soft-tier release infra doctor — surfaces §2 prerequisites as WARNINGS.

This is the *soft* tier of ``release-preflight`` (notes/releasing.md §8 P1 #6):
it reports infra gaps that would otherwise surface mid-release as a billing /
quota / auth error *after* a tag has already been pushed. It is best-effort by
design and **can never fail a run** — it always exits 0. The hard tier
(``prepare.py --check``) is what gates a release.

It checks only what is cheap and non-fragile:

  * Repo-level secrets visible to the gate job (``HF_TOKEN``, ``MODAL_TOKEN_*``).
  * Environment-scoped secrets (``CARGO_REGISTRY_TOKEN``, ``NPM_TOKEN``) are
    NOT visible from a non-environment job, so they are reported as
    "verify manually" rather than warned on — their real gate is the
    environment that owns them at publish time.
  * GitHub Pages enabled (best-effort, only if ``gh`` + a token are present).

Billing / spending-limit / artifact-storage-quota stay a MANUAL §2 checklist
item: they cannot be proven cheaply or non-fragilely from CI. The durable
mitigation for that failure class is the runner migration + publisher atomicity
(§8 P0 #3), not this script.

    uv run --no-project scripts/release/doctor.py   # human run, locally or in CI

Always exits 0. Emits ``::warning::`` lines on GitHub Actions so gaps are
visible in the run summary without failing it.
"""

from __future__ import annotations

import os
import subprocess
import sys

# Repo-level secrets the gate job CAN see when the workflow maps them into env.
REPO_SECRETS = ["HF_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]
# Environment-scoped secrets — only present inside their protected environment
# (crates-io / npm) at publish time, never in the preflight gate job.
ENV_SCOPED_SECRETS = ["CARGO_REGISTRY_TOKEN", "NPM_TOKEN"]

ON_GHA = os.environ.get("GITHUB_ACTIONS") == "true"


def warn(msg: str) -> None:
    print(f"::warning::{msg}" if ON_GHA else f"WARN: {msg}")


def info(msg: str) -> None:
    print(msg)


def check_repo_secrets() -> int:
    warnings = 0
    for name in REPO_SECRETS:
        if os.environ.get(name):
            info(f"  ok   {name} is present")
        else:
            warn(f"{name} is not set in this job's env — canary downloads / the "
                 f"cu12 Modal build will fail if it is genuinely absent (§2).")
            warnings += 1
    return warnings


def note_env_scoped_secrets() -> None:
    for name in ENV_SCOPED_SECRETS:
        present = "present" if os.environ.get(name) else "not visible here"
        info(f"  --   {name}: {present} (environment-scoped; verify in its "
             f"protected environment — §2)")


def check_pages_enabled() -> int:
    """Best-effort: is GitHub Pages enabled? Never fails; warns at most once."""
    repo = os.environ.get("GITHUB_REPOSITORY")
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not repo or not token:
        info("  --   GitHub Pages: skipped (no GITHUB_REPOSITORY / token in env)")
        return 0
    try:
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/pages"],
            capture_output=True, text=True, timeout=20,
            env={**os.environ, "GH_TOKEN": token},
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        info("  --   GitHub Pages: skipped (gh CLI unavailable)")
        return 0
    if proc.returncode == 0:
        info("  ok   GitHub Pages is enabled")
        return 0
    warn("GitHub Pages does not appear to be enabled — the cu12 PEP 503 index "
         "(wheel-index.yml) needs it (§2). This is best-effort; verify manually.")
    return 1


def main() -> int:
    info("release infra doctor (soft tier — warnings only, never gates)\n")
    info("repo-level secrets:")
    warnings = check_repo_secrets()
    info("\nenvironment-scoped secrets (informational):")
    note_env_scoped_secrets()
    info("\ninfra reachability:")
    warnings += check_pages_enabled()

    info("\nMANUAL (§2, not provable from CI): GitHub billing current, spending "
         "limit adequate, Actions artifact-storage quota has headroom, PyPI prod "
         "trusted publishers registered for both projects.")

    info(f"\ndoctor: {warnings} warning(s). (Soft tier — this never fails the run.)")
    return 0  # by contract: the doctor cannot, by itself, fail a release.


if __name__ == "__main__":
    raise SystemExit(main())
