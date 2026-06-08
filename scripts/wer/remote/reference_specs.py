from __future__ import annotations

import json
from pathlib import Path

from fingerprints import env_fingerprint


def resolve_reference(
    root: Path,
    spec: str,
    runner_override: str = "",
) -> dict:
    """Map a variant spec to everything the reference cell needs.

    spec is '<family>:<variant>' (unambiguous, preferred) or '<variant>'
    (resolved by scanning reports/porting/*/<variant>/intake.json). Reads the
    intake for the upstream repo + framework, picks the runner by globbing
    scripts/wer/run_reference_<family>_*.py (shortest name wins; pass --runner
    to override when a family has several, e.g. parakeet's streaming variants).
    """
    if ":" in spec:
        family, variant = spec.split(":", 1)
        intake = root / "reports" / "porting" / family / variant / "intake.json"
    else:
        variant = spec
        hits = list(root.glob(f"reports/porting/*/{variant}/intake.json"))
        if len(hits) != 1:
            raise SystemExit(
                f"{spec!r}: found {len(hits)} matching intakes; "
                f"disambiguate with '<family>:{variant}'")
        intake = hits[0]
        family = intake.parents[1].name
    if not intake.exists():
        raise SystemExit(f"no intake.json at {intake}")
    d = json.loads(intake.read_text())
    upstream_repo = d.get("hf_repo")
    framework = d.get("reference_framework") or "unknown"
    if not upstream_repo:
        raise SystemExit(f"{intake}: missing hf_repo")

    if runner_override:
        runner_rel = runner_override
        if not (root / runner_rel).is_file():
            raise SystemExit(f"--runner {runner_rel} not found under {root}")
    else:
        cands = sorted(root.glob(f"scripts/wer/run_reference_{family}_*.py"),
                       key=lambda p: len(p.name))
        if not cands:
            raise SystemExit(
                f"no scripts/wer/run_reference_{family}_*.py; "
                f"pass --runner <path> (a new family needs one -- copy the "
                f"closest same-framework runner)")
        runner_rel = str(cands[0].relative_to(root))

    env_dir = root / "scripts" / "envs" / family
    if not (env_dir / "pyproject.toml").is_file():
        raise SystemExit(f"no reference env at {env_dir}/pyproject.toml")

    return {
        "family": family, "variant": variant, "framework": framework,
        "upstream_repo": upstream_repo, "runner_rel": runner_rel,
        "env_fp": env_fingerprint(root, family, runner_rel),
    }
