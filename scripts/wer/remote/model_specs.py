from __future__ import annotations

from pathlib import Path


def resolve_model(root: Path, spec: str) -> tuple[str, list[str] | None]:
    """Map a model spec to (HF repo, filenames or None).

    Rules:
    - Spec contains '/': treat as a HF repo path. Filenames are None
      (caller will discover via the HF API at dispatch time).
    - Otherwise: treat as an hf_card slug. Reads
      scripts/hf_cards/<slug>.yaml, returns its target_repo and the
      pinned `quants[].filename` list.
    """
    if "/" in spec:
        return spec, None
    card_path = root / "scripts" / "hf_cards" / f"{spec}.yaml"
    if not card_path.exists():
        raise SystemExit(
            f"no hf_card at {card_path}; pass a HF repo path "
            f"(e.g. handy-computer/{spec}-gguf) if the card doesn't exist yet"
        )
    # Tiny manual parser so the local entrypoint has no non-stdlib deps. The card
    # schema has target_repo at top level and filename: only under quants[].
    target_repo: str | None = None
    filenames: list[str] = []
    for raw in open(card_path):
        line = raw.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if line.startswith("target_repo:"):
            target_repo = line.split(":", 1)[1].strip()
        elif stripped.startswith("filename:"):
            filenames.append(stripped.split(":", 1)[1].strip())
    if not target_repo:
        raise SystemExit(f"hf_card {spec!r}: missing target_repo")
    if not filenames:
        raise SystemExit(f"hf_card {spec!r} has no quants[].filename entries")
    return target_repo, filenames
