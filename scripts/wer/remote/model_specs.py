from __future__ import annotations

import json
from pathlib import Path


def resolve_model(root: Path, spec: str) -> tuple[str, list[str] | None]:
    """Map a model spec to (HF repo, filenames or None).

    Rules:
    - Spec contains '/': treat as a HF repo path. Filenames are None
      (caller will discover via the HF API at dispatch time).
    - Otherwise: treat as a variant slug and read catalog/<slug>.json for
      `published_repo` and the `downloads[].filename` list.

    The catalog is the source here because that is where the published repo
    and quant set now live. scripts/hf_cards/<slug>.yaml used to carry
    `target_repo` and `quants[].filename`; both are derived from the catalog
    now, so a card no longer states them and parsing it finds nothing.
    """
    if "/" in spec:
        return spec, None

    record_path = root / "catalog" / f"{spec}.json"
    if not record_path.exists():
        raise SystemExit(
            f"no catalog record at {record_path}; pass a HF repo path "
            f"(e.g. handy-computer/{spec}-gguf) if the variant isn't in the "
            f"catalog yet"
        )
    record = json.loads(record_path.read_text())
    repo = record.get("published_repo")
    if not repo:
        raise SystemExit(f"catalog record {spec!r}: no published_repo")
    filenames = [d["filename"] for d in record.get("downloads", []) if d.get("filename")]
    if not filenames:
        raise SystemExit(f"catalog record {spec!r} has no downloads[].filename entries")
    return repo, filenames
