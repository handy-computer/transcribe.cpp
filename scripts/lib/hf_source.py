"""Hugging Face source resolution shared across converters.

Every HF-sourced converter needs the same three things: decide whether a
`--model` argument is a repo id or a local path, download a snapshot into
`$TRANSCRIBE_MODELS_DIR/<slug>/` (falling back to the HF cache), and hand back
a local directory. This module is the single copy of that logic; before it,
`_download_snapshot` / `_looks_like_repo_id` / `hf_resolve` were duplicated
near-verbatim across a dozen converters.

NeMo-sourced families (parakeet, canary) resolve checkpoints through
`ASRModel.from_pretrained` instead and do not use this module.
"""

from __future__ import annotations

import os
from pathlib import Path

from lib.gguf_common import slug_from_repo_id


def looks_like_repo_id(s: str) -> bool:
    """True for an `org/name` string with no matching local filesystem path.

    `expanduser()` so a real local checkpoint like `~/models/foo` is not
    mistaken for an HF repo id (the pre-centralization copies omitted this).
    """
    return "/" in s and not Path(s).expanduser().exists()


def download_snapshot(repo_id: str, revision: str | None = None) -> Path:
    """Download an HF snapshot and return its local directory.

    When `$TRANSCRIBE_MODELS_DIR` is set the snapshot lands in
    `<models_dir>/<slug>/`; otherwise `snapshot_download` uses the default HF
    cache and returns whatever local path it resolves to. Call this after the
    caller has already decided `repo_id` is a repo (see `looks_like_repo_id`).
    """
    from huggingface_hub import snapshot_download

    slug = slug_from_repo_id(repo_id)
    models_root = os.environ.get("TRANSCRIBE_MODELS_DIR")
    local_dir = Path(models_root) / slug if models_root else None
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
    if revision:
        print(f"Downloading {repo_id}@{revision} from Hugging Face...")
    else:
        print(f"Downloading {repo_id} from Hugging Face "
              f"(no revision pin; reproducibility depends on upstream)...")
    resolved = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir) if local_dir is not None else None,
    )
    return Path(resolved)


def resolve_model_dir(model_arg: str, revision: str | None = None) -> Path:
    """Return a local directory for `model_arg`, downloading if needed.

    If `model_arg` is an existing directory it is returned as-is; otherwise it
    is treated as an HF repo id and downloaded. This is the one-call form for
    converters that do not need to distinguish the two cases at the call site.
    """
    p = Path(model_arg).expanduser()
    if p.is_dir():
        return p.resolve()
    return download_snapshot(model_arg, revision)
