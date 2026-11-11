"""Hugging Face source resolution shared across converters.

Every HF-sourced converter needs the same three things: decide whether a
`--model` argument is a repo id or a local path, download a snapshot into
`$TRANSCRIBE_MODELS_DIR/<slug>/` (`<slug>@<commit>/` for a pinned commit,
falling back to the HF cache), and hand back a local directory. This
module is the single copy of that logic; before it, `_download_snapshot` /
`_looks_like_repo_id` / `hf_resolve` were duplicated near-verbatim across a
dozen converters.

NeMo-sourced families (parakeet, canary) resolve checkpoints through
`ASRModel.from_pretrained` instead and do not use this module.
"""

from __future__ import annotations

import os
import re
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
    `<models_dir>/<slug>/`, or in `<models_dir>/<slug>@<revision>/` when
    `revision` is a commit hash, full or abbreviated; otherwise
    `snapshot_download` uses the default HF cache and returns whatever local
    path it resolves to. Call this after the caller has already decided
    `repo_id` is a repo (see `looks_like_repo_id`).

    When the Hub is unreachable or HF_HUB_OFFLINE=1 is set,
    `snapshot_download` returns a non-empty local directory without checking
    what it holds. A directory per commit holds only the files
    huggingface_hub downloaded at that commit, so a pinned load never reads
    a file of another commit or one another tool saved beside the snapshot,
    and an absent commit raises LocalEntryNotFoundError that names the
    directory. The commit directories sit beside `<slug>/`, not in it, so an
    unpinned load never mistakes them for its snapshot. A branch or tag
    named like a hash (7 to 40 lowercase hex characters) is taken for a
    commit and gets such a directory too, which holds files of more than
    one commit once that ref moves.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    slug = slug_from_repo_id(repo_id)
    models_root = os.environ.get("TRANSCRIBE_MODELS_DIR")
    local_dir = None
    per_commit = bool(revision and _COMMIT_HASH.fullmatch(revision))
    if models_root:
        local_dir = Path(models_root) / (f"{slug}@{revision}" if per_commit else slug)
        local_dir.mkdir(parents=True, exist_ok=True)
    if revision:
        print(f"Downloading {repo_id}@{revision} from Hugging Face...")
    else:
        print(f"Downloading {repo_id} from Hugging Face "
              f"(no revision pin; reproducibility depends on upstream)...")
    try:
        resolved = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir) if local_dir is not None else None,
        )
    except LocalEntryNotFoundError as e:
        if local_dir is None or not per_commit:
            raise
        raise LocalEntryNotFoundError(
            f"{repo_id}@{revision} is not in {local_dir}, and HF_HUB_OFFLINE is "
            f"set or the Hugging Face Hub cannot be reached. A pinned commit is read "
            f"only from its own directory, not from {local_dir.parent / slug}/. Unset "
            f"HF_HUB_OFFLINE if it is set, or restore network access, and rerun to "
            f"download it there."
        ) from e
    return Path(resolved)


# A full or abbreviated commit hash, as the Hub accepts in `revision`.
_COMMIT_HASH = re.compile(r"[0-9a-f]{7,40}")


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
