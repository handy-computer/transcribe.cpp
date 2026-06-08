from __future__ import annotations

import hashlib
import os
from pathlib import Path


def source_fingerprint(root: Path) -> str:
    """Hash of the C++ source set that affects the build binary.

    Includes CMakeLists.txt content + the path list of C++ source/header
    files. Does NOT include Python scripts (they don't affect the build).
    Both the laptop (REPO = repo root) and the container (REPO = /src) reach
    the same hash because add_local_dir's ignore list mirrors the skip set.
    """
    if not root.is_dir():
        return "unknown"
    # Mirrors the add_local_dir(ignore=...) list. envs is the big one:
    # scripts/envs/<family>/ contains tens of thousands of vendored Python
    # files. Pruning at traversal time keeps this function sub-second.
    #
    # `skip_dir_prefixes` mirrors the glob patterns in add_local_dir's
    # ignore list ("build-*/**", ...) -- the bare set above only catches
    # literal name matches, which leaves stale `build-vulkan/`,
    # `build-san/`, etc. visible to the laptop while the container's
    # add_local_dir excludes them. The two enumerators must agree or
    # the build-cache lookup downstream looks at a different SRC_FP
    # than the build wrote.
    skip_dirs = {"build", ".cache", ".git", ".venv", "__pycache__",
                 "reports", "models", "samples", "envs", "node_modules"}
    skip_dir_prefixes = ("build-",)
    h = hashlib.sha256()
    entries: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_dirs
                       and not d.startswith(".")
                       and not any(d.startswith(p) for p in skip_dir_prefixes)]
        for fn in filenames:
            p = Path(dirpath, fn)
            entries.append((str(p.relative_to(root)), p))
    entries.sort()
    for rel, p in entries:
        rel_b = rel.encode()
        if p.name == "CMakeLists.txt":
            h.update(b"cm\0"); h.update(rel_b); h.update(b"\0")
            h.update(p.read_bytes()); h.update(b"\0")
        elif p.suffix in (".cpp", ".cu", ".h", ".hpp", ".cuh"):
            # Content is hashed: source edits rotate BUILD_DIR, sidestepping
            # ninja mtime quirks when /src is uploaded with reset timestamps.
            h.update(b"src\0"); h.update(rel_b); h.update(b"\0")
            h.update(p.read_bytes()); h.update(b"\0")
    return h.hexdigest()[:12]


def hyp_extra_hash(root: Path) -> str:
    """Hash of the Python pieces that affect hyp output: run.py + ingest.py.

    Folded with SRC_FP into the hyp cache key. Edits to the dispatcher
    (modal_sweep.py) or to local-only scripts (score.py) do NOT invalidate
    the hyp cache because they cannot change what the cell produces.
    """
    h = hashlib.sha256()
    for name in ("run.py", "ingest.py"):
        p = root / "scripts" / "wer" / name
        if p.is_file():
            h.update(name.encode()); h.update(b"\0")
            h.update(p.read_bytes()); h.update(b"\0")
    return h.hexdigest()[:12]


def hyp_fingerprint(root: Path, src_fp: str) -> str:
    return hashlib.sha256(
        (src_fp + hyp_extra_hash(root)).encode()
    ).hexdigest()[:12]


def env_fingerprint(root: Path, family: str, runner_rel: str) -> str:
    """Hash of what determines a family's reference venv + runner behavior:
    its pyproject.toml + uv.lock + the runner script. Computed identically on
    the laptop and in the container (ref_image keeps these files under /src)."""
    h = hashlib.sha256()
    for rel in (f"scripts/envs/{family}/pyproject.toml",
                f"scripts/envs/{family}/uv.lock",
                runner_rel):
        p = root / rel
        h.update(rel.encode()); h.update(b"\0")
        if p.is_file():
            h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]
