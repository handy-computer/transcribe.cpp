from __future__ import annotations

import os
import shutil

from dataset_specs import dataset_id


def prepare_work(spec: str | None = None) -> str:
    """Build a writable /work mirror of /src with samples/wer/* pointed at the
    /data volume. Returns the manifest path for the given spec (if any)."""
    if os.path.exists("/work"):
        shutil.rmtree("/work")
    shutil.copytree("/src", "/work")
    os.makedirs("/data/wer", exist_ok=True)
    samples_wer = "/work/samples/wer"
    os.makedirs(samples_wer, exist_ok=True)

    # Per-known-subdir symlinks. Keeps unrelated /src/samples/wer/* untouched.
    for sub in ("raw",):
        link, target = os.path.join(samples_wer, sub), f"/data/wer/{sub}"
        os.makedirs(target, exist_ok=True)
        replace_with_symlink(link, target)

    if spec is not None:
        ds_id = dataset_id(spec)
        for sub in (ds_id,):
            link, target = os.path.join(samples_wer, sub), f"/data/wer/{sub}"
            os.makedirs(target, exist_ok=True)
            replace_with_symlink(link, target)
        manifest_link = os.path.join(samples_wer, f"{ds_id}.manifest.jsonl")
        manifest_target = f"/data/wer/{ds_id}.manifest.jsonl"
        replace_with_symlink(manifest_link, manifest_target)
        return manifest_target
    return ""


def replace_with_symlink(link: str, target: str) -> None:
    # The rmtree branch only fires on real dirs at /work/samples/wer/<sub>; the
    # add_local_dir ignore list keeps those paths empty in /src, so we are only
    # deleting the placeholder dir created moments earlier by prepare_work.
    if os.path.lexists(link):
        if os.path.islink(link) or os.path.isfile(link):
            os.unlink(link)
        elif os.path.isdir(link):
            shutil.rmtree(link)
    os.symlink(target, link)
