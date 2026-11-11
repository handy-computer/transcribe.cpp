#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "gguf>=0.10",
#   "huggingface-hub>=0.30",
#   "numpy",
# ]
# ///
"""Unit tests for lib.hf_source.download_snapshot with $TRANSCRIBE_MODELS_DIR
set and the Hub unreachable, where huggingface_hub returns an existing
non-empty local directory without checking what it holds.

Run standalone (exit-code driven):   uv run scripts/lib/test_hf_source.py
Or under pytest:                      pytest scripts/lib/test_hf_source.py
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Read by huggingface_hub at import; lib.hf_source imports it lazily.
os.environ["HF_HUB_OFFLINE"] = "1"

from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo scripts/
from lib.hf_source import download_snapshot  # noqa: E402

_PINNED = "1111111111111111111111111111111111111111"
_OTHER = "2222222222222222222222222222222222222222"


def _add_file(directory: Path, name: str, record_commit: str | None = None) -> None:
    """Write `name` into `directory`, with a huggingface_hub download record
    for `record_commit` unless it is None."""
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    if record_commit is not None:
        meta = directory / ".cache" / "huggingface" / "download" / f"{name}.metadata"
        meta.parent.mkdir(parents=True, exist_ok=True)
        meta.write_text(f"{record_commit}\netag\n{path.stat().st_mtime}\n", encoding="utf-8")


class DownloadSnapshotLocalDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.models = Path(tmp.name)
        self.slug_dir = self.models / "model"
        env = mock.patch.dict(os.environ, {"TRANSCRIBE_MODELS_DIR": str(self.models)})
        env.start()
        self.addCleanup(env.stop)

    def commit_dir(self, revision: str) -> Path:
        return self.models / f"model@{revision}"

    def test_a_pinned_commit_is_read_from_its_own_directory(self):
        _add_file(self.commit_dir(_PINNED), "config.json", _PINNED)
        self.assertEqual(download_snapshot("org/model", _PINNED), self.commit_dir(_PINNED).resolve())

    def test_files_in_the_repo_directory_are_not_a_snapshot_of_a_pinned_commit(self):
        # A recorded config.json with an unrecorded checkpoint and a converted
        # GGUF, as an earlier layout or another tool leaves in <slug>/.
        _add_file(self.slug_dir, "config.json", _PINNED)
        _add_file(self.slug_dir, "pytorch_model.bin")
        _add_file(self.slug_dir, "model-F32.gguf")
        with self.assertRaisesRegex(
            LocalEntryNotFoundError,
            f"is not in {re.escape(str(self.commit_dir(_PINNED)))}, .* Unset HF_HUB_OFFLINE",
        ):
            download_snapshot("org/model", _PINNED)

    def test_the_directory_of_another_commit_is_not_used(self):
        _add_file(self.commit_dir(_OTHER), "config.json", _OTHER)
        with self.assertRaisesRegex(LocalEntryNotFoundError,
                                    f"is not in {re.escape(str(self.commit_dir(_PINNED)))}"):
            download_snapshot("org/model", _PINNED)

    def test_an_abbreviated_commit_is_not_read_from_the_repo_directory(self):
        _add_file(self.slug_dir, "config.json", _OTHER)
        with self.assertRaisesRegex(LocalEntryNotFoundError,
                                    f"is not in {re.escape(str(self.commit_dir('e37978b')))}"):
            download_snapshot("org/model", "e37978b")

    def test_a_branch_revision_reads_the_repo_directory(self):
        _add_file(self.slug_dir, "config.json", _OTHER)
        self.assertEqual(download_snapshot("org/model", "main"), self.slug_dir.resolve())

    def test_a_branch_revision_does_not_take_a_commit_directory_for_its_snapshot(self):
        _add_file(self.commit_dir(_PINNED), "config.json", _PINNED)
        with self.assertRaises(LocalEntryNotFoundError) as caught:
            download_snapshot("org/model", "main")
        self.assertNotIn("pinned commit", str(caught.exception))

    def test_six_hex_characters_are_read_as_a_branch(self):
        _add_file(self.slug_dir, "config.json", _OTHER)
        self.assertEqual(download_snapshot("org/model", "abcdef"), self.slug_dir.resolve())

    def test_forty_one_hex_characters_are_read_as_a_branch(self):
        _add_file(self.slug_dir, "config.json", _OTHER)
        self.assertEqual(download_snapshot("org/model", _PINNED + "1"), self.slug_dir.resolve())


if __name__ == "__main__":
    unittest.main()
