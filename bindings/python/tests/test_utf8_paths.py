"""Non-ASCII (UTF-8) path handling through the C ABI, end to end.

Regression coverage for Handy issue #1585: on Windows, model loading and
backend artifact-dir validation failed for any path containing a
non-ASCII character (e.g. ``C:\\Users\\Jerôme\\...``) because the
library's narrow ``::stat()`` / ``ifstream`` calls read the UTF-8 path
bytes in the process ANSI code page. src/transcribe-path.h now routes
all release-path file access through an explicit UTF-8 -> wide
conversion on Windows.

The C++ counterpart (tests/utf8_path_unit.cpp) pins the same contract
white-box, but ctest only runs on Linux/macOS in CI. THIS file is what
executes on a real Windows runner — the wheel lanes run the full pytest
suite via scripts/ci/wheel_smoke.py — so it is the test that actually
exercises the conversion on the platform the bug lives on.

Two tiers, mirroring the suite convention:

  - Model-free tests (junk bytes / missing file / init_backends) always
    run. The junk-file case is the exact #1585 failure shape: the file
    EXISTS at the non-ASCII path, so anything but "found and rejected
    on content" is a path-encoding bug.
  - The full-load test needs a real GGUF and skips when absent
    (``TRANSCRIBE_SMOKE_MODEL``, set by the wheel lanes).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import transcribe_cpp as t
from transcribe_cpp import errors

# Same characters as the C++ counterpart's directory name.
NON_ASCII_DIR = "transcribe-utf8-Jerôme-日本語"


def _nonascii_dir(tmp_path: Path) -> Path:
    d = tmp_path / NON_ASCII_DIR
    try:
        d.mkdir()
    except OSError as e:  # filesystem that cannot represent the name
        pytest.skip(f"cannot create non-ASCII directory: {e}")
    return d


def test_missing_file_in_nonascii_dir_raises_file_not_found(tmp_path):
    # The not-found classification must survive the wide-path port: a
    # genuinely absent file is FILE_NOT_FOUND, not some generic error.
    d = _nonascii_dir(tmp_path)
    with pytest.raises(t.ModelFileNotFound) as ei:
        t.Model(d / "definitely-missing.gguf")
    assert ei.value.status == errors.ERR_FILE_NOT_FOUND


def test_junk_file_in_nonascii_dir_is_found_then_rejected(tmp_path):
    # THE #1585 shape: the file exists at the non-ASCII path. Pre-fix
    # Windows raised ModelFileNotFound here (ANSI-code-page stat could
    # not see the file). ModelLoadError — a sibling class, so this
    # cannot pass via FILE_NOT_FOUND — proves the file was found,
    # opened, and rejected on content.
    d = _nonascii_dir(tmp_path)
    junk = d / "junk.gguf"
    junk.write_bytes(b"this is not a gguf file" * 64)
    with pytest.raises(t.ModelLoadError):
        t.Model(junk)


def test_smoke_model_loads_from_nonascii_dir(model_path, tmp_path):
    # Full successful load through the non-ASCII path: existence
    # pre-check, magic sniff, gguf parse, and the tensor-data streaming
    # reopen all consume the path. Constructing the Model is the
    # assertion.
    d = _nonascii_dir(tmp_path)
    local = d / model_path.name
    shutil.copyfile(model_path, local)
    with t.Model(local):
        pass


def test_init_backends_nonascii_dirs(tmp_path):
    # Multiple init_backends calls are tolerated (test_backends.py
    # already relies on this post-import). The existing non-ASCII dir
    # must not be misread as absent; FILE_NOT_FOUND specifically is the
    # #1585 encoding bug ("...is not an existing directory"). We do not
    # assert OK: a module-less dir may legitimately report ERR_BACKEND
    # depending on build posture.
    lib = t._lib
    d = _nonascii_dir(tmp_path)
    st = lib.transcribe_init_backends(os.fspath(d).encode("utf-8"))
    assert st != errors.ERR_FILE_NOT_FOUND

    missing = d / "missing-subdir"
    st = lib.transcribe_init_backends(os.fspath(missing).encode("utf-8"))
    assert st == errors.ERR_FILE_NOT_FOUND
