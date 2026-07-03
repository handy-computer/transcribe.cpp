"""Non-ASCII (UTF-8) path handling through the C ABI.

Regression coverage for Handy issue #1585: on Windows, model loading
and backend artifact-dir validation failed for non-ASCII paths because
the library's narrow ``::stat()`` / ``ifstream`` calls read the UTF-8
path bytes in the process ANSI code page. src/transcribe-path.h now
converts UTF-8 to wide paths on Windows.

The C++ counterpart (tests/utf8_path_unit.cpp) pins the same contract,
but ctest only runs on Linux/macOS in CI — this file is what executes
on a real Windows runner (the wheel lanes run the pytest suite via
scripts/ci/wheel_smoke.py).

Model-free tests always run; the full-load test skips without a model
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
NON_ASCII_DIR = "transcribe-utf8-café-日本語"


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
    # The #1585 failure shape: the file EXISTS at the non-ASCII path,
    # but pre-fix Windows raised ModelFileNotFound (the ANSI-code-page
    # stat could not see it). ModelLoadError is a sibling class, so
    # this cannot pass via FILE_NOT_FOUND.
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
    # The existing non-ASCII dir must not be misread as absent;
    # FILE_NOT_FOUND specifically is the #1585 encoding bug. We do not
    # assert OK: a module-less dir may legitimately report ERR_BACKEND
    # depending on build posture.
    lib = t._lib
    d = _nonascii_dir(tmp_path)
    st = lib.transcribe_init_backends(os.fspath(d).encode("utf-8"))
    assert st != errors.ERR_FILE_NOT_FOUND

    missing = d / "missing-subdir"
    st = lib.transcribe_init_backends(os.fspath(missing).encode("utf-8"))
    assert st == errors.ERR_FILE_NOT_FOUND
