"""Error-mapping coverage: status codes → exception classes, end to end.

Two layers, both model-free:

  - **Unit**: ``raise_for_status`` / ``exception_for_status`` map every
    ``transcribe_status`` to its documented exception subclass with
    ``.status`` set, and unknown codes degrade to the base class instead of
    KeyError.
  - **Integration**: real native calls produce the mapped exceptions
    (missing file → ModelFileNotFound, junk bytes → ModelLoadError) — the
    paths a user actually hits first.
"""

from __future__ import annotations

import pytest

import transcribe_cpp as t
from transcribe_cpp import errors


def test_every_status_maps_to_documented_subclass():
    expected = {
        errors.ERR_INVALID_ARG: t.InvalidArgument,
        errors.ERR_NOT_IMPLEMENTED: t.NotImplementedByModel,
        errors.ERR_FILE_NOT_FOUND: t.ModelFileNotFound,
        errors.ERR_GGUF: t.ModelLoadError,
        errors.ERR_UNSUPPORTED_ARCH: t.ModelLoadError,
        errors.ERR_UNSUPPORTED_VARIANT: t.ModelLoadError,
        errors.ERR_OOM: t.OutOfMemory,
        errors.ERR_BACKEND: t.BackendError,
        errors.ERR_SAMPLE_RATE: t.InvalidArgument,
        errors.ERR_UNSUPPORTED_LANGUAGE: t.UnsupportedRequest,
        errors.ERR_UNSUPPORTED_TASK: t.UnsupportedRequest,
        errors.ERR_UNSUPPORTED_TIMESTAMPS: t.UnsupportedRequest,
        errors.ERR_ABORTED: t.Aborted,
        errors.ERR_BAD_STRUCT_SIZE: t.AbiError,
        errors.ERR_UNSUPPORTED_PNC: t.UnsupportedRequest,
        errors.ERR_UNSUPPORTED_ITN: t.UnsupportedRequest,
        errors.ERR_INPUT_TOO_LONG: t.InputTooLong,
        errors.ERR_OUTPUT_TRUNCATED: t.OutputTruncated,
    }
    # The mapping table covers every non-OK status the header defines, and
    # nothing else (a new C status must be mapped deliberately, not by
    # accident).
    assert set(errors._STATUS_TO_EXC) == set(expected)
    for status, exc_type in expected.items():
        with pytest.raises(exc_type) as ei:
            errors.raise_for_status(status, "synthetic", "ctx")
        assert ei.value.status == status
        assert isinstance(ei.value, t.TranscribeError)
        assert "ctx: synthetic" in str(ei.value)


def test_ok_does_not_raise():
    assert errors.raise_for_status(errors.OK, "ok") is None


def test_unknown_status_degrades_to_base_class():
    exc = errors.exception_for_status(999, "mystery")
    assert type(exc) is t.TranscribeError
    assert exc.status == 999


def test_exception_for_status_builds_without_raising():
    exc = errors.exception_for_status(errors.ERR_ABORTED, "aborted", "run")
    assert isinstance(exc, t.Aborted)
    assert exc.partial_result is None      # default until a caller attaches one
    assert exc.utterance_index is None


def test_status_constants_come_from_generated_layer():
    # errors.py aliases the generated constants instead of hand-mirroring
    # ints; this pins the alias so a regeneration cannot drift from it.
    from transcribe_cpp import _generated

    assert errors.ERR_ABORTED == _generated.TRANSCRIBE_ERR_ABORTED
    assert errors.ERR_OUTPUT_TRUNCATED == _generated.TRANSCRIBE_ERR_OUTPUT_TRUNCATED
    assert errors.OK == _generated.TRANSCRIBE_OK


# --- integration: real native calls raise the mapped classes ----------------


def test_missing_model_file_raises_model_file_not_found(tmp_path):
    with pytest.raises(t.ModelFileNotFound) as ei:
        t.Model(tmp_path / "definitely-missing.gguf")
    assert ei.value.status == errors.ERR_FILE_NOT_FOUND


def test_junk_model_file_raises_model_load_error(tmp_path):
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"this is not a gguf file" * 64)
    with pytest.raises(t.ModelLoadError):
        t.Model(junk)


def test_invalid_spec_k_drafts_rejected_before_native_call():
    from transcribe_cpp import _build_run_params

    with pytest.raises(t.InvalidArgument, match="spec_k_drafts"):
        _build_run_params("transcribe", None, None, "none", False, -2)
    with pytest.raises(t.InvalidArgument, match="spec_k_drafts"):
        _build_run_params("transcribe", None, None, "none", False, "many")
