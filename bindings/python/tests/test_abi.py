"""No-model tests: ABI layout, version gate, and status-code/enum agreement.

These run anywhere the native library loads — no GGUF required. Importing
``transcribe_cpp`` already ran the loader, ``_abi.verify_layouts``, and the
version gate; here we pin each surface explicitly so a regression names itself.
"""

from __future__ import annotations

import pytest

import transcribe_cpp as t
from transcribe_cpp import _abi, _generated, errors


def test_native_version_matches_binding():
    # The import-time gate compares *base* versions; assert the same here so a
    # base mismatch is a named failure rather than a bare ImportError.
    assert t._base_version(t.native_version()) == t._base_version(t.__version__)


def test_abi_layouts_reverify():
    # Idempotent: import did this once; a second pass must still agree.
    _abi.verify_layouts(t._lib)


def test_every_abi_struct_known_to_native():
    assert _generated.ABI_STRUCT_IDS, "no ABI struct ids generated"
    for name, abi_id in _generated.ABI_STRUCT_IDS.items():
        assert t._lib.transcribe_abi_struct_size(abi_id) > 0, name


def test_struct_layout_offsets_present():
    # A guard against an empty/garbage generated layout silently passing.
    n_fields = sum(len(lo["offsets"]) for lo in _generated.STRUCT_LAYOUT.values())
    assert len(_generated.STRUCT_LAYOUT) >= 10
    assert n_fields >= 50


@pytest.mark.parametrize(
    "version,base",
    [
        ("0.0.1", "0.0.1"),
        ("0.0.1.post1", "0.0.1"),
        ("0.0.1.post42", "0.0.1"),
        ("0.1.0.dev3", "0.1.0"),
        ("1.2.3rc1", "1.2.3"),
        ("1.2.3a4", "1.2.3"),
        ("0.0.1+local.build", "0.0.1"),
        ("2.0.0", "2.0.0"),
    ],
)
def test_base_version_strips_suffixes(version, base):
    assert t._base_version(version) == base


# --- status-code coverage (M2) ---------------------------------------------
#
# errors.py imports its status constants FROM the generated layer (single
# source — nothing hand-kept to drift). What still needs pinning is COVERAGE:
# every status the native header defines must be aliased in errors.py and
# (for errors) mapped to an exception class, so adding a code natively fails
# CI until the Python layer handles it deliberately.

_STATUS_NAMES = [n for n in dir(errors) if n == "OK" or n.startswith("ERR_")]


def test_status_constants_discovered():
    # transcribe_status has OK + 18 error codes as of this writing; never let
    # this collapse to a near-empty set that would make the checks vacuous.
    assert len(_STATUS_NAMES) >= 19


def test_every_generated_status_is_aliased_and_mapped():
    # Every TRANSCRIBE_OK / TRANSCRIBE_ERR_* the native library exposes must
    # have an errors.py alias AND (for errors) a mapped exception.
    py_values = {getattr(errors, n) for n in _STATUS_NAMES}
    for gen_name in dir(_generated):
        if gen_name == "TRANSCRIBE_OK" or gen_name.startswith("TRANSCRIBE_ERR_"):
            value = getattr(_generated, gen_name)
            assert value in py_values, f"{gen_name} not aliased in errors.py"
            if gen_name != "TRANSCRIBE_OK":
                assert value in errors._STATUS_TO_EXC, (
                    f"{gen_name} has no mapped exception in errors._STATUS_TO_EXC"
                )
