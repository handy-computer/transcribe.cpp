"""Unit tests for native-provider discovery and selection (_library).

No real provider package is installed in CI, so these drive the selection and
contract machinery with synthetic descriptors — the logic that decides which
artifact loads and rejects an ABI/version-mismatched provider before dlopen.
"""

from __future__ import annotations

import pytest

import transcribe_cpp as t
from transcribe_cpp import _generated, _library
from transcribe_cpp.errors import TranscribeError

GOOD_HASH = _generated.PUBLIC_HEADER_HASH
API_BASE = _library._api_version_base()


def make(name, backends, *, version=None, header_hash=GOOD_HASH,
         library_path="/x/libtranscribe.so", artifact_dir=None):
    return _library._Provider(
        name,
        {
            "name": name,
            "backends": backends,
            "version": version if version is not None else (API_BASE or "0.0.1"),
            "header_hash": header_hash,
            "library_path": library_path,
            "artifact_dir": artifact_dir,
        },
    )


# --- descriptor normalization & matching ----------------------------------


def test_descriptor_from_mapping_and_object():
    class Obj:
        name = "transcribe-cpp-native-cpu"
        backends = ["cpu"]
        version = "0.0.1"
        header_hash = GOOD_HASH
        library_path = "/x/libtranscribe.so"
        artifact_dir = None

    p = _library._Provider("cpu", Obj())
    assert p.name == "transcribe-cpp-native-cpu"
    assert p.backends == ("cpu",)
    assert p.library_path().name.startswith("libtranscribe")


def test_library_path_from_artifact_dir():
    p = make("transcribe-cpp-native-cpu", ["cpu"],
             library_path=None, artifact_dir="/opt/art")
    assert str(p.library_path()).startswith("/opt/art")


def test_library_path_requires_a_source():
    p = make("x", ["cpu"], library_path=None, artifact_dir=None)
    with pytest.raises(TranscribeError):
        p.library_path()


@pytest.mark.parametrize(
    "request_str,expected",
    [
        ("transcribe-cpp-native-cpu", True),
        ("cpu", True),  # backend kind and name suffix
        ("metal", False),
        ("transcribe-cpp-native-metal", False),
    ],
)
def test_matches_request(request_str, expected):
    p = make("transcribe-cpp-native-cpu", ["cpu"])
    assert p.matches_request(request_str) is expected


def test_rank_orders_accelerated_above_cpu():
    assert make("m", ["metal", "cpu"]).rank == 3
    assert make("v", ["vulkan"]).rank == 2
    assert make("c", ["cpu"]).rank == 1
    assert make("u", ["wat"]).rank == 0


# --- selection policy ------------------------------------------------------


def test_select_prefers_accelerated():
    cpu = make("transcribe-cpp-native-cpu", ["cpu"])
    metal = make("transcribe-cpp-native-metal", ["metal", "cpu"])
    assert _library._select_provider([cpu, metal], None) is metal


def test_select_is_deterministic_on_ties():
    a = make("transcribe-cpp-native-cpu", ["cpu"])
    b = make("transcribe-cpp-native-cpu2", ["cpu"])
    # Same rank → alphabetical by name, stable regardless of input order.
    assert _library._select_provider([a, b], None).name == "transcribe-cpp-native-cpu"
    assert _library._select_provider([b, a], None).name == "transcribe-cpp-native-cpu"


def test_select_explicit_request():
    cpu = make("transcribe-cpp-native-cpu", ["cpu"])
    metal = make("transcribe-cpp-native-metal", ["metal", "cpu"])
    assert _library._select_provider([cpu, metal], "cpu") is cpu


def test_select_none_when_no_providers():
    assert _library._select_provider([], None) is None


def test_select_unknown_request_raises():
    cpu = make("transcribe-cpp-native-cpu", ["cpu"])
    with pytest.raises(TranscribeError, match="not installed"):
        _library._select_provider([cpu], "vulkan")


def test_select_broken_only_raises():
    broken = _library._BrokenProvider("transcribe-cpp-native-cpu", RuntimeError("boom"))
    with pytest.raises(TranscribeError, match="failed to load"):
        _library._select_provider([broken], None)


def test_select_request_for_broken_raises():
    broken = _library._BrokenProvider("transcribe-cpp-native-cpu", RuntimeError("boom"))
    with pytest.raises(TranscribeError, match="failed to load"):
        _library._select_provider([broken], "transcribe-cpp-native-cpu")


# --- contract validation ---------------------------------------------------


def test_contract_ok_for_matching_provider():
    make("transcribe-cpp-native-cpu", ["cpu"]).validate_contract()  # no raise


def test_contract_rejects_header_hash_mismatch():
    p = make("transcribe-cpp-native-cpu", ["cpu"], header_hash="deadbeefdeadbeef")
    with pytest.raises(TranscribeError, match="different\\s+public ABI"):
        p.validate_contract()


def test_contract_skips_hash_check_when_provider_omits_it():
    # A provider that declares no header hash is trusted on ABI (the import-time
    # struct-layout check in _abi is still the deep backstop).
    make("transcribe-cpp-native-cpu", ["cpu"], header_hash=None).validate_contract()


@pytest.mark.skipif(API_BASE is None, reason="API distribution not installed")
def test_contract_rejects_version_base_mismatch():
    major = int(API_BASE.split(".")[0])
    bumped = f"{major + 1}.0.0"
    p = make("transcribe-cpp-native-cpu", ["cpu"], version=bumped)
    with pytest.raises(TranscribeError, match="requires a matching"):
        p.validate_contract()


@pytest.mark.skipif(API_BASE is None, reason="API distribution not installed")
def test_contract_accepts_post_release_provider():
    # A .postN packaging fix on the provider keeps the same base and must load.
    p = make("transcribe-cpp-native-cpu", ["cpu"], version=f"{API_BASE}.post3")
    p.validate_contract()  # no raise


def test_no_provider_installed_in_test_env():
    # Sanity: this binding ran from the dev tree, not a provider package.
    assert t.native_provider() is None
