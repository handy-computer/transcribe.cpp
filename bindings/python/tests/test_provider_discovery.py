"""Unit tests for native-provider discovery and selection (_library).

No real provider package is installed in CI, so these drive the selection and
contract machinery with synthetic descriptors — the logic that decides which
artifact loads and rejects an ABI/version-mismatched provider before dlopen.
"""

from __future__ import annotations

from pathlib import Path

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
    # Path comparison, not string prefix: Windows renders this with
    # backslashes (str(WindowsPath('/opt/art/...')) == '\\opt\\art\\...').
    path = p.library_path()
    assert path.parent == Path("/opt/art")
    assert path.name in ("libtranscribe.so", "libtranscribe.dylib", "transcribe.dll")


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


# --- prepare hook ------------------------------------------------------------


def test_prepare_hook_runs_when_present():
    calls = []
    p = _library._Provider(
        "cu12",
        {"name": "transcribe-cpp-native-cu12", "backends": ["cuda", "cpu"],
         "library_path": "/x/libtranscribe.so", "prepare": lambda: calls.append(1)},
    )
    p.prepare()
    assert calls == [1]


def test_prepare_hook_absent_is_noop():
    make("transcribe-cpp-native-cpu", ["cpu"]).prepare()  # no raise


def test_prepare_hook_failure_names_the_provider():
    def boom():
        raise RuntimeError("nvidia wheels missing")

    p = _library._Provider(
        "cu12",
        {"name": "transcribe-cpp-native-cu12", "backends": ["cuda"],
         "library_path": "/x/libtranscribe.so", "prepare": boom},
    )
    with pytest.raises(TranscribeError, match="transcribe-cpp-native-cu12.*prepare"):
        p.prepare()


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


def test_provider_state_is_consistent():
    # Two legitimate environments run this suite: the dev tree (no provider
    # installed / TRANSCRIBE_LIBRARY override -> native_provider() is None) and
    # a wheel-test env where a transcribe-cpp-native* package (default or an
    # accelerator variant like -cu12) supplied the library. Pin that the
    # reported state matches where the library came from.
    provider = t.native_provider()
    if provider is None:
        assert "transcribe_cpp_native" not in t.library_path()
    else:
        assert provider.startswith("transcribe-cpp-native"), provider
        assert "transcribe_cpp_native" in t.library_path()


# --- end to end, through the real entry-point machinery --------------------
#
# The unit tests above drive selection/contract with synthetic _Provider
# objects. These build an actual installed-distribution shape on disk — a
# module plus a .dist-info advertising the transcribe_cpp.native group — and
# run the full path: importlib.metadata discovery → ep.load() → descriptor →
# selection → contract → ctypes dlopen of the real library this suite loaded.


def _install_fake_provider(
    site_dir, module_name, *, header_hash=GOOD_HASH, version=API_BASE
):
    lib = t.library_path()
    dist = module_name.replace("_", "-")
    marker = site_dir / f"{module_name}.prepared"
    (site_dir / f"{module_name}.py").write_text(
        "def _prepare():\n"
        f"    open({str(marker)!r}, 'w').close()\n"
        "\n"
        "def descriptor():\n"
        "    return {\n"
        f"        'name': {dist!r},\n"
        f"        'library_path': {lib!r},\n"
        f"        'version': {version!r},\n"
        f"        'header_hash': {header_hash!r},\n"
        "        'backends': ['cpu'],\n"
        "        'prepare': _prepare,\n"
        "    }\n"
    )
    di = site_dir / f"{module_name}-0.0.1.dist-info"
    di.mkdir()
    (di / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {dist}\nVersion: 0.0.1\n"
    )
    (di / "entry_points.txt").write_text(
        f"[transcribe_cpp.native]\n{module_name} = {module_name}:descriptor\n"
    )
    return dist


@pytest.fixture
def library_globals_restored():
    """load_library mutates module globals; put them back for later tests."""
    provider, art = _library._selected_provider, _library._artifact_dir
    yield
    _library._selected_provider = provider
    _library._artifact_dir = art


def test_entry_point_end_to_end_loads(tmp_path, monkeypatch, library_globals_restored):
    dist = _install_fake_provider(tmp_path, "fake_native_provider_ok")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("TRANSCRIBE_LIBRARY", raising=False)

    # Explicit request: deterministic even if a real provider package is also
    # installed in this environment (auto-selection policy is unit-tested).
    cdll, path = _library.load_library(provider=dist)
    assert str(path) == t.library_path()
    assert _library.selected_provider() == dist
    assert _library.artifact_dir() == path.parent
    # The dlopen is real: the loaded handle exposes the C API.
    assert cdll.transcribe_version is not None
    # The selected provider's prepare() hook ran before the dlopen.
    assert (tmp_path / "fake_native_provider_ok.prepared").is_file()


def test_entry_point_end_to_end_rejects_abi_mismatch(
    tmp_path, monkeypatch, library_globals_restored
):
    dist = _install_fake_provider(
        tmp_path, "fake_native_provider_badhash", header_hash="deadbeefdeadbeef"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("TRANSCRIBE_LIBRARY", raising=False)

    # The contract check must refuse before any dlopen happens.
    with pytest.raises(TranscribeError, match="public ABI"):
        _library.load_library(provider=dist)
