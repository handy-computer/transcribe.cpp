"""Backend-module loading and device discovery.

No model required: importing the package already called
``transcribe_init_backends`` on the artifact directory (a no-op for
compiled-in builds, the module-loading step for dynamic-backend builds).
These pin the discovery surface and the degradation contract on whatever
build configuration the suite runs against.
"""

from __future__ import annotations

import pytest

import transcribe_cpp as t
from transcribe_cpp import _library


def test_at_least_one_device_registered():
    devices = t.backends()
    assert devices, "no compute devices registered — nothing could run"
    for dev in devices:
        assert dev.name
        assert dev.kind in {
            "cpu", "accel", "metal", "vulkan", "cuda", "sycl", "gpu", "unknown"
        }


def test_cpu_always_available():
    # Every shipped configuration includes a CPU backend (compiled in or as
    # the baseline module); a process without one is mispackaged.
    assert t.backend_available("cpu") is True
    assert t.backend_available("auto") is True
    assert any(d.kind == "cpu" for d in t.backends())


def test_unavailable_backend_answers_false_not_error():
    # The degradation probe: asking about every known kind must answer a
    # bool, never raise — this is what turns backend="vulkan" on a machine
    # without Vulkan into a clear Python-level answer.
    for kind in ("cpu", "metal", "vulkan", "cuda", "cpu_accel"):
        assert t.backend_available(kind) in (True, False)


def test_unknown_backend_kind_raises():
    with pytest.raises(t.InvalidArgument, match="unknown backend"):
        t.backend_available("quantum")


def test_available_kinds_match_device_list():
    kinds = {d.kind for d in t.backends()}
    for request, device_kind in (("metal", "metal"), ("vulkan", "vulkan"),
                                 ("cuda", "cuda")):
        assert t.backend_available(request) == (device_kind in kinds)


def test_artifact_dir_is_library_dir():
    # The loader records where the native artifact lives; backend modules
    # are loaded from there and nowhere else (package-local contract).
    adir = _library.artifact_dir()
    assert adir is not None
    assert adir == _library._artifact_dir
    import pathlib

    assert pathlib.Path(t.library_path()).parent == adir


def test_init_backends_rejects_bad_dirs():
    lib = t._lib
    # NULL/empty/missing directory: clean status codes, never a crash.
    assert lib.transcribe_init_backends(None) != 0
    assert lib.transcribe_init_backends(b"") != 0
    assert lib.transcribe_init_backends(b"/nonexistent-transcribe-dir") != 0


def test_init_backends_idempotent():
    lib = t._lib
    adir = str(_library.artifact_dir()).encode("utf-8")
    n = lib.transcribe_backend_device_count()
    assert lib.transcribe_init_backends(adir) == 0
    assert lib.transcribe_backend_device_count() == n  # no re-registration
