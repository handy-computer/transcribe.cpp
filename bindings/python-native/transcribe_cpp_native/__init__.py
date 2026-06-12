"""transcribe-cpp-native: prebuilt native artifact for the transcribe-cpp bindings.

This package contains no Python API. It bundles the native ``transcribe``
shared library (plus its ggml libraries and any dynamic backend modules) in
``_native/`` and advertises it to the pure-Python ``transcribe-cpp`` package
through the ``transcribe_cpp.native`` entry point declared in pyproject.toml.

``descriptor()`` returns the provider contract the binding validates before
dlopen — see bindings/python/src/transcribe_cpp/_library.py for the consuming
side. ``_contract.py`` is generated at build time by CMake
(cmake/python-wheel-install.cmake) and stamps the native library version, the
public-header hash the FFI layer was generated from, and the backend kinds
this artifact was built with.
"""

from __future__ import annotations

from pathlib import Path


def descriptor() -> dict:
    """The transcribe_cpp.native provider contract for this package."""
    from . import _contract

    return {
        "name": "transcribe-cpp-native",
        "artifact_dir": str(Path(__file__).resolve().parent / "_native"),
        "version": _contract.VERSION,
        "header_hash": _contract.PUBLIC_HEADER_HASH,
        "backends": list(_contract.BACKENDS),
    }
