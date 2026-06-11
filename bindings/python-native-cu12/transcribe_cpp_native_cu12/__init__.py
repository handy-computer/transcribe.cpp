"""transcribe-cpp-native-cu12: CUDA 12 native artifact for transcribe-cpp.

No Python API. Bundles libtranscribe + ggml backend modules (CPU + CUDA) in
``_native/`` and advertises them via the ``transcribe_cpp.native`` entry
point. ``_contract.py`` is stamped at build time by CMake
(cmake/python-wheel-install.cmake).

The CUDA runtime is NOT vendored in the wheel: ``prepare()`` (invoked by the
binding after this provider is *selected*, before any dlopen) loads
cudart/cublas from the NVIDIA runtime wheels — they install under
``site-packages/nvidia/*/lib``, which the platform loader never searches.
``libcuda.so.1`` is the system driver; when absent, the ggml-cuda module
quietly fails to load and CPU keeps working.
"""

from __future__ import annotations

from pathlib import Path

#: Versioned sonames the ggml-cuda module links, in dependency order
#: (cublasLt before cublas), relative to the nvidia namespace package.
_NVIDIA_LIBS = (
    "cuda_runtime/lib/libcudart.so.12",
    "cublas/lib/libcublasLt.so.12",
    "cublas/lib/libcublas.so.12",
)


def prepare() -> None:
    """Load the NVIDIA runtime libraries into the process (RTLD_GLOBAL) so
    the ggml-cuda module's DT_NEEDED entries resolve from the loaded set."""
    import ctypes

    try:
        import nvidia
    except ImportError:
        # Runtime wheels absent (e.g. --no-deps install): the cuda module
        # will fail to load quietly and CPU still works.
        return
    for root in map(Path, nvidia.__path__):
        for rel in _NVIDIA_LIBS:
            lib = root / rel
            if lib.is_file():
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # Same posture as a missing driver: degrade, don't break
                    # the process for the CPU path.
                    pass


def descriptor() -> dict:
    """The transcribe_cpp.native provider contract for this package."""
    from . import _contract

    return {
        "name": "transcribe-cpp-native-cu12",
        "artifact_dir": str(Path(__file__).resolve().parent / "_native"),
        "version": _contract.VERSION,
        "header_hash": _contract.PUBLIC_HEADER_HASH,
        "backends": list(_contract.BACKENDS),
        "prepare": prepare,
    }
