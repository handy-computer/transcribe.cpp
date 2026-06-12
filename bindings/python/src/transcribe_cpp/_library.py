"""Locate and load the native ``transcribe`` shared library.

Two worlds load the library, and this module is the single choke point for both:

**Distribution (wheels).** The pure-Python API package carries no native code.
A *provider* package ships the artifact and advertises it through a Python
*entry point* in group ``transcribe_cpp.native``. Two providers exist today:
``transcribe-cpp-native`` (the default — platform wheels bundling CPU+Metal on
macOS arm64, CPU+Vulkan on Linux/Windows) and ``transcribe-cpp-native-cu12``
(opt-in CUDA 12, installed via the ``transcribe-cpp[cu12]`` extra). The entry
point resolves to a zero-argument callable returning a descriptor (a mapping
or an object) with the contract fields:

    name          provider distribution name (e.g. "transcribe-cpp-native-cpu")
    library_path  absolute path to the shared library to dlopen
                  (or artifact_dir + the platform filename)
    artifact_dir  directory holding the library and its sibling ggml libs
    version       native library version it was built for (base must match ours)
    header_hash   the _generated.PUBLIC_HEADER_HASH it was built against
    backends      supported backend kinds, e.g. ["cpu"] / ["metal", "cpu"]
    prepare       optional zero-argument callable invoked after this provider
                  is SELECTED and validated, before any dlopen — e.g. the cu12
                  provider preloads the NVIDIA runtime-wheel libraries
                  (site-packages/nvidia/*/lib) that the platform loader would
                  never find on its own

Selecting a provider picks which native artifact loads into the process; the
per-model ``backend=`` request is a *separate* axis resolved inside it. Selection
policy: explicit ``provider`` argument → ``TRANSCRIBE_NATIVE_PROVIDER`` env var →
best accelerated (CUDA/Metal, then Vulkan) → CPU. A discovered provider whose
declared version or header hash disagrees with this binding is a hard error
*before* dlopen — pip pins are not enough; this runtime check is the backstop.

**Development (working tree).** No provider is installed; ``TRANSCRIBE_LIBRARY``
points at a hand-built library, or one is discovered under the repo's
``build-shared/`` / ``build/`` tree. This path skips the provider contract (the
developer owns the build); the import-time ABI layout check in ``_abi`` is the
correctness backstop either way.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Callable, Iterator, Optional

from . import _generated
from .errors import TranscribeError

#: Entry-point group a native provider package registers under.
ENTRY_POINT_GROUP = "transcribe_cpp.native"

#: Backend-kind preference when auto-selecting among installed providers. Higher
#: wins; a provider's rank is the max over the kinds it advertises.
_BACKEND_RANK = {"cuda": 3, "metal": 3, "vulkan": 2, "cpu_accel": 1, "cpu": 1}

#: Set after a successful load so the package can surface it for diagnostics.
#: None means the library came from the dev-tree / explicit-path fallback.
_selected_provider: Optional[str] = None

#: Directory holding the native library and (in dynamic-backend builds) its
#: ggml backend modules. The import-time bootstrap passes this to
#: transcribe_init_backends so module loading stays package-local.
_artifact_dir: Optional[Path] = None


def selected_provider() -> Optional[str]:
    """Name of the provider package the loaded library came from, or None for a
    dev-tree / ``TRANSCRIBE_LIBRARY`` load."""
    return _selected_provider


def artifact_dir() -> Optional[Path]:
    """Directory the native library (and any backend modules) live in."""
    return _artifact_dir


def _library_filename() -> str:
    if sys.platform == "darwin":
        return "libtranscribe.dylib"
    if sys.platform == "win32":
        return "transcribe.dll"
    return "libtranscribe.so"


def _base_version(version: str) -> str:
    """Leading dotted-numeric release segment, suffix stripped (PEP 440)."""
    import re

    m = re.match(r"\d+(?:\.\d+)*", version.strip())
    return m.group(0) if m else version.strip()


# --- provider discovery ---------------------------------------------------


def _descriptor_field(descriptor, key: str):
    """Read a contract field from a mapping- or attribute-style descriptor."""
    if isinstance(descriptor, dict):
        return descriptor.get(key)
    return getattr(descriptor, key, None)


def _iter_native_entry_points():
    """Entry points in ENTRY_POINT_GROUP, across the 3.9 vs 3.10+ APIs."""
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):  # 3.10+
        return list(eps.select(group=ENTRY_POINT_GROUP))
    return list(eps.get(ENTRY_POINT_GROUP, []))  # 3.9


class _Provider:
    """A discovered, normalized provider descriptor."""

    def __init__(self, ep_name: str, descriptor):
        self.ep_name = ep_name
        self.name = _descriptor_field(descriptor, "name") or ep_name
        self.version = _descriptor_field(descriptor, "version")
        self.header_hash = _descriptor_field(descriptor, "header_hash")
        self.backends = tuple(_descriptor_field(descriptor, "backends") or ())
        self._library_path = _descriptor_field(descriptor, "library_path")
        self._artifact_dir = _descriptor_field(descriptor, "artifact_dir")
        self._prepare = _descriptor_field(descriptor, "prepare")

    def prepare(self) -> None:
        """Run the provider's post-selection hook (no-op when absent). Errors
        here are the provider's fault — surface them with its name."""
        if not callable(self._prepare):
            return
        try:
            self._prepare()
        except Exception as exc:
            raise TranscribeError(
                message=(
                    f"native provider {self.name!r} failed in its prepare() "
                    f"hook: {exc}"
                )
            ) from exc

    @property
    def rank(self) -> int:
        return max((_BACKEND_RANK.get(b, 0) for b in self.backends), default=0)

    def matches_request(self, request: str) -> bool:
        """Whether an explicit provider request names this provider — by dist
        name, by an advertised backend kind, or by the conventional
        ``…-native-<kind>`` suffix."""
        request = request.strip().lower()
        name = (self.name or "").lower()
        return (
            request == name
            or request == self.ep_name.lower()
            or request in {b.lower() for b in self.backends}
            or name.endswith(f"-{request}")
            or name.endswith(f"-native-{request}")
        )

    def library_path(self) -> Path:
        if self._library_path:
            return Path(self._library_path)
        if self._artifact_dir:
            return Path(self._artifact_dir) / _library_filename()
        raise TranscribeError(
            message=(
                f"native provider {self.name!r} supplied neither 'library_path' "
                "nor 'artifact_dir' in its entry-point descriptor"
            )
        )

    def validate_contract(self) -> None:
        """Hard-fail if the provider disagrees with this binding on ABI or
        version. Raises TranscribeError with an actionable message."""
        expected_hash = _generated.PUBLIC_HEADER_HASH
        if self.header_hash and self.header_hash != expected_hash:
            raise TranscribeError(
                message=(
                    f"native provider {self.name!r} was built against a different "
                    f"public ABI (header hash {self.header_hash!r}, this binding "
                    f"expects {expected_hash!r}). Install a provider matching "
                    "transcribe-cpp, or upgrade/downgrade transcribe-cpp to match "
                    "the provider."
                )
            )
        api_base = _api_version_base()
        if self.version and api_base and _base_version(self.version) != api_base:
            raise TranscribeError(
                message=(
                    f"native provider {self.name!r} is version {self.version}, but "
                    f"transcribe-cpp is {api_base}.x: pre-1.0 requires a matching "
                    "base (MAJOR.MINOR.PATCH). Install matching versions of "
                    "transcribe-cpp and its native provider."
                )
            )


def _api_version_base() -> Optional[str]:
    """Base version of the installed API distribution, or None if not installed
    as a distribution (a pure source/PYTHONPATH run, where no provider exists)."""
    try:
        return _base_version(importlib.metadata.version("transcribe-cpp"))
    except importlib.metadata.PackageNotFoundError:
        return None


def _discover_providers() -> list:
    providers = []
    for ep in _iter_native_entry_points():
        try:
            factory: Callable = ep.load()
            descriptor = factory() if callable(factory) else factory
            providers.append(_Provider(ep.name, descriptor))
        except Exception as exc:  # a broken provider must not hide the others
            providers.append(_BrokenProvider(ep.name, exc))
    return providers


class _BrokenProvider:
    """A provider whose entry point failed to load — kept so selection can name
    it in diagnostics rather than silently dropping it."""

    rank = -1
    backends: tuple = ()

    def __init__(self, ep_name: str, error: Exception):
        self.ep_name = ep_name
        self.name = ep_name
        self.error = error

    def matches_request(self, request: str) -> bool:
        return request.strip().lower() in (self.ep_name.lower(), self.name.lower())


def _select_provider(providers: list, request: Optional[str]) -> Optional["_Provider"]:
    """Apply the selection policy. Returns the chosen provider, or None when no
    providers are installed. Raises when an explicit request cannot be honored."""
    healthy = [p for p in providers if isinstance(p, _Provider)]

    if request:
        for p in providers:
            if p.matches_request(request):
                if isinstance(p, _BrokenProvider):
                    raise TranscribeError(
                        message=(
                            f"requested native provider {request!r} failed to "
                            f"load: {p.error}"
                        )
                    )
                return p
        installed = ", ".join(sorted(p.name for p in providers)) or "(none)"
        # Suggest a command that actually exists: cu12 is the only extra;
        # every other backend kind ships inside the default provider.
        if request.strip().lower() in ("cu12", "cuda", "transcribe-cpp-native-cu12"):
            hint = 'Install it with: pip install "transcribe-cpp[cu12]".'
        else:
            hint = (
                "The default provider (pip install transcribe-cpp) bundles "
                "cpu plus the platform accelerator (metal on macOS arm64, "
                "vulkan on Linux/Windows)."
            )
        raise TranscribeError(
            message=(
                f"requested native provider {request!r} is not installed. "
                f"Installed providers: {installed}. {hint}"
            )
        )

    if not providers:
        return None
    if not healthy:
        errors = "; ".join(f"{p.name}: {p.error}" for p in providers)
        raise TranscribeError(
            message=f"every installed native provider failed to load: {errors}"
        )
    # Best accelerated first (CUDA/Metal, then Vulkan, then CPU); deterministic
    # tie-break by name so selection is stable across runs.
    healthy.sort(key=lambda p: (-p.rank, p.name))
    return healthy[0]


# --- developer / bundled fallback candidates ------------------------------


def _ascend_to_repo_root(start: Path) -> Optional[Path]:
    """Walk up from *start* to the transcribe.cpp repo root, if we are in one."""
    for parent in (start, *start.parents):
        if (parent / "CMakeLists.txt").is_file() and (
            parent / "include" / "transcribe.h"
        ).is_file():
            return parent
    return None


def _fallback_candidate_paths() -> Iterator[Path]:
    """In-package bundled artifact, then dev-tree build outputs."""
    name = _library_filename()

    pkg_dir = Path(__file__).resolve().parent
    yield pkg_dir / "_native" / name
    yield pkg_dir / name

    repo_root = _ascend_to_repo_root(pkg_dir)
    if repo_root is not None:
        for build_dir in ("build-shared", "build"):
            yield repo_root / build_dir / "src" / name
            yield repo_root / build_dir / "bin" / name


def _cdll(path: Path) -> ctypes.CDLL:
    # Windows resolves sibling DLLs (ggml, …) via the DLL search path, not
    # rpath; add the library's own directory before loading it.
    if sys.platform == "win32":
        os.add_dll_directory(str(path.parent))
    return ctypes.CDLL(str(path))


def load_library(provider: Optional[str] = None) -> tuple:
    """Return ``(CDLL, path)`` for the native library, or raise TranscribeError.

    Resolution order:

    1. ``TRANSCRIBE_LIBRARY`` — explicit path (developer escape hatch).
    2. An installed provider package (entry-point discovery + contract check).
    3. A bundled in-package ``_native/`` artifact, then a dev-tree build.

    *provider* (or ``TRANSCRIBE_NATIVE_PROVIDER``) forces a specific installed
    provider; otherwise the best accelerated one is chosen, falling back to CPU.
    """
    global _selected_provider, _artifact_dir
    _selected_provider = None
    _artifact_dir = None

    # 1. Explicit path override — bypasses provider discovery entirely.
    override = os.environ.get("TRANSCRIBE_LIBRARY")
    if override:
        path = Path(override)
        if not path.is_file():
            raise TranscribeError(
                message=f"TRANSCRIBE_LIBRARY points at a missing file: {path}"
            )
        _artifact_dir = path.parent
        return _cdll(path), path

    # 2. Installed provider packages.
    request = provider or os.environ.get("TRANSCRIBE_NATIVE_PROVIDER")
    providers = _discover_providers()
    chosen = _select_provider(providers, request)
    if chosen is not None:
        chosen.validate_contract()
        path = chosen.library_path()
        if not path.is_file():
            raise TranscribeError(
                message=(
                    f"native provider {chosen.name!r} points at a missing "
                    f"library: {path}"
                )
            )
        # Only the selected provider prepares (e.g. preloading NVIDIA runtime
        # libs) — discovery alone must stay side-effect free.
        chosen.prepare()
        cdll = _cdll(path)
        _selected_provider = chosen.name
        _artifact_dir = (
            Path(chosen._artifact_dir) if chosen._artifact_dir else path.parent
        )
        return cdll, path

    # 3. Bundled / dev-tree fallback.
    tried: list = []
    for path in _fallback_candidate_paths():
        tried.append(str(path))
        if path.is_file():
            _artifact_dir = path.parent
            return _cdll(path), path

    raise TranscribeError(
        message=(
            "could not locate the native transcribe library. Install the "
            "native provider (pip install transcribe-cpp pulls it in), set "
            "TRANSCRIBE_LIBRARY to a built library, or build a shared library "
            "(cmake -DTRANSCRIBE_BUILD_SHARED=ON). Searched:\n  "
            + "\n  ".join(tried)
        )
    )
