# Install rules for the transcribe-cpp-native Python provider wheel.
#
# Only included when scikit-build-core drives the build (SKBUILD is set, see
# the gate in the top-level CMakeLists.txt). Everything the wheel ships is
# COMPONENT wheel; pyproject.toml restricts packaging to that component, so
# the vendored ggml's default-component install rules (headers, cmake config,
# pkg-config) never leak into the wheel.
#
# The artifact directory is FLAT: every shared library and ggml backend module
# sits next to libtranscribe in transcribe_cpp_native/_native/, resolved via
# $ORIGIN / @loader_path rpaths (Windows uses os.add_dll_directory in the
# binding's loader). transcribe_init_backends() scans exactly this directory
# for backend modules, so module search never escapes the package.

# --- the library set --------------------------------------------------------
# ggml maintains GGML_AVAILABLE_BACKENDS (internal cache) as backends register;
# in GGML_BACKEND_DL builds these are MODULE libraries (the provider-directory
# shape), otherwise ordinary shared libraries linked into libggml.
set(_wheel_targets transcribe ggml ggml-base)
foreach(_backend IN LISTS GGML_AVAILABLE_BACKENDS)
    if(TARGET ${_backend})
        list(APPEND _wheel_targets ${_backend})
    endif()
endforeach()
list(REMOVE_DUPLICATES _wheel_targets)

foreach(_tgt IN LISTS _wheel_targets)
    # Strip VERSION/SOVERSION decoration inside the wheel: versioned library
    # names install as symlink chains (libggml.so -> libggml.so.0 -> ...), and
    # wheels cannot carry symlinks — zip would materialize each link as a full
    # duplicate copy. Undecorated names also make the SONAME/install-name the
    # plain file name, so sibling resolution needs nothing but the rpath.
    # Pre-1.0 the binding's exact version/header-hash check is the real
    # compatibility gate, not the SONAME.
    set_property(TARGET ${_tgt} PROPERTY VERSION)
    set_property(TARGET ${_tgt} PROPERTY SOVERSION)
    if(APPLE)
        set_property(TARGET ${_tgt} PROPERTY INSTALL_RPATH "@loader_path")
    else()
        set_property(TARGET ${_tgt} PROPERTY INSTALL_RPATH "$ORIGIN")
    endif()
    install(TARGETS ${_tgt}
        # Shared libraries / backend modules (RUNTIME = DLLs on Windows).
        LIBRARY DESTINATION _native COMPONENT wheel
        RUNTIME DESTINATION _native COMPONENT wheel
        # Windows import .lib stubs: link-time only, never packaged.
        ARCHIVE DESTINATION _native-dev COMPONENT wheel-dev)
endforeach()

# --- the build-time contract stamp (_contract.py) ---------------------------
# The binding hard-fails on a provider whose version or public-header hash
# disagrees with it (_library.validate_contract). Version comes from the
# header macros via PROJECT_VERSION; the header hash is the digest the FFI
# generator emitted into the committed _generated.py (drift-gated in CI), so
# wheel and binding are stamped from the same source.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python/src/transcribe_cpp/_generated.py"
     _hash_line REGEX "^PUBLIC_HEADER_HASH = ")
string(REGEX MATCH "\"([0-9a-f]+)\"" _ "${_hash_line}")
set(_public_header_hash "${CMAKE_MATCH_1}")
if(NOT _public_header_hash MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
        "python-wheel-install: failed to parse PUBLIC_HEADER_HASH from "
        "bindings/python/src/transcribe_cpp/_generated.py")
endif()

# Backend kinds this artifact supports, for the descriptor's `backends` field
# (and the binding's accelerated-first provider ranking). Derived from the
# ggml backend target names; CPU ISA variants (ggml-cpu-haswell, ...) all
# collapse to "cpu".
set(_kinds "")
foreach(_backend IN LISTS GGML_AVAILABLE_BACKENDS)
    string(REGEX REPLACE "^ggml-" "" _kind "${_backend}")
    string(REGEX REPLACE "^cpu-.*$" "cpu" _kind "${_kind}")
    list(APPEND _kinds "${_kind}")
endforeach()
list(REMOVE_DUPLICATES _kinds)
# Accelerated kinds first, cpu last — readability only; ranking is the
# binding's job.
set(_backend_kinds "")
foreach(_kind IN ITEMS cuda metal vulkan)
    if(_kind IN_LIST _kinds)
        list(APPEND _backend_kinds "${_kind}")
        list(REMOVE_ITEM _kinds "${_kind}")
    endif()
endforeach()
list(REMOVE_ITEM _kinds cpu)
list(APPEND _backend_kinds ${_kinds} cpu)

list(JOIN _backend_kinds "\", \"" _backends_joined)
set(TRANSCRIBE_WHEEL_BACKENDS_PY "\"${_backends_joined}\"")
set(TRANSCRIBE_PUBLIC_HEADER_HASH "${_public_header_hash}")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/python-native/_contract.py.in"
    "${CMAKE_CURRENT_BINARY_DIR}/python-wheel/_contract.py"
    @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/python-wheel/_contract.py"
        DESTINATION . COMPONENT wheel)

message(STATUS
    "python wheel: transcribe-cpp-native ${PROJECT_VERSION} "
    "(header ${_public_header_hash}, backends: ${_backend_kinds})")
