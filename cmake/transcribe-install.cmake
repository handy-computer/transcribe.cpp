# Install rules for consuming the C library OUTSIDE Python packaging —
# the foundation the Rust -sys crate's build.rs (and any plain C/C++
# consumer) builds on. Included from the top-level CMakeLists.txt when
# TRANSCRIBE_INSTALL is ON (default for top-level non-SKBUILD builds; the
# Python wheel has its own component-filtered install in
# python-wheel-install.cmake).
#
# What `cmake --install` produces:
#   include/   transcribe.h, transcribe.abihash, transcribe/*.h
#              (plus ggml's public headers via ggml's own install rules)
#   lib/       libtranscribe (static or shared per TRANSCRIBE_BUILD_SHARED)
#              + ggml/ggml-base/backend libs (ggml's own install rules)
#              + transcribe-link.json (the link manifest, see below)
#
# transcribe-link.json is the machine-readable link interface for non-CMake
# consumers: which archives to link in which order, plus the system
# libraries/frameworks/flags they drag in. It is generated from the
# configured build (backend set, OpenMP/BLAS posture), not hardcoded by the
# consumer — the whisper-rs drift class this avoids. The link-smoke CI lane
# (scripts/ci/link_smoke.py) compiles a toy C consumer from NOTHING but this
# manifest, in both postures, so a wrong manifest is a red check, not a
# downstream surprise.
#
# Windows note: the static-posture system-library translation below is
# provisional (zlib via vcpkg spells differently, MSVC has no -framework/
# -lstdc++); it gets exercised and completed when the Rust branch does its
# MSVC shakeout. The smoke lanes cover linux + macos.

include(GNUInstallDirs)
include("${CMAKE_CURRENT_LIST_DIR}/transcribe-backend-kinds.cmake")

# Resolve the absolute path of the static zlib archive find_package(ZLIB) gave
# us. Used on Windows, where vcpkg spells the lib `zlib.lib` (not `z`) and a
# bare `-l z` is unsatisfiable: we stage the real archive into the lib dir and
# link it by name. Tries the imported target's per-config location first, then
# the configuration-less location, then the raw ZLIB_LIBRARY cache var.
function(_transcribe_zlib_archive out)
    set(_loc "")
    # src/CMakeLists.txt stashes this from the find_package(ZLIB) scope: the
    # imported target and ZLIB_LIBRARY are NOT visible at this top-level scope
    # (subdirectory-local target / normal var), so prefer the captured cache var.
    if(TRANSCRIBE_ZLIB_ARCHIVE)
        set(_loc "${TRANSCRIBE_ZLIB_ARCHIVE}")
    endif()
    if(NOT _loc AND TARGET ZLIB::ZLIB)
        get_target_property(_loc ZLIB::ZLIB IMPORTED_LOCATION_RELEASE)
        if(NOT _loc)
            get_target_property(_loc ZLIB::ZLIB IMPORTED_LOCATION)
        endif()
    endif()
    if(NOT _loc AND ZLIB_LIBRARY)
        set(_loc "${ZLIB_LIBRARY}")
    endif()
    # Only a single real archive on disk is stage-able (a debug/optimized
    # keyworded list or a bare -l name is not).
    if(_loc AND EXISTS "${_loc}")
        set(${out} "${_loc}" PARENT_SCOPE)
    else()
        set(${out} "" PARENT_SCOPE)
    endif()
endfunction()

# --- headers + the ABI digest ------------------------------------------------
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe.abihash"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/transcribe"
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")

# --- libtranscribe ------------------------------------------------------------
# ggml's own install rules already cover ggml / ggml-base / the backend libs.
# In shared mode, give every library a self-referential rpath so the
# installed lib dir is self-contained: under RUNPATH semantics the LOADING
# object's rpath resolves its deps, so libtranscribe needs $ORIGIN to find
# its sibling libggml*, not just the consumer binary.
transcribe_backend_kinds(_kinds _backend_targets)
if(TRANSCRIBE_BUILD_SHARED)
    foreach(_tgt transcribe ggml ggml-base ${_backend_targets})
        if(APPLE)
            set_property(TARGET ${_tgt} PROPERTY INSTALL_RPATH "@loader_path")
        else()
            set_property(TARGET ${_tgt} PROPERTY INSTALL_RPATH "$ORIGIN")
        endif()
    endforeach()
endif()
install(TARGETS transcribe
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# --- the link manifest (transcribe-link.json) --------------------------------
set(_libraries transcribe)
set(_library_paths "")
set(_system_libs "")
set(_frameworks "")
set(_link_flags "")

if(NOT TRANSCRIBE_BUILD_SHARED)
    # Static: the consumer links the whole archive set. Order is
    # single-pass-ld safe: each archive's undefined refs resolve in a later
    # one (transcribe -> ggml -> backends -> ggml-base).
    list(APPEND _libraries ggml ${_backend_targets} ggml-base)

    # The archives are C++; the consumer may be C or Rust.
    if(APPLE)
        list(APPEND _system_libs c++ m)
    elseif(WIN32)
        # MSVC links the CRT and the C++ runtime implicitly. ggml-cpu reads the
        # registry for CPU feature detection (RegOpenKeyEx/RegCloseKey), so the
        # static consumer must pull in advapi32 — ggml's own CMake never links
        # it (it rides MSVC default-lib auto-linking the manifest reconstruction
        # loses).
        list(APPEND _system_libs advapi32)
    elseif(UNIX)
        list(APPEND _system_libs stdc++ m pthread dl)
    endif()

    # Translate libtranscribe's PRIVATE link list (the configure-time truth
    # for zlib / OpenMP / Accelerate / system BLAS) into consumer terms.
    get_target_property(_transcribe_links transcribe LINK_LIBRARIES)
    foreach(_dep IN LISTS _transcribe_links)
        if(_dep STREQUAL "ZLIB::ZLIB")
            if(WIN32)
                # MSVC/vcpkg name the static lib `zlib.lib`, not `z`, so the
                # POSIX `-l z` is unsatisfiable. Stage the resolved archive next
                # to libtranscribe (keeping the lib dir self-contained) and link
                # it by name through the same search path as the rest of the set.
                _transcribe_zlib_archive(_zlib_archive)
                if(_zlib_archive)
                    install(FILES "${_zlib_archive}"
                        DESTINATION ${CMAKE_INSTALL_LIBDIR})
                    get_filename_component(_zlib_name "${_zlib_archive}" NAME_WE)
                    list(APPEND _libraries "${_zlib_name}")
                else()
                    list(APPEND _system_libs zlib)  # last resort: a bare name
                endif()
            else()
                list(APPEND _system_libs z)
            endif()
        elseif(_dep STREQUAL "OpenMP::OpenMP_CXX")
            list(APPEND _link_flags -fopenmp)
        elseif(_dep MATCHES "^-framework (.+)$")
            list(APPEND _frameworks "${CMAKE_MATCH_1}")
        elseif(_dep MATCHES "\\.(a|so[.0-9]*|dylib|tbd|lib)$")
            # Absolute paths (e.g. find_package(BLAS) results).
            list(APPEND _library_paths "${_dep}")
        endif()
    endforeach()

    # Per-backend system dependencies the ggml archives drag in.
    if("metal" IN_LIST _kinds)
        list(APPEND _frameworks Foundation Metal MetalKit)
    endif()
    if("blas" IN_LIST _kinds AND APPLE)
        list(APPEND _frameworks Accelerate)
    endif()
    if("vulkan" IN_LIST _kinds)
        list(APPEND _system_libs vulkan)
    endif()
    if(_frameworks)
        list(REMOVE_DUPLICATES _frameworks)
    endif()
    if(_system_libs)
        list(REMOVE_DUPLICATES _system_libs)
    endif()
endif()
# Shared: the consumer links libtranscribe alone; the ggml libraries are
# runtime dependencies resolved through the rpaths set above.

set(_metal_embed false)
if("metal" IN_LIST _kinds AND GGML_METAL_EMBED_LIBRARY)
    set(_metal_embed true)
endif()
if(TRANSCRIBE_BUILD_SHARED)
    set(_shared_json true)
else()
    set(_shared_json false)
endif()
if(TRANSCRIBE_GGML_BACKEND_DL)
    set(_backend_dl_json true)
    # ggml installs backend MODULES to GGML_BACKEND_DIR when set, else to
    # bin/. Record where, so a consumer knows what directory to hand to
    # transcribe_init_backends() (DL builds compile in NO backends — without
    # that call the process has zero devices).
    if(GGML_BACKEND_DIR)
        set(_module_dir_json "\"${GGML_BACKEND_DIR}\"")
    else()
        set(_module_dir_json "\"${CMAKE_INSTALL_BINDIR}\"")
    endif()
else()
    set(_backend_dl_json false)
    set(_module_dir_json null)
endif()

# JSON list helper: emits `"a", "b"` (empty list -> empty string -> []).
function(_transcribe_json_strings out)
    set(_quoted "")
    foreach(_item IN LISTS ARGN)
        list(APPEND _quoted "\"${_item}\"")
    endforeach()
    list(JOIN _quoted ", " _joined)
    set(${out} "${_joined}" PARENT_SCOPE)
endfunction()
_transcribe_json_strings(_kinds_json ${_kinds})
_transcribe_json_strings(_libraries_json ${_libraries})
_transcribe_json_strings(_library_paths_json ${_library_paths})
_transcribe_json_strings(_system_libs_json ${_system_libs})
_transcribe_json_strings(_frameworks_json ${_frameworks})
_transcribe_json_strings(_link_flags_json ${_link_flags})

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/transcribe-link.json"
"{
  \"version\": \"${PROJECT_VERSION}\",
  \"commit\": \"${TRANSCRIBE_BUILD_COMMIT}\",
  \"shared\": ${_shared_json},
  \"backend_dl\": ${_backend_dl_json},
  \"module_dir\": ${_module_dir_json},
  \"backends\": [${_kinds_json}],
  \"metal_embed\": ${_metal_embed},
  \"include_dir\": \"${CMAKE_INSTALL_INCLUDEDIR}\",
  \"lib_dir\": \"${CMAKE_INSTALL_LIBDIR}\",
  \"libraries\": [${_libraries_json}],
  \"library_paths\": [${_library_paths_json}],
  \"system_libs\": [${_system_libs_json}],
  \"frameworks\": [${_frameworks_json}],
  \"link_flags\": [${_link_flags_json}]
}
")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/transcribe-link.json"
    DESTINATION ${CMAKE_INSTALL_LIBDIR})

message(STATUS
    "transcribe install: ${PROJECT_VERSION} shared=${TRANSCRIBE_BUILD_SHARED} "
    "backends: ${_kinds} (link manifest: lib/transcribe-link.json)")
