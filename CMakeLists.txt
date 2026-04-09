cmake_minimum_required(VERSION 3.16)
project(transcribe LANGUAGES C CXX)

# -----------------------------------------------------------------------------
# Standards
# -----------------------------------------------------------------------------

set(CMAKE_C_STANDARD            11)
set(CMAKE_C_STANDARD_REQUIRED   ON)
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS        OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Hidden symbol visibility by default per PLAN.md.
set(CMAKE_C_VISIBILITY_PRESET       hidden)
set(CMAKE_CXX_VISIBILITY_PRESET     hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# -----------------------------------------------------------------------------
# Apple Silicon detection (drives Metal default)
# -----------------------------------------------------------------------------

set(TRANSCRIBE_IS_APPLE_SILICON OFF)
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(TRANSCRIBE_IS_APPLE_SILICON ON)
endif()

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------

option(TRANSCRIBE_BUILD_TESTS    "Build unit / smoke tests" ON)
option(TRANSCRIBE_BUILD_EXAMPLES "Build example CLI"        ON)
option(TRANSCRIBE_BUILD_TOOLS    "Build inspect / quantize" OFF)  # post pass 2
option(TRANSCRIBE_METAL          "Enable Metal backend"     ${TRANSCRIBE_IS_APPLE_SILICON})
option(TRANSCRIBE_VULKAN         "Enable Vulkan backend"    OFF)  # post-v1
option(TRANSCRIBE_BLAS           "Enable BLAS for encoder matmuls" ON)
option(TRANSCRIBE_SANITIZE       "Enable ASan + UBSan"      OFF)
option(TRANSCRIBE_LTO            "Enable LTO in Release"    OFF)

# Real-model gated tests. OFF by default because the tests need a
# converted Parakeet GGUF on disk (~2.4 GB) and CI can't ship one.
# When ON, the test reads TRANSCRIBE_REAL_PARAKEET_GGUF from the
# environment at run time. See tests/parakeet_real_smoke.cpp.
option(TRANSCRIBE_BUILD_REAL_MODEL_TESTS
    "Build tests that load real (multi-GB) model files via env var" OFF)

# -----------------------------------------------------------------------------
# Warnings (no -Werror in v1 per PLAN.md)
# -----------------------------------------------------------------------------
#
# transcribe_apply_warnings(target): apply our project warning flags to one
# target. Used per-target so we don't pollute the vendored ggml subdirectory
# with -Wshadow / -Wpedantic, which fire on a lot of legitimate ggml code.

function(transcribe_apply_warnings tgt)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        target_compile_options(${tgt} PRIVATE
            -Wall
            -Wextra
            -Wshadow
            -Wpedantic
            -Wno-unused-parameter
        )
    endif()
endfunction()

# Sanitizers must be applied to every target in the build (including ggml)
# so the resulting binary is consistent. add_compile_options /
# add_link_options is the right scope for this.
#
# -fno-sanitize-recover=undefined makes UBSan reports fatal at runtime
# instead of merely printing them. ASan already aborts on error by
# default; UBSan does not, which would otherwise let ctest report
# success while the sanitizer printed real diagnostics.
if(TRANSCRIBE_SANITIZE)
    add_compile_options(
        -fsanitize=address,undefined
        -fno-sanitize-recover=undefined
        # ggml's `incr_ptr_aligned` (ggml/src/ggml.c) uses the
        # standard "use NULL as base, walk by offsets to compute
        # struct layout sizes" trick. UBSan flags this as a
        # `pointer-overflow` runtime error even though it's a
        # widely-used and safe-in-practice C idiom. Make this
        # specific check non-fatal so the encoder run path doesn't
        # abort under build-san; real bugs in our own code still
        # fail loud (the rest of -fno-sanitize-recover=undefined
        # is unaffected). When ggml gets patched upstream this can
        # come back out.
        -fsanitize-recover=pointer-overflow
        -fno-omit-frame-pointer
    )
    add_link_options(-fsanitize=address,undefined)
endif()

# -----------------------------------------------------------------------------
# Output layout
# -----------------------------------------------------------------------------
#
# Put all binaries in build/bin/ regardless of which subdirectory built them.
# ggml's CMakeLists already does this when invoked standalone; pinning it
# at the top level keeps the same layout when ggml is added as a subdirectory.

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# -----------------------------------------------------------------------------
# Vendored ggml
# -----------------------------------------------------------------------------
#
# ggml lives at the repo root (matching whisper.cpp / llama.cpp). The pinned
# upstream SHA is recorded in ggml/UPSTREAM. We map our TRANSCRIBE_* options
# onto ggml's GGML_* options BEFORE add_subdirectory() so ggml's option()
# calls become no-ops and our defaults win.

set(GGML_METAL    ${TRANSCRIBE_METAL}  CACHE BOOL "" FORCE)
set(GGML_VULKAN   ${TRANSCRIBE_VULKAN} CACHE BOOL "" FORCE)
set(GGML_BLAS     ${TRANSCRIBE_BLAS}   CACHE BOOL "" FORCE)

# On Apple, Accelerate provides BLAS; elsewhere let ggml's find_package
# probe for OpenBLAS / MKL / BLIS. If nothing is found, ggml gracefully
# disables the BLAS backend at configure time.
if(TRANSCRIBE_BLAS)
    if(APPLE)
        set(GGML_BLAS_VENDOR "Apple" CACHE STRING "" FORCE)
    elseif(NOT DEFINED CACHE{GGML_BLAS_VENDOR})
        set(GGML_BLAS_VENDOR "OpenBLAS" CACHE STRING "")
    endif()
endif()

# We don't want ggml's tests / examples leaking into our build tree. They
# default to OFF when ggml is added as a subdirectory (GGML_STANDALONE OFF),
# but pin them anyway so the contract is explicit.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Build ggml as a static library so the resulting transcribe binary can be
# distributed without chasing libggml.dylib at runtime.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# ggml's CMake declares its own warning flags; let it.
add_subdirectory(ggml)

# -----------------------------------------------------------------------------
# Subdirs
# -----------------------------------------------------------------------------

add_subdirectory(src)

if(TRANSCRIBE_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(TRANSCRIBE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(TRANSCRIBE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()
