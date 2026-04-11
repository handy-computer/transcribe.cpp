# transcribe library
#
# Pass 2C: real Parakeet weight ingest. ParakeetModel now owns a
# ggml_context with every weight tensor's data; load() does a two-stage
# open (header-only via Loader, then full data via gguf_init_from_file
# with no_alloc=false). run() is still NOT_IMPLEMENTED — compute is
# phase 3 (frontend) + phase 4 (encoder / decoder). The public ABI is
# unchanged.

add_library(transcribe
    transcribe.cpp
    transcribe-loader.cpp
    transcribe-arch.cpp
    transcribe-meta.cpp
    transcribe-mel.cpp
    transcribe-model.cpp
    transcribe-tokenizer.cpp
    transcribe-debug.cpp
    transcribe-flash-policy.cpp
    transcribe-load-common.cpp
    transcribe-weights-util.cpp
    conformer/conformer.cpp
    arch/parakeet/model.cpp
    arch/parakeet/capabilities.cpp
    arch/parakeet/weights.cpp
    arch/parakeet/encoder.cpp
    arch/parakeet/decoder.cpp
    arch/cohere/model.cpp
    arch/cohere/capabilities.cpp
    arch/cohere/weights.cpp
    arch/cohere/encoder.cpp
    arch/cohere/decoder.cpp
)

target_include_directories(transcribe
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        # Internal headers (transcribe-loader.h, transcribe-arch.h) live
        # alongside the .cpp files. PRIVATE so they never escape into the
        # public include path.
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(transcribe
    PRIVATE
        TRANSCRIBE_BUILD
)

# Phase 4 step 1: Parakeet::load tries Metal first when compiled in.
# When TRANSCRIBE_METAL is OFF the source falls through to CPU
# unconditionally and ggml-metal.h is never included.
# Backend-specific compile defs are no longer needed on the library —
# model.cpp discovers backends at runtime via ggml's device registry.
# TRANSCRIBE_METAL / TRANSCRIBE_VULKAN still control what ggml itself
# compiles (set in the top-level CMakeLists.txt → GGML_METAL / GGML_VULKAN).

# ggml is the runtime substrate. Linked PRIVATE because the public header
# stays ggml-free (callers never include <ggml.h>).
target_link_libraries(transcribe
    PRIVATE
        ggml
)

# BLAS for the host decoder (cblas_sgemv in decoder.cpp). On Apple we
# link Accelerate directly; elsewhere we probe for OpenBLAS / MKL /
# BLIS via find_package(BLAS). When no BLAS is found the decoder falls
# back to a scalar loop (~14x slower but functionally correct).
if(APPLE)
    target_link_libraries(transcribe PRIVATE "-framework Accelerate")
    target_compile_definitions(transcribe PRIVATE TRANSCRIBE_HAS_BLAS=1)
else()
    # BLA_VENDOR can be set by the user to prefer a specific BLAS
    # (e.g. -DBLA_VENDOR=OpenBLAS). find_package probes the system.
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        # find_package(BLAS) may find a Fortran-only BLAS (e.g. libblas3
        # on Debian/Ubuntu) that lacks the cblas C wrappers. Verify that
        # cblas.h exists before enabling; otherwise the build compiles
        # cblas_sgemv calls but the linker can't resolve them.
        include(CheckIncludeFile)
        check_include_file(cblas.h HAVE_CBLAS_H)
        if(HAVE_CBLAS_H)
            target_link_libraries(transcribe PRIVATE ${BLAS_LIBRARIES})
            target_compile_definitions(transcribe PRIVATE TRANSCRIBE_HAS_BLAS=1)
            message(STATUS "transcribe: BLAS found — decoder will use cblas_sgemv")
        else()
            message(STATUS "transcribe: BLAS found but cblas.h missing — decoder uses scalar fallback (install libopenblas-dev)")
        endif()
    else()
        message(STATUS "transcribe: no BLAS found — decoder uses scalar fallback")
    endif()
endif()

set_target_properties(transcribe PROPERTIES
    OUTPUT_NAME transcribe
    POSITION_INDEPENDENT_CODE ON
)

transcribe_apply_warnings(transcribe)
