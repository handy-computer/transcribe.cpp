# transcribe library.
#
# Sources are listed explicitly. Per-family code lives under arch/<fam>/;
# shared scaffolding (loader, mel frontend, tokenizer, load-common,
# backend classification, flash-policy, ...) lives alongside the
# dispatcher at the top level of src/.
#
# Backend libraries (Metal, Vulkan, ...) are compiled into ggml by the
# top-level CMake via GGML_METAL / GGML_VULKAN. This target discovers
# them at runtime via the ggml device registry; no per-backend compile
# defs are needed on libtranscribe itself.

add_library(transcribe
    transcribe.cpp
    transcribe-loader.cpp
    transcribe-arch.cpp
    transcribe-meta.cpp
    transcribe-kaldi-fbank.cpp
    transcribe-mel.cpp
    transcribe-model.cpp
    transcribe-tokenizer.cpp
    transcribe-unicode.cpp
    transcribe-unicode-data.cpp
    transcribe-debug.cpp
    transcribe-flash-policy.cpp
    transcribe-backend.cpp
    transcribe-load-common.cpp
    transcribe-weights-util.cpp
    transcribe-bin-loader.cpp
    transcribe-batch-util.cpp
    conformer/conformer.cpp
    sanm/sanm.cpp
    causal_lm/causal_lm.cpp
    granite_conformer/shaw_attn.cpp
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
    arch/canary/model.cpp
    arch/canary/capabilities.cpp
    arch/canary/weights.cpp
    arch/canary/encoder.cpp
    arch/canary/decoder.cpp
    arch/qwen3_asr/model.cpp
    arch/qwen3_asr/capabilities.cpp
    arch/qwen3_asr/weights.cpp
    arch/qwen3_asr/encoder.cpp
    arch/qwen3_asr/decoder.cpp
    arch/voxtral/model.cpp
    arch/voxtral/capabilities.cpp
    arch/voxtral/weights.cpp
    arch/voxtral/encoder.cpp
    arch/voxtral/decoder.cpp
    arch/voxtral_realtime/model.cpp
    arch/voxtral_realtime/capabilities.cpp
    arch/voxtral_realtime/weights.cpp
    arch/voxtral_realtime/encoder.cpp
    arch/voxtral_realtime/decoder.cpp
    arch/canary_qwen/model.cpp
    arch/canary_qwen/capabilities.cpp
    arch/canary_qwen/weights.cpp
    arch/canary_qwen/encoder.cpp
    arch/canary_qwen/decoder.cpp
    arch/whisper/model.cpp
    arch/whisper/capabilities.cpp
    arch/whisper/weights.cpp
    arch/whisper/encoder.cpp
    arch/whisper/decoder.cpp
    arch/whisper/bin_load.cpp
    arch/whisper/public.cpp
    arch/moonshine/model.cpp
    arch/moonshine/capabilities.cpp
    arch/moonshine/weights.cpp
    arch/moonshine/encoder.cpp
    arch/moonshine/decoder.cpp
    arch/moonshine_streaming/model.cpp
    arch/moonshine_streaming/capabilities.cpp
    arch/moonshine_streaming/weights.cpp
    arch/moonshine_streaming/encoder.cpp
    arch/moonshine_streaming/decoder.cpp
    arch/sensevoice/model.cpp
    arch/sensevoice/capabilities.cpp
    arch/sensevoice/weights.cpp
    arch/sensevoice/encoder.cpp
    arch/funasr_nano/model.cpp
    arch/funasr_nano/capabilities.cpp
    arch/funasr_nano/weights.cpp
    arch/funasr_nano/encoder.cpp
    arch/funasr_nano/decoder.cpp
    arch/funasr_nano/adaptor.cpp
    arch/gigaam/model.cpp
    arch/gigaam/capabilities.cpp
    arch/gigaam/weights.cpp
    arch/gigaam/encoder.cpp
    arch/gigaam/decoder.cpp
    arch/gigaam/mel.cpp
    arch/granite/model.cpp
    arch/granite/capabilities.cpp
    arch/granite/weights.cpp
    arch/granite/encoder.cpp
    arch/granite/projector.cpp
    arch/granite/decoder.cpp
    arch/granite_nar/model.cpp
    arch/granite_nar/capabilities.cpp
    arch/granite_nar/weights.cpp
    arch/granite_nar/encoder.cpp
    arch/granite_nar/projector.cpp
    arch/granite_nar/decoder.cpp
    arch/medasr/model.cpp
    arch/medasr/capabilities.cpp
    arch/medasr/weights.cpp
    arch/medasr/encoder.cpp
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

# zlib for Whisper's temperature-fallback compression_ratio metric
# (HF generation_whisper.py:_retrieve_compression_ratio uses
# len(token_bytes) / len(zlib.compress(token_bytes)) to detect decoder
# token-id loops). System zlib is ubiquitous; compile fails if absent.
find_package(ZLIB REQUIRED)
target_link_libraries(transcribe PRIVATE ZLIB::ZLIB)

transcribe_apply_warnings(transcribe)
