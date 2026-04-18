# transcribe.cpp test suite.
#
# Tests in pass 2A:
#
#   transcribe_api_smoke    - pure C smoke of the public ABI. Built as
#                             C11 (not C++) on purpose so it doubles as
#                             a "header is C-clean" canary.
#
#   transcribe_loader_smoke - C++ fixture-driven test of the GGUF loader
#                             and per-family arch dispatch. Reads three
#                             tiny GGUFs from tests/fixtures/ that are
#                             generated at BUILD TIME by
#                             make_gguf_fixtures.py via uv. Only
#                             registered when uv is on PATH.
#
#   transcribe_wav_loader_smoke - example WAV loader unit smoke.
#
#   transcribe_cli_smoke    - end-to-end exit-0 + duration-line check
#                             on the example CLI.
#
# Fixture pipeline rationale: previous attempts ran the generator at
# configure time via execute_process and treated failure as a warning.
# That created a real false-green path: a broken generator could be
# silently masked by stale .gguf files left over from a previous
# successful run, because the only post-generation check was a runtime
# existence test. The current setup uses add_custom_command(OUTPUT)
# with DEPENDS on the script, so generation runs at BUILD time (not
# configure time), the build graph tracks the script as a dependency,
# and any uv failure produces a hard build error with the actual uv
# message in the build log. When uv is not installed at all, the
# loader test is not registered — a skipped test is honest, a passed
# test against stale fixtures is not.

# -----------------------------------------------------------------------------
# Pure C ABI smoke
# -----------------------------------------------------------------------------

add_executable(transcribe_api_smoke
    api_smoke.c
)

set_target_properties(transcribe_api_smoke PROPERTIES
    C_STANDARD          11
    C_STANDARD_REQUIRED ON
    LINKER_LANGUAGE     C
)

target_link_libraries(transcribe_api_smoke PRIVATE transcribe)

# C-only target: do not apply our C++-flavored shadow/pedantic flags here.
# transcribe_apply_warnings would still work for C, but the test file is
# tiny enough that the existing -Wall -Wextra inherited from CMAKE flags
# is enough.

add_test(NAME transcribe_api_smoke COMMAND transcribe_api_smoke)

# -----------------------------------------------------------------------------
# Backend classification unit test
# -----------------------------------------------------------------------------
#
# Covers the pure backend-kind mapping used by the runtime backend selector.
# This keeps vendor registry names such as ggml Metal's "MTL" tested without
# requiring that CI have the corresponding backend hardware available.

add_executable(transcribe_backend_classification_unit
    backend_classification_unit.cpp)

target_link_libraries(transcribe_backend_classification_unit PRIVATE transcribe ggml)

target_include_directories(transcribe_backend_classification_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_backend_classification_unit)

add_test(NAME transcribe_backend_classification_unit
    COMMAND transcribe_backend_classification_unit)

# -----------------------------------------------------------------------------
# Backend init_backends() runtime behavior
# -----------------------------------------------------------------------------
#
# Covers the runtime backend-selection matrix: strict CPU invariants,
# conditional explicit-backend error paths, AUTO fallback, and invalid
# enum rejection. Hardware-dependent cases (Metal/Vulkan present or
# absent) are probed at runtime inside the test binary — the test
# always runs but exercises different branches depending on what the
# build configured and what hardware is available.

add_executable(transcribe_backend_init_unit
    backend_init_unit.cpp)

target_link_libraries(transcribe_backend_init_unit PRIVATE transcribe ggml)

target_include_directories(transcribe_backend_init_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_backend_init_unit)

add_test(NAME transcribe_backend_init_unit
    COMMAND transcribe_backend_init_unit)

# -----------------------------------------------------------------------------
# F16 → F32 conv pointwise promotion unit test
# -----------------------------------------------------------------------------
#
# Tests promote_conv_pw_f16_to_f32_on_cpu() directly with
# programmatically created F16 tensors rather than GGUF fixtures.
# The synthetic Parakeet fixture emits only F32 tensors, so a
# fixture-based test would be a false positive. This creates real F16
# data, verifies the CPU path promotes and round-trips correctly, and
# confirms the non-CPU path is a clean no-op.

add_executable(transcribe_conv_pw_promote_unit
    conv_pw_promote_unit.cpp)

target_link_libraries(transcribe_conv_pw_promote_unit PRIVATE transcribe ggml)

target_include_directories(transcribe_conv_pw_promote_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_conv_pw_promote_unit)

add_test(NAME transcribe_conv_pw_promote_unit
    COMMAND transcribe_conv_pw_promote_unit)

# -----------------------------------------------------------------------------
# Dispatcher-level transcribe_run behavior
# -----------------------------------------------------------------------------

add_executable(transcribe_run_dispatch_unit
    run_dispatch_unit.cpp)

target_link_libraries(transcribe_run_dispatch_unit PRIVATE transcribe)

target_include_directories(transcribe_run_dispatch_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_run_dispatch_unit)

add_test(NAME transcribe_run_dispatch_unit
    COMMAND transcribe_run_dispatch_unit)

# -----------------------------------------------------------------------------
# Synthetic GGUF fixtures (for the loader smoke test)
# -----------------------------------------------------------------------------

set(TRANSCRIBE_FIXTURE_GENERATOR
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_gguf_fixtures.py)

set(TRANSCRIBE_FIXTURE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_parakeet.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_unknown.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/corrupt_magic.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_minimal.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_minimal_v3.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_cohere_minimal.gguf)

find_program(TRANSCRIBE_UV uv)

if(TRANSCRIBE_UV)
    # add_custom_command(OUTPUT ...) makes the .gguf files first-class
    # build outputs. The build graph treats the generator script as a
    # source dependency, so the fixtures are regenerated whenever the
    # script changes — and a failing uv invocation produces a hard
    # build error rather than a configure-time warning.
    add_custom_command(
        OUTPUT  ${TRANSCRIBE_FIXTURE_FILES}
        COMMAND ${TRANSCRIBE_UV} run --quiet ${TRANSCRIBE_FIXTURE_GENERATOR}
        DEPENDS ${TRANSCRIBE_FIXTURE_GENERATOR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
        COMMENT "Generating synthetic GGUF fixtures"
        VERBATIM)

    # Aggregate target. The loader test target depends on this; building
    # `fixtures` directly is the supported way to hand-regenerate.
    add_custom_target(fixtures
        DEPENDS ${TRANSCRIBE_FIXTURE_FILES})
else()
    message(STATUS
        "uv not found on PATH; transcribe_loader_smoke will not be "
        "registered. Install uv (https://docs.astral.sh/uv/) and "
        "reconfigure to enable.")
endif()

# -----------------------------------------------------------------------------
# GGUF loader / arch dispatch smoke
# -----------------------------------------------------------------------------
#
# Only registered when uv is available. The build dependency on the
# `fixtures` target guarantees the .gguf files are present and current
# at test time. loader_smoke.cpp keeps a runtime existence backstop for
# the pathological "fixtures deleted between build and test" case, but
# under normal use it should never fire.

if(TRANSCRIBE_UV)
    add_executable(transcribe_loader_smoke
        loader_smoke.cpp)

    target_link_libraries(transcribe_loader_smoke PRIVATE transcribe)

    target_compile_definitions(transcribe_loader_smoke PRIVATE
        "TRANSCRIBE_TEST_FIXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\"")

    transcribe_apply_warnings(transcribe_loader_smoke)

    # Build dependency: the loader test cannot link until the fixtures
    # custom target has succeeded, which means uv must have produced
    # current .gguf files for the current generator script.
    add_dependencies(transcribe_loader_smoke fixtures)

    add_test(NAME transcribe_loader_smoke COMMAND transcribe_loader_smoke)
    set_tests_properties(transcribe_loader_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# -----------------------------------------------------------------------------
# Tokenizer ingest smoke (success path through the public ABI)
# -----------------------------------------------------------------------------
#
# Goes one level deeper than loader_smoke: it actually loads
# tokenizer_minimal.gguf, then reaches into internal headers to inspect
# the resulting Tokenizer's vocabulary, special token ids, and decode
# behavior. The internal-header reach is fenced behind a PRIVATE include
# of ${CMAKE_SOURCE_DIR}/src so the test target can pull in
# transcribe-model.h / transcribe-tokenizer.h / arch/parakeet/parakeet.h
# without exposing them on the public include path.

if(TRANSCRIBE_UV)
    add_executable(transcribe_tokenizer_smoke
        tokenizer_smoke.cpp)

    target_link_libraries(transcribe_tokenizer_smoke PRIVATE transcribe ggml)

    # Internal-header access. PRIVATE so it stays scoped to this test
    # target.
    target_include_directories(transcribe_tokenizer_smoke PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

    target_compile_definitions(transcribe_tokenizer_smoke PRIVATE
        "TRANSCRIBE_TEST_FIXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\"")

    transcribe_apply_warnings(transcribe_tokenizer_smoke)

    add_dependencies(transcribe_tokenizer_smoke fixtures)

    add_test(NAME transcribe_tokenizer_smoke COMMAND transcribe_tokenizer_smoke)
    set_tests_properties(transcribe_tokenizer_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# -----------------------------------------------------------------------------
# Parakeet weight ingest smoke (full Parakeet model load through the public ABI)
# -----------------------------------------------------------------------------
#
# Loads the same tokenizer_minimal.gguf fixture as transcribe_tokenizer_smoke
# (which is now a structurally complete minimal Parakeet model with 78
# weight tensors filled with deterministic toy float32 data) and exercises
# the 2C tensor ingest path. Asserts hparams match the fixture, every named
# slot in ParakeetWeights is populated, a couple of tensor shapes line up
# with the per-block formulas in arch/parakeet/weights.cpp, and the first
# element of two specific tensors equals the deterministic value the
# fixture wrote (proves the data section was actually copied).
#
# Same internal-header access pattern as tokenizer_smoke: PRIVATE include
# of ${CMAKE_SOURCE_DIR}/src so the test can pull in
# arch/parakeet/parakeet.h and arch/parakeet/weights.h without exposing
# them on the public include path.

if(TRANSCRIBE_UV)
    add_executable(transcribe_parakeet_smoke
        parakeet_smoke.cpp)

    target_link_libraries(transcribe_parakeet_smoke PRIVATE transcribe ggml)

    target_include_directories(transcribe_parakeet_smoke PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

    target_compile_definitions(transcribe_parakeet_smoke PRIVATE
        "TRANSCRIBE_TEST_FIXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\"")

    transcribe_apply_warnings(transcribe_parakeet_smoke)

    add_dependencies(transcribe_parakeet_smoke fixtures)

    add_test(NAME transcribe_parakeet_smoke COMMAND transcribe_parakeet_smoke)
    set_tests_properties(transcribe_parakeet_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# -----------------------------------------------------------------------------
# Real-model gated test (developer-local, never CI)
# -----------------------------------------------------------------------------
#
# Loads a real Parakeet GGUF (the output of scripts/convert-parakeet.py
# against a real exported model directory) and asserts that the
# loader produces a fully populated ParakeetModel with the right
# 0.6B-variant hparams and a populated 697-tensor catalog.
#
# Gated two ways: by the CMake option above (so the binary is not
# built unless the developer opts in) AND by the runtime
# TRANSCRIBE_REAL_PARAKEET_GGUF env var (so a configured-but-no-file
# build still skips the test cleanly via SKIP_RETURN_CODE 77 instead
# of failing the suite).

if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_parakeet_real_smoke
        parakeet_real_smoke.cpp)

    target_link_libraries(transcribe_parakeet_real_smoke PRIVATE transcribe ggml)

    target_include_directories(transcribe_parakeet_real_smoke PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

    transcribe_apply_warnings(transcribe_parakeet_real_smoke)

    add_test(NAME transcribe_parakeet_real_smoke
        COMMAND transcribe_parakeet_real_smoke)
    set_tests_properties(transcribe_parakeet_real_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# -----------------------------------------------------------------------------
# Decoder + result accessor accuracy test (real-model gated)
# -----------------------------------------------------------------------------
#
# Phase 5 acceptance gate: end-to-end transcription on jfk.wav. Loads
# a real Parakeet v2 GGUF, runs transcribe_run, and asserts the full
# text is within Levenshtein edit distance 3 of the canonical JFK
# reference text. Tensor-level numerical validation lives in
# scripts/validate.py.
# Also exercises every public result accessor (segment / word / token
# levels) including the safe-sentinel out-of-bounds path.
#
# Same gating as the other real-model tests: built only when
# TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON; skipped at run time when
# TRANSCRIBE_REAL_PARAKEET_GGUF is unset. The reference text is
# baked into the test source, not a golden file on disk, so the
# test is hermetic — it does not need a golden generation step.

if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_decoder_smoke
        decoder_smoke.cpp)

    target_link_libraries(transcribe_decoder_smoke
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_decoder_smoke PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_decoder_smoke)

    add_test(NAME transcribe_decoder_smoke COMMAND transcribe_decoder_smoke)
    set_tests_properties(transcribe_decoder_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# -----------------------------------------------------------------------------
# Cohere ASR synthetic fixture smoke (default-built)
# -----------------------------------------------------------------------------
#
# Same role as transcribe_parakeet_smoke for the Parakeet family:
# loads arch_cohere_minimal.gguf (a structurally complete minimal
# Cohere ASR model with the full encoder + enc_dec_proj + decoder +
# head tensor catalog at toy dims), then reaches into internal headers
# to assert hparams round-trip, every CohereWeights slot is populated,
# the post-load BN fuse + Q-bias fuse + conv-pw promote steps ran, and
# the streamed tensor data is non-zero at known indices. Catches
# converter/loader drift in default CI without needing a real model.
#
# Only registered when uv is available (the fixture generator is a uv
# script). The transcribe_cohere_e2e_smoke and transcribe_cohere_real_smoke
# tests below cover the real-model paths.

if(TRANSCRIBE_UV)
    add_executable(transcribe_cohere_smoke
        cohere_smoke.cpp)

    target_link_libraries(transcribe_cohere_smoke PRIVATE transcribe ggml)

    target_include_directories(transcribe_cohere_smoke PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

    target_compile_definitions(transcribe_cohere_smoke PRIVATE
        "TRANSCRIBE_TEST_FIXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\"")

    transcribe_apply_warnings(transcribe_cohere_smoke)

    add_dependencies(transcribe_cohere_smoke fixtures)

    add_test(NAME transcribe_cohere_smoke COMMAND transcribe_cohere_smoke)
    set_tests_properties(transcribe_cohere_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# -----------------------------------------------------------------------------
# Cohere ASR end-to-end smoke test (real-model gated)
# -----------------------------------------------------------------------------
#
# Loads a real Cohere ASR GGUF, runs transcription on a WAV file, and
# verifies the output text contains expected content words, timing info
# is present and reasonable, and no crashes or leaks occur. This is an
# API and structural behavior test, not a numerical accuracy gate;
# tensor-level validation lives in scripts/validate.py.
#
# Same gating as the Parakeet real-model tests: built only when
# TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON. At run time,
# TRANSCRIBE_COHERE_MODEL must point at a Cohere ASR GGUF. Exits 77
# (CTest "skipped") when the env var is unset or the file is missing.

if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_cohere_e2e_smoke
        cohere_e2e_smoke.cpp)

    target_link_libraries(transcribe_cohere_e2e_smoke
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_cohere_e2e_smoke PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_cohere_e2e_smoke)

    add_test(NAME transcribe_cohere_e2e_smoke COMMAND transcribe_cohere_e2e_smoke)
    set_tests_properties(transcribe_cohere_e2e_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# -----------------------------------------------------------------------------
# Cohere ASR structural / loader test (real-model gated)
# -----------------------------------------------------------------------------
#
# Loads a real Cohere ASR GGUF (same one cohere_smoke uses), then
# reaches into the CohereModel via PRIVATE include of ${CMAKE_SOURCE_DIR}/src
# and asserts the full hparams catalog, the 2103-tensor count formula,
# the language list ordering, and a handful of canonical weight shapes.
# This is the converter regression gate -- if read_cohere_hparams or
# build_cohere_weights ever drift from the converter tables, this test
# trips well before any numerical test would.
#
# Gated two ways: by TRANSCRIBE_BUILD_REAL_MODEL_TESTS (so the binary
# is not built unless the developer opts in) AND by the runtime
# TRANSCRIBE_COHERE_MODEL env var (same var that cohere_e2e_smoke.cpp
# uses, so a single export drives both tests). Skipped via
# SKIP_RETURN_CODE 77 when the env var is unset or the file is
# missing.

if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_cohere_real_smoke
        cohere_real_smoke.cpp)

    target_link_libraries(transcribe_cohere_real_smoke PRIVATE transcribe ggml)

    target_include_directories(transcribe_cohere_real_smoke PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

    transcribe_apply_warnings(transcribe_cohere_real_smoke)

    add_test(NAME transcribe_cohere_real_smoke
        COMMAND transcribe_cohere_real_smoke)
    set_tests_properties(transcribe_cohere_real_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# -----------------------------------------------------------------------------
# Debug tensor dumper round-trip unit
# -----------------------------------------------------------------------------
#
# Phase 4 step 2: per-stage tensor dumping infrastructure for the
# numerical accuracy harness. This unit test sets TRANSCRIBE_DUMP_DIR
# at runtime, builds a tiny CPU-backed ggml tensor with deterministic
# values, dumps it via transcribe::debug::dump_tensor, and verifies
# the .f32 file bytes round-trip exactly and the .json sidecar carries
# the right shape (slow-to-fast / numpy-style with trailing 1s
# dropped). Also exercises the negative paths (unsafe names + null
# tensor) to confirm dumping never propagates errors. CPU-only on
# purpose; the Metal path runs end-to-end via the encoder graph in
# step 3.

add_executable(transcribe_debug_dump_unit
    debug_dump_unit.cpp)

target_link_libraries(transcribe_debug_dump_unit PRIVATE transcribe ggml)

target_include_directories(transcribe_debug_dump_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_debug_dump_unit)

add_test(NAME transcribe_debug_dump_unit
    COMMAND transcribe_debug_dump_unit)

# -----------------------------------------------------------------------------
# Mel frontend unit tests (precomputed window + filterbank)
# -----------------------------------------------------------------------------
#
# Validates the constructor outputs of transcribe::MelFrontend in
# isolation: the symmetric Hann window zero-padded to n_fft, and the
# librosa Slaney mel filterbank. Reference values are bit-precise
# constants captured from numpy / librosa 0.11; no Python dependency
# at test time. Catches off-by-one bugs in window/filterbank
# construction before they cascade through the full pipeline.

add_executable(transcribe_mel_unit
    mel_unit.cpp)

target_link_libraries(transcribe_mel_unit PRIVATE transcribe)

target_include_directories(transcribe_mel_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_mel_unit)

add_test(NAME transcribe_mel_unit COMMAND transcribe_mel_unit)

# -----------------------------------------------------------------------------
# WAV loader unit smoke (example helper)
# -----------------------------------------------------------------------------
#
# The example WAV loader is in examples/common, not the core library, but
# we still cover it here so a regression at the example boundary trips
# the same gate as the public ABI tests. Only registered when examples
# are built (which is the default and the only configuration that builds
# transcribe-common-example).

if(TRANSCRIBE_BUILD_EXAMPLES)
    add_executable(transcribe_wav_loader_smoke
        wav_loader_smoke.cpp)

    target_link_libraries(transcribe_wav_loader_smoke
        PRIVATE transcribe-common-example)

    target_compile_definitions(transcribe_wav_loader_smoke PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_wav_loader_smoke)

    add_test(NAME transcribe_wav_loader_smoke
        COMMAND transcribe_wav_loader_smoke)
endif()

# -----------------------------------------------------------------------------
# CLI smoke (end-to-end exit-0 + duration-line check)
# -----------------------------------------------------------------------------
#
# Closes the Phase 1 review gap: the example CLI itself was not exercised
# by ctest, so a regression that broke the WAV path or the example
# binary's exit-status handling could only be caught by manual smoke. We
# run the binary against the canonical jfk.wav fixture and assert it
# exits 0 with a "duration:" line on stdout. CTest's
# PASS_REGULAR_EXPRESSION combines both checks: the test fails unless
# the exit code is 0 AND the regex matches.

if(TRANSCRIBE_BUILD_EXAMPLES)
    add_test(NAME transcribe_cli_smoke
        COMMAND $<TARGET_FILE:transcribe-cli>
            ${CMAKE_SOURCE_DIR}/samples/jfk.wav)
    set_tests_properties(transcribe_cli_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "duration:")
endif()
