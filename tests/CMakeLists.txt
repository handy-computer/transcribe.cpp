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

add_test(
    NAME transcribe_extension_umbrella_check
    COMMAND ${CMAKE_COMMAND}
        -DTRANSCRIBE_SOURCE_DIR=${PROJECT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_extension_umbrella.cmake)

# -----------------------------------------------------------------------------
# Shared-library build: only the public-ABI smoke is meaningful
# -----------------------------------------------------------------------------
#
# Every test below this point is a C++ white-box test: it links `transcribe`
# AND reaches into src/ internal headers, calling non-TRANSCRIBE_API symbols.
# Those symbols are hidden in a shared libtranscribe (visibility=hidden), so the
# white-box suite cannot link against it. A shared build (TRANSCRIBE_BUILD_SHARED
# =ON) is the packaging/binding configuration; build static (the default) to run
# the full suite. The pure-C api_smoke above stays — against a shared lib it
# doubles as an exported-surface canary: it proves every public symbol resolves
# from a clean C consumer with no internal-header reach.
if(BUILD_SHARED_LIBS)
    message(STATUS
        "transcribe: shared-library build — registering only the public-ABI "
        "smoke; the C++ white-box test suite requires a static build "
        "(-DTRANSCRIBE_BUILD_SHARED=OFF).")
    return()
endif()

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
# Parakeet chunked_limited_with_rc attention mask unit test
# -----------------------------------------------------------------------------
#
# Validates compute_chunked_limited_with_rc_mask against an inline
# reproduction of NeMo's ConformerEncoder._create_masks formula for the
# chunked_limited_with_rc branch. The buffered streaming driver
# (parakeet-unified-en-0.6b) feeds this mask into rel_pos_mhsa; any
# divergence from NeMo would break the streaming WER gate.

add_executable(transcribe_parakeet_chunked_limited_with_rc_mask_unit
    parakeet_chunked_limited_with_rc_mask_unit.cpp)

target_link_libraries(transcribe_parakeet_chunked_limited_with_rc_mask_unit
    PRIVATE transcribe ggml)

target_include_directories(transcribe_parakeet_chunked_limited_with_rc_mask_unit
    PRIVATE ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_parakeet_chunked_limited_with_rc_mask_unit)

add_test(NAME transcribe_parakeet_chunked_limited_with_rc_mask_unit
    COMMAND transcribe_parakeet_chunked_limited_with_rc_mask_unit)

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
# Dispatcher-level streaming entry-point behavior
# -----------------------------------------------------------------------------

add_executable(transcribe_stream_dispatch_unit
    stream_dispatch_unit.cpp)

target_link_libraries(transcribe_stream_dispatch_unit PRIVATE transcribe)

target_include_directories(transcribe_stream_dispatch_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_stream_dispatch_unit)

add_test(NAME transcribe_stream_dispatch_unit
    COMMAND transcribe_stream_dispatch_unit)

# -----------------------------------------------------------------------------
# Log sink contract (three states + ggml bridge level mapping)
# -----------------------------------------------------------------------------

add_executable(transcribe_log_unit
    log_unit.cpp)

# ggml for ggml_log_get / GGML_LOG_LEVEL_*: the test drives the bridge
# transcribe_log_set installs, pinning the cross-enum level mapping.
target_link_libraries(transcribe_log_unit PRIVATE transcribe ggml)

target_include_directories(transcribe_log_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_log_unit)

add_test(NAME transcribe_log_unit
    COMMAND transcribe_log_unit)

# -----------------------------------------------------------------------------
# CPU thread-count helpers (affinity-aware default + sched resolution)
# -----------------------------------------------------------------------------
#
# Pins default_n_threads()/configure_sched_n_threads() in transcribe-batch-util:
# cap+range, the null-sched resolution contract, and (Linux) that the default
# honors the process affinity mask rather than the host core count — the
# property whose absence caused the oversubscription livelock.

add_executable(transcribe_thread_default_unit
    thread_default_unit.cpp)

target_link_libraries(transcribe_thread_default_unit PRIVATE transcribe ggml)

target_include_directories(transcribe_thread_default_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_thread_default_unit)

add_test(NAME transcribe_thread_default_unit
    COMMAND transcribe_thread_default_unit)

# -----------------------------------------------------------------------------
# Threadpool barrier under CPU oversubscription (regression for the MSVC
# non-OpenMP ggml_barrier livelock; see ggml-cpu.c ggml_thread_yield)
# -----------------------------------------------------------------------------
#
# Pins the process to 2 logical CPUs and runs a 4-thread graph in a loop with a
# watchdog: a barrier deadlock turns into exit 2 instead of an indefinite hang.
# Only exercises the custom barrier on a GGML_OPENMP=OFF build (a GGML_OPENMP=ON
# build uses `#pragma omp barrier` and passes trivially) — it is load-bearing in
# the OpenMP-OFF lanes. Links ggml only; no transcribe/model dependency.

add_executable(transcribe_threadpool_oversubscription
    threadpool_oversubscription.c)

target_link_libraries(transcribe_threadpool_oversubscription PRIVATE ggml)

transcribe_apply_warnings(transcribe_threadpool_oversubscription)

add_test(NAME transcribe_threadpool_oversubscription
    COMMAND transcribe_threadpool_oversubscription)

# -----------------------------------------------------------------------------
# Synthetic GGUF fixtures (for the loader smoke test)
# -----------------------------------------------------------------------------

set(TRANSCRIBE_FIXTURE_GENERATOR
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/make_gguf_fixtures.py)

set(TRANSCRIBE_FIXTURE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_parakeet.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_sensevoice.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_funasr_nano.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_unknown.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/corrupt_magic.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_minimal.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_minimal_v3.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_minimal_streaming.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_minimal_streaming_cache_aware.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/tokenizer_minimal_streaming_buffered.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_cohere_minimal.gguf
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures/arch_qwen3_asr_minimal.gguf)

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

    # -------------------------------------------------------------------------
    # Streaming capability gate against a real loaded model (Phase 3)
    # -------------------------------------------------------------------------
    #
    # Uses two parakeet fixtures (default capability + KV-override) to
    # pin (a) supports_streaming reads correctly from both family default
    # and stt.capability.streaming KV, and (b) transcribe_stream_begin
    # returns NOT_IMPLEMENTED in both cases — once on the capability
    # gate, once on the dispatcher's hook-set check. Public-API only
    # (no internal-header reach).
    add_executable(transcribe_stream_capability_unit
        stream_capability_unit.cpp)

    target_link_libraries(transcribe_stream_capability_unit PRIVATE transcribe)

    target_compile_definitions(transcribe_stream_capability_unit PRIVATE
        "TRANSCRIBE_TEST_FIXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\"")

    transcribe_apply_warnings(transcribe_stream_capability_unit)

    add_dependencies(transcribe_stream_capability_unit fixtures)

    add_test(NAME transcribe_stream_capability_unit
        COMMAND transcribe_stream_capability_unit)
    set_tests_properties(transcribe_stream_capability_unit PROPERTIES
        SKIP_RETURN_CODE 77)

    # -------------------------------------------------------------------------
    # Parakeet streaming extension sentinel-reject regression
    # -------------------------------------------------------------------------
    #
    # Pins the per-field "< -1 sentinel returns INVALID_ARG" rule for
    # both the cache-aware (transcribe_parakeet_stream_ext) and the
    # chunked-attention buffered (transcribe_parakeet_buffered_stream_ext)
    # streaming extensions. Uses toy ChunkedLimited and ChunkedLimitedWithRc
    # fixtures so the parakeet stream_begin hook routes into each
    # validation path; the rejection fires before any compute, so the
    # toy fixtures without real weights are sufficient.
    add_executable(transcribe_parakeet_stream_ext_reject_unit
        parakeet_stream_ext_reject_unit.cpp)

    target_link_libraries(transcribe_parakeet_stream_ext_reject_unit
        PRIVATE transcribe)

    target_compile_definitions(transcribe_parakeet_stream_ext_reject_unit PRIVATE
        "TRANSCRIBE_TEST_FIXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\"")

    transcribe_apply_warnings(transcribe_parakeet_stream_ext_reject_unit)

    add_dependencies(transcribe_parakeet_stream_ext_reject_unit fixtures)

    add_test(NAME transcribe_parakeet_stream_ext_reject_unit
        COMMAND transcribe_parakeet_stream_ext_reject_unit)
    set_tests_properties(transcribe_parakeet_stream_ext_reject_unit PROPERTIES
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
# TRANSCRIBE_PARAKEET_GGUF env var (so a configured-but-no-file
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
# TRANSCRIBE_PARAKEET_GGUF is unset. The reference text is
# baked into the test source, not a golden file on disk, so the
# test is hermetic — it does not need a golden generation step.

if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_moonshine_streaming_stream_parity
        moonshine_streaming_stream_parity.cpp)

    target_link_libraries(transcribe_moonshine_streaming_stream_parity
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_moonshine_streaming_stream_parity PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_moonshine_streaming_stream_parity)

    add_test(NAME transcribe_moonshine_streaming_stream_parity
        COMMAND transcribe_moonshine_streaming_stream_parity)
    set_tests_properties(transcribe_moonshine_streaming_stream_parity PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# Per-utterance OUTPUT_TRUNCATED in transcribe_run_batch (moonshine_streaming).
# Skipped (RC 77) when TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF is unset.
if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_moonshine_streaming_batch_truncation
        moonshine_streaming_batch_truncation.cpp)

    target_link_libraries(transcribe_moonshine_streaming_batch_truncation
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_moonshine_streaming_batch_truncation PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_moonshine_streaming_batch_truncation)

    add_test(NAME transcribe_moonshine_streaming_batch_truncation
        COMMAND transcribe_moonshine_streaming_batch_truncation)
    set_tests_properties(transcribe_moonshine_streaming_batch_truncation PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# Per-utterance OUTPUT_TRUNCATED in transcribe_run / transcribe_run_batch for a
# causal_lm (LLM-decoder) family. Guards the shared src/causal_lm batched step
# loop's truncation detection (the encoder-decoder twin is covered by
# transcribe_moonshine_streaming_batch_truncation). Skipped (RC 77) when
# TRANSCRIBE_QWEN3_ASR_0_6B_GGUF is unset.
if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_qwen3_asr_batch_truncation
        qwen3_asr_batch_truncation.cpp)

    target_link_libraries(transcribe_qwen3_asr_batch_truncation
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_qwen3_asr_batch_truncation PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_qwen3_asr_batch_truncation)

    add_test(NAME transcribe_qwen3_asr_batch_truncation
        COMMAND transcribe_qwen3_asr_batch_truncation)
    set_tests_properties(transcribe_qwen3_asr_batch_truncation PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

# Input-length contract for the chunked/unbounded voxtral_realtime family:
# n_ctx is a no-op, limits stay unbounded, and a tiny n_ctx neither rejects
# (one-shot) nor crashes (batch). Skipped (RC 77) when
# TRANSCRIBE_VOXTRAL_REALTIME_GGUF is unset.
if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_voxtral_realtime_real_smoke
        voxtral_realtime_real_smoke.cpp)

    target_link_libraries(transcribe_voxtral_realtime_real_smoke
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_voxtral_realtime_real_smoke PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_voxtral_realtime_real_smoke)

    add_test(NAME transcribe_voxtral_realtime_real_smoke
        COMMAND transcribe_voxtral_realtime_real_smoke)
    set_tests_properties(transcribe_voxtral_realtime_real_smoke PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_stream_committed_pointer_stability
        stream_committed_pointer_stability.cpp)

    target_link_libraries(transcribe_stream_committed_pointer_stability
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_stream_committed_pointer_stability PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_stream_committed_pointer_stability)

    add_test(NAME transcribe_stream_committed_pointer_stability
        COMMAND transcribe_stream_committed_pointer_stability)
    set_tests_properties(transcribe_stream_committed_pointer_stability PROPERTIES
        SKIP_RETURN_CODE 77)
endif()

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
# Qwen3-ASR loader smoke test (fixture-backed, always on)
# -----------------------------------------------------------------------------
#
# Loads arch_qwen3_asr_minimal.gguf and asserts the hparams, tokenizer,
# capabilities, chat-template token resolution, and the full Qwen3-ASR
# weight catalog all come through the loader cleanly. Does not call
# transcribe_run — toy weights produce nonsense. Run-path coverage is
# in qwen3_asr_real_smoke_{0_6b,1_7b}.cpp.

if(TRANSCRIBE_UV)
    add_executable(transcribe_qwen3_asr_smoke
        qwen3_asr_smoke.cpp)

    target_link_libraries(transcribe_qwen3_asr_smoke PRIVATE transcribe ggml)

    target_include_directories(transcribe_qwen3_asr_smoke PRIVATE
        ${CMAKE_SOURCE_DIR}/src)

    target_compile_definitions(transcribe_qwen3_asr_smoke PRIVATE
        "TRANSCRIBE_TEST_FIXTURES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/fixtures\"")

    transcribe_apply_warnings(transcribe_qwen3_asr_smoke)

    add_dependencies(transcribe_qwen3_asr_smoke fixtures)

    add_test(NAME transcribe_qwen3_asr_smoke COMMAND transcribe_qwen3_asr_smoke)
    set_tests_properties(transcribe_qwen3_asr_smoke PROPERTIES
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
# TRANSCRIBE_COHERE_GGUF must point at a Cohere ASR GGUF. Exits 77
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
# Whisper end-to-end / capability smoke test (real-model gated)
# -----------------------------------------------------------------------------
#
# Loads a real Whisper GGUF and verifies the public capabilities match the
# runtime behavior: language detection runs when no language hint is supplied,
# default params use no-timestamp decode, explicit SEGMENT returns segment
# timestamps, and WORD timestamps are rejected until the DTW alignment path lands.

if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    add_executable(transcribe_whisper_e2e_smoke
        whisper_e2e_smoke.cpp)

    target_link_libraries(transcribe_whisper_e2e_smoke
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_whisper_e2e_smoke PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_whisper_e2e_smoke)

    add_test(NAME transcribe_whisper_e2e_smoke
        COMMAND transcribe_whisper_e2e_smoke)
    set_tests_properties(transcribe_whisper_e2e_smoke PROPERTIES
        SKIP_RETURN_CODE 77)

    # Legacy whisper.cpp `.bin` end-to-end smoke. Gated by
    # TRANSCRIBE_WHISPER_BIN_TINY_Q8_0 so CI without an upstream
    # `.bin` skips cleanly (RC 77). Same gating pattern as the
    # other real-model tests.
    add_executable(transcribe_whisper_bin_e2e_smoke
        whisper_bin_e2e_smoke.cpp)

    target_link_libraries(transcribe_whisper_bin_e2e_smoke
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_whisper_bin_e2e_smoke PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_whisper_bin_e2e_smoke)

    add_test(NAME transcribe_whisper_bin_e2e_smoke
        COMMAND transcribe_whisper_bin_e2e_smoke)
    set_tests_properties(transcribe_whisper_bin_e2e_smoke PROPERTIES
        SKIP_RETURN_CODE 77)

    # Tokenizer parity: tiktoken-style raw-bytes encoder (.bin) vs
    # HF GPT-2 byte-level BPE with merges (GGUF). Both surfaces feed
    # transcribe_tokenize; the parity test asserts byte-for-byte
    # identical token id sequences across a representative probe set.
    add_executable(transcribe_whisper_bin_tokenize_parity
        whisper_bin_tokenize_parity.cpp)

    target_link_libraries(transcribe_whisper_bin_tokenize_parity
        PRIVATE transcribe)

    transcribe_apply_warnings(transcribe_whisper_bin_tokenize_parity)

    add_test(NAME transcribe_whisper_bin_tokenize_parity
        COMMAND transcribe_whisper_bin_tokenize_parity)
    set_tests_properties(transcribe_whisper_bin_tokenize_parity PROPERTIES
        SKIP_RETURN_CODE 77)

    # Stage 1 parity: HF-byte-identical tokenization for Whisper,
    # including digit and contraction edge cases that the GPT-2
    # pretokenizer regex handles differently from Qwen2.
    add_executable(transcribe_whisper_tokenize_parity
        whisper_tokenize_parity.cpp)

    target_link_libraries(transcribe_whisper_tokenize_parity
        PRIVATE transcribe)

    transcribe_apply_warnings(transcribe_whisper_tokenize_parity)

    add_test(NAME transcribe_whisper_tokenize_parity
        COMMAND transcribe_whisper_tokenize_parity)
    set_tests_properties(transcribe_whisper_tokenize_parity PROPERTIES
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
# TRANSCRIBE_COHERE_GGUF env var (same var that cohere_e2e_smoke.cpp
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
# Qwen3-ASR structural + end-to-end tests (real-model gated)
# -----------------------------------------------------------------------------
#
# Structural loader tests are now split by variant so the 0.6B and
# 1.7B GGUFs can each be validated against their own hparam roster.
# Each structural test looks for its variant-specific env var and
# skips (RC 77) when unset:
#
#   TRANSCRIBE_QWEN3_ASR_0_6B_GGUF  -> transcribe_qwen3_asr_real_smoke_0_6b
#   TRANSCRIBE_QWEN3_ASR_1_7B_GGUF  -> transcribe_qwen3_asr_real_smoke_1_7b
#
# The e2e smoke still uses TRANSCRIBE_QWEN3_ASR_GGUF (variant-agnostic
# — either variant is a valid e2e target). Built only when
# TRANSCRIBE_BUILD_REAL_MODEL_TESTS=ON.

if(TRANSCRIBE_BUILD_REAL_MODEL_TESTS)
    foreach(_variant 0_6b 1_7b)
        set(_target transcribe_qwen3_asr_real_smoke_${_variant})
        add_executable(${_target} qwen3_asr_real_smoke_${_variant}.cpp)
        target_link_libraries(${_target} PRIVATE transcribe ggml)
        target_include_directories(${_target} PRIVATE
            ${CMAKE_SOURCE_DIR}/src)
        transcribe_apply_warnings(${_target})
        add_test(NAME ${_target} COMMAND ${_target})
        set_tests_properties(${_target} PROPERTIES SKIP_RETURN_CODE 77)
    endforeach()

    # BPE encoder parity test (TRANSCRIBE_QWEN3_ASR_0_6B_GGUF gated).
    # Loads the real 0.6B tokenizer and compares Tokenizer::encode and
    # encode_language_prefix output against HF-generated expected ids
    # from tests/fixtures/qwen3_asr_bpe_parity.inc. That fixture is
    # hand-regenerated via scripts/tokenizer-parity-fixture.py when
    # the vocab or publisher language list moves.
    add_executable(transcribe_qwen3_asr_bpe_parity
        qwen3_asr_bpe_parity.cpp)
    target_link_libraries(transcribe_qwen3_asr_bpe_parity PRIVATE transcribe ggml)
    target_include_directories(transcribe_qwen3_asr_bpe_parity PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR})
    transcribe_apply_warnings(transcribe_qwen3_asr_bpe_parity)
    add_test(NAME transcribe_qwen3_asr_bpe_parity
        COMMAND transcribe_qwen3_asr_bpe_parity)
    set_tests_properties(transcribe_qwen3_asr_bpe_parity PROPERTIES
        SKIP_RETURN_CODE 77)

    add_executable(transcribe_qwen3_asr_e2e_smoke
        qwen3_asr_e2e_smoke.cpp)

    target_link_libraries(transcribe_qwen3_asr_e2e_smoke
        PRIVATE transcribe transcribe-common-example)

    target_compile_definitions(transcribe_qwen3_asr_e2e_smoke PRIVATE
        "TRANSCRIBE_TEST_SAMPLES_DIR=\"${CMAKE_SOURCE_DIR}/samples\"")

    transcribe_apply_warnings(transcribe_qwen3_asr_e2e_smoke)

    add_test(NAME transcribe_qwen3_asr_e2e_smoke
        COMMAND transcribe_qwen3_asr_e2e_smoke)
    set_tests_properties(transcribe_qwen3_asr_e2e_smoke PROPERTIES
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
# Tokenizer decode-only mode unit test
# -----------------------------------------------------------------------------
#
# Covers the in-memory load_decode_only_gpt2 / load_decode_only_raw_bytes
# constructors that the legacy whisper.cpp .bin adapter (PR3) will use.
# No fixtures: the test builds tiny in-memory vocabularies and asserts
# decode() picks the right strategy (byte-unicode inversion vs verbatim
# concatenation), encoder is reported absent, and DecodeOnlySpecials
# values are propagated.

add_executable(transcribe_tokenizer_decode_only_unit
    tokenizer_decode_only_unit.cpp)

target_link_libraries(transcribe_tokenizer_decode_only_unit PRIVATE transcribe)

target_include_directories(transcribe_tokenizer_decode_only_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_tokenizer_decode_only_unit)

add_test(NAME transcribe_tokenizer_decode_only_unit
    COMMAND transcribe_tokenizer_decode_only_unit)

# -----------------------------------------------------------------------------
# Whisper .bin parser unit test
# -----------------------------------------------------------------------------
#
# Covers the legacy whisper.cpp `.bin` parser surfaces:
#   - missing path → ERR_FILE_NOT_FOUND
#   - non-Whisper `ggml`-magic file (Silero VAD) → ERR_UNSUPPORTED_ARCH
#     (gated by TRANSCRIBE_WHISPER_BIN_SILERO)
#   - truncated `for-tests-*.bin` → ERR_GGUF
#     (gated by TRANSCRIBE_WHISPER_BIN_TRUNCATED)
#   - real ggml-tiny-q8_0.bin parses to expected metadata
#     (gated by TRANSCRIBE_WHISPER_BIN_TINY_Q8_0)
#
# Always built; sub-tests skip when their gating env var is unset.

add_executable(transcribe_whisper_bin_parser_unit
    whisper_bin_parser_unit.cpp)

target_link_libraries(transcribe_whisper_bin_parser_unit PRIVATE transcribe ggml)

target_include_directories(transcribe_whisper_bin_parser_unit PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

transcribe_apply_warnings(transcribe_whisper_bin_parser_unit)

add_test(NAME transcribe_whisper_bin_parser_unit
    COMMAND transcribe_whisper_bin_parser_unit)

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
