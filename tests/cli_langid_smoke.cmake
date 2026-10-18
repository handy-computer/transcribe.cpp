# transcribe-cli routes a LANGID model to the language ID path: prints the
# `language:` / `candidate:` lines, writes the candidates with -o, honours
# --allow / --top / --repeat, and refuses --batch and --stream-chunk-ms.
# Structural checks only, on the toy fixture (MODEL).

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
set(_output "${TEST_DIR}/candidates.txt")

function(run_cli out_var rc_var err_var)
    execute_process(
        COMMAND "${CLI}" -q --backend cpu ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err)
    set(${out_var} "${out}" PARENT_SCOPE)
    set(${rc_var} "${rc}" PARENT_SCOPE)
    set(${err_var} "${err}" PARENT_SCOPE)
endfunction()

run_cli(out rc err -m "${MODEL}" -o "${_output}" "${WAV}")
if(NOT rc EQUAL 0 OR NOT out MATCHES "language: (aa|bb|cc|dd|ee) index=[0-4] p=")
    message(FATAL_ERROR "default run: rc=${rc}\n${out}\n${err}")
endif()
file(READ "${_output}" written)
if(NOT written MATCHES "^candidate: 1 [a-e][a-e] index=[0-4] p=[0-9.]+ logit=.*candidate: 5 ")
    message(FATAL_ERROR "-o wrote unexpected candidates:\n${written}")
endif()

run_cli(out rc err -m "${MODEL}" --allow bb,dd --repeat 2 "${WAV}")
if(NOT rc EQUAL 0 OR NOT out MATCHES "language: (bb|dd) " OR out MATCHES "candidate: 3 " OR NOT out MATCHES "2 allowed")
    message(FATAL_ERROR "--allow bb,dd: rc=${rc}\n${out}\n${err}")
endif()

run_cli(out rc err -m "${MODEL}" --top 1 "${WAV}")
if(NOT rc EQUAL 0 OR out MATCHES "candidate: 2 ")
    message(FATAL_ERROR "--top 1: rc=${rc}\n${out}\n${err}")
endif()

run_cli(out rc err -m "${MODEL}" --stream-chunk-ms 100 "${WAV}")
if(rc EQUAL 0 OR NOT err MATCHES "no streaming entry point")
    message(FATAL_ERROR "--stream-chunk-ms on a langid model: rc=${rc}\n${err}")
endif()

file(WRITE "${TEST_DIR}/list.txt" "${WAV}\n")
run_cli(out rc err -m "${MODEL}" --batch "${TEST_DIR}/list.txt")
if(rc EQUAL 0 OR NOT err MATCHES "--batch is ASR-only")
    message(FATAL_ERROR "--batch on a langid model: rc=${rc}\n${err}")
endif()
