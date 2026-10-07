# transcribe-cli routes a DIARIZE model to the diarize path and prints speaker
# segments for the 2-speaker oracle mix, writes them with -o, honours --repeat,
# and refuses --batch and --stream-chunk-ms. Gated on TRANSCRIBE_SORTFORMER_GGUF (reported as skipped).

if("$ENV{TRANSCRIBE_SORTFORMER_GGUF}" STREQUAL "")
    message("SKIP: TRANSCRIBE_SORTFORMER_GGUF not set")
    return()
endif()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
set(_output "${TEST_DIR}/segments.txt")

execute_process(
    COMMAND "${CLI}" -q --backend cpu -m "$ENV{TRANSCRIBE_SORTFORMER_GGUF}" -o "${_output}" "${WAV}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "transcribe-cli exited ${rc}\n${out}\n${err}")
endif()
if(NOT out MATCHES "speaker segments: [1-9].*S2")
    message(FATAL_ERROR "unexpected output:\n${out}")
endif()
file(READ "${_output}" written)
if(NOT written MATCHES "^\\[ +0\\.[0-9]+ -> +[0-9.]+\\] S[0-9].*S2\n$")
    message(FATAL_ERROR "-o wrote unexpected segments:\n${written}")
endif()

file(WRITE "${TEST_DIR}/list.txt" "${WAV}\n")
execute_process(
    COMMAND "${CLI}" -q --backend cpu -m "$ENV{TRANSCRIBE_SORTFORMER_GGUF}" --batch "${TEST_DIR}/list.txt"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(rc EQUAL 0 OR NOT err MATCHES "--batch is ASR-only")
    message(FATAL_ERROR "--batch on a diarize model: rc=${rc}\n${err}")
endif()

execute_process(
    COMMAND "${CLI}" -q --backend cpu -m "$ENV{TRANSCRIBE_SORTFORMER_GGUF}" --repeat 2 "${WAV}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(NOT rc EQUAL 0 OR NOT out MATCHES "speaker segments: [1-9].*S2")
    message(FATAL_ERROR "--repeat 2 on a diarize model: rc=${rc}\n${out}\n${err}")
endif()

execute_process(
    COMMAND "${CLI}" -q --backend cpu -m "$ENV{TRANSCRIBE_SORTFORMER_GGUF}" --stream-chunk-ms 100 "${WAV}"
    RESULT_VARIABLE rc
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err)
if(rc EQUAL 0 OR NOT err MATCHES "no streaming entry point")
    message(FATAL_ERROR "--stream-chunk-ms on a diarize model: rc=${rc}\n${err}")
endif()
