foreach(_var CLI WAV)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR "${_var} is required")
    endif()
endforeach()

set(_model "$ENV{TRANSCRIBE_QWEN3_ASR_GGUF}")
if(_model STREQUAL "" OR NOT EXISTS "${_model}")
    message("SKIP: TRANSCRIBE_QWEN3_ASR_GGUF unset or missing")
    return()
endif()

execute_process(
    COMMAND "${CLI}"
        -q
        -m "${_model}"
        --context "Vocabulary: GGUF, ggml, Qwen3-ASR"
        "${WAV}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "context command failed (${_result}):\n${_stdout}\n${_stderr}")
endif()
