foreach(_var CLI WAV TEST_DIR)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR "${_var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")

set(_output "${TEST_DIR}/transcript.txt")
file(WRITE "${_output}" "stale output\n")
execute_process(
    COMMAND "${CLI}" -q -o "${_output}" "${WAV}"
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "output command failed (${_result}):\n${_stdout}\n${_stderr}")
endif()
file(SIZE "${_output}" _output_size)
if(NOT _output_size EQUAL 0)
    message(FATAL_ERROR "--output did not truncate stale output")
endif()

set(_bad_output "${TEST_DIR}/missing/transcript.txt")
execute_process(
    COMMAND "${CLI}" -q -o "${_bad_output}" "${WAV}"
    RESULT_VARIABLE _bad_result
    OUTPUT_VARIABLE _bad_stdout
    ERROR_VARIABLE _bad_stderr)
if(_bad_result EQUAL 0)
    message(FATAL_ERROR "unwritable output path unexpectedly succeeded")
endif()
if(EXISTS "${_bad_output}")
    message(FATAL_ERROR "unwritable output path unexpectedly created a file")
endif()
