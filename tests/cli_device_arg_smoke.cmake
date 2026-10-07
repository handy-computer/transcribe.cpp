# Verify --device rejects malformed, negative, trailing-junk, and overflowing
# values instead of silently treating them as exact device 0.

if(NOT DEFINED CLI OR NOT DEFINED WAV)
    message(FATAL_ERROR "CLI and WAV are required")
endif()

foreach(value IN ITEMS abc 0x1 1tail -1 2147483648)
    execute_process(
        COMMAND "${CLI}" --device "${value}" "${WAV}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr)
    if(result EQUAL 0)
        message(FATAL_ERROR
            "--device ${value} unexpectedly succeeded\nstdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    if(NOT stderr MATCHES "--device must be an integer index >= 0")
        message(FATAL_ERROR
            "--device ${value} returned the wrong diagnostic\nstderr:\n${stderr}")
    endif()
endforeach()
