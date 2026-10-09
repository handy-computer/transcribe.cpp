# Cheap VAD CLI regression on seeded synthetic weights (probabilities < 1).
# Pin strict parsing, streaming thresholds/counts (including the final feed
# and padded flush), and the speech lines written by -o. No real model needed.

foreach(_var CLI MODEL WAV TEST_DIR)
    if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
        message(FATAL_ERROR "${_var} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
set(_output "${TEST_DIR}/speech.txt")

function(run_cli out_var rc_var err_var)
    execute_process(
        COMMAND "${CLI}" -q --backend cpu --threads 1 ${ARGN}
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err)
    set(${out_var} "${out}" PARENT_SCOPE)
    set(${rc_var} "${rc}" PARENT_SCOPE)
    set(${err_var} "${err}" PARENT_SCOPE)
endfunction()

# An empty argument must survive command construction, not become a missing
# argument or be silently dropped by expansion of ARGN in the helper above.
foreach(value IN ITEMS "" garbage 0.5tail "0.5 " " 0.5" NaN nan Inf -Inf 1e999 1e-999 -0.1 1.1)
    execute_process(
        COMMAND "${CLI}" --vad-threshold "${value}" "${WAV}"
        RESULT_VARIABLE rc
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err)
    if(rc EQUAL 0 OR NOT err MATCHES "--vad-threshold must be a finite number in \\[0, 1\\]")
        message(FATAL_ERROR "malformed threshold '${value}': rc=${rc}\n${out}\n${err}")
    endif()
endforeach()
execute_process(
    COMMAND "${CLI}" --vad-threshold
    RESULT_VARIABLE rc
    ERROR_VARIABLE err)
if(rc EQUAL 0 OR NOT err MATCHES "--vad-threshold requires an argument")
    message(FATAL_ERROR "missing threshold: rc=${rc}\n${err}")
endif()

# The offline count is also the expected streaming count, including padding.
run_cli(out rc err -m "${MODEL}" "${WAV}")
if(NOT rc EQUAL 0 OR NOT out MATCHES "frames: +([1-9][0-9]*) ")
    message(FATAL_ERROR "offline VAD: rc=${rc}\n${out}\n${err}")
endif()
set(n_frames "${CMAKE_MATCH_1}")

run_cli(out rc err -m "${MODEL}" --stream-chunk-ms 100 "${WAV}")
if(NOT rc EQUAL 0 OR NOT out MATCHES "speech: [0-9]+ of ${n_frames} frames at p >= 0\\.5")
    message(FATAL_ERROR "default streaming threshold: rc=${rc}\n${out}\n${err}")
endif()

# A large last feed produces frames which must be counted before flush replaces
# them; small chunks also exercise feeds which produce no complete frame.
foreach(chunk IN ITEMS 10 10000)
    foreach(threshold IN ITEMS 0 1)
        run_cli(out rc err -m "${MODEL}" --stream-chunk-ms "${chunk}"
            --vad-threshold "${threshold}" -o "${_output}" "${WAV}")
        if(threshold EQUAL 0)
            set(n_speech "${n_frames}")
        else()
            set(n_speech 0)
        endif()
        set(line "speech: ${n_speech} of ${n_frames} frames at p >= ${threshold}")
        if(NOT rc EQUAL 0 OR NOT out MATCHES "${line}\n")
            message(FATAL_ERROR "stream chunk=${chunk} threshold=${threshold}: rc=${rc}\n${out}\n${err}")
        endif()
        file(READ "${_output}" written)
        if(NOT written STREQUAL "${line}\n")
            message(FATAL_ERROR "-o wrote unexpected speech counts:\n${written}")
        endif()
    endforeach()
endforeach()
