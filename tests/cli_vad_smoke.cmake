# transcribe-cli routes a VAD model to the VAD path: prints the `segment:`
# lines, writes them with -o, honours --vad-threshold, and refuses --batch and
# --stream-chunk-ms. Structural checks only, on the toy fixture (MODEL).

file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")
set(_output "${TEST_DIR}/segments.txt")

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

# Threshold 0 marks every frame as speech: one segment over the whole clip.
run_cli(out rc err -m "${MODEL}" --vad-threshold 0 -o "${_output}" "${WAV}")
if(NOT rc EQUAL 0 OR NOT out MATCHES "speech segments: 1\n  segment: 0 start=0 end=[1-9][0-9]* "
   OR NOT out MATCHES "frames: +[1-9]")
    message(FATAL_ERROR "--vad-threshold 0: rc=${rc}\n${out}\n${err}")
endif()
file(READ "${_output}" written)
if(NOT written MATCHES "^segment: 0 start=0 end=[1-9][0-9]* t0=0\\.000 t1=[0-9.]+\n$")
    message(FATAL_ERROR "-o wrote unexpected segments:\n${written}")
endif()

run_cli(out rc err -m "${MODEL}" --repeat 2 "${WAV}")
if(NOT rc EQUAL 0 OR NOT out MATCHES "speech segments: [0-9]+\n")
    message(FATAL_ERROR "default run: rc=${rc}\n${out}\n${err}")
endif()

run_cli(out rc err -m "${MODEL}" --vad-threshold 1.5 "${WAV}")
if(rc EQUAL 0 OR NOT err MATCHES "--vad-threshold must be in \\[0, 1\\]")
    message(FATAL_ERROR "--vad-threshold 1.5: rc=${rc}\n${err}")
endif()

run_cli(out rc err -m "${MODEL}" --stream-chunk-ms 100 "${WAV}")
if(rc EQUAL 0 OR NOT err MATCHES "no streaming CLI mode")
    message(FATAL_ERROR "--stream-chunk-ms on a VAD model: rc=${rc}\n${err}")
endif()

file(WRITE "${TEST_DIR}/list.txt" "${WAV}\n")
run_cli(out rc err -m "${MODEL}" --batch "${TEST_DIR}/list.txt")
if(rc EQUAL 0 OR NOT err MATCHES "--batch is ASR-only")
    message(FATAL_ERROR "--batch on a VAD model: rc=${rc}\n${err}")
endif()
