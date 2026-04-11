# transcribe-bench: one-shot perf measurement binary for a single
# (model, sample) pair. Links against the public transcribe ABI and
# the shared WAV loader in transcribe-common-example. The Python
# driver invokes this binary across a matrix of models and samples.

add_executable(transcribe-bench
    main.cpp
)

target_link_libraries(transcribe-bench
    PRIVATE
        transcribe
        transcribe-common-example
)

transcribe_apply_warnings(transcribe-bench)

set_target_properties(transcribe-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
