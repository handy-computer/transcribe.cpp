# transcribe-turn-eval: scores many clips with one loaded TURN model through
# the public ABI. Reads clips from stdin (int32 n_samples, then n_samples
# float32 PCM, repeated) and prints one `p ended` line per clip. The Python
# driver (scripts/turn/eval_cpp.py) feeds it a dataset.

add_executable(transcribe-turn-eval
    main.cpp
)

target_link_libraries(transcribe-turn-eval
    PRIVATE
        transcribe
)

transcribe_apply_warnings(transcribe-turn-eval)

set_target_properties(transcribe-turn-eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
