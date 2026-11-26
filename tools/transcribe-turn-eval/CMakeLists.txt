# transcribe-turn-eval: scores clips from stdin with one TURN model (see
# main.cpp); scripts/turn/eval_cpp.py feeds it a dataset.

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
