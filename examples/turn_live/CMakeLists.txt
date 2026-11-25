# transcribe-turn-live: a live turn-taking loop over a WAV (see main.cpp).

add_executable(transcribe-turn-live
    main.cpp
)

target_link_libraries(transcribe-turn-live
    PRIVATE
        transcribe
        transcribe-common-example
)

transcribe_apply_warnings(transcribe-turn-live)

set_target_properties(transcribe-turn-live PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
