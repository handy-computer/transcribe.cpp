# transcribe-turn-live: a live turn-taking loop over a WAV fed in 20 ms
# chunks, the way a voice service drives the VAD and TURN roles: speech_start
# and turn events. Two setups: a VAD model plus a clip turn model (Silero +
# Smart Turn), or one model that streams both (SAVA). The turn rule is the
# caller's (docs/turn.md); this example implements upstream's rules.

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
