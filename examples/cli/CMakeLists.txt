# transcribe-cli: example CLI binary.
#
# Pass 2: loads a WAV, prints metadata, and surfaces the stub library's
# NOT_IMPLEMENTED status via transcribe_status_string. The same binary
# will run real transcription once the loader and decoder land.

add_executable(transcribe-cli
    main.cpp
)

target_link_libraries(transcribe-cli
    PRIVATE
        transcribe
        transcribe-common-example
)

transcribe_apply_warnings(transcribe-cli)
