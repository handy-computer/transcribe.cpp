# transcribe-hello-stream: minimal streaming-API example.
#
# Pure-C peer of transcribe-hello. Drives transcribe_stream_begin /
# _feed / _finalize over a chunked WAV and prints committed-vs-tentative
# tokens on every update. Intended as the canonical reference for what
# a streaming consumer looks like through the public C ABI; binding
# authors should be able to read this and the offline hello/main.c
# back-to-back to understand the full surface.

add_executable(transcribe-hello-stream
    main.c
)

target_link_libraries(transcribe-hello-stream
    PRIVATE
        transcribe
)

# dr_wav.h lives next to the common example helpers.
target_include_directories(transcribe-hello-stream
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(main.c PROPERTIES COMPILE_OPTIONS
        "-Wno-sign-compare;-Wno-unused-function")
endif()
