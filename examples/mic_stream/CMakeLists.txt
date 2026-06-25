# transcribe-mic-stream: live microphone streaming demo.
#
# Reads raw S16_LE 16 kHz mono PCM from stdin (pipe from arecord/ffmpeg)
# and rewrites one terminal line with the committed/tentative transcript
# in real time. See the header comment in main.c for invocations.

add_executable(transcribe-mic-stream
    main.c
)

target_link_libraries(transcribe-mic-stream
    PRIVATE
        transcribe
)
