# Example targets. Built when TRANSCRIBE_BUILD_EXAMPLES is ON.
#
# These targets are NOT part of the public ABI surface; they exist to drive
# the library from a real entry point.

add_subdirectory(common)
add_subdirectory(cli)
add_subdirectory(bench)
add_subdirectory(hello)
add_subdirectory(hello_stream)
add_subdirectory(mic_stream)
