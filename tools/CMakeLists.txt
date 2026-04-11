# tools/ - developer tools that are NOT part of the public ABI.
#
# Gated on TRANSCRIBE_BUILD_TOOLS at the root CMakeLists.txt level so a
# default `cmake -B build` produces just the library + example CLI. Each
# tool is its own subdirectory with its own add_executable; this file
# is just a dispatch.

add_subdirectory(transcribe-quantize)
add_subdirectory(transcribe-bench)
