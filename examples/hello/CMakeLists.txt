# transcribe-hello: the minimal canonical example.
#
# A pure-C program that loads a model and transcribes one WAV via the
# stable C ABI (transcribe_open / transcribe_run / transcribe_full_text /
# transcribe_close). Intended as the first thing a new user or a binding
# author reads. Links only the core library; the WAV decode uses dr_wav
# directly (from examples/common) so the example stays self-contained.

add_executable(transcribe-hello
    main.c
)

target_link_libraries(transcribe-hello
    PRIVATE
        transcribe
)

# dr_wav.h lives next to the common example helpers.
target_include_directories(transcribe-hello
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# dr_wav.h is third-party and trips our project warnings; silence the ones
# it raises (matching examples/common), but keep warnings on our own code.
if(CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
    set_source_files_properties(main.c PROPERTIES COMPILE_OPTIONS
        "-Wno-sign-compare;-Wno-unused-function")
endif()
