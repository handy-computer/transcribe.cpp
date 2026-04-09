# transcribe-common-example: tiny static helpers shared by example binaries.
#
# Currently just the WAV loader. dr_wav.h lives next to wav.cpp because the
# core library does not decode WAV files - this is example-only code.

add_library(transcribe-common-example STATIC
    wav.cpp
)

target_include_directories(transcribe-common-example
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(transcribe-common-example PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Our project warnings, but not the ones that fire inside dr_wav.h itself.
transcribe_apply_warnings(transcribe-common-example)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(transcribe-common-example PRIVATE
        -Wno-sign-compare
        -Wno-shadow
        -Wno-unused-function
    )
endif()
