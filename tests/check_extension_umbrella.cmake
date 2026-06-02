if(NOT DEFINED TRANSCRIBE_SOURCE_DIR)
    message(FATAL_ERROR "TRANSCRIBE_SOURCE_DIR is required")
endif()

set(include_dir "${TRANSCRIBE_SOURCE_DIR}/include/transcribe")
set(umbrella "${include_dir}/extensions.h")

if(NOT EXISTS "${umbrella}")
    message(FATAL_ERROR "missing umbrella header: ${umbrella}")
endif()

file(READ "${umbrella}" umbrella_contents)
file(GLOB family_headers "${include_dir}/*.h")

set(missing)
foreach(header IN LISTS family_headers)
    get_filename_component(name "${header}" NAME)
    if(name STREQUAL "extensions.h")
        continue()
    endif()

    set(expected "#include \"transcribe/${name}\"")
    string(FIND "${umbrella_contents}" "${expected}" found_at)
    if(found_at EQUAL -1)
        list(APPEND missing "${expected}")
    endif()
endforeach()

if(missing)
    string(REPLACE ";" "\n  " missing_lines "${missing}")
    message(FATAL_ERROR
        "include/transcribe/extensions.h is missing family header includes:\n"
        "  ${missing_lines}")
endif()
