cmake_minimum_required(VERSION 3.19)

project(transcribe_ep_prefix_parent LANGUAGES C CXX)

if(NOT TRANSCRIBE_SOURCE_DIR)
    message(FATAL_ERROR "TRANSCRIBE_SOURCE_DIR is required")
endif()

# Model an embedding project that keeps all ExternalProject state in a
# consumer-owned location. transcribe must inherit this instead of replacing it
# with its Windows MAX_PATH default.
include(ExternalProject)
set(_expected_ep_prefix "${CMAKE_CURRENT_BINARY_DIR}/consumer-ep")
set_property(DIRECTORY PROPERTY EP_PREFIX "${_expected_ep_prefix}")

set(TRANSCRIBE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(TRANSCRIBE_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${TRANSCRIBE_SOURCE_DIR}" transcribe)

get_property(
    _actual_ep_prefix
    DIRECTORY "${TRANSCRIBE_SOURCE_DIR}"
    PROPERTY EP_PREFIX)
if(NOT _actual_ep_prefix STREQUAL _expected_ep_prefix)
    message(FATAL_ERROR
        "transcribe replaced the embedding project's EP_PREFIX: "
        "expected '${_expected_ep_prefix}', got '${_actual_ep_prefix}'")
endif()

message(STATUS "transcribe preserved inherited EP_PREFIX: ${_actual_ep_prefix}")
