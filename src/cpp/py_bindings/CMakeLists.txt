# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.26)

set(CMAKE_GENERATOR_TOOLSET "v142")

get_cmake_property(_variableNames VARIABLES)
add_subdirectory(../ model_api/cpp)

set(Python_FIND_VIRTUALENV FIRST)
project(_vision_api LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)


file(GLOB BINDINGS_SOURCES src/vision_api/*.cpp)
file(GLOB BINDINGS_HEADERS src/vision_api/*.hpp)

message(INFO ${BINDINGS_SOURCES})

nanobind_add_module(_vision_api NB_STATIC STABLE_ABI LTO ${BINDINGS_SOURCES} ${BINDINGS_HEADERS})

set_target_properties(_vision_api PROPERTIES
    # LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/vision_api"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/src/vision_api"
)

include(OpenCVDependencies.cmake)

set(PYTHON_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/vision_api)
foreach(lib ${DEPENDENCIES_TO_COPY})
    add_custom_command(
        TARGET _vision_api POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${lib}
            ${PYTHON_PACKAGE_DIR}
        COMMENT "Copying ${lib} to ${PYTHON_PACKAGE_DIR}"
    )
endforeach()

target_link_libraries(_vision_api PRIVATE model_api)
