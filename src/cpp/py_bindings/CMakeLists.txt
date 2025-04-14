# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.26)

if (MSVC)
set(CMAKE_GENERATOR_TOOLSET "v142")
endif()

get_cmake_property(_variableNames VARIABLES)
add_subdirectory(../ model_api/cpp)

set(Python_FIND_VIRTUALENV FIRST)
project(_vision_api LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

#execute_process(
#  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
#  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

set(VISION_API_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/vision_api)
file(GLOB BINDINGS_SOURCES ${VISION_API_SOURCE_DIR}/*.cpp)
file(GLOB BINDINGS_HEADERS ${VISION_API_SOURCE_DIR}/*.hpp)

message(INFO ${BINDINGS_SOURCES})

nanobind_add_module(_vision_api NB_STATIC STABLE_ABI LTO ${BINDINGS_SOURCES} ${BINDINGS_HEADERS})

include(OpenCVDependencies.cmake)

set_target_properties(_vision_api PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${VISION_API_SOURCE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${VISION_API_SOURCE_DIR}
)

target_link_libraries(_vision_api PRIVATE model_api)
