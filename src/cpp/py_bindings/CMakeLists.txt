# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.26)

add_subdirectory(../ model_api/cpp)

set(Python_FIND_VIRTUALENV FIRST)
project(vision_api LANGUAGES CXX)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)


file(GLOB BINDINGS_SOURCES ./src/*.cpp)
file(GLOB BINDINGS_HEADERS ./src/*.hpp)

nanobind_add_module(vision_api_impl NB_STATIC STABLE_ABI LTO ${BINDINGS_SOURCES} ${BINDINGS_HEADERS})

target_link_libraries(vision_api_impl PRIVATE model_api)

install(TARGETS vision_api_impl LIBRARY DESTINATION vision_api)

# nanobind_add_stub(
#     py_model_api_stub
#     MODULE vision_sdk
#     OUTPUT py_model_api.pyi
#     PYTHON_PATH $<TARGET_FILE_DIR:py_model_api>
#     DEPENDS py_model_api
# )
