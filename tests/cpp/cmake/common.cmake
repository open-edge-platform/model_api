# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.26)

macro(add_test)
    set(oneValueArgs NAME OPENCV_VERSION_REQUIRED)
    set(multiValueArgs SOURCES HEADERS DEPENDENCIES INCLUDE_DIRECTORIES)
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})

    if(TEST_OPENCV_VERSION_REQUIRED AND OpenCV_VERSION VERSION_LESS TEST_OPENCV_VERSION_REQUIRED)
        message(WARNING "${TEST_NAME} is disabled; required OpenCV version ${TEST_OPENCV_VERSION_REQUIRED}, provided ${OpenCV_VERSION}")
        return()
    endif()

    # Create named folders for the sources within the .vcproj
    # Empty name lists them directly under the .vcproj
    source_group("src" FILES ${TEST_SOURCES})
    if(TEST_HEADERS)
        source_group("include" FILES ${TEST_HEADERS})
    endif()

    # Create executable file from sources
    add_executable(${TEST_NAME} ${TEST_SOURCES} ${TEST_HEADERS})

    if(WIN32)
        set_target_properties(${TEST_NAME} PROPERTIES COMPILE_PDB_NAME ${TEST_NAME})
    endif()

    include_directories(${GTEST_INCLUDE_DIRS})

    if(TEST_INCLUDE_DIRECTORIES)
        target_include_directories(${TEST_NAME} PRIVATE ${TEST_INCLUDE_DIRECTORIES})
    endif()

    target_link_libraries(${TEST_NAME} PRIVATE ${OpenCV_LIBRARIES} ${TEST_DEPENDENCIES})

    if(UNIX)
        target_link_libraries(${TEST_NAME} PRIVATE pthread)
    endif()

    target_link_libraries(${TEST_NAME} PRIVATE gtest gtest_main)
    target_link_libraries(${TEST_NAME} PRIVATE nlohmann_json::nlohmann_json)

endmacro()
