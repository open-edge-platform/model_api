cmake_minimum_required(VERSION 3.26)


project(tests)

add_subdirectory(../../../src/cpp ${tests_BINARY_DIR}/src/cpp)

include(FetchContent)

FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git
                          GIT_TAG d41ca94fa85d5119852e2f7a3f94335cc7cb0486  # PR #4709, fixes cmake deprecation warnings
                    )
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG a7f443b80b105f940225332ed3c31f2790092f47  # latest main
)
FetchContent_MakeAvailable(json googletest)

find_package(OpenCV REQUIRED COMPONENTS imgcodecs)

add_executable(test_sanity test_sanity.cpp)
target_link_libraries(test_sanity gtest_main nlohmann_json::nlohmann_json model_api ${OpenCV_LIBRARIES})
