cmake_minimum_required(VERSION 3.26)

set(vision_api_VERSION 0.0.0)

project(vision_api
        VERSION ${vision_api_VERSION}
        DESCRIPTION "OpenVINO Vision API"
        HOMEPAGE_URL "https://github.com/openvinotoolkit/model_api/"
        LANGUAGES CXX C)

include(FetchContent)
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git
                          GIT_TAG d41ca94fa85d5119852e2f7a3f94335cc7cb0486  # PR #4709, fixes cmake deprecation warnings
                    )
FetchContent_Declare(
        onnxrt
        URL      https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
        URL_HASH MD5=2b3a728a057226a3d6577335604db9bc
)

FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(onnxrt)


find_package(OpenCV REQUIRED COMPONENTS core imgproc)

find_package(OpenVINO REQUIRED
             COMPONENTS Runtime Threading)

file(GLOB TASK_SOURCES src/tasks/**/*.cpp)
file(GLOB TASKS_SOURCES src/tasks/*.cpp)
file(GLOB UTILS_SOURCES src/utils/*.cpp)
file(GLOB ADAPTERS_SOURCES src/adapters/*.cpp)

add_library(model_api STATIC ${TASK_SOURCES} ${TASKS_SOURCES} ${UTILS_SOURCES} ${ADAPTERS_SOURCES} ${TILERS_SOURCES})

target_link_libraries(model_api PUBLIC openvino::runtime opencv_core opencv_imgproc PRIVATE nlohmann_json::nlohmann_json)
target_include_directories(model_api PUBLIC ${PROJECT_SOURCE_DIR}/include)

# onnxrt doesn't have a working CMake config, set the related directories manually
set(ONNXRUNTIME_ROOTDIR ${CMAKE_BINARY_DIR}/_deps/onnxrt-src)

target_include_directories(model_api PUBLIC ${ONNXRUNTIME_ROOTDIR}/include)

target_link_directories(model_api PUBLIC "${ONNXRUNTIME_ROOTDIR}/lib")
target_link_libraries(model_api PUBLIC libonnxruntime.so)