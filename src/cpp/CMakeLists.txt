cmake_minimum_required(VERSION 3.26)

set(vision_api_VERSION 0.0.0)

project(vision_api
        VERSION ${vision_api_VERSION}
        DESCRIPTION "OpenVINO Vision API"
        HOMEPAGE_URL "https://github.com/openvinotoolkit/model_api/"
        LANGUAGES CXX C)

find_package(OpenCV REQUIRED COMPONENTS core imgproc)

find_package(OpenVINO REQUIRED
             COMPONENTS Runtime Threading)

file(GLOB TASK_SOURCES src/tasks/**/*.cpp)
file(GLOB TASKS_SOURCES src/tasks/*.cpp)
file(GLOB UTILS_SOURCES src/utils/*.cpp)
file(GLOB ADAPTERS_SOURCES src/adapters/*.cpp)

add_library(model_api STATIC ${TASK_SOURCES} ${TASKS_SOURCES} ${UTILS_SOURCES} ${ADAPTERS_SOURCES} ${TILERS_SOURCES})

target_link_libraries(model_api PUBLIC openvino::runtime opencv_core opencv_imgproc)
target_include_directories(model_api PUBLIC ${PROJECT_SOURCE_DIR}/include)
