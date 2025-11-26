cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang")
set(CMAKE_CXX_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(tft_ngarch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# 🔹 Find dependencies
find_package(Torch REQUIRED)

# 🔹 Include directories
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/libtorch_tft/include
)

# 🔹 Collect all source files (except test files)
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
file(GLOB_RECURSE TFT_SOURCE_FILES libtorch_tft/src/*.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_test.cpp$")
list(FILTER TFT_SOURCE_FILES EXCLUDE REGEX ".*_test.cpp$")

# Combine all source files
set(ALL_SOURCE_FILES ${SOURCE_FILES} ${TFT_SOURCE_FILES})

# 🔹 Create static library for main project
add_library(tft_ngarch_lib STATIC ${ALL_SOURCE_FILES})

# 🔹 Link dependencies
target_link_libraries(tft_ngarch_lib PRIVATE ${TORCH_LIBRARIES})

# 🔹 Collect and build test files (if needed)
file(GLOB TEST_FILES src/*_test.cpp)
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE tft_ngarch_lib)
endforeach()

# 🔹 Set additional compiler options
# set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
