cmake_minimum_required(VERSION 3.18)

if(EXISTS "${CMAKE_SOURCE_DIR}/cmake/LocalConfig.cmake")
    include("${CMAKE_SOURCE_DIR}/cmake/LocalConfig.cmake")
endif()

project(vol_nn_garch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# 🔹 Find dependencies
find_package(Torch REQUIRED)
# find_package(CUDA REQUIRED)     if you have CUDA

# 🔹 Include directories
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)


# 🔹 Collect all source files (except test files)
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_test.cpp$")

# 🔹 Create static library for main project
add_library(vol_nn_garch_lib STATIC ${SOURCE_FILES})

# 🔹 Link dependencies
target_link_libraries(vol_nn_garch_lib PRIVATE ${TORCH_LIBRARIES})

# 🔹 Collect and build test files (if needed)
file(GLOB TEST_FILES src/*_test.cpp)
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE vol_nn_garch_lib)
endforeach()

# 🔹 Set additional compiler options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")