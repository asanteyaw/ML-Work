cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang")
set(CMAKE_CXX_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(hn_garch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# 🔹 Find dependencies
find_package(Torch REQUIRED)

# 🔹 Include directories
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external/dataframe/include  # Corrected include path
)

# 🔹 Add subdirectory for dataframe library
add_subdirectory(${CMAKE_SOURCE_DIR}/external/dataframe dataframe_lib)

# 🔹 Collect all source files (except test files)
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_test.cpp$")

# 🔹 Create static library for main project
add_library(hn_garch_lib STATIC ${SOURCE_FILES})

# 🔹 Link dependencies
target_link_libraries(hn_garch_lib PRIVATE dataframe ${TORCH_LIBRARIES})

# 🔹 Enable AddressSanitizer in Debug mode
# if(CMAKE_BUILD_TYPE MATCHES Debug)
#     message(STATUS "✅ AddressSanitizer enabled for Debug build in main project")
#     add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
#     add_link_options(-fsanitize=address)
# endif()

# 🔹 Collect and build test files (if needed)
file(GLOB TEST_FILES src/*_test.cpp)
foreach(TEST_FILE ${TEST_FILES})
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_FILE})
    target_link_libraries(${TEST_NAME} PRIVATE hn_garch_lib)
endforeach()

# 🔹 Set additional compiler options
# set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")