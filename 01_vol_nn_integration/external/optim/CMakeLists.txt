# cmake_minimum_required(VERSION 3.18)

# set(CMAKE_C_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang")
# set(CMAKE_CXX_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang++")
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# project(DFO)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# # 🔹 Find dependencies
# find_package(Torch REQUIRED)

# # 🔹 Include directories
# include_directories(
#     ${TORCH_INCLUDE_DIRS}
#     ${CMAKE_SOURCE_DIR}/include
#     # ${CMAKE_SOURCE_DIR}/../../Xperimental/optim/include  # Corrected include path
# )

# # 🔹 Add subdirectory for optim library
# # add_subdirectory(${CMAKE_SOURCE_DIR}/../../Xperimental/optim optim_lib)

# # 🔹 Collect all source files (except test files)
# file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
# list(FILTER SOURCE_FILES EXCLUDE REGEX ".*_test.cpp$")

# # 🔹 Create static library for main project
# add_library(DFO_lib STATIC ${SOURCE_FILES})

# # 🔹 Link dependencies
# target_link_libraries(DFO_lib PRIVATE ${TORCH_LIBRARIES})

# # 🔹 Collect and build test files (if needed)
# file(GLOB TEST_FILES src/*_test.cpp)
# foreach(TEST_FILE ${TEST_FILES})
#     get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
#     add_executable(${TEST_NAME} ${TEST_FILE})
#     target_link_libraries(${TEST_NAME} PRIVATE DFO_lib)
# endforeach()

# # 🔹 Set additional compiler options
# # set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type")
# # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -g -O1")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(optim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure ABI compatibility
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# Find dependencies
find_package(Torch REQUIRED)

# Include necessary headers
include_directories(
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Find all .cpp files in src/
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

# Create a static library (no executables)
add_library(optim STATIC ${SOURCE_FILES})

# Ensure headers are accessible
target_include_directories(optim PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link Torch libraries
link_directories(${TORCH_INSTALL_PREFIX}/lib)
target_link_libraries(optim PRIVATE "${TORCH_LIBRARIES}")

# Use optimized Release Mode
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Enable full optimization with safer flags for ARM64
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-variable -Wno-sign-compare")

# Set architecture for macOS arm64
# set(CMAKE_OSX_ARCHITECTURES "arm64")

# Display configuration summary
message(STATUS "✅ Building optim Library with C++20 and Torch Integration")