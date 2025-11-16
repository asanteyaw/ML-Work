# cmake_minimum_required(VERSION 3.18)

# # ✅ Override compiler only for dataframe
# set(CMAKE_C_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang")
# set(CMAKE_CXX_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang++")
# set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# project(dataframe)

# set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # 🔹 Ensure ABI compatibility
# add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# # 🔹 Find dependencies
# find_package(Torch REQUIRED)

# # 🔹 Automatically includes necessary headers
# include_directories(
#     ${TORCH_INCLUDE_DIRS}
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
# )

# # 🔹 Find all .cpp files in src/
# file(GLOB_RECURSE SOURCE_FILES src/*.cpp)

# # 🔹 Create a static library (remove `main.cpp`)
# add_library(dataframe STATIC ${SOURCE_FILES})

# # 🔹 Ensure headers are accessible
# target_include_directories(dataframe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# # 🔹 Link Torch libraries (manually specify necessary components)
# target_link_libraries(dataframe PRIVATE "${TORCH_LIBRARIES}")

# # Explicitly link directories for Torch
# link_directories(${TORCH_INSTALL_PREFIX}/lib)

# # ✅ Use optimized Release Mode
# set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# # ✅ Enable full optimization with safer flags for ARM64
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# # 🔹 Add executables for *_test.cpp files
# file(GLOB_RECURSE TEST_FILES src/*_test.cpp)
# foreach(TEST_FILE ${TEST_FILES})
#     get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
#     add_executable(${TEST_NAME} ${TEST_FILE})
#     target_link_libraries(${TEST_NAME} PRIVATE dataframe)
#     message(STATUS "🔍 Added test executable: ${TEST_NAME}")
# endforeach()

# # 🔹 Add these flags to reduce warning noise
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-variable -Wno-sign-compare")

# # 🔹 Ensure proper architecture
# set(CMAKE_OSX_ARCHITECTURES "arm64")

cmake_minimum_required(VERSION 3.18)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(dataframe)

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
add_library(dataframe STATIC ${SOURCE_FILES})

# Ensure headers are accessible
target_include_directories(dataframe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link Torch libraries
link_directories(${TORCH_INSTALL_PREFIX}/lib)
target_link_libraries(dataframe PRIVATE "${TORCH_LIBRARIES}")

# Use optimized Release Mode
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

# Enable full optimization with safer flags for ARM64
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# Add warning flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-variable -Wno-sign-compare")

# Set architecture for macOS arm64
# set(CMAKE_OSX_ARCHITECTURES "arm64")

# Display configuration summary
message(STATUS "✅ Building DataFrame Library with C++20 and Torch Integration")
