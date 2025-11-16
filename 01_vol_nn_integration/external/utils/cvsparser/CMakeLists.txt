cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang")
set(CMAKE_CXX_COMPILER "/Users/yawasante/software/miniconda3/envs/cpp_env/bin/clang++")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(CSVParser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# 🔹 Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 🔹 Add the test executable
add_executable(CSVParserTest CSVParserTest.cpp)

# 🔹 Link with pthread for multi-threading support
target_link_libraries(CSVParserTest PRIVATE pthread)

# 🔹 Set additional compiler options
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# 🔹 Add a custom target to run the test with the US Accidents dataset
add_custom_target(run_test
    COMMAND CSVParserTest
    DEPENDS CSVParserTest
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running CSV Parser test with US Accidents dataset"
)
