cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(libtorch_tft)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add source files
file(GLOB_RECURSE SOURCES "src/*.cpp")
file(GLOB_RECURSE HEADERS "include/*.h")

# Create library
add_library(tft_lib ${SOURCES} ${HEADERS})
target_link_libraries(tft_lib ${TORCH_LIBRARIES})

# Create executable for training
add_executable(train_tft examples/train.cpp)
target_link_libraries(train_tft tft_lib ${TORCH_LIBRARIES})

# Create executable for inference
add_executable(predict_tft examples/predict.cpp)
target_link_libraries(predict_tft tft_lib ${TORCH_LIBRARIES})

# Copy dlls to output directory on Windows
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET train_tft
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:train_tft>)
endif (MSVC)
