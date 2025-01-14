cmake_minimum_required(VERSION 3.18)
project(GTSAMExample)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find GTSAM
# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)

# Include GTSAM
include_directories(${GTSAM_INCLUDE_DIR})

# Add executable
add_executable(GTSAMExample main.cpp)

# Link GTSAM
target_link_libraries(GTSAMExample Eigen3::Eigen gtsam)