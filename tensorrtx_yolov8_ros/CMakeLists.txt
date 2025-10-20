cmake_minimum_required(VERSION 3.10)
project(tensorrtx_yolov8_ros)

## Compile as C++17, supported in ROS Kinetic and newer
add_compile_options(-std=c++17)
add_definitions(-DAPI_EXPORTS)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS 
  roscpp 
  rospy
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  vision_msgs
  image_geometry
  pcl_conversions
  pcl_ros
  message_filters
)

catkin_package(
  CATKIN_DEPENDS 
  roscpp 
  rospy 
  std_msgs 
  sensor_msgs 
  image_transport 
  cv_bridge
  vision_msgs
  image_geometry
  pcl_conversions
  pcl_ros
  message_filters
)

## Enable CUDA
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  message("embed_platform on")
  include_directories(/usr/local/cuda/targets/aarch64-linux/include)
  link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
  message("embed_platform off")
  # cuda
  include_directories(/usr/local/cuda/include)
  link_directories(/usr/local/cuda/lib64)

  # tensorrt
  include_directories(/usr/local/TensorRT-8.6.1.6/include)
  link_directories(/usr/local/TensorRT-8.6.1.6/lib)
endif()

# Add YOLOv8 plugins library
add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/include/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Gather all source files
file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.cu)

add_executable(yolov8_seg src/yolov8_seg.cpp ${SRCS})
target_link_libraries(yolov8_seg
  ${catkin_LIBRARIES}
  nvinfer
  cudart
  myplugins
  ${OpenCV_LIBS}
)

