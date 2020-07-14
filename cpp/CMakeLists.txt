cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(trt_infer)

if(WIN32)
	set(CMAKE_PREFIX_PATH "D:/programming/opencv/opencv-4.3.0/build")
else()
	set(CMAKE_PREFIX_PATH "/home/gyz/programming/TensorRT-7.1.3.4")
endif()

include_directories(/home/gyz/programming/TensorRT-7.1.3.4/include)
link_directories(/home/gyz/programming/TensorRT-7.1.3.4/lib)

find_package(OpenCV REQUIRED)

add_executable(trt_infer main.cpp)
target_link_libraries(trt_infer ${OpenCV_LIBS} nvinfer)

set_property(TARGET trt_infer PROPERTY CXX_STANDARD 14)