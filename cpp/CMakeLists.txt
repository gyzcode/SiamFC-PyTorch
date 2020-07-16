cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(trt_infer)

include_directories("./common")

if(WIN32)
	set(CMAKE_PREFIX_PATH "D:/programming/opencv/opencv-4.3.0/build")
else()
	set(CMAKE_PREFIX_PATH "/home/gyz/programming/TensorRT-7.1.3.4")
endif()

include_directories(/home/gyz/programming/TensorRT-7.1.3.4/include)
link_directories(/home/gyz/programming/TensorRT-7.1.3.4/lib)

find_package(OpenCV REQUIRED)

find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
	include_directories(${CUDA_INCLUDE_DIRS})
else(${CUDA_FOUND})
	MESSAGE(STATUS "cuda not found!")
endif(${CUDA_FOUND})

add_executable(trt_infer main.cpp common/logger.cpp)
target_link_libraries(trt_infer ${OpenCV_LIBS} ${CUDA_LIBRARIES} nvinfer nvcaffe_parser)

set_property(TARGET trt_infer PROPERTY CXX_STANDARD 14)