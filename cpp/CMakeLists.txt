cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(trt_infer)

set(AMD64 0)
 
if(WIN32)
	MESSAGE(STATUS "WIN32")
	set(CMAKE_PREFIX_PATH "D:/programming/opencv/opencv-4.3.0/build")
elseif(AMD64)
	MESSAGE(STATUS "AMD64")
	set(CMAKE_PREFIX_PATH "/home/gyz/programming/TensorRT-7.0.0.11")
	set(Torch_DIR /home/gyz/programming/libtorch-1.5.1/share/cmake/Torch)
	include_directories(/home/gyz/programming/TensorRT-7.0.0.11/include)
	link_directories(/home/gyz/programming/TensorRT-7.0.0.11/lib)
else()
	MESSAGE(STATUS "ARM64")
	set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
	set(Torch_DIR /home/gyz/.local/lib/python3.6/site-packages/torch/share/cmake/Torch)
	#set(CUDNN_LIBRARY_PATH /usr/lib/aarch64-linux-gnu)
endif()



find_package(OpenCV REQUIRED)
if(${OPENCV_FOUND})
	include_directories(${OPENCV_INCLUDE_DIRS})
else(${OPENCV_FOUND})
	MESSAGE(STATUS "opencv not found!")
endif(${OPENCV_FOUND})

find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
	include_directories(${CUDA_INCLUDE_DIRS})
else(${CUDA_FOUND})
	MESSAGE(STATUS "cuda not found!")
endif(${CUDA_FOUND})

find_package(Torch REQUIRED)
if(${TORCH_FOUND})
	include_directories(${TORCH_INCLUDE_DIRS})
else(${TORCH_FOUND})
	MESSAGE(STATUS "torch not found!")
endif(${TORCH_FOUND})

#add_executable(trt_infer main.cpp common/logger.cpp)
add_executable(trt_infer main1.cpp tracker.cpp)
target_link_libraries(trt_infer ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} nvinfer)

set_property(TARGET trt_infer PROPERTY CXX_STANDARD 14)