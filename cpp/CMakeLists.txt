cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(trt_infer)

set(AMD64 0)
 
if(WIN32)
	MESSAGE(STATUS "WIN32")
	set(CMAKE_PREFIX_PATH "D:/programming/opencv/opencv-4.3.0/build")
elseif(AMD64)
	MESSAGE(STATUS "AMD64")
	set(CMAKE_PREFIX_PATH "/home/gyz/programming/TensorRT-7.1.3.4")
	set(Torch_DIR /home/gyz/programming/libtorch-latest/share/cmake/Torch)
	include_directories(/home/gyz/programming/TensorRT-7.1.3.4/include)
	link_directories(/home/gyz/programming/TensorRT-7.1.3.4/lib)
else()
	MESSAGE(STATUS "ARM64")
	set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
	set(Torch_DIR /home/gyz/.local/lib/python3.6/site-packages/torch/share/cmake/Torch)
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
add_executable(trt_infer offline_test.cpp tracker.cpp)
target_link_libraries(trt_infer ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TORCH_LIBRARIES} nvinfer)

include_directories(./common)
add_executable(sample_dynamic_reshape sampleDynamicReshape.cpp ./common/logger.cpp)
target_link_libraries(sample_dynamic_reshape ${CUDA_LIBRARIES} ${OpenCV_LIBS} nvinfer nvonnxparser)

set_property(TARGET trt_infer PROPERTY CXX_STANDARD 14)
set_property(TARGET sample_dynamic_reshape PROPERTY CXX_STANDARD 14)