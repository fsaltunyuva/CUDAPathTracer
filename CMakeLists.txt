cmake_minimum_required(VERSION 3.20)
project(GPUPathTracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -use_fast_math")

add_executable(PathTracerCUDA main.cu)

target_link_libraries(PathTracerCUDA PRIVATE curand)
