cmake_minimum_required(VERSION 3.12)
project(onnx_inference)
set(CMAKE_CXX_STANDARD 17)

set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation")
add_executable(onnx_inference ort_inference_from_model.cpp)

include_directories(
        ${ONNXRUNTIME_DIR}/build/native/include
        ${CMAKE_SOURCE_DIR}/include
    )
target_link_libraries(
        onnx_inference 
        ${ONNXRUNTIME_DIR}/runtimes/linux-x64/native/libonnxruntime.so.1
    )