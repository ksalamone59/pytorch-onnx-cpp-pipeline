cmake_minimum_required(VERSION 3.12)

project(onnx_inference)

set(CMAKE_CXX_STANDARD 17)
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime installation")
set(ORT_LIB ${ONNXRUNTIME_DIR}/runtimes/linux-x64/native/libonnxruntime.so.1)

add_executable(onnx_inference ort_inference_from_model.cpp)
target_include_directories(onnx_inference PRIVATE
    ${ONNXRUNTIME_DIR}/build/native/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(onnx_inference PRIVATE ${ORT_LIB})

add_executable(onnx_benchmark benchmark.cpp)
target_include_directories(onnx_benchmark PRIVATE
    ${ONNXRUNTIME_DIR}/build/native/include
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(onnx_benchmark PRIVATE ${ORT_LIB})