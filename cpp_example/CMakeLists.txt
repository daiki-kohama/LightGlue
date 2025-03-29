cmake_minimum_required(VERSION 3.10)
project(onnx_cpp_demo)

find_package(OpenCV REQUIRED)

# define ONNXRUNTIME_DIR as a cache variable
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime directory")

# if ONNXRUNTIME_DIR is not set, check if the environment variable is set
if(NOT ONNXRUNTIME_DIR)
    set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
endif()

# if ONNXRUNTIME_DIR is still not set, throw an error
if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "ONNXRUNTIME_DIR not set. Set it via -D or environment variable.")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Require user to define model paths at configure time
if(NOT DEFINED EXTRACTOR_MODEL_PATH OR NOT DEFINED MATCHER_MODEL_PATH)
    message(FATAL_ERROR "You must define EXTRACTOR_MODEL_PATH and MATCHER_MODEL_PATH using -DEXTRACTOR_MODEL_PATH=... and -DMATCHER_MODEL_PATH=...")
endif()

# Add preprocessor definitions
add_compile_definitions(
    EXTRACTOR_MODEL_PATH="${EXTRACTOR_MODEL_PATH}"
    MATCHER_MODEL_PATH="${MATCHER_MODEL_PATH}"
)

add_executable(main main.cpp extractor_matcher.cpp)
target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS} onnxruntime)
