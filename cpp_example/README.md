# How to use LightGlue ONNX models in c++

## Environment

- CMake
- CUDA
- OpenCV
- ONNX Runtime
  - Install onnxruntime compatible with CUDA via [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/releases) and unzip.

## Compile

```shell
mkdir -p cpp_example/build && cd cpp_example/build
cmake .. \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime \
  -DEXTRACTOR_MODEL_PATH="/absolute/path/to/extractor.onnx" \
  -DMATCHER_MODEL_PATH="/absolute/path/to/matcher.onnx"
make
```

## Run

```shell
./main <image_path_1> <image_path_2>
```
