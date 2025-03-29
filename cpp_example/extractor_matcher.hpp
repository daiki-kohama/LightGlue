// extractor_matcher.hpp
#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <utility>

std::vector<float> preprocess_for_extract(const cv::Mat &bgr_img);
std::vector<cv::Point2f> normalize_keypoints(const std::vector<cv::Point2f> &keypts, float width, float height);

class Extractor
{
public:
    explicit Extractor(const char *model_path);

    void Run(std::vector<cv::Mat> &images,
             std::vector<std::vector<cv::Point2f>> &imgs_keypts,
             std::vector<std::vector<std::vector<float>>> &imgs_descriptors);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime::Extractor"};
    Ort::Session session_{nullptr};
};

class Matcher
{
public:
    explicit Matcher(const char *model_path);

    void Run(std::vector<std::vector<cv::Point2f>> &imgs_keypts,
             std::vector<std::vector<std::vector<float>>> &imgs_descriptors,
             unsigned int width,
             unsigned int height,
             std::vector<std::pair<unsigned int, unsigned int>> &matches,
             std::vector<float> &scores);

private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime::Matcher"};
    Ort::Session session_{nullptr};
};
