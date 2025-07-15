// main.cpp
#include "extractor_matcher.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

#ifndef EXTRACTOR_MODEL_PATH
#error "EXTRACTOR_MODEL_PATH is not defined. Please define it using -DEXTRACTOR_MODEL_PATH=\"path/to/model.onnx\"."
#endif

#ifndef MATCHER_MODEL_PATH
#error "MATCHER_MODEL_PATH is not defined. Please define it using -DMATCHER_MODEL_PATH=\"path/to/model.onnx\"."
#endif

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image_path_1> <image_path_2>" << std::endl;
        return 1;
    }

    cv::Mat img0 = cv::imread(argv[1]);
    cv::Mat img1 = cv::imread(argv[2]);
    if (img0.empty() || img1.empty())
    {
        std::cerr << "Failed to load images." << std::endl;
        return 1;
    }

    std::vector<cv::Mat> images = {img0, img1};

    // Extract features
    Extractor extractor(EXTRACTOR_MODEL_PATH);
    std::vector<std::vector<cv::Point2f>> imgs_keypts;
    std::vector<std::vector<std::vector<float>>> imgs_descriptors;
    extractor.Run(images, imgs_keypts, imgs_descriptors);

    // Draw keypoints
    for (int i = 0; i < images.size(); ++i)
    {
        cv::Mat img_keypts = images[i].clone();
        std::cout << "image: " << i << ", keypoints count: " << imgs_keypts[i].size() << std::endl;

        for (const auto &kp : imgs_keypts[i])
        {
            cv::circle(img_keypts, kp, 3, cv::Scalar(0, 0, 255), -1);
        }

        cv::Mat img_keypts_resized;
        cv::resize(img_keypts, img_keypts_resized, cv::Size(1920, 960));
        cv::imshow("keypoints" + std::to_string(i), img_keypts_resized);
        cv::waitKey(0);
    }

    // Match features
    Matcher matcher(MATCHER_MODEL_PATH);
    std::vector<std::pair<unsigned int, unsigned int>> matches;
    std::vector<float> scores;
    matcher.Run(imgs_keypts, imgs_descriptors, img0.cols, img0.rows, matches, scores);

    // Draw matches
    cv::Mat img_combined;
    cv::vconcat(img0, img1, img_combined);
    std::cout << "matches num: " << matches.size() << std::endl;
    unsigned int match_num = 0;
    for (int i = 0; i < matches.size(); ++i)
    {
        if (scores[i] < 0.5)
            continue;
        cv::Point2f pt1 = imgs_keypts[0][matches[i].first];
        cv::Point2f pt2 = imgs_keypts[1][matches[i].second] + cv::Point2f(0, img0.rows);
        cv::circle(img_combined, pt1, 3, cv::Scalar(0, 0, 255), -1);
        cv::circle(img_combined, pt2, 3, cv::Scalar(0, 0, 255), -1);
        cv::line(img_combined, pt1, pt2, cv::Scalar(0, 255, 0), 1);
        match_num++;
    }

    std::cout << "match_num (threshold: 0.5): " << match_num << std::endl;
    cv::Mat img_combined_resized;
    cv::resize(img_combined, img_combined_resized, cv::Size(1200, 1200));
    cv::imshow("matches", img_combined_resized);
    cv::waitKey(0);

    return 0;
}
