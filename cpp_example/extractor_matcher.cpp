// extractor_matcher.cpp
#include "extractor_matcher.hpp"

// Convert BGR image to normalized grayscale tensor (1x1xHxW)
std::vector<float> preprocess_for_extract(const cv::Mat &bgr_img)
{
    cv::Mat rgb_img;
    cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
    rgb_img.convertTo(rgb_img, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> rgb_channels(3);
    cv::split(rgb_img, rgb_channels);
    cv::Mat gray = 0.299 * rgb_channels[0] + 0.587 * rgb_channels[1] + 0.114 * rgb_channels[2];

    int height = gray.rows;
    int width = gray.cols;
    std::vector<float> input_tensor(1 * 1 * height * width);
    std::memcpy(input_tensor.data(), gray.data, height * width * sizeof(float));

    return input_tensor;
}

std::vector<cv::Point2f> normalize_keypoints(const std::vector<cv::Point2f> &keypts, float width, float height)
{
    std::vector<cv::Point2f> normalized;
    normalized.reserve(keypts.size());

    for (const auto &kp : keypts)
    {
        float x = 2.0f * kp.x / width - 1.0f;
        float y = 2.0f * kp.y / height - 1.0f;
        normalized.emplace_back(x, y);
    }

    return normalized;
}

Extractor::Extractor(const char *model_path)
{
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_ = Ort::Session(env_, model_path, session_options);
}

void Extractor::Run(std::vector<cv::Mat> &images,
                    std::vector<std::vector<cv::Point2f>> &imgs_keypts,
                    std::vector<std::vector<std::vector<float>>> &imgs_descriptors)
{
    const unsigned int width = images[0].cols;
    const unsigned int height = images[0].rows;
    const unsigned int image_num = images.size();

    std::vector<float> input_tensor_values;
    for (const auto &img : images)
    {
        std::vector<float> preprocessed = preprocess_for_extract(img);
        input_tensor_values.insert(input_tensor_values.end(), preprocessed.begin(), preprocessed.end());
    }

    std::vector<int64_t> input_shape = {image_num, 1, height, width};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    const char *input_names[] = {"images"};
    const char *output_names[] = {"top_keypoints", "top_scores", "top_descriptors"};
    auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 3);

    Ort::Value &keypts_tensor = output_tensors[0];
    int16_t *keypts_data = keypts_tensor.GetTensorMutableData<int16_t>();
    auto keypts_shape = keypts_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int num_keypts = keypts_shape[1];

    imgs_keypts.resize(image_num);
    for (int i = 0; i < image_num; ++i)
    {
        const unsigned int offset_i = i * num_keypts * 2;
        imgs_keypts[i].resize(num_keypts);
        for (int j = 0; j < num_keypts; ++j)
        {
            const unsigned int offset_j = j * 2;
            float x = static_cast<float>(keypts_data[offset_i + offset_j]);
            float y = static_cast<float>(keypts_data[offset_i + offset_j + 1]);
            imgs_keypts[i][j] = cv::Point2f(x, y);
        }
    }

    Ort::Value &descriptors_tensor = output_tensors[2];
    float *descriptors_data = descriptors_tensor.GetTensorMutableData<float>();
    auto descriptors_shape = descriptors_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int descriptor_size = descriptors_shape[2];

    imgs_descriptors.resize(image_num);
    for (int i = 0; i < image_num; ++i)
    {
        imgs_descriptors[i].resize(num_keypts);
        const unsigned int offset_i = i * num_keypts * descriptor_size;
        for (int j = 0; j < num_keypts; ++j)
        {
            imgs_descriptors[i][j].resize(descriptor_size);
            const unsigned int offset_j = j * descriptor_size;
            std::copy(
                descriptors_data + offset_i + offset_j,
                descriptors_data + offset_i + offset_j + descriptor_size,
                imgs_descriptors[i][j].begin());
        }
    }
}

Matcher::Matcher(const char *model_path)
{
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_ = Ort::Session(env_, model_path, session_options);
}

void Matcher::Run(std::vector<std::vector<cv::Point2f>> &imgs_keypts,
                  std::vector<std::vector<std::vector<float>>> &imgs_descriptors,
                  unsigned int width,
                  unsigned int height,
                  std::vector<std::pair<unsigned int, unsigned int>> &matches,
                  std::vector<float> &scores)
{
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<std::vector<cv::Point2f>> normalized_imgs_keypts;
    normalized_imgs_keypts.reserve(imgs_keypts.size());
    for (const auto &keypts : imgs_keypts)
    {
        normalized_imgs_keypts.emplace_back(normalize_keypoints(keypts, width, height));
    }

    std::vector<float> normalized_imgs_keypts_values;
    for (const auto &keypts : normalized_imgs_keypts)
    {
        for (const auto &keypt : keypts)
        {
            normalized_imgs_keypts_values.push_back(keypt.x);
            normalized_imgs_keypts_values.push_back(keypt.y);
        }
    }

    std::vector<int64_t> normalized_imgs_keypts_shape = {
        static_cast<int64_t>(normalized_imgs_keypts.size()),
        static_cast<int64_t>(normalized_imgs_keypts[0].size()),
        2};

    Ort::Value normalized_imgs_keypts_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        normalized_imgs_keypts_values.data(),
        normalized_imgs_keypts_values.size(),
        normalized_imgs_keypts_shape.data(),
        normalized_imgs_keypts_shape.size());

    std::vector<float> imgs_descriptors_values;
    imgs_descriptors_values.reserve(imgs_descriptors.size() * imgs_descriptors[0].size() * imgs_descriptors[0][0].size());
    for (const auto &descriptors : imgs_descriptors)
    {
        for (const auto &descriptor : descriptors)
        {
            imgs_descriptors_values.insert(imgs_descriptors_values.end(), descriptor.begin(), descriptor.end());
        }
    }

    std::vector<int64_t> imgs_descriptors_shape = {
        static_cast<int64_t>(imgs_descriptors.size()),
        static_cast<int64_t>(imgs_descriptors[0].size()),
        static_cast<int64_t>(imgs_descriptors[0][0].size())};

    Ort::Value imgs_descriptors_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        imgs_descriptors_values.data(),
        imgs_descriptors_values.size(),
        imgs_descriptors_shape.data(),
        imgs_descriptors_shape.size());

    std::array<Ort::Value, 2> input_tensors = {
        std::move(normalized_imgs_keypts_tensor),
        std::move(imgs_descriptors_tensor)};

    const char *input_names[] = {"keypoints", "descriptors"};
    const char *output_names[] = {"matches", "mscores"};

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names, input_tensors.data(), 2,
        output_names, 2);

    Ort::Value &matches_tensor = output_tensors[0];
    int16_t *matches_data = matches_tensor.GetTensorMutableData<int16_t>();
    auto matches_shape = matches_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const unsigned int match_num = matches_shape[0];

    matches.resize(match_num);
    for (int i = 0; i < match_num; ++i)
    {
        const unsigned int offset_i = i * 3;
        int idx_1 = static_cast<int>(matches_data[offset_i + 1]);
        int idx_2 = static_cast<int>(matches_data[offset_i + 2]);
        matches[i] = std::make_pair(idx_1, idx_2);
    }

    Ort::Value &scores_tensor = output_tensors[1];
    float *scores_data = scores_tensor.GetTensorMutableData<float>();
    scores.resize(match_num);
    std::copy(scores_data, scores_data + match_num, scores.begin());
}
