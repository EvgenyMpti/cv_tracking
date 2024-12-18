#pragma once
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <opencv2/core/types.hpp>

cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape, 
    std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true);
void clip_boxes(cv::Rect& box, const cv::Size& shape);
void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape);
void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape);
void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape);

std::vector<float> scale_coords(const cv::Size& img1_shape, std::vector<float>& coords, const cv::Size& img0_shape);

std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
non_max_suppression(const cv::Mat& output0, int class_names_num, int total_features_num, double conf_threshold, float iou_threshold);

void letterbox(const cv::Mat& image,
    cv::Mat& outImage,
    const cv::Size& newShape = cv::Size(640, 640),
    cv::Scalar_<double> color = cv::Scalar(),
    bool auto_ = true,
    bool scaleFill = false,
    bool scaleUp = true,
    int stride = 32
);

class Timer {
public:
    Timer(double& accumulator, bool isEnabled = true);
    void Stop();

private:
    double& accumulator;
    bool isEnabled;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

std::wstring get_win_path(const std::string& path);
std::vector<std::string> parseVectorString(const std::string& input);
std::vector<int> convertStringVectorToInts(const std::vector<std::string>& input);
std::unordered_map<int, std::string> parseNames(const std::string& input);
int64_t vector_product(const std::vector<int64_t>& vec);