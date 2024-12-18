#pragma once
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <opencv2/core/mat.hpp>
#include <onnxruntime_cxx_api.h>

struct YoloResults {
    int class_idx{};
    float conf{};
    cv::Rect_<float> bbox;
    std::vector<float> keypoints{};
    std::string class_name{};
};

class YoLoOnnx {
public:
    enum Type
    {
        KP = 0,
        Seg = 1
    };
    // constructors
    YoLoOnnx(const std::string& modelPath, const std::string& provider, const std::string& ident);

    // getters
    const int& getStride();
    const int& getCh();
    const std::unordered_map<int, std::string>& getNames();
    const std::vector<int64_t>& getInputTensorShape();
    const int& getWidth();
    const int& getHeight();
    const cv::Size& getCvSize();

    std::vector<YoloResults> Predict
        (cv::Mat& image, Type, float& conf, float& iou, int conversionCode = -1, bool verbose = true);

    void fill_blob(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);

    void ProcessKpts(cv::Mat& output0, cv::Size& image_info, std::vector<YoloResults>& output,
        int& class_names_num, float& conf_threshold, float& iou_threshold);
    void ProcessDetects(cv::Mat& output0, cv::Size image_info, std::vector<YoloResults>& output,
        int& class_names_num, float& conf_threshold, float& iou_threshold);
   
protected:
    Ort::Env env_{ nullptr };
    Ort::Session session_{ nullptr };

    std::vector<std::string> inputNodeNames_;
    std::vector<std::string> outputNodeNames_;
    Ort::ModelMetadata model_metadata_{ nullptr };
    std::unordered_map<std::string, std::string> metadata_;
    std::vector<const char*> outputNamesCStr_;
    std::vector<const char*> inputNamesCStr_;
    ///////////////////

    std::vector<int> imgsz_;
    int stride_ = -1;
    int nc_ = -1;
    int ch_ = 3;
    std::unordered_map<int, std::string> names_;
    std::vector<int64_t> inputTensorShape_;
    cv::Size cvSize_;

    const Ort::ModelMetadata& getModelMetadata();
    std::vector<Ort::Value> forward(std::vector<Ort::Value>& inputTensors);
};
