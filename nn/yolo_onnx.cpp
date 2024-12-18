#pragma once

#include "nn/yolo_onnx.h"

#include <iostream>
#include <ostream>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

#include "misc.h"

namespace fs = std::filesystem;


YoLoOnnx::YoLoOnnx(const std::string& modelPath, const std::string& provider, const std::string& ident)
{
    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, ident.c_str());
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();

    if (provider == "cuda") {
        std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
        auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
        if (cudaAvailable != availableProviders.end()) {
            OrtCUDAProviderOptions cudaOption;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
        }
    }

#ifdef _WIN32
    auto modelPathW = get_win_path(modelPath);  // For Windows (wstring)
    session_ = Ort::Session(env_, modelPathW.c_str(), sessionOptions);
#else
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);  // For Linux (string)
#endif
    //session = Ort::Session(env)
    // https://github.com/microsoft/onnxruntime/issues/14157
    //std::vector<const char*> inputNodeNames; //
    // ----------------
    // init input names
    inputNodeNames_;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings; // <-- newly added
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNodesNum = session_.GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session_.GetInputNameAllocated(i, allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNodeNames_.push_back(inputNodeNameAllocatedStrings.back().get());
    }
    // -----------------
    // init output names
    outputNodeNames_;
    auto outputNodesNum = session_.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings; // <-- newly added
    Ort::AllocatorWithDefaultOptions output_names_allocator;
    for (int i = 0; i < outputNodesNum; i++)
    {
        auto output_name = session_.GetOutputNameAllocated(i, output_names_allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNodeNames_.push_back(outputNodeNameAllocatedStrings.back().get());
    }
    // -------------------------
    // initialize model metadata
    model_metadata_ = session_.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions metadata_allocator;

    std::vector<Ort::AllocatedStringPtr> metadataAllocatedKeys = model_metadata_.GetCustomMetadataMapKeysAllocated(metadata_allocator);
    std::vector<std::string> metadata_keys;
    metadata_keys.reserve(metadataAllocatedKeys.size());

    for (const Ort::AllocatedStringPtr& allocatedString : metadataAllocatedKeys) {
        metadata_keys.emplace_back(allocatedString.get());
    }

    // -------------------------
    // initialize metadata as the dict
    // even though we know exactly what metadata we intend to use
    // base onnx class should not have any ultralytics yolo-specific attributes like stride, task etc, so keep it clean as much as possible
    for (const std::string& key : metadata_keys) {
        Ort::AllocatedStringPtr metadata_value = model_metadata_.LookupCustomMetadataMapAllocated(key.c_str(), metadata_allocator);
        if (metadata_value != nullptr) {
            auto raw_metadata_value = metadata_value.get();
            metadata_[key] = std::string(raw_metadata_value);
        }
    }

    // initialize cstr
    for (const std::string& name : outputNodeNames_) {
        outputNamesCStr_.push_back(name.c_str());
    }

    for (const std::string& name : inputNodeNames_)
    {
        inputNamesCStr_.push_back(name.c_str());
    }
    // then try to get additional info from metadata like imgsz, stride etc;
    //  ideally you should get all of them but you'll raise error if smth is not in metadata (or not under the appropriate keys)

    // post init imgsz
    if (imgsz_.empty()) {
        auto imgsz_iterator = metadata_.find("imgsz");
        if (imgsz_iterator != metadata_.end()) {
            imgsz_ = convertStringVectorToInts(parseVectorString(imgsz_iterator->second));
        }
    }

    // post init stride
    if (stride_ == -1) {
        auto stride_item = metadata_.find("stride");
        if (stride_item != metadata_.end()) {
            stride_ = std::stoi(stride_item->second);
        }
    }

    // post init names
    if (names_.empty()) {
    auto names_item = metadata_.find("names");
    if (names_item != metadata_.end()) {
            names_ = parseNames(names_item->second);
        }
    }

    // post init number of classes - you can do that only and only if names_ is not empty and nc was not initialized previously
    if (nc_ == -1 && !names_.empty()) {
        nc_ = names_.size();
    }
    else {
        std::cerr << "Warning: Cannot get nc value from metadata (probably names wasn't set)" << std::endl;
    }

    if (!imgsz_.empty() && inputTensorShape_.empty())
    {
        inputTensorShape_ = { 1, ch_, getHeight(), getWidth() };
    }

    if (!imgsz_.empty())
    {
        // Initialize cvSize_ using getHeight() and getWidth()
        //cvSize_ = cv::MatSize()
        cvSize_ = cv::Size(getWidth(), getHeight());
        //cvMatSize_ = cv::MatSize(cvSize_.width, cvSize_.height);
    }

}


const int& YoLoOnnx::getHeight()
{
    return imgsz_[0];
}

const int& YoLoOnnx::getWidth()
{
    return imgsz_[1];
}

const int& YoLoOnnx::getStride() {
    return stride_;
}

const int& YoLoOnnx::getCh() {
    return ch_;
}

const std::unordered_map<int, std::string>& YoLoOnnx::getNames() {
    return names_;
}


const cv::Size& YoLoOnnx::getCvSize()
{
    return cvSize_;
}

const std::vector<int64_t>& YoLoOnnx::getInputTensorShape()
{
    return inputTensorShape_;
}

std::vector<YoloResults> YoLoOnnx::Predict(cv::Mat& image_src, Type typ, float& conf, float& iou, int conversionCode, bool verbose) {
    double preprocess_time = 0.0;
    double inference_time = 0.0;
    double postprocess_time = 0.0;
    Timer preprocess_timer = Timer(preprocess_time, verbose);
    // 1. preprocess
    float* blob = nullptr;
    //double* blob = nullptr;
    std::vector<Ort::Value> inputTensors;
    cv::Mat image = image_src.clone();
    if (conversionCode >= 0) {
        cv::cvtColor(image, image, conversionCode);
    }
    std::vector<int64_t> inputTensorShape;
    // TODO: for classify task preprocessed image will be different (!):
    cv::Mat preprocessed_img;
    cv::Size new_shape = cv::Size(getWidth(), getHeight());
    const bool& scaleFill = false;  // false
    const bool& auto_ = false; // true
    letterbox(image, preprocessed_img, new_shape, cv::Scalar(), auto_, scaleFill, true, getStride());
    fill_blob(preprocessed_img, blob, inputTensorShape);
    int64_t inputTensorSize = vector_product(inputTensorShape);
    std::vector<float> inputTensorValues(blob, blob + inputTensorSize);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize,
        inputTensorShape.data(), inputTensorShape.size()
    ));
    preprocess_timer.Stop();
    Timer inference_timer = Timer(inference_time, verbose);
    // 2. inference
    std::vector<Ort::Value> outputTensors = forward(inputTensors);
    inference_timer.Stop();
    Timer postprocess_timer = Timer(postprocess_time, verbose);
    // create container for the results
    std::vector<YoloResults> results;
    // 3. postprocess based on task:
    std::unordered_map<int, std::string> names = this->getNames();
    // 4. cleanup blob since it was created using the "new" keyword during the `fill_blob` func call
    delete[] blob;

    int class_names_num = names.size();

    std::vector<int64_t> outputTensor0Shape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* all_data0 = outputTensors[0].GetTensorMutableData<float>();
    cv::Mat output0 = cv::Mat(cv::Size((int)outputTensor0Shape[2], (int)outputTensor0Shape[1]), CV_32F, all_data0).t();  // [bs, features, preds_num]=>[bs, preds_num, features]
    if (typ == KP)
        ProcessKpts(output0, image.size(), results, class_names_num, conf, iou);
    else if (typ == Seg)
        ProcessDetects(output0, image.size(), results, class_names_num, conf, iou);
    else
        return {};

    postprocess_timer.Stop();
    if (verbose) {
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "image: " << preprocessed_img.rows << "x" << preprocessed_img.cols << " " << results.size() << " objs, ";
        std::cout << (preprocess_time + inference_time + postprocess_time) * 1000.0 << "ms" << std::endl;
        std::cout << "Speed: " << (preprocess_time * 1000.0) << "ms preprocess, ";
        std::cout << (inference_time * 1000.0) << "ms inference, ";
        std::cout << (postprocess_time * 1000.0) << "ms postprocess per image ";
        std::cout << "at shape (1, " << image.channels() << ", " << preprocessed_img.rows << ", " << preprocessed_img.cols << ")" << std::endl;
    }

    return results;
}

void YoLoOnnx::ProcessKpts(cv::Mat& output0, cv::Size& image_info, std::vector<YoloResults>& output,
                                          int& class_names_num, float& conf_threshold, float& iou_threshold)
{
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> rest;
    std::tie(boxes, confidences, class_ids, rest) = non_max_suppression(output0, class_names_num, output0.cols, conf_threshold, iou_threshold);
    cv::Size img1_shape = getCvSize();
    auto bound_bbox = cv::Rect_ <float> (0, 0, image_info.width, image_info.height);
    for (int i = 0; i < boxes.size(); i++) {
        //             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        //            pred_kpts = pred[:, 6:].view(len(pred), *self.model.kpt_shape) if len(pred) else pred[:, 6:]
        //            pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, shape)
        //            path = self.batch[0]
        //            img_path = path[i] if isinstance(path, list) else path
        //            results.append(
        //                Results(orig_img=orig_img,
        //                        path=img_path,
        //                        names=self.model.names,
        //                        boxes=pred[:, :6],
        //                        keypoints=pred_kpts))
        cv::Rect_<float> bbox = boxes[i];
        auto scaled_bbox = scale_boxes(img1_shape, bbox, image_info);
        scaled_bbox = scaled_bbox & bound_bbox;
//        cv::Mat kpt = cv::Mat(rest[i]).t();
//        scale_coords(img1_shape, kpt, image_info.raw_size);
        // TODO: overload scale_coords so that will accept cv::Mat of shape [17, 3]
        //      so that it will be more similar to what we have in python
        std::vector<float> kpt = scale_coords(img1_shape, rest[i], image_info);
        YoloResults tmp_res = { class_ids[i], confidences[i], scaled_bbox, kpt, names_[class_ids[i]] };
        output.push_back(tmp_res);
    }
}

void YoLoOnnx::ProcessDetects(cv::Mat& output0, cv::Size image_info, std::vector<YoloResults>& output,
    int& class_names_num, float& conf_threshold, float& iou_threshold)
{
    output.clear();
    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> masks;
    // 4 - your default number of rect parameters {x, y, w, h}
    int data_width = class_names_num + 4;
    int rows = output0.rows;
    float* pdata = (float*)output0.data;

    for (int r = 0; r < rows; ++r)
    {
        cv::Mat scores(1, class_names_num, CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_conf;
        minMaxLoc(scores, nullptr, &max_conf, nullptr, &class_id);

        if (max_conf > conf_threshold)
        {
            masks.emplace_back(pdata + 4 + class_names_num, pdata + data_width);
            class_ids.push_back(class_id.x);
            confidences.push_back((float)max_conf);

            float out_w = pdata[2];
            float out_h = pdata[3];
            float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
            float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);

            cv::Rect_ <float> bbox = cv::Rect_ <float>(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
            cv::Rect_<float> scaled_bbox = scale_boxes(getCvSize(), bbox, image_info);

            boxes.push_back(scaled_bbox);
        }
        pdata += data_width; // next pred
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result); // , nms_eta, top_k);
    for (int idx : nms_result)
    {
        boxes[idx] = boxes[idx] & cv::Rect(0, 0, image_info.width, image_info.height);
        YoloResults result = { class_ids[idx] ,confidences[idx] ,boxes[idx] };
        result.class_name = names_[class_ids[idx]];
        output.push_back(result);
    }
}

void YoLoOnnx::fill_blob(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape) {

	cv::Mat floatImage;
    if (inputTensorShape.empty())
    {
        inputTensorShape = getInputTensorShape();
    }
    int inputChannelsNum = inputTensorShape[1];
    int rtype = CV_32FC3;
    image.convertTo(floatImage, rtype, 1.0f / 255.0);
    blob = new float[floatImage.cols * floatImage.rows * floatImage.channels()];
    cv::Size floatImageSize{ floatImage.cols, floatImage.rows };

    // hwc -> chw
    std::vector<cv::Mat> chw(floatImage.channels());
    for (int i = 0; i < floatImage.channels(); ++i)
    {
        chw[i] = cv::Mat(floatImageSize, CV_32FC1, blob + i * floatImageSize.width * floatImageSize.height);
    }
    cv::split(floatImage, chw);
}

const Ort::ModelMetadata& YoLoOnnx::getModelMetadata()
{
    return model_metadata_;
}


std::vector<Ort::Value> YoLoOnnx::forward(std::vector<Ort::Value>& inputTensors)
{
    // todo: make runOptions parameter here

    return session_.Run(Ort::RunOptions{ nullptr },
        inputNamesCStr_.data(),
        inputTensors.data(),
        inputNamesCStr_.size(),
        outputNamesCStr_.data(),
        outputNamesCStr_.size());
}

