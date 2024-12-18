#pragma once

#include <string>

namespace cfg
{
    const std::string& PATH = "Z:/work/dentimus/cpp/camsec/";

    const std::string& MODEL_PATH = PATH + "camsec/build/models/";
    const std::string& YOLO_POSE = MODEL_PATH + "yolo11n-pose.onnx";
    const std::string& YOLO_EMOTION = MODEL_PATH + "weights.onnx";
    const std::string& FRec_MODEL = MODEL_PATH + "face_recognition_sface_2021dec.onnx";
    const std::string& FDet_MODEL = MODEL_PATH + "face_detection_yunet_2023mar.onnx";
    const std::string& EMPLFACE_DIR = PATH + "faces/empl_big/";
    const std::string& EMPLFACE_FILE = EMPLFACE_DIR + "empls.txt";

    const std::string& ONNX_PROVIDER = "cpu"; 

    float CONF_THRESHOLD = 0.50f;
    float IOU_THRESHOLD = 0.4f;  //  0.70f;
    int CONVERSION_CODE = cv::COLOR_BGR2RGB;

    static const float SCORE_THRESHOLD = 0.6;
    static const float NMS_THRESHOLD = 0.3;
    static const int TOPK = 5000;

    double COSINE_SIMILAR_THRESH = 0.363;
    double L2NORM_SIMILAR_THRESH = 1.128;
}
