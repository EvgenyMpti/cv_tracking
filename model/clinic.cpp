#include <numeric>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include "clinic.h"
#include "config.h"

namespace CamSec
{

Clinic::Clinic()
    : pose_model_(cfg::YOLO_POSE, cfg::ONNX_PROVIDER, "YoloPose")
    , emotion_model_(cfg::YOLO_EMOTION, cfg::ONNX_PROVIDER, "YoloEmotions")
{
	facedet_model_ = cv::FaceDetectorYN::create(cfg::FDet_MODEL, "", cv::Size(320, 320), cfg::SCORE_THRESHOLD, cfg::NMS_THRESHOLD, cfg::TOPK);
    facerec_model_ = cv::FaceRecognizerSF::create(cfg::FRec_MODEL, "");
    
    std::vector<std::pair<std::string, cv::Mat>> empl_faces;
    empl_faces = ReadFacesFromTxt(cfg::EMPLFACE_FILE);
    
    if (empl_faces.empty()) {
        empl_faces = BuildEmplFaces();
        saveFacesToTxt(cfg::EMPLFACE_FILE, empl_faces);
    }
    auto el_cnt = persons_.size();
    for (const auto& el : empl_faces)
        persons_.try_emplace(++el_cnt, el_cnt, el.first, el.second);

	//plan + rooms
    Corners r1_corners{
        cv::Point2f(0, 0),
            cv::Point2f(150, 0),
            cv::Point2f(150, 200),
            cv::Point2f(0, 200)
    };
    Corners r2_corners{
        cv::Point2f(0, 200),
            cv::Point2f(150, 200),
            cv::Point2f(150, 400),
            cv::Point2f(0, 400)
    };
    Corners r2_door{
        cv::Point2f(0, 200),
        cv::Point2f(50, 200),
        cv::Point2f(50, 230),
        cv::Point2f(0, 230)
    };

    auto r1 = rooms_.try_emplace(rooms_.size() + 1, "Room1", r1_corners, r2_corners);
    auto r2 = rooms_.try_emplace(rooms_.size() + 1, "Room2", r2_corners, r2_door);
    clinic_map_ = cv::Mat(400, 150, CV_8UC3, cv::Scalar(255, 255, 255)); // Белый фон на всю клинику
    cv::rectangle(clinic_map_, r1_corners[0], r1_corners[2], cv::Scalar(0, 0, 0), 2, cv::LINE_8);
    cv::rectangle(clinic_map_, r2_corners[0], r2_corners[2], cv::Scalar(0, 0, 0), 2, cv::LINE_8);
    cv::rectangle(clinic_map_, r2_door[0], r2_door[2], cv::Scalar(0, 0, 255), 2, cv::LINE_8);
    

    const std::string cam1_name{ "Z:/work/dentimus/cpp/camsec/video/4.mp4" };
    const Corners cam1_real {
        cv::Point2f(0, 200),
        cv::Point2f(150, 200),
        cv::Point2f(150, 400),
        cv::Point2f(0, 400)
    };
    const Corners cam1_image {
        cv::Point2f(1422, 1063),
        cv::Point2f(2579, 1283),
        cv::Point2f(1494, 2173),
        cv::Point2f(362, 1453)
    };

    if (r2.second)
        r2.first->second.AddCamera(cam1_name, cam1_real, cam1_image);
    
}

std::vector<const Camera*> Clinic::GetCameras()
{
    std::vector<const Camera*> result;
    for (const auto& [id, room] : rooms_)
    {
        const auto& room_cams = room.GetCameras();
        if (!room_cams.empty())
            result.insert(result.end(), room_cams.begin(), room_cams.end());
    }
    return result;
}

//нахождение признаков лица на изображении
cv::Mat Clinic::DetectFace(cv::Mat& img)
{
    if (img.cols > 1000 || img.rows > 1000)
    {
        double scale = std::min(800.0 / img.cols, 600.0 / img.rows);
        cv::Size newSize(scale * img.cols, scale * img.rows);
        resize(img, img, newSize);
    }

    facedet_model_->setInputSize(img.size());

    cv::Mat faces;
    facedet_model_->detect(img, faces);
    if (faces.rows < 1)
        return {};

    cv::Mat aligned_face;
    facerec_model_->alignCrop(img, faces.row(0), aligned_face);

    // Run feature extraction with given aligned_face
    cv::Mat feature;
    facerec_model_->feature(aligned_face, feature);
    return feature;
}

//строим вектор признаков лиц сотрудников
std::vector<std::pair<std::string, cv::Mat>> Clinic::BuildEmplFaces()
{
    std::vector<std::pair<std::string, cv::Mat>> features;
    for (const auto& entry : std::filesystem::directory_iterator(cfg::EMPLFACE_DIR))
    {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") 
            {
                cv::Mat img = cv::imread(entry.path().string());
                if (img.empty())
                {
                    std::cerr << "Cannot read image: " << entry << std::endl;
                    continue;
                }
                int prev_size = img.total();
                int current_size;

                do {
                    current_size = img.total();
                    cv::waitKey(100);
                } while (current_size != prev_size);

                const auto& feature = DetectFace(img);
                if (feature.empty())
                {
                    std::cerr << "No features: " << entry << std::endl;
                    continue;
                }
                features.push_back({ entry.path().stem().string(), feature.clone() });
            }
        }
    }
    return features;
}

std::vector<std::pair<std::string, cv::Mat>> Clinic::ReadFacesFromTxt(const std::string& filename) {
    std::vector<std::pair<std::string, cv::Mat>> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "ReadFaces file open err: " << filename << std::endl;
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string name;
        int rows, cols;
        iss >> name >> rows >> cols;

        cv::Mat mat(rows, cols, CV_32FC1);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                float value;
                iss >> value;
                mat.at<float>(i, j) = value;
            }
        }

        data.emplace_back(name, mat);
    }

    file.close();
    return data;
}

std::vector<obj::Body> Clinic::GetBodies(std::vector<YoloResults>& results)
{
    std::vector<obj::Body> bodies;
    for (const auto& res : results) {
        if (res.keypoints.size() == 51) {

            const auto& kps = res.keypoints;

            
            std::array<cv::Point2f, 17> kps_arr;
            for (int i = 0; i < 17; i++) {
                int idx = i * 3;
                kps_arr[i] = { kps[idx], kps[idx + 1] };
            }
            obj::Body body(kps_arr, res.bbox, res.conf);
            bodies.push_back(body);
        }
    }
    return bodies;
}

std::pair<std::string, float> Clinic::GetEmotions(cv::Mat roi)
{
    cv::Mat grey;
    cv::cvtColor(roi, grey, cv::COLOR_BGR2GRAY);

    const auto& emotions = emotion_model_.Predict(grey, YoLoOnnx::Seg, cfg::CONF_THRESHOLD, cfg::IOU_THRESHOLD, cfg::CONVERSION_CODE);

    std::string emotion_name;
    float emotion_conf = 0;
    for (const auto& em : emotions)
        if (em.conf > emotion_conf)
        {
            emotion_name = em.class_name;
            emotion_conf = em.conf;
        }
    return { emotion_name, emotion_conf };
}

cv::Mat Clinic::HandleFrame(cv::Mat& frame, const Camera* camera)
{

    // Уменьшаем размер кадра с сохранением пропорций
    double scale = std::min(800.0 / frame.cols, 600.0 / frame.rows);
    cv::Size newSize(scale * frame.cols, scale * frame.rows);
    cv::Mat small;
    resize(frame, small, newSize);

    std::vector<YoloResults> objs = pose_model_.Predict(small, YoLoOnnx::KP, cfg::CONF_THRESHOLD, cfg::IOU_THRESHOLD, cfg::CONVERSION_CODE);
    clinic_map_.convertTo(clinic_map_, CV_32F);
    clinic_map_ *= 1.05;
    clinic_map_.convertTo(clinic_map_, CV_8U);
    cv::Mat clinic_img(clinic_map_.clone());
    cv::Mat faces_mat = cv::Mat{ 250, 900, CV_8UC3, cv::Scalar(255, 255, 255) };
    size_t face_offset = 0;
    for (auto& body : GetBodies(objs))
    {
        //get face
        obj::Person* person{};
        auto face_rect = body.GetFaceRect();
        if (!face_rect.empty())
        {
            auto face_orig = face_rect;
            face_orig.x /= scale;
            face_orig.y /= scale;
            face_orig.width /= scale;
            face_orig.height /= scale;

            cv::Mat roi = frame(face_orig);
            //
            

            cv::Mat face_feature = DetectFace(roi);
            if (!face_feature.empty())
            {
                double max_cos = -1000;
                for (auto& [id, el] : persons_)
                {
                    const auto& el_feature = el.GetFaceFeatures();
                    if (el_feature.empty())
                        continue;
                    double cos_score = facerec_model_->match(el_feature, face_feature, cv::FaceRecognizerSF::DisType::FR_COSINE);
                    //double L2_score = facerec_model_->match(el_feature, face_feature, cv::FaceRecognizerSF::DisType::FR_NORM_L2);
                    if (cos_score > max_cos)
                    {
                        max_cos = cos_score;
                        person = &el;
                    }
                }
                if (max_cos >= 0.3)
                {
                    cv::Scalar text_color;
                    if (max_cos >= 0.6)
                        text_color = cv::Scalar(0, 255.0, 0);
                    else if (max_cos >= 0.5)
                        text_color = cv::Scalar(0, 255.0, 255.0);
                    else
                        text_color = cv::Scalar(0, 0, 255.0);
                    putText(small, person->GetName(), cv::Point(face_rect.x - 1.5, face_rect.y - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
                }
                if (max_cos < 0.5)
                    person = nullptr;

                const auto& [emotion_name, emotion_conf] = GetEmotions(roi);
                if (emotion_conf)
                {
                    cv::Scalar text_color;
                    if (emotion_conf >= 0.6)
                        text_color = cv::Scalar(0, 255.0, 0); //green
                    else if (emotion_conf >= 0.5)
                        text_color = cv::Scalar(0, 255.0, 255.0); // yellow
                    else
                        text_color = cv::Scalar(0, 0, 255.0); //red

                    putText(roi, emotion_name, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2);
                }
            }
            cv::Mat faceROI(faces_mat, cv::Rect(face_offset, 0, roi.cols, roi.rows));
            roi.clone().copyTo(faceROI);
            face_offset += face_orig.height;
        }

        const auto& kps = body.GetKP();
        cv::Point2f ancle = (kps[obj::Body::Ancle_r] + kps[obj::Body::Ancle_l]) / 2;
        cv::Point2f head = std::accumulate(kps.begin(), kps.begin() + 5, cv::Point2f{}) / 5;

        ancle /= scale;

        cv::Mat ancle_mat = (cv::Mat_<double>(3, 1) << ancle.x, ancle.y, 1);
        cv::Mat room_mat = camera->GetHomography() * ancle_mat;

        double x = room_mat.at<double>(0, 0) / room_mat.at<double>(2, 0);
        double y = room_mat.at<double>(1, 0) / room_mat.at<double>(2, 0);

        cv::Point2f room_point(x, y);

        const auto& updated = camera->GetRoom()->Update(head, room_point, body, person);
        if (!updated)
            continue;

        cv::circle(clinic_map_, room_point, 3, updated->GetColor(), cv::FILLED);
        putText(clinic_img, updated->GetName(), cv::Point(room_point.x - 1.5, room_point.y - 12.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

    }
    //imshow("Clinic", clinic_img);
    //cv::waitKey(0);

    cv::Mat samllROI(full_frame_, cv::Rect(0, 0, small.cols, small.rows));
    cv::Mat naviROI(full_frame_, cv::Rect(700, 0, clinic_img.cols, clinic_img.rows));
    cv::Mat facesROI(full_frame_, cv::Rect(0, 450, faces_mat.cols, faces_mat.rows));
    small.copyTo(samllROI);
    clinic_img.copyTo(naviROI);
    faces_mat.copyTo(facesROI);
    putText(full_frame_, std::to_string(camera->GetRoom()->GetCount()), cv::Point(755, 120), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(0, 0, 0), 2);
    return full_frame_;//small;
    
}

void Clinic::saveFacesToTxt(const std::string& filename, const std::vector<std::pair<std::string, cv::Mat>>& data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Ошибка открытия файла: " << filename << std::endl;
        return;
    }

    for (const auto& pair : data) {
        file << pair.first << " " << pair.second.rows << " " << pair.second.cols;
        for (int i = 0; i < pair.second.rows; ++i) {
            for (int j = 0; j < pair.second.cols; ++j) {
                file << " " << pair.second.at<float>(i, j);
            }
        }
        file << std::endl;
    }
    file.close();
}

} //namespace CamSec

