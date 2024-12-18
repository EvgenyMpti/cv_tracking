#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include "opencv2/face.hpp"

#include "person.h"
#include "../nn/yolo_onnx.h"

namespace CamSec
{
using Corners = std::vector<cv::Point2f>;
class Room;

class Camera
{
public:
    Camera(Room* room, const std::string& source, const Corners& real_corn, const Corners& image_corn)
        : room_(room), source_(source), real_corn_(real_corn), image_corn_(image_corn)
    {
        homography_ = cv::findHomography(image_corn_, real_corn_);
    }
    Camera(const Camera&) = delete;
    Camera& operator=(const Camera&) = delete;

    std::string GetSource() const
    {
        return source_;
    }
    cv::Mat GetHomography() const
    {
        return homography_;
    }
    Room* GetRoom() const
    {
        return room_;
    }
private:
    Room* room_;
    std::string source_;
    // Реальные координаны на карте
    Corners real_corn_;
    // Координаты углов параллелограмма на изображении
    Corners image_corn_;
    cv::Mat homography_;
    
};

class Room
{
public:
    explicit Room(const std::string& name, const Corners& coords, const Corners& door)
        : name_(name), coords_(coords), door_(door)
    {}
    Room(const Room&) = delete;
    Room& operator=(const Room&) = delete;

    void AddCamera(const std::string& source, const Corners& real_corn, const Corners& image_corn)
    {
        cameras_.try_emplace(cameras_.size()+1, this, source, real_corn, image_corn);
    }

    std::vector<const Camera*> GetCameras() const
    {
        std::vector<const Camera*> result(cameras_.size());

        std::transform(cameras_.begin(), cameras_.end(), result.begin(),
            [](const auto& cam) -> const Camera* {
                return &cam.second;
            }
        );
        return result;
    }
    std::string GetName() const
    {
        return name_;
    }
    size_t GetCount() const
    {
        return persons_.size();
    }
    obj::Person* Update(cv::Point2f head, cv::Point2f room_pos, const obj::Body& body, obj::Person* person)
    {
        int closest = 0;
        double last_norm = -1;
        for (auto& [id, p] : persons_)
        {
            if (p.GetLastPos() == cv::Point2f{})
                continue;
            auto norm = cv::norm(p.GetLastPos() - head);
            if (last_norm == -1 || norm < last_norm)
            {
                last_norm = norm;
                closest = id;
            }
        }
        bool need_insert = false;

        //need to create a new person (no closest persons)
        if (last_norm == -1 || last_norm > std::max(coords_[2].x, coords_[2].y) * 0.1)
        {
            need_insert = 1;
            closest = 0;
        }
        //we got closest person and not the same
        else if (person && person->GetId() != closest)
        {
            //copy closest data to person
            persons_.erase(closest);
            need_insert = 1;
        }
        bool is_exit = cv::pointPolygonTest(door_, room_pos, false) >= 0 || room_pos.x < coords_[0].x || room_pos.y < coords_[0].y;
        if (is_exit)
        {
            if (closest)
                persons_.erase(closest);
            return {};
        }
        int id = closest;
        if (need_insert && !is_exit)
        {
            if (!person)
            {
                persons_.try_emplace(--unknown_, unknown_, "Unknown " + std::to_string(abs(unknown_)), cv::Mat{});
                id = unknown_;
            }
            else {
                id = person->GetId();
                persons_.insert({ id, *person });
            }
        }
        auto& p = persons_.at(id);
        p.Update(head, body);
        return &p;
    }
	//add person
	//remove
	//update coord+status
	//Get
private:
    std::string name_;
    //координаты на общем плане клиники
    Corners coords_;
    Corners door_;
    std::unordered_map<size_t, Camera> cameras_;
    std::unordered_map<int, obj::Person> persons_;
    int unknown_{};
};

class Clinic 
{
public:
    Clinic();
    Clinic(const Clinic&) = delete;
    Clinic& operator=(const Clinic&) = delete;

    std::vector<const Camera*> GetCameras();
    cv::Mat HandleFrame(cv::Mat& frame, const Camera* camera);

private:
    YoLoOnnx pose_model_;
    YoLoOnnx emotion_model_;
    cv::Ptr<cv::FaceDetectorYN> facedet_model_;
    cv::Ptr<cv::FaceRecognizerSF> facerec_model_;

    std::mutex lock_;

	std::unordered_map<size_t, Room> rooms_;
    std::unordered_map<size_t, obj::Person> persons_;

    //temp to paint map
    cv::Mat clinic_map_;
    cv::Mat full_frame_{ 700, 900, CV_8UC3, cv::Scalar(255, 255, 255)};

    cv::Mat DetectFace(cv::Mat& img);
    std::vector<std::pair<std::string, cv::Mat>> BuildEmplFaces();
    std::vector<std::pair<std::string, cv::Mat>> ReadFacesFromTxt(const std::string& filename);
    std::vector<obj::Body> GetBodies(std::vector<YoloResults>& results);
    std::pair<std::string, float> Clinic::GetEmotions(cv::Mat roi);
    void saveFacesToTxt(const std::string& filename, const std::vector<std::pair<std::string, cv::Mat>>& data);
};

}

