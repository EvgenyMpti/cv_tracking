#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <unordered_map>
#include <string>
#include <vector>
#include <random>

namespace obj
{

class Body 
{
public:
	enum Names
	{
		Nose = 0,
		Eye_r = 1,
		Eye_l = 2,
		Ear_r = 3,
		Ear_l = 4,
		Shoulder_r = 5,
		Shoulder_l = 6,
		Elbow_r = 7,
		Elbow_l = 8,
		Wrist_r = 9,
		Wrist_l = 10,
		Hip_r = 11,
		Hip_l = 12,
		Knee_r = 13,
		Knee_l = 14,
		Ancle_r = 15,
		Ancle_l = 16
	};

	explicit Body(const std::array<cv::Point2f, 17>& kps, const cv::Rect_<float>& rect, float conf)
		: kps_(kps), body_rect_(rect), conf_(conf) {}
	explicit Body(const cv::Mat& face)
		:face_features_(face.clone())
	{ }

	cv::Rect_<float> GetFaceRect()
	{
		if (!face_rect_.empty())
			return face_rect_;
		if (kps_[Ear_l].x <= kps_[Nose].x && kps_[Nose].x <= kps_[Ear_r].x)
		{
			float w = (kps_[Ear_r].x - kps_[Ear_l].x) * 1.2f;
			float h = w * 1.2;
			face_rect_ = { abs(kps_[Nose].x - w/2), abs(kps_[Nose].y - h/2), w, h};
			return face_rect_;
		}
		return {};
	}

	cv::Rect_<float> GetBodyRect()
	{
		if (!body_rect_.empty())
			return body_rect_;
		const auto [x1, x2] = std::minmax_element(kps_.begin(), kps_.end(), [](const auto& a, const auto& b) { return a.x < b.x; });
		const auto [y1, y2] = std::minmax_element(kps_.begin(), kps_.end(), [](const auto& a, const auto& b) { return a.y < b.y; });
		body_rect_ = { x1->x, y1->y, abs(x2->x-x1->x), abs(y2->y-y1->y)};
		return body_rect_;
	}

	void SetConfidence(float conf)
	{
		conf_ = conf;
	}
	void SetKeyPoints(const std::array<cv::Point2f, 17>& kps)
	{
		kps_ = kps;
	}
	void SetRect(const cv::Rect_<float>& rect)
	{
		body_rect_ = rect;
	}
	float GetConfidence() const
	{
		return conf_;
	}
	std::vector<cv::Point2f> GetFaceDlibKP() const
	{
		return std::vector<cv::Point2f>{ kps_[Ear_r], kps_[Eye_r], kps_[Ear_l] , kps_[Eye_l] , kps_[Nose] };
	}
	std::array<cv::Point2f, 17> GetKP() const
	{
		return kps_;
	}
	cv::Mat GetFaceFeatures() const
	{
		return face_features_;
	}
private:
	std::array<cv::Point2f, 17> kps_;
	cv::Rect_<float> face_rect_, body_rect_;
	cv::Mat face_features_;
	std::vector<std::vector<cv::Point2f>> marks_;
	bool is_marks_ = false;
	float conf_ = 0;
};

class Person 
{
public:
	enum Type
	{
		Guest = 0,
		Patient = 1,
		Employe = 2,
	};
	Person(int id, const std::string& name, const cv::Mat& face)
		: id_(id), name_(name), body_(face)
	{ 
		std::random_device rd; // Obtain a random seed from the OS
		std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
		std::uniform_int_distribution<> distrib(30, 150); // Define the range
		color_ = cv::Scalar(distrib(gen), distrib(gen), distrib(gen));
	}

	cv::Mat GetFaceFeatures() const
	{
		return body_.GetFaceFeatures();
	}
	std::string GetName() const
	{
		return name_;
	}
	int GetId() const
	{
		return id_;
	}
	cv::Point2f GetLastPos() const
	{
		if (vector_.empty())
			return {};
		return vector_.back();
	}
	void Update(cv::Point2f head, const Body& body)
	{
		vector_.push_back(head);
	}
	cv::Scalar GetColor() const
	{
		return color_;
	}
private:
	int id_;
	std::string name_;
	Body body_;
	Type type_{};
	std::vector<cv::Point2f> vector_; //вектор движения головы выровненный по пространству

	cv::Scalar color_{};
};

}

/*
Transform y-n-y to y-n-m
		cv::Mat align_kp = cv::Mat(1, 15, CV_32FC1);


		float mouth_ratio_x1 = 0.1835f;
		float mouth_ratio_x2 = 0.84f;
		float mouth_ratio_y1 = 2.02937;
		float mouth_ratio_y2 = 2.01148;
		const auto& face_kp = body.GetFaceKP();

		//float dst[5][2] = { {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f}, {41.5493f, 92.3655f}, {70.7299f, 92.2041f} };

		float ml_x = (face_kp[obj::Body::Nose].x - face_kp[obj::Body::Eye_l].x) * mouth_ratio_x1 + face_kp[obj::Body::Eye_l].x;
		float mr_x = (face_kp[obj::Body::Eye_r].x - face_kp[obj::Body::Nose].x) * mouth_ratio_x2 + face_kp[obj::Body::Nose].x;

		float ml_y = (face_kp[obj::Body::Nose].y - face_kp[obj::Body::Eye_l].y) * mouth_ratio_y1 + face_kp[obj::Body::Eye_l].y;
		float mr_y = (face_kp[obj::Body::Nose].y - face_kp[obj::Body::Eye_r].y) * mouth_ratio_y2 + face_kp[obj::Body::Eye_r].y;

		std::vector<float> formatted_kp{
			face_rect->x, face_rect->y, face_rect->width, face_rect->height,
			face_kp[obj::Body::Eye_l].x, face_kp[obj::Body::Eye_l].y,
			face_kp[obj::Body::Eye_r].x, face_kp[obj::Body::Eye_r].y,
			face_kp[obj::Body::Nose].x, face_kp[obj::Body::Nose].y,
			ml_x, ml_y,
			mr_x, mr_y,
			body.GetConfidence()
		};
		for (auto i = 0; i < formatted_kp.size(); ++i)
		{
			formatted_kp[i] /= scale;
			align_kp.at<float>(0, i) = formatted_kp[i];
		}

		cv::circle(frame, cv::Point2f{ formatted_kp[4], formatted_kp[5] }, 3, cv::Scalar(255, 0, 0), cv::FILLED);
		cv::circle(frame, cv::Point2f{ formatted_kp[6], formatted_kp[7] }, 3, cv::Scalar(255, 0, 0), cv::FILLED);
		cv::circle(frame, cv::Point2f{ formatted_kp[8], formatted_kp[9] }, 3, cv::Scalar(255, 0, 0), cv::FILLED);
		cv::circle(frame, cv::Point2f{ formatted_kp[12], formatted_kp[13] }, 3, cv::Scalar(255, 0, 0), cv::FILLED);
		cv::circle(frame, cv::Point2f{ formatted_kp[10], formatted_kp[11] }, 3, cv::Scalar(255, 0, 0), cv::FILLED);
		imshow("frame", frame);
		cv::waitKey(0);

		//cv_detector->setInputSize(frame_orig.size());

		//cv::Mat faces1;
		//cv_detector->detect(frame_orig, faces1);

		//cv::Mat aligned_face;
		//facerec_model_->alignCrop(frame, align_kp, aligned_face);

*/