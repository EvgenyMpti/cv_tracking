#include <opencv2/opencv.hpp>

#include <vector>
#include <thread>

#include "model/clinic.h"

/*
cv::Mat alignFace(const cv::Mat& src, obj::Body& body)
{
    auto face_kp = body.GetFaceKP();
    const auto face_rect = *body.GetFace();
    //cv::Mat face_mat = src(face_rect);
    //for (auto& kp : face_kp)
    //    kp -= face_rect.tl();

    // ������� ���������� (��������������)
    std::vector<cv::Point2f> dst_pts = {
        face_kp[0],  // ��� (����� ���� �� ���������, 1/3 ������)
        face_kp[1],  // ����� ����
        face_kp[2],   // ������ ����
        face_kp[3],  // ����� ��� (�������� ������, �������� ������)
        cv::Point2f(face_rect.width * 3 / 4, face_rect.height / 2)  // ������ ���
    };
    
    // ���������� ������� ��������� � ����������
    float avg_eye_y = (face_kp[1].y + face_kp[2].y) / 2;
    float avg_ear_y = (face_kp[3].y + face_kp[4].y) / 2;

    // ������������� ������� ��������� �� ������ �������
    //dst_pts[1].y = dst_pts[2].y = avg_eye_y;
    //dst_pts[3].y = dst_pts[4].y = avg_ear_y;

    // ���������� ������� ����������
    //cv::Mat H = findHomography(face_kp, dst_pts);

    // ���������� ��������������
    //cv::Mat warped_img;
    //warpPerspective(face_mat, warped_img, H, face_rect.size());
    //cv::circle(warped_img, dst_pts[4], 2, cv::Scalar(0, 255, 0), -1);
    //////////////////////////////

    //std::vector<cv::Point2d> image_points = face_kp;

    return {};// warped_img;
}

*/

void processCamera(CamSec::Clinic& clinic, const CamSec::Camera* camera) {

    cv::VideoCapture cap(camera->GetSource());
    if (!cap.isOpened()) {
        std::cerr << "Error opening video input" << std::endl;
        return;
    }

    std::string filename = "output1.mp4"; // ��� ��������� ����������
    int fps = 15;                         // ����� � ������� (FPS)
    cv::Size frameSize(900, 700);        // ������ ����� (������, ������)
    bool isColor = true;                // ������� ����� (true) ��� �����-����� (false)

    cv::VideoWriter video(filename, cv::VideoWriter::fourcc('M', 'P', '4', 'V'), fps, frameSize, isColor);

    if (!video.isOpened()) {
        std::cerr << "Could not open the output video for write\n";
        return;
    }

    int frame_count = 0;

    while (cap.isOpened()) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        // ������������ ������ 5-� ����
        //if (frame_count % 5 == 0) {
            const auto& result = clinic.HandleFrame(frame, camera);
            imshow("Video", result);
            video.write(result);
        //}
        frame_count++;

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    video.release();
    cap.release();
}

int main() 
{
    try {

        CamSec::Clinic clinic;
        const auto& cameras = clinic.GetCameras();
        std::vector<std::thread> workers;
        workers.reserve(cameras.size());
        for (const auto& camera : cameras)
        {
            workers.emplace_back([&clinic, camera]() {
                processCamera(clinic, camera);
                }
            );
        }

        // ������� ���������� ���� �������
        for (auto& thread : workers) {
            thread.join();
        }
    }
    catch (const std::exception& ex) {
        std::cerr << "server exited, exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    cv::destroyAllWindows();

    return 0;
}
