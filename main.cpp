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

    // Целевые координаты (приблизительно)
    std::vector<cv::Point2f> dst_pts = {
        face_kp[0],  // Нос (центр лица по вертикали, 1/3 высоты)
        face_kp[1],  // Левый глаз
        face_kp[2],   // Правый глаз
        face_kp[3],  // Левое ухо (четверть ширины, середина высоты)
        cv::Point2f(face_rect.width * 3 / 4, face_rect.height / 2)  // Правое ухо
    };
    
    // Вычисление средних координат и расстояний
    float avg_eye_y = (face_kp[1].y + face_kp[2].y) / 2;
    float avg_ear_y = (face_kp[3].y + face_kp[4].y) / 2;

    // Корректировка целевых координат на основе средних
    //dst_pts[1].y = dst_pts[2].y = avg_eye_y;
    //dst_pts[3].y = dst_pts[4].y = avg_ear_y;

    // Вычисление матрицы гомографии
    //cv::Mat H = findHomography(face_kp, dst_pts);

    // Применение преобразования
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

    std::string filename = "output1.mp4"; // Имя выходного видеофайла
    int fps = 15;                         // Кадры в секунду (FPS)
    cv::Size frameSize(900, 700);        // Размер кадра (ширина, высота)
    bool isColor = true;                // Цветное видео (true) или черно-белое (false)

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

        // Обрабатываем каждый 5-й кадр
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

        // Ожидаем завершения всех потоков
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
