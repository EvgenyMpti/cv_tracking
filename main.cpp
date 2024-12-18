#include <opencv2/opencv.hpp>

#include <vector>
#include <thread>

#include "model/clinic.h"

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
