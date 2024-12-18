# Проектная работа МФТИ - cv_tracking
Анализ видео с камер наблюдения в клинике с дальнейшей интеграцией с учетной системой и сбора статистики

- Подробное описание проекта можно найти в [ноутбуке разработки](https://colab.research.google.com/drive/1C6qhr3qGnRfvE8HPyjo9KgK0RXPHhGjX?usp=sharing)
- [Оригинальное видео работы программы](https://github.com/EvgenyMpti/cv_tracking/raw/refs/heads/main/result/output.mp4) 

[![Пример работы программы](/result/output.gif)](https://www.youtube.com/watch?v=TLlQZG0YQ5g)

[YouTube](https://www.youtube.com/watch?v=TLlQZG0YQ5g)


# Требования:
- Unix/Windows
- С++ >=14
- OpenCV >= 3.0
- ONNXRUNTIME
- Cuda опционально

Тестирование проводилось под Microsoft Visual Studio Community 2022 (64-bit) + ONNXRUNTIME as Nuget, 
на других платформах также должно работать, т.к. все компонетны являются кросс платформенными

# Для сборки можно воспользоваться CMake
Установить окружение в CMaLists.txt (проверить пути к библиотекам)
- set (CMAKE_PREFIX_PATH "D:/code/opencv/build/install")
- set(ONNXRUNTIME_VERSION 1.20.1)
- set(ONNXRUNTIME_ROOT "D:/code/camsec/build/packages/Microsoft.ML.OnnxRuntime.${ONNXRUNTIME_VERSION}/")

- cmake -B build .
- cmake --build build

# Настройка
- Необходимо предварительно настроить model/config.h -> указать все требуемые пути
- Разместить папку с фотографиями сотрудников для предварительного составления вектора особенностей лица
- необходимо настроить источник видео, прописать формат клиники, ее кабинетов и указать точки соответствия углов изображений с камеры, реальному положению кабинета

# Ссылки
* [YOLOv11 by Ultralytics](https://github.com/ultralytics/ultralytics)
* [ONNX](https://onnx.ai)
* [OpenCV](https://opencv.org)
