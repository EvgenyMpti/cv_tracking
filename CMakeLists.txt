cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME cv_tracking)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)


# -------------- Support C++17 for using filesystem  ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)


# -------------- OpenCV  ------------------#
set(USE_CUDA OFF)

#set (CMAKE_PREFIX_PATH "D:/code/opencv/build/install")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# -------------- Compile CUDA for FP16 inference if needed  ------------------#
#option(USE_CUDA "Enable CUDA support" ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

# -------------- ONNXRUNTIME  ------------------#

# Set ONNXRUNTIME_VERSION
set(ONNXRUNTIME_VERSION 1.20.1)
#set(ONNXRUNTIME_ROOT "Z:/work/dentimus/cpp/camsec/camsec/build/packages/Microsoft.ML.OnnxRuntime.${ONNXRUNTIME_VERSION}/")

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/build/native/include)

set(PROJECT_SOURCES
	main.cpp
	model/clinic.h
	model/clinic.h
	model/config.h
	model/person.h
	nn/yolo_onnx.cpp
	nn/yolo_onnx.h
	nn/misc.cpp
	nn/misc.h
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

if (WIN32)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/runtimes/win-x64/native/onnxruntime.lib)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (LINUX)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)
    if (USE_CUDA)
        target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    endif ()
elseif (APPLE)
    target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)
endif ()

