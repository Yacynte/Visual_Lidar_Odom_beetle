#include <opencv2/core/mat.hpp>
#include <iostream>
#include "vo_process.h"
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

// typedef void (*StereoOdometryFunc)(cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&, cv::Mat&);
typedef void* VisualOdometryHandle;
extern "C" {
    // ... (ConvertBytesToMat and ReleaseMatData declarations)
    EXPORT bool ConvertBytesToMat(const unsigned char* imageData, int width, int height, int channels, unsigned char** matData, int* matRows, int* matCols, int* matType);
    EXPORT void ReleaseMatData(unsigned char* data);

    EXPORT VisualOdometryHandle CreateVisualOdometry();
    EXPORT void DestroyVisualOdometry(VisualOdometryHandle handle);
    EXPORT void ProcessFrame(VisualOdometryHandle handle, unsigned char* preMatDataLeft, unsigned char* preMatDataRight, unsigned char* curMatDataLeft, unsigned char* curMatDataRight,
                             int rows, int cols, int type, float* pose);
}

EXPORT bool ConvertBytesToMat(const unsigned char* imageData, int width, int height, int channels, unsigned char** matData, int* matRows, int* matCols, int* matType) {
    if (!imageData || width <= 0 || height <= 0 || channels <= 0) {
        return false;
    }

    cv::Mat mat;
    if (channels == 1) {
        mat = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(imageData));
    } else if (channels == 3) {
        mat = cv::Mat(height, width, CV_8UC3, const_cast<unsigned char*>(imageData));
    } else if (channels == 4) {
        mat = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(imageData));
    } else {
        return false; // Unsupported number of channels
    }

    if (mat.empty()) {
        return false;
    }

    // Allocate memory for the Mat data that can be accessed from C#
    size_t totalBytes = mat.total() * mat.elemSize();
    unsigned char* managedData = new unsigned char[totalBytes];
    memcpy(managedData, mat.data, totalBytes);

    // Return the pointer and dimensions to C#
    *matData = managedData;
    *matRows = mat.rows;
    *matCols = mat.cols;
    *matType = mat.type(); // Return the OpenCV Mat type

    return true;
}

EXPORT void ReleaseMatData(unsigned char* data) {
    delete[] data;
}

EXPORT VisualOdometryHandle CreateVisualOdometry() {
    // Create a new instance of VisualOdometry
    VisualOdometry* voInstance = new VisualOdometry();
    return static_cast<VisualOdometryHandle>(voInstance);
}

EXPORT void DestroyVisualOdometry(VisualOdometryHandle handle) {
    VisualOdometry* voInstance = static_cast<VisualOdometry*>(handle);
    delete voInstance;
}

EXPORT void ProcessFrame(VisualOdometryHandle handle, unsigned char* preMatDataLeft, unsigned char* preMatDataRight, unsigned char* curMatDataLeft, unsigned char* curMatDataRight,
                        int rows, int cols, int type, float* pose)
{
    // Convert the byte arrays to cv::Mat
    cv::Mat preImageLeft(rows, cols, type, preMatDataLeft);
    cv::Mat preImageRight(rows, cols, type, preMatDataRight);
    cv::Mat curImageLeft(rows, cols, type, curMatDataLeft);
    cv::Mat curImageRight(rows, cols, type, curMatDataRight);
    cv::Mat prevTranslation_ = (cv::Mat_<double>(3, 1) << pose[0], pose[1], pose[2]);
    cv::Mat prevRotation_ = (cv::Mat_<double>(3, 1) << pose[3], pose[4], pose[5]); 
    cv::Mat prevTranslation, prevRotation;
    prevRotation_.convertTo(prevRotation, CV_32F);
    prevTranslation_.convertTo(prevTranslation, CV_32F);

    VisualOdometry* voInstance = static_cast<VisualOdometry*>(handle);
    // Call the StereoOdometry function
    if( voInstance->StereoOdometry (preImageLeft, curImageLeft, preImageRight, curImageRight, prevRotation, prevTranslation)){
        voInstance->updatePose(prevRotation, prevTranslation, prevTranslation, prevRotation);
        pose[0] = prevTranslation.at<float>(0);
        pose[1] = prevTranslation.at<float>(1);
        pose[2] = prevTranslation.at<float>(2);
        pose[3] = prevRotation.at<float>(0);
        pose[4] = prevRotation.at<float>(1);
        pose[5] = prevRotation.at<float>(2);
        // std::cout<<"Pose updated!"<<std::endl;
        std::cout<<"Success!"<<std::endl;
    }


}