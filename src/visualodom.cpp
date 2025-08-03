#include <opencv2/core/mat.hpp>
#include <iostream>
#include <stdexcept> // For std::bad_alloc and other exceptions
#include "vo_process.h"
#include "visualodom.h"



EXPORT void SetDebugLogCallback(DebugLogCallback callback) {
    unityDebugLog = callback;
}


EXPORT void UnityLog(const char* message) {
    if (unityDebugLog != nullptr) {
        unityDebugLog(message);
    }
}

/*

EXPORT bool add(float* a) {
    if (a == nullptr) {
        UnityLog("Error: Null pointer passed to add function.");
        return false;
    }

    // a[2] = a[0] + a[1];
    a[2] = 2.0f;
    // printf("add() called: %f + %f = %f\n", a[2], a[1]);
    return true;
}

bool ConvertBytesToMat(const unsigned char* imageData, int width, int height, int channels, unsigned char** matData, int* matRows, int* matCols, int* matType) {
    if (!imageData || width <= 0 || height <= 0 || channels <= 0) {
        // std::cerr << "Error: Invalid input parameters in ConvertBytesToMat." << std::endl;
        UnityLog("Error: Invalid input parameters in ConvertBytesToMat.");
        return false;
    }

    cv::Mat mat;
    try {
        if (channels == 1) {
            mat = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char*>(imageData));
        } else if (channels == 3) {
            mat = cv::Mat(height, width, CV_8UC3, const_cast<unsigned char*>(imageData));
        } else if (channels == 4) {
            mat = cv::Mat(height, width, CV_8UC4, const_cast<unsigned char*>(imageData));
        } else {
            // std::cerr << "Error: Unsupported number of channels in ConvertBytesToMat: " << channels << std::endl;
            UnityLog("Error: Unsupported number of channels in ConvertBytesToMat:");
            UnityLog(std::to_string(channels).c_str());
            return false; // Unsupported number of channels
        }
    } catch (const std::bad_alloc& e) {
        // std::cerr << "Error: Memory allocation failed in ConvertBytesToMat: " << e.what() << std::endl;
        UnityLog("Error: Memory allocation failed in ConvertBytesToMat:");
        UnityLog(e.what());
        return false;
    } catch (const std::exception& e) {
        // std::cerr << "Error creating cv::Mat in ConvertBytesToMat: " << e.what() << std::endl;
        UnityLog("Error creating cv::Mat in ConvertBytesToMat:");
        UnityLog(e.what());
        return false;
    }

    if (mat.empty()) {
        // std::cerr << "Error: Created cv::Mat is empty in ConvertBytesToMat." << std::endl;
        UnityLog("Error: Created cv::Mat is empty in ConvertBytesToMat.");
        return false;
    }

    size_t totalBytes = mat.total() * mat.elemSize();
    unsigned char* managedData = nullptr;
    try {
        managedData = new unsigned char[totalBytes];
    } catch (const std::bad_alloc& e) {
        // std::cerr << "Error: Memory allocation failed for managedData in ConvertBytesToMat: " << e.what() << std::endl;
        UnityLog("Error: Memory allocation failed for managedData in ConvertBytesToMat:");
        UnityLog(e.what());
        return false;
    }

    try {
        memcpy(managedData, mat.data, totalBytes);
    } catch (const std::exception& e) {
        // std::cerr << "Error copying data in ConvertBytesToMat: " << e.what() << std::endl;
        UnityLog("Error copying data in ConvertBytesToMat:");
        UnityLog(e.what());
        delete[] managedData;
        return false;
    }

    *matData = managedData;
    *matRows = mat.rows;
    *matCols = mat.cols;
    *matType = mat.type(); // Return the OpenCV Mat type

    return true;
}

EXPORT void ReleaseMatData(unsigned char* data) {
    if (data) {
        delete[] data;
        data = nullptr; // Good practice to set to null after deleting
    } else {
        // std::cerr << "Warning: ReleaseMatData called with a null pointer." << std::endl;
        UnityLog("Warning: ReleaseMatData called with a null pointer.");
    }
}

EXPORT VisualOdometryHandle CreateVisualOdometry() {
    VisualOdometry* voInstance = nullptr;
    try {
        voInstance = new VisualOdometry();
    } catch (const std::bad_alloc& e) {
        // std::cerr << "Error: Memory allocation failed in CreateVisualOdometry: " << e.what() << std::endl;
        UnityLog("Error: Memory allocation failed in CreateVisualOdometry:");
        UnityLog(e.what());
        return nullptr;
    } catch (const std::exception& e) {
        // std::cerr << "Error creating VisualOdometry instance: " << e.what() << std::endl;
        UnityLog("Error creating VisualOdometry instance:");
        UnityLog(e.what());
        delete voInstance;
        return nullptr;
    }
    return static_cast<VisualOdometryHandle>(voInstance);
}

EXPORT void DestroyVisualOdometry(VisualOdometryHandle handle) {
    if (handle) {
        VisualOdometry* voInstance = static_cast<VisualOdometry*>(handle);
        delete voInstance;
    } else {
        // std::cerr << "Warning: DestroyVisualOdometry called with a null handle." << std::endl;
        UnityLog("Warning: DestroyVisualOdometry called with a null handle.");
    }
}

EXPORT bool ProcessFrame(VisualOdometryHandle handle, unsigned char* preMatDataLeft, 
                        unsigned char* preMatDataRight, unsigned char* curMatDataLeft, 
                        unsigned char* curMatDataRight, int rows, int cols, float* pose)
{

    // UnityLog("Entering ProcessFrame");

    // char msg[128];
    // snprintf(msg, sizeof(msg), "Pointers: preMatDataLeft=%p, preMatDataRight=%p, curMatDataLeft=%p, curMatDataRight=%p, pose=%p", 
    //         preMatDataLeft, preMatDataRight, curMatDataLeft, curMatDataRight, pose);
    // UnityLog(msg);

    // snprintf(msg, sizeof(msg), "Rows: %d, Cols: %d", rows, cols);
    // UnityLog(msg);


    // if (!handle) {
    //     UnityLog("Error: Invalid VisualOdometry handle in ProcessFrame.");
    // return false;
    // }
    // if (!preMatDataLeft || !preMatDataRight || !curMatDataLeft || !curMatDataRight || rows <= 0 || cols <= 0 || !pose) {
    //     UnityLog("Error: Invalid input data in ProcessFrame.");
    //     return false;
    // }
    // UnityLog("Input data validated successfully in ProcessFrame.");
    try {


        // Create colored image matrices first
        // cv::Mat preImageLeftColor(rows, cols, CV_8UC3, preMatDataLeft);
        cv::Mat preImageRightColor(rows, cols, CV_8UC3, preMatDataRight);
        cv::Mat curImageLeftColor(rows, cols, CV_8UC3, curMatDataLeft);
        cv::Mat curImageRightColor(rows, cols, CV_8UC3, curMatDataRight);

        // Convert to grayscale
        cv::Mat preImageLeft, preImageRight, curImageLeft, curImageRight;

        // Convert raw byte* data to a vector
        std::vector<uchar> buffer(preMatDataLeft, preMatDataLeft + rows*cols*3);  // dataSize = total size of the byte array
        std::vector<uchar> buffer2(preMatDataRight, preMatDataRight + rows*cols*3);
        std::vector<uchar> buffer3(curMatDataLeft, curMatDataLeft + rows*cols*3);
        std::vector<uchar> buffer4(curMatDataRight, curMatDataRight + rows*cols*3);

        // Decode the image from the buffer
        preImageLeft = cv::imdecode(buffer, cv::IMREAD_COLOR);
        preImageRight = cv::imdecode(buffer2, cv::IMREAD_COLOR);
        curImageLeft = cv::imdecode(buffer3, cv::IMREAD_COLOR);
        curImageRight = cv::imdecode(buffer4, cv::IMREAD_COLOR);
        // Check if the images were decoded successfully
        if (preImageLeft.empty() || preImageRight.empty() || curImageLeft.empty() || curImageRight.empty()) {
            UnityLog("Error: Failed to decode images from byte array.");
            return false;
        }
        
        // UnityLog("Images decoded from byte array successfully.");


        cv::Mat prevTranslation_ = (cv::Mat_<double>(3, 1) << pose[0], pose[1], pose[2]);
        cv::Mat prevRotation_ = (cv::Mat_<double>(3, 1) << pose[3], pose[4], pose[5]);
        cv::Mat prevTranslation, prevRotation;
        prevRotation_.convertTo(prevRotation, CV_32F);
        prevTranslation_.convertTo(prevTranslation, CV_32F);

        cv::Mat rel_rotation_vector = cv::Mat::zeros(3, 1, CV_32F);
        cv::Mat rel_translation_vector = cv::Mat::zeros(3, 1, CV_32F);
        // UnityLog("Pose data converted to cv::Mat successfully.");

        VisualOdometry* voInstance = static_cast<VisualOdometry*>(handle);
        // UnityLog("VisualOdometry instance casted successfully.");
    


    // Check if the VisualOdometry instance is valid
    if (voInstance) {
        // UnityLog("VisualOdometry instance is valid in ProcessFrame.");
        // Call the StereoOdometry function
        try {
            voInstance->StereoOdometry(preImageLeft, curImageLeft, preImageRight, curImageRight, rel_rotation_vector, rel_translation_vector);
            voInstance->updatePose(prevTranslation, prevRotation, rel_translation_vector, rel_rotation_vector); // Review this logic
            pose[0] = prevTranslation.at<float>(0);
            pose[1] = prevTranslation.at<float>(1);
            pose[2] = prevTranslation.at<float>(2);
            pose[3] = prevRotation.at<float>(0);
            pose[4] = prevRotation.at<float>(1);
            pose[5] = prevRotation.at<float>(2);
            UnityLog("Pose updated! Success!");
            return true;
        } catch (const cv::Exception e) {
            UnityLog("Warning: StereoOdometry failed.");
            UnityLog(e.what());
            return false;
        }
        catch (const std::exception e) {
            UnityLog(" Standard exception in Calculation the odometry.");
            UnityLog(e.what());
            return false;
        }
        catch (...) {
            UnityLog("Unknown exception in Stereo Odometry.");
            return false;
        }
        
    } else {
        UnityLog("Error: VisualOdometry instance is null in ProcessFrame.");
        return false;
        }
    } catch (const cv::Exception& e) {
        UnityLog("OpenCV error in Process Frame:");
        UnityLog(e.what());
        return false;
    } catch (const std::exception& e) {
        UnityLog("Standard exception in ProcessFrame:");
        UnityLog(e.what());
        return false;
    } catch (...) {
        UnityLog("Unknown exception in ProcessFrame.");
        return false;
    }
}

*/