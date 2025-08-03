#pragma once
#include <opencv2/core/mat.hpp>
#include <iostream>
#include <stdexcept> // For std::bad_alloc and other exceptions
#include "vo_process.h"

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT void UnityLog(const char* message);

typedef void* VisualOdometryHandle;

// Delegate type for the Unity Debug.Log function (defined in C#)
typedef void (*DebugLogCallback)(const char* message);
static DebugLogCallback unityDebugLog;

// extern "C" {
//     // EXPORT int add(int a, int b);
//     EXPORT bool add(float* a);
//     EXPORT void SetDebugLogCallback(DebugLogCallback callback);
//     // EXPORT void LogMessageFromNative(const char* message);

//     // EXPORT bool ConvertBytesToMat(const unsigned char* imageData, int width, int height, int channels, unsigned char** matData, int* matRows, int* matCols, int* matType);
//     EXPORT void ReleaseMatData(unsigned char* data);
//     EXPORT VisualOdometryHandle CreateVisualOdometry();
//     EXPORT void DestroyVisualOdometry(VisualOdometryHandle handle);
//     EXPORT bool ProcessFrame(VisualOdometryHandle handle, unsigned char* preMatDataLeft,
//                             unsigned char* preMatDataRight, unsigned char* curMatDataLeft, 
//                             unsigned char* curMatDataRight, int rows, int cols, float* pose);
// }

