// VisualOdometry.h
#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/opencv.hpp>

namespace vo_ceres {

    struct ReprojectionLeft {
        ReprojectionLeft(const cv::Point3d& P, const cv::Point2d& obs, const cv::Mat& K);
        template <typename T>
        bool operator()(const T* const camera, T* residuals) const;
        static ceres::CostFunction* Create(const cv::Point3d& P, const cv::Point2d& p, const cv::Mat& K);

        cv::Point3d P;
        cv::Point2d p;
        double fx, fy, cx, cy;
    };

    struct ReprojectionRight {
        ReprojectionRight(const cv::Point3d& P, const cv::Point2d& obs, const cv::Mat& K,
                        const cv::Mat& R_lr_, const cv::Mat& t_lr_);
        template <typename T>
        bool operator()(const T* const camera, T* residuals) const;
        static ceres::CostFunction* Create(const cv::Point3d& P, const cv::Point2d& p, const cv::Mat& K,
                                        const cv::Mat& R_lr, const cv::Mat& t_lr);

        cv::Point3d P;
        cv::Point2d p;
        double fx, fy, cx, cy;
        double R_lr[9];
        double t_lr[3];
    };

    // helper
    cv::Mat deg2rad_vec(const cv::Mat& v_deg);

    struct TranslationNormPrior {
        TranslationNormPrior(double baseline, double weight);

        template<typename T>
        bool operator()(const T* const camera, T* residuals) const;
        static ceres::CostFunction* Create(double baseline, double weight);

        
        double baseline;
        double w;
    };

} // namespace vo_ceres

