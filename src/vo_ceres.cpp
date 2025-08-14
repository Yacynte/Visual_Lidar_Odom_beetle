
#include <cmath>
#include <iostream>
#include <vo_ceres.h>

namespace vo_ceres {

// --- ReprojectionLeft ---
ReprojectionLeft::ReprojectionLeft(const cv::Point3d& P_, const cv::Point2d& obs, const cv::Mat& K)
    : P(P_), p(obs) {
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
}

template <typename T>
bool ReprojectionLeft::operator()(const T* const camera, T* residuals) const {
    T P_t[3] = { T(P.x), T(P.y), T(P.z) };
    T pcam[3];
    ceres::AngleAxisRotatePoint(camera, P_t, pcam);
    pcam[0] += camera[3];
    pcam[1] += camera[4];
    pcam[2] += camera[5];
    T xp = pcam[0] / pcam[2];
    T yp = pcam[1] / pcam[2];
    T u = T(fx) * xp + T(cx);
    T v = T(fy) * yp + T(cy);
    residuals[0] = u - T(p.x);
    residuals[1] = v - T(p.y);
    return true;
}

ceres::CostFunction* ReprojectionLeft::Create(const cv::Point3d& P, const cv::Point2d& p, const cv::Mat& K) {
    return new ceres::AutoDiffCostFunction<ReprojectionLeft, 2, 6>(
        new ReprojectionLeft(P, p, K));
}

// --- ReprojectionRight ---
ReprojectionRight::ReprojectionRight(
    const cv::Point3d& P_, const cv::Point2d& obs, const cv::Mat& K,
    const cv::Mat& R_lr_, const cv::Mat& t_lr_)
    : P(P_), p(obs) {
    fx = K.at<double>(0, 0);
    fy = K.at<double>(1, 1);
    cx = K.at<double>(0, 2);
    cy = K.at<double>(1, 2);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c)
            R_lr[3*r + c] = R_lr_.at<double>(r, c);
        t_lr[r] = t_lr_.at<double>(r, 0);
    }
}

template <typename T>
bool ReprojectionRight::operator()(const T* const camera, T* residuals) const {
    T P_t[3] = { T(P.x), T(P.y), T(P.z) };
    T p_left[3];
    ceres::AngleAxisRotatePoint(camera, P_t, p_left);
    p_left[0] += camera[3];
    p_left[1] += camera[4];
    p_left[2] += camera[5];

    T p_right[3];
    for (int r = 0; r < 3; ++r) {
        p_right[r] = T(0);
        for (int c = 0; c < 3; ++c)
            p_right[r] += T(R_lr[3*r + c]) * p_left[c];
        p_right[r] += T(t_lr[r]);
    }

    T xp = p_right[0] / p_right[2];
    T yp = p_right[1] / p_right[2];
    T u = T(fx) * xp + T(cx);
    T v = T(fy) * yp + T(cy);
    residuals[0] = u - T(p.x);
    residuals[1] = v - T(p.y);
    return true;
}

ceres::CostFunction* ReprojectionRight::Create(const cv::Point3d& P, const cv::Point2d& p,
                                               const cv::Mat& K, const cv::Mat& R_lr, const cv::Mat& t_lr) {
    return new ceres::AutoDiffCostFunction<ReprojectionRight, 2, 6>(
        new ReprojectionRight(P, p, K, R_lr, t_lr));
}

// --- helper ---
cv::Mat deg2rad_vec(const cv::Mat& v_deg) {
    cv::Mat v;
    v_deg.convertTo(v, CV_64F);
    for (int i = 0; i < 3; ++i)
        v.at<double>(i) *= M_PI / 180.0;
    return v;
}

// --- TranslationNormPrior ---
TranslationNormPrior::TranslationNormPrior(double baseline, double weight) : baseline(baseline), w(weight) {}

template<typename T>
bool TranslationNormPrior::operator()(const T* const camera, T* residuals) const {
    // camera[3..5] are tx,ty,tz
    T tx = camera[3], ty = camera[4], tz = camera[5];
    T norm = ceres::sqrt(tx*tx + ty*ty + tz*tz);
    residuals[0] = T(w) * (norm - T(baseline));
    return true;
}

ceres::CostFunction* TranslationNormPrior::Create(double baseline, double weight) {
    return new ceres::AutoDiffCostFunction<TranslationNormPrior, 1, 6>(
        new TranslationNormPrior(baseline, weight));
    }

} // namespace vo_ceres
