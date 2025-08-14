// StereoCamera.h
#pragma once

#include <vo_ceres.h>
#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
 // ORB and feature detection
#include <opencv2/calib3d.hpp>

#include <cassert> // Include this header

#include <opencv2/core/mat.hpp>

#include <vector>
#include <string>
#include <filesystem> // For directory and file handling

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <omp.h>
#include <numeric>      // std::accumulate
#include <unordered_set>


#include <algorithm>
#include <set>

#include <string>
#include <cstdlib> // for std::stoi


#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
// #include <opencv2/xfeatures2d.hpp>

struct MarkerInfo {
            int id;
            std::vector<cv::Point2f> corners;
        };

struct RelativePose {
        // cv::Mat R, t;
        Eigen::Vector3f R = Eigen::Vector3f::Zero();
        Eigen::Vector3f t = Eigen::Vector3f::Zero();
        bool valid = false;
        std::string sensor_type; // "camera" or "lidar"
        float timestamp; // Timestamp of the pose
    };

struct CountourPose {
        // cv::Mat R, t;
        Eigen::Matrix3f R = Eigen::Vector3f::Zero();
        Eigen::Vector3f t;
        int position;
        bool valid = false;
    };


class VisualOdometry {
    public:
        // Constructor to initialize fx, fy, cx, cy
        VisualOdometry() {
            fx = K1.at<double>(0, 0);
            fy = K1.at<double>(1, 1);
            cx = K1.at<double>(0, 2);
            cy = K1.at<double>(1, 2);

            K1.convertTo(K1_float, CV_32F);
            D1.convertTo(D1_float, CV_32F);
            K2.convertTo(K2_float, CV_32F);
            D2.convertTo(D2_float, CV_32F);
        }
        
        // Method to compute stereo odometry
        bool StereoOdometry(cv::Mat leftImage_color, cv::Mat preLeftImage, cv::Mat curLeftImage, cv::Mat preRightImage, cv::Mat curRightImage, 
                            RelativePose* rel_pose, CountourPose* contour_pose);
                            // cv::Mat& rotation_vector, cv::Mat& translation_vector, CountourPose* contour_pose); //, cv::Mat init_R, cv::Mat init_T);

        void updatePose(cv::Mat& tot_translation_vector, cv::Mat& tot_rotation_vector,  cv::Mat& rel_translation_vector, cv::Mat& rel_rotation_vector);


        bool detectArucoMarkers(const cv::Mat& image, std::map<int, MarkerInfo>& detectedMarkers);

        bool detectContourMarkers(const cv::Mat& image, std::vector<cv::Point2f>& contourPoints);

        void estimateMarkersPose(const cv::Mat& imageLeft, const cv::Mat& imageRight,
                                std::vector<cv::Point2f>& contourPoints,
                                cv::Mat& rvec, cv::Mat& tvec);

        void estimateMarkersPose(const cv::Mat& imageLeft, const cv::Mat& imageRight,
                                std::map<int, MarkerInfo>& detectedLeftMarkers,
                                std::map<int, MarkerInfo>& detectedRightMarkers,
                                CountourPose* contour_pose);

    private:

        // Match corners based on shared marker IDs
        std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matchMarkerCorners(
            const std::map<int, MarkerInfo>& leftMarkers, const std::map<int, MarkerInfo>& rightMarkers);
                
        // Motion estimation from two sets of 2D points and depth map.
        bool motionEstimation(const cv::Mat& leftImage_color, const std::vector<cv::Point2f>& image1_points_L, const std::vector<cv::Point2f>& image2_points_L,
                            // const std::vector<cv::Point2f>& image1_points_R, const std::vector<cv::Point2f>& image2_points_R,
                            const cv::Mat& depth, RelativePose* rel_pose, cv::Mat leftImage_cur_,
                            CountourPose* contour_pose);


        cv::Mat K1 = (cv::Mat_<double>(3, 3) << 9.597910e+02, 0, 6.960217e+02,
                                                0, 9.569251e+02, 2.241806e+02,
                                                0, 0, 1);

        cv::Mat D1 = (cv::Mat_<double>(1, 5) << -3.691481e-01, 1.968681e-01, 1.353473e-03, 5.677587e-04, -6.770705e-02);

        cv::Mat R2 = (cv::Mat_<double>(3, 3) << 9.999758e-01, -5.267463e-03, -4.552439e-03,
                                                5.251945e-03, 9.999804e-01, -3.413835e-03,
                                                4.570332e-03, 3.389843e-03, 9.999838e-01);

        cv::Mat T2 = (cv::Mat_<double>(3, 1) << 5.956621e-02,
                                                2.900141e-04,
                                                2.577209e-03);


        cv::Mat K2 = (cv::Mat_<double>(3, 3) << 9.037596e+02, 0, 6.957519e+02,
                                                0, 9.019653e+02, 2.242509e+02,
                                                0, 0, 1);

        cv::Mat D2 = (cv::Mat_<double>(1, 5) << -3.639558e-01, 1.788651e-01, 6.029694e-04, -3.922424e-04, -5.382460e-02);
        
        // You might have another rotation and translation if R_03 and T_03 represent the pose of the second camera
        cv::Mat R = (cv::Mat_<double>(3, 3) << 9.995599e-01, 1.699522e-02, -2.431313e-02,
                                                    -1.704422e-02, 9.998531e-01, -1.809756e-03,
                                                    2.427880e-02, 2.223358e-03, 9.997028e-01);
        
        cv::Mat T = (cv::Mat_<double>(3, 1) << -4.731050e-01,
                                                5.551470e-03,
                                                -5.250882e-03);


       
        cv::Ptr<cv::ORB> orb = cv::ORB::create();

        float fx, fy, cx, cy, threshold = 1.0f;
        cv::Mat K1_float, D1_float, K2_float, D2_float;

        cv::Mat prev_disparity; // Store this from previous frame
        
        // cv::Mat disparity;
        // cv::Mat depth_map;
        // cv::Mat rvec;                           // Sotre Rotation vector
        // cv::Mat rotation_matrix;                // Sotre Rotation matrix 
        // cv::Mat translation_vector;
        
        // Method to rectify images
        void RectifyImage(cv::Mat& leftImage, cv::Mat& rightImage);

        void reconstruct3D(const std::vector<cv::Point2f>& image_points, const cv::Mat& depth,
                           std::vector<cv::Point3f>& points_3D, std::vector<size_t>& outliers);

        //Compute mean of 3D points.
        cv::Point3f computeMean3D(const std::vector<cv::Point3f>& points);

        // Compute Disparity
        cv::Mat computeDisparity(const cv::Mat& left, const cv::Mat& right);

        // Compute depth map
        cv::Mat computeDepth(const cv::Mat& left, const cv::Mat& right, const float max_depth = 500.0f); 

        void feature_matching(const cv::Mat& left_prev, const cv::Mat& left_cur, 
                            std::vector<cv::Point2f>& pts_prev_L, std::vector<cv::Point2f>& pts_cur_L);
        void matchWithSIFT(const cv::Mat& img1, const cv::Mat& img2,
                                std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) ;
        void filterKeypointsByDistance(std::vector<cv::KeyPoint>&  keypoints, cv::Mat& descriptors, 
                            std::vector<cv::KeyPoint>&  filtered_keypoints, cv::Mat& filtered_descriptors, double min_distance);

        void filterKeypointsByGrid(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
                                std::vector<cv::KeyPoint>& filtered_keypoints, cv::Mat& filtered_descriptors,
                                int grid_rows, int grid_cols, int max_per_cell);

        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        
        
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
       // Optional: Detector parameters (can be fine-tuned)
        // cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::aruco::DetectorParameters parameters = cv::aruco::DetectorParameters();

        float aruco_corner_dist = 0.3f;
        float half_dist = aruco_corner_dist / 2.0f;
        std::vector<cv::Point3f> aruco_objectPoints = {
            {-half_dist,  half_dist, 0.0f},  // top-left
            { half_dist,  half_dist, 0.0f},  // top-right
            { half_dist, -half_dist, 0.0f},  // bottom-right
            {-half_dist, -half_dist, 0.0f}   // bottom-left
        };

        void optimizePoseWithCeres( std::vector<int> inliers_L, std::vector<int> inliers_R, 
                                            const cv::Mat& points3D_L, const cv::Mat& points2D_L,
                                            const cv::Mat& points3D_R, const cv::Mat& points2D_R,
                                            cv::Mat& rvec_degree, cv::Mat& tvec);

protected:
        // -------------------- Utilities --------------------
    static std::mutex cout_mtx;
    void safe_log(const std::string &s);

};  

