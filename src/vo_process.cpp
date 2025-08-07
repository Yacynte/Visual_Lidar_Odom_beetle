#include "vo_process.h"
#include "visualodom.h"
#include <ceres/ceres.h>


// bool VisualOdometry::StereoOdometry(cv::Mat leftImage_color, cv::Mat leftImage_pre_, cv::Mat leftImage_cur_, cv::Mat rightImage_pre_, cv::Mat rightImage_cur_, 
//                                     cv::Mat& rotation_vector, cv::Mat& translation_vector, CountourPose* contour_pose) {

bool VisualOdometry::StereoOdometry(cv::Mat leftImage_color, cv::Mat leftImage_pre, cv::Mat leftImage_cur, cv::Mat rightImage_pre, cv::Mat rightImage_cur, 
                                            cv::Mat& rotation_vector, cv::Mat& translation_vector, CountourPose* contour_pose) {

    // if (leftImage_pre_.empty(r << ) || leftImage_cur_.empty() || rightImage_pre_.empty() || rightImage_cur_.empty()) {
    if (leftImage_pre.empty() || leftImage_cur.empty() || rightImage_pre.empty() || rightImage_cur.empty()) {
        std::cerr << "One or all images are Empty!" << std::endl;
        UnityLog("One or all images are Empty!");
        // Initialize rotation and translation vectors to -1
        cv::Mat rotation_vector = cv::Mat(3, 1, CV_32F, cv::Scalar(-1));
        cv::Mat translation_vector =cv::Mat(3, 1, CV_32F, cv::Scalar(-1));
    }

    // Half the image size
    // int new_width = leftImage_pre_.cols / 2;
    // int new_height = leftImage_pre_.rows / 2;
    // cv::Size new_size(new_width, new_height);

    // // Create a matrix to store the resized image
    // cv::Mat leftImage_pre, leftImage_cur, rightImage_pre, rightImage_cur;
    // // Resize the images using high-quality interpolation
    // cv::resize(leftImage_pre_, leftImage_pre, new_size, 0, 0, cv::INTER_AREA);
    // cv::resize(leftImage_cur_, leftImage_cur, new_size, 0, 0, cv::INTER_AREA);
    // cv::resize(rightImage_pre_, rightImage_pre, new_size, 0, 0, cv::INTER_AREA);
    // cv::resize(rightImage_cur_, rightImage_cur, new_size, 0, 0, cv::INTER_AREA);


    // Step 1: Rectify images
    std::thread t1([&]() { RectifyImage(leftImage_pre, rightImage_pre); });
    std::thread t2([&]() { RectifyImage(leftImage_cur, rightImage_cur); });

    t1.join();
    t2.join();
    
    //Compute depth map of previous Image pair
    cv::Mat depth_map = computeDepth(leftImage_pre, rightImage_pre);
    
    // Vectors to hold the matched points
    std::vector<cv::Point2f> pts_prev_L, pts_cur_L, pts_prev_R, pts_cur_R;

    // Call the feature matching function
    // std::cout<<"feature matching"<<std::endl;
    // feature_matching(leftImage_pre, leftImage_cur, pts_prev_L, pts_cur_L);
    matchWithSIFT(leftImage_pre, leftImage_cur, pts_prev_L, pts_cur_L);
    matchWithSIFT(rightImage_pre, rightImage_cur, pts_prev_R, pts_cur_R);
    int num_matching = pts_prev_L.size();
    // std::string message = "Number of matching points: " + std::to_string(num_matching);
    // UnityLog(message.c_str());
    // std::cout<<"Motion Estimation"<<std::endl;
    bool state = motionEstimation(leftImage_color, pts_prev_L, pts_cur_L, pts_prev_R, pts_cur_R, depth_map, rotation_vector, translation_vector, leftImage_cur, contour_pose);
    
    return state;
}

void VisualOdometry::RectifyImage(cv::Mat& leftImage, cv::Mat& rightImage) {
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(K1, D1, K2, D2, leftImage.size(), R, T, R1, R2, P1, P2, Q);

    cv::Mat map1x, map1y, map2x, map2y;
    cv::Mat rectifiedLeft, rectifiedRight;
    cv::Size imageSize = leftImage.size();

    // Lambda function for rectifying the left image
    auto rectifyLeft = [&]() {
        cv::initUndistortRectifyMap(K1, D1, R1, P1, imageSize, CV_32FC1, map1x, map1y);
        cv::remap(leftImage, rectifiedLeft, map1x, map1y, cv::INTER_LINEAR);
        leftImage = rectifiedLeft.clone();
    };

    // Lambda function for rectifying the right image
    auto rectifyRight = [&]() {
        cv::initUndistortRectifyMap(K2, D2, R2, P2, imageSize, CV_32FC1, map2x, map2y);
        cv::remap(rightImage, rectifiedRight, map2x, map2y, cv::INTER_LINEAR);
        rightImage = rectifiedRight.clone();
    };

    // Create and launch the threads
    std::thread leftThread(rectifyLeft);
    std::thread rightThread(rectifyRight);

    // Wait for both threads to finish
    leftThread.join();
    rightThread.join();

}

void VisualOdometry::reconstruct3D(const std::vector<cv::Point2f>& image_points, const cv::Mat& depth,
                                   std::vector<cv::Point3f>& points_3D, std::vector<size_t>& outliers) {
    points_3D.clear();
    outliers.clear();

    for (size_t i = 0; i < image_points.size(); ++i) {
        float u = image_points[i].x;
        float v = image_points[i].y;
        float z = depth.at<float>(static_cast<int>(v), static_cast<int>(u));

        // Ignore points with invalid depth
        if (z <= 0) {
            outliers.push_back(i);
            continue;
        }

        float x = z * (u - cx) / fx;
        float y = z * (v - cy) / fy;
        points_3D.emplace_back(x, y, z);
    }
    // std::cout << "image_points: " << image_points.size() << " points 3d: " << points_3D.size() << " outliers : " << outliers.size()  <<std::endl;
}

cv::Point3f VisualOdometry::computeMean3D(const std::vector<cv::Point3f>& points) {
    if (points.empty()) return cv::Point3f(0, 0, 0);

    cv::Point3f mean = std::accumulate(points.begin(), points.end(), cv::Point3f(0, 0, 0));
    mean *= (1.0f / points.size());
    return mean;
}

bool VisualOdometry::motionEstimation(const cv::Mat& leftImage_color, const std::vector<cv::Point2f>& image1_points_L,
                                        const std::vector<cv::Point2f>& image2_points_L, const std::vector<cv::Point2f>& image1_points_R,
                                        const std::vector<cv::Point2f>& image2_points_R, const cv::Mat& depth, cv::Mat& rvec, 
                                        cv::Mat& translation_vector, cv::Mat leftImage_cur_, CountourPose* contour_pose) 
{
    if (image1_points_L.size() != image2_points_L.size() || image1_points_R.size() != image2_points_R.size()) {
        std::cerr << "Error: Point sets must have the same size." << std::endl;
        return false;
    }

    // Step 1: Reconstruct 3D points
    std::vector<cv::Point3f> points_3D_L, points_3D_R;
    std::vector<size_t> outliers_L, outliers_R;
    reconstruct3D(image1_points_L, depth, points_3D_L, outliers_L);
    reconstruct3D(image1_points_R, depth, points_3D_R, outliers_R);

    std::vector<cv::Point2f> contourPoses2d;
    std::vector<cv::Point3f> contourPoses;
    // if(detectContourMarkers(leftImage_cur_, contourPoses2d)){
    //     reconstruct3D(contourPoses2d, depth, contourPoses, outliers, max_depth);
    //     cv::Point3f meanPose = computeMean3D(contourPoses);
    //     contour_pose->R = cv::Mat::eye(3, 3, CV_32F);
    //     contour_pose->t = cv::Mat(meanPose);
    //     contour_pose->valid = true;
    // }
    
    // Step 2: Use an unordered_set for fast outlier lookup
    std::unordered_set<size_t> outlier_set_L(outliers_L.begin(), outliers_L.end());
    std::unordered_set<size_t> outlier_set_R(outliers_R.begin(), outliers_R.end());

    // Filter points, reserving space to avoid multiple reallocations
    std::vector<cv::Point2f> filtered_image2_points_L, filtered_image2_points_R;
    std::vector<cv::Point3f> filtered_points_3D_L, filtered_points_3D_R;
    size_t num_points_L = image1_points_L.size();
    size_t num_points_R = image1_points_R.size();
    // filtered_image1_points.reserve(image1_points.size());
    // filtered_image2_points.reserve(image2_points.size());
    // filtered_points_3D.reserve(points_3D.size());


    for (size_t i = 0; i < num_points_L; ++i) {
        if (outlier_set_L.find(i) == outlier_set_L.end()) {
            // filtered_image1_points.push_back(image1_points[i]);
            filtered_image2_points_L.push_back(image2_points_L[i]);
            // filtered_points_3D.push_back(points_3D[i]);
        }
    }

    for (size_t i = 0; i < num_points_R; ++i) {
        if (outlier_set_R.find(i) == outlier_set_R.end()) {
            // filtered_image1_points.push_back(image1_points[i]);
            filtered_image2_points_R.push_back(image2_points_R[i]);
            // filtered_points_3D.push_back(points_3D[i]);
        }
    }

    // Step 3: Solve PnP with RANSAC
    std::cout << "Left image points size: " << filtered_image2_points_L.size() << std::endl;
    std::cout << " Left 3D points size: " << points_3D_L.size() << std::endl;
    std::cout << " Left mage 1 points size: " << image1_points_L.size() << std::endl;

    std::cout << " Right image points size: " << filtered_image2_points_R.size() << std::endl;
    std::cout << " Right 3D points size: " << points_3D_R.size() << std::endl;
    std::cout << " Right mage 1 points size: " << image1_points_R.size() << std::endl;

    cv::Mat filtered_points_3D_mat_L(points_3D_L), filtered_image2_points_mat_L(filtered_image2_points_L);
    filtered_points_3D_mat_L.convertTo(filtered_points_3D_mat_L, CV_32F);
    filtered_image2_points_mat_L.convertTo(filtered_image2_points_mat_L, CV_32F);

    cv::Mat filtered_points_3D_mat_R(points_3D_R), filtered_image2_points_mat_R(filtered_image2_points_R);
    filtered_points_3D_mat_R.convertTo(filtered_points_3D_mat_R, CV_32F);
    filtered_image2_points_mat_R.convertTo(filtered_image2_points_mat_R, CV_32F);
    // std::vector<int> inliers;
    // std::string message = "Filtered image points size: " + std::to_string(filtered_image2_points_mat.size[0]);
    // UnityLog(message.c_str());
    // std::string message2 = "Filtered 3D points size: " + std::to_string(points_3D_mat.size[0]);
    // UnityLog(message2.c_str());
    // std::cout << " length of outliers: " << outliers.size() << std::endl;
    // std::cout << " length of unfiltered 2d points: " << image2_points.size() << std::endl;
    // std::cout << "Filtered image points size: " << filtered_image2_points_mat.size[0] << std::endl;
    // std::cout << "Filtered 3D points size: " << filtered_points_3D_mat.size[0] << std::endl;
    // std::cout << "3D points: " << points_3D.size() << std::endl;
    std::vector<int> inliers_L, inliers_R; ;
    cv::Mat rvec_L, tvec_L, rvec_R, tvec_R;
    bool success_L = cv::solvePnPRansac(filtered_points_3D_mat_L, filtered_image2_points_mat_L, K1_float, 
                                      D1_float, rvec_L, tvec_L, false, 100, 8.0, 0.99, inliers_L, cv::SOLVEPNP_ITERATIVE);
    bool success_R = cv::solvePnPRansac(filtered_points_3D_mat_R, filtered_image2_points_mat_R, K2_float, 
                                      D2_float, rvec_R, tvec_R, false, 100, 8.0, 0.99, inliers_R, cv::SOLVEPNP_ITERATIVE);
   
    // std::cout << "Translation: " << rvec << std::endl;
    if (success_L == false){ std::cout << "RansacPnP Failed for left images \n";}
    // if (success_R == false){ std::cout << "RansacPnP Failed for right images \n";}

    std::cout << "Transformation left: " << tvec_L << std::endl;
    std::cout << "Transformation right: " << tvec_R << std::endl;

    double camera_L[6], camera_R[6];
    // rvec_L.convertTo(rvec_L, CV_64F);
    // tvec_L.convertTo(tvec_L, CV_64F);
    // rvec_R.convertTo(rvec_R, CV_64F);
    // tvec_R.convertTo(tvec_R, CV_64F);
    for (int i = 0; i < 3; ++i) {
        camera_L[i] = rvec_L.at<double>(i);
        camera_L[i + 3] = tvec_L.at<double>(i);
        camera_R[i] = rvec_R.at<double>(i);
        camera_R[i + 3] = tvec_R.at<double>(i);
    }
    // std::cout << "Camera L: " << camera_L[0] << ", " << camera_L[1] << ", " << camera_L[2] << ", "
    //           << camera_L[3] << ", " << camera_L[4] << ", " << camera_L[5] << std::endl;
    // fill camera[] with rvec (angle-axis) and tvec from OpenCV solvePnP (as double)
    ceres::Problem problem;
    for(size_t i=0;i<inliers_L.size();++i) {
        int idx = inliers_L[i];
        ceres::CostFunction* cf = ReprojectionError::Create(points_3D_L[idx], filtered_image2_points_L[idx], K1_float);
        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cf, loss, camera_L);
    }

    for(size_t i=0;i<inliers_R.size();++i) {
        int idx = inliers_R[i];
        ceres::CostFunction* cf = ReprojectionError::Create(points_3D_R[idx], filtered_image2_points_R[idx], K2_float);
        ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cf, loss, camera_R);
    }

    // Optionally: keep scale or impose a small prior on translation magnitude:
    // ceres::CostFunction* prior =
    //     new ceres::AutoDiffCostFunction<TranslationPrior,3,6>(new TranslationPrior(initial_t));
    // problem.AddResidualBlock(prior, new ceres::ScaledLoss(nullptr, 1e-4, ceres::TAKE_OWNERSHIP), camera);

    // Solve
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.FullReport() << "\n";

    rvec_L.convertTo(rvec, CV_64F);
    tvec_L.convertTo(translation_vector, CV_64F);
    
    for (int i = 0; i < 3; ++i) {
        rvec.at<double>(i) = camera_L[i];
        translation_vector.at<double>(i) = camera_L[i + 3];
    }

    return success_L && success_R; // Return true if both PnP solutions were successful
    
}

cv::Mat VisualOdometry::computeDisparity(const cv::Mat& left, const cv::Mat& right) {
    // Create StereoSGBM object
    // auto stereo = cv::StereoSGBM::create(
    //     0,            // Min disparity
    //     4 * 16,       // Reduced number of disparities for faster computation
    //     9,            // Smaller block size for faster computation
    //     8 * 3 * 9 * 9,    // P1 with block size of 5
    //     32 * 3 * 9 * 9,   // P2 with block size of 5
    //     31,           // Pre-filter cap
    //     15,           // Uniqueness ratio
    //     50,            // Speckle window size
    //     2,            // Speckle range
    //     cv::StereoSGBM::MODE_SGBM // Mode
    // );

    // // Compute disparity
    // // cv::Mat disparity(left.size(), CV_16S);
    // cv::Mat disparity;
    // stereo->compute(left, right, disparity);

    // // Convert to float for bilateral filtering
    // cv::Mat disparity_float;
    // disparity.convertTo(disparity_float, CV_32F);  // scale back to original disparity, SGBM divides disparity by 16

    // std::cout << "Disparity computed with size: " << disparity_float.size() << std::endl;

    // Create StereoSGBM object
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create(0, 4*16, 5);
    stereo->setP1(8 * left.channels() * 5 * 5);
    stereo->setP2(32 * left.channels() * 5 * 5);
    stereo->setPreFilterCap(63);
    stereo->setMode(cv::StereoSGBM::MODE_SGBM);

    // Compute disparity map
    cv::Mat disparity;
    stereo->compute(left, right, disparity);

    // Convert disparity to float
    cv::Mat disparity_float;
    disparity.convertTo(disparity_float, CV_32F, 1.0 / 16.0);

    return disparity_float;
}
// Compute depth from disparity
cv::Mat VisualOdometry::computeDepth(const cv::Mat& left, const cv::Mat& right, const float max_depth) {
    cv::Mat disparity = computeDisparity(left, right);  // Assuming this is a pre-defined function

    double focal_length_ = K1.at<double>(0, 0);
    double baseline_ = T2.at<double>(0) - T.at<double>(0);
    // double baseline_ = 0.5; // KITTI baseline value, adjust as needed

    cv::Mat depth(disparity.size(), CV_32F);

    // Define a parallel loop for processing each pixel
    cv::parallel_for_(cv::Range(0, disparity.rows), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y) {
            for (int x = 0; x < disparity.cols; ++x) {
                float d = disparity.at<float>(y, x) ;
                if (d > 0) { // Avoid division by zero
                    float depth_ = (focal_length_ * baseline_) / d; // Depth calculation
                    if (depth_ > max_depth) {
                        depth_ = 0.0f; // Set to 0 if depth exceeds max_depth
                    }
                    depth.at<float>(y, x) = depth_; // Store the computed depth
                } else {
                    depth.at<float>(y, x) = 0.0f; // Invalid depth
                }
            }
        }
    });

    std::cout << "Depth map computed with size: " << depth.size() << std::endl;
    double minVal, maxVal;
    cv::minMaxLoc(depth, &minVal, &maxVal);
    std::cout << "Min value: " << minVal << ", Max value: " << maxVal << std::endl;
    return depth;
}


// Feature matching function
void VisualOdometry::feature_matching(const cv::Mat& left_prev, const cv::Mat& left_cur,
    std::vector<cv::Point2f>& pts_prev_L, std::vector<cv::Point2f>& pts_cur_L) {
    
    int MAX_FEATURES = 1000, grid_rows = 3, grid_cols = 8, min_distance = 0;
    
    cv::Ptr<cv::ORB> orb1 = cv::ORB::create(MAX_FEATURES);
    cv::Ptr<cv::ORB> orb2 = cv::ORB::create(MAX_FEATURES);
    
    // cv::Ptr<cv::SURF> orb1 =cv::SURF::create(MAX_FEATURES);

    std::vector<cv::KeyPoint> keypoints1, keypoints2, filtered_keypoints1, filtered_keypoints2;
    cv::Mat descriptors1, descriptors2, filtered_descriptors1, filtered_descriptors2;

    // Feature detection + description
    orb1->detectAndCompute(left_prev, cv::noArray(), keypoints1, descriptors1);
    orb2->detectAndCompute(left_cur, cv::noArray(), keypoints2, descriptors2);

    // filterKeypointsByDistance(keypoints1, descriptors1, filtered_keypoints1, filtered_descriptors1, min_distance);
    // filterKeypointsByDistance(keypoints2, descriptors2, filtered_keypoints2, filtered_descriptors2, min_distance);

    filtered_keypoints1 = keypoints1, filtered_keypoints2 = keypoints2;
    filtered_descriptors1 = descriptors1, filtered_descriptors2 = descriptors2;
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(filtered_descriptors1, filtered_descriptors2, knn_matches, 2);

    // std::string message2 = "KNN matches found: " + std::to_string(knn_matches.size());
    // UnityLog(message2.c_str());

    // 4. Lowe's ratio test
    const float ratio_thresh = 0.95f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    
    // std::cout << "Good matches found: " << good_matches.size() << std::endl;
    // std::string message3 = "Good matches found: " + std::to_string(good_matches.size());
    // UnityLog(message3.c_str());

    // 5. Extract matched points
    pts_prev_L.clear();
    pts_cur_L.clear();
    for (const auto& match : good_matches) {
        pts_prev_L.push_back(filtered_keypoints1[match.queryIdx].pt);
        pts_cur_L.push_back(filtered_keypoints2[match.trainIdx].pt);
    }
    
    std::cout << "filtered_keypoints found: " << pts_prev_L.size() << "\n";

    // UnityLog("Number of matched points: ");
    // UnityLog(std::to_string(pts_prev_L.size()).c_str());
}



void VisualOdometry::updatePose(cv::Mat& tot_translation_vector, cv::Mat& tot_rotation_vector,
                                cv::Mat& rel_translation_vector, cv::Mat& rel_rotation_vector) {

    rel_translation_vector.convertTo(rel_translation_vector, CV_32F);
    rel_rotation_vector.convertTo(rel_rotation_vector, CV_32F);

    // Convert relative rotation vector to rotation matrix
    cv::Mat rel_rotation_matrix(3, 3, CV_32F);
    cv::Rodrigues(rel_rotation_vector, rel_rotation_matrix);
    // rel_rotation_matrix.convertTo(rel_rotation_matrix, CV_32F); // Explicitly ensure float

    // Convert total rotation vector from previous step to rotation matrix
    cv::Mat tot_rotation_matrix_prev(3, 3, CV_32F);
    cv::Rodrigues(tot_rotation_vector, tot_rotation_matrix_prev);
    // tot_rotation_matrix_prev.convertTo(tot_rotation_matrix_prev, CV_32F); // Explicitly ensure float


    // 1. Update Rotation: Compose the relative rotation with the previous total rotation
    cv::Mat tot_rotation_matrix_curr = rel_rotation_matrix * tot_rotation_matrix_prev;
    cv::Rodrigues(tot_rotation_matrix_curr, tot_rotation_vector); // Convert back to rotation vector
    // tot_rotation_vector.convertTo(tot_rotation_vector, CV_32F); // Explicitly ensure float

    // 2. Update Translation: Transform the relative translation and add to the previous total
    cv::Mat rotated_rel_translation = tot_rotation_matrix_prev * rel_translation_vector;
    tot_translation_vector += rotated_rel_translation;

    // std::cout << "Updated total translation vector: " << tot_translation_vector.t() << std::endl;
    // std::cout << "Updated total rotation vector: " << tot_rotation_vector.t() << std::endl;
}



void VisualOdometry::matchWithSIFT(const cv::Mat& img1, const cv::Mat& img2,
                   std::vector<cv::Point2f>& pts1, std::vector<cv::Point2f>& pts2) {
    int MAX_FEATURES = 1500, grid_rows = 3, grid_cols = 8, min_distance = 3;

    
    // int max_per_cell = 20;
    // 1. Create SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(MAX_FEATURES);

    // 2. Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2, filtered_keypoints1, filtered_keypoints2;
    cv::Mat descriptors1, descriptors2, filtered_descriptors1, filtered_descriptors2;
    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    int max_per_cell = (int)std::ceil((float)keypoints1.size() / (grid_rows * grid_cols));

    // filterKeypointsByGrid(img1, keypoints1, descriptors1, filtered_keypoints1, filtered_descriptors1, grid_rows, grid_cols, max_per_cell);
    // filterKeypointsByGrid(img2, keypoints2, descriptors2, filtered_keypoints2, filtered_descriptors2, grid_rows, grid_cols, max_per_cell);
    filterKeypointsByDistance(keypoints1, descriptors1, filtered_keypoints1, filtered_descriptors1, min_distance);
    filterKeypointsByDistance(keypoints2, descriptors2, filtered_keypoints2, filtered_descriptors2, min_distance);
    // filtered_descriptors1 = descriptors1;
    // filtered_descriptors2 = descriptors2;
    // filtered_keypoints1 = keypoints1;
    // filtered_keypoints2 = keypoints2;
    // 3. Match descriptors using BFMatcher (L2 norm for SIFT)
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(filtered_descriptors1, filtered_descriptors2, knn_matches, 2);

    // std::string message2 = "KNN matches found: " + std::to_string(knn_matches.size());
    // UnityLog(message2.c_str());
    

    // 4. Lowe's ratio test
    const float ratio_thresh = 0.95f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() >= 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    // std::cout << "Good matches found: " << good_matches.size() << std::endl;
    // std::string message3 = "Good matches found: " + std::to_string(good_matches.size());
    // UnityLog(message3.c_str());

    // 5. Extract matched points
    pts1.clear();
    pts2.clear();
    for (const auto& match : good_matches) {
        pts1.push_back(filtered_keypoints1[match.queryIdx].pt);
        pts2.push_back(filtered_keypoints2[match.trainIdx].pt);
    }

    std::cout << "filtered_keypoints found: " << pts1.size() << "\n";

    // // Optional: draw matches
    // cv::Mat img_matches;
    // cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);
    // cv::imshow("SIFT Matches", img_matches);
    // cv::waitKey(1); // for display
}

void VisualOdometry::filterKeypointsByDistance(std::vector<cv::KeyPoint>&  keypoints, cv::Mat& descriptors, 
                            std::vector<cv::KeyPoint>&  filtered_keypoints, cv::Mat& filtered_descriptors, double min_distance){
    
    for(size_t i = 0; i < keypoints.size(); ++i){
        bool too_close = false;
        for (size_t j = 0; j < filtered_keypoints.size(); ++j){
            double dx = keypoints[i].pt.x - filtered_keypoints[j].pt.x;
            double dy = keypoints[i].pt.y - filtered_keypoints[j].pt.y;
            if (std::sqrt(dx*dx + dy*dy) < min_distance){
                too_close = true;
                break;
            }
        }

        if(!too_close){
            filtered_keypoints.push_back(keypoints[i]);
            filtered_descriptors.push_back(descriptors.row((int)i));
        }
    }

}

void VisualOdometry::filterKeypointsByGrid(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors,
                            std::vector<cv::KeyPoint>& filtered_keypoints, cv::Mat& filtered_descriptors,
                            int grid_rows = 8,int grid_cols = 8, int max_per_cell = 5){

    int cell_width = image.cols / grid_cols;
    int cell_height = image.rows / grid_rows;

    filtered_keypoints.clear();
    filtered_descriptors = cv::Mat();

    for (int row = 0; row < grid_rows; ++row) {
        for (int col = 0; col < grid_cols; ++col) {
        // Collect keypoints in the current cell
        std::vector<std::pair<float, int>> strength_idx;

        for (size_t i = 0; i < keypoints.size(); ++i) {
            const auto& kp = keypoints[i];
            if (kp.pt.x >= col * cell_width && kp.pt.x < (col + 1) * cell_width &&
                kp.pt.y >= row * cell_height && kp.pt.y < (row + 1) * cell_height) {
                strength_idx.emplace_back(kp.response, i);
                }
            }

            // Sort by strength (response)
            std::sort(strength_idx.rbegin(), strength_idx.rend());

            // Take up to max_per_cell strongest keypoints
            for (int k = 0; k < std::min((int)strength_idx.size(), max_per_cell); ++k) {
                int idx = strength_idx[k].second;
                filtered_keypoints.push_back(keypoints[idx]);
                filtered_descriptors.push_back(descriptors.row(idx));
            }
        }
    }
}

// Detect ArUco markers and return map from ID to corners
// std::map<int, VisualOdometry::MarkerInfo> 
bool VisualOdometry::detectArucoMarkers(const cv::Mat& image, std::map<int, MarkerInfo>& detectedMarkers) {

    cv::aruco::ArucoDetector detector(dictionary, parameters);
    detector.detectMarkers(image, markerCorners, markerIds, rejectedCandidates);

    for (size_t i = 0; i < markerIds.size(); ++i) {
        detectedMarkers[markerIds[i]] = { markerIds[i], markerCorners[i] };
    }

    bool hasMarkers = !detectedMarkers.empty();

    return hasMarkers;
}

// Match corners based on shared marker IDs
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> VisualOdometry::matchMarkerCorners( const std::map<int, MarkerInfo>& leftMarkers, 
                                                                                    const std::map<int, MarkerInfo>& rightMarkers) 
{
    // std::vector<std::pair<cv::Point2f, cv::Point2f>> matchedPoints;
    std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>> matchedPoints;
    std::vector<cv::Point2f> leftCorners, rightCorners;

    for (const auto& [id, leftMarker] : leftMarkers) {
        auto it = rightMarkers.find(id);
        if (it != rightMarkers.end()) {
            const auto& rightMarker = it->second;
            // Match all 4 corners (assuming order is preserved)
            for (int i = 0; i < 4; ++i) {
                // matchedPoints.emplace_back(leftMarker.corners[i], rightMarker.corners[i]);
                leftCorners.push_back(leftMarker.corners[i]);
                rightCorners.push_back(rightMarker.corners[i]);
            }
        }
    }
    matchedPoints.first = leftCorners;
    matchedPoints.second = rightCorners;

    return matchedPoints;
}

void VisualOdometry::estimateMarkersPose(const cv::Mat& imageLeft, const cv::Mat& imageRight,
                                        std::map<int, MarkerInfo>& detectedLeftMarkers,
                                        std::map<int, MarkerInfo>& detectedRightMarkers,
                                        cv::Mat& rvec, cv::Mat& tvec) {


    auto matchedPoints = matchMarkerCorners(detectedLeftMarkers, detectedRightMarkers);

    // std::cout << "Matched ArUCo Points: " << matchedPoints.size() << "\n";

    // Estimate pose for each marker
    for (size_t i = 0; i < detectedRightMarkers.size(); ++i) {
        // Estimate pose of the marker
        // cv::aruco::estimatePoseSingleMarkers(corners[i], 0.05, K1_float, D1_float, rvec, tvec);

        std::vector<cv::Point2f> leftImagePoints = matchedPoints.first; // from ArUco detection
        std::vector<cv::Point2f> RightImagePoints = matchedPoints.second; 
        
        cv::solvePnPRansac(aruco_objectPoints, leftImagePoints, K1_float, D1_float, rvec, tvec);


    }
}


bool VisualOdometry::detectContourMarkers(const cv::Mat& image, std::vector<cv::Point2f>& contourPoints) {
    // Convert to HSV for better red detection
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<std::vector<cv::Point>> contoursOut;

    // Define lower and upper bounds for red in HSV (wraps around hue = 0)
    cv::Mat mask1, mask2;
    cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), mask1);   // lower reds
    cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), mask2); // upper reds

    cv::Mat redMask = mask1 | mask2;

    // Optional: Morphological operations to clean mask
    cv::morphologyEx(redMask, redMask, cv::MORPH_OPEN, cv::Mat(), cv::Point(-1,-1), 2);

    // Detect contours
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(redMask, contoursOut, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& pt : contoursOut[0]) {
        contourPoints.push_back(cv::Point2f(pt.x, pt.y));
    }

    return !contoursOut.empty();
}

