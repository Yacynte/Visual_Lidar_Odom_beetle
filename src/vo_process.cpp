#include "vo_process.h"
#include "visualodom.h"



bool VisualOdometry::StereoOdometry(cv::Mat leftImage_pre_, cv::Mat leftImage_cur_, cv::Mat rightImage_pre_, cv::Mat rightImage_cur_, 
                                    cv::Mat& rotation_vector, cv::Mat& translation_vector) {

// bool VisualOdometry::StereoOdometry(cv::Mat leftImage_pre, cv::Mat leftImage_cur, cv::Mat rightImage_pre, cv::Mat rightImage_cur, 
//                                             cv::Mat& rotation_vector, cv::Mat& translation_vector) {

    if (leftImage_pre_.empty() || leftImage_cur_.empty() || rightImage_pre_.empty() || rightImage_cur_.empty()) {
        std::cerr << "One or all images are Empty!" << std::endl;
        UnityLog("One or all images are Empty!");
        // Initialize rotation and translation vectors to -1
        cv::Mat rotation_vector = cv::Mat(3, 1, CV_32F, cv::Scalar(-1));
        cv::Mat translation_vector =cv::Mat(3, 1, CV_32F, cv::Scalar(-1));
    }
    // Half the image size
    int new_width = leftImage_pre_.cols / 2;
    int new_height = leftImage_pre_.rows / 2;
    cv::Size new_size(new_width, new_height);

    // Create a matrix to store the resized image
    cv::Mat leftImage_pre, leftImage_cur, rightImage_pre, rightImage_cur;
    // Resize the images using high-quality interpolation
    cv::resize(leftImage_pre_, leftImage_pre, new_size, 0, 0, cv::INTER_AREA);
    cv::resize(leftImage_cur_, leftImage_cur, new_size, 0, 0, cv::INTER_AREA);
    cv::resize(rightImage_pre_, rightImage_pre, new_size, 0, 0, cv::INTER_AREA);
    cv::resize(rightImage_cur_, rightImage_cur, new_size, 0, 0, cv::INTER_AREA);


    // Step 1: Rectify images
    std::thread t1([&]() { RectifyImage(leftImage_pre, rightImage_pre); });
    std::thread t2([&]() { RectifyImage(leftImage_cur, rightImage_cur); });

    t1.join();
    t2.join();
    
    //Compute depth map of previous Image pair
    auto depth_map = computeDepth(leftImage_pre, rightImage_pre);
    
    // Vectors to hold the matched points
    // std::cout<<"Init 2d points"<<std::endl;
    std::vector<cv::Point2f> pts_prev_L, pts_cur_L;

    // Call the feature matching function
    // std::cout<<"feature matching"<<std::endl;
    // feature_matching(leftImage_pre, leftImage_cur, pts_prev_L, pts_cur_L);
    matchWithSIFT(leftImage_pre, leftImage_cur, pts_prev_L, pts_cur_L);
    int num_matching = pts_prev_L.size();
    // std::string message = "Number of matching points: " + std::to_string(num_matching);
    // UnityLog(message.c_str());
    // std::cout<<"Motion Estimation"<<std::endl;
    bool state = motionEstimation(pts_prev_L, pts_cur_L, depth_map, rotation_vector, translation_vector);
    
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
                                   std::vector<cv::Point3f>& points_3D, std::vector<size_t>& outliers, float max_depth) {
    points_3D.clear();
    outliers.clear();

    for (size_t i = 0; i < image_points.size(); ++i) {
        float u = image_points[i].x;
        float v = image_points[i].y;
        float z = depth.at<float>(static_cast<int>(v), static_cast<int>(u));

        // Ignore points with invalid depth
        if (z > max_depth) {
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

bool VisualOdometry::motionEstimation(const std::vector<cv::Point2f>& image1_points, const std::vector<cv::Point2f>& image2_points,
    const cv::Mat& depth, cv::Mat& rvec, cv::Mat& translation_vector, float max_depth) {
    if (image1_points.size() != image2_points.size()) {
        std::cerr << "Error: Point sets must have the same size." << std::endl;
        UnityLog("Error: Point sets must have the same size.");
        return false;
    }

    // Step 1: Reconstruct 3D points
    std::vector<cv::Point3f> points_3D;
    std::vector<size_t> outliers;
    reconstruct3D(image1_points, depth, points_3D, outliers, max_depth);

    // Step 2: Use an unordered_set for fast outlier lookup
    std::unordered_set<size_t> outlier_set(outliers.begin(), outliers.end());

    // Filter points, reserving space to avoid multiple reallocations
    std::vector<cv::Point2f> filtered_image2_points; //filtered_image1_points
    std::vector<cv::Point3f> filtered_points_3D;
    size_t num_points = image1_points.size();
    // filtered_image1_points.reserve(image1_points.size());
    filtered_image2_points.reserve(image2_points.size());
    filtered_points_3D.reserve(points_3D.size());

    // UnityLog("length of image1_points: ");
    // UnityLog(std::to_string(image1_points.size()).c_str());
    // UnityLog("length of image2_points: ");
    // UnityLog(std::to_string(image2_points.size()).c_str());
    // UnityLog("length of points_3D: ");
    // UnityLog(std::to_string(points_3D.size()).c_str());
    // UnityLog("length of outliers: ");
    // UnityLog(std::to_string(outliers.size()).c_str());
    // std::cout << "length of image1_points: " << image1_points.size() << std::endl;

    for (size_t i = 0; i < num_points; ++i) {
        if (outlier_set.find(i) == outlier_set.end()) {
            // filtered_image1_points.push_back(image1_points[i]);
            filtered_image2_points.push_back(image2_points[i]);
            filtered_points_3D.push_back(points_3D[i]);
        }
    }

    

    // Step 3: Solve PnP with RANSAC
    cv::Mat points_3D_mat(filtered_points_3D), filtered_image2_points_mat(filtered_image2_points);
    cv::Mat(points_3D).convertTo(points_3D_mat, CV_32F);
    cv::Mat(filtered_image2_points).convertTo(filtered_image2_points_mat, CV_32F);

    // std::vector<int> inliers;
    // std::string message = "Filtered image points size: " + std::to_string(filtered_image2_points_mat.size[0]);
    // UnityLog(message.c_str());
    // std::string message2 = "Filtered 3D points size: " + std::to_string(points_3D_mat.size[0]);
    // UnityLog(message2.c_str());
    // std::cout << "Filtered image points size: " << filtered_image2_points_mat.size[0] << std::endl;
    // std::cout << "Filtered 3D points size: " << points_3D_mat.size[0] << std::endl;
    std::vector<int> inliers ;
    bool success = cv::solvePnPRansac(points_3D_mat, filtered_image2_points_mat, K1_float, D1_float,
                                    rvec, translation_vector, false, 100, 8.0, 0.99, inliers, cv::SOLVEPNP_ITERATIVE);
    
    // std::string message3 = "Inliers size: " + std::to_string(inliers.size());
    // UnityLog(message3.c_str());
    // std::cout << "Inliers size: " << inliers.size() << std::endl;
    
    return success;
    
}

cv::Mat VisualOdometry::computeDisparity(const cv::Mat& left, const cv::Mat& right) {
    // Create StereoSGBM object
    auto stereo = cv::StereoSGBM::create(
        0,            // Min disparity
        4 * 16,       // Reduced number of disparities for faster computation
        7,            // Smaller block size for faster computation
        8 * 7 * 7,    // P1 with block size of 5
        32 * 7 * 7,   // P2 with block size of 5
        31,           // Pre-filter cap
        10,           // Uniqueness ratio
        0,            // Speckle window size
        0,            // Speckle range
        cv::StereoSGBM::MODE_SGBM // Mode
    );

    // Compute disparity
    cv::Mat disparity(left.size(), CV_16S);
    stereo->compute(left, right, disparity);

    // Convert to float for bilateral filtering
    cv::Mat disparity_float;
    disparity.convertTo(disparity_float, CV_32F, 1.0 / 16.0);  // scale back to original disparity


    // Normalize disparity for visualization (optional)
    // cv::Mat disparity_normalized;
    // cv::normalize(disparity, disparity_normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat disparity_filtered_float;
    cv::bilateralFilter(disparity_float, disparity_filtered_float, 9, 75, 75);

    // if (!prev_disparity.empty()) {
    //     for (int y = 0; y < disparity_filtered_float.rows; ++y) {
    //         for (int x = 0; x < disparity_filtered_float.cols; ++x) {
    //             float curr = static_cast<float>(disparity_filtered_float.at<short>(y, x)) / 16.0;
    //             float prev = static_cast<float>(prev_disparity.at<short>(y, x)) / 16.0;
    //             if (std::abs(curr - prev) > threshold) {
    //                 disparity_filtered_float.at<short>(y, x) = prev_disparity.at<short>(y, x); // Or use neighborhood average
    //             }
    //         }
    //     }
    // }
    // prev_disparity = disparity_filtered_float.clone();


    cv::Mat disparity_filtered;
    disparity_filtered_float.convertTo(disparity_filtered, CV_16S, 16.0);  // back to fixed-point disparity

    return disparity_filtered;
}
// Compute depth from disparity
cv::Mat VisualOdometry::computeDepth(const cv::Mat& left, const cv::Mat& right) {
    auto disparity = computeDisparity(left, right);  // Assuming this is a pre-defined function

    double focal_length_ = K1.at<double>(0, 0);
    double baseline_ = T.at<double>(0);

    cv::Mat depth(disparity.size(), CV_32F);

    // Define a parallel loop for processing each pixel
    cv::parallel_for_(cv::Range(0, disparity.rows), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y) {
            for (int x = 0; x < disparity.cols; ++x) {
                float d = static_cast<float>(disparity.at<short>(y, x)) / 16.0;  // SGBM divides disparity by 16
                if (d > 0) { // Avoid division by zero
                    depth.at<float>(y, x) = (focal_length_ * baseline_) / d;
                } else {
                    depth.at<float>(y, x) = 0.0f; // Invalid depth
                }
            }
        }
    });

    return depth;
}


// Feature matching function
void VisualOdometry::feature_matching(const cv::Mat& left_prev, const cv::Mat& left_cur,
    std::vector<cv::Point2f>& pts_prev_L, std::vector<cv::Point2f>& pts_cur_L) {
    
    int MAX_FEATURES = 3000, grid_rows = 3, grid_cols = 8, min_distance = 0;
    
    cv::Ptr<cv::ORB> orb1 = cv::ORB::create(MAX_FEATURES);
    cv::Ptr<cv::ORB> orb2 = cv::ORB::create(MAX_FEATURES);
    
    // cv::Ptr<cv::SURF> orb1 =cv::SURF::create(MAX_FEATURES);

    std::vector<cv::KeyPoint> keypoints1, keypoints2, filtered_keypoints1, filtered_keypoints2;
    cv::Mat descriptors1, descriptors2, filtered_descriptors1, filtered_descriptors2;

    // Feature detection + description
    orb1->detectAndCompute(left_prev, cv::noArray(), keypoints1, descriptors1);
    orb2->detectAndCompute(left_cur, cv::noArray(), keypoints2, descriptors2);

    filterKeypointsByDistance(keypoints1, descriptors1, filtered_keypoints1, filtered_descriptors1, min_distance);
    filterKeypointsByDistance(keypoints2, descriptors2, filtered_keypoints2, filtered_descriptors2, min_distance);


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
    int MAX_FEATURES = 3000, grid_rows = 3, grid_cols = 8, min_distance = 3;

    int max_per_cell = (int)std::ceil((float)MAX_FEATURES / (grid_rows * grid_cols));
    // int max_per_cell = 20;
    // 1. Create SIFT detector
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(MAX_FEATURES);

    // 2. Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints1, keypoints2, filtered_keypoints1, filtered_keypoints2;
    cv::Mat descriptors1, descriptors2, filtered_descriptors1, filtered_descriptors2;
    sift->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    // filterKeypointsByGrid(img1, keypoints1, descriptors1, filtered_keypoints1, filtered_descriptors1, grid_rows, grid_cols, max_per_cell);
    // filterKeypointsByGrid(img2, keypoints2, descriptors2, filtered_keypoints2, filtered_descriptors2, grid_rows, grid_cols, max_per_cell);
    filterKeypointsByDistance(keypoints1, descriptors1, filtered_keypoints1, filtered_descriptors1, min_distance);
    filterKeypointsByDistance(keypoints2, descriptors2, filtered_keypoints2, filtered_descriptors2, min_distance);

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
std::map<int, VisualOdometry::MarkerInfo> VisualOdometry::detectArucoMarkers(const cv::Mat& image) {

    cv::aruco::ArucoDetector detector(dictionary, parameters);
    detector.detectMarkers(image, markerCorners, markerIds, rejectedCandidates);

    for (size_t i = 0; i < markerIds.size(); ++i) {
        detectedMarkers[markerIds[i]] = { markerIds[i], markerCorners[i] };
    }

    return detectedMarkers;
}

// Match corners based on shared marker IDs
std::vector<std::pair<cv::Point2f, cv::Point2f>> VisualOdometry::matchMarkerCorners( const std::map<int, MarkerInfo>& leftMarkers, 
                                                                                    const std::map<int, MarkerInfo>& rightMarkers) 
{
    std::vector<std::pair<cv::Point2f, cv::Point2f>> matchedPoints;

    for (const auto& [id, leftMarker] : leftMarkers) {
        auto it = rightMarkers.find(id);
        if (it != rightMarkers.end()) {
            const auto& rightMarker = it->second;
            // Match all 4 corners (assuming order is preserved)
            for (int i = 0; i < 4; ++i) {
                matchedPoints.emplace_back(leftMarker.corners[i], rightMarker.corners[i]);
            }
        }
    }

    return matchedPoints;
}

void VisualOdometry::estimateMarkersPose(const cv::Mat& imageLeft, const cv::Mat& imageRight,
                                         const std::vector<std::vector<cv::Point2f>>& corners,
                                         const std::vector<int>& ids, cv::Mat& rvec, cv::Mat& tvec) {
    if (ids.empty() || corners.empty()) {
        std::cerr << "No markers " << std::endl;
        return;
    }
    // Detect markers in both images
    auto leftMarkers = detectArucoMarkers(imageLeft);
    auto rightMarkers = detectArucoMarkers(imageRight);

    auto matchedPoints = matchMarkerCorners(leftMarkers, rightMarkers);

    std::cout << "Matched ARCO Points: " << matchedPoints.size() << "\n";

    // Estimate pose for each marker
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0) continue; // Skip invalid IDs
        // Estimate pose of the marker
        cv::aruco::estimatePoseSingleMarkers(corners[i], 0.05, K1_float, D1_float, rvec, tvec);

        std::vector<cv::Point2f> imagePoints = corners[i]; // from ArUco detection

        
        cv::solvePnPRansac(aruco_objectPoints, corners[i], K1_float, D1_float, rvec, tvec);

    }
}