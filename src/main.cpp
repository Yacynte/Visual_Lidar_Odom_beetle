#include <iostream>
#include <string>
#include <vector>
#include <filesystem> //  C++17 and above
#include "vo_process.h"
#define optimal_distance 1

namespace fs = std::filesystem;

// Initialize a vector to hold the relative poses



int main( int c, char** argv) {
    // Check if the correct number of arguments is provided
    if (c < 2 || c > 4) {
        std::cerr << "Algorithm takes 2 or 3 arguments " << std::endl;
        return 0;
    }

    int total_frames;
    int old_frames = 0;
    std::istringstream(argv[1]) >> total_frames;
    float speed = 1.0f, frame_rate = 1;
    if (c > 2) {
        // std::cout << "Skip value: " << argv[2] << std::endl;
        std::istringstream(argv[2]) >> speed;
        std::istringstream(argv[3]) >> frame_rate;
    }
    float dist = speed/frame_rate;
    int skip = std::max(int(optimal_distance /dist), 1);
    std::cout << "Skip value: " << skip << std::endl;
    // std::cout<< "dist: " << dist<< ", opt dist: "<< optimal_distance <<std::endl;
    // std::cout << int(optimal_distance /dist) << std::endl;

    // initial time
    auto start = std::chrono::high_resolution_clock::now();

    std::string left_folder = "../images/left_images/data";
    std::string right_folder = "../images/right_images/data";

    // Create an instance of your VisualOdometry class
    VisualOdometry vo;

    // Vectors to store the paths of the left and right images
    std::vector<fs::path> left_image_paths;
    std::vector<fs::path> right_image_paths;

    // Load image paths from the left folder
    if (fs::exists(left_folder) && fs::is_directory(left_folder)) {
        for (const auto& entry : fs::directory_iterator(left_folder)) {
            if (entry.is_regular_file() && (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg")) {
                left_image_paths.push_back(entry.path());
            }
        }
        std::sort(left_image_paths.begin(), left_image_paths.end()); // Sort to ensure correct order
    } else {
        std::cerr << "Error: Left image folder does not exist or is not a directory." << std::endl;
        return 1;
    }

    // Load image paths from the right folder
    if (fs::exists(right_folder) && fs::is_directory(right_folder)) {
        for (const auto& entry : fs::directory_iterator(right_folder)) {
            if (entry.is_regular_file() && (entry.path().extension() == ".png" || entry.path().extension() == ".jpg" || entry.path().extension() == ".jpeg")) {
                right_image_paths.push_back(entry.path());
            }
        }
        std::sort(right_image_paths.begin(), right_image_paths.end()); // Sort to ensure correct order
    } else {
        std::cerr << "Error: Right image folder does not exist or is not a directory." << std::endl;
        return 1;
    }

    // Check if the number of left and right images matches
    if (left_image_paths.size() != right_image_paths.size()) {
        std::cerr << "Error: Number of left and right images does not match." << std::endl;
        return 1;
    }

    if (left_image_paths.empty() || right_image_paths.empty()) {
        std::cout << "No images found in the specified folders." << std::endl;
        return 0;
    }
    cv::Mat R_global = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_global = cv::Mat::zeros(3, 1, CV_64F);

    // Initialize a vector to hold the relative poses
    // struct RelativePose {
    //     cv::Mat R, t;
    //     bool valid = false;
    // };
    std::vector<RelativePose> rel_poses;

    int last_img_set = 0;
    // std::cout << "before while loop "<< std::endl;
    std::cout << "Number of images : " << left_image_paths.size() << std::endl;

    while (old_frames <= left_image_paths.size() - skip -1){

        cv::Mat leftImage_pre, leftImage_cur, rightImage_pre, rightImage_cur;
        cv::Mat rotation_vector, translation_vector;
        cv::Mat tot_rotation_vector = cv::Mat::zeros(3, 1, CV_32F);
        cv::Mat tot_translation_vector = cv::Mat::zeros(3, 1, CV_32F);

        
        // int total_frames = 100; // Must be even
        // int total_frames = left_image_paths.size();
        if (left_image_paths.size() - old_frames < total_frames) {
            total_frames = left_image_paths.size() - old_frames; // Ensure even number of frames
            std::cout << "Adjusted total frames to: " << total_frames << std::endl;
        }
        
        // std::cout<< "resize rel_pose \n";
        rel_poses.resize(int((old_frames + total_frames)/skip)+1); // For each i -> i+skip
        // std::cout<<"rel_pose resized \n";

        auto compute_rel_pose = [&](int i, int step) {
            cv::Mat right_cur, left_cur, left_cur_color; bool isLast = false;
            cv::Mat left_pre = cv::imread(left_image_paths[i].string(), cv::IMREAD_GRAYSCALE);
            cv::Mat right_pre = cv::imread(right_image_paths[i].string(), cv::IMREAD_GRAYSCALE);

            if (left_image_paths.size() - i <= skip){
                int last = left_image_paths.size() -1;
                left_cur = cv::imread(left_image_paths[last].string(), cv::IMREAD_GRAYSCALE);
                right_cur = cv::imread(right_image_paths[last].string(), cv::IMREAD_GRAYSCALE);
                left_cur_color = cv::imread(left_image_paths[last].string());
                isLast = true;
            }
            else{
                left_cur = cv::imread(left_image_paths[i + step].string(), cv::IMREAD_GRAYSCALE);
                right_cur = cv::imread(right_image_paths[i + step].string(), cv::IMREAD_GRAYSCALE);
                left_cur_color = cv::imread(left_image_paths[i + step].string());
            }

        
            RelativePose rp;
            cv::Mat r;
            CountourPose contour_pose;
            if (!left_pre.empty() && !left_cur.empty() && !right_pre.empty() && !right_cur.empty()) {
                
                if (vo.StereoOdometry(left_cur_color, left_pre, left_cur, right_pre, right_cur, r, rp.t, &contour_pose)) {
                    cv::Rodrigues(r, rp.R);
                    rel_poses[int(i/skip)] = rp;
                    rel_poses[int(i/skip)].valid = true;
                    std::cout << "Calculated relative pose: " << rp.t << std::endl;
                    // std::cout << "Processing frame pair: " << i << " and " << i + step << std::endl;
                    // std::cout << "rel pose for frame "<< i << " to frame "<<  i + skip <<": " << rp.t.t() <<std::endl;
                }
                else{
                    std::cerr << "StereoOdometry failed for images: " << i << " and " << i + skip << std::endl;
                    rel_poses[int(i/skip)].valid = false;
                }
                std::map<int, MarkerInfo> detectLeftMarkers, detectRightMarkers;
                RelativePose marker_pose;
                if (vo.detectArucoMarkers(left_cur, detectLeftMarkers) && vo.detectArucoMarkers(right_cur, detectRightMarkers) && !contour_pose.valid) {
                    // std::cout << "Detected markers in frame pair: " << i << " and " << i + step << std::endl;
                    // Estimate pose of the markers
                    vo.estimateMarkersPose(left_cur, right_cur, detectLeftMarkers, detectRightMarkers, contour_pose.R, contour_pose.t);
                    contour_pose.position = i; // Store the position index
                    contour_pose.valid = true;
                    
                }
                if (contour_pose.valid){
                    std::cout << "Contour pose valid for frame pair: " << i << " and " << i + step << std::endl;
                    std::cout << "Contour pose translation: " << contour_pose.t << std::endl;
                }
            }
        };

        unsigned int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        // std::cout << "Total loop: " <<old_frames + total_frames<<std::endl;
        // std::cout << " Old frames: " << old_frames <<std::endl;
        // for (int i = std::max(0, old_frames); i < old_frames + total_frames - skip; i+= skip) {
        for (int i = 0; i < total_frames; i += skip) {
            compute_rel_pose(i, skip);
        //     threads.emplace_back(compute_rel_pose, i, skip);
        //     // std::cout << "Processing frame pair: " << i << " and " << i + skip << std::endl;

        //     // Limit concurrent threads
        //     if (threads.size() >= num_threads) {
        //         for (auto& t : threads) t.join();
        //         threads.clear();
        //     }
        //     last_img_set = i + skip;
        }
        // if ( left_image_paths.size() - last_img_set <= skip) threads.emplace_back(compute_rel_pose, last_img_set, skip);

        // // Join remaining threads
        // for (auto& t : threads) t.join();
        // std::cout << "Joined threads \n";
        // R_global = cv::Mat::eye(3, 3, CV_64F);
        // t_global = cv::Mat::zeros(3, 1, CV_64F);

        for (int i = std::max(0, int(old_frames/skip)); i <= rel_poses.size()-1; ++i) {
            if (!rel_poses[i].valid) continue;

            // t' = t + R * t_i
            t_global += R_global * rel_poses[i].t;
            // R' = R * R_i
            R_global = R_global * rel_poses[i].R;
            
            // std::cout << "Updating pose for frame pair: " << i << " and " << i + 1 << std::endl;
        }

        old_frames = last_img_set;
    }

    std::cout << "  Translation: " << t_global.t() << "\n";
    std::cout << "  Rotation:\n" << R_global << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;
    

    // // Process images in pairs
    // for (size_t i = 0; i < left_image_paths.size() - 1; i+=2) {
    //     // Load the current and previous left images
    //     leftImage_pre = cv::imread(left_image_paths[i].string(), cv::IMREAD_GRAYSCALE);
    //     leftImage_cur = cv::imread(left_image_paths[i + 2].string(), cv::IMREAD_GRAYSCALE);

    //     // Load the current and previous right images (assuming corresponding names)
    //     rightImage_pre = cv::imread(right_image_paths[i].string(), cv::IMREAD_GRAYSCALE);
    //     rightImage_cur = cv::imread(right_image_paths[i + 2].string(), cv::IMREAD_GRAYSCALE);

    //     // Check if images were loaded successfully
    //     if (leftImage_pre.empty() || leftImage_cur.empty() || rightImage_pre.empty() || rightImage_cur.empty()) {
    //         std::cerr << "Error: Could not load one or more images." << std::endl;
    //         continue; // Skip to the next pair
    //     }

    //     // Call your stereo visual odometry function
    //     if (vo.StereoOdometry(leftImage_pre, leftImage_cur, rightImage_pre, rightImage_cur, rotation_vector, translation_vector)) {
    //         vo.updatePose(tot_translation_vector, tot_rotation_vector, translation_vector, rotation_vector);
    //         std::cout << "Frame " << i  << " to " << i + 2 << ": " << std::endl;
    //         std::cout << "  Rotation Vector: " << tot_rotation_vector.t() << std::endl;
    //         std::cout << "  Translation Vector: " << tot_translation_vector.t() << std::endl;
    //         // std::cout << "Frame " << i + 1 << " to " << i + 2 << ": " << std::endl;
    //         // std::cout << "  Rotation Vector: " << rotation_vector << std::endl;
    //         // std::cout << "  Translation Vector: " << translation_vector << std::endl;
    //         // You can further process or store the rotation and translation vectors here
    //     } else {
    //         std::cout << "Stereo odometry failed for frames " << i + 1 << " and " << i + 2 << std::endl;
    //     }

    //     // Optionally add a delay to visualize the process
    //     // cv::waitKey(100);
    // }

    std::cout << "Finished processing all image pairs." << std::endl;

    return 0;
}