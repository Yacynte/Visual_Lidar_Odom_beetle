#include <iostream>
#include <string>
#include <vector>
#include <filesystem> //  C++17 and above
#include "vo_process.h"

#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
const int PORT = 8080;


#define optimal_distance 1

namespace fs = std::filesystem;

// Initialize a vector to hold the relative poses



int main( int c, char** argv) {

    int server_fd, client_fd;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    // Create socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == 0) {
        perror("Socket failed");
        exit(EXIT_FAILURE);
    }

    // Bind to port
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY; // 0.0.0.0
    address.sin_port = htons(PORT);

    int opt = 1;
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    // Bind the socket to the address and port
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // Listen for connections
    listen(server_fd, 3);
    std::cout << "Waiting for connection on port " << PORT << "...\n";

    // Accept one client
    client_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t *)&addrlen);
    if (client_fd < 0) {
        perror("Accept failed");
        exit(EXIT_FAILURE);
    }
    std::cout << "Client connected!\n";



    // Check if the correct number of arguments is provided
    // if (c < 2 || c > 4) {
    //     std::cerr << "Algorithm takes 2 or 3 arguments " << std::endl;
    //     return 0;
    // }

    // int PORT;
    // std::istringstream(argv[1]) >> PORT;

    // initial time
    auto start = std::chrono::high_resolution_clock::now();

    std::string left_folder = "/home/divan/beetle/VisualOdometry/Data/LeftImages/sync_Left_" ;// "Data/LeftImages/sync_Left_";
    std::string right_folder = "/home/divan/beetle/VisualOdometry/Data/RightImages/sync_Right_"; // "Data/RightImages/sync_Right_";

    // Create an instance of your VisualOdometry class
    VisualOdometry vo;


    if (left_folder.empty() || right_folder.empty()) {
        std::cout << "No images found in the specified folders." << std::endl;
        return 0;
    }
    cv::Mat R_global = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat t_global = cv::Mat::zeros(3, 1, CV_64F);

    // Initialize a vector to hold the relative poses
    
    std::cout << "Starting to at cliend: " << client_fd << std::endl;
    while (true){

        // cv::Mat leftImage_pre, leftImage_cur, rightImage_pre, rightImage_cur;
        // cv::Mat rotation_vector, translation_vector;
        // cv::Mat tot_rotation_vector = cv::Mat::zeros(3, 1, CV_32F);
        // cv::Mat tot_translation_vector = cv::Mat::zeros(3, 1, CV_32F);

        // Compute odometry asychronously
        auto compute_rel_pose = [&](std::string pre_image, std::string current_image, RelativePose* rel_pose, 
                                CountourPose* contour_pose, int rel_index) {
            cv::Mat right_cur, left_cur, right_pre, left_pre, left_pre_color;

            try {
                // std::cout << "Loading: " << left_folder+pre_image+".png" << std::endl;
                // std::cout << "Loading: " << right_folder+pre_image+".png" << std::endl;

                left_pre = cv::imread(left_folder + pre_image+".png", cv::IMREAD_GRAYSCALE);
                right_pre = cv::imread(right_folder + pre_image+".png", cv::IMREAD_GRAYSCALE);

                left_cur = cv::imread(left_folder + current_image+".png", cv::IMREAD_GRAYSCALE);
                right_cur = cv::imread(right_folder + current_image+".png", cv::IMREAD_GRAYSCALE);

                left_pre_color = cv::imread(left_folder + current_image+".png");
                // right_pre = cv::imread(right_folder+pre_image+".png");

                // left_cur = cv::imread(left_folder+current_image+".png");
                // right_cur = cv::imread(right_folder+current_image+".png");
            }
            catch (const std::exception& e) {
                std::cerr << "Error loading images: " << e.what() << std::endl;
                return;
            }

            // std::cout << "Previous image timestamp: "<< pre_image << std::endl;
            cv::Mat rp;
            cv::Mat rel_R, rel_t;
            // std::vector<cv::Point3f> contour_poses;
            try{
                if (vo.StereoOdometry(left_pre_color, left_pre, left_cur, right_pre, right_cur, rel_R, rel_t, contour_pose)) {
                    cv::Rodrigues(rel_R, rp);
                    // std::cout << "Calculated relative pose: " << rel_t << std::endl;
                    rel_pose->valid = true;
                    if(contour_pose->valid) {
                        contour_pose->position = rel_index; // Store the position index
                    }
                    
                }
                else {
                    std::cerr << "StereoOdometry failed for images: " << pre_image << " and " << current_image << std::endl;
                    rel_pose->valid = false;
                }
                rel_pose->R = rp;
                rel_pose->t = rel_t;

                std::map<int, MarkerInfo> detectLeftMarkers, detectRightMarkers;
                if (vo.detectArucoMarkers(left_cur, detectLeftMarkers) && vo.detectArucoMarkers(right_cur, detectRightMarkers) && !contour_pose->valid) {
                    // std::cout << "Detected markers in frame pair: " << i << " and " << i + step << std::endl;
                    // Estimate pose of the markers
                    vo.estimateMarkersPose(left_cur, right_cur, detectLeftMarkers, detectRightMarkers, contour_pose->R, contour_pose->t);
                    contour_pose->position = rel_index; // Store the position index
                    contour_pose->valid = true;
                    
                }
                else {
                    contour_pose = nullptr; // No markers detected, set to nullptr
                }
            }
            catch (const cv::Exception& e) {
                std::cerr << "Error in StereoOdometry: " << e.what() << std::endl;
                return;
            }
        
        };

        unsigned int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::cout << "Number of threads available: " << num_threads << std::endl;
        char buffer[1024];  // buffer for incoming data
        int bytes_received = recv(client_fd, buffer, sizeof(buffer) - 1, 0);
        std::cout << "Bytes received: " << bytes_received << std::endl;
        std::string jsonStr;

        json j;
        if (bytes_received > 0) {
            buffer[bytes_received] = '\0';  // null-terminate to safely use as C-string
            jsonStr = std::string(buffer);  // convert to std::string

            try {
                j = json::parse(jsonStr);
                std::cout << "Received image timestamps:\n"; // << j.dump(4) << std::endl;
                // processJsonObject(j);
            } catch (json::parse_error& e) {
                std::cerr << "JSON parse error: " << e.what() << std::endl;
            }
        }

        int rel_index = 0;
        bool first_index = true;
        std::string pre_timestamp;
        std::vector<RelativePose> rel_poses;
        std::vector<CountourPose> contour_poses;
        rel_poses.resize(int(j.size())-1);  // Resize to hold all relative poses
        contour_poses.resize(int(j.size())-1);  // Resize to hold all contour poses except the first one
        // CountourPose contour_pose;
        for (const auto& [timestamp, value] : j.items()) {
            bool detected = value.get<bool>();
            // std::cout << "  " << timestamp << " => " << (detected ? "Detected" : "Not Detected") << "\n";
            if (first_index) {
                pre_timestamp = timestamp;  // Initialize the previous timestamp
                first_index = false;
                continue;  // Skip the first iteration
            }
            else{
                // RelativePose rel_pose;
                // Compute the relative pose between the previous and current timestamp
                threads.emplace_back(compute_rel_pose, pre_timestamp, timestamp, &rel_poses[rel_index], &contour_poses[rel_index], rel_index);
                // Limit concurrent threads
                if (threads.size() >= num_threads) {
                    for (auto& t : threads) t.join();
                    threads.clear();
                }
                // rel_poses.push_back(rel_pose);
                if (!(contour_poses[rel_index].R.empty() && contour_poses[rel_index].t.empty())) {
                    // contour_pose.R, contour_pose.t = findContoursAndPose(timestamp);
                    // contour_poses[irr].position = irr;
                    // contour_poses.push_back(CountourPose());
                    std::cout << "Contour detected at timestamp: " << timestamp << std::endl;
                }
            }
            rel_index++;
            pre_timestamp = timestamp;
        }
        // Join remaining threads
        for (auto& t : threads) t.join();
        // std::cout << "Third poses: " << rel_poses[3].R << std::endl;

        json pose_array = json::array();  // holds all pose entries
        json contour_array = json::array();
        json full_json;
        json countour_entry;
        if (!contour_poses.empty()) {
            for (const auto& cp : contour_poses) {
                if (!cp.valid) continue;  // Skip invalid poses
                countour_entry["contour_position"] = {
                    {"x", cp.t.at<double>(0)},
                    {"y", cp.t.at<double>(1)},
                    {"z", cp.t.at<double>(2)}
                };
                countour_entry["contour_rotation"] = {
                    {"x", cp.R.at<double>(0)},
                    {"y", cp.R.at<double>(1)},
                    {"z", cp.R.at<double>(2)}
                };
                countour_entry["contour_position_index"] = cp.position;
            }
            if (!countour_entry.empty()) {
                contour_array.push_back(countour_entry);  // add to array  
            } 
            
        }
        full_json["contour_poses"] = contour_array; // wrap contour poses in an json object

        // Add relative poses to the JSON array

        for (const auto& rp : rel_poses) {
            if (!rp.valid) continue;  // Skip invalid poses
            json pose_entry;
            pose_entry["drone_position"] = {
                {"x", rp.t.at<double>(0)},
                {"y", rp.t.at<double>(1)},
                {"z", rp.t.at<double>(2)}
            };
            pose_entry["drone_rotation"] = {
                {"x", rp.R.at<double>(0)},
                {"y", rp.R.at<double>(1)},
                {"z", rp.R.at<double>(2)}
            };

            pose_array.push_back(pose_entry);  // add to array
        }
        // Wrap the array in a JSON object to send as a response
        
        full_json["poses"] = pose_array;  // optionally wrap in another object
        std::string response = full_json.dump();
        send(client_fd, response.c_str(), response.size(), 0);

        std::cout<< "Sent poses to GUI with size: " << response.size()  << std::endl;
    }

    std::cout << "Finished processing all image pairs." << std::endl;

    return 0;
}