#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <zlib.h>
#include <sys/socket.h> // recv

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include "vo_process.h"

class PointCloudReceiver {
public:
    PointCloudReceiver(int socket) : sock(socket) {};
    cv::Mat receiveLidarData(int sock);

private:
    bool recvAll(int sock, void* buffer, size_t length);
    uint64_t unpack_u64_be(const uint8_t* data);
    double unpack_double_be(const uint8_t* data);
    bool decompressData(const std::vector<uchar>& compressed, std::vector<uchar>& decompressed);
    
    int sock;

};

class LidarOdometry: public VisualOdometry {

public:
    bool lidarOdom(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2, RelativePose* rel_pos);
private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr loadKittiBin(const std::string& path);
    bool processLidarOdometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
                             Eigen::Matrix4f& transformation);
    // std::string lidar_path = "/home/divan/beetle/VisualOdometry/Data/lidar/" ;
};