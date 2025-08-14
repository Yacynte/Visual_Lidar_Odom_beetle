#include "receive_cloud.h"
#include <iostream>

// Helper to receive exactly n bytes
bool PointCloudReceiver::recvAll(int sock, void* buffer, size_t length) {
    char* buf = reinterpret_cast<char*>(buffer);
    size_t received = 0;
    while (received < length) {
        ssize_t ret = recv(sock, buf + received, length - received, 0);
        if (ret <= 0) return false; // error or disconnect
        received += ret;
    }
    return true;
}

// Unpack big-endian uint64_t
uint64_t PointCloudReceiver::unpack_u64_be(const uint8_t* data) {
    uint64_t val = 0;
    for (int i = 0; i < 8; i++) {
        val = (val << 8) | data[i];
    }
    return val;
}

// Unpack big-endian double (IEEE 754)
double PointCloudReceiver::unpack_double_be(const uint8_t* data) {
    uint64_t val = unpack_u64_be(data);
    double d;
    std::memcpy(&d, &val, sizeof(d));
    return d;
}

// Decompress zlib data
bool PointCloudReceiver::decompressData(const std::vector<uchar>& compressed, std::vector<uchar>& decompressed) {
    uLongf decompressedSize = compressed.size() * 10; // guess max decompressed size (adjust as needed)
    decompressed.resize(decompressedSize);

    int ret = uncompress(decompressed.data(), &decompressedSize, compressed.data(), compressed.size());
    if (ret != Z_OK) {
        std::cerr << "Decompression failed: " << ret << std::endl;
        return false;
    }
    decompressed.resize(decompressedSize);
    return true;
}

cv::Mat PointCloudReceiver::receiveLidarData(int sock) {
    uint8_t header[16];
    if (!recvAll(sock, header, sizeof(header))) {
        throw std::runtime_error("Failed to receive header");
    }

    uint64_t compressedSize = unpack_u64_be(header);
    double timestamp = unpack_double_be(header + 8);
    std::cout << "Compressed size: " << compressedSize << ", timestamp: " << timestamp << std::endl;

    std::vector<uchar> compressedData(compressedSize);
    if (!recvAll(sock, compressedData.data(), compressedSize)) {
        throw std::runtime_error("Failed to receive compressed cloud data");
    }

    std::vector<uchar> decompressedData;
    if (!decompressData(compressedData, decompressedData)) {
        throw std::runtime_error("Failed to decompress lidar cloud data");
    }

    // reconstruct cv::Mat
    size_t totalBytes = decompressedData.size();
    size_t pointCount = totalBytes / (2 * sizeof(float));
    cv::Mat cloudMat(pointCount, 2, CV_32F);

    std::memcpy(cloudMat.data, decompressedData.data(), totalBytes);

    return cloudMat;
}

bool LidarOdometry::lidarOdom(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2, RelativePose* rel_pos) {
    if (cloud1->empty() || cloud2->empty()) {
        std::cerr << "Input point clouds are empty!" << std::endl;
        return false;
    }
    Eigen::Matrix4f transformation;
    // Optional: Downsample to improve speed and stability
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setLeafSize(0.2f, 0.2f, 0.2f);

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered1(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered2(new pcl::PointCloud<pcl::PointXYZ>());

    voxel_grid.setInputCloud(cloud1);
    voxel_grid.filter(*filtered1);

    voxel_grid.setInputCloud(cloud2);
    voxel_grid.filter(*filtered2);

    // Setup NDT
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setResolution(1.0);  // voxel grid resolution
    ndt.setMaximumIterations(50);

    ndt.setInputSource(filtered1);
    ndt.setInputTarget(filtered2);

    // Initial guess (identity or known rough transform)
    Eigen::Matrix4f initial_guess = Eigen::Matrix4f::Identity();
    initial_guess(0, 3) = 10.0f; // if you know there's about 1m shift

    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    ndt.align(*output_cloud, initial_guess);

    if (ndt.hasConverged()) {
        std::cout << "[NDT] Converged." << std::endl;
        std::cout << "[NDT] Fitness score: " << ndt.getFitnessScore() << std::endl;
        std::cout << "[NDT] Final transformation:\n" << ndt.getFinalTransformation() << std::endl;
    } else {
        std::cerr << "[NDT] Failed to converge!" << std::endl;
    }

    // Setup ICP
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    // pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(filtered1);
    icp.setInputTarget(filtered2);

    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final, initial_guess);

    icp.setMaximumIterations(50);
    icp.setMaxCorrespondenceDistance(0.05);
    icp.setTransformationEpsilon(1e-8);
    icp.setEuclideanFitnessEpsilon(1e-5);

    bool success = icp.hasConverged();
    if (success) {
        // std::cout << "ICP converged.\n";
        // std::cout << "Score: " << icp.getFitnessScore() << "\n";
        // std::cout << "Transformation:\n" << icp.getFinalTransformation() << std::endl;
        transformation = icp.getFinalTransformation();
        
        // Extract rotation (top-left 3x3)
        Eigen::Matrix3f R_eig = transformation.block<3,3>(0,0);

        // Extract translation (top-right 3x1)
        Eigen::Vector3f t_eig = transformation.block<3,1>(0,3);

        // Convert to OpenCV Mat
        cv::Mat R_cv(3, 3, CV_32F), R_cv_v;
        cv::Mat t_cv(3, 1, CV_32F);

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_cv.at<float>(i, j) = R_eig(i, j);
            }
        }
        Rodrigues(R_cv, R_cv_v); // Convert rotation matrix to rotation vector
        Eigen::Vector3f R_eig_v;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                R_eig_v(i, j) = R_cv_v.at<float>(i, j);
            }
        }
        rel_pos->R = R_eig_v; // Store rotation matrix
        rel_pos->t = transformation.block<3,1>(0,3); // Store translation vector
        rel_pos->valid = true;
        rel_pos->sensor_type = "lidar"; // Set sensor type
        std::cout << "Lidar odometry computed successfully." << std::endl;
    } 
    return success;
}