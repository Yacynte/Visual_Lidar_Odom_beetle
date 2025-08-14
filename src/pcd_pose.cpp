#include <iostream>
#include <fstream>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <iostream>

#include "receive_cloud.h"

const char* SERVER_IP = "192.168.1.100";
const int SERVER_PORT = 6080;

pcl::PointCloud<pcl::PointXYZ>::Ptr loadKittiBin(const std::string& path) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::ifstream input(path, std::ios::binary);

    if (!input) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return cloud;
    }

    // Each point is 3 floats (x, y, z)
    while (!input.eof()) {
        pcl::PointXYZ point;
        input.read(reinterpret_cast<char*>(&point.x), sizeof(float));
        input.read(reinterpret_cast<char*>(&point.y), sizeof(float));
        input.read(reinterpret_cast<char*>(&point.z), sizeof(float));

        cloud->push_back(point);
    }

    std::cout << "length of cloud: " << cloud->points.size() << std::endl;

    if (cloud->empty()) {
        std::cerr << "No points loaded from file: " << path << std::endl;
        return cloud;
    }

    input.close();
    return cloud;
}

int InitializeCom() {
     // Create socket
    int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        std::cerr << "Failed to create socket\n";
        return -1;
    }

    sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(SERVER_PORT);

    if (inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr) <= 0) {
        std::cerr << "Invalid server IP address\n";
        close(sock_fd);
        return -1;
    }

    // Connect to server
    if (connect(sock_fd, (sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        std::cerr << "Connection to server failed\n";
        close(sock_fd);
        return -1;
    }

    std::cout << "Connected to server, waiting for lidar data...\n";
    return sock_fd;
}



int main( int c, char** argv) {

    // Initialize communication
    int sock_fd = InitializeCom();
    if (sock_fd < 0) {
        std::cerr << "Failed to initialize communication\n";
        return -1;
    }
    PointCloudReceiver point_cloud_receiver(sock_fd);
    

    auto cloud1 = loadKittiBin("/home/divan/beetle/VisualOdometry/images/velodyne_points/data/0000000000.bin");
    auto cloud2 = loadKittiBin("/home/divan/beetle/VisualOdometry/images/velodyne_points/data/0000000001.bin");

    if (cloud1->empty() || cloud2->empty()) {
        std::cerr << "Error: one of the point clouds is empty." << std::endl;
        return -1;
    }

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

    if (icp.hasConverged()) {
        std::cout << "ICP converged.\n";
        std::cout << "Score: " << icp.getFitnessScore() << "\n";
        std::cout << "Transformation:\n" << icp.getFinalTransformation() << std::endl;
    } else {
        std::cout << "ICP did not converge.\n";
    }

    return 0;
}




// using PointT = pcl::PointXYZ;
// using FeatureT = pcl::FPFHSignature33;

// int main(int c, char** argv) {
//     // Load source and target
//     // pcl::PointCloud<PointT>::Ptr source(new pcl::PointCloud<PointT>());
//     // pcl::PointCloud<PointT>::Ptr target(new pcl::PointCloud<PointT>());
//     // pcl::io::loadPCDFile("source.pcd", *source);
//     // pcl::io::loadPCDFile("target.pcd", *target);

//     auto source = loadKittiBin("/home/divan/beetle/VisualOdometry/images/velodyne_points/data/0000000000.bin");
//     auto target = loadKittiBin("/home/divan/beetle/VisualOdometry/images/velodyne_points/data/0000000001.bin");

//     if (source->empty() || target->empty()) {
//         std::cerr << "Error: one of the point clouds is empty." << std::endl;
//         return -1;
//     }

//     pcl::VoxelGrid<PointT> voxel;
//     voxel.setLeafSize(0.05f, 0.05f, 0.05f);
//     pcl::PointCloud<PointT>::Ptr src_ds(new pcl::PointCloud<PointT>());
//     pcl::PointCloud<PointT>::Ptr tgt_ds(new pcl::PointCloud<PointT>());
//     voxel.setInputCloud(source);
//     voxel.filter(*src_ds);
//     voxel.setInputCloud(target);
//     voxel.filter(*tgt_ds);

//     // Estimate normals
//     pcl::NormalEstimation<PointT, pcl::Normal> ne;
//     ne.setRadiusSearch(0.1);
//     pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
//     ne.setSearchMethod(tree);

//     pcl::PointCloud<pcl::Normal>::Ptr src_normals(new pcl::PointCloud<pcl::Normal>());
//     pcl::PointCloud<pcl::Normal>::Ptr tgt_normals(new pcl::PointCloud<pcl::Normal>());
//     ne.setInputCloud(src_ds);
//     ne.compute(*src_normals);
//     ne.setInputCloud(tgt_ds);
//     ne.compute(*tgt_normals);

//     // Compute FPFH features
//     pcl::FPFHEstimation<PointT, pcl::Normal, FeatureT> fpfh;
//     fpfh.setRadiusSearch(0.2);
//     fpfh.setSearchMethod(tree);

//     pcl::PointCloud<FeatureT>::Ptr src_fpfh(new pcl::PointCloud<FeatureT>());
//     pcl::PointCloud<FeatureT>::Ptr tgt_fpfh(new pcl::PointCloud<FeatureT>());
//     fpfh.setInputCloud(src_ds);
//     fpfh.setInputNormals(src_normals);
//     fpfh.compute(*src_fpfh);

//     fpfh.setInputCloud(tgt_ds);
//     fpfh.setInputNormals(tgt_normals);
//     fpfh.compute(*tgt_fpfh);

//     // RANSAC alignment
//     pcl::SampleConsensusPrerejective<PointT, PointT, FeatureT> align;
//     align.setInputSource(src_ds);
//     align.setSourceFeatures(src_fpfh);
//     align.setInputTarget(tgt_ds);
//     align.setTargetFeatures(tgt_fpfh);
//     align.setMaximumIterations(50000);
//     align.setNumberOfSamples(3);      // Number of points to sample for generating/prerejecting a pose
//     align.setCorrespondenceRandomness(5); // Number of nearest features to use
//     align.setSimilarityThreshold(0.9f);
//     align.setMaxCorrespondenceDistance(2.5f * 0.05f);
//     align.setInlierFraction(0.25f);

//     pcl::PointCloud<PointT> aligned;
//     align.align(aligned);

//     if (align.hasConverged()) {
//         std::cout << "\nRANSAC alignment succeeded." << std::endl;
//         std::cout << "Fitness score: " << align.getFitnessScore() << std::endl;
//         std::cout << "Transformation:\n" << align.getFinalTransformation() << std::endl;
//     } else {
//         std::cerr << "RANSAC alignment failed." << std::endl;
//     }

//     return 0;
// }
