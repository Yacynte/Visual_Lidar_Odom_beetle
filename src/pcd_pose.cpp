#include <iostream>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/features/normal_3d.h>
// #include <pcl/features/fpfh.h>
// #include <pcl/registration/sample_consensus_prerejective.h>
// #include <pcl/visualization/pcl_visualizer.h>



// Function to load .bin file into a point cloud
// pcl::PointCloud<pcl::PointXYZ>::Ptr loadBinToPointCloud(const std::string& filename) {
//     pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

//     std::ifstream input(filename, std::ios::binary);
//     if (!input) {
//         std::cerr << "Failed to open " << filename << std::endl;
//         return cloud;
//     }

//     while (!input.eof()) {
//         float x, y, z, intensity;
//         input.read(reinterpret_cast<char*>(&x), sizeof(float));
//         input.read(reinterpret_cast<char*>(&y), sizeof(float));
//         input.read(reinterpret_cast<char*>(&z), sizeof(float));
//         input.read(reinterpret_cast<char*>(&intensity), sizeof(float));  // we ignore intensity

//         if (input.gcount() < 16) break;  // avoid reading incomplete points

//         cloud->points.emplace_back(x, y, z);
//     }

//     cloud->width = static_cast<uint32_t>(cloud->points.size());
//     cloud->height = 1;
//     cloud->is_dense = false;

//     return cloud;
// }

pcl::PointCloud<pcl::PointXYZ>::Ptr loadKittiBin(const std::string& path) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    std::ifstream input(path, std::ios::binary);

    if (!input) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return cloud;
    }

    // Each point is 4 floats (x, y, z, intensity)
    while (!input.eof()) {
        pcl::PointXYZ point;
        input.read(reinterpret_cast<char*>(&point.x), sizeof(float));
        input.read(reinterpret_cast<char*>(&point.y), sizeof(float));
        input.read(reinterpret_cast<char*>(&point.z), sizeof(float));
        // input.read(reinterpret_cast<char*>(&point.intensity), sizeof(float));
        // std::cout << "point: " << point.x << " " << point.y << " " << point.z << std::endl;
        // if (input.gcount() == sizeof(float) * 3) {
        //     cloud->push_back(point);
        // }
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


int main( int c, char** argv) {
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
