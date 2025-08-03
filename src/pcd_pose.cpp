#include <iostream>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>

// Function to load .bin file into a point cloud
pcl::PointCloud<pcl::PointXYZ>::Ptr loadBinToPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Failed to open " << filename << std::endl;
        return cloud;
    }

    while (!input.eof()) {
        float x, y, z, intensity;
        input.read(reinterpret_cast<char*>(&x), sizeof(float));
        input.read(reinterpret_cast<char*>(&y), sizeof(float));
        input.read(reinterpret_cast<char*>(&z), sizeof(float));
        input.read(reinterpret_cast<char*>(&intensity), sizeof(float));  // we ignore intensity

        if (input.gcount() < 16) break;  // avoid reading incomplete points

        cloud->points.emplace_back(x, y, z);
    }

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = false;

    return cloud;
}

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

    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(cloud1);
    icp.setInputTarget(cloud2);

    pcl::PointCloud<pcl::PointXYZ> Final;
    icp.align(Final);

    if (icp.hasConverged()) {
        std::cout << "ICP converged.\n";
        std::cout << "Score: " << icp.getFitnessScore() << "\n";
        std::cout << "Transformation:\n" << icp.getFinalTransformation() << std::endl;
    } else {
        std::cout << "ICP did not converge.\n";
    }

    return 0;
}
