#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <optional>
#include <cmath>

/**
 * PoseFusionEKF
 *  - Nominal state: position p (world), orientation q (world<-body)
 *  - Error state:   δx = [δp (3), δθ (3)]
 *  - Predict:       from CAMERA incremental motion (high-rate VO)
 *  - Update:        with LiDAR absolute pose (lower rate LO)
 *
 * Frames & conventions:
 *  - Common "body/world" convention follows REP-103: x-forward, y-left, z-up.
 *  - Camera input pose is z-forward (typical CV: x-right, y-down, z-forward).
 *  - LiDAR input pose is x-forward (already REP-103-ish).
 *  - If sensors are not co-located, set extrinsics T_body_from_cam, T_body_from_lidar.
 *
 * Dependencies: Eigen3
 */

struct Pose {
    Eigen::Vector3d p;          // position
    Eigen::Quaterniond q;       // orientation (world <- frame) // [w,x,y,z]
    double t;                   // timestamp (seconds)
    bool valid = true; // whether pose is valid
};

class PoseFusionEKF {
public:
    // --- Types ---
    using Mat6 = Eigen::Matrix<double,6,6>;
    using Vec6 = Eigen::Matrix<double,6,1>;

    struct Params {
        // Process noise for CAMERA incremental motion (per predict step)
        double sigma_cam_pos = 0.02;     // meters (per step)
        double sigma_cam_ang = 0.5 * M_PI/180.0; // radians (per step)

        // Measurement noise for LiDAR pose (absolute)
        double sigma_lidar_pos = 0.03;   // meters
        double sigma_lidar_ang = 0.3 * M_PI/180.0; // radians

        // Static extrinsics (body from sensors). Defaults assume co-located origins with pure rotations.
        // If you know real extrinsics, set them after construction.
        Eigen::Isometry3d T_body_from_cam = defaultCamToBody();   // maps camera coords -> body coords
        Eigen::Isometry3d T_body_from_lidar = Eigen::Isometry3d::Identity(); // LiDAR already x-forward

        // Whether input poses are of the *sensor* frame in world (usual odometry), e.g., T_w_cam, T_w_lidar.
        // Fusion will convert them into T_w_body using the above extrinsics.
    };

    PoseFusionEKF(const Params& p);

    void reset(const Eigen::Vector3d& p0 = Eigen::Vector3d::Zero(),
               const Eigen::Quaterniond& q0 = Eigen::Quaterniond::Identity(),
               double t0 = 0.0);

    // --- Setters for extrinsics/noise at runtime ---
    void setCamToBody(const Eigen::Isometry3d& T_body_from_cam);
    void setLidarToBody(const Eigen::Isometry3d& T_body_from_lidar);
    void setNoise(double sigma_cam_pos, double sigma_cam_ang,
                  double sigma_lidar_pos, double sigma_lidar_ang);

    // --- Feed CAMERA (VO) pose (high rate): acts as PREDICT via relative motion ---
    // Input: pose of CAMERA in its world (or shared world) frame: T_w_cam = [p_c, q_c]
    void pushCameraPose(const Pose& cam_world_pose);
    // --- Feed LiDAR (LO) pose (low rate): acts as UPDATE (absolute) ---
    // Input: pose of LiDAR in its world (or shared world) frame: T_w_l = [p_l, q_l]
    void pushLidarPose(const Pose& lidar_world_pose);

    // --- Access fused state ---
    Pose fusedPose() const;

    const Mat6& covariance() const; // { return P_; }

    // --- Helper: default rotation from CAMERA (x-right, y-down, z-forward) to BODY (x-forward,y-left,z-up) ---
    // Maps cam axes:  x_cam -> -y_body,  y_cam -> -z_body,  z_cam -> x_body
    static Eigen::Isometry3d defaultCamToBody();

    // --- Helper: convert small-angle vector to quaternion ---
    static Eigen::Quaterniond quatFromSmallAngle(const Eigen::Vector3d& dtheta) ;

private:
    // --- Quaternion small-angle utilities ---
    static Eigen::Vector3d quatLog(const Eigen::Quaterniond& q_in);

    static Eigen::Isometry3d isoFromPose(const Pose& P);

private:
    Params prm_;
    Eigen::Vector3d    state_p_{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond state_q_{Eigen::Quaterniond::Identity()};
    Mat6 P_{Mat6::Zero()};
    std::optional<Pose> last_cam_pose_;
    double last_time_{0.0};
};
