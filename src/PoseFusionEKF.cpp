#include "PoseFusionEKF.h"

// --- Types ---
using Mat6 = Eigen::Matrix<double,6,6>;
using Vec6 = Eigen::Matrix<double,6,1>;


PoseFusionEKF::PoseFusionEKF(const Params& p = Params{}) : prm_(p) {
        reset();
    }

void PoseFusionEKF::reset(const Eigen::Vector3d& p0,
               const Eigen::Quaterniond& q0,
               double t0)
    {
        state_p_ = p0;
        state_q_ = q0.normalized();
        P_.setZero();
        // small initial uncertainty
        P_.topLeftCorner<3,3>()      = Eigen::Matrix3d::Identity() * 1e-4; // pos
        P_.bottomRightCorner<3,3>()  = Eigen::Matrix3d::Identity() * (0.1 * M_PI/180.0)*(0.1 * M_PI/180.0); // ang
        last_cam_pose_.reset();
        last_time_ = t0;
    }

void PoseFusionEKF::setCamToBody(const Eigen::Isometry3d& T_body_from_cam) { prm_.T_body_from_cam = T_body_from_cam; }
void PoseFusionEKF::setLidarToBody(const Eigen::Isometry3d& T_body_from_lidar) { prm_.T_body_from_lidar = T_body_from_lidar; }
void PoseFusionEKF::setNoise(double sigma_cam_pos, double sigma_cam_ang,
                double sigma_lidar_pos, double sigma_lidar_ang) {
    prm_.sigma_cam_pos = sigma_cam_pos;
    prm_.sigma_cam_ang = sigma_cam_ang;
    prm_.sigma_lidar_pos = sigma_lidar_pos;
    prm_.sigma_lidar_ang = sigma_lidar_ang;
}

void PoseFusionEKF::pushCameraPose(const Pose& cam_world_pose) {
        // Convert sensor pose to BODY pose in world: T_w_b = T_w_cam * T_cam_b
        Eigen::Isometry3d T_w_cam = isoFromPose(cam_world_pose);
        Eigen::Isometry3d T_cam_b = prm_.T_body_from_cam.inverse();
        Eigen::Isometry3d T_w_b   = T_w_cam * T_cam_b;

        Pose body_pose;
        body_pose.p = T_w_b.translation();
        body_pose.q = Eigen::Quaterniond(T_w_b.rotation()).normalized();
        body_pose.t = cam_world_pose.t;

        if (!last_cam_pose_) {
            // Initialize nominal state from first camera pose if not set
            state_p_ = body_pose.p;
            state_q_ = body_pose.q;
            last_cam_pose_ = body_pose;
            last_time_ = body_pose.t;
            return;
        }

        // Compute relative motion between last camera BODY pose and current camera BODY pose
        Eigen::Isometry3d T_w_b_prev = isoFromPose(*last_cam_pose_);
        Eigen::Isometry3d T_w_b_curr = isoFromPose(body_pose);
        Eigen::Isometry3d T_bprev_bcurr = T_w_b_prev.inverse() * T_w_b_curr;

        Eigen::Vector3d p_rel = T_bprev_bcurr.translation();               // in prev body frame
        Eigen::Quaterniond q_rel(T_bprev_bcurr.rotation());                // body_prev -> body_curr

        // --- Predict: apply relative motion to nominal state ---
        // world update: p_k+ = p_k + R(q_k) * p_rel ; q_k+ = q_k * q_rel
        state_p_ += state_q_ * p_rel;
        state_q_  = (state_q_ * q_rel).normalized();

        // --- Covariance propagation (simple additive with process noise per step) ---
        Mat6 Q = Mat6::Zero();
        Q.topLeftCorner<3,3>().setIdentity();        Q.topLeftCorner<3,3>()      *= prm_.sigma_cam_pos * prm_.sigma_cam_pos;
        Q.bottomRightCorner<3,3>().setIdentity();    Q.bottomRightCorner<3,3>()  *= prm_.sigma_cam_ang * prm_.sigma_cam_ang;
        // For pure relative pose compounding, state transition ~ I in the error space => P = P + Q
        P_ += Q;

        last_cam_pose_ = body_pose;
        last_time_     = body_pose.t;
    }


void PoseFusionEKF::pushLidarPose(const Pose& lidar_world_pose) {
        // Convert to BODY pose in world: T_w_b = T_w_lidar * T_lidar_b
        Eigen::Isometry3d T_w_l = isoFromPose(lidar_world_pose);
        Eigen::Isometry3d T_l_b = prm_.T_body_from_lidar.inverse();
        Eigen::Isometry3d T_w_b = T_w_l * T_l_b;

        Eigen::Vector3d p_meas = T_w_b.translation();
        Eigen::Quaterniond q_meas(T_w_b.rotation());
        q_meas.normalize();

        // Innovation: position & small-angle orientation
        Eigen::Vector3d dp = p_meas - state_p_;
        Eigen::Quaterniond q_err = q_meas * state_q_.inverse(); // world<-body: q_meas ⊗ q_state^{-1}
        Eigen::Vector3d dtheta = quatLog(q_err);

        Vec6 y; y << dp, dtheta;

        // H = I (direct measurement in error space)
        Mat6 H = Mat6::Identity();

        // Measurement covariance
        Mat6 R = Mat6::Zero();
        R.topLeftCorner<3,3>().setIdentity();       R.topLeftCorner<3,3>()      *= prm_.sigma_lidar_pos * prm_.sigma_lidar_pos;
        R.bottomRightCorner<3,3>().setIdentity();   R.bottomRightCorner<3,3>()  *= prm_.sigma_lidar_ang * prm_.sigma_lidar_ang;

        // Kalman gain
        Mat6 S = P_ + R;
        Mat6 K = P_ * S.inverse();

        // Correction
        Vec6 dx = K * y;
        state_p_ += dx.head<3>();
        state_q_  = (state_q_ * quatFromSmallAngle(dx.tail<3>())).normalized();

        // Joseph-form covariance update for numerical stability
        Mat6 I = Mat6::Identity();
        P_ = (I - K*H) * P_ * (I - K*H).transpose() + K * R * K.transpose();
    }


Pose PoseFusionEKF::fusedPose() const {
        Pose out;
        out.p = state_p_;
        out.q = state_q_;
        out.t = last_time_;
        return out;
    }

const Mat6& PoseFusionEKF::covariance() const { return P_; }


Eigen::Isometry3d PoseFusionEKF::defaultCamToBody() {
        Eigen::Matrix3d R;
        R <<  0,  0, 1,
             -1,  0, 0,
              0, -1, 0;
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = R;
        return T;
    }

Eigen::Quaterniond PoseFusionEKF::quatFromSmallAngle(const Eigen::Vector3d& dtheta) {
    double angle = dtheta.norm();
    if (angle < 1e-12) {
        // first-order
        Eigen::Quaterniond q(1.0, 0.5*dtheta.x(), 0.5*dtheta.y(), 0.5*dtheta.z());
        return q.normalized();
    }
    Eigen::Vector3d axis = dtheta / angle;
    double half = 0.5 * angle;
    return Eigen::Quaterniond(std::cos(half), axis.x()*std::sin(half), axis.y()*std::sin(half), axis.z()*std::sin(half)).normalized();
}

Eigen::Vector3d PoseFusionEKF::quatLog(const Eigen::Quaterniond& q_in) {
        Eigen::Quaterniond q = q_in.normalized();
        double w = q.w();
        Eigen::Vector3d v(q.x(), q.y(), q.z());
        double nv = v.norm();
        if (nv < 1e-12) return Eigen::Vector3d::Zero();
        double theta = 2.0 * std::atan2(nv, w);
        return v * (theta / nv);
    }

Eigen::Isometry3d PoseFusionEKF::isoFromPose(const Pose& P) {
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.linear() = P.q.normalized().toRotationMatrix();
        T.translation() = P.p;
        return T;
    }