#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace gtsam;

using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

struct ImuData {
  double timestamp{0};  // in seconds
  gtsam::Vector3 linear_acceleration{0, 0, 0};
  gtsam::Vector3 angular_velocity{0, 0, 0};

  // Default constructor
  ImuData() = default;
  void print() {
    std::cout << std::fixed << std::setprecision(9) << timestamp << ", " << linear_acceleration.transpose() << ", "
              << angular_velocity.transpose() << std::endl;
  }
};

struct VisualOdometryData {
  double timestamp{0};                                       // in seconds
  gtsam::Pose3 pose{gtsam::Rot3(), gtsam::Point3(0, 0, 0)};  // Identity rotation and zero translation
  gtsam::Matrix6 covariance{gtsam::Matrix6::Zero()};

  // Default constructor
  VisualOdometryData() = default;
  void print() {
    std::cout << std::fixed << std::setprecision(9) << timestamp << ", " << pose.translation().transpose() << ", "
              << pose.rotation().toQuaternion() << std::endl;
  }
};

// Add helper functions to load data
std::vector<ImuData> loadImuData(const std::string &filename) {
  std::vector<ImuData> imu_measurements;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Failed to open IMU file: " << filename << std::endl;
    return imu_measurements;
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    ImuData data;
    double timestamp_sec;

    // Read timestamp and convert to nanoseconds
    if (!(iss >> timestamp_sec)) continue;
    data.timestamp = timestamp_sec;

    // Read linear acceleration and angular velocity
    for (int i = 0; i < 3; i++) iss >> data.linear_acceleration[i];
    for (int i = 0; i < 3; i++) iss >> data.angular_velocity[i];

    imu_measurements.push_back(data);
  }

  return imu_measurements;
}

std::vector<VisualOdometryData> loadVoData(const std::string &filename) {
  std::vector<VisualOdometryData> vo_measurements;
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open()) {
    std::cerr << "Failed to open VO file: " << filename << std::endl;
    return vo_measurements;
  }

  while (std::getline(file, line)) {
    std::istringstream iss(line);
    VisualOdometryData data;
    double timestamp_sec;
    Vector3 translation;
    Vector4 rotation_quat;  // w, x, y, z

    // Read timestamp and convert to nanoseconds
    if (!(iss >> timestamp_sec)) continue;
    data.timestamp = timestamp_sec;

    // Read translation
    for (int i = 0; i < 3; i++) iss >> translation[i];

    // Read rotation quaternion
    for (int i = 0; i < 4; i++) iss >> rotation_quat[i];

    // Create Pose3 from translation and rotation
    data.pose =
        Pose3(Rot3::Quaternion(rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]), translation);

    vo_measurements.push_back(data);
  }

  return vo_measurements;
}

int main(int argc, char const *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " OPTIMIZATION_TYPE=(lm|isam) <imu_file.txt> <vo_file.txt> " << std::endl;
    return 1;
  }
  PreintegratedImuMeasurements *imu_preintegrated_;

  std::string optimization_type = argv[1];
  if (optimization_type != "lm" && optimization_type != "isam") {
    std::cerr << "Invalid optimization type. Choose 'lm' for Levenberg-Marquardt or 'isam' for iSAM2." << std::endl;
    return 1;
  }

  // Load IMU and VO data
  std::vector<ImuData> imu_data = loadImuData(argv[2]);
  std::vector<VisualOdometryData> vo_data = loadVoData(argv[3]);

  std::cout << "IMU data size: " << imu_data.size() << std::endl;
  std::cout << "VO data size: " << vo_data.size() << std::endl;

  std::cout << "First VO data:" << std::endl;
  vo_data.front().print();
  std::cout << "Last VO data:" << std::endl;
  vo_data.back().print();

  std::cout << "First IMU data:" << std::endl;
  imu_data.front().print();
  std::cout << "Last IMU data:" << std::endl;
  imu_data.back().print();

  if (imu_data.empty() || vo_data.empty()) {
    std::cerr << "Failed to load data files" << std::endl;
    return 1;
  }

  ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.1;
  parameters.relinearizeSkip = 1;
  ISAM2 isam(parameters);

  const auto T_imu_cam_ = gtsam::Pose3(gtsam::Rot3::Quaternion(0.7123015, -0.0077072, 0.0104993, 0.7017528),
                                       gtsam::Point3(-0.0216401454975, -0.064676986768, 0.00981073058949));

  Rot3 prior_rotation = Rot3::Quaternion(1.0, 0.0, 0.0, 0.0);
  Point3 prior_position(0.0, 0.0, 0.0);
  Pose3 prior_pose(prior_rotation, prior_position);
  Vector3 prior_vel(0.0, 0.0, 0.0);
  imuBias::ConstantBias prior_imu_bias;
  Values initial_values;
  int correction_count = 0;
  initial_values.insert(X(correction_count), prior_pose);
  initial_values.insert(V(correction_count), prior_vel);
  initial_values.insert(B(correction_count), prior_imu_bias);

  noiseModel::Diagonal::shared_ptr pose_noise_model =
      noiseModel::Diagonal::Sigmas((Vector(6) << 0.01, 0.01, 0.01, 0.5, 0.5, 0.5).finished());   // rad,rad,rad,m, m, m
  noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.1);  // m/s
  noiseModel::Diagonal::shared_ptr bias_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector6() << gtsam::Vector3::Constant(3.0000e-3), gtsam::Vector3::Constant(0.5e-05)).finished());

  // Add all prior factors (pose, velocity, bias) to the graph.
  NonlinearFactorGraph graph;
  graph.emplace_shared<PriorFactor<Pose3>>(X(correction_count), prior_pose, pose_noise_model);
  graph.emplace_shared<PriorFactor<Vector3>>(V(correction_count), prior_vel, velocity_noise_model);
  graph.emplace_shared<PriorFactor<imuBias::ConstantBias>>(B(correction_count), prior_imu_bias, bias_noise_model);

  // We use the sensor specs to build the noise model for the IMU factor.
  double accel_noise_sigma = 2.0000e-3;
  double gyro_noise_sigma = 1.6968e-04;
  Matrix33 measured_acc_cov = Matrix33::Identity(3, 3) * pow(accel_noise_sigma, 2);
  Matrix33 measured_omega_cov = Matrix33::Identity(3, 3) * pow(gyro_noise_sigma, 2);
  Matrix33 integration_error_cov =
      Matrix33::Identity(3, 3) * 1e-8;  // error committed in integrating position from velocities
  // Matrix33 bias_acc_cov = Matrix33::Identity(3,3) * pow(accel_bias_rw_sigma,2);
  // Matrix33 bias_omega_cov = Matrix33::Identity(3,3) * pow(gyro_bias_rw_sigma,2);
  // Matrix66 bias_acc_omega_int = Matrix::Identity(6,6)*1e-5; // error in the bias used for preintegration

  std::shared_ptr<PreintegratedImuMeasurements::Params> p = PreintegratedImuMeasurements::Params::MakeSharedU();
  // PreintegrationBase params:
  p->accelerometerCovariance = measured_acc_cov;     // acc white noise in continuous
  p->integrationCovariance = integration_error_cov;  // integration uncertainty continuoustgf
  p->gyroscopeCovariance = measured_omega_cov;       // gyro white noise in continuous
  // // PreintegrationCombinedMeasurements params:
  // p->biasAccCovariance = bias_acc_cov; // acc bias in continuous
  // p->biasOmegaCovariance = bias_omega_cov; // gyro bias in continuous
  // p->biasAccOmegaInt = bias_acc_omega_int;

  imu_preintegrated_ = new PreintegratedImuMeasurements(p, prior_imu_bias);
  double dt = 0.0;
  NavState prev_state(prior_pose, prior_vel);
  NavState last_state;
  imuBias::ConstantBias prev_bias = prior_imu_bias;

  std::vector<VisualOdometryData> optimized_data;

  // All priors have been set up, now iterate through the data.
  Eigen::Matrix<double, 6, 1> imu = Eigen::Matrix<double, 6, 1>::Zero();
  for (auto iter_odom = vo_data.begin() + 1; iter_odom != vo_data.end(); ++iter_odom) {
    auto del_point = imu_data.begin();
    for (auto iter_imu = imu_data.begin(); iter_imu->timestamp <= iter_odom->timestamp; ++iter_imu) {
      if (iter_imu == imu_data.begin()) {
        dt = 1e-9;
      } else {
        dt = iter_imu->timestamp - std::prev(iter_imu)->timestamp;
        std::cout << "dt: " << dt << ", current tms: " << iter_imu->timestamp
                  << ", pre tms: " << std::prev(iter_imu)->timestamp << std::endl;
      }
      imu << iter_imu->linear_acceleration.x(), iter_imu->linear_acceleration.y(), iter_imu->linear_acceleration.z(),
          iter_imu->angular_velocity.x(), iter_imu->angular_velocity.y(), iter_imu->angular_velocity.z();
      imu_preintegrated_->integrateMeasurement(imu.head<3>(), imu.tail<3>(), dt);

      del_point = iter_imu;
    }
    correction_count++;
    ImuFactor imu_factor(X(correction_count - 1), V(correction_count - 1), X(correction_count), V(correction_count),
                         B(correction_count - 1), *imu_preintegrated_);
    graph.add(imu_factor);
    imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
    graph.add(BetweenFactor<imuBias::ConstantBias>(B(correction_count - 1), B(correction_count), zero_bias,
                                                   bias_noise_model));

    gtsam::Pose3 current_vo_in_imu = T_imu_cam_ * iter_odom->pose * T_imu_cam_.inverse();
    gtsam::Pose3 last_vo_in_imu = T_imu_cam_ * std::prev(iter_odom)->pose * T_imu_cam_.inverse();
    gtsam::Pose3 relative_pose = last_vo_in_imu.between(current_vo_in_imu);

    auto vo_noise_model = gtsam::noiseModel::Isotropic::Sigma(6, 0.001);  // Tune sigma based on VO accuracy
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(X(correction_count - 1), X(correction_count), relative_pose,
                                                 vo_noise_model));
    // Now optimize and compare results.
    last_state = imu_preintegrated_->predict(prev_state, prev_bias);
    initial_values.insert(X(correction_count), last_state.pose());
    initial_values.insert(V(correction_count), last_state.v());
    initial_values.insert(B(correction_count), prev_bias);

    Values result;
    if (optimization_type == "lm") {
      LevenbergMarquardtOptimizer optimizer(graph, initial_values);
      result = optimizer.optimize();
    } else {
      isam.update(graph, initial_values);
      isam.update();
      result = isam.calculateEstimate();
    }

    // Overwrite the beginning of the preintegration for the next step.
    prev_state = NavState(result.at<Pose3>(X(correction_count)), result.at<Vector3>(V(correction_count)));
    prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));
    // Reset the preintegration object.
    imu_preintegrated_->resetIntegrationAndSetBias(prev_bias);
    graph.resize(0);
    initial_values.clear();

    if (optimization_type == "lm") {
      initial_values.insert(X(correction_count), prev_state.pose());
      initial_values.insert(V(correction_count), prev_state.v());
      initial_values.insert(B(correction_count), prev_bias);
      graph.emplace_shared<PriorFactor<Pose3>>(X(correction_count), prev_state.pose(), pose_noise_model);
      graph.emplace_shared<PriorFactor<Vector3>>(V(correction_count), prev_state.v(), velocity_noise_model);
      graph.emplace_shared<PriorFactor<imuBias::ConstantBias>>(B(correction_count), prev_bias, bias_noise_model);
    }

    // Print out the position and orientation.
    Vector3 gtsam_position = prev_state.pose().translation();
    Quaternion gtsam_quat = prev_state.pose().rotation().toQuaternion();
    optimized_data.push_back({iter_odom->timestamp, prev_state.pose()});
    std::cout << iter_odom->timestamp << " " << gtsam_position(0) << " " << gtsam_position(1) << " "
              << gtsam_position(2) << " " << gtsam_quat.x() << " " << gtsam_quat.y() << " " << gtsam_quat.z() << " "
              << gtsam_quat.w() << std::endl;
    imu_data.erase(imu_data.begin(), del_point);
  }
  std::cout << "size of optimized data: " << optimized_data.size() << std::endl;

  // Save VO and IMU data separately
  std::ofstream est_file("../estimated.txt");

  if (!est_file.is_open()) {
    return false;
  }

  // Write VO data
  for (const auto &pose_data : optimized_data) {
    const auto &quat = pose_data.pose.rotation().toQuaternion();
    const auto &trans = pose_data.pose.translation();
    est_file << std::fixed << std::setprecision(9) << pose_data.timestamp << " " << trans.x() << " " << trans.y() << " "
             << trans.z() << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
  }

  est_file.close();

  return 0;
}
