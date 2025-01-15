/******************************************************************************
 * File: main.cpp
 * Author: aabouee
 * Date: 13.01.2025
 * Description: This program performs sensor fusion of IMU and Visual Odometry
 *              data to estimate the trajectory of a moving platform. It
 *              utilizes the GTSAM library for factor graph optimization,
 *              offering both Levenberg-Marquardt and iSAM2 optimization
 *              methods.
 ******************************************************************************/

#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
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

// Define symbolic variables for brevity
using gtsam::symbol_shorthand::B;  // Bias  (ax,ay,az,gx,gy,gz)
using gtsam::symbol_shorthand::V;  // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::X;  // Pose3 (x,y,z,r,p,y)

/******************************************************************************
 * Struct: ImuData
 * Description: Represents a single IMU measurement, including timestamp,
 *              linear acceleration, and angular velocity.
 ******************************************************************************/
struct ImuData {
  double timestamp{0};                          // Timestamp of the measurement in seconds
  gtsam::Vector3 linear_acceleration{0, 0, 0};  // Linear acceleration vector
  gtsam::Vector3 angular_velocity{0, 0, 0};     // Angular velocity vector

  // Default constructor
  ImuData() = default;

  // Print the IMU data to the console
  void print() {
    std::cout << std::fixed << std::setprecision(9) << timestamp << ", " << linear_acceleration.transpose() << ", "
              << angular_velocity.transpose() << std::endl;
  }
};

/******************************************************************************
 * Struct: VisualOdometryData
 * Description: Represents a single Visual Odometry measurement, including
 *              timestamp, pose (position and orientation), and covariance.
 ******************************************************************************/
struct VisualOdometryData {
  double timestamp{0};                                       // Timestamp of the measurement in seconds
  gtsam::Pose3 pose{gtsam::Rot3(), gtsam::Point3(0, 0, 0)};  // Pose (rotation and translation)
  gtsam::Matrix6 covariance{gtsam::Matrix6::Zero()};         // Covariance matrix associated with the pose

  // Default constructor
  VisualOdometryData() = default;

  // Print the Visual Odometry data to the console
  void print() {
    std::cout << std::fixed << std::setprecision(9) << timestamp << ", " << pose.translation().transpose() << ", "
              << pose.rotation().toQuaternion() << std::endl;
  }
};

/******************************************************************************
 * Function: loadImuData
 * Description: Loads IMU data from a file. Each line in the file is expected
 *              to contain a timestamp, followed by linear acceleration (x, y, z)
 *              and angular velocity (x, y, z).
 * Parameters:
 *   - filename: The path to the IMU data file.
 * Returns:
 *   - A vector of ImuData structs containing the loaded IMU measurements.
 ******************************************************************************/
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

    // Read timestamp and convert to seconds
    if (!(iss >> timestamp_sec)) continue;
    data.timestamp = timestamp_sec;

    // Read linear acceleration and angular velocity
    for (int i = 0; i < 3; i++) iss >> data.linear_acceleration[i];
    for (int i = 0; i < 3; i++) iss >> data.angular_velocity[i];

    imu_measurements.push_back(data);
  }

  return imu_measurements;
}

/******************************************************************************
 * Function: loadVoData
 * Description: Loads Visual Odometry data from a file. Each line in the file
 *              is expected to contain a timestamp, followed by translation
 *              (x, y, z) and a rotation quaternion (w, x, y, z).
 * Parameters:
 *   - filename: The path to the VO data file.
 * Returns:
 *   - A vector of VisualOdometryData structs containing the loaded VO
 *     measurements.
 ******************************************************************************/
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
    gtsam::Vector3 translation;
    gtsam::Vector4 rotation_quat;  // w, x, y, z

    // Read timestamp
    if (!(iss >> timestamp_sec)) continue;
    data.timestamp = timestamp_sec;

    // Read translation
    for (int i = 0; i < 3; i++) iss >> translation[i];

    // Read rotation quaternion
    for (int i = 0; i < 4; i++) iss >> rotation_quat[i];

    // Create Pose3 from translation and rotation
    data.pose = gtsam::Pose3(
        gtsam::Rot3::Quaternion(rotation_quat[3], rotation_quat[0], rotation_quat[1], rotation_quat[2]), translation);

    vo_measurements.push_back(data);
  }

  return vo_measurements;
}

/******************************************************************************
 * Function: saveToFile
 * Description: Saves the optimized trajectory data to a file.
 * Parameters:
 *   - filename: The path to the output file.
 *   - optimized_data: A vector of VisualOdometryData containing the optimized
 *                     trajectory.
 * Returns:
 *   - True if the file was saved successfully, false otherwise.
 ******************************************************************************/
bool saveToFile(const std::string &filename, std::vector<VisualOdometryData> &optimized_data) {
  // Save VO and IMU data separately
  std::ofstream est_file(filename);

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
  return true;
}

/******************************************************************************
 * Function: main
 * Description: The main function of the program. It loads IMU and VO data,
 *              performs sensor fusion using GTSAM, and saves the optimized
 *              trajectory to a file.
 * Parameters:
 *   - argc: The number of command-line arguments.
 *   - argv: An array of command-line argument strings.
 * Returns:
 *   - 0 if the program executed successfully, 1 otherwise.
 ******************************************************************************/
int main(int argc, char const *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " OPTIMIZATION_TYPE=(lm|isam) <imu_file.txt> <vo_file.txt> " << std::endl;
    return 1;
  }

  // Declare a pointer to store the preintegrated IMU measurements.
  // std::unique_ptr<gtsam::PreintegratedImuMeasurements> imu_preintegrated_;

  // Get the optimization type from the command-line arguments.
  std::string optimization_type = argv[1];
  if (optimization_type != "lm" && optimization_type != "isam") {
    std::cerr << "Invalid optimization type. Choose 'lm' for Levenberg-Marquardt or 'isam' for iSAM2." << std::endl;
    return 1;
  }

  // Load IMU and VO data from the specified files.
  std::vector<ImuData> imu_data = loadImuData(argv[2]);
  std::vector<VisualOdometryData> vo_data = loadVoData(argv[3]);

  // Check if the data was loaded successfully.
  if (imu_data.empty() || vo_data.empty()) {
    std::cerr << "Failed to load data files" << std::endl;
    return 1;
  } else {
    std::cout << imu_data.size() << " IMU data loaded." << std::endl;
    std::cout << vo_data.size() << " VO data loaded." << std::endl;
  }

  int correction_count = 0;

  // iSAM2 optimization parameters.
  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.1;  // Threshold for relinearization.
  parameters.relinearizeSkip = 1;         // Perform relinearization every 'relinearizeSkip' updates.
  gtsam::ISAM2 isam(parameters);

  // Transformation from camera frame to IMU frame.
  const auto T_imu_cam_ = gtsam::Pose3(gtsam::Rot3::Quaternion(0.7123015, -0.0077072, 0.0104993, 0.7017528),
                                       gtsam::Point3(-0.0216401454975, -0.064676986768, 0.00981073058949));

  // Prior values for pose, velocity, and IMU bias.
  gtsam::Rot3 prior_rotation = gtsam::Rot3::Quaternion(1.0, 0.0, 0.0, 0.0);  // Identity rotation.
  gtsam::Point3 prior_position(0.0, 0.0, 0.0);                               // Zero initial position.
  gtsam::Pose3 prior_pose(prior_rotation, prior_position);                   // Initial pose.
  gtsam::Vector3 prior_vel(0.0, 0.0, 0.0);                                   // Zero initial velocity.
                                                                             // Create initial bias sigma
  auto bias_sigma = gtsam::Vector6::Constant(1e-3);
  gtsam::imuBias::ConstantBias prior_imu_bias(bias_sigma);  // Zero initial IMU bias.

  // Initial values for the optimization.
  gtsam::Values initial_values;
  initial_values.insert(X(correction_count), prior_pose);
  initial_values.insert(V(correction_count), prior_vel);
  initial_values.insert(B(correction_count), prior_imu_bias);

  // Noise models for prior factors.
  auto pose_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector(6) << 0.01, 0.01, 0.01, 0.1, 0.1, 0.).finished());     // rad,rad,rad,m, m, m
  auto velocity_noise_model = gtsam::noiseModel::Isotropic::Sigma(3, 0.1);  // m/s
  auto bias_noise_model = gtsam::noiseModel::Diagonal::Sigmas(
      (gtsam::Vector6() << gtsam::Vector3::Constant(3.0000e-3), gtsam::Vector3::Constant(0.5e-05)).finished());

  // Create an empty nonlinear factor graph.
  gtsam::NonlinearFactorGraph graph;

  // Add prior factors for pose, velocity, and bias to the graph.
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(correction_count), prior_pose, pose_noise_model);
  graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(correction_count), prior_vel, velocity_noise_model);
  graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(B(correction_count), prior_imu_bias,
                                                                         bias_noise_model);

  // IMU noise models
  auto imu_accel_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(2.0000e-3, 2.0000e-3, 2.0000e-3));
  auto imu_gyro_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(1.6968e-04, 1.6968e-04, 1.6968e-04));
  auto imu_integration_noise_model = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(1e-8, 1e-8, 1e-8));

  const auto preint_params = gtsam::PreintegratedCombinedMeasurements::Params::MakeSharedU(9.81);
  preint_params->accelerometerCovariance = imu_accel_noise_model->covariance();
  preint_params->gyroscopeCovariance = imu_gyro_noise_model->covariance();
  preint_params->integrationCovariance = imu_integration_noise_model->covariance();
  preint_params->biasAccCovariance = bias_noise_model->covariance().block<3, 3>(0, 0);
  preint_params->biasOmegaCovariance = bias_noise_model->covariance().block<3, 3>(3, 3);
  preint_params->biasAccOmegaInt = gtsam::Matrix::Identity(6, 6) * 1e-10;

  // Initialize preintegration
  auto preint_imu_combined = std::make_shared<gtsam::PreintegratedCombinedMeasurements>(preint_params, prior_imu_bias);

  // Variables to store previous state and bias for IMU preintegration.
  double dt = 0.0;
  gtsam::NavState prev_state(prior_pose, prior_vel);
  gtsam::NavState last_state;
  gtsam::imuBias::ConstantBias prev_bias = prior_imu_bias;

  // Vector to store optimized trajectory data.
  std::vector<VisualOdometryData> optimized_data;

  // Main loop: Iterate through the VO data and integrate IMU measurements.
  for (auto iter_odom = vo_data.begin() + 1; iter_odom != vo_data.end(); ++iter_odom) {
    // Find IMU measurements between the current and previous VO timestamps.
    auto del_point = imu_data.begin();
    for (auto iter_imu = imu_data.begin(); iter_imu->timestamp <= iter_odom->timestamp; ++iter_imu) {
      // Calculate time difference between consecutive IMU measurements.
      // For beginning of the data, use a small time step.
      if (iter_imu == imu_data.begin()) {
        dt = 1e-9;
      } else {
        dt = iter_imu->timestamp - std::prev(iter_imu)->timestamp;
        std::cout << "dt: " << dt << ", current tms: " << iter_imu->timestamp
                  << ", pre tms: " << std::prev(iter_imu)->timestamp << std::endl;
      }
      // Integrate the IMU measurement.
      preint_imu_combined->integrateMeasurement(iter_imu->linear_acceleration, iter_imu->angular_velocity, dt);
      del_point = iter_imu;
    }

    // Increment the correction count.
    correction_count++;

    // Add the IMU factor to the graph.
    graph.emplace_shared<gtsam::CombinedImuFactor>(X(correction_count - 1), V(correction_count - 1),
                                                   X(correction_count), V(correction_count), B(correction_count - 1),
                                                   B(correction_count), *preint_imu_combined);

    // Transform VO measurements to the IMU frame.
    gtsam::Pose3 current_vo_in_imu = T_imu_cam_ * iter_odom->pose * T_imu_cam_.inverse();
    gtsam::Pose3 last_vo_in_imu = T_imu_cam_ * std::prev(iter_odom)->pose * T_imu_cam_.inverse();

    // Calculate the relative pose between consecutive VO measurements.
    gtsam::Pose3 relative_pose = last_vo_in_imu.between(current_vo_in_imu);

    // Noise model for the VO factor (tune sigma based on VO accuracy).
    auto vo_noise_model = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);

    // Add a between factor for the relative pose from VO.
    graph.add(gtsam::BetweenFactor<gtsam::Pose3>(X(correction_count - 1), X(correction_count), relative_pose,
                                                 vo_noise_model));

    // Predict the current state using IMU preintegration.
    last_state = preint_imu_combined->predict(prev_state, prev_bias);
    // Insert the predicted state into the initial values.
    initial_values.insert(X(correction_count), last_state.pose());
    initial_values.insert(V(correction_count), last_state.v());
    initial_values.insert(B(correction_count), prev_bias);

    // Optimize the factor graph.
    gtsam::Values result;
    if (optimization_type == "lm") {
      // Use Levenberg-Marquardt optimizer.
      gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_values);
      result = optimizer.optimize();
    } else {
      // Use iSAM2 optimizer.
      isam.update(graph, initial_values);
      isam.update();  // Additional update for better convergence (can be adjusted).
      result = isam.calculateEstimate();
      if (result.empty()) {
        std::cerr << "Optimization failed: Empty result" << std::endl;
        return 1;
      }
    }

    // Update the previous state and bias with the optimized values.
    prev_state =
        gtsam::NavState(result.at<gtsam::Pose3>(X(correction_count)), result.at<gtsam::Vector3>(V(correction_count)));
    prev_bias = result.at<gtsam::imuBias::ConstantBias>(B(correction_count));
    // Reset the IMU preintegration with the updated bias.
    preint_imu_combined->resetIntegrationAndSetBias(prev_bias);

    // Clear the graph and initial values for the next iteration (except for lm which needs it).
    graph.resize(0);
    initial_values.clear();

    if (optimization_type == "lm") {
      // For Levenberg-Marquardt, re-add prior factors for the current state after each optimization step.
      initial_values.insert(X(correction_count), prev_state.pose());
      initial_values.insert(V(correction_count), prev_state.v());
      initial_values.insert(B(correction_count), prev_bias);
      graph.emplace_shared<gtsam::PriorFactor<gtsam::Pose3>>(X(correction_count), prev_state.pose(), pose_noise_model);
      graph.emplace_shared<gtsam::PriorFactor<gtsam::Vector3>>(V(correction_count), prev_state.v(),
                                                               velocity_noise_model);
      graph.emplace_shared<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(B(correction_count), prev_bias,
                                                                             bias_noise_model);
    }

    // Print and store the optimized pose.
    gtsam::Vector3 gtsam_position = prev_state.pose().translation();
    gtsam::Quaternion gtsam_quat = prev_state.pose().rotation().toQuaternion();
    optimized_data.push_back({iter_odom->timestamp, prev_state.pose()});
    std::cout << iter_odom->timestamp << " " << gtsam_position(0) << " " << gtsam_position(1) << " "
              << gtsam_position(2) << " " << gtsam_quat.x() << " " << gtsam_quat.y() << " " << gtsam_quat.z() << " "
              << gtsam_quat.w() << std::endl;

    // Remove processed IMU data points to avoid redundant integration in the next iteration.
    imu_data.erase(imu_data.begin(), del_point);
  }
  std::cout << "size of optimized data: " << optimized_data.size() << std::endl;

  bool res = saveToFile("../estimated.txt", optimized_data);
  if (res) std::cout << "The optimized data has been saved to estimated.txt" << std::endl;
  return 0;
}