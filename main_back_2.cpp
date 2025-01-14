#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/dataset.h>

std::vector<gtsam::Pose3> loadPosesFromFile(const std::string &filename)
{
    std::vector<gtsam::Pose3> poses;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return poses;
    }

    double tx, ty, tz, qx, qy, qz, qw;
    while (file >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
    {
        gtsam::Rot3 rotation = gtsam::Rot3::Quaternion(qw, qx, qy, qz);
        gtsam::Point3 translation(tx, ty, tz);
        gtsam::Pose3 pose(rotation, translation);
        poses.push_back(pose);
    }

    file.close();
    return poses;
}

std::vector<gtsam::Pose3> convert_to_ENU(const std::vector<gtsam::Pose3> &poses)
{
    std::vector<gtsam::Pose3> enu_poses;
    enu_poses.reserve(poses.size());

    gtsam::Pose3 T_imu_cam = gtsam::Pose3(gtsam::Rot3::RzRyRx(M_PI, M_PI * 15 / 180, M_PI), gtsam::Point3(0.0, 0.0, 0.0));
    std::cout << "T_imu_cam: " << T_imu_cam.matrix() << std::endl;

    for (const auto &pose : poses)
    {
        // Create new ENU pose
        const gtsam::Pose3 enu_pose = T_imu_cam * pose * T_imu_cam.inverse();
        enu_poses.push_back(enu_pose);
    }

    return enu_poses;
}

int main()
{
    auto source_poses = loadPosesFromFile("/home/aabouee/Workspace/libs/test_gtsam/input/gt_poses.txt");
    auto target_poses = loadPosesFromFile("/home/aabouee/Workspace/libs/test_gtsam/input/es_poses.txt");

    auto transformed_poses = convert_to_ENU(target_poses);

    std::cout << "Size of sources: " << source_poses.size() << ", while transformed has: " << transformed_poses.size() << std::endl;

    // Step 4: Use GTSAM to compute the similarity transformation
    std::vector<std::pair<gtsam::Pose3, gtsam::Pose3>> correspondences;

    for (size_t i = 0; i < source_poses.size(); ++i)
    {
        correspondences.emplace_back(transformed_poses[i], source_poses[i]);
    }

    // Use the align function
    gtsam::Similarity3 similarity_transform = gtsam::Similarity3::Align(correspondences);

    // Print the ground-truth and computed transformations
    // cout << "Ground-truth transformation:" << endl;
    // cout << "Rotation:\n"
    //      << R << endl;
    // cout << "Translation: " << t.transpose() << endl;
    // cout << "Scale: " << s << endl;

    std::cout << "Computed transformation:" << std::endl;
    std::cout << "Rotation:\n"
              << similarity_transform.rotation().matrix() << std::endl;
    std::cout << "Translation: " << similarity_transform.translation().transpose() << std::endl;
    std::cout << "Scale: " << similarity_transform.scale() << std::endl;

    // Eigen::Matrix4d ip;
    // ip << 0.159982, 0.971829, -0.173071, -1.78644, 0.961056, -0.113321, 0.252052, -0.974415, 0.225339, -0.206654, -0.952111, -3.70283, 0, 0, 0, 0.30716;
    // Similarity3 similarity_transform_new = Similarity3(ip);

    // cout << "Computed transformation:" << endl;
    // cout << "Rotation:\n"
    //      << similarity_transform_new.rotation().matrix() << endl;
    // cout << "Translation: " << similarity_transform_new.translation().transpose() << endl;
    // cout << "Scale: " << similarity_transform_new.scale() << endl;

    // Step 5: Project the source points to the target using the computed transformation and compute the error
    double total_error = 0.0;
    for (size_t i = 0; i < source_poses.size(); ++i)
    {
        Eigen::Vector3d projected_point = similarity_transform.transformFrom(source_poses[i].translation());
        double error = (projected_point - target_poses[i].translation()).norm();
        total_error += error;
    }
    double mean_error = total_error / source_poses.size();
    std::cout << "Mean projection error: " << mean_error << std::endl;

    return 0;
}
