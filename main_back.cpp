#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Similarity3.h>

using namespace std;
using namespace Eigen;
using namespace gtsam;

// Function to generate Gaussian noise
double generateGaussianNoise(double mean, double stddev)
{
    static std::mt19937 generator(std::random_device{}());
    std::normal_distribution<double> distribution(mean, stddev);
    return distribution(generator);
}

vector<Vector3d> generateSourceTrajectory(int n_poses, double noise_stddev)
{
    vector<Vector3d> source_positions;
    for (int i = 0; i < n_poses; ++i)
    {
        Vector3d pos(0, i, 0);
        pos += Vector3d(generateGaussianNoise(1.05, noise_stddev),
                        generateGaussianNoise(0.01, noise_stddev),
                        generateGaussianNoise(1.2, noise_stddev));
        std::cout << "pos " << i << ": " << pos.transpose() << std::endl;
        source_positions.push_back(pos);
    }
    return source_positions;
}

vector<Rot3> generateSourceOrientations(int n_poses, double noise_stddev)
{
    vector<Rot3> source_orientations;
    for (int i = 0; i < n_poses; ++i)
    {
        Vector3d euler_angles(0, 0, 0);
        euler_angles += Vector3d(generateGaussianNoise(0.02, noise_stddev),
                                 generateGaussianNoise(0.04, noise_stddev),
                                 generateGaussianNoise(0.05, noise_stddev));
        std::cout << "euler_angles " << i << ": " << euler_angles.transpose() << std::endl;
        source_orientations.push_back(Rot3::RzRyRx(euler_angles[0], euler_angles[1], euler_angles[2]));
    }
    return source_orientations;
}

pair<vector<Vector3d>, vector<Rot3>> applySimilarityTransform(
    const vector<Vector3d> &positions,
    const vector<Rot3> &orientations,
    const Matrix3d &R,
    const Vector3d &t,
    double s)
{
    vector<Vector3d> transformed_positions;
    vector<Rot3> transformed_orientations;
    for (size_t i = 0; i < positions.size(); ++i)
    {
        Vector3d new_pos = s * (R * positions[i]) + t;
        Rot3 new_ori = Rot3(R * orientations[i].matrix());
        transformed_positions.push_back(new_pos);
        transformed_orientations.push_back(new_ori);
    }
    return {transformed_positions, transformed_orientations};
}

pair<vector<Vector3d>, vector<Rot3>> transformToEdn(
    const vector<Vector3d> &positions,
    const vector<Rot3> &orientations,
    const Matrix3d &R_enu_to_edn)
{

    vector<Vector3d> edn_positions;
    vector<Rot3> edn_orientations;
    for (size_t i = 0; i < positions.size(); ++i)
    {
        Vector3d edn_pos = R_enu_to_edn * positions[i];
        Rot3 edn_ori = Rot3(R_enu_to_edn * orientations[i].matrix() * R_enu_to_edn.transpose());
        edn_positions.push_back(edn_pos);
        edn_orientations.push_back(edn_ori);
    }
    return make_pair(edn_positions, edn_orientations);
}

Matrix3d enuToEdnTransform()
{
    Matrix3d R_enu_to_edn;
    R_enu_to_edn << 1, 0, 0,
        0, 0, -1,
        0, 1, 0;
    return R_enu_to_edn;
}

vector<Pose3> loadPosesFromFile(const string &filename)
{
    vector<Pose3> poses;
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "Error opening file: " << filename << endl;
        return poses;
    }

    double tx, ty, tz, qx, qy, qz, qw;
    while (file >> tx >> ty >> tz >> qx >> qy >> qz >> qw)
    {
        Rot3 rotation = Rot3::Quaternion(qw, qx, qy, qz);
        Point3 translation(tx, ty, tz);
        Pose3 pose(rotation, translation);
        poses.push_back(pose);
    }

    file.close();
    return poses;
}

int main()
{
    int n_poses = 100;
    double noise_stddev = 0.2;

    // Step 1: Generate the source trajectory with Gaussian noise
    vector<Vector3d> source_positions = generateSourceTrajectory(n_poses, noise_stddev);
    vector<Rot3> source_orientations = generateSourceOrientations(n_poses, noise_stddev);

    // Step 2: Define the similarity transform
    double angle_x = 5.0 * M_PI / 180.0; // 5 degrees in radians
    double angle_y = 7.0 * M_PI / 180.0; // 7 degrees in radians
    double angle_z = 3.0 * M_PI / 180.0; // 3 degrees in radians
    Matrix3d R_x, R_y, R_z;
    R_x << 1, 0, 0,
        0, cos(angle_x), -sin(angle_x),
        0, sin(angle_x), cos(angle_x);

    R_y << cos(angle_y), 0, sin(angle_y),
        0, 1, 0,
        -sin(angle_y), 0, cos(angle_y);

    R_z << cos(angle_z), -sin(angle_z), 0,
        sin(angle_z), cos(angle_z), 0,
        0, 0, 1;

    Matrix3d R_enu_to_edn = enuToEdnTransform();
    Matrix3d R = R_enu_to_edn * R_z * R_y * R_x; // Apply ENU to EDN transform and then rotate around x, y, and z axes
    Vector3d t(0.1, 1.1, 120.005);
    double s = 1.06;

    // Apply the similarity transform to generate the target trajectory
    auto [target_positions, target_orientations] = applySimilarityTransform(source_positions, source_orientations, R, t, s);

    // Step 3: Transform the target trajectory to EDN coordinate system
    // Matrix3d R_enu_to_edn = enuToEdnTransform();
    // auto [edn_positions, edn_orientations] = transformToEdn(target_positions, target_orientations, R_enu_to_edn);

    // Step 4: Use GTSAM to compute the similarity transformation
    std::vector<std::pair<Pose3, Pose3>> correspondences;

    for (size_t i = 0; i < source_positions.size(); ++i)
    {
        gtsam::Pose3 source_pose(gtsam::Rot3(source_orientations[i]), gtsam::Point3(source_positions[i]));
        gtsam::Pose3 target_pose(gtsam::Rot3(target_orientations[i]), gtsam::Point3(target_positions[i]));
        correspondences.emplace_back(target_pose, source_pose);
    }

    // Use the align function
    Similarity3 similarity_transform = Similarity3::Align(correspondences);

    // Print the ground-truth and computed transformations
    cout << "Ground-truth transformation:" << endl;
    cout << "Rotation:\n"
         << R << endl;
    cout << "Translation: " << t.transpose() << endl;
    cout << "Scale: " << s << endl;

    cout << "Computed transformation:" << endl;
    cout << "Rotation:\n"
         << similarity_transform.rotation().matrix() << endl;
    cout << "Translation: " << similarity_transform.translation().transpose() << endl;
    cout << "Scale: " << similarity_transform.scale() << endl;

    // Step 5: Project the source points to the target using the computed transformation and compute the error
    double total_error = 0.0;
    for (size_t i = 0; i < source_positions.size(); ++i)
    {
        Vector3d projected_point = similarity_transform.transformFrom(source_positions[i]);
        double error = (projected_point - target_positions[i]).norm();
        total_error += error;
    }
    double mean_error = total_error / source_positions.size();
    cout << "Mean projection error: " << mean_error << endl;

    return 0;
}
