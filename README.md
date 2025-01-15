# GTSAM Visual Odometry (VIO) Project

This project performs a simple sensor fusion of IMU and Visual Odometry (VO) data to estimate the trajectory of a moving platform. It utilizes the GTSAM (Georgia Tech Smoothing and Mapping) library for factor graph optimization, offering both Levenberg-Marquardt and iSAM2 optimization methods.

## Table of Contents

- [GTSAM Visual Odometry (VIO) Project](#gtsam-visual-odometry-vio-project)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Dependencies](#dependencies)
  - [Running the Project](#running-the-project)
  - [License](#license)

## Description

The project includes the following main components:

- **IMU Data Loading**: Loads IMU data from a text file.
- **Visual Odometry Data Loading**: Loads VO data from a text file in tum format.
- **Sensor Fusion**: Performs sensor fusion using GTSAM to estimate the trajectory.
- **Optimization**: Uses either Levenberg-Marquardt or iSAM2 for optimization.
- **Output**: Saves the optimized trajectory to a file.

## Dependencies

- **GTSAM**: Georgia Tech Smoothing and Mapping library.
- **Eigen**: Linear algebra library.
- **CMake**: Build system generator.

## Running the Project

To run the project, use the following command:

```sh
./gtsam_vio OPTIMIZATION_TYPE=(lm|isam) <imu_file.txt> <vo_file.txt>
```

- `OPTIMIZATION_TYPE`: Specify `lm` for Levenberg-Marquardt or `isam` for iSAM2.
- `<imu_file.txt>`: Path to the IMU data file.
- `<vo_file.txt>`: Path to the VO data file.

Example:

```sh
./gtsam_vio isam ./v1_01/imu_data.txt ./v1_01/vo_data.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.