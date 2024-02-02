#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <iostream>
#include <opencv2/core.hpp>

using Timestamp = int64_t;

struct TrackingInfo {
 public:
  TrackingInfo() = default;
  ~TrackingInfo() = default;

  TrackingInfo(const Timestamp timestamp,
               const uint32_t num_tracked_keypoints,
               const uint32_t num_new_keypoints,
               const std::vector<int>& keypoints_per_quadrant,
               const std::vector<int>& covisibilities,
               const std::vector<float>& keypoints_response_strengths,
               const std::vector<float>& points_quality)
      : timestamp_(timestamp),
        num_tracked_keypoints_(num_tracked_keypoints),
        num_new_keypoints_(num_new_keypoints),
        keypoints_per_quadrant_(keypoints_per_quadrant),
        covisibilities_(covisibilities),
        keypoints_response_strengths_(keypoints_response_strengths),
        points_quality_(points_quality){};

  Timestamp timestamp_;
  uint32_t num_tracked_keypoints_;
  uint32_t num_new_keypoints_;
  std::vector<int> keypoints_per_quadrant_;
  std::vector<int> covisibilities_;
  std::vector<float> keypoints_response_strengths_;
  std::vector<float> points_quality_;
};

struct KeyframeInfo {
 public:
  KeyframeInfo() = default;
  ~KeyframeInfo() = default;

  KeyframeInfo(const int64_t& keyframe_index,
               const cv::Mat& image,
               const Eigen::Vector3d& translation,
               const Eigen::Matrix3d& rotation,
               const TrackingInfo& tracking_info,
               const std::vector<cv::Point3f>& keyframe_points,
               const std::vector<cv::KeyPoint>& keypoints,
               const std::vector<Eigen::Vector3i>& points_ids,
               const std::vector<std::vector<int64_t>>& points_covisibilities)
      : keyframe_index_(keyframe_index),
        keyframe_image_(image),
        translation_(translation),
        rotation_(rotation),
        tracking_info_(tracking_info),
        keyfame_points_(keyframe_points),
        cv_keypoints_(keypoints),
        keypoint_ids_(points_ids),
        point_covisibilities_(points_covisibilities) {
    timestamp_ = tracking_info_.timestamp_;
  };

  KeyframeInfo(const Timestamp& timestamp,
               const int64_t& keyframe_index,
               const cv::Mat& image,
               const Eigen::Vector3d& translation,
               const Eigen::Matrix3d& rotation,
               const std::vector<cv::Point3f>& keyframe_points,
               const std::vector<cv::KeyPoint>& keypoints,
               const std::vector<Eigen::Vector3i>& points_ids,
               const std::vector<std::vector<int64_t>>& points_observations)
      : timestamp_(timestamp),
        keyframe_index_(keyframe_index),
        keyframe_image_(image),
        translation_(translation),
        rotation_(rotation),
        keyfame_points_(keyframe_points),
        cv_keypoints_(keypoints),
        keypoint_ids_(points_ids),
        point_covisibilities_(points_observations){};

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Timestamp timestamp_;
  int64_t keyframe_index_;
  cv::Mat keyframe_image_;
  Eigen::Vector3d translation_;
  Eigen::Matrix3d rotation_;
  TrackingInfo tracking_info_;
  std::vector<cv::Point3f> keyfame_points_;
  std::vector<cv::KeyPoint> cv_keypoints_;
  // @Reloc: landmarkId, mfId, keypointIdx related to each point
  std::vector<Eigen::Vector3i> keypoint_ids_;
  std::vector<std::vector<int64_t>> point_covisibilities_;
};

enum TrackingStatus { NOT_INITIALIZED = 0, TRACKING_VIO = 1, TRACKING_PRIMITIVE_ESTIMATOR = 2 };

typedef std::function<void(const uint64_t)> EventCallback;
typedef std::function<void(std::unique_ptr<KeyframeInfo>)> KeyframeCallback;
typedef std::function<void(std::unique_ptr<std::pair<Timestamp, cv::Mat>>)> CVMatCallback;
typedef std::function<void(const std::pair<Timestamp, Eigen::Matrix4d>&)> PoseCallback;
typedef std::function<void(const std::vector<std::pair<Timestamp, Eigen::Matrix4d>>&)> PathCallback;
typedef std::function<void(const std::pair<Timestamp, Eigen::Matrix4d>&,
                           const std::pair<Eigen::Vector3d, Eigen::Vector3d>&)>
    KeframeWithLoopClosureCallback;
typedef std::function<void(const std::vector<std::pair<Timestamp, Eigen::Matrix4d>>&,
                           const std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>&)>
    PathWithLoopClosureCallback;
typedef std::function<void(pcl::PointCloud<pcl::PointXYZRGB>::Ptr&)> PointCloudCallback;

struct CameraCalibration {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Matrix<double, 4, 4> T_imu_cam0_;  ///< Transformation from camera to sensor (IMU) frame.
  Eigen::Vector2i image_dimension_;         ///< Image dimension. [pixels]
  cv::Mat distortion_coefficients_;         ///< Distortion Coefficients.
  Eigen::Vector2d focal_length_;            ///< Focal length.
  Eigen::Vector2d principal_point_;         ///< Principal point.
  std::string distortion_type_;             ///< Distortion type. ('radialtangential' 'plumb_bob' 'equidistant')

  void print() {
    std::cout << "CameraCalibration: " << std::endl;
    std::cout << "T_IMU_Cam0: \n" << T_imu_cam0_ << std::endl;
    std::cout << "Image dimension: " << image_dimension_.transpose() << std::endl;
    std::cout << "Distortion Coefficients: \n" << distortion_coefficients_ << std::endl;
    std::cout << "Focal length: " << focal_length_.transpose() << std::endl;
    std::cout << "Principal point: " << principal_point_.transpose() << std::endl;
    std::cout << "Distortion type: " << distortion_type_ << std::endl;
  }
};