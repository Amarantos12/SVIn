/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file Estimator.cpp
 * @brief Source file for the Estimator class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>

#include <limits>
#include <map>
#include <memory>
#include <okvis/Estimator.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/DepthError.hpp>                      // @Sharmin
#include <okvis/ceres/HomogeneousPointError.hpp>           // @Sharmin
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>  // @Sharmin
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/ceres/SonarError.hpp>  // @Sharmin
//#include <okvis/ceres/ForwardSonarError.hpp>  // @Shu Pan
#include <okvis/ceres/ForwardSonarErrorAutodiff.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <utility>
#include <vector>

#define pi 3.1415926535898

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor if a ceres map is already available.
Estimator::Estimator(std::shared_ptr<okvis::ceres::Map> mapPtr)
    : mapPtr_(mapPtr),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0) {}

// The default constructor.
Estimator::Estimator()
    : mapPtr_(new okvis::ceres::Map()),
      referencePoseId_(0),
      cauchyLossFunctionPtr_(new ::ceres::CauchyLoss(1)),
      huberLossFunctionPtr_(new ::ceres::HuberLoss(1)),
      marginalizationResidualId_(0) {}

Estimator::~Estimator() {}

// Add a camera to the configuration. Sensors can only be added and never removed.
int Estimator::addCamera(const ExtrinsicsEstimationParameters& extrinsicsEstimationParameters) {
  extrinsicsEstimationParametersVec_.push_back(extrinsicsEstimationParameters);
  return extrinsicsEstimationParametersVec_.size() - 1;
}

// Add an IMU to the configuration.
int Estimator::addImu(const ImuParameters& imuParameters) {
  if (imuParametersVec_.size() > 1) {
    LOG(ERROR) << "only one IMU currently supported";
    return -1;
  }
  imuParametersVec_.push_back(imuParameters);
  return imuParametersVec_.size() - 1;
}

// Remove all cameras from the configuration
void Estimator::clearCameras() { extrinsicsEstimationParametersVec_.clear(); }

// Remove all IMUs from the configuration.
void Estimator::clearImus() { imuParametersVec_.clear(); }

void Estimator::forwardsonarMatchW3D(const okvis::ForwardSonarMeasurement& keyforwardsonarMeasurements,
                                     const okvis::ForwardSonarMeasurement& forwardsonarMeasurements,
                                     const okvis::VioParameters& params,
                                     const okvis::kinematics::Transformation& T_WS,
                                     const okvis::kinematics::Transformation& T_WSlast,
                                     std::vector<Eigen::Vector3d>& lastpts,
                                     std::vector<Eigen::Vector3d>& curpts)
{
      std::vector<cv::KeyPoint> last_keypoints = keyforwardsonarMeasurements.measurement.keypoints;
      std::vector<cv::KeyPoint> cur_keypoints = forwardsonarMeasurements.measurement.keypoints;
      cv::Mat lastImage = keyforwardsonarMeasurements.measurement.image.clone();
      cv::Mat curImage = forwardsonarMeasurements.measurement.image.clone();
      cv::Mat last_descriptors = keyforwardsonarMeasurements.measurement.descriptors;
      cv::Mat cur_descriptors = forwardsonarMeasurements.measurement.descriptors;
      double orign_x = curImage.cols / 2;
      double orign_y = curImage.rows;
       // matching
      BFMatcher matcher(NORM_L2);
      std::vector<DMatch> matches;
      matcher.match(last_descriptors, cur_descriptors, matches);

      // good matches
      std::vector<DMatch> goodMatches;
      double minDist = 10000, maxDist = 0;
      for (size_t i = 0; i < matches.size(); i++)
      {
         double dist = matches[i].distance;
         if (dist < minDist)
              minDist = dist;
         if (dist > maxDist)
              maxDist = dist;
      }
      for (size_t i = 0; i < matches.size(); i++)
      {
         double dist = matches[i].distance;
         if (dist < max(2.5 * minDist, 0.02))
              goodMatches.push_back(matches[i]);
      }

      std::vector<Point2f> obj;
      std::vector<Point2f> scene;
      for( unsigned int i = 0; i < goodMatches.size(); i++ )
      {
         obj.push_back( last_keypoints[ goodMatches[i].queryIdx ].pt );
         scene.push_back( cur_keypoints[ goodMatches[i].trainIdx ].pt );
      }
      std::vector<unsigned char> listpoints;
      std::vector<DMatch> RansacMatches;
      if((obj.size() >= 4) || (scene.size() >= 4))
      {
         Mat H = findHomography( obj, scene, RANSAC, 3, listpoints);//计算透视变换
         for (int i = 0; i < listpoints.size();i++)
         {
            if ((int)listpoints[i])
            {
               RansacMatches.push_back(goodMatches[i]);
            }
         }
//         imshow("lastImage", lastImage);
//         imshow("curImage", curImage);
//         cv::Mat ransacMatchesImage;
//         drawMatches(lastImage, last_keypoints, curImage, cur_keypoints, RansacMatches, ransacMatchesImage,\
//            Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//         cv::imshow("ransacMatchesImage", ransacMatchesImage);
//                waitKey(1);
      }
      LOG(INFO) << "RansacMatches size: " << RansacMatches.size();
      if(RansacMatches.size() >= 10)
      {
          okvis::kinematics::Transformation T_WSo = T_WS * params.sonar.T_SSo;
          okvis::kinematics::Transformation T_WSoL = T_WSlast * params.sonar.T_SSo;
          for(int i = 0; i < RansacMatches.size(); i++)
          {
                double angle;
                double range;
                if (last_keypoints[RansacMatches[i].queryIdx].pt.x <= orign_x)
                {
                     double y = orign_y - last_keypoints[RansacMatches[i].queryIdx].pt.y;
                     double x = orign_x - last_keypoints[RansacMatches[i].queryIdx].pt.x;
                     double angle = -1 * (pi / 2 - atan2(y, x));    // sonar angle is 90 - angle
                     double r = sqrt(pow(x, 2) + pow(y, 2));
                     double range = r * resolution_;
                     okvis::kinematics::Transformation sonar_p1(Eigen::Vector3d(range * sin(angle), 0, range * cos(angle)),
                                                                Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
                     okvis::kinematics::Transformation T_WSo_p1 = T_WSoL * sonar_p1;
                     Eigen::Vector3d WSp1 = T_WSo_p1.r();
                     lastpts.push_back(WSp1);
                }
                else
                {
                     double y = orign_y - last_keypoints[RansacMatches[i].queryIdx].pt.y;;
                     double x = last_keypoints[RansacMatches[i].queryIdx].pt.x - orign_x;
                     double angle = (pi / 2 - atan2(y, x));
                     double r = sqrt(pow(x, 2) + pow(y, 2));
                     double range = r * resolution_;
                     okvis::kinematics::Transformation sonar_p1(Eigen::Vector3d(range * sin(angle), 0, range * cos(angle)),
                                                                Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
                     okvis::kinematics::Transformation T_WSo_p1 = T_WSoL * sonar_p1;
                     Eigen::Vector3d WSp1 = T_WSo_p1.r();
                     lastpts.push_back(WSp1);
                 }
                if (cur_keypoints[RansacMatches[i].trainIdx].pt.x <= orign_x)
                {
                    double y = orign_y - cur_keypoints[RansacMatches[i].trainIdx].pt.y;
                    double x = orign_x - cur_keypoints[RansacMatches[i].trainIdx].pt.x;
                    angle = -1 * (pi / 2 - atan2(y, x));    // sonar angle is 90 - angle
                    double r = sqrt(pow(x, 2) + pow(y, 2));
                    range = r * resolution_;
                    okvis::kinematics::Transformation sonar_p2(Eigen::Vector3d(range * sin(angle), 0, range * cos(angle)),
                                                               Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
                    okvis::kinematics::Transformation T_WSo_p2 = T_WSo * sonar_p2;
                    Eigen::Vector3d WSp2 = T_WSo_p2.r();
                    curpts.push_back(WSp2);
                }
                else
                {
                    double y = orign_y - cur_keypoints[RansacMatches[i].trainIdx].pt.y;;
                    double x = cur_keypoints[RansacMatches[i].trainIdx].pt.x - orign_x;
                    angle = (pi / 2 - atan2(y, x));
                    double r = sqrt(pow(x, 2) + pow(y, 2));
                    range = r * resolution_;
                    okvis::kinematics::Transformation sonar_p2(Eigen::Vector3d(range * sin(angle), 0, range * cos(angle)),
                                                               Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
                    okvis::kinematics::Transformation T_WSo_p2 = T_WSo * sonar_p2;
                    Eigen::Vector3d WSp2 = T_WSo_p2.r();
                    curpts.push_back(WSp2);
                }
          }
      }
}

void Estimator::drawVerticalMatches(const cv::Mat& image1, const std::vector<cv::KeyPoint>& keypoints1,
                         const cv::Mat& image2, const std::vector<cv::KeyPoint>& keypoints2,
                         const std::vector<cv::DMatch>& matches, cv::Mat& output)
{
    int rows1 = image1.rows;
    int cols1 = image1.cols;
    int rows2 = image2.rows;
    int cols2 = image2.cols;

    int outputRows = rows1 + rows2;
    int outputCols = std::max(cols1, cols2);

    output.create(outputRows, outputCols, CV_8UC3);
    output.setTo(cv::Scalar(0, 0, 0));

    cv::Mat roi1(output, cv::Rect(0, 0, cols1, rows1));
    cv::Mat roi2(output, cv::Rect(0, rows1, cols2, rows2));

    cv::cvtColor(image1, roi1, cv::COLOR_GRAY2BGR);
    cv::cvtColor(image2, roi2, cv::COLOR_GRAY2BGR);

    for (const cv::DMatch& match : matches)
    {
        const cv::KeyPoint& kp1 = keypoints1[match.queryIdx];
        const cv::KeyPoint& kp2 = keypoints2[match.trainIdx];

        cv::Point2f pt1(kp1.pt.x, kp1.pt.y);
        cv::Point2f pt2(kp2.pt.x, kp2.pt.y + rows1);

        cv::line(output, pt1, pt2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::circle(output, pt1, 2, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);
        cv::circle(output, pt2, 2, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    }
}

void Estimator::forwardsonarMatching(const okvis::ForwardSonarMeasurement& keyforwardsonarMeasurements,
                                     const okvis::ForwardSonarMeasurement& forwardsonarMeasurements,
                                     std::vector<Eigen::Vector3d>& lastpts,
                                     std::vector<Eigen::Vector3d>& curpts)
{
      std::vector<cv::KeyPoint> last_keypoints = keyforwardsonarMeasurements.measurement.keypoints;
      std::vector<cv::KeyPoint> cur_keypoints = forwardsonarMeasurements.measurement.keypoints;
      cv::Mat lastImage = keyforwardsonarMeasurements.measurement.image.clone();
      cv::Mat curImage = forwardsonarMeasurements.measurement.image.clone();
      cv::Mat last_descriptors = keyforwardsonarMeasurements.measurement.descriptors;
      cv::Mat cur_descriptors = forwardsonarMeasurements.measurement.descriptors;
      double orign_x = curImage.cols / 2;
      double orign_y = curImage.rows;
       // matching
      BFMatcher matcher(NORM_L2);
      std::vector<DMatch> matches;
      matcher.match(last_descriptors, cur_descriptors, matches);

      // good matches
      std::vector<DMatch> goodMatches;
      double minDist = 10000, maxDist = 0;
      for (size_t i = 0; i < matches.size(); i++)
      {
         double dist = matches[i].distance;
         if (dist < minDist)
              minDist = dist;
         if (dist > maxDist)
              maxDist = dist;
      }
      for (size_t i = 0; i < matches.size(); i++)
      {
         double dist = matches[i].distance;
         if (dist < max(1.5 * minDist, 0.02))
              goodMatches.push_back(matches[i]);
      }
//      LOG(INFO) << "!!!!!!!keypoints1: " << last_keypoints.size() << " keypoints2: " << cur_keypoints.size() << " matches: " << goodMatches.size();
      std::vector<Point2f> obj;
      std::vector<Point2f> scene;
      for( unsigned int i = 0; i < goodMatches.size(); i++ )
      {
         obj.push_back( last_keypoints[ goodMatches[i].queryIdx ].pt );
         scene.push_back( cur_keypoints[ goodMatches[i].trainIdx ].pt );
      }
      std::vector<unsigned char> listpoints;
      std::vector<DMatch> RansacMatches;
      if((obj.size() >= 4) || (scene.size() >= 4))
      {
         Mat H = findHomography( obj, scene, RANSAC, 3, listpoints);//计算透视变换
         for (int i = 0; i < listpoints.size();i++)
         {
            if ((int)listpoints[i])
            {
               RansacMatches.push_back(goodMatches[i]);
            }
         }
//         imshow("lastImage", lastImage);
//         imshow("curImage", curImage);
         if(RansacMatches.size() <= 100)
         {
             cv::Mat ransacMatchesImage;
             drawMatches(lastImage, last_keypoints, curImage, cur_keypoints, RansacMatches, ransacMatchesImage,\
                Scalar(0, 255, 0), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //         drawVerticalMatches(lastImage, last_keypoints, curImage, cur_keypoints, RansacMatches, ransacMatchesImage);
             cv::imshow("ransacMatchesImage", ransacMatchesImage);
    //                waitKey(1);
         }
      }
      LOG(INFO) << "RansacMatches size: " << RansacMatches.size();
      if((RansacMatches.size() >= 8) && (RansacMatches.size() <= 80))
      {
          for(int i = 0; i < RansacMatches.size(); i++)
          {
                double angle;
                double range;
                if (last_keypoints[RansacMatches[i].queryIdx].pt.x <= orign_x)
                {
                     double y = orign_y - last_keypoints[RansacMatches[i].queryIdx].pt.y;
                     double x = orign_x - last_keypoints[RansacMatches[i].queryIdx].pt.x;
                     double angle = -1 * (pi / 2 - atan2(y, x));    // sonar angle is 90 - angle
                     double r = sqrt(pow(x, 2) + pow(y, 2));
                     double range = r * resolution_;
                     Eigen::Vector3d WSp1(Eigen::Vector3d(range * sin(angle), 0, range * cos(angle)));
                     lastpts.push_back(WSp1);
                }
                else
                {
                     double y = orign_y - last_keypoints[RansacMatches[i].queryIdx].pt.y;;
                     double x = last_keypoints[RansacMatches[i].queryIdx].pt.x - orign_x;
                     double angle = (pi / 2 - atan2(y, x));
                     double r = sqrt(pow(x, 2) + pow(y, 2));
                     double range = r * resolution_;
                     Eigen::Vector3d WSp1(Eigen::Vector3d(range * sin(angle), 0, range * cos(angle)));
                     lastpts.push_back(WSp1);
                 }
                if (cur_keypoints[RansacMatches[i].trainIdx].pt.x <= orign_x)
                {
                    double y = orign_y - cur_keypoints[RansacMatches[i].trainIdx].pt.y;
                    double x = orign_x - cur_keypoints[RansacMatches[i].trainIdx].pt.x;
                    angle = -1 * (pi / 2 - atan2(y, x));    // sonar angle is 90 - angle
                    double r = sqrt(pow(x, 2) + pow(y, 2));
                    range = r * resolution_;
                    Eigen::Vector3d WSp2(Eigen::Vector3d(range * sin(angle), 0, range * cos(angle)));
                    curpts.push_back(WSp2);
                }
                else
                {
                    double y = orign_y - cur_keypoints[RansacMatches[i].trainIdx].pt.y;;
                    double x = cur_keypoints[RansacMatches[i].trainIdx].pt.x - orign_x;
                    angle = (pi / 2 - atan2(y, x));
                    double r = sqrt(pow(x, 2) + pow(y, 2));
                    range = r * resolution_;
                    Eigen::Vector3d WSp2(Eigen::Vector3d(range * sin(angle), 0, range * cos(angle)));
                    curpts.push_back(WSp2);
                }
          }
      }
}

void Estimator::sonar_pose_estimation(const std::vector<Eigen::Vector3d> &pts1,
                                     const std::vector<Eigen::Vector3d> &pts2,
                                     Eigen::Matrix3d &R, Eigen::Vector3d &t) {
    Eigen::Vector3d p1(0,0,0);
    Eigen::Vector3d p2(0,0,0);     // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1[0] += pts1[i][0];
        p1[1] += pts1[i][1];
        p1[2] += pts1[i][2];
        p2[0] += pts2[i][0];
        p2[1] += pts2[i][1];
        p2[2] += pts2[i][2];
    }
    p1[0] = p1[0] / N;
    p1[1] = p1[1] / N;
    p1[2] = p1[2] / N;
    p2[0] = p2[0] / N;
    p2[1] = p2[1] / N;
    p2[2] = p2[2] / N;
    std::vector<Eigen::Vector3d> q1, q2; // remove the center
    q1.resize(N);
    q2.resize(N);
    for (int i = 0; i < N; i++) {
        q1[i][0] = pts1[i][0] - p1[0];
        q1[i][1] = pts1[i][1] - p1[1];
        q1[i][2] = pts1[i][2] - p1[2];
        q2[i][0] = pts2[i][0] - p2[0];
        q2[i][1] = pts2[i][1] - p2[1];
        q2[i][2] = pts2[i][2] - p2[2];
    }

    // compute q1*q2^T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i][0], q1[i][1], q1[i][2]) * Eigen::Vector3d(q2[i][0], q2[i][1], q2[i][2]).transpose();
    }
//    cout << "W=" << W << endl;

    // SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

//    cout << "U=" << U << endl;
//    cout << "V=" << V << endl;

    R = U * (V.transpose());
    if (R.determinant() < 0) {
        R = -R;
    }
    t = Eigen::Vector3d(p1[0], p1[1], p1[2]) - R * Eigen::Vector3d(p2[0], p2[1], p2[2]);
}

// Add a pose to the state.
bool Estimator::addStates(okvis::MultiFramePtr multiFrame,
                          const okvis::ImuMeasurementDeque& imuMeasurements,
                          const okvis::VioParameters& params,                    /* @Sharmin */
                          const okvis::ForwardSonarMeasurement& keyforwardsonarMeasurements, /* @ShuPan */
                          const okvis::ForwardSonarMeasurement& forwardsonarMeasurements, /* @ShuPan */
                          const okvis::DepthMeasurementDeque& depthMeasurements,
                          double firstDepth, /* @Sharmin */
                          bool asKeyframe,
                          okvis::kinematics::Transformation T_WSL,
                          double sonar_resolution) {
  // Note Sharmin: this is for imu propagation no matter isScaleRefined_ is true/false.
  // TODO(Sharmin): Start actual optimization when isScaleRefined_ = true.

  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBias speedAndBias;
  std::vector<Eigen::Vector3d> lastpts;
  std::vector<Eigen::Vector3d> curpts;
  okvis::kinematics::Transformation T_So1_So2;
  resolution_ = sonar_resolution;
  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    bool success0 = initPoseFromImu(imuMeasurements, T_WS);
    OKVIS_ASSERT_TRUE_DBG(Exception, success0, "pose could not be initialized from imu measurements.");
    if (!success0) return false;

    // Sharmin
    if (multiFrame->numKeypoints() > 15) {
      std::cout << "Initialized!! As enough keypoints are found" << std::endl;
      std::cout << "Initial T_WS: " << T_WS.parameters();
    } else {
      return false;
    }
    // End Sharmin

    speedAndBias.setZero();
    speedAndBias.segment<3>(6) = imuParametersVec_.at(0).a0;
  } else {
    // get the previous states
    uint64_t T_WS_id = statesMap_.rbegin()->second.id;  // n-th state
    uint64_t speedAndBias_id =
        statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).id;
    OKVIS_ASSERT_TRUE_DBG(
        Exception, mapPtr_->parameterBlockExists(T_WS_id), "this is an okvis bug. previous pose does not exist.");
    T_WS = std::static_pointer_cast<ceres::PoseParameterBlock>(mapPtr_->parameterBlockPtr(T_WS_id))->estimate();
//    LOG(INFO) << "************Estimate time: " << statesMap_.at(T_WS_id).timestamp << " id: " << T_WS_id;
    // OKVIS_ASSERT_TRUE_DBG(
    //    Exception, speedAndBias_id,
    //    "this is an okvis bug. previous speedAndBias does not exist.");
    speedAndBias =
        std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(mapPtr_->parameterBlockPtr(speedAndBias_id))
            ->estimate();

    // propagate pose and speedAndBias
    // @Sharmin: using only imu measurements
    // Modified to add scale
    // Eigen::Matrix<double, 15, 15> covariance; // not used
    // Eigen::Matrix<double, 15, 15> jacobian;  // not used
    Eigen::Vector3d acc_doubleinteg;
    Eigen::Vector3d acc_integ;
    double Del_t;

    if (params.sensorList.isForwardSonarUsed)
    {
//        LOG(INFO) << "numKeypoints(0): " << multiFrame->numKeypoints(0)  << " numKeypoints(1): " << multiFrame->numKeypoints(1) << " CameraMatchedSize_: " << CameraMatchedSize_;
        LOG(INFO) <<" CameraMatchedSize_: " << CameraMatchedSize_;
//        if(multiFrame->numKeypoints(0) < 200 && multiFrame->numKeypoints(1) < 200)
//         {
//              sonarOptimized_ = 1;
//         }
//         else if(multiFrame->numKeypoints(0) < 100 && multiFrame->numKeypoints(1) < 100)
//         {
//              sonarOptimized_ = 2;
//         }
//         else
//             sonarOptimized_ = 0;
         if((CameraMatchedSize_ < 20) && (CameraMatchedSize_ >= 8))
         {
              sonarOptimized_ = 1;
         }
         else if(CameraMatchedSize_ < 8)
         {
              sonarOptimized_ = 2;
         }
         else
             sonarOptimized_ = 0;
//         LOG(INFO) << "sonarOptimized_: " << sonarOptimized_;
         forwardsonarMatching(keyforwardsonarMeasurements, forwardsonarMeasurements, lastpts, curpts);
         Eigen::Matrix3d R;
         Eigen::Vector3d t;
         if(curpts.size() > 0)
         {
            sonar_pose_estimation(lastpts, curpts, R, t);
            Eigen::Quaterniond q(R);
            okvis::kinematics::Transformation T_S1S2(t, q);
            T_S1_S2_ = T_S1S2;
//            if(multiFrame->numKeypoints(0) > 100 && multiFrame->numKeypoints(1) > 100)
//               T_WS = T_WS * params.sonar.T_SSo * T_S1S2 * params.sonar.T_SSo.inverse();
//            else
//            if(multiFrame->numKeypoints(0) < 100 && multiFrame->numKeypoints(1) < 100)
//               T_WS = T_WSL * params.sonar.T_SSo * T_S1S2 * params.sonar.T_SSo.inverse();
          }
     }

    int numUsedImuMeasurements = ceres::ImuError::propagation(imuMeasurements,
                                                              imuParametersVec_.at(0),
                                                              T_WS,
                                                              speedAndBias,
                                                              statesMap_.rbegin()->second.timestamp,
                                                              multiFrame->timestamp(),
                                                              0,
                                                              0,
                                                              acc_doubleinteg,
                                                              acc_integ,
                                                              Del_t);

    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1, "propagation failed");
    if (numUsedImuMeasurements < 1) {
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }

    // Added by Sharmin
    setImuPreIntegral(multiFrame->id(), acc_doubleinteg, acc_integ, Del_t);
  }

  // create a states object:
  States states(asKeyframe, multiFrame->id(), multiFrame->timestamp());
  // Added by Sharmin
  stateCount_ = stateCount_ + 1;
  // LOG (INFO) << "No. of state created: "<< stateCount_;
  // LOG (INFO) << "statesMap_ size: "<< statesMap_.size();

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(
      Exception, statesMap_.find(states.id) == statesMap_.end(), "pose ID" << states.id << " was used before!");

  // create global states
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
      new okvis::ceres::PoseParameterBlock(T_WS, states.id, multiFrame->timestamp()));
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;

  if (statesMap_.empty()) {
    referencePoseId_ = states.id;  // set this as reference pose
    if (!mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d)) {
      return false;
    }
  } else {
    if (!mapPtr_->addParameterBlock(poseParameterBlock, ceres::Map::Pose6d)) {
      return false;
    }
  }

  // End @Sharmin

  // add to buffer
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));
  multiFramePtrMap_.insert(std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator = statesMap_.rbegin();
  lastElementIterator++;

  // cameras:
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer cameraInfos(2);
    cameraInfos.at(CameraSensorStates::T_SCi).exists = true;
    cameraInfos.at(CameraSensorStates::Intrinsics).exists = false;
    if (((extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation < 1e-12) ||
         (extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation < 1e-12)) &&
        (statesMap_.size() > 1)) {
      // use the same block...
      cameraInfos.at(CameraSensorStates::T_SCi).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id;
//      LOG(INFO) << "!!!!!!!!!!!!!!!!!Camera extrinsicsEstimationParametersVec_ too low !!!!!!!!!!!!!";
    } else {
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(
          new okvis::ceres::PoseParameterBlock(T_SC, id, multiFrame->timestamp()));
      if (!mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr, ceres::Map::Pose6d)) {
        return false;
      }
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;
      LOG(INFO) << "!!!!!!!!!!!!!!!!!Camera extrinsicsEstimationParametersVec_ normal !!!!!!!!!!!!!";
    }
    // update the states info
    statesMap_.rbegin()->second.sensors.at(SensorStates::Camera).push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
    SpecificSensorStatesContainer imuInfo(2);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;
    uint64_t id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(
        new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));

    if (!mapPtr_->addParameterBlock(speedAndBiasParameterBlock)) {
      return false;
    }
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;
    statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }

  // @Sharmin
  // Depth
  if (depthMeasurements.size() != 0) {
    // Though there should not be more than one depth data
    double mean_depth = 0.0;
    for (auto depthMeasurements_it = depthMeasurements.begin(); depthMeasurements_it != depthMeasurements.end();
         ++depthMeasurements_it) {
      mean_depth += depthMeasurements_it->measurement.depth;
    }
    mean_depth = mean_depth / depthMeasurements.size();

    double information_depth = 5.0;  // TODO(Sharmin) doublre check with the manual

    std::shared_ptr<ceres::DepthError> depthError(new ceres::DepthError(mean_depth, information_depth, firstDepth));
    mapPtr_->addResidualBlock(depthError, NULL, poseParameterBlock);
    std::cout << "Residual block z: " << (*poseParameterBlock->parameters()) + 2 << std::endl;
  }
  // Sonar
//  std::vector<Eigen::Vector3d> landmarkSubset;
//  Eigen::Vector3d sonar_landmark;
//  double range = 0.0, heading = 0.0;

  // @Sharmin
//  if (sonarMeasurements.size() != 0) {
//    auto last_sonarMeasurement_it = sonarMeasurements.rbegin();
//
//    // Taking the nearest range value to the n+1 th frame
//    range = last_sonarMeasurement_it->measurement.range;
//    heading = last_sonarMeasurement_it->measurement.heading;
//
//    okvis::kinematics::Transformation T_WSo = T_WS * params.sonar.T_SSo;
//
//    okvis::kinematics::Transformation sonar_point(Eigen::Vector3d(range * cos(heading), range * sin(heading), 0.0),
//                                                  Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
//    okvis::kinematics::Transformation T_WSo_point = T_WSo * sonar_point;
//
//    sonar_landmark = T_WSo_point.r();
//    // std::cout << "T_WSo: " << T_WSo.r() << std::endl;
//    // std::cout << "T_WSo_point: " << sonar_landmark << std::endl;
//
//    // may be the reverse searching is faster
//    for (PointMap::const_reverse_iterator rit = landmarksMap_.rbegin(); rit != landmarksMap_.rend(); ++rit) {
//      Eigen::Vector3d visual_landmark;
//      if (fabs(rit->second.point[3]) > 1.0e-8) {
//        visual_landmark = (rit->second.point / rit->second.point[3]).head<3>();
//        // LOG (INFO) << "Visual Landmark: " << visual_landmark;
//      }
//      // double distance_from_sonar = (sonar_landmark - visual_landmark).norm();  //Euclidean distance
//      if (fabs(sonar_landmark[0] - visual_landmark[0]) < 0.1 && fabs(sonar_landmark[1] - visual_landmark[1]) < 0.1 &&
//          fabs(sonar_landmark[2] - visual_landmark[2]) < 0.1) {
//        // TODO(sharmin) parameter!!
//        // searching around 10 cm of sonar landmark
//
//        landmarkSubset.push_back(visual_landmark);
//      }
//    }
//
//    // std::cout << "Size of visual patch: "<<landmarkSubset.size() << std::endl;
//
//    if (landmarkSubset.size() > 0) {
//      // LOG (INFO) << " Sonar added for ceres optimization";
//      // @Sharmin
//      // add sonarError and related addResidualBlock
//      double information_sonar = 1.0;  // TODO(sharmin) calculate properly?
//
//          std::shared_ptr<ceres::SonarError> sonarError(
//              new ceres::SonarError(params, range, heading, information_sonar, landmarkSubset));
//          mapPtr_->addResidualBlock(sonarError, NULL, poseParameterBlock);
//        }
//        // End @Sharmin
//  }
    forwardsonarFramePtrMap_.insert(std::pair<uint64_t, okvis::ForwardSonarMeasurement>(states.id, forwardsonarMeasurements));
//    if ((forwardsonarMeasurements.measurement.keypoints.size() != 0) &&
//        (keyforwardsonarMeasurements.measurement.keypoints.size() != 0)) {
   if (curpts.size() > 0) {
//            Eigen::Vector3d sp1(0,0,0);
//            Eigen::Vector3d sp2(0,0,0);     // center of mass
//            int N = curpts.size();
//            for (int i = 0; i < N; i++) {
//                sp1[0] += lastpts[i][0];
//                sp1[1] += lastpts[i][1];
//                sp1[2] += lastpts[i][2];
//                sp2[0] += curpts[i][0];
//                sp2[1] += curpts[i][1];
//                sp2[2] += curpts[i][2];
//             }
//             sp1[0] = sp1[0] / N;
//             sp1[1] = sp1[1] / N;
//             sp1[2] = sp1[2] / N;
//             sp2[0] = sp2[0] / N;
//             sp2[1] = sp2[1] / N;
//             sp2[2] = sp2[2] / N;
//          for(int i = 0; i < curpts.size(); i++)
//          {
           double information_forwardsonar = 1.0;   // subsea 8
           if(sonarOptimized_ == 0)
                information_forwardsonar = 1.0;    // subsea 90
           if(sonarOptimized_ == 1)
                information_forwardsonar = 5.0;    // subsea 90
           if(sonarOptimized_ == 2)
                information_forwardsonar = 9.0;    // subsea 90
           LOG(INFO) << "information_forwardsonar: " << information_forwardsonar;
//           std::shared_ptr<ceres::SonarError> forwardsonarError(
//                    new ceres::SonarError(params, information_forwardsonar, sp1, sp2, T_WSL, T_S1_S2_));
           std::shared_ptr<ceres::SonarError> forwardsonarError(
                    new ceres::SonarError(params, information_forwardsonar, T_WSL, T_S1_S2_));
           mapPtr_->addResidualBlock(forwardsonarError, NULL, poseParameterBlock);
  }

  // depending on whether or not this is the very beginning, we will add priors or relative terms to the last state:
  if (statesMap_.size() == 1) {
    // let's add a prior
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    information(5, 5) = 1.0e8;
    information(0, 0) = 1.0e8;
    information(1, 1) = 1.0e8;
    information(2, 2) = 1.0e8;
    std::shared_ptr<ceres::PoseError> poseError(new ceres::PoseError(T_WS, information));
    /*auto id2= */ mapPtr_->addResidualBlock(poseError, NULL, poseParameterBlock);
    // mapPtr_->isJacobianCorrect(id2,1.0e-6);

    // sensor states
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
      double translationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_translation;
      double translationVariance = translationStdev * translationStdev;
      double rotationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_orientation;
      double rotationVariance = rotationStdev * rotationStdev;
      if (translationVariance > 1.0e-16 && rotationVariance > 1.0e-16) {
        const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
        std::shared_ptr<ceres::PoseError> cameraPoseError(
            new ceres::PoseError(T_SC, translationVariance, rotationVariance));
        // add to map
        mapPtr_->addResidualBlock(
            cameraPoseError,
            NULL,
            mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id));
        // mapPtr_->isJacobianCorrect(id,1.0e-6);
      } else {
        mapPtr_->setParameterBlockConstant(
            states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id);
      }
    }
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      Eigen::Matrix<double, 6, 1> variances;
      // get these from parameter file
      const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
      const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
      std::shared_ptr<ceres::SpeedAndBiasError> speedAndBiasError(
          new ceres::SpeedAndBiasError(speedAndBias, 1.0, sigma_bg * sigma_bg, sigma_ba * sigma_ba));
      // add to map
      mapPtr_->addResidualBlock(
          speedAndBiasError,
          NULL,
          mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      // mapPtr_->isJacobianCorrect(id,1.0e-6);
    }

    // @Sharmin
    // End @Sharmin
  } else {
    // add IMU error terms
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      std::shared_ptr<ceres::ImuError> imuError(new ceres::ImuError(
          imuMeasurements, imuParametersVec_.at(i), lastElementIterator->second.timestamp, states.timestamp));
      /*::ceres::ResidualBlockId id = */ mapPtr_->addResidualBlock(
          imuError,
          NULL,
          mapPtr_->parameterBlockPtr(lastElementIterator->second.id),
          mapPtr_->parameterBlockPtr(
              lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id),
          mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
      // imuError->setRecomputeInformation(false);
      // mapPtr_->isJacobianCorrect(id,1.0e-9);
      // imuError->setRecomputeInformation(true);
    }

    // add relative sensor state errors
    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
      if (lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id !=
          states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id) {
        // i.e. they are different estimated variables, so link them with a temporal error term
        double dt = (states.timestamp - lastElementIterator->second.timestamp).toSec();
        double translationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation;
        double translationVariance = translationSigmaC * translationSigmaC * dt;
        double rotationSigmaC = extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation;
        double rotationVariance = rotationSigmaC * rotationSigmaC * dt;
        std::shared_ptr<ceres::RelativePoseError> relativeExtrinsicsError(
            new ceres::RelativePoseError(translationVariance, rotationVariance));
        mapPtr_->addResidualBlock(
            relativeExtrinsicsError,
            NULL,
            mapPtr_->parameterBlockPtr(
                lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id),
            mapPtr_->parameterBlockPtr(states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id));
        // mapPtr_->isJacobianCorrect(id,1.0e-6);
      }
    }
    // only camera. this is slightly inconsistent, since the IMU error term contains both
    // a term for global states as well as for the sensor-internal ones (i.e. biases).
    // TODO(sharmin): magnetometer, pressure, ...
  }

  return true;
}

//// @Sharmin
//// Add a sonar landmark.
//// TODO(sharmin) check whether it's okvis::ceres::Map::Sonar or okvis::ceres::Map::HomogeneousPoint
//bool Estimator::addSonarLandmark(uint64_t landmarkId, const Eigen::Vector4d& landmark) {
//  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> pointParameterBlock(
//      new okvis::ceres::HomogeneousPointParameterBlock(landmark, landmarkId));
//  if (!mapPtr_->addParameterBlock(pointParameterBlock, okvis::ceres::Map::HomogeneousPoint)) {
//    return false;
//  }
//
//  /*std::shared_ptr<okvis::ceres::SonarParameterBlock> pointParameterBlock(
//      new okvis::ceres::SonarParameterBlock(landmark, landmarkId));
//  if (!mapPtr_->addParameterBlock(pointParameterBlock,
//                                  okvis::ceres::Map::Sonar)) {
//    return false;
//  }*/
//
//  /*Eigen::Matrix<double,3,3> information = Eigen::Matrix<double,3,3>::Zero();
//          information(0,0) = 1.0; information(1,1) = 1.0; information(2,2) = 1.0;
//  // TODO(Sharmin): check Runtime error, parameterBlockExists = false
//  std::shared_ptr<ceres::SonarError > sonarError(
//                        new ceres::SonarError(landmark, information, landmarkSubset));
//  // add to map
//  mapPtr_->addResidualBlock(sonarError, NULL, pointParameterBlock);*/
//
//  // TODO(Sharmin) check it!!
//  /*std::shared_ptr<okvis::ceres::HomogeneousPointError> homogeneousPointError(
//               new okvis::ceres::HomogeneousPointError(
//                               landmark, 0.01));
//
//  mapPtr_->addResidualBlock(
//               homogeneousPointError, NULL, pointParameterBlock);*/
//
//  // remember
//  double dist = std::numeric_limits<double>::max();
//  if (fabs(landmark[3]) > 1.0e-8) {
//    dist = (landmark / landmark[3]).head<3>().norm();  // euclidean distance
//  }
//  //  (sharmin) check landmarksMap
//  landmarksMap_.insert(std::pair<uint64_t, MapPoint>(landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
//  OKVIS_ASSERT_TRUE_DBG(
//      Exception, isLandmarkAdded(landmarkId), "bug adding sonar landmark: inconsistend landmarkdMap_ with mapPtr_.");
//  return true;
//}
// @Sharmin
// Add a sonar landmark.
// TODO(sharmin) check whether it's okvis::ceres::Map::Sonar or okvis::ceres::Map::HomogeneousPoint
bool Estimator::addSonarLandmark(uint64_t landmarkId,
                                 const okvis::VioParameters& params,
                                 const Eigen::Vector3d& pts1,
                                 const Eigen::Vector3d& pts2,
                                 okvis::kinematics::Transformation& T_WS,
                                 okvis::Time timestamp
                                 )
{
  std::shared_ptr<okvis::ceres::PoseParameterBlock> sonarposeParameterBlock(
     new okvis::ceres::PoseParameterBlock(T_WS, landmarkId, timestamp));
  if (!mapPtr_->addParameterBlock(sonarposeParameterBlock, ceres::Map::Pose6d)) {
     return false;
  }

//  std::shared_ptr<ceres::SonarError> forwardsonarError(
//                  new ceres::SonarError(params, pts1, pts2, T_WSlast_));
//  mapPtr_->addResidualBlock(forwardsonarError, NULL, sonarposeParameterBlock);

  /*std::shared_ptr<okvis::ceres::SonarParameterBlock> pointParameterBlock(
      new okvis::ceres::SonarParameterBlock(landmark, landmarkId));
  if (!mapPtr_->addParameterBlock(pointParameterBlock,
                                  okvis::ceres::Map::Sonar)) {
    return false;
  }*/

  /*Eigen::Matrix<double,3,3> information = Eigen::Matrix<double,3,3>::Zero();
          information(0,0) = 1.0; information(1,1) = 1.0; information(2,2) = 1.0;
  // TODO(Sharmin): check Runtime error, parameterBlockExists = false
  std::shared_ptr<ceres::SonarError > sonarError(
                        new ceres::SonarError(landmark, information, landmarkSubset));
  // add to map
  mapPtr_->addResidualBlock(sonarError, NULL, pointParameterBlock);*/

  // TODO(Sharmin) check it!!
  /*std::shared_ptr<okvis::ceres::HomogeneousPointError> homogeneousPointError(
               new okvis::ceres::HomogeneousPointError(
                               landmark, 0.01));

  mapPtr_->addResidualBlock(
               homogeneousPointError, NULL, pointParameterBlock);*/

  // remember
//  T_WSlast_ = T_WSLast;
//  resolution_ = sonar_resolution;
//  double dist = std::numeric_limits<double>::max();
//  if (fabs(landmark[2]) > 1.0e-8) {
//    dist = (landmark / landmark[2]).head<3>().norm();  // euclidean distance
//  }
//  //  (sharmin) check landmarksMap
//  landmarksMap_.insert(std::pair<uint64_t, MapPoint>(landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
//  OKVIS_ASSERT_TRUE_DBG(
//      Exception, isLandmarkAdded(landmarkId), "bug adding sonar landmark: inconsistend landmarkdMap_ with mapPtr_.");
  return true;
}

//bool Estimator::addSonarParaments(double sonar_resolution,
//                                 okvis::kinematics::Transformation& T_WSLast)
//{
//  T_WSlast_ = T_WSLast;
//  resolution_ = sonar_resolution;
//}

// Add a landmark.
bool Estimator::addLandmark(uint64_t landmarkId, const Eigen::Vector4d& landmark) {
  std::shared_ptr<okvis::ceres::HomogeneousPointParameterBlock> pointParameterBlock(
      new okvis::ceres::HomogeneousPointParameterBlock(landmark, landmarkId));
  if (!mapPtr_->addParameterBlock(pointParameterBlock, okvis::ceres::Map::HomogeneousPoint)) {
    return false;
  }

  // TODO(sharmin) check it!!
  /* std::shared_ptr<okvis::ceres::HomogeneousPointError> homogeneousPointError(
          new okvis::ceres::HomogeneousPointError(
                          landmark, 0.1));

   mapPtr_->addResidualBlock(
          homogeneousPointError, NULL, pointParameterBlock);*/

  // remember
  double dist = std::numeric_limits<double>::max();
  if (fabs(landmark[3]) > 1.0e-8) {
    dist = (landmark / landmark[3]).head<3>().norm();  // euclidean distance
  }
  landmarksMap_.insert(std::pair<uint64_t, MapPoint>(landmarkId, MapPoint(landmarkId, landmark, 0.0, dist)));
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "bug: inconsistend landmarkdMap_ with mapPtr_.");
  return true;
}

// Remove an observation from a landmark.
bool Estimator::removeObservation(::ceres::ResidualBlockId residualBlockId) {
  const ceres::Map::ParameterBlockCollection parameters = mapPtr_->parameters(residualBlockId);
  const uint64_t landmarkId = parameters.at(1).first;
  // remove in landmarksMap
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator it = mapPoint.observations.begin();
       it != mapPoint.observations.end();) {
    if (it->second == uint64_t(residualBlockId)) {
      it = mapPoint.observations.erase(it);
    } else {
      it++;
    }
  }
  // remove residual block
  mapPtr_->removeResidualBlock(residualBlockId);
  return true;
}

// Remove an observation from a landmark, if available.
bool Estimator::removeObservation(uint64_t landmarkId, uint64_t poseId, size_t camIdx, size_t keypointIdx) {
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    for (PointMap::iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      LOG(INFO) << it->first << ", no. obs = " << it->second.observations.size();
    }
    LOG(INFO) << landmarksMap_.at(landmarkId).id;
  }
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");

  okvis::KeypointIdentifier kid(poseId, camIdx, keypointIdx);
  MapPoint& mapPoint = landmarksMap_.at(landmarkId);
  std::map<okvis::KeypointIdentifier, uint64_t>::iterator it = mapPoint.observations.find(kid);
  if (it == landmarksMap_.at(landmarkId).observations.end()) {
    return false;  // observation not present
  }

  // remove residual block
  mapPtr_->removeResidualBlock(reinterpret_cast< ::ceres::ResidualBlockId>(it->second));

  // remove also in local map
  mapPoint.observations.erase(it);

  return true;
}

/**
 * @brief Does a vector contain a certain element.
 * @tparam Class of a vector element.
 * @param vector Vector to search element in.
 * @param query Element to search for.
 * @return True if query is an element of vector.
 */
template <class T>
bool vectorContains(const std::vector<T>& vector, const T& query) {
  for (size_t i = 0; i < vector.size(); ++i) {
    if (vector[i] == query) {
      return true;
    }
  }
  return false;
}

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
bool Estimator::applyMarginalizationStrategy(size_t numKeyframes,
                                             size_t numImuFrames,
                                             okvis::MapPointVector& removedLandmarks) {
  // keep the newest numImuFrames
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  for (size_t k = 0; k < numImuFrames; k++) {
    rit++;
    if (rit == statesMap_.rend()) {
      // nothing to do.
      return true;
    }
  }

  // remove linear marginalizationError, if existing
  if (marginalizationErrorPtr_ && marginalizationResidualId_) {
    bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);
    OKVIS_ASSERT_TRUE_DBG(Exception, success, "could not remove marginalization error");
    marginalizationResidualId_ = 0;
    if (!success) return false;
  }

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;
  std::vector<bool> keepParameterBlocks;

  if (!marginalizationErrorPtr_) {
    marginalizationErrorPtr_.reset(new ceres::MarginalizationError(*mapPtr_.get()));
  }

  // distinguish if we marginalize everything or everything but pose
  std::vector<uint64_t> removeFrames;
  std::vector<uint64_t> removeAllButPose;
  std::vector<uint64_t> allLinearizedFrames;
  size_t countedKeyframes = 0;
  while (rit != statesMap_.rend()) {
    if (!rit->second.isKeyframe || countedKeyframes >= numKeyframes) {
      removeFrames.push_back(rit->second.id);
    } else {
      countedKeyframes++;
    }
    removeAllButPose.push_back(rit->second.id);
    allLinearizedFrames.push_back(rit->second.id);
    ++rit;  // check the next frame
  }

  // marginalize everything but pose:
  for (size_t k = 0; k < removeAllButPose.size(); ++k) {
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeAllButPose[k]);
    for (size_t i = 0; i < it->second.global.size(); ++i) {
      if (i == GlobalStates::T_WS) {
        continue;  // we do not remove the pose here.
      }
      if (!it->second.global[i].exists) {
        continue;  // if it doesn't exist, we don't do anything.
      }
      if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if (checkit->second.global[i].exists && checkit->second.global[i].id == it->second.global[i].id) {
        continue;
      }
      it->second.global[i].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[i].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
        if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }
    // add all error terms of the sensor states.
    for (size_t i = 0; i < it->second.sensors.size(); ++i) {
      for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
        for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) {
          if (i == SensorStates::Camera && k == CameraSensorStates::T_SCi) {
            continue;  // we do not remove the extrinsics pose here.
          }
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) {
            continue;  // we never eliminate fixed blocks.
          }
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          if (checkit->second.sensors[i][j][k].exists &&
              checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id) {
            continue;
          }
          it->second.sensors[i][j][k].exists = false;  // remember we removed
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) {
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
            if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
            }
          }
        }
      }
    }
  }
  // marginalize ONLY pose now:
  bool reDoFixation = false;
  for (size_t k = 0; k < removeFrames.size(); ++k) {
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeFrames[k]);

    // schedule removal - but always keep the very first frame.
    // if(it != statesMap_.begin()){
    if (true) {
      it->second.global[GlobalStates::T_WS].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
      keepParameterBlocks.push_back(false);
    }

    // add remaing error terms
    ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.global[GlobalStates::T_WS].id);

    for (size_t r = 0; r < residuals.size(); ++r) {
      if (std::dynamic_pointer_cast<ceres::PoseError>(
              residuals[r].errorInterfacePtr)) {  // avoids linearising initial pose error
        mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
        reDoFixation = true;
        continue;
      }
      std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
          std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
      if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
        marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
      }
    }

    // add remaining error terms of the sensor states.
    size_t i = SensorStates::Camera;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
      size_t k = CameraSensorStates::T_SCi;
      if (!it->second.sensors[i][j][k].exists) {
        continue;
      }
      if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if (checkit->second.sensors[i][j][k].exists &&
          checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id) {
        continue;
      }
      it->second.sensors[i][j][k].exists = false;  // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(it->second.sensors[i][j][k].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
        if (!reprojectionError) {  // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }

    // now finally we treat all the observations.
    OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size() > 0, "bug");
    uint64_t currentKfId = allLinearizedFrames.at(0);

    {
      for (PointMap::iterator pit = landmarksMap_.begin(); pit != landmarksMap_.end();) {
        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

        // first check if we can skip
        bool skipLandmark = true;
        bool hasNewObservations = false;
        bool justDelete = false;
        bool marginalize = true;
        bool errorTermAdded = false;
        std::map<uint64_t, bool> visibleInFrame;
        size_t obsCount = 0;
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            // since we have implemented the linearisation to account for robustification,
            // we don't kick out bad measurements here any more like
            // if(vectorContains(allLinearizedFrames,poseId)){ ...
            //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
            // }
            if (vectorContains(removeFrames, poseId)) {
              skipLandmark = false;
            }
            if (poseId >= currentKfId) {
              marginalize = false;
              hasNewObservations = true;
            }
            if (vectorContains(allLinearizedFrames, poseId)) {
              visibleInFrame.insert(std::pair<uint64_t, bool>(poseId, true));
              obsCount++;
            }
          }
        }

        if (residuals.size() == 0) {
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        if (skipLandmark) {
          pit++;
          continue;
        }

        // so, we need to consider it.
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            if ((vectorContains(removeFrames, poseId) && hasNewObservations) ||
                (!vectorContains(allLinearizedFrames, poseId) && marginalize)) {
              // ok, let's ignore the observation.
              removeObservation(residuals[r].residualBlockId);
              residuals.erase(residuals.begin() + r);
              r--;
            } else if (marginalize && vectorContains(allLinearizedFrames, poseId)) {
              // TODO(sharmin): consider only the sensible ones for marginalization
              if (obsCount < 2) {  // visibleInFrame.size()
                removeObservation(residuals[r].residualBlockId);
                residuals.erase(residuals.begin() + r);
                r--;
              } else {
                // add information to be considered in marginalization later.
                errorTermAdded = true;
                marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId, false);
              }
            }
            // check anything left
            if (residuals.size() == 0) {
              justDelete = true;
              marginalize = false;
            }
          }
        }

        if (justDelete) {
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }
        if (marginalize && errorTermAdded) {
          paremeterBlocksToBeMarginalized.push_back(pit->first);
          keepParameterBlocks.push_back(false);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        pit++;
      }
    }

    // update book-keeping and go to the next frame
    // if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    if (true) {  ///// DEBUG
      multiFramePtrMap_.erase(it->second.id);
      statesMap_.erase(it->second.id);
    }
  }

  // now apply the actual marginalization
  if (paremeterBlocksToBeMarginalized.size() > 0) {
    std::vector< ::ceres::ResidualBlockId> addedPriors;
    marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
  }

  // update error computation
  if (paremeterBlocksToBeMarginalized.size() > 0) {
    marginalizationErrorPtr_->updateErrorComputation();
  }

  // add the marginalization term again
  if (marginalizationErrorPtr_->num_residuals() == 0) {
    marginalizationErrorPtr_.reset();
  }
  if (marginalizationErrorPtr_) {
    std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> > parameterBlockPtrs;
    marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
    marginalizationResidualId_ = mapPtr_->addResidualBlock(marginalizationErrorPtr_, NULL, parameterBlockPtrs);
    OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_, "could not add marginalization error");
    if (!marginalizationResidualId_) return false;
  }

  if (reDoFixation) {
    // finally fix the first pose properly
    // mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
    okvis::kinematics::Transformation T_WS_0;
    get_T_WS(statesMap_.begin()->first, T_WS_0);
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    information(5, 5) = 1.0e14;
    information(0, 0) = 1.0e14;
    information(1, 1) = 1.0e14;
    information(2, 2) = 1.0e14;
    std::shared_ptr<ceres::PoseError> poseError(new ceres::PoseError(T_WS_0, information));
    mapPtr_->addResidualBlock(poseError, NULL, mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
  }

  return true;
}

// Prints state information to buffer.
void Estimator::printStates(uint64_t poseId, std::ostream& buffer) const {
  buffer << "GLOBAL: ";
  for (size_t i = 0; i < statesMap_.at(poseId).global.size(); ++i) {
    if (statesMap_.at(poseId).global.at(i).exists) {
      uint64_t id = statesMap_.at(poseId).global.at(i).id;
      if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << "(";
      buffer << "id=" << id << ":";
      buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
      if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << ")";
      buffer << ", ";
    }
  }
  buffer << "SENSOR: ";
  for (size_t i = 0; i < statesMap_.at(poseId).sensors.size(); ++i) {
    for (size_t j = 0; j < statesMap_.at(poseId).sensors.at(i).size(); ++j) {
      for (size_t k = 0; k < statesMap_.at(poseId).sensors.at(i).at(j).size(); ++k) {
        if (statesMap_.at(poseId).sensors.at(i).at(j).at(k).exists) {
          uint64_t id = statesMap_.at(poseId).sensors.at(i).at(j).at(k).id;
          if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << "(";
          buffer << "id=" << id << ":";
          buffer << mapPtr_->parameterBlockPtr(id)->typeInfo();
          if (mapPtr_->parameterBlockPtr(id)->fixed()) buffer << ")";
          buffer << ", ";
        }
      }
    }
  }
  buffer << std::endl;
}

// Initialise pose from IMU measurements. For convenience as static.
bool Estimator::initPoseFromImu(const okvis::ImuMeasurementDeque& imuMeasurements,
                                okvis::kinematics::Transformation& T_WS) {
  // set translation to zero, unit rotation
  T_WS.setIdentity();

  if (imuMeasurements.size() == 0) return false;

  // acceleration vector
  Eigen::Vector3d acc_B = Eigen::Vector3d::Zero();
  for (okvis::ImuMeasurementDeque::const_iterator it = imuMeasurements.begin(); it < imuMeasurements.end(); ++it) {
    acc_B += it->measurement.accelerometers;
  }
  acc_B /= static_cast<double>(imuMeasurements.size());
  Eigen::Vector3d e_acc = acc_B.normalized();

  // align with ez_W:
  Eigen::Vector3d ez_W(0.0, 0.0, 1.0);
  Eigen::Matrix<double, 6, 1> poseIncrement;
  poseIncrement.head<3>() = Eigen::Vector3d::Zero();
  poseIncrement.tail<3>() = ez_W.cross(e_acc).normalized();
  double angle = std::acos(ez_W.transpose() * e_acc);
  poseIncrement.tail<3>() *= angle;
  T_WS.oplus(-poseIncrement);

  return true;
}

// Start ceres optimization.
#ifdef USE_OPENMP
void Estimator::optimize(size_t numIter, size_t numThreads, bool verbose)
#else
void Estimator::optimize(size_t numIter,
                         size_t /*numThreads*/,
                         bool verbose)  // avoid warning since numThreads unused
#warning openmp not detected, your system may be slower than expected
#endif

{
  // assemble options
  mapPtr_->options.linear_solver_type = ::ceres::SPARSE_SCHUR;
  // mapPtr_->options.initial_trust_region_radius = 1.0e4;
  // mapPtr_->options.initial_trust_region_radius = 2.0e6;
  // mapPtr_->options.preconditioner_type = ::ceres::IDENTITY;
  mapPtr_->options.trust_region_strategy_type = ::ceres::DOGLEG;
  // mapPtr_->options.trust_region_strategy_type = ::ceres::LEVENBERG_MARQUARDT;
  // mapPtr_->options.use_nonmonotonic_steps = true;
  // mapPtr_->options.max_consecutive_nonmonotonic_steps = 10;
  // mapPtr_->options.function_tolerance = 1e-12;
  // mapPtr_->options.gradient_tolerance = 1e-12;
  // mapPtr_->options.jacobi_scaling = false;
#ifdef USE_OPENMP
  mapPtr_->options.num_threads = numThreads;
#endif
  mapPtr_->options.max_num_iterations = numIter;

  if (verbose) {
    mapPtr_->options.minimizer_progress_to_stdout = true;
  } else {
    mapPtr_->options.minimizer_progress_to_stdout = false;
  }

  // call solver
  mapPtr_->solve();

  // update landmarks
  {
    for (auto it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
      Eigen::MatrixXd H(3, 3);
      mapPtr_->getLhs(it->first, H);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(H);
      Eigen::Vector3d eigenvalues = saes.eigenvalues();
      const double smallest = (eigenvalues[0]);
      const double largest = (eigenvalues[2]);
      if (smallest < 1.0e-12) {
        // this means, it has a non-observable depth
        it->second.quality = 0.0;
      } else {
        // OK, well constrained
        it->second.quality = sqrt(smallest) / sqrt(largest);
      }

      // update coordinates
      it->second.point =
          std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(it->first))
              ->estimate();
    }
  }

  // @Sharmin: Calculating covariance

  // mapPtr_->options_covar.sparse_linear_algebra_library_type = ::ceres::SUITE_SPARSE;
  // mapPtr_->options_covar.algorithm_type = ::ceres::SPARSE_QR;

  /*::ceres::Covariance::Options cov_options;
  ::ceres::Covariance covariance(cov_options);

  std::vector<std::pair<const double*, const double*> > covariance_blocks; // @Sharmin

  // Note: mapPtr_->parameterBlockPtr(this->currentKeyframeId()) is a PoseParameterBlock
  std::cout<< "Covar Cal Type: " << mapPtr_->parameterBlockPtr(this->currentKeyframeId())->typeInfo()<< std::endl;

  // Note: if this is a ceres::Map::Pose6d then get parameterPtr() from T_WS
  const double* pose_block = mapPtr_->parameterBlockPtr(this->currentKeyframeId())->parameters();
  covariance_blocks.push_back(std::make_pair(pose_block, pose_block));

  mapPtr_->computeCovariance(covariance_blocks, pose_block);*/

  // End Calculating covariance

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

// Set a time limit for the optimization process.
bool Estimator::setOptimizationTimeLimit(double timeLimit, int minIterations) {
  if (ceresCallback_ != nullptr) {
    if (timeLimit < 0.0) {
      // no time limit => set minimum iterations to maximum iterations
      ceresCallback_->setMinimumIterations(mapPtr_->options.max_num_iterations);
      return true;
    }
    ceresCallback_->setTimeLimit(timeLimit);
    ceresCallback_->setMinimumIterations(minIterations);
    return true;
  } else if (timeLimit >= 0.0) {
    ceresCallback_ = std::unique_ptr<okvis::ceres::CeresIterationCallback>(
        new okvis::ceres::CeresIterationCallback(timeLimit, minIterations));
    mapPtr_->options.callbacks.push_back(ceresCallback_.get());
    return true;
  }
  // no callback yet registered with ceres.
  // but given time limit is lower than 0, so no callback needed
  return true;
}

// getters
// Get a specific landmark.
bool Estimator::getLandmark(uint64_t landmarkId, MapPoint& mapPoint) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  if (landmarksMap_.find(landmarkId) == landmarksMap_.end()) {
    OKVIS_THROW_DBG(Exception, "landmark with id = " << landmarkId << " does not exist.")
    return false;
  }
  mapPoint = landmarksMap_.at(landmarkId);
  return true;
}

// Checks whether the landmark is initialized.
bool Estimator::isLandmarkInitialized(uint64_t landmarkId) const {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");
  return std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(landmarkId))
      ->initialized();
}

// Get a copy of all the landmarks as a PointMap.
size_t Estimator::getLandmarks(PointMap& landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks = landmarksMap_;
  return landmarksMap_.size();
}

// Get a copy of all the landmark in a MapPointVector. This is for legacy support.
// Use getLandmarks(okvis::PointMap&) if possible.
size_t Estimator::getLandmarks(MapPointVector& landmarks) const {
  std::lock_guard<std::mutex> l(statesMutex_);
  landmarks.clear();
  landmarks.reserve(landmarksMap_.size());
  for (PointMap::const_iterator it = landmarksMap_.begin(); it != landmarksMap_.end(); ++it) {
    landmarks.push_back(it->second);
  }
  return landmarksMap_.size();
}

// Get pose for a given pose ID.
bool Estimator::get_T_WS(uint64_t poseId, okvis::kinematics::Transformation& T_WS) const {
  if (!getGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId, GlobalStates::T_WS, T_WS)) {
    return false;
  }

  return true;
}

// Added by Sharmin
bool Estimator::getImuPreIntegral(uint64_t poseId,
                                  Eigen::Vector3d& acc_doubleintegral,
                                  Eigen::Vector3d& acc_integral,
                                  double& Delta_t) const {
  std::map<uint64_t, imu_integrals>::const_iterator it = imuIntegralsMap_.find(poseId);
  if (it != imuIntegralsMap_.end()) {
    acc_doubleintegral = it->second.acc_doubleintegral;
    acc_integral = it->second.acc_integral;
    Delta_t = it->second.Delta_t;

    // std::cout <<"Accessing Imu integral Values: "<< it->second.acc_doubleintegral << ", "
    //<< it->second.acc_integral<< ", " << it->second.Delta_t <<std::endl;
    return true;
  }
  return false;
}

// Feel free to implement caching for them...
// Get speeds and IMU biases for a given pose ID.
bool Estimator::getSpeedAndBias(uint64_t poseId, uint64_t imuIdx, okvis::SpeedAndBias& speedAndBias) const {
  if (!getSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
          poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBias)) {
    return false;
  }
  return true;
}

// Get camera states for a given pose ID.
bool Estimator::getCameraSensorStates(uint64_t poseId,
                                      size_t cameraIdx,
                                      okvis::kinematics::Transformation& T_SCi) const {
  return getSensorStateEstimateAs<ceres::PoseParameterBlock>(
      poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi, T_SCi);
}

// Get the ID of the current keyframe.
uint64_t Estimator::currentKeyframeId() const {
  for (std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin(); rit != statesMap_.rend(); ++rit) {
    if (rit->second.isKeyframe) {
      return rit->first;
    }
  }
  OKVIS_THROW_DBG(Exception, "no keyframes existing...");
  return 0;
}

// Get the ID of an older frame.
uint64_t Estimator::frameIdByAge(size_t age) const {
  std::map<uint64_t, States>::const_reverse_iterator rit = statesMap_.rbegin();
  for (size_t i = 0; i < age; ++i) {
    ++rit;
    OKVIS_ASSERT_TRUE_DBG(Exception, rit != statesMap_.rend(), "requested age " << age << " out of range.");
  }
  return rit->first;
}

// Get the ID of the newest frame added to the state.
uint64_t Estimator::currentFrameId() const {
  OKVIS_ASSERT_TRUE_DBG(Exception, statesMap_.size() > 0, "no frames added yet.")
  return statesMap_.rbegin()->first;
}

// Checks if a particular frame is still in the IMU window
bool Estimator::isInImuWindow(uint64_t frameId) const {
  if (statesMap_.at(frameId).sensors.at(SensorStates::Imu).size() == 0) {
    return false;  // no IMU added
  }
  return statesMap_.at(frameId).sensors.at(SensorStates::Imu).at(0).at(ImuSensorStates::SpeedAndBias).exists;
}

// Set pose for a given pose ID.
bool Estimator::set_T_WS(uint64_t poseId, const okvis::kinematics::Transformation& T_WS) {
  if (!setGlobalStateEstimateAs<ceres::PoseParameterBlock>(poseId, GlobalStates::T_WS, T_WS)) {
    return false;
  }

  return true;
}

// Added by Sharmin
void Estimator::setImuPreIntegral(uint64_t poseId,
                                  Eigen::Vector3d& acc_doubleintegral,
                                  Eigen::Vector3d& acc_integral,
                                  double& Delta_t) {
  imu_integrals imu_int(acc_doubleintegral, acc_integral, Delta_t);
  imuIntegralsMap_.insert(std::pair<uint64_t, imu_integrals>(poseId, imu_int));
  // std::cout <<"Imu integral Values: "<< imu_int.acc_doubleintegral << ", "
  //<< imu_int.acc_integral<< ", " << imu_int.Delta_t <<std::endl;
}

// Set the speeds and IMU biases for a given pose ID.
bool Estimator::setSpeedAndBias(uint64_t poseId, size_t imuIdx, const okvis::SpeedAndBias& speedAndBias) {
  return setSensorStateEstimateAs<ceres::SpeedAndBiasParameterBlock>(
      poseId, imuIdx, SensorStates::Imu, ImuSensorStates::SpeedAndBias, speedAndBias);
}

// Set the transformation from sensor to camera frame for a given pose ID.
bool Estimator::setCameraSensorStates(uint64_t poseId,
                                      size_t cameraIdx,
                                      const okvis::kinematics::Transformation& T_SCi) {
  return setSensorStateEstimateAs<ceres::PoseParameterBlock>(
      poseId, cameraIdx, SensorStates::Camera, CameraSensorStates::T_SCi, T_SCi);
}

// Set the homogeneous coordinates for a landmark.
bool Estimator::setLandmark(uint64_t landmarkId, const Eigen::Vector4d& landmark) {
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(landmarkId);
#ifndef NDEBUG
  std::shared_ptr<ceres::HomogeneousPointParameterBlock> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(landmark);
#else
  std::static_pointer_cast<ceres::HomogeneousPointParameterBlock>(parameterBlockPtr)->setEstimate(landmark);
#endif

  // also update in map
  landmarksMap_.at(landmarkId).point = landmark;
  return true;
}

// Set the landmark initialization state.
void Estimator::setLandmarkInitialized(uint64_t landmarkId, bool initialized) {
  OKVIS_ASSERT_TRUE_DBG(Exception, isLandmarkAdded(landmarkId), "landmark not added");
  std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(mapPtr_->parameterBlockPtr(landmarkId))
      ->setInitialized(initialized);
}

// private stuff
// getters
bool Estimator::getGlobalStateParameterBlockPtr(uint64_t poseId,
                                                int stateType,
                                                std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW(Exception, "pose with id = " << id << " does not exist.")
    return false;
  }

  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getGlobalStateParameterBlockAs(uint64_t poseId,
                                               int stateType,
                                               PARAMETER_BLOCK_T& stateParameterBlock) const {
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getGlobalStateParameterBlockPtr(poseId, stateType, parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    LOG(INFO) << "--" << parameterBlockPtr->typeInfo();
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception,
                    "wrong pointer type requested: requested " << info->typeInfo() << " but is of type"
                                                               << parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
#endif
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getGlobalStateEstimateAs(uint64_t poseId,
                                         int stateType,
                                         typename PARAMETER_BLOCK_T::estimate_t& state) const {
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getGlobalStateParameterBlockAs(poseId, stateType, stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

bool Estimator::getSensorStateParameterBlockPtr(uint64_t poseId,
                                                int sensorIdx,
                                                int sensorType,
                                                int stateType,
                                                std::shared_ptr<ceres::ParameterBlock>& stateParameterBlockPtr) const {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }
  stateParameterBlockPtr = mapPtr_->parameterBlockPtr(id);
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateParameterBlockAs(uint64_t poseId,
                                               int sensorIdx,
                                               int sensorType,
                                               int stateType,
                                               PARAMETER_BLOCK_T& stateParameterBlock) const {
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getSensorStateParameterBlockPtr(poseId, sensorIdx, sensorType, stateType, parameterBlockPtr)) {
    return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
    OKVIS_THROW_DBG(Exception,
                    "wrong pointer type requested: requested " << info->typeInfo() << " but is of type"
                                                               << parameterBlockPtr->typeInfo())
    return false;
  }
  stateParameterBlock = *derivedParameterBlockPtr;
#else
  stateParameterBlock = *std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
#endif
  return true;
}
template <class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateEstimateAs(uint64_t poseId,
                                         int sensorIdx,
                                         int sensorType,
                                         int stateType,
                                         typename PARAMETER_BLOCK_T::estimate_t& state) const {
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getSensorStateParameterBlockAs(poseId, sensorIdx, sensorType, stateType, stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
}

template <class PARAMETER_BLOCK_T>
bool Estimator::setGlobalStateEstimateAs(uint64_t poseId,
                                         int stateType,
                                         const typename PARAMETER_BLOCK_T::estimate_t& state) {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).global.at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(state);
#endif
  return true;
}

template <class PARAMETER_BLOCK_T>
bool Estimator::setSensorStateEstimateAs(uint64_t poseId,
                                         int sensorIdx,
                                         int sensorType,
                                         int stateType,
                                         const typename PARAMETER_BLOCK_T::estimate_t& state) {
  // check existence in states set
  if (statesMap_.find(poseId) == statesMap_.end()) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  // obtain the parameter block ID
  uint64_t id = statesMap_.at(poseId).sensors.at(sensorType).at(sensorIdx).at(stateType).id;
  if (!mapPtr_->parameterBlockExists(id)) {
    OKVIS_THROW_DBG(Exception, "pose with id = " << poseId << " does not exist.")
    return false;
  }

  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr = mapPtr_->parameterBlockPtr(id);
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
      std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if (!derivedParameterBlockPtr) {
    OKVIS_THROW_DBG(Exception, "wrong pointer type requested.")
    return false;
  }
  derivedParameterBlockPtr->setEstimate(state);
#else
  std::static_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr)->setEstimate(state);
#endif
  return true;
}

}  // namespace okvis
