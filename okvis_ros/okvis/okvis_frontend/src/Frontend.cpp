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
 *  Created on: Mar 27, 2015
 *      Author: Andreas Forster (an.forster@gmail.com)
 *    Modified: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file Frontend.cpp
 * @brief Source file for the Frontend class.
 * @author Andreas Forster
 * @author Stefan Leutenegger
 * @Modified by Sharmin Rahman
 * @Last Modified: 09/19/2018
 */

#include <brisk/brisk.h>
#include <glog/logging.h>

#include <okvis/Frontend.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/VioKeyframeWindowMatchingAlgorithm.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// cameras and distortions
#include <algorithm>
#include <memory>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <vector>
// Kneip RANSAC
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/FrameAbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRelativePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/FrameRotationOnlySacProblem.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor.
Frontend::Frontend(size_t numCameras)
    : isInitialized_(false),
      numCameras_(numCameras),
      briskDetectionOctaves_(0),
      briskDetectionThreshold_(50.0),
      briskDetectionAbsoluteThreshold_(800.0),
      briskDetectionMaximumKeypoints_(450),
      briskDescriptionRotationInvariance_(true),
      briskDescriptionScaleInvariance_(false),
      briskMatchingThreshold_(60.0),
      matcher_(std::unique_ptr<okvis::DenseMatcher>(new okvis::DenseMatcher(4))),
      keyframeInsertionOverlapThreshold_(0.6),
      keyframeInsertionMatchingRatioThreshold_(0.2) {
  // create mutexes for feature detectors and descriptor extractors
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectorMutexes_.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
  }
  initialiseBriskFeatureDetectors();
}

// Detection and descriptor extraction on a per image basis.
bool Frontend::detectAndDescribe(size_t cameraIndex,
                                 std::shared_ptr<okvis::MultiFrame> frameOut,
                                 const okvis::kinematics::Transformation& T_WC,
                                 const std::vector<cv::KeyPoint>* keypoints) {
  OKVIS_ASSERT_TRUE_DBG(Exception, cameraIndex < numCameras_, "Camera index exceeds number of cameras.");
  std::lock_guard<std::mutex> lock(*featureDetectorMutexes_[cameraIndex]);

  // check there are no keypoints here
  OKVIS_ASSERT_TRUE(Exception, keypoints == nullptr, "external keypoints currently not supported")

  frameOut->setDetector(cameraIndex, featureDetectors_[cameraIndex]);
  frameOut->setExtractor(cameraIndex, descriptorExtractors_[cameraIndex]);

  frameOut->detect(cameraIndex);

  // ExtractionDirection == gravity direction in camera frame
  Eigen::Vector3d g_in_W(0, 0, -1);
  Eigen::Vector3d extractionDirection = T_WC.inverse().C() * g_in_W;
  frameOut->describe(cameraIndex, extractionDirection);

  // set detector/extractor to nullpointer? TODO(later) or not?
  return true;
}

// Matching as well as initialization of landmarks and state.
bool Frontend::dataAssociationAndInitialization(
    okvis::Estimator& estimator,
    okvis::kinematics::Transformation& /*T_WS_propagated*/,  // TODO(sleutenegger): why is this not used here?
    const okvis::VioParameters& params,
    const std::shared_ptr<okvis::MapPointVector> /*map*/,  // TODO(sleutenegger): why is this not used here?
    std::shared_ptr<okvis::MultiFrame> framesInOut,
    bool* asKeyframe,
    bool* asKeySonarframe) {
  // match new keypoints to existing landmarks/keypoints
  // initialise new landmarks (states)
  // outlier rejection by consistency check
  // RANSAC (2D2D / 3D2D)
  // decide keyframe
  // left-right stereo match & init

  // find distortion type
  okvis::cameras::NCameraSystem::DistortionType distortionType = params.nCameraSystem.distortionType(0);
  for (size_t i = 1; i < params.nCameraSystem.numCameras(); ++i) {
    OKVIS_ASSERT_TRUE(
        Exception, distortionType == params.nCameraSystem.distortionType(i), "mixed frame types are not supported yet");
  }
  int num3dMatches = 0;

  // first frame? (did do addStates before, so 1 frame minimum in estimator)
  if (estimator.numFrames() > 2) {
    int requiredMatches = 5;

    double uncertainMatchFraction = 0;
    bool rotationOnly = false;
    bool sonarchangeKf = false;
    // match to last keyframe
    TimerSwitchable matchKeyframesTimer("2.4.1 matchToKeyframes");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        num3dMatches = matchToKeyframes<VioKeyframeWindowMatchingAlgorithm<
            okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, sonarchangeKf, false, &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        num3dMatches = matchToKeyframes<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(), rotationOnly, sonarchangeKf, false, &uncertainMatchFraction);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        num3dMatches = matchToKeyframes<VioKeyframeWindowMatchingAlgorithm<
            okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(), rotationOnly, sonarchangeKf, false, &uncertainMatchFraction);
        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchKeyframesTimer.stop();
    if (!isInitialized_) {
      if (!rotationOnly) {
        isInitialized_ = true;
        LOG(INFO) << "Initialized!";
      }
    }

    if (num3dMatches <= requiredMatches) {
      LOG(WARNING) << "Tracking failure. Number of 3d2d-matches: " << num3dMatches;
    }

    // keyframe decision, at the moment only landmarks that match with keyframe are initialised
    *asKeyframe = *asKeyframe || doWeNeedANewKeyframe(estimator, framesInOut);
//    if(sonarchangeKf == true || (*asKeyframe == true))
//    {
//        *asKeySonarframe = true;
//    }
    // match to last frame
    TimerSwitchable matchToLastFrameTimer("2.4.2 matchToLastFrame");
    switch (distortionType) {
      case okvis::cameras::NCameraSystem::RadialTangential: {
        matchToLastFrame<VioKeyframeWindowMatchingAlgorithm<
            okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(
            estimator, params, framesInOut->id(), false);
        break;
      }
      case okvis::cameras::NCameraSystem::Equidistant: {
        matchToLastFrame<
            VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > >(
            estimator, params, framesInOut->id(), false);
        break;
      }
      case okvis::cameras::NCameraSystem::RadialTangential8: {
        matchToLastFrame<VioKeyframeWindowMatchingAlgorithm<
            okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > >(
            estimator, params, framesInOut->id(), false);

        break;
      }
      default:
        OKVIS_THROW(Exception, "Unsupported distortion type.")
        break;
    }
    matchToLastFrameTimer.stop();
  } else {
    *asKeyframe = true;  // first frame needs to be keyframe
  }
  // do stereo match to get new landmarks
  TimerSwitchable matchStereoTimer("2.4.3 matchStereo");
  switch (distortionType) {
    case okvis::cameras::NCameraSystem::RadialTangential: {
      matchStereo<VioKeyframeWindowMatchingAlgorithm<
          okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion> > >(estimator, params, framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::Equidistant: {
      matchStereo<
          VioKeyframeWindowMatchingAlgorithm<okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion> > >(
          estimator, params, framesInOut);
      break;
    }
    case okvis::cameras::NCameraSystem::RadialTangential8: {
      matchStereo<VioKeyframeWindowMatchingAlgorithm<
          okvis::cameras::PinholeCamera<okvis::cameras::RadialTangentialDistortion8> > >(
          estimator, params, framesInOut);
      break;
    }
    default:
      OKVIS_THROW(Exception, "Unsupported distortion type.")
      break;
  }
  matchStereoTimer.stop();

  return true;
}

// Propagates pose, speeds and biases with given IMU measurements.
bool Frontend::propagation(const okvis::ImuMeasurementDeque& imuMeasurements,
                           const okvis::ImuParameters& imuParams,
                           okvis::kinematics::Transformation& T_WS_propagated,
                           okvis::SpeedAndBias& speedAndBiases,
                           const okvis::Time& t_start,
                           const okvis::Time& t_end,
                           Eigen::Matrix<double, 15, 15>* covariance,
                           Eigen::Matrix<double, 15, 15>* jacobian) const {
  if (imuMeasurements.size() < 2) {
    LOG(WARNING) << "- Skipping propagation as only one IMU measurement has been given to frontend."
                 << " Normal when starting up.";
    return 0;
  }
  int measurements_propagated = okvis::ceres::ImuError::propagation(
      imuMeasurements, imuParams, T_WS_propagated, speedAndBiases, t_start, t_end, covariance, jacobian);

  return measurements_propagated > 0;
}

// Decision whether a new frame should be keyframe or not.
bool Frontend::doWeNeedANewKeyframe(okvis::Estimator& estimator,
                                    std::shared_ptr<okvis::MultiFrame> currentFrame) {
  // Sharmin: Modified for Scale refinement
  // if (estimator.numFrames() < 2) {
  if (estimator.numFrames() < 2 || estimator.stateCount_ < 6) {
    // just starting, so yes, we need this as a new keyframe
    return true;
  }

  if (!isInitialized_) return false;

  double overlap = 0.0;
  double ratio = 0.0;

  uint64_t MatchedSize = 0;
  // go through all the frames and try to match the initialized keypoints
  for (size_t im = 0; im < currentFrame->numFrames(); ++im) {
    // get the hull of all keypoints in current frame
    std::vector<cv::Point2f> frameBPoints, frameBHull;
    std::vector<cv::Point2f> frameBMatches, frameBMatchesHull;
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > frameBLandmarks;

    const size_t numB = currentFrame->numKeypoints(im);
    frameBPoints.reserve(numB);
    frameBLandmarks.reserve(numB);
    Eigen::Vector2d keypoint;
    for (size_t k = 0; k < numB; ++k) {
      currentFrame->getKeypoint(im, k, keypoint);
      // insert it
      frameBPoints.push_back(cv::Point2f(keypoint[0], keypoint[1]));
      // also remember matches
      if (currentFrame->landmarkId(im, k) != 0) {
        frameBMatches.push_back(cv::Point2f(keypoint[0], keypoint[1]));
      }
    }
    MatchedSize = frameBMatches.size();
    if (frameBPoints.size() < 3) continue;
    cv::convexHull(frameBPoints, frameBHull);
    if (frameBMatches.size() < 3) continue;
    cv::convexHull(frameBMatches, frameBMatchesHull);

    // areas
    double frameBArea = cv::contourArea(frameBHull);
    double frameBMatchesArea = cv::contourArea(frameBMatchesHull);

    // overlap area
    double overlapArea = frameBMatchesArea / frameBArea;
    // matching ratio inside overlap area: count
    int pointsInFrameBMatchesArea = 0;
    if (frameBMatchesHull.size() > 2) {
      for (size_t k = 0; k < frameBPoints.size(); ++k) {
        if (cv::pointPolygonTest(frameBMatchesHull, frameBPoints[k], false) > 0) {
          pointsInFrameBMatchesArea++;
        }
      }
    }
//    LOG(INFO) << "MatchedSize: " << MatchedSize;
    double matchingRatio = static_cast<double>(frameBMatches.size()) / static_cast<double>(pointsInFrameBMatchesArea);
    // calculate overlap score
    overlap = std::max(overlapArea, overlap);
    ratio = std::max(matchingRatio, ratio);
  }
  estimator.SetCameraMatchedSize(MatchedSize);
//  LOG(INFO) << "keypoints camera 0: " << currentFrame->numKeypoints(0) << " camera 1: " << currentFrame->numKeypoints(1)
//            << " MatchedSize: " << MatchedSize << " visualfailtime_: " << visualfailtime_;
  // take a decision
  if (overlap > keyframeInsertionOverlapThreshold_ && ratio > keyframeInsertionMatchingRatioThreshold_)
    return false;
  else
  {
      if (isforwardsonarUsed_ && isInitialized_)
      {
            if((currentFrame->numKeypoints(0) >= 80) && (currentFrame->numKeypoints(1) >= 80) && (MatchedSize >= 3))
            {
                visualfailtime_ = 0;
                return true;
            }
            else
            {
                visualfailtime_++;
                if(visualfailtime_ >= 2)
                {
                    visualfailtime_ = 0;
                    return true;
                }
                return false;
            }
      }
      else
         return true;
  }
}

// Match a new multiframe to existing keyframes
template <class MATCHING_ALGORITHM>
int Frontend::matchToKeyframes(okvis::Estimator& estimator,
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool& rotationOnly,
                               bool& sonarchangeKf,
                               bool usePoseUncertainty,
                               double* uncertainMatchFraction,
                               bool removeOutliers) {
  rotationOnly = true;
  sonarchangeKf = false;
  bool sonarchangeKf_tmp = false;
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  int retCtr = 0;
  int numUncertainMatches = 0;

  // go through all the frames and try to match the initialized keypoints
  size_t kfcounter = 0;
  for (size_t age = 1; age < estimator.numFrames(); ++age) {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId)) continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      MATCHING_ALGORITHM matchingAlgorithm(
          estimator, MATCHING_ALGORITHM::Match3D2D, briskMatchingThreshold_, usePoseUncertainty);
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 3D-2D
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();
    }
    kfcounter++;
    if (kfcounter > 2) break;
  }

  kfcounter = 0;
  bool firstFrame = true;
  // Note Sharmin: age = 0 is the current frame. age is a reverse iterator over StateMap
  for (size_t age = 1; age < estimator.numFrames(); ++age) {
    uint64_t olderFrameId = estimator.frameIdByAge(age);
    if (!estimator.isKeyframe(olderFrameId)) continue;
    for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
      MATCHING_ALGORITHM matchingAlgorithm(
          estimator, MATCHING_ALGORITHM::Match2D2D, briskMatchingThreshold_, usePoseUncertainty);
      matchingAlgorithm.setFrames(olderFrameId, currentFrameId, im, im);

      // match 2D-2D for initialization of new (mono-)correspondences
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
      retCtr += matchingAlgorithm.numMatches();
      numUncertainMatches += matchingAlgorithm.numUncertainMatches();
    }

//    LOG(INFO) << "matchToKeyframes retCtr: " << retCtr
//              << " olderId: " << olderFrameId
//              << " currentId: " << currentFrameId;
    bool rotationOnly_tmp = false;
//    bool sonarchangeKf_tmp = false;
    // remove outliers
    // only do RANSAC 3D2D with most recent KF
    if (kfcounter == 0 && isInitialized_)
    {
      size_t ransac3d2dNum = runRansac3d2d(estimator, params.nCameraSystem, estimator.multiFrame(currentFrameId), removeOutliers);
//      if(ransac3d2dNum < 5)
//        runRansac2d2d(estimator, params, currentFrameId, olderFrameId, true, removeOutliers, rotationOnly_tmp, sonarchangeKf_tmp);
    }
    // do RANSAC 2D2D for initialization only
    if (!isInitialized_) {
      runRansac2d2d(estimator, params, currentFrameId, olderFrameId, true, removeOutliers, rotationOnly_tmp, sonarchangeKf_tmp);
    }
//    if(isInitialized_ && retCtr < 50)
//    {
//       for (size_t age = 1; age < estimator.numFrames(); ++age) {
//         uint64_t olderSonarFrameId = estimator.frameIdByAge(age);
//         if (!estimator.isKeySonarframe(olderSonarFrameId)) continue;
//         runRansac2d2d(estimator, params, currentFrameId, olderSonarFrameId, true, removeOutliers, rotationOnly_tmp, sonarchangeKf_tmp);
//         break;
//       }
//    }
//    LOG(INFO) << "isInitialized_: " << isInitialized_;

    // Sharmin: commented for scale
    if (firstFrame) {
      rotationOnly = rotationOnly_tmp;
      sonarchangeKf = sonarchangeKf_tmp;
      firstFrame = false;
    }

    kfcounter++;
    if (kfcounter > 1) break;
  }

  // calculate fraction of safe matches
  if (uncertainMatchFraction && retCtr > 0) {
    *uncertainMatchFraction = static_cast<double>(numUncertainMatches) / static_cast<double>(retCtr);
  }

  return retCtr;
}

// Match a new multiframe to the last frame.
template <class MATCHING_ALGORITHM>
int Frontend::matchToLastFrame(okvis::Estimator& estimator,
                               const okvis::VioParameters& params,
                               const uint64_t currentFrameId,
                               bool usePoseUncertainty,
                               bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // just starting, so yes, we need this as a new keyframe
    return 0;
  }

  uint64_t lastFrameId = estimator.frameIdByAge(1);

  if (estimator.isKeyframe(lastFrameId)) {
    // already done
    return 0;
  }

  int retCtr = 0;
  bool sonarchangeKf = false;

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    MATCHING_ALGORITHM matchingAlgorithm(
        estimator, MATCHING_ALGORITHM::Match3D2D, briskMatchingThreshold_, usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 3D-2D
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
  }

  runRansac3d2d(estimator, params.nCameraSystem, estimator.multiFrame(currentFrameId), removeOutliers);

  for (size_t im = 0; im < params.nCameraSystem.numCameras(); ++im) {
    MATCHING_ALGORITHM matchingAlgorithm(
        estimator, MATCHING_ALGORITHM::Match2D2D, briskMatchingThreshold_, usePoseUncertainty);
    matchingAlgorithm.setFrames(lastFrameId, currentFrameId, im, im);

    // match 2D-2D for initialization of new (mono-)correspondences
    matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    retCtr += matchingAlgorithm.numMatches();
  }
//  LOG(INFO) << "matchToLastFrame retCtr: " << retCtr
//            << " lastId: " << lastFrameId
//            << " currentId: " << currentFrameId;
  // remove outliers
  bool rotationOnly = false;
  if (!isInitialized_)
    runRansac2d2d(estimator, params, currentFrameId, lastFrameId, false, removeOutliers, rotationOnly, sonarchangeKf);
//  if(isInitialized_ && retCtr < 50 && isforwardsonarUsed_){
//    const size_t numCameras = params.nCameraSystem.numCameras();
//    for (size_t im = 0; im < numCameras; ++im) {
//      okvis::ForwardSonarMeasurement forwardsonarframeA = estimator.forwardsonarFrame(lastSonarFrameId);
//      okvis::ForwardSonarMeasurement forwardsonarframeB = estimator.forwardsonarFrame(currentFrameId);
////////          LOG(INFO) << "frameA id " << olderFrameId << ": " << forwardsonarframeA.measurement.keypoints.size()
////////                    << " frameB pts " << currentFrameId << ": " << forwardsonarframeB.measurement.keypoints.size();
//      std::vector<Eigen::Vector3d> lastpts;
//      std::vector<Eigen::Vector3d> curpts;
//      estimator.forwardsonarMatching(forwardsonarframeA, forwardsonarframeB, lastpts, curpts);
//      LOG(INFO) << "matchToLastFrame lastpts size: " << lastpts.size()
//                << " curpts size: " << curpts.size()
//                << " frameA id " << lastSonarFrameId
//                << " frameB id " << currentFrameId;
//      if(curpts.size() < 10)
//      {
//        sonarchangeKf = true;
//        continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!
//      }
//      else{
//          Eigen::Matrix3d R;
//          Eigen::Vector3d t;
//          sonar_pose_estimation(lastpts, curpts, R, t);
//          Eigen::Quaterniond q(R);
//          okvis::kinematics::Transformation T_S1S2(t, q);
//          LOG(INFO) << "matchToLastFrame T_S1S2: " << T_S1S2.T3x4();
//          okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
//          uint64_t idA = lastSonarFrameId;
//          uint64_t id0 = currentFrameId;
//          estimator.getCameraSensorStates(idA, im, T_SCA);
//          estimator.get_T_WS(idA, T_WSA);
//          estimator.getCameraSensorStates(id0, im, T_SC0);
//          estimator.get_T_WS(id0, T_WS0);
////               set.
////          estimator.set_T_WS(id0, T_WSA * T_SCA * (*params.nCameraSystem.T_SC(0)).inverse() * params.sonar.T_SSo * T_S1S2 * T_SC0.inverse());
//          estimator.set_T_WS(id0, T_WSA * params.sonar.T_SSo * T_S1S2 * params.sonar.T_SSo.inverse());
//          if(curpts.size() < 35)
//             sonarchangeKf = true;
//       }
//       break;
//     }
//     LOG(INFO) << "#####matchToLastFrame Point to rare!: " << sonarchangeKf;
//  }
  return retCtr;
}

// Match the frames inside the multiframe to each other to initialise new landmarks.
// Sharmin: modified to add scale-refinement
template <class MATCHING_ALGORITHM>
void Frontend::matchStereo(okvis::Estimator& estimator,
                           const okvis::VioParameters& params,
                           std::shared_ptr<okvis::MultiFrame> multiFrame) {
  bool useSCM = false;  // Sharmin
  const size_t camNumber = multiFrame->numFrames();
  const uint64_t mfId = multiFrame->id();

  for (size_t im0 = 0; im0 < camNumber; im0++) {
    for (size_t im1 = im0 + 1; im1 < camNumber; im1++) {
      // first, check the possibility for overlap
      // TODO(test): implement this in the Multiframe.

      // check overlap
      if (!multiFrame->hasOverlap(im0, im1)) {
        continue;
      }

      MATCHING_ALGORITHM matchingAlgorithm(
          estimator,
          MATCHING_ALGORITHM::Match2D2D,
          briskMatchingThreshold_,
          false);  // TODO(test): make sure this is changed when switching back to uncertainty based matching
      matchingAlgorithm.setFrames(mfId, mfId, im0, im1);  // newest frame

      // match 2D-2D
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);

      // match 3D-2D
      matchingAlgorithm.setMatchingType(MATCHING_ALGORITHM::Match3D2D);
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);

      // match 2D-3D
      matchingAlgorithm.setFrames(mfId, mfId, im1, im0);  // newest frame
      matcher_->match<MATCHING_ALGORITHM>(matchingAlgorithm);
    }
  }

  /***********   Scale Refinement: Added by Sharmin ****************/

  bool rotationOnly_tmp = false;
  bool removeOutliers = true;
  // do RANSAC 2D2D for initialization only
  if (!isScaleRefined_ && numStatesToRefineScale_ <= 5) {
    std::cout << "Performing Ransac2d2d to refine scale." << std::endl;
    int numInliers = runRansac2d2dToRefineScale(estimator, params, mfId, mfId, true, removeOutliers, rotationOnly_tmp);
    std::cout << "ransac2d2d num_inliers: " << numInliers << std::endl;
    if (numInliers > 15) {
      std::cout << "To refine scale: num_state " << numStatesToRefineScale_ << " num_inliers: " << numInliers
                << std::endl;
      numStatesToRefineScale_ += 1;  // Sharmin
    }
  }

  if (!isScaleRefined_ && numStatesToRefineScale_ > 5) {
    int n_state = numStatesToRefineScale_ * 3 + 3 + 1;

    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    for (size_t i = 0; i < ransac2d2d_R_WS.size() - 1; i++) {
      Eigen::MatrixXd tmp_A(6, 10);
      tmp_A.setZero();
      Eigen::VectorXd tmp_b(6);
      tmp_b.setZero();

      double dt = imu_interal_dt.at(i);

      tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
      tmp_A.block<3, 3>(0, 6) = ransac2d2d_R_WS.at(i).transpose() * dt * dt / 2 * Eigen::Matrix3d::Identity();
      tmp_A.block<3, 1>(0, 9) =
          ransac2d2d_R_WS.at(i).transpose() * (ransac2d2d_t_WC.at(i + 1) - ransac2d2d_t_WC.at(i)) / 100.0;
      tmp_b.block<3, 1>(0, 0) =
          imu_interal_deltaP.at(i) +
          ransac2d2d_R_WS.at(i).transpose() * ransac2d2d_R_WS.at(i + 1) * ransac2d2d_t_SC.at(i + 1) -
          ransac2d2d_t_SC.at(i);
      // cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
      tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
      tmp_A.block<3, 3>(3, 3) = ransac2d2d_R_WS.at(i).transpose() * ransac2d2d_R_WS.at(i + 1);
      tmp_A.block<3, 3>(3, 6) = ransac2d2d_R_WS.at(i).transpose() * dt * Eigen::Matrix3d::Identity();
      tmp_b.block<3, 1>(3, 0) = imu_interal_deltaV.at(i);
      // cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

      Eigen::Matrix<double, 6, 6> cov_inv = Eigen::Matrix<double, 6, 6>::Zero();
      // cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
      // MatrixXd cov_inv = cov.inverse();
      cov_inv.setIdentity();

      Eigen::MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
      Eigen::VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;

      A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
      b.segment<6>(i * 3) += r_b.head<6>();

      A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
      b.tail<4>() += r_b.tail<4>();

      A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
      A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }

    A = A * 1000.0;
    b = b * 1000.0;
    Eigen::VectorXd x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;

    std::cout << "================= Scale =================== " << std::endl;
    std::cout << "estimated scale: " << s << std::endl;

    isScaleRefined_ = true;
  }

  // rotationOnly = rotationOnly_tmp;

  /***********   End Scale Refinement: Added by Sharmin ****************/

  // TODO(test): for more than 2 cameras check that there were no duplications!

  // TODO(test): ensure 1-1 matching.

  // TODO(test): no RANSAC ?

  for (size_t im = 0; im < camNumber; im++) {
    const size_t ksize = multiFrame->numKeypoints(im);
    for (size_t k = 0; k < ksize; ++k) {
      if (multiFrame->landmarkId(im, k) != 0) {
        continue;  // already identified correspondence
      }
      multiFrame->setLandmarkId(im, k, okvis::IdProvider::instance().newId());
    }
  }

  // Added by Sharmin
  /*for (size_t im = 0; im < camNumber; im++) {
      const size_t ksize = multiFrame->contour_numKeypoints(im);
      for (size_t k = 0; k < ksize; ++k) {
        if (multiFrame->landmarkId(im, k) != 0) {
          continue;  // already identified correspondence
        }
        multiFrame->setLandmarkId(im, k, okvis::IdProvider::instance().newId());
      }
   }*/
  // End Added by Sharmin
}

// Perform 3D/2D RANSAC.
int Frontend::runRansac3d2d(okvis::Estimator& estimator,
                            const okvis::cameras::NCameraSystem& nCameraSystem,
                            std::shared_ptr<okvis::MultiFrame> currentFrame,
                            bool removeOutliers) {
  if (estimator.numFrames() < 2) {
    // nothing to match against, we are just starting up.
    return 1;
  }

  /////////////////////
  //   KNEIP RANSAC
  /////////////////////
  int numInliers = 0;

  // absolute pose adapter for Kneip toolchain
  opengv::absolute_pose::FrameNoncentralAbsoluteAdapter adapter(estimator, nCameraSystem, currentFrame);

  size_t numCorrespondences = adapter.getNumberCorrespondences();
  if (numCorrespondences < 5) return numCorrespondences;

  // create a RelativePoseSac problem and RANSAC
  opengv::sac::Ransac<opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> ransac;
  std::shared_ptr<opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem> absposeproblem_ptr(
      new opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::FrameAbsolutePoseSacProblem::Algorithm::GP3P));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = 9;
  ransac.max_iterations_ = 50;
  // initial guess not needed...
  // run the ransac
  ransac.computeModel(0);

  // assign transformation
  numInliers = ransac.inliers_.size();
  if (numInliers >= 10) {
    // kick out outliers:
    std::vector<bool> inliers(numCorrespondences, false);
    for (size_t k = 0; k < ransac.inliers_.size(); ++k) {
      inliers.at(ransac.inliers_.at(k)) = true;
    }

    for (size_t k = 0; k < numCorrespondences; ++k) {
      if (!inliers[k]) {
        // get the landmark id:
        size_t camIdx = adapter.camIndex(k);
        size_t keypointIdx = adapter.keypointIndex(k);
        uint64_t lmId = currentFrame->landmarkId(camIdx, keypointIdx);

        // reset ID:
        currentFrame->setLandmarkId(camIdx, keypointIdx, 0);

        // remove observation
        if (removeOutliers) {
          estimator.removeObservation(lmId, currentFrame->id(), camIdx, keypointIdx);
        }
      }
    }
  }
  return numInliers;
}

// Added by Sharmin
// Perform 2D/2D RANSAC.
int Frontend::runRansac2d2dToRefineScale(okvis::Estimator& estimator,
                                         const okvis::VioParameters& params,
                                         uint64_t currentFrameId,
                                         uint64_t olderFrameId,
                                         bool initializePose,
                                         bool removeOutliers,
                                         bool& rotationOnly) {
  // match 2d2d
  rotationOnly = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  size_t totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  // for (size_t im = 0; im < numCameras; ++im) {

  // relative pose adapter for Kneip toolchain
  // Sharmin: to get relative pose in a stereo pair
  opengv::relative_pose::FrameRelativeAdapter adapter(
      estimator, params.nCameraSystem, olderFrameId, 0, currentFrameId, 1);

  size_t numCorrespondences = adapter.getNumberCorrespondences();

//  if (numCorrespondences < 10)
//    return 0;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!

 if (numCorrespondences < 10)
    {
        if (isforwardsonarUsed_)
        {
          std::vector<Eigen::Vector3d> PtsC1, PtsC2;
          rotationOnly = true;
          okvis::ForwardSonarMeasurement forwardsonarframeA = estimator.forwardsonarFrame(olderFrameId);
          okvis::ForwardSonarMeasurement forwardsonarframeB = estimator.forwardsonarFrame(currentFrameId);
//          LOG(INFO) << "frameA id " << olderFrameId << ": " << forwardsonarframeA.measurement.keypoints.size()
//                    << " frameB pts " << currentFrameId << ": " << forwardsonarframeB.measurement.keypoints.size();
          okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
          uint64_t idA = olderFrameId;  // idA, id0 same
          uint64_t id0 = currentFrameId;
          estimator.getCameraSensorStates(idA, 0, T_SCA);  // Sharmin: camIndex = 0
          estimator.get_T_WS(idA, T_WSA);
          estimator.getCameraSensorStates(id0, 1, T_SC0);  // Sharmin: camIndex = 1
          estimator.get_T_WS(id0, T_WS0);
          std::vector<Eigen::Vector3d> lastpts;
          std::vector<Eigen::Vector3d> curpts;
          estimator.forwardsonarMatching(forwardsonarframeA, forwardsonarframeB,
                                         lastpts, curpts);
          for(size_t i = 0; i < curpts.size(); i++)
          {
             okvis::kinematics::Transformation sonar_point(lastpts[i], Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
             okvis::kinematics::Transformation T_PC1 = (*params.nCameraSystem.T_SC(0)).inverse() * params.sonar.T_SSo * sonar_point;
             Eigen::Vector3d PC1 = T_PC1.r();
             PtsC1.push_back(PC1);
          }
          for(size_t i = 0; i < curpts.size(); i++)
          {
             okvis::kinematics::Transformation sonar_point(curpts[i], Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
             okvis::kinematics::Transformation T_PC2 = (*params.nCameraSystem.T_SC(1)).inverse() * params.sonar.T_SSo * sonar_point;
             Eigen::Vector3d PC2 = T_PC2.r();
             PtsC2.push_back(PC2);
          }

          if(curpts.size() < 10)
            return 0;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!
          else{
              Eigen::Matrix3d R;
              Eigen::Vector3d t;
//              sonar_pose_estimation(PtsC1, PtsC2, R, t);
              estimator.sonar_pose_estimation(lastpts, curpts, R, t);
              Eigen::Quaterniond q(R);
              okvis::kinematics::Transformation T_S1S2(t, q);
//              okvis::kinematics::Transformation T_C1C2(t, q);

//              okvis::kinematics::Transformation T_WS_ransac2d2d =
//                T_WSA * T_SCA * T_C1C2 * T_SC0.inverse();
//              ransac2d2d_R_WS.push_back(T_WS_ransac2d2d.q().toRotationMatrix());
//
//              okvis::kinematics::Transformation T_WCA_ransac = T_WS_ransac2d2d * T_SCA;
//              ransac2d2d_t_WC.push_back(T_WCA_ransac.r());
//
//              okvis::kinematics::Transformation T_SCA_ransac =
//                 T_SCA * T_C1C2 * T_SC0.inverse() * T_SCA;
//              ransac2d2d_t_SC.push_back(T_SCA_ransac.r());

////              okvis::kinematics::Transformation T_WS_ransac2d2d =
////                    T_WSA * T_SCA * (*params.nCameraSystem.T_SC(0)).inverse() * params.sonar.T_SSo * T_S1S2 * T_SC0.inverse();
              okvis::kinematics::Transformation T_WS_ransac2d2d = T_WSA * params.sonar.T_SSo * T_S1S2 * params.sonar.T_SSo.inverse();
              ransac2d2d_R_WS.push_back(T_WS_ransac2d2d.q().toRotationMatrix());
//
              okvis::kinematics::Transformation T_WCA_ransac = T_WS_ransac2d2d * T_SCA;
              ransac2d2d_t_WC.push_back(T_WCA_ransac.r());
//
////              okvis::kinematics::Transformation T_SCA_ransac =
////                T_SCA * (*params.nCameraSystem.T_SC(0)).inverse() * params.sonar.T_SSo * T_S1S2 * T_SC0.inverse() * T_SCA;
              okvis::kinematics::Transformation T_SCA_ransac =
                (T_SCA.inverse() * params.sonar.T_SSo * T_S1S2 * params.sonar.T_SSo.inverse()).inverse();
              ransac2d2d_t_SC.push_back(T_SCA_ransac.r());

              Eigen::Vector3d del_p, del_v;
              double del_t;
              estimator.getImuPreIntegral(idA, del_p, del_v, del_t);
              imu_interal_deltaP.push_back(del_p);
              imu_interal_deltaV.push_back(del_v);
              imu_interal_dt.push_back(del_t);
//              if(abs(t[0]) < 0.001 && abs(t[1]) < 0.001 && abs(t[2]) < 0.001)
              return curpts.size();
          }
        }
        else
        {
            return 0;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!
        }
    }
  // try both the rotation-only RANSAC and the relative one:

  // create a RelativePoseSac problem and RANSAC
  typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
  opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
  std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(new FrameRotationOnlySacProblem(adapter));
  rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
  rotation_only_ransac.threshold_ = 9;
  rotation_only_ransac.max_iterations_ = 50;

  // run the ransac
  rotation_only_ransac.computeModel(0);

  // get quality
  int rotation_only_inliers = rotation_only_ransac.inliers_.size();
  float rotation_only_ratio = static_cast<float>(rotation_only_inliers) / static_cast<float>(numCorrespondences);

  // now the rel_pose one:
  typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem FrameRelativePoseSacProblem;
  opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
  std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(
      new FrameRelativePoseSacProblem(adapter, FrameRelativePoseSacProblem::STEWENIUS));
  rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
  rel_pose_ransac.threshold_ = 9;  // (1.0 - cos(0.5/600));
  rel_pose_ransac.max_iterations_ = 50;

  // run the ransac
  rel_pose_ransac.computeModel(0);

  // assess success
  int rel_pose_inliers = rel_pose_ransac.inliers_.size();
  float rel_pose_ratio = static_cast<float>(rel_pose_inliers) / static_cast<float>(numCorrespondences);

  // decide on success and fill inliers
  std::vector<bool> inliers(numCorrespondences, false);
  if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8) {
    if (rotation_only_inliers > 10) {
      rotation_only_success = true;
    }
    rotationOnly = true;
    totalInlierNumber += rotation_only_inliers;
    for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
      inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
    }
  } else {
    if (rel_pose_inliers > 10) {
      rel_pose_success = true;
    }
    totalInlierNumber += rel_pose_inliers;
    for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k) {
      inliers.at(rel_pose_ransac.inliers_.at(k)) = true;
    }
  }

  // failure?
  if (!rel_pose_success) {
    return 0;
  }

  // Sharmin: No need to kick out outliers!
  // Sharmin: This matching/ransac only for computing scale. This is not added/removed from estimator

  // initialize pose if necessary
  if (initializePose && !isInitialized_) {
    if (rel_pose_success) LOG(INFO) << "Refining scale from 2D-2D RANSAC";

    Eigen::Matrix4d T_C1C2_mat = Eigen::Matrix4d::Identity();

    okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
    uint64_t idA = olderFrameId;  // idA, id0 same
    uint64_t id0 = currentFrameId;
    estimator.getCameraSensorStates(idA, 0, T_SCA);  // Sharmin: camIndex = 0
    estimator.get_T_WS(idA, T_WSA);
    estimator.getCameraSensorStates(id0, 1, T_SC0);  // Sharmin: camIndex = 1
    estimator.get_T_WS(id0, T_WS0);
    if (rel_pose_success) {
      // update pose
      // if the IMU is used, this will be quickly optimized to the correct scale. Hopefully.
      T_C1C2_mat.topLeftCorner<3, 4>() = rel_pose_ransac.model_coefficients_;

      // initialize with projected length according to motion prior.

      // CA==C1, C0==C2
      okvis::kinematics::Transformation T_C1C2 = T_SCA.inverse() * T_WSA.inverse() * T_WS0 * T_SC0;
      T_C1C2_mat.topRightCorner<3, 1>() =
          T_C1C2_mat.topRightCorner<3, 1>() *
          std::max(0.0, static_cast<double>(T_C1C2_mat.topRightCorner<3, 1>().transpose() * T_C1C2.r()));
    }
    okvis::kinematics::Transformation T_WS_ransac2d2d =
        T_WSA * T_SCA * okvis::kinematics::Transformation(T_C1C2_mat) * T_SC0.inverse();
    ransac2d2d_R_WS.push_back(T_WS_ransac2d2d.q().toRotationMatrix());

    okvis::kinematics::Transformation T_WCA_ransac = T_WS_ransac2d2d * T_SCA;
    ransac2d2d_t_WC.push_back(T_WCA_ransac.r());

    okvis::kinematics::Transformation T_SCA_ransac =
        T_SCA * okvis::kinematics::Transformation(T_C1C2_mat) * T_SC0.inverse() * T_SCA;
    ransac2d2d_t_SC.push_back(T_SCA_ransac.r());

    Eigen::Vector3d del_p, del_v;
    double del_t;
    estimator.getImuPreIntegral(idA, del_p, del_v, del_t);
    imu_interal_deltaP.push_back(del_p);
    imu_interal_deltaV.push_back(del_v);
    imu_interal_dt.push_back(del_t);

    // set.
    // TODO(Sharmin)
    // estimator.set_T_WS(
    //    id0, T_WS_ransac2d2d);
  }
  //}

  if (rel_pose_success) {
    return totalInlierNumber;
  } else {
    // rotationOnly = true;  // hack...
    return -1;
  }

  return 0;
}

//void Frontend::sonar_pose_estimation(const std::vector<Eigen::Vector3d> &pts1,
//                                     const std::vector<Eigen::Vector3d> &pts2,
//                                     Eigen::Matrix3d &R, Eigen::Vector3d &t) {
//    Eigen::Vector3d p1(0,0,0);
//    Eigen::Vector3d p2(0,0,0);     // center of mass
//    int N = pts1.size();
//    for (int i = 0; i < N; i++) {
//        p1[0] += pts1[i][0];
//        p1[1] += pts1[i][1];
//        p1[2] += pts1[i][2];
//        p2[0] += pts2[i][0];
//        p2[1] += pts2[i][1];
//        p2[2] += pts2[i][2];
//    }
//    p1[0] = p1[0] / N;
//    p1[1] = p1[1] / N;
//    p1[2] = p1[2] / N;
//    p2[0] = p2[0] / N;
//    p2[1] = p2[1] / N;
//    p2[2] = p2[2] / N;
//    std::vector<Eigen::Vector3d> q1, q2; // remove the center
//    q1.resize(N);
//    q2.resize(N);
//    for (int i = 0; i < N; i++) {
//        q1[i][0] = pts1[i][0] - p1[0];
//        q1[i][1] = pts1[i][1] - p1[1];
//        q1[i][2] = pts1[i][2] - p1[2];
//        q2[i][0] = pts2[i][0] - p2[0];
//        q2[i][1] = pts2[i][1] - p2[1];
//        q2[i][2] = pts2[i][2] - p2[2];
//    }
//
//    // compute q1*q2^T
//    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
//    for (int i = 0; i < N; i++) {
//        W += Eigen::Vector3d(q1[i][0], q1[i][1], q1[i][2]) * Eigen::Vector3d(q2[i][0], q2[i][1], q2[i][2]).transpose();
//    }
////    cout << "W=" << W << endl;
//
//    // SVD on W
//    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
//    Eigen::Matrix3d U = svd.matrixU();
//    Eigen::Matrix3d V = svd.matrixV();
//
////    cout << "U=" << U << endl;
////    cout << "V=" << V << endl;
//
//    R = U * (V.transpose());
//    if (R.determinant() < 0) {
//        R = -R;
//    }
//    t = Eigen::Vector3d(p1[0], p1[1], p1[2]) - R * Eigen::Vector3d(p2[0], p2[1], p2[2]);
//}

//void Frontend::sonar_pose_estimation(const std::vector<cv::Point3f> &pts1,
//                                     const std::vector<cv::Point3f> &pts2,
//                                     Eigen::Matrix3d &R, Eigen::Vector3d &t) {
//    Point3f p1, p2;     // center of mass
//    int N = pts1.size();
//    for (int i = 0; i < N; i++) {
//        p1 += pts1[i];
//        p2 += pts2[i];
//    }
//    p1 = Point3f(Vec3f(p1) / N);
//    p2 = Point3f(Vec3f(p2) / N);
//    vector<Point3f> q1(N), q2(N); // remove the center
//    for (int i = 0; i < N; i++) {
//        q1[i] = pts1[i] - p1;
//        q2[i] = pts2[i] - p2;
//    }
//
//    // compute q1*q2^T
//    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
//    for (int i = 0; i < N; i++) {
//        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
//    }
////    cout << "W=" << W << endl;
//
//    // SVD on W
//    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
//    Eigen::Matrix3d U = svd.matrixU();
//    Eigen::Matrix3d V = svd.matrixV();
//
////    cout << "U=" << U << endl;
////    cout << "V=" << V << endl;
//
//    R = U * (V.transpose());
//    if (R.determinant() < 0) {
//        R = -R;
//    }
//    t = Eigen::Vector3d(p1.x, p1.y, p1.z) - R * Eigen::Vector3d(p2.x, p2.y, p2.z);
//}

// Perform 2D/2D RANSAC.
int Frontend::runRansac2d2d(okvis::Estimator& estimator,
                            const okvis::VioParameters& params,
                            uint64_t currentFrameId,
                            uint64_t olderFrameId,
                            bool initializePose,
                            bool removeOutliers,
                            bool& rotationOnly,
                            bool& sonarchangeKf) {
  // match 2d2d
  rotationOnly = false;
  sonarchangeKf = false;
  const size_t numCameras = params.nCameraSystem.numCameras();

  size_t totalInlierNumber = 0;
  bool rotation_only_success = false;
  bool rel_pose_success = false;

  // run relative RANSAC
  for (size_t im = 0; im < numCameras; ++im) {
    // relative pose adapter for Kneip toolchain
    opengv::relative_pose::FrameRelativeAdapter adapter(
        estimator, params.nCameraSystem, olderFrameId, im, currentFrameId, im);

    size_t numCorrespondences = adapter.getNumberCorrespondences();
    LOG(INFO) << "numCorrespondences: " << numCorrespondences;
//    if (numCorrespondences < 10)
//      continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!
    // Modified by Shu Pan
    // When the number of Correspondences or points low than a threshold, leverage sonar frame
    if (numCorrespondences < 10)
    {
        if (isforwardsonarUsed_)
        {
          std::vector<Eigen::Vector3d> PtsC1, PtsC2;
          okvis::ForwardSonarMeasurement forwardsonarframeA = estimator.forwardsonarFrame(olderFrameId);
          okvis::ForwardSonarMeasurement forwardsonarframeB = estimator.forwardsonarFrame(currentFrameId);
          LOG(INFO) << "frameA id " << olderFrameId << ": " << forwardsonarframeA.measurement.keypoints.size()
                    << " frameB pts " << currentFrameId << ": " << forwardsonarframeB.measurement.keypoints.size();
          std::vector<Eigen::Vector3d> lastpts;
          std::vector<Eigen::Vector3d> curpts;
          estimator.forwardsonarMatching(forwardsonarframeA, forwardsonarframeB, lastpts, curpts);
//          std::vector<cv::Point3f> lastpts, curpts;
//          estimator.forwardsonaraddMatch(forwardsonarframeA, forwardsonarframeB, lastpts, curpts);
          LOG(INFO) << "lastpts size: " << lastpts.size()
                    << " curpts size: " << curpts.size();
          for(size_t i = 0; i < lastpts.size(); i++)
          {
             okvis::kinematics::Transformation sonar_point(lastpts[i], Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
             okvis::kinematics::Transformation T_PC1 = (*params.nCameraSystem.T_SC(0)).inverse() * params.sonar.T_SSo * sonar_point;
             Eigen::Vector3d PC1 = T_PC1.r();
             PtsC1.push_back(PC1);
          }
          for(size_t i = 0; i < curpts.size(); i++)
          {
             okvis::kinematics::Transformation sonar_point(curpts[i], Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
             okvis::kinematics::Transformation T_PC2 = (*params.nCameraSystem.T_SC(1)).inverse() * params.sonar.T_SSo * sonar_point;
             Eigen::Vector3d PC2 = T_PC2.r();
             PtsC2.push_back(PC2);
          }
          if(curpts.size() < 10)
          {
            sonarchangeKf = true;
            continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!
          }
          else{
              Eigen::Matrix3d R;
              Eigen::Vector3d t;
              estimator.sonar_pose_estimation(lastpts, curpts, R, t);
//              sonar_pose_estimation(PtsC1, PtsC2, R, t);
              Eigen::Quaterniond q(R);
              okvis::kinematics::Transformation T_S1S2(t, q);
//              okvis::kinematics::Transformation T_C1C2(t, q);
              LOG(INFO) << "T_S1S2: " << T_S1S2.T3x4();
              okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
              uint64_t idA = olderFrameId;
              uint64_t id0 = currentFrameId;
              estimator.getCameraSensorStates(idA, im, T_SCA);
              estimator.get_T_WS(idA, T_WSA);
              estimator.getCameraSensorStates(id0, im, T_SC0);
              estimator.get_T_WS(id0, T_WS0);

//              Eigen::Matrix4d T_C1C2_mat = T_C1C2.T();
//              okvis::kinematics::Transformation T_C1C2_est = T_SCA.inverse() * T_WSA.inverse() * T_WS0 * T_SC0;
//              T_C1C2_mat.topRightCorner<3, 1>() =
//                    T_C1C2_mat.topRightCorner<3, 1>() *
//                    std::max(0.0, static_cast<double>(T_C1C2_mat.topRightCorner<3, 1>().transpose() * T_C1C2_est.r()));
//               set.
//              estimator.set_T_WS(id0, T_WSA * T_SCA * (*params.nCameraSystem.T_SC(0)).inverse() * params.sonar.T_SSo * T_S1S2 * T_SC0.inverse());
              estimator.set_T_WS(id0, T_WSA * params.sonar.T_SSo * T_S1S2 * params.sonar.T_SSo.inverse());
//              estimator.set_T_WS(id0, T_WSA * T_SCA * T_C1C2 * T_SC0.inverse());
//              if(abs(t[0]) < 0.001 && abs(t[1]) < 0.001 && abs(t[2]) < 0.001)
              if(isScaleRefined_ == false)
                 rotationOnly = true;
              if(curpts.size() < 30 || (currentFrameId - olderFrameId) > 5)
                 sonarchangeKf = true;
              return 0;
          }
        }
        else
        {
              continue;  // won't generate meaningful results. let's hope the few correspondences we have are all inliers!!
        }
    }
    // try both the rotation-only RANSAC and the relative one:

    // create a RelativePoseSac problem and RANSAC
    typedef opengv::sac_problems::relative_pose::FrameRotationOnlySacProblem FrameRotationOnlySacProblem;
    opengv::sac::Ransac<FrameRotationOnlySacProblem> rotation_only_ransac;
    std::shared_ptr<FrameRotationOnlySacProblem> rotation_only_problem_ptr(new FrameRotationOnlySacProblem(adapter));
    rotation_only_ransac.sac_model_ = rotation_only_problem_ptr;
    rotation_only_ransac.threshold_ = 9;
    rotation_only_ransac.max_iterations_ = 50;

    // run the ransac
    rotation_only_ransac.computeModel(0);

    // get quality
    int rotation_only_inliers = rotation_only_ransac.inliers_.size();
    float rotation_only_ratio = static_cast<float>(rotation_only_inliers) / static_cast<float>(numCorrespondences);

    // now the rel_pose one:
    typedef opengv::sac_problems::relative_pose::FrameRelativePoseSacProblem FrameRelativePoseSacProblem;
    opengv::sac::Ransac<FrameRelativePoseSacProblem> rel_pose_ransac;
    std::shared_ptr<FrameRelativePoseSacProblem> rel_pose_problem_ptr(
        new FrameRelativePoseSacProblem(adapter, FrameRelativePoseSacProblem::STEWENIUS));
    rel_pose_ransac.sac_model_ = rel_pose_problem_ptr;
    rel_pose_ransac.threshold_ = 9;  // (1.0 - cos(0.5/600));
    rel_pose_ransac.max_iterations_ = 50;

    // run the ransac
    rel_pose_ransac.computeModel(0);

    // assess success
    int rel_pose_inliers = rel_pose_ransac.inliers_.size();
    float rel_pose_ratio = static_cast<float>(rel_pose_inliers) / static_cast<float>(numCorrespondences);

    // decide on success and fill inliers
    std::vector<bool> inliers(numCorrespondences, false);
    if (rotation_only_ratio > rel_pose_ratio || rotation_only_ratio > 0.8) {
      if (rotation_only_inliers > 10) {
        rotation_only_success = true;
      }
      rotationOnly = true;
      totalInlierNumber += rotation_only_inliers;
      for (size_t k = 0; k < rotation_only_ransac.inliers_.size(); ++k) {
        inliers.at(rotation_only_ransac.inliers_.at(k)) = true;
      }
    } else {
      if (rel_pose_inliers > 10) {
        rel_pose_success = true;
      }
      totalInlierNumber += rel_pose_inliers;
      for (size_t k = 0; k < rel_pose_ransac.inliers_.size(); ++k) {
        inliers.at(rel_pose_ransac.inliers_.at(k)) = true;
      }
    }
    LOG(INFO) << "rel_pose_inliers: " << rel_pose_inliers << " rotation_only_inliers: " << rotation_only_inliers;
    // failure?
    if (!rotation_only_success && !rel_pose_success) {
      continue;
    }

    // otherwise: kick out outliers!
    std::shared_ptr<okvis::MultiFrame> multiFrame = estimator.multiFrame(currentFrameId);

    for (size_t k = 0; k < numCorrespondences; ++k) {
      size_t idxB = adapter.getMatchKeypointIdxB(k);
      if (!inliers[k]) {
        uint64_t lmId = multiFrame->landmarkId(im, k);
        // reset ID:
        multiFrame->setLandmarkId(im, k, 0);
        // remove observation
        if (removeOutliers) {
          if (lmId != 0 && estimator.isLandmarkAdded(lmId)) {
            estimator.removeObservation(lmId, currentFrameId, im, idxB);
          }
        }
      }
    }

    // initialize pose if necessary
    if (initializePose && !isInitialized_) {
      if (rel_pose_success)
        LOG(INFO) << "Initializing pose from 2D-2D RANSAC";
      else
        LOG(INFO) << "Initializing pose from 2D-2D RANSAC: orientation only";

      Eigen::Matrix4d T_C1C2_mat = Eigen::Matrix4d::Identity();

      okvis::kinematics::Transformation T_SCA, T_WSA, T_SC0, T_WS0;
      uint64_t idA = olderFrameId;
      uint64_t id0 = currentFrameId;
      estimator.getCameraSensorStates(idA, im, T_SCA);
      estimator.get_T_WS(idA, T_WSA);
      estimator.getCameraSensorStates(id0, im, T_SC0);
      estimator.get_T_WS(id0, T_WS0);
      if (rel_pose_success) {
        // update pose
        // if the IMU is used, this will be quickly optimized to the correct scale. Hopefully.
        T_C1C2_mat.topLeftCorner<3, 4>() = rel_pose_ransac.model_coefficients_;

        // initialize with projected length according to motion prior.

        okvis::kinematics::Transformation T_C1C2 = T_SCA.inverse() * T_WSA.inverse() * T_WS0 * T_SC0;
        T_C1C2_mat.topRightCorner<3, 1>() =
            T_C1C2_mat.topRightCorner<3, 1>() *
            std::max(0.0, static_cast<double>(T_C1C2_mat.topRightCorner<3, 1>().transpose() * T_C1C2.r()));
      } else {
        // rotation only assigned...
        T_C1C2_mat.topLeftCorner<3, 3>() = rotation_only_ransac.model_coefficients_;
      }

      // set.
      estimator.set_T_WS(id0, T_WSA * T_SCA * okvis::kinematics::Transformation(T_C1C2_mat) * T_SC0.inverse());
    }
  }

  if (rel_pose_success || rotation_only_success) {
    return totalInlierNumber;
  } else {
    rotationOnly = true;  // hack...
    return -1;
  }

  return 0;
}

// (re)instantiates feature detectors and descriptor extractors. Used after settings changed or at startup.
void Frontend::initialiseBriskFeatureDetectors() {
  for (auto it = featureDetectorMutexes_.begin(); it != featureDetectorMutexes_.end(); ++it) {
    (*it)->lock();
  }
  featureDetectors_.clear();
  descriptorExtractors_.clear();
  for (size_t i = 0; i < numCameras_; ++i) {
    featureDetectors_.push_back(std::shared_ptr<cv::FeatureDetector>(
#ifdef __ARM_NEON__
        new cv::GridAdaptedFeatureDetector(new cv::FastFeatureDetector(briskDetectionThreshold_),
                                           briskDetectionMaximumKeypoints_,
                                           7,
                                           4)));  // from config file, except the 7x4...
#else
        new brisk::ScaleSpaceFeatureDetector<brisk::HarrisScoreCalculator>(briskDetectionThreshold_,
                                                                           briskDetectionOctaves_,
                                                                           briskDetectionAbsoluteThreshold_,
                                                                           briskDetectionMaximumKeypoints_)));
    std::cout << "briskDetectionThreshold_: " << briskDetectionThreshold_ << std::endl;
    std::cout << "briskDetectionOctaves_: " << briskDetectionOctaves_ << std::endl;
    std::cout << "briskDetectionAbsoluteThreshold_: " << briskDetectionAbsoluteThreshold_ << std::endl;
    std::cout << "briskDetectionMaximumKeypoints_: " << briskDetectionMaximumKeypoints_ << std::endl;
#endif
    descriptorExtractors_.push_back(std::shared_ptr<cv::DescriptorExtractor>(
        new brisk::BriskDescriptorExtractor(briskDescriptionRotationInvariance_, briskDescriptionScaleInvariance_)));
  }
  for (auto it = featureDetectorMutexes_.begin(); it != featureDetectorMutexes_.end(); ++it) {
    (*it)->unlock();
  }
}

}  // namespace okvis
