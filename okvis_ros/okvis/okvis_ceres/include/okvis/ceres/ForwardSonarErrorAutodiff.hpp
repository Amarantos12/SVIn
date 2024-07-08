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
 *  Created on: Sep 10, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file SonarError.hpp
 * @brief Header file for the SonarError class.
 * @author Sharmin Rahman
 */

#ifndef INCLUDE_OKVIS_CERES_FORWARDSONARERRORAUTODIFF_HPP_
#define INCLUDE_OKVIS_CERES_FORWARDSONARERRORAUTODIFF_HPP_

#include <okvis/VioParametersReader.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <string>
#include <vector>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <okvis/ceres/PoseLocalParameterization.hpp>

#include "ceres/ceres.h"

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

class ForwardSonarErrorAutodiff{
public:

  ForwardSonarErrorAutodiff(const Eigen::Vector3d& pts1,
                            const Eigen::Vector3d& pts2,
                            const okvis::kinematics::Transformation T_WSL)
      : pts1_(pts1), pts2_(pts2), T_WSL_(T_WSL)
  {
//     setMeasurement(pts1, pts2, T_WSL);
       LOG(INFO) << "###########ForwardSonarErrorAutodiff###########";
  }

  template <typename T>
  bool operator()(const T* const parameters, T* residuals) const {

    LOG(INFO) << "###########operator()###########";
//    Eigen::Matrix<T, 3, 1> translation2(parameters);
//    Eigen::Quaternion<T> quaternion2(parameters[6], parameters[3], parameters[4], parameters[5]);
//    Eigen::Matrix<T, 4, 4> T_WS;
//    T_WS.setIdentity();
//    T_WS.block(0, 0, 3, 3) = quaternion2.toRotationMatrix();
//    T_WS.block(0, 3, 3, 1) = translation2;
//    LOG(INFO) << "T_WS: " << T_WS;
//
//    Eigen::Matrix<T, 3, 1> translation1(T_WSL_.r().cast<T>());
//    Eigen::Quaternion<T> quaternion1(T_WSL_.q().cast<T>());
//    Eigen::Matrix<T, 4, 4> T_WSLast;
//    T_WSLast.setIdentity();
//    T_WSLast.block(0, 0, 3, 3) = quaternion1.toRotationMatrix();
//    T_WSLast.block(0, 3, 3, 1) = translation1;
//    LOG(INFO) << "T_WSLast: " << T_WSLast;
//
////    Eigen::Matrix<T, 4, 4> T_WSo = T_WS * params_.sonar.T_SSo.T().cast<T>();
////    Eigen::Matrix<T, 4, 4> T_WSoL = T_WSLast * params_.sonar.T_SSo.T().cast<T>();
//    Eigen::Matrix<T, 4, 4> T_SoL_So = T_WSLast.inverse() * T_WS;
////
////    Eigen::Matrix<T, 3, 3> T_WSoL_R;
////    Eigen::Matrix<T, 3, 1> T_WSoL_t;
////    T_WSoL_R = T_WSoL.block(0, 0, 3, 3);
////    T_WSoL_t = T_WSoL.block(0, 3, 3, 1);
////
////    Eigen::Matrix<T, 3, 3> T_WSo_R;
////    Eigen::Matrix<T, 3, 1> T_WSo_t;
////    T_WSo_R = T_WSo.block(0, 0, 3, 3);
////    T_WSo_t = T_WSo.block(0, 3, 3, 1);
////
//    Eigen::Matrix<T, 3, 3> T_SoL_So_R;
//    Eigen::Matrix<T, 3, 1> T_SoL_So_t;
//    T_SoL_So_R = T_SoL_So.block(0, 0, 3, 3);
//    T_SoL_So_t = T_SoL_So.block(0, 3, 3, 1);
//
////    Eigen::Matrix<T, 3, 1> T_WSo_p1 = T_WSoL_R * pts1_.cast<T>() + T_WSoL_t;
////    LOG(INFO) << "pts1_: " << pts1_;
////
////    Eigen::Matrix<T, 3, 1> T_WSo_p2 = T_WSo_R * pts2_.cast<T>() + T_WSo_t;
////    LOG(INFO) << "pts2_: " << pts2_;
////
//    Eigen::Matrix<T, 3, 1> error = pts1_.cast<T>() - (T_SoL_So_R * pts2_ + T_SoL_So_t);
//    residuals[0] = error[0];
//    residuals[1] = error[1];
//    residuals[2] = error[2];
//    LOG(INFO) << "error: " << error;

    return true;
  }

protected:
  // the measurement
  Eigen::Vector3d pts1_;
  Eigen::Vector3d pts2_;
  okvis::kinematics::Transformation T_WSL_;

private:
    okvis::VioParameters params_;
};
}
}

#endif /* INCLUDE_OKVIS_CERES_SONARERRORAUTODIFF_HPP_ */