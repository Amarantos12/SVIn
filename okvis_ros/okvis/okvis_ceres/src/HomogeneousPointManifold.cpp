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
 *  Created on: Aug 30, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file HomogeneousPointManifold.cpp
 * @brief Source file for the HomogeneousPointManifold class.
 * @author Stefan Leutenegger
 */

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/HomogeneousPointManifold.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Generalization of the addition operation,
//        x_plus_delta = Plus(x, delta)
//        with the condition that Plus(x, 0) = x.
bool HomogeneousPointManifold::Plus(const double* x, const double* delta, double* x_plus_delta) const {
  return plus(x, delta, x_plus_delta);
}

// Generalization of the addition operation,
//        x_plus_delta = Plus(x, delta)
//        with the condition that Plus(x, 0) = x.
bool HomogeneousPointManifold::plus(const double* x, const double* delta, double* x_plus_delta) {
  Eigen::Map<const Eigen::Vector3d> delta_(delta);
  Eigen::Map<const Eigen::Vector4d> x_(x);
  Eigen::Map<Eigen::Vector4d> x_plus_delta_(x_plus_delta);

  // Euclidean style
  x_plus_delta_ = x_ + Eigen::Vector4d(delta_[0], delta_[1], delta_[2], 0);

  return true;
}

// Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
bool HomogeneousPointManifold::Minus(const double* x_plus_delta, const double* x, double* delta) const {
  return minus(x_plus_delta, x, delta);
}

bool HomogeneousPointManifold::ComputeLiftJacobian(const double* x, double* jacobian) const {
  return liftJacobian(x, jacobian);
}

// Computes the minimal difference between a variable x and a perturbed variable x_plus_delta.
bool HomogeneousPointManifold::minus(const double* x_plus_delta, const double* x, double* delta) {
  Eigen::Map<Eigen::Vector3d> delta_(delta);
  Eigen::Map<const Eigen::Vector4d> x_(x);
  Eigen::Map<const Eigen::Vector4d> x_plus_delta_(x_plus_delta);

  // Euclidean style
  OKVIS_ASSERT_TRUE_DBG(Exception,
                        fabs((x_plus_delta_ - x_)[3]) < 1e-12,
                        "comparing homogeneous points with different scale " << x_plus_delta_[3] << " vs. " << x_[3]);
  delta_ = (x_plus_delta_ - x_).head<3>();

  return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
bool HomogeneousPointManifold::PlusJacobian(const double* x, double* jacobian) const {
  return plusJacobian(x, jacobian);
}

// Compute the derivative of Minus(y, x) w.r.t y at y = x.
bool HomogeneousPointManifold::MinusJacobian(const double* x, double* jacobian) const {
  return minusJacobian(x, jacobian);
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
bool HomogeneousPointManifold::plusJacobian(const double*, double* jacobian) {
  Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor> > Jp(jacobian);

  // Euclidean-style
  Jp.setZero();
  Jp.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();

  return true;
}

// Compute the derivative of Minus(y, x) w.r.t y at y = x.
bool HomogeneousPointManifold::minusJacobian(const double*, double* jacobian) {
  Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > Jp(jacobian);

  // Euclidean-style
  Jp.setZero();
  Jp.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();

  return true;
}

// Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
bool HomogeneousPointManifold::liftJacobian(const double*, double* jacobian) {
  Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor> > Jp(jacobian);

  // Euclidean-style
  Jp.setZero();
  Jp.topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity();

  return true;
}

}  // namespace ceres
}  // namespace okvis
