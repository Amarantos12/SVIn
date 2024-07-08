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
 *********************************************************************************/

/**
 * @file SonarError.cpp
 * @brief Source file for the SonarError class.
 * @author Sharmin Rahman
 */

#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/SonarError.hpp>
#include <vector>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurement and information matrix.
/*SonarError::SonarError(
    const Eigen::Vector4d & measurement, const information_t & information) {
  setMeasurement(measurement);
  setInformation(information);
}*/

SonarError::SonarError(const okvis::VioParameters& params,
                       const information_t& information,
                       const okvis::kinematics::Transformation T_WSL,
                       const okvis::kinematics::Transformation T_SLS)
      : params_(params)
  {
//    mutable_parameter_block_sizes()->push_back(7); // T2的参数块大小为7
//    set_num_residuals(pts1.size() * 3); // 残差的数量为P1的大小乘以3
    setMeasurement(T_WSL, T_SLS);
    setInformation(information);
//    LOG(INFO) << "########ForwardSonarError::ForwardSonarError######";
  }

/*void SonarError::computeCovarianceMatrix(const std::vector<Eigen::Vector4d> &landmarkSubset) {
        Eigen::Matrix<double,3,3> covariance = Eigen::Matrix<double,3,3>::Zero();
        double mean[3] = {0, 0, 0};
        for (auto it = landmarkSubset.begin(); it != landmarkSubset.end(); ++it){
                mean[0] += (*it)[0] / (*it)[3]; // it[3] is always 1
                mean[1] += (*it)[1] / (*it)[3]; // it[3] is always 1
                mean[2] += (*it)[2] / (*it)[3]; // it[3] is always 1
        }
        mean[0] = mean[0] / landmarkSubset.size();
        mean[1] = mean[1] / landmarkSubset.size();
        mean[2] = mean[2] / landmarkSubset.size();

        for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++) {
                        covariance(i,j) = 0.0;
                        for (auto it = landmarkSubset.begin(); it != landmarkSubset.end(); ++it)
                                covariance(i,j) += (mean[i] - (*it)[0]) * (mean[j] - (*it)[1]);
                        covariance(i,j) /= landmarkSubset.size() - 1;
                }
        covariance_ = covariance;
        information_ = covariance.inverse();
        // perform the Cholesky decomposition on order to obtain the correct error weighting
        Eigen::LLT<information_t> lltOfInformation(information_);
        _squareRootInformation = lltOfInformation.matrixL().transpose();
        LOG(INFO) << covariance_;
        LOG(INFO) << information_;
        LOG(INFO) << _squareRootInformation;
  }*/

void SonarError::setInformation(const information_t& information) {
  information_ = information;
  covariance_ = 1 / information;
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  // Eigen::LLT<information_t> lltOfInformation(information_);
  // TODO(@sharmin): Check if it's correct
  _squareRootInformation = sqrt(information);
}

// This evaluates the error term and additionally computes the Jacobians.
bool SonarError::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
bool SonarError::EvaluateWithMinimalJacobians(double const* const* parameters,
                                              double* residuals,
                                              double** jacobians,
                                              double** jacobiansMinimal) const {
//   LOG(INFO) << "######pts2_: " << pts2_ << " pts1_: " << pts1_;
   // compute error
   okvis::kinematics::Transformation T_WS(
      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));

   okvis::kinematics::Transformation T_WSo = T_WS * params_.sonar.T_SSo;
   okvis::kinematics::Transformation T_WSoL = T_WSL_ * params_.sonar.T_SSo;
   okvis::kinematics::Transformation T_WSoL_So = T_WSoL * T_SL_S_;
//   okvis::kinematics::Transformation sonar_p1(pts1_, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
//   okvis::kinematics::Transformation sonar_p2(pts2_, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
//   okvis::kinematics::Transformation SP1 = T_WSoL * sonar_p1;
//   okvis::kinematics::Transformation TSP1 = T_SoL_So * sonar_p2;
   Eigen::Matrix4d MTWSo = T_WSo.T();
   Eigen::Matrix4d MTSoLSo = T_WSoL_So.T();
   Eigen::Matrix4d Mdp = MTSoLSo - MTWSo;
   okvis::kinematics::Transformation dp(Mdp);

   Eigen::Matrix<double, 6, 1> error;
   Eigen::Vector3d dtheta = 2 * dp.q().coeffs().head<3>();
//   error.head<3>() = SP1.r() - TSP1.r();
//   error.tail<3>() = dtheta1 - dtheta2;
   error.head<3>() = dp.r();
   error.tail<3>() = dtheta;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, 6, 1> > weighted_error(residuals);
  weighted_error = _squareRootInformation * error;

//  LOG(INFO) << "########weighted_error: " << weighted_error << " _squareRootInformation: " << _squareRootInformation;

    // compute Jacobian...
  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor> > J0(jacobians[0]);
      Eigen::Matrix<double, 6, 6, Eigen::RowMajor> J0_minimal;
      J0_minimal.setIdentity();
      J0_minimal *= -1.0;
      J0_minimal.block<3, 3>(3, 3) = -okvis::kinematics::plus(dp.q()).topLeftCorner<3, 3>();
      J0_minimal = (_squareRootInformation * J0_minimal).eval();

      // pseudo inverse of the local parametrization Jacobian:
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[0], J_lift.data());

      // hallucinate Jacobian w.r.t. state
      J0 = J0_minimal * J_lift;

      if (jacobiansMinimal != NULL) {
        if (jacobiansMinimal[0] != NULL) {
          Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
          J0_minimal_mapped = J0_minimal;
        }
      }
    }
  }
      return true;
}

}  // namespace ceres
}  // namespace okvis

//bool SonarError::EvaluateWithMinimalJacobians(double const* const* parameters,
//                                              double* residuals,
//                                              double** jacobians,
//                                              double** jacobiansMinimal) const {
//   // compute error
//   okvis::kinematics::Transformation T_WS(
//      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
//      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));
//
//   okvis::kinematics::Transformation T_WSo = T_WS * params_.sonar.T_SSo;
//   okvis::kinematics::Transformation T_WSoL = T_WSL_ * params_.sonar.T_SSo;
//   okvis::kinematics::Transformation T_SoL_So = T_WSoL.inverse() * T_WSo;
//   Eigen::Vector3d error = (T_SoL_So.C() * pts2_ + T_SoL_So.r()) - pts1_;
//   residuals[0] =  error[0];
//   residuals[1] =  error[1];
//   residuals[2] =  error[2];
//   if (jacobians != NULL) {
//        if (jacobians[0] != NULL) {
//        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J0(jacobians[0]);
//        Eigen::Matrix<double, 3, 7> J0_minimal_temp;
//        J0_minimal_temp.setZero();
//
//        const double sx = pts2_[0];
//        const double sy = pts2_[1];
//        const double sz = pts2_[2];
//        const double qw = T_SoL_So.q().w();
//        const double qx = T_SoL_So.q().x();
//        const double qy = T_SoL_So.q().y();
//        const double qz = T_SoL_So.q().z();
//        const double x = T_SoL_So.r()[0];
//        const double y = T_SoL_So.r()[1];
//        const double z = T_SoL_So.r()[2];
//
//        J0_minimal_temp.resize(3, 7);
//
////         Analytic Jacobians with Multiple Residuals, see:
////         https://groups.google.com/g/ceres-solver/c/nVZdc4hu5zw
////         calculate jacobians
////             qx  qy  qz  x  y  z (qw = sqrt(1-qx^2-qy^2-qz^2)
////         r1                           ri = tar(i) - tar_p_(i)
////         r2
////         r3
////         dr1/dq
////          dri/dt
//        J0_minimal_temp(0,0) = 1.;  // dr1/dx
//        J0_minimal_temp(0,1) = 0.;
//        J0_minimal_temp(0,2) = 0.;
//        J0_minimal_temp(1,0) = 0.;
//        J0_minimal_temp(1,1) = 1.;  // dr2/dy
//        J0_minimal_temp(1,2) = 0.;
//        J0_minimal_temp(2,0) = 0.;
//        J0_minimal_temp(2,1) = 0.;
//        J0_minimal_temp(2,2) = 1.;  // dr3/dz
//
//        J0_minimal_temp(0,3) = 2 * (qy * sy + qz * sz);
//        J0_minimal_temp(0,4) = 2 * (-2 * qy * sx + qx * sy + qw * sz);
//        J0_minimal_temp(0,5) = 2 * (-2 * qz * sx - qw * sy + qw * sz);
//        J0_minimal_temp(0,6) = 2 * (-qz * sy + qy * sz);  // dr1/dqw
//        // dr2/dq
//        J0_minimal_temp(1,3) = 2 * (qy * sx - 2 * qx * sy - qw * sz);
//        J0_minimal_temp(1,4) = 2 * (qx * sx + qz * sz);
//        J0_minimal_temp(1,5) = 2 * (qw * sx - 2 * qz * sy + qy * sz);
//        J0_minimal_temp(1,6) = 2 * (qz * sx - qx * sz);  // dr2/dqw
//        // dr3/dq
//        J0_minimal_temp(2,3) = 2 * (qz * sx + qw * sy - 2 * qx * sz);
//        J0_minimal_temp(2,4) = 2 * (-qw * sx + qz * sy - 2 * qy * sz);
//        J0_minimal_temp(2,5) = 2 * (qx * sx + qy * sy);
//        J0_minimal_temp(2,6) = 2 * (-qy * sx + qx * sy);  // dr3/dqw
//
//        J0 = J0_minimal_temp;
//
//     }
//   }
//
//      return true;
//}

//###########################################################################################//
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
 *********************************************************************************/

/**
 * @file SonarError.cpp
 * @brief Source file for the SonarError class.
 * @author Sharmin Rahman
 */

//#include <okvis/ceres/PoseLocalParameterization.hpp>
//#include <okvis/ceres/SonarError.hpp>
//#include <vector>
//
///// \brief okvis Main namespace of this package.
//namespace okvis {
///// \brief ceres Namespace for ceres-related functionality implemented in okvis.
//namespace ceres {
//
//// Construct with measurement and information matrix.
///*SonarError::SonarError(
//    const Eigen::Vector4d & measurement, const information_t & information) {
//  setMeasurement(measurement);
//  setInformation(information);
//}*/
//
//SonarError::SonarError(const okvis::VioParameters& params,
//                       const information_t& information,
//                       const Eigen::Vector3d& pts1,
//                       const Eigen::Vector3d& pts2,
//                       const okvis::kinematics::Transformation T_WSL,
//                       const okvis::kinematics::Transformation T_SLS)
//      : params_(params)
//  {
////    mutable_parameter_block_sizes()->push_back(7); // T2的参数块大小为7
////    set_num_residuals(pts1.size() * 3); // 残差的数量为P1的大小乘以3
//    setMeasurement(pts1, pts2, T_WSL, T_SLS);
//    setInformation(information);
////    LOG(INFO) << "########ForwardSonarError::ForwardSonarError######";
//  }
//
///*void SonarError::computeCovarianceMatrix(const std::vector<Eigen::Vector4d> &landmarkSubset) {
//        Eigen::Matrix<double,3,3> covariance = Eigen::Matrix<double,3,3>::Zero();
//        double mean[3] = {0, 0, 0};
//        for (auto it = landmarkSubset.begin(); it != landmarkSubset.end(); ++it){
//                mean[0] += (*it)[0] / (*it)[3]; // it[3] is always 1
//                mean[1] += (*it)[1] / (*it)[3]; // it[3] is always 1
//                mean[2] += (*it)[2] / (*it)[3]; // it[3] is always 1
//        }
//        mean[0] = mean[0] / landmarkSubset.size();
//        mean[1] = mean[1] / landmarkSubset.size();
//        mean[2] = mean[2] / landmarkSubset.size();
//
//        for (int i = 0; i < 3; i++)
//                for (int j = 0; j < 3; j++) {
//                        covariance(i,j) = 0.0;
//                        for (auto it = landmarkSubset.begin(); it != landmarkSubset.end(); ++it)
//                                covariance(i,j) += (mean[i] - (*it)[0]) * (mean[j] - (*it)[1]);
//                        covariance(i,j) /= landmarkSubset.size() - 1;
//                }
//        covariance_ = covariance;
//        information_ = covariance.inverse();
//        // perform the Cholesky decomposition on order to obtain the correct error weighting
//        Eigen::LLT<information_t> lltOfInformation(information_);
//        _squareRootInformation = lltOfInformation.matrixL().transpose();
//        LOG(INFO) << covariance_;
//        LOG(INFO) << information_;
//        LOG(INFO) << _squareRootInformation;
//  }*/
//
//void SonarError::setInformation(const information_t& information) {
//  information_ = information;
//  covariance_ = 1 / information;
//  // perform the Cholesky decomposition on order to obtain the correct error weighting
//  // Eigen::LLT<information_t> lltOfInformation(information_);
//  // TODO(@sharmin): Check if it's correct
//  _squareRootInformation = sqrt(information);
//}
//
//// This evaluates the error term and additionally computes the Jacobians.
//bool SonarError::Evaluate(double const* const* parameters, double* residuals, double** jacobians) const {
//  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
//}
//
//// This evaluates the error term and additionally computes
//// the Jacobians in the minimal internal representation.
//bool SonarError::EvaluateWithMinimalJacobians(double const* const* parameters,
//                                              double* residuals,
//                                              double** jacobians,
//                                              double** jacobiansMinimal) const {
////   LOG(INFO) << "######pts2_: " << pts2_ << " pts1_: " << pts1_;
//   // compute error
//   okvis::kinematics::Transformation T_WS(
//      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
//      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));
//
//   okvis::kinematics::Transformation T_WSo = T_WS * params_.sonar.T_SSo;
//   okvis::kinematics::Transformation T_WSoL = T_WSL_ * params_.sonar.T_SSo;
//   okvis::kinematics::Transformation T_SoL_So = T_SL_S_ * T_WSo;
////   Eigen::Vector3d error = (T_SoL_So.C() * pts2_ + T_SoL_So.r()) - pts1_;
//   okvis::kinematics::Transformation sonar_p1(pts1_, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
//   okvis::kinematics::Transformation sonar_p2(pts2_, Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0));
//
////   Eigen::Vector3d error = (T_SoL_So.C() * pts2_ + T_SoL_So.r()) - pts1_;
//   Eigen::Vector3d error = (T_SoL_So * sonar_p2).r() - sonar_p1.r();
//   LOG(INFO) << "########error: " << error;
//   residuals[0] =  _squareRootInformation * error[0];
//   residuals[1] =  _squareRootInformation * error[1];
////   residuals[1] =  0;
//   residuals[2] =  _squareRootInformation * error[2];
////   LOG(INFO) << " residuals[0]: " << residuals[0] << " residuals[1]: " << residuals[1] << " residuals[2]: " << residuals[2];
//   if (jacobians != NULL) {
//        if (jacobians[0] != NULL) {
//        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J0(jacobians[0]);
//        Eigen::Matrix<double, 3, 7, Eigen::RowMajor> J0_minimal_temp;
////        Eigen::Matrix<double, 3, 7> J0_minimal;
////        J0_minimal.setZero();
//        J0_minimal_temp.setZero();
//
////        const double sx = pts2_[0];
////        const double sy = pts2_[1];
////        const double sz = pts2_[2];
////        const double qw = T_WSo.q().w();
////        const double qx = T_WSo.q().x();
////        const double qy = T_WSo.q().y();
////        const double qz = T_WSo.q().z();
////        const double x = T_WSo.r()[0];
////        const double y = T_WSo.r()[1];
////        const double z = T_WSo.r()[2];
//        const double sx = sonar_p2.r()[0];
//        const double sy = sonar_p2.r()[1];
//        const double sz = sonar_p2.r()[2];
//        const double qw = T_SoL_So.q().w();
//        const double qx = T_SoL_So.q().x();
//        const double qy = T_SoL_So.q().y();
//        const double qz = T_SoL_So.q().z();
//        const double x = T_SoL_So.r()[0];
//        const double y = T_SoL_So.r()[1];
//        const double z = T_SoL_So.r()[2];
//
////         Analytic Jacobians with Multiple Residuals, see:
////         https://groups.google.com/g/ceres-solver/c/nVZdc4hu5zw
////         calculate jacobians
////             qx  qy  qz  x  y  z (qw = sqrt(1-qx^2-qy^2-qz^2)
////         r1                           ri = tar(i) - tar_p_(i)
////         r2
////         r3
////         dr1/dq
////          dri/dt
//        J0_minimal_temp(0,0) = 1.;  // dr1/dx
//        J0_minimal_temp(0,1) = 0.;
//        J0_minimal_temp(0,2) = 0.;
//        J0_minimal_temp(1,0) = 0.;
//        J0_minimal_temp(1,1) = 1.;  // dr2/dy
//        J0_minimal_temp(1,2) = 0.;
//        J0_minimal_temp(2,0) = 0.;
//        J0_minimal_temp(2,1) = 0.;
//        J0_minimal_temp(2,2) = 1.;  // dr3/dz
//
//        J0_minimal_temp(0,3) = 2 * (qy * sy + qz * sz);
//        J0_minimal_temp(0,4) = 2 * (-2 * qy * sx + qx * sy + qw * sz);
//        J0_minimal_temp(0,5) = 2 * (-2 * qz * sx - qw * sy + qw * sz);
//        J0_minimal_temp(0,6) = 2 * (-qz * sy + qy * sz);  // dr1/dqw
//        // dr2/dq
//        J0_minimal_temp(1,3) = 2 * (qy * sx - 2 * qx * sy - qw * sz);
//        J0_minimal_temp(1,4) = 2 * (qx * sx + qz * sz);
//        J0_minimal_temp(1,5) = 2 * (qw * sx - 2 * qz * sy + qy * sz);
//        J0_minimal_temp(1,6) = 2 * (qz * sx - qx * sz);  // dr2/dqw
//        // dr3/dq
//        J0_minimal_temp(2,3) = 2 * (qz * sx + qw * sy - 2 * qx * sz);
//        J0_minimal_temp(2,4) = 2 * (-qw * sx + qz * sy - 2 * qy * sz);
//        J0_minimal_temp(2,5) = 2 * (qx * sx + qy * sy);
//        J0_minimal_temp(2,6) = 2 * (-qy * sx + qx * sy);  // dr3/dqw
//        LOG(INFO) << J0_minimal_temp;
////        for(int i = 0; i < 3; i++)
////        {
////           okvis::kinematics::Transformation T_Jacobian(
////               Eigen::Vector3d(J0_minimal_temp(i,0), J0_minimal_temp(i,1), J0_minimal_temp(i,2)),
////               Eigen::Quaterniond(J0_minimal_temp(i,6), J0_minimal_temp(i,3), J0_minimal_temp(i,4), J0_minimal_temp(i,5)));
////           okvis::kinematics::Transformation T_J0_minimal = T_WSoL.inverse() * T_Jacobian;
//////           okvis::kinematics::Transformation T_J0_minimal = T_Jacobian;
////           J0_minimal(i,0) = T_J0_minimal.r()[0];
////           J0_minimal(i,1) = T_J0_minimal.r()[1];
////           J0_minimal(i,2) = T_J0_minimal.r()[2];
////           J0_minimal(i,3) = T_J0_minimal.q().x();
////           J0_minimal(i,4) = T_J0_minimal.q().y();
////           J0_minimal(i,5) = T_J0_minimal.q().z();
////           J0_minimal(i,6) = T_J0_minimal.q().w();
////        }
////        LOG(INFO) << "J0_minimal_temp: " << J0_minimal_temp;
//////        J0.block<3, 7>(0, 0) = J0_minimal_temp;
//        J0_minimal_temp = (_squareRootInformation * J0_minimal_temp).eval();
//        J0 = J0_minimal_temp;
////        LOG(INFO) << "J0: " << J0;
////        J0 = J0_minimal;
//
//
////        if (jacobiansMinimal != NULL) {
////            if (jacobiansMinimal[0] != NULL) {
////                Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor> > J0_minimal_mapped(jacobiansMinimal[0]);
////                J0_minimal_mapped = J0_minimal_temp;
////            }
////         }
//
//     }
//   }
//
//      return true;
//}
//
//}  // namespace ceres
//}  // namespace okvis
//
////bool SonarError::EvaluateWithMinimalJacobians(double const* const* parameters,
////                                              double* residuals,
////                                              double** jacobians,
////                                              double** jacobiansMinimal) const {
////   // compute error
////   okvis::kinematics::Transformation T_WS(
////      Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]),
////      Eigen::Quaterniond(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]));
////
////   okvis::kinematics::Transformation T_WSo = T_WS * params_.sonar.T_SSo;
////   okvis::kinematics::Transformation T_WSoL = T_WSL_ * params_.sonar.T_SSo;
////   okvis::kinematics::Transformation T_SoL_So = T_WSoL.inverse() * T_WSo;
////   Eigen::Vector3d error = (T_SoL_So.C() * pts2_ + T_SoL_So.r()) - pts1_;
////   residuals[0] =  error[0];
////   residuals[1] =  error[1];
////   residuals[2] =  error[2];
////   if (jacobians != NULL) {
////        if (jacobians[0] != NULL) {
////        Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J0(jacobians[0]);
////        Eigen::Matrix<double, 3, 7> J0_minimal_temp;
////        J0_minimal_temp.setZero();
////
////        const double sx = pts2_[0];
////        const double sy = pts2_[1];
////        const double sz = pts2_[2];
////        const double qw = T_SoL_So.q().w();
////        const double qx = T_SoL_So.q().x();
////        const double qy = T_SoL_So.q().y();
////        const double qz = T_SoL_So.q().z();
////        const double x = T_SoL_So.r()[0];
////        const double y = T_SoL_So.r()[1];
////        const double z = T_SoL_So.r()[2];
////
////        J0_minimal_temp.resize(3, 7);
////
//////         Analytic Jacobians with Multiple Residuals, see:
//////         https://groups.google.com/g/ceres-solver/c/nVZdc4hu5zw
//////         calculate jacobians
//////             qx  qy  qz  x  y  z (qw = sqrt(1-qx^2-qy^2-qz^2)
//////         r1                           ri = tar(i) - tar_p_(i)
//////         r2
//////         r3
//////         dr1/dq
//////          dri/dt
////        J0_minimal_temp(0,0) = 1.;  // dr1/dx
////        J0_minimal_temp(0,1) = 0.;
////        J0_minimal_temp(0,2) = 0.;
////        J0_minimal_temp(1,0) = 0.;
////        J0_minimal_temp(1,1) = 1.;  // dr2/dy
////        J0_minimal_temp(1,2) = 0.;
////        J0_minimal_temp(2,0) = 0.;
////        J0_minimal_temp(2,1) = 0.;
////        J0_minimal_temp(2,2) = 1.;  // dr3/dz
////
////        J0_minimal_temp(0,3) = 2 * (qy * sy + qz * sz);
////        J0_minimal_temp(0,4) = 2 * (-2 * qy * sx + qx * sy + qw * sz);
////        J0_minimal_temp(0,5) = 2 * (-2 * qz * sx - qw * sy + qw * sz);
////        J0_minimal_temp(0,6) = 2 * (-qz * sy + qy * sz);  // dr1/dqw
////        // dr2/dq
////        J0_minimal_temp(1,3) = 2 * (qy * sx - 2 * qx * sy - qw * sz);
////        J0_minimal_temp(1,4) = 2 * (qx * sx + qz * sz);
////        J0_minimal_temp(1,5) = 2 * (qw * sx - 2 * qz * sy + qy * sz);
////        J0_minimal_temp(1,6) = 2 * (qz * sx - qx * sz);  // dr2/dqw
////        // dr3/dq
////        J0_minimal_temp(2,3) = 2 * (qz * sx + qw * sy - 2 * qx * sz);
////        J0_minimal_temp(2,4) = 2 * (-qw * sx + qz * sy - 2 * qy * sz);
////        J0_minimal_temp(2,5) = 2 * (qx * sx + qy * sy);
////        J0_minimal_temp(2,6) = 2 * (-qy * sx + qx * sy);  // dr3/dqw
////
////        J0 = J0_minimal_temp;
////
////     }
////   }
////
////      return true;
////}