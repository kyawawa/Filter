// -*- mode: C++; coding: utf-8-unix; -*-

/**
 * State space representation:
 * x = Ax + Bu + w ( w ~ N(0, Q) )
 * y = Hx + v ( v ~ N(0, R) )
 */

#ifndef __KALMANFILTER_H__
#define __KALMANFILTER_H__

#include <Eigen/Core>
#include "discrete_argebraic_riccati_equation.h"

template<size_t dim_state, size_t dim_observe, size_t dim_input>
class KalmanFilter
{
    using VectorState      = Eigen::Matrix<double, dim_state, 1>;
    using MatrixState      = Eigen::Matrix<double, dim_state, dim_state>;
    using MatrixInput      = Eigen::Matrix<double, dim_state, dim_input>;
    using MatrixObserve    = Eigen::Matrix<double, dim_observe, dim_state>;
    using MatrixObserveSqu = Eigen::Matrix<double, dim_observe, dim_observe>;
    using MatrixKalmanGain = Eigen::Matrix<double, dim_state, dim_observe>;

  public:
    KalmanFilter(const Eigen::Ref<const VectorState>& _Xinit,
                 const Eigen::Ref<const MatrixState>& _A,
                 const Eigen::Ref<const MatrixInput>& _B,
                 const Eigen::Ref<const MatrixState>& _Q,
                 const Eigen::Ref<const MatrixObserve>& _H,
                 const Eigen::Ref<const MatrixObserveSqu>& _R)
        : X(_Xinit), A(_A), B(_B), Q(_Q), H(_H), R(_R) {}

    VectorState getStateVector() const { return X; };

    void setStateVector(const Eigen::Ref<const VectorState>& _X) { X = _X; };
    void setStateMatrix(const Eigen::Ref<const MatrixState>& _A) { A = _A; };
    void setInputMatrix(const Eigen::Ref<const MatrixInput>& _B) { B = _B; };
    void setProcessNoiseCovMatrix(const Eigen::Ref<const MatrixState>& _Q) { Q = _Q; };
    void setObservationMatrix(const Eigen::Ref<const MatrixObserve>& _H) { H = _H; };
    void setObservationNoiseCovMatrix(const Eigen::Ref<const MatrixObserveSqu>& _R) { R = _R; };

    virtual void predictNextState()
    {
        X = A * X;
        P = A * P * A.transpose() + Q;
    }

    virtual void estimateCurrentState(const Eigen::Ref<const Eigen::Matrix<double, dim_observe, 1>>& _y)
    {
        K = P * H.transpose() * (H * P * H.transpose() + R).inverse(); // TODO
        X = X + K * (_y - H * X);
        P = (MatrixState::Identity() - K * H) * P;
    }

  protected:
    // Estimate
    VectorState      X;  // State vector
    MatrixState      P;  // Error covariance matrix
    MatrixKalmanGain K;  // Kalman gain
    // Parameter
    MatrixState      A;  // State matrix
    MatrixInput      B;  // Input matrix
    MatrixState      Q;  // Covariance matrix of the process noise
    MatrixObserve    H;  // Observation matrix
    MatrixObserveSqu R;  // Covariance matrix of the observation noise
};

template<size_t dim_state, size_t dim_observe, size_t dim_input>
class SteadyKalmanFilter : public KalmanFilter<dim_state, dim_observe, dim_input>
{
    using VectorState      = Eigen::Matrix<double, dim_state, 1>;
    using MatrixState      = Eigen::Matrix<double, dim_state, dim_state>;
    using MatrixInput      = Eigen::Matrix<double, dim_state, dim_input>;
    using MatrixObserve    = Eigen::Matrix<double, dim_observe, dim_state>;
    using MatrixObserveSqu = Eigen::Matrix<double, dim_observe, dim_observe>;

    using super = KalmanFilter<dim_state, dim_observe, dim_input>;
    using super::X;
    using super::P;
    using super::K;
    using super::A;
    using super::B;
    using super::Q;
    using super::H;
    using super::R;

  public:
    SteadyKalmanFilter(const Eigen::Ref<const VectorState>& _Xinit,
                       const Eigen::Ref<const MatrixState>& _A,
                       const Eigen::Ref<const MatrixInput>& _B,
                       const Eigen::Ref<const MatrixState>& _Q,
                       const Eigen::Ref<const MatrixObserve>& _H,
                       const Eigen::Ref<const MatrixObserveSqu>& _R)
        : super(_Xinit, _A, _B, _Q, _H, _R)
    {
        P = solveDiscreteAlgebraicRiccati(_A.transpose(), _H.transpose(), _Q, _R);
        K = P * H.transpose() * (H * P * H.transpose() + R).inverse(); // TODO
    }

    void predictNextState() override { X = A * X; }
    void estimateCurrentState(const Eigen::Ref<const Eigen::Matrix<double, dim_observe, 1>>& _y) override
    {
        X = X + K * (_y - H * X);
    }
};

#endif // __KALMANFILTER_H__
