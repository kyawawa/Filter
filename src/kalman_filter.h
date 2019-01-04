// -*- mode: C++; coding: utf-8-unix; -*-

/**
 * State space representation:
 * x = Ax + Bu + w ( w ~ N(0, Q) )
 * y = Hx + v ( v ~ N(0, R) )
 */

#ifndef __KALMANFILTER_H__
#define __KALMANFILTER_H__

#include <Eigen/Core>
#include "discrete_algebraic_riccati_equation.h"

template<size_t dim_state, size_t dim_observe, size_t dim_input>
class KalmanFilter
{
    using VectorState      = Eigen::Matrix<double, dim_state, 1>;
    using MatrixState      = Eigen::Matrix<double, dim_state, dim_state>;
    using VectorInput      = Eigen::Matrix<double, dim_input, 1>;
    using MatrixInput      = Eigen::Matrix<double, dim_state, dim_input>;
    using MatrixObserve    = Eigen::Matrix<double, dim_observe, dim_state>;
    using MatrixObserveSqu = Eigen::Matrix<double, dim_observe, dim_observe>;
    using MatrixKalmanGain = Eigen::Matrix<double, dim_state, dim_observe>;

    bool is_steady;  // Calculate Kalman gain at each step or not
    // Estimate
    VectorState      X;  // State vector
    MatrixState      P;  // Error covariance matrix
    MatrixKalmanGain K;  // Kalman gain
    // Parameter
    MatrixState      A;  // State matrix
    VectorInput      U;  // Input vector
    MatrixInput      B;  // Input matrix
    MatrixState      Q;  // Covariance matrix of the process noise
    MatrixObserve    H;  // Observation matrix
    MatrixObserveSqu R;  // Covariance matrix of the observation noise

  public:
    KalmanFilter(const Eigen::Ref<const VectorState>& _Xinit,
                 const Eigen::Ref<const MatrixState>& _A,
                 const Eigen::Ref<const VectorInput>& _U,
                 const Eigen::Ref<const MatrixInput>& _B,
                 const Eigen::Ref<const MatrixState>& _Q,
                 const Eigen::Ref<const MatrixObserve>& _H,
                 const Eigen::Ref<const MatrixObserveSqu>& _R,
                 const bool _is_steady = false)
        : X(_Xinit), A(_A), U(_U), B(_B), Q(_Q), H(_H), R(_R), is_steady(_is_steady)
    {
        if (is_steady) calcSteadyKalmanGain(_A, _Q, _H, _R);
        else K = Q * H.transpose() * (H * Q * H.transpose() + R).inverse(); // TODO
        P = (MatrixState::Identity() - K * H) * Q;
    }

    auto getStateVector() const { return X; };

    void setStateVector(const Eigen::Ref<const VectorState>& _X) { X = _X; }
    void setStateMatrix(const Eigen::Ref<const MatrixState>& _A) { A = _A; }
    void setInputVector(const Eigen::Ref<const MatrixInput>& _U) { U = _U; }
    void setInputMatrix(const Eigen::Ref<const MatrixInput>& _B) { B = _B; }
    void setProcessNoiseCovMatrix(const Eigen::Ref<const MatrixState>& _Q) { Q = _Q; }
    void setObservationMatrix(const Eigen::Ref<const MatrixObserve>& _H) { H = _H; }
    void setObservationNoiseCovMatrix(const Eigen::Ref<const MatrixObserveSqu>& _R) { R = _R; }

    void unsetSteadyKalmanGain() { is_steady = false; }
    void calcSteadyKalmanGain(const Eigen::Ref<const MatrixState>& _A,
                              const Eigen::Ref<const MatrixState>& _Q,
                              const Eigen::Ref<const MatrixObserve>& _H,
                              const Eigen::Ref<const MatrixObserveSqu>& _R)
    {
        const auto P_final = solveDiscreteAlgebraicRiccati<dim_state, dim_observe>(_A.transpose(), _H.transpose(), _Q, _R);
        K = P_final * _H.transpose() * (_H * P_final * _H.transpose() + _R).inverse(); // TODO
        is_steady = true;
    }

    virtual void predictNextState()
    {
        X = A * X + B * U;
        P = A * P * A.transpose() + Q;
    }

    virtual void estimateCurrentState(const Eigen::Ref<const Eigen::Matrix<double, dim_observe, 1>>& _y)
    {
        if (!is_steady) K = P * H.transpose() * (H * P * H.transpose() + R).inverse(); // TODO
        X = X + K * (_y - H * X);
        P = (MatrixState::Identity() - K * H) * P;
    }
};





#endif // __KALMANFILTER_H__
