// -*- mode: C++; coding: utf-8-unix; -*-

/**
 * State space representation:
 * x = Ax + Bu + w ( w ~ N(0, Q) )
 * y = Hx + v ( v ~ N(0, R) )
 * @file  kalman_filter.h
 * @brief Linear Kalman filter
 * @author Tatsuya Ishikawa
 */

#ifndef __KALMANFILTER_H__
#define __KALMANFILTER_H__

#include <deque>
#include <iterator>
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

    bool is_steady;  // Calculate Kalman gain at each step or not

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

    void predictNextState()
    {
        X = A * X + B * U;
        P = A * P * A.transpose() + Q;
    }

    void estimateCurrentState(const Eigen::Ref<const Eigen::Matrix<double, dim_observe, 1>>& _y)
    {
        if (!is_steady) K = P * H.transpose() * (H * P * H.transpose() + R).inverse(); // TODO
        X = X + K * (_y - H * X);
        P = (MatrixState::Identity() - K * H) * P;
    }
};

template<size_t dim_state, size_t dim_observe, size_t dim_input>
class KalmanFilterSmoother
{
    using VectorState      = Eigen::Matrix<double, dim_state, 1>;
    using MatrixState      = Eigen::Matrix<double, dim_state, dim_state>;
    using VectorInput      = Eigen::Matrix<double, dim_input, 1>;
    using MatrixInput      = Eigen::Matrix<double, dim_state, dim_input>;
    using MatrixObserve    = Eigen::Matrix<double, dim_observe, dim_state>;
    using MatrixObserveSqu = Eigen::Matrix<double, dim_observe, dim_observe>;
    using MatrixKalmanGain = Eigen::Matrix<double, dim_state, dim_observe>;

  protected:
    unsigned int smooth_size = 0;
    /// Estimated parameter and deque for Kalman Smoothing
    /// X_est[k] = x_k|n, X_pre[k] = x_k|(k-1)
    /// P_est[k] = P_k|n, P_pre[k] = P_k|(k-1)
    std::deque<VectorState> X_est;    // Estimated state vector
    std::deque<VectorState> X_pre;    // Predicted state vector
    std::deque<VectorState> X_smth;  // Smoothed state vector
    std::deque<MatrixState> P_est;    // Estimated error covariance matrix
    std::deque<MatrixState> P_pre;    // Predicted error covariance matrix
    std::deque<MatrixState> C_deque;  // Used to run Kalman Smoothing. Size: X_est - 1 (Without Latest info
    MatrixKalmanGain K;  // Kalman gain
    // Parameter
    MatrixState      A;  // State matrix
    VectorInput      U;  // Input vector
    MatrixInput      B;  // Input matrix
    MatrixState      Q;  // Covariance matrix of the process noise
    MatrixObserve    H;  // Observation matrix
    MatrixObserveSqu R;  // Covariance matrix of the observation noise

  public:
    KalmanFilterSmoother(const Eigen::Ref<const VectorState>& _Xinit,
                         const Eigen::Ref<const MatrixState>& _A,
                         const Eigen::Ref<const VectorInput>& _U,
                         const Eigen::Ref<const MatrixInput>& _B,
                         const Eigen::Ref<const MatrixState>& _Q,
                         const Eigen::Ref<const MatrixObserve>& _H,
                         const Eigen::Ref<const MatrixObserveSqu>& _R)
        : A(_A), U(_U), B(_B), Q(_Q), H(_H), R(_R)
    {
        X_pre.emplace_front(_Xinit);
        X_est.emplace_front(_Xinit);
        P_pre.emplace_front(Q);
        K = Q * H.transpose() * (H * Q * H.transpose() + R).inverse(); // TODO
        P_est.emplace_front((MatrixState::Identity() - K * H) * Q);
    }

    auto getEstimatedStateVector() const { return X_est.front(); }
    auto getPredictedStateVector() const { return X_pre.front(); }
    auto getAllSmoothedStateVector() const { return X_smth; }
    auto getStateVectorDeque() const { return X_est; }

    void setSmoothSize(const unsigned int _size) { smooth_size = _size; }
    void setStateMatrix(const Eigen::Ref<const MatrixState>& _A) { A = _A; }
    void setInputVector(const Eigen::Ref<const MatrixInput>& _U) { U = _U; }
    void setInputMatrix(const Eigen::Ref<const MatrixInput>& _B) { B = _B; }
    void setProcessNoiseCovMatrix(const Eigen::Ref<const MatrixState>& _Q) { Q = _Q; }
    void setObservationMatrix(const Eigen::Ref<const MatrixObserve>& _H) { H = _H; }
    void setObservationNoiseCovMatrix(const Eigen::Ref<const MatrixObserveSqu>& _R) { R = _R; }

    virtual void predictNextState()
    {
        X_pre.emplace_front(A * X_est.front() + B * U);
        P_pre.emplace_front(A * P_est.front() * A.transpose() + Q);

        if (X_pre.size() > smooth_size + 1) {
            X_pre.pop_back();
            P_pre.pop_back();
        }
    }

    virtual void estimateCurrentState(const Eigen::Ref<const Eigen::Matrix<double, dim_observe, 1>>& _y)
    {
        C_deque.emplace_front(P_est.front() * A.transpose() * P_pre.front().inverse()); // TODO
        K = P_pre.front() * H.transpose() * (H * P_pre.front() * H.transpose() + R).inverse(); // TODO
        X_est.emplace_front(X_pre.front() + K * (_y - H * X_pre.front()));
        X_smth.emplace_front(X_est.front());
        P_est.emplace_front((MatrixState::Identity() - K * H) * P_pre.front());

        // Put condition here not to pop_back the empty C_deque
        if (X_est.size() > smooth_size + 1) {
            C_deque.pop_back();
            X_est.pop_back();
            X_smth.pop_back();
            P_est.pop_back();
        }

        // Kalman Smoothing
        if (smooth_size > 0) {
            // Don't need P_est_itr
            auto X_est_itr = X_est.begin();
            auto X_pre_itr = X_pre.begin();
            auto X_smth_itr = X_smth.begin();
            auto P_est_itr = P_est.begin();
            auto P_pre_itr = P_pre.begin();
            auto C_itr = C_deque.begin();
            std::advance(X_est_itr, 1);
            std::advance(X_pre_itr, 1);
            std::advance(X_smth_itr, 1);
            std::advance(P_est_itr, 1);
            std::advance(P_pre_itr, 1);
            for (; X_est_itr != X_est.end(); ++X_est_itr, ++X_pre_itr, ++X_smth_itr,
                                             ++P_est_itr, ++P_pre_itr, ++C_itr) {
                *X_smth_itr = *X_est_itr + *C_itr * (*(X_smth_itr - 1) - *X_pre_itr);
            }
        }
    }
};

#endif // __KALMANFILTER_H__
