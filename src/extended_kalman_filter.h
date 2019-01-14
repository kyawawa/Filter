// -*- mode: C++; coding: utf-8-unix; -*-

/**
 * @file  extended_kalman_filter.h
 * @brief
 * @author Tatsuya Ishikawa
 */

#ifndef __EXTENDED_KALMAN_FILTER_H__
#define __EXTENDED_KALMAN_FILTER_H__

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

namespace filter {
template<size_t dim_state, size_t dim_observe, size_t dim_input>
class ExtendedKalmanFilter
{
    using VectorState      = Eigen::Matrix<double, dim_state, 1>;
    using MatrixState      = Eigen::Matrix<double, dim_state, dim_state>;
    using VectorInput      = Eigen::Matrix<double, dim_input, 1>;
    using MatrixInput      = Eigen::Matrix<double, dim_state, dim_input>;
    using MatrixObserve    = Eigen::Matrix<double, dim_observe, dim_state>;
    using VectorObserve    = Eigen::Matrix<double, dim_observe, 1>;
    using MatrixObserveSqu = Eigen::Matrix<double, dim_observe, dim_observe>;
    using MatrixKalmanGain = Eigen::Matrix<double, dim_state, dim_observe>;
    // For autodiff
    using ActiveScalar     = Eigen::AutoDiffScalar<VectorState>;
    using ActiveState      = Eigen::Matrix<ActiveScalar, dim_state, 1>;
    using ActiveObserve    = Eigen::Matrix<ActiveScalar, dim_observe, 1>;
    using StateFunction    = std::function<ActiveState(const ActiveState&,
                                                       const VectorInput&)>;
    using ObserveFunction  = std::function<ActiveObserve(const ActiveState&)>;

    // Estimate
    VectorState      X;  // State vector
    MatrixState      P;  // Error covariance matrix
    MatrixKalmanGain K;  // Kalman gain
    // Parameter
    StateFunction    state_func;  // State function
    VectorInput      U;  // Input vector
    MatrixState      Q;  // Covariance matrix of the process noise
    ObserveFunction  observe_func;  // Observe function
    MatrixObserveSqu R;  // Covariance matrix of the observation noise

    struct state_functor
    {
        const ExtendedKalmanFilter<dim_state, dim_observe, dim_input>& super;
        using InputType    = VectorState;
        using ValueType    = VectorState;
        using JacobianType = MatrixState;

        enum {
            InputsAtCompileTime = InputType::RowsAtCompileTime,
            ValuesAtCompileTime = ValueType::RowsAtCompileTime
        };

        state_functor(const ExtendedKalmanFilter<dim_state, dim_observe, dim_input>& _super) : super(_super) {}

        size_t inputs() const { return InputsAtCompileTime; }

        void operator() (const InputType &input, ValueType *output) const
        {
            ActiveState ax = input.template cast<ActiveScalar>();
            ActiveState active_out = super.state_func(ax, super.U);
            for (size_t i = 0; i < dim_state; ++i)
                (*output)[i] = active_out[i].value();
        }

        void operator() (const ActiveState &input, ActiveState *output) const
        {
            *output = super.state_func(input, super.U);
        }
    };

    struct observe_functor
    {
        const ExtendedKalmanFilter<dim_state, dim_observe, dim_input>& super;
        using InputType    = VectorState;
        using ValueType    = VectorObserve;
        using JacobianType = MatrixObserve;

        enum {
            InputsAtCompileTime = InputType::RowsAtCompileTime,
            ValuesAtCompileTime = ValueType::RowsAtCompileTime
        };

        observe_functor(const ExtendedKalmanFilter<dim_state, dim_observe, dim_input>& _super) : super(_super) {}

        size_t inputs() const { return InputsAtCompileTime; }

        void operator() (const InputType &input, ValueType *output) const
        {
            ActiveState ax = input.template cast<ActiveScalar>();
            ActiveObserve active_out = super.observe_func(ax);
            for (size_t i = 0; i < dim_observe; ++i)
                (*output)[i] = active_out[i].value();
        }

        void operator() (const ActiveState &input, ActiveObserve *output) const
        {
            *output = super.observe_func(input);
        }
    };

    Eigen::AutoDiffJacobian<state_functor> state_jacobian{*this};
    Eigen::AutoDiffJacobian<observe_functor> observe_jacobian{*this};

  public:
    ExtendedKalmanFilter(const Eigen::Ref<const VectorState>& _Xinit,
                         StateFunction _state_func,
                         const Eigen::Ref<const VectorInput>& _U,
                         const Eigen::Ref<const MatrixState>& _Q,
                         ObserveFunction _observe_func,
                         const Eigen::Ref<const MatrixObserveSqu>& _R)
        : X(_Xinit), state_func(_state_func), U(_U), Q(_Q), observe_func(_observe_func), R(_R)
    {
        MatrixObserve observe_jac;
        VectorObserve estimated_observe;
        observe_jacobian(X, &estimated_observe, &observe_jac);

        K = Q * observe_jac.transpose() * (observe_jac * Q * observe_jac.transpose() + R).inverse(); // TODO
        P = (MatrixState::Identity() - K * observe_jac) * Q;
    }

    auto getStateVector() const { return X; };

    void setStateVector(const Eigen::Ref<const VectorState>& _X) { X = _X; }
    void setStateFunction(StateFunction _state_func) { state_func = _state_func; }
    void setInputVector(const Eigen::Ref<const MatrixInput>& _U) { U = _U; }
    void setProcessNoiseCovMatrix(const Eigen::Ref<const MatrixState>& _Q) { Q = _Q; }
    void setObservationFunction(ObserveFunction _observe_func) { observe_func = _observe_func; }
    void setObservationNoiseCovMatrix(const Eigen::Ref<const MatrixObserveSqu>& _R) { R = _R; }

    void predictNextState()
    {
        MatrixState state_jac;
        state_jacobian(X, &X, &state_jac);  //  Calc jacobian and update X
        std::cerr << "sta jac:\n" << state_jac << std::endl;
        P = state_jac * P * state_jac.transpose() + Q;
    }

    void estimateCurrentState(const Eigen::Ref<const VectorObserve>& _y)
    {
        MatrixObserve observe_jac;
        VectorObserve estimated_observe;
        observe_jacobian(X, &estimated_observe, &observe_jac);
        std::cerr << "est obs:\n" << estimated_observe << std::endl;
        std::cerr << "obs jac:\n" << observe_jac << std::endl;

        K = P * observe_jac.transpose() * (observe_jac * P * observe_jac.transpose() + R).inverse(); // TODO
        X = X + K * (_y - estimated_observe);
        P = (MatrixState::Identity() - K * observe_jac) * P;
    }
};
} // end of namespace filter

#endif // __EXTENDED_KALMAN_FILTER_H__
