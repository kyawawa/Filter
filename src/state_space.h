// -*- mode: C++; coding: utf-8-unix; -*-

/**
 * @file  state_space.h
 * @brief State-space representation with noise
 * @author Tatsuya Ishikawa
 */

#ifndef __STATE_SPACE_H__
#define __STATE_SPACE_H__

#include <functional>
#include <Eigen/Core>

/// x = Ax + Bu
/// y = Hx + Du
template<size_t dim_state, size_t dim_observe, size_t dim_input>
class LinearSystem
{
    using VectorState       = Eigen::Matrix<double, dim_state, 1>;
    using MatrixState       = Eigen::Matrix<double, dim_state, dim_state>;
    using VectorInput       = Eigen::Matrix<double, dim_input, 1>;
    using MatrixInput       = Eigen::Matrix<double, dim_state, dim_input>;
    using VectorObserve     = Eigen::Matrix<double, dim_observe, 1>;
    using MatrixObserve     = Eigen::Matrix<double, dim_observe, dim_state>;
    using MatrixFeedThrough = Eigen::Matrix<double, dim_observe, dim_input>;

    VectorState       X_;  // State vector
    MatrixState       A_;  // State matrix
    VectorInput       U_;  // Input vector
    MatrixInput       B_;  // Input matrix
    VectorObserve     Y_;  // Observation vector
    MatrixObserve     H_;  // Observation matrix
    MatrixFeedThrough D_;  // Feedthrough matrix

    virtual VectorState calcNextState() { return A_ * X_ + B_ * U_; }
    virtual VectorObserve calcObservedState() { return H_ * X_ + D_ * U_; }

  public:
    LinearSystem(const Eigen::Ref<const VectorState>& _Xinit,
                 const Eigen::Ref<const MatrixState>& _A,
                 const Eigen::Ref<const VectorInput>& _U,
                 const Eigen::Ref<const MatrixInput>& _B,
                 const Eigen::Ref<const MatrixObserve>& _H,
                 const Eigen::Ref<const MatrixFeedThrough>& _D)
        : X_(_Xinit), A_(_A), U_(_U), B_(_B), H_(_H), D_(_D)
    {
        Y_.setZero();
    }

    auto getStateVector()       const { return X_; }  // Only for debug
    auto getStateMatrix()       const { return A_; }
    auto getInputVector()       const { return U_; }
    auto getInputMatrix()       const { return B_; }
    auto getObservedState()     const { return Y_; }
    auto getObservationMatrix() const { return H_; }
    auto getFeedThroughMatrix() const { return D_; }

    void setStateVector(const Eigen::Ref<const VectorState>& _X) { X_ = _X; };
    void setStateMatrix(const Eigen::Ref<const MatrixState>& _A) { A_ = _A; };
    void setInputVector(const Eigen::Ref<const MatrixInput>& _U) { U_ = _U; };
    void setInputMatrix(const Eigen::Ref<const MatrixInput>& _B) { B_ = _B; };
    void setObservationMatrix(const Eigen::Ref<const MatrixObserve>& _H) { H_ = _H; };
    void setFeedThroughMatrix(const Eigen::Ref<const MatrixFeedThrough>& _D) { D_ = _D; };

    void goNextStep()
    {
        X_ = calcNextState();
        Y_ = calcObservedState();
    }
};


/// x = f(t, x(t), u(t))
/// y = h(t, x(t), u(t))
template<size_t dim_state, size_t dim_observe, size_t dim_input>
class NonLinearSystem
{
    using VectorState      = Eigen::Matrix<double, dim_state, 1>;
    using VectorInput      = Eigen::Matrix<double, dim_input, 1>;
    using VectorObserve    = Eigen::Matrix<double, dim_observe, 1>;
    using StateFunction    = std::function<VectorState
                                           (const double t,
                                            const Eigen::Ref<const VectorState>& _X,
                                            const Eigen::Ref<const VectorInput>& _U)>;
    using ObserveFunction  = std::function<VectorObserve
                                           (const double t,
                                            const Eigen::Ref<const VectorState>& _X,
                                            const Eigen::Ref<const VectorInput>& _U)>;

    VectorState     X_;  // State vector
    VectorInput     U_;  // Input vector
    VectorObserve   Y_;  // Observation vector
    StateFunction   f_;
    ObserveFunction h_;

    const double dt_;
    unsigned int count_ = 0;

    virtual VectorState calcNextState() { return f_(count_ * dt_, X_, U_); }
    virtual VectorObserve calcObservedState() { return h_(count_ * dt_, X_, U_); }

  public:
    NonLinearSystem(const Eigen::Ref<const VectorState>& _Xinit,
                    const Eigen::Ref<const VectorInput>& _U,
                    const StateFunction _f,
                    const ObserveFunction _h,
                    const double _dt)
        : X_(_Xinit), U_(_U), f_(_f), h_(_h), dt_(_dt)
    {
        Y_.setZero();
    }

    auto getStateVector()     const { return X_; }  // Only for debug
    auto getInputVector()     const { return U_; }
    auto getObservedState()   const { return Y_; }
    auto getStateFunction()   const { return f_; }
    auto getObserveFunction() const { return h_; }

    void setStateVector(const Eigen::Ref<const VectorState>& _X) { X_ = _X; };
    void setInputVector(const Eigen::Ref<const VectorInput>& _U) { U_ = _U; };
    void setStateFunction(const StateFunction _f) { f_ = _f; };
    void setObserveFunction(const ObserveFunction _h) { h_ = _h; };

    void goNextStep()
    {
        X_ = calcNextState();
        Y_ = calcObservedState();
        ++count_;
    }
};

#endif // __STATE_SPACE_H__
