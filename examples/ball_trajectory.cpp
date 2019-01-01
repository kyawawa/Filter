// -*- mode: C++; coding: utf-8-unix; -*-

/**
 * @file  ball_trajectory.cpp
 * @brief Estimate ball trajectory using with Kalman filter
 * @author Tatsuya Ishikawa
 */

#include <random>
#include <memory>
#include <type_traits>
#include <csignal>
#include "matplotlibcpp.h"
#include "../src/kalman_filter.h"

namespace {
namespace plt = matplotlibcpp;
constexpr double G_ACC = -9.80665;
}

/*
 * X = AX + u
 * X: (x z dx dz)', u: (0 0 0 -g)'
 */

class BallTrajectory
{
  protected:
    // True value
    double x0;
    double z0;
    double dx0;
    double dz0;
    // Discrete state-space representation
    unsigned int count = 0;
    double dt;
    Eigen::Vector4d X;  // State vector
    Eigen::Matrix4d A;  // State matrix
    Eigen::Matrix4d B;  // Input matrix
    Eigen::Vector4d U;  // Input (gravity)
    Eigen::Matrix4d Q;  // Covariance matrix of the process noise
    Eigen::Vector2d Y;  // Observation vector
    Eigen::Matrix<double, 2, 4> H;  // Observation matrix
    Eigen::Matrix2d R;  // Covariance matrix of the observation noise

    virtual Eigen::Vector2d calcObservedState() { return H * X; }

  public:
    BallTrajectory(const double x_init, const double z_init,
                   const double dx_init, const double dz_init,
                   const double _dt = 0.05)
        : x0(x_init), z0(z_init), dx0(dx_init), dz0(dz_init), dt(_dt)
    {
        X << x_init, z_init, dx_init, dz_init;
        A << 1, 0, _dt, 0,
             0, 1, 0, _dt,
             0, 0, 1, 0,
             0, 0, 0, 1;
        B.setIdentity();
        Q = 0.00001 * decltype(Q)::Identity();
        U << 0, G_ACC / 2 * _dt*_dt, 0, G_ACC * _dt;
        Y << 0, 0;
        H << 1, 0, 0, 0,
             0, 1, 0, 0;
        R.setZero();
    }

    auto getStateVector() const { return X; }
    auto getStateMatrix() const { return A; }
    auto getInputVector() const { return U; }
    auto getInputMatrix() const { return B; }
    auto getProcessNoiseCovMatrix() const { return Q; }
    auto getObservationMatrix() const { return H; }
    auto getObservationNoiseCovMatrix() const { return R; }

    void goNextStep()
    {
        X = A * X + U;
        Y = calcObservedState();
        ++count;
    }
    virtual Eigen::Vector2d getObservedState() { return Y; }

    // Return true value
    Eigen::Vector2d calcTrueCurrentPosition() const
    {
        const double t = count * dt;
        return Eigen::Vector2d(x0 + dx0 * t, z0 + G_ACC / 2 * t*t + dz0 * t);
    }
};

// Add gaussian observation noise
class GaussianNoiseBallTrajectory : public BallTrajectory
{
    // Temporary until implement multivariate normal distributions
    std::mt19937 rand_engine{12345};
    std::normal_distribution<double> x_noise;
    std::normal_distribution<double> z_noise;

    Eigen::Vector2d calcObservedState() override
    {
        return H * X + Eigen::Vector2d(x_noise(rand_engine), z_noise(rand_engine));
    }

  public:
    GaussianNoiseBallTrajectory(const double x_init, const double z_init,
                                const double dx_init, const double dz_init,
                                const double _dt = 0.05,
                                const double var_x = 0.05, const double var_z = 0.01)
        : BallTrajectory(x_init, z_init, dx_init, dz_init, _dt)
    {
        x_noise = std::normal_distribution<double>(0.0, sqrt(var_x));
        z_noise = std::normal_distribution<double>(0.0, sqrt(var_z));
        R << var_x, 0, 0, var_z;
    }
};


template<class StateSpace, class KalmanFilter>
class StateSpaceKalmanFilter
{
    std::unique_ptr<StateSpace> state_space;
    std::unique_ptr<KalmanFilter> kalman_filter;

  public:
    StateSpaceKalmanFilter(std::unique_ptr<StateSpace> _state, std::unique_ptr<KalmanFilter> _kalman)
        : state_space(std::move(_state)), kalman_filter(std::move(_kalman))
    {
    }

    auto getTrueStateVector() { return state_space->getStateVector(); }
    auto getObservedState() { return state_space->getObservedState(); }
    auto calcTrueCurrentPosition() { return state_space->calcTrueCurrentPosition(); }
    auto getEstimatedStateVector() { return kalman_filter->getStateVector(); }

    void goNextStep()
    {
        state_space->goNextStep();
        kalman_filter->estimateCurrentState(state_space->getObservedState());
    }
    void predictNextState() { kalman_filter->predictNextState(); }
    // void estimateCurrentState() { kalman_filter->estimateCurrentState(); }
};

int main(int argc, char **argv)
{
    constexpr double dt = 0.02;

    auto ball = std::make_unique<GaussianNoiseBallTrajectory>(0, 1, 0.5, 0.5, dt, 0.002, 0.001);
    decltype(ball->getStateVector()) kalman_state_init(0, 10, 0.7, 0.3);
    // auto kalman = std::make_unique<SteadyKalmanFilter<4, 2, 4>>
    auto kalman = std::make_unique<KalmanFilter<4, 2, 4>>
        (kalman_state_init, ball->getStateMatrix(),
         ball->getInputVector(), ball->getInputMatrix(),
         ball->getProcessNoiseCovMatrix(), ball->getObservationMatrix(),
         ball->getObservationNoiseCovMatrix());
    auto ball_kalman = StateSpaceKalmanFilter
        <std::decay<decltype(*ball)>::type, std::decay<decltype(*kalman)>::type>
        (std::move(ball), std::move(kalman));

    constexpr double max_time = 10.0;  // [s]
    constexpr double max_count = max_time / dt;
    std::vector<double> time_list;
    std::vector<double> true_pos_x;
    std::vector<double> true_pos_z;
    std::vector<double> true_state_x;
    std::vector<double> true_state_z;
    std::vector<double> predicted_pos_x;
    std::vector<double> predicted_pos_z;
    std::vector<double> estimated_pos_x;
    std::vector<double> estimated_pos_z;
    std::vector<double> diff_state_estimated_x;
    std::vector<double> diff_state_estimated_z;
    // std::vector<double> diff_estimated_predict_x;
    // std::vector<double> diff_estimated_predict_z;
    time_list.reserve(max_count);
    true_pos_x.reserve(max_count);
    true_pos_z.reserve(max_count);
    true_state_x.reserve(max_count);
    true_state_z.reserve(max_count);
    predicted_pos_x.reserve(max_count);
    predicted_pos_z.reserve(max_count);
    estimated_pos_x.reserve(max_count);
    estimated_pos_z.reserve(max_count);
    diff_state_estimated_x.reserve(max_count);
    diff_state_estimated_z.reserve(max_count);
    // diff_estimated_predict_x.reserve(max_count);
    // diff_estimated_predict_z.reserve(max_count);

    // Issue: Now, first state is dropped
    for (size_t i = 0; i < max_count; ++i) {
        ball_kalman.predictNextState();

        time_list.emplace_back(i * dt);
        {
            const auto predicted_state = ball_kalman.getEstimatedStateVector().segment<2>(0);
            predicted_pos_x.emplace_back(predicted_state[0]);
            predicted_pos_z.emplace_back(predicted_state[1]);
        }

        ball_kalman.goNextStep();

        {
            const auto true_pos = ball_kalman.calcTrueCurrentPosition();
            true_pos_x.emplace_back(true_pos[0]);
            true_pos_z.emplace_back(true_pos[1]);
        }
        {
            const auto true_state = ball_kalman.getTrueStateVector();
            true_state_x.emplace_back(true_state[0]);
            true_state_z.emplace_back(true_state[1]);
        }
        {
            const auto estimated_state = ball_kalman.getEstimatedStateVector().segment<2>(0);
            estimated_pos_x.emplace_back(estimated_state[0]);
            estimated_pos_z.emplace_back(estimated_state[1]);
        }
        {
            const auto diff = ball_kalman.getTrueStateVector() - ball_kalman.getEstimatedStateVector();
            diff_state_estimated_x.emplace_back(diff[0]);
            diff_state_estimated_z.emplace_back(diff[1]);
        }
    }

    plt::subplot(2, 2, 1);
    plt::title("Ball Pos X");
    plt::named_plot("True Pos X", time_list, true_pos_x, "--b");
    plt::named_plot("True State X", time_list, true_state_x, "--r");
    plt::named_plot("Predicted X", time_list, predicted_pos_x, "--g");
    plt::named_plot("Estimated X", time_list, estimated_pos_x, "--y");
    plt::legend();

    plt::subplot(2, 2, 2);
    plt::title("Ball Pos Z");
    plt::named_plot("True Pos Z", time_list, true_pos_z, "--b");
    plt::named_plot("True State Z", time_list, true_state_z, "--r");
    plt::named_plot("Predicted Z", time_list, predicted_pos_z, "--g");
    plt::named_plot("Estimated Z", time_list, estimated_pos_z, "--y");
    plt::legend();

    plt::subplot(2, 2, 3);
    plt::title("Diff True and Estimated X");
    plt::plot(time_list, diff_state_estimated_x, "--b");
    plt::legend();

    plt::subplot(2, 2, 4);
    plt::title("Diff True and Estimated Z");
    plt::plot(time_list, diff_state_estimated_z, "--b");
    plt::legend();

    std::signal(SIGINT, SIG_DFL);  // Kill plot by Ctrl-c
    plt::show();

    return 0;
}
