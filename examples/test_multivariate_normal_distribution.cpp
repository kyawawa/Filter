// -*- mode: C++; coding: utf-8-unix; -*-

/**
 * @file  test_multivariate_normal_distribution.cpp
 * @brief
 * @author Tatsuya Ishikawa
 */

#include <csignal>
#include <vector>
#include "matplotlibcpp.h"
#include "../src/multivariate_normal_distribution.h"

namespace {
namespace plt = matplotlibcpp;
}

int main()
{
    std::mt19937 rand_engine{12345};
    const auto mean = Eigen::Vector2d::Ones();
    Eigen::Matrix2d cov;
    cov << 1, 0.5, 0.5, 1;
    auto dist = filter::MultiVariateNormalDistribution<2>(mean, cov);

    constexpr size_t plot_num = 1000;
    std::vector<double> x;
    std::vector<double> y;
    x.reserve(plot_num);
    y.reserve(plot_num);
    for (size_t i = 0; i < plot_num; ++i) {
        const Eigen::Vector2d data = dist(rand_engine);
        x.emplace_back(data[0]);
        y.emplace_back(data[1]);
    }

    plt::title("Multi Variate Normal Distribution: mean (1,1), cov(1,0.5,0.5,1)");
    plt::plot(x, y, "o ");
    plt::grid(true);
    std::signal(SIGINT, SIG_DFL);
    plt::show();

    return 0;
}
