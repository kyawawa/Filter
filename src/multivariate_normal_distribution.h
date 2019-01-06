// -*- mode: C++; coding: utf-8-unix; -*-

/**
 * @file  multivariate_normal_distribution.h
 * @brief
 * @author Tatsuya Ishikawa
 */

// https://forum.kde.org/viewtopic.php?f=74&t=95260

#ifndef __MULTIVARIATE_NORMAL_DISTRIBUTION_H__
#define __MULTIVARIATE_NORMAL_DISTRIBUTION_H__

#include <random>
#include <Eigen/Dense>

template<size_t dim>
class MultiVariateNormalDistribution
{
    using VectorDim = Eigen::Matrix<double, dim, 1>;
    using MatrixDim = Eigen::Matrix<double, dim, dim>;

    std::normal_distribution<double> standard_normal_dist{0.0, 1.0};

    VectorDim mean;
    MatrixDim covar;
    MatrixDim transform;

  public:
    MultiVariateNormalDistribution(const Eigen::Ref<const VectorDim>& _mean,
                                   const Eigen::Ref<const MatrixDim>& _covar)
        : mean(_mean), covar(_covar)
    {
        const Eigen::LLT<MatrixDim> llt_solver(covar);
        if (llt_solver.info() == Eigen::Success) {
            transform = llt_solver.matrixL();
        } else {
            const Eigen::LDLT<MatrixDim> ldlt_solver(covar);
            if (ldlt_solver.info() == Eigen::Success) {
                transform = ldlt_solver.matrixL();
            } else {
                const Eigen::SelfAdjointEigenSolver<MatrixDim> eigen_solver(covar);
                transform = eigen_solver.eigenvectors() *
                    eigen_solver.eigenvalues().cwiseSqrt().asDiagonal();
            }
        }
    }

    template<class URBG>
    const VectorDim operator() (URBG& g)
    {
        VectorDim standard_dist_vec;
        // Is it better to use NullaryExpr?
        for (size_t i = 0; i < dim; ++i) {
            standard_dist_vec[i] = standard_normal_dist(g);
        }
        return mean + transform * standard_dist_vec;
    }
};

#endif // __MULTIVARIATE_NORMAL_DISTRIBUTION_H__
