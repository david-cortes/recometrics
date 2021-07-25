/*   BSD 2-Clause License

    Copyright (c) 2021, David Cortes
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
#ifndef _FOR_R

#include "recometrics.hpp"

bool get_has_openmp()
{
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif
}

/* Note: for some reason, Cython generates uncompilable C++ code when using the
   templates in the header, so they had to be defined with explicit types.
   Otherwise, this file and the 'signatures' header are not needed. */

void calc_metrics_double
(
    const double *restrict A, const size_t lda, const double *restrict B, const size_t ldb,
    const int32_t m, const int32_t n, const int32_t k,
    const int32_t *restrict Xtrain_csr_p, const int32_t *restrict Xtrain_csr_i,
    const int32_t *restrict Xtest_csr_p, int32_t *restrict Xtest_csr_i, const double *restrict Xtest_csr,
    const int32_t k_metrics,
    const bool cumulative,
    const bool break_ties_with_noise,
    double *restrict p_at_k,
    double *restrict tp_at_k,
    double *restrict r_at_k,
    double *restrict ap_at_k,
    double *restrict tap_at_k,
    double *restrict ndcg_at_k,
    double *restrict hit_at_k,
    double *restrict rr_at_k,
    double *restrict roc_auc,
    double *restrict pr_auc,
    const bool consider_cold_start,
    int32_t min_items_pool,
    int32_t min_pos_test,
    int32_t nthreads,
    uint64_t seed
)
{
    return calc_metrics<double>(
        A, lda, B, ldb, m, n, k,
        Xtrain_csr_p, Xtrain_csr_i,
        Xtest_csr_p, Xtest_csr_i, Xtest_csr,
        k_metrics,
        cumulative,
        break_ties_with_noise,
        p_at_k,
        tp_at_k,
        r_at_k,
        ap_at_k,
        tap_at_k,
        ndcg_at_k,
        hit_at_k,
        rr_at_k,
        roc_auc,
        pr_auc,
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        nthreads,
        seed
    );
}

void calc_metrics_float
(
    const float *restrict A, const size_t lda, const float *restrict B, const size_t ldb,
    const int32_t m, const int32_t n, const int32_t k,
    const int32_t *restrict Xtrain_csr_p, const int32_t *restrict Xtrain_csr_i,
    const int32_t *restrict Xtest_csr_p, int32_t *restrict Xtest_csr_i, const float *restrict Xtest_csr,
    const int32_t k_metrics,
    const bool cumulative,
    const bool break_ties_with_noise,
    float *restrict p_at_k,
    float *restrict tp_at_k,
    float *restrict r_at_k,
    float *restrict ap_at_k,
    float *restrict tap_at_k,
    float *restrict ndcg_at_k,
    float *restrict hit_at_k,
    float *restrict rr_at_k,
    float *restrict roc_auc,
    float *restrict pr_auc,
    const bool consider_cold_start,
    int32_t min_items_pool,
    int32_t min_pos_test,
    int32_t nthreads,
    uint64_t seed
)
{
    return calc_metrics<float>(
        A, lda, B, ldb, m, n, k,
        Xtrain_csr_p, Xtrain_csr_i,
        Xtest_csr_p, Xtest_csr_i, Xtest_csr,
        k_metrics,
        cumulative,
        break_ties_with_noise,
        p_at_k,
        tp_at_k,
        r_at_k,
        ap_at_k,
        tap_at_k,
        ndcg_at_k,
        hit_at_k,
        rr_at_k,
        roc_auc,
        pr_auc,
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        nthreads,
        seed
    );
}

void split_data_selected_users_double
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const double *restrict X_csr,
    const int32_t m, const int32_t n,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<double> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<double> &Xtest_csr,
    const double test_fraction,
    uint64_t seed
)
{
    return split_data_selected_users<double>(
        X_csr_p,
        X_csr_i,
        X_csr,
        m, n,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        test_fraction,
        seed
    );
}

void split_data_selected_users_float
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const float *restrict X_csr,
    const int32_t m, const int32_t n,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<float> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<float> &Xtest_csr,
    const float test_fraction,
    uint64_t seed
)
{
    return split_data_selected_users<float>(
        X_csr_p,
        X_csr_i,
        X_csr,
        m, n,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        test_fraction,
        seed
    );
}

void split_data_separate_users_double
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const double *restrict X_csr,
    int32_t m, int32_t n,
    std::vector<int32_t> &users_test,
    std::vector<int32_t> &Xrem_csr_p,
    std::vector<int32_t> &Xrem_csr_i,
    std::vector<double> &Xrem_csr,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<double> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<double> &Xtest_csr,
    const int32_t n_users_test,
    const double test_fraction,
    const bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    uint64_t seed
)
{
    return split_data_separate_users<double>(
        X_csr_p,
        X_csr_i,
        X_csr,
        m, n,
        users_test,
        Xrem_csr_p,
        Xrem_csr_i,
        Xrem_csr,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        n_users_test,
        test_fraction,
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        seed
    );
}

void split_data_separate_users_float
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const float *restrict X_csr,
    int32_t m, int32_t n,
    std::vector<int32_t> &users_test,
    std::vector<int32_t> &Xrem_csr_p,
    std::vector<int32_t> &Xrem_csr_i,
    std::vector<float> &Xrem_csr,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<float> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<float> &Xtest_csr,
    const int32_t n_users_test,
    const float test_fraction,
    const bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    uint64_t seed
)
{
    return split_data_separate_users<float>(
        X_csr_p,
        X_csr_i,
        X_csr,
        m, n,
        users_test,
        Xrem_csr_p,
        Xrem_csr_i,
        Xrem_csr,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        n_users_test,
        test_fraction,
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        seed
    );
}

void split_data_joined_users_double
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const double *restrict X_csr,
    int32_t m, int32_t n,
    std::vector<int32_t> &users_test,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<double> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<double> &Xtest_csr,
    const int32_t n_users_test,
    const double test_fraction,
    const bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    uint64_t seed
)
{
    return split_data_joined_users<double>(
        X_csr_p,
        X_csr_i,
        X_csr,
        m, n,
        users_test,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        n_users_test,
        test_fraction,
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        seed
    );
}

void split_data_joined_users_float
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const float *restrict X_csr,
    int32_t m, int32_t n,
    std::vector<int32_t> &users_test,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<float> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<float> &Xtest_csr,
    const int32_t n_users_test,
    const float test_fraction,
    const bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    uint64_t seed
)
{
    return split_data_joined_users<float>(
        X_csr_p,
        X_csr_i,
        X_csr,
        m, n,
        users_test,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        n_users_test,
        test_fraction,
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        seed
    );
}

#endif
