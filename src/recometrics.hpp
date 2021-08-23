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
#pragma once
#include <algorithm>
#include <numeric>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <cinttypes>
#include <random>
#include <vector>
#include <stdexcept>
#include <csignal>
#ifdef _OPENMP
#   include <omp.h>
#else
#   define omp_get_thread_num() 0
#endif

using std::size_t;
using std::uint64_t;
using std::int32_t;


#if defined(__GNUG__) || defined(__GNUC__) || defined(_MSC_VER) || defined(__clang__) || defined(__INTEL_COMPILER) || defined(SUPPORTS_RESTRICT)
#   define restrict __restrict
#else
#   define restrict 
#endif

#define restrict __restrict

#if defined(_FOR_R)
extern "C" double ddot_(const int *n, const double *dx, const int *incx, const double *dy, const int *incy);
extern "C" float sdot_(const int *n, const float *dx, const int *incx, const float *dy, const int *incy);
#endif

#ifndef _FOR_R
#   define NAN_ NAN
#else
#   include <R.h>
#   define NAN_ ((std::is_same<real_t, double>::value)? NA_REAL : NAN)
#endif

constexpr const int one = 1;

[[gnu::hot]]
static inline double dot1(const double *restrict x, const double *restrict y, const int n)
{
#if defined(_FOR_R)
    return ddot_(&n, x, &one, y, &one);
#else
    double res = 0;
    #ifndef _MSC_VER
    #pragma omp simd
    #endif
    for (int32_t ix = 0; ix < n; ix++) res += x[ix]*y[ix];
    return res;
#endif
}

[[gnu::hot]]
static inline float dot1(const float *restrict x, const float *restrict y, const int n)
{
#if defined(_FOR_R)
    return sdot_(&n, x, &one, y, &one);
#else
    float res = 0;
    #ifndef _MSC_VER
    #pragma omp simd
    #endif
    for (int32_t ix = 0; ix < n; ix++) res += x[ix]*y[ix];
    return res;
#endif
}

/* This is in order to catch interrupt signals */
bool interrupt_switch = false;
bool handle_is_locked = false;
typedef void (*sig_t_)(int);
extern "C"
void set_interrup_global_variable(int s)
{
    #pragma omp critical
    {
        interrupt_switch = true;
    }
}
class SignalSwitcher
{
public:
    sig_t_ old_sig;
    bool is_active;
    SignalSwitcher() {
        #pragma omp critical
        {
            if (!handle_is_locked) {
                handle_is_locked = true;
                interrupt_switch = false;
                this->old_sig = std::signal(SIGINT, set_interrup_global_variable);
                this->is_active = true;
            }

            else {
                this->is_active = false;
            }
        }
    }
    
    ~SignalSwitcher() {
        #pragma omp critical
        {
            if (this->is_active && handle_is_locked)
                interrupt_switch = false;
        }
    }

    void restore_handle() {
        #pragma omp critical
        {
            if (this->is_active && handle_is_locked) {
                std::signal(SIGINT, this->old_sig);
                this->is_active = false;
                handle_is_locked = false;
            }
        }
    }

    void check_interrupt_switch() {
        if (interrupt_switch) {
            this->restore_handle();
            std::raise(SIGINT);
            throw std::runtime_error("Error: procedure was interrupted.\n");
        }
    }
};

/*  Note: at a first glance, it might seem that some things here are rather inefficient.
    Nevertheless, for a typical use case, the following ideas would only result in a slowdown
    when compiled with GCC10, and would not result in something that runs faster than GCC10
    when compiled with CLANG10.

    List of failed ideas:
    - Using BLAS gemv when the amount of items to predict is large (with MKL, it can speed up by
      around 10% the 'float' version, but slightly slows down the 'double' version, whereas with
      OpenBLAS it makes both slower).
    - Adding an extra layer of indices in such a way that the predictions would only need to
      be filled in a contiguous array (i.e. 'pred_thread[ix]' vs. 'pred_thread[ind_thread[ix]]').
      Running time is the same as current, but it would use more memory.
    - Allocating the temporary buffers contiguously for the same thread (no speed up).
    - Using a struct of index+value (slowdown for 'double', no effect for 'float').
*/


/*  Calculate implicit-feedback recommendation metrics for matrix factorization models

    Note: the metrics for a given user will be set to NAN in the following situations:
        - All the rankeable items have the exact same predicted score.
        - One or more of the predicted scores evaluates to NAN.
        - There are only negative entries (no non-zero entries in the test data).
        - The number of available items to rank (between positive and negative) is
          smaller than the requested 'k', and the metric is not affected by the exact order
          within the top-K items (i.e. precision, recall, hit, will be NAN if there's
          'K' or fewer items after discarding those from the training data).
        - There are inconsistencies in the data (e.g. number of entries being greater
          than 'n').
        - A user does not meet the minimum criteria set by the configurable parameters
          for this function.
        - There are only positive entries (i.e. the user already consumed all the items).
          In this case, NDCG@K will still be calculated, while the rest will be set
          to NAN.
    
    Parameters
    ==========
    - A[m*lda]
        The user-factors model matrix, in row-major order.
        It is assumed that the model determines the score for a given item 'j' for user 'i' as:
            score = <A[i*k : (i+1)*k], B[j*k : (j+1)*k]>
        Where '<x, y>' denotes the inner product (a.k.a. dot product) between vectors 'x' and 'y'.
        It is assumed that higher scores are better, and the items for each user are ranked according
        to this score, in descending order.
    - lda
        Leading dimension of 'A'. Typically this is equal to 'k'.
        It is assumed that the vector for user 'i' ranges from positions
        'i*lda' through 'i*lda + k'.
    - B[n*ldb]
        The item-factors model matrix, in row-major order.
    - ldb
        Leading dimension of 'B'.
    - m
        The number of users (rows of 'A').
    - n
        The number of items (rows of 'B').
    - k
        Number of latent factors/components in the model (columns of 'A' and 'B').
    - Xtrain_csr_p[m+1]
        The training data, in CSR format (compressed sparse row). This one denotes the index pointer.
        The training data should be a sparse matrix of dimensions [m, n], holding the non-zero entries
        of items that were consumed by each user (users are rows, items are columns, and non-zero values
        denote confidence scores for the items that were consumed). Items in the test set should not be
        included in the training data passed to this function. This matrix should only contain users
        that are also in the test set, and they both should be in the same order (i.e. user representing
        row 'i' must be the same in 'A', in 'Xtrain', and in 'Xtest').
        Negative values are not supported, and if passed, will be assumed that such items are still
        deemed better for the user than those which are missing.
    - Xtrain_csr_i[nnz_train]
        The training data, in CSR format. This one denotes the non-zero indices of the items (rows of B).
        For a given user 'i', the items that he/she consumed start in this array at the position given in
        'Xtrain_csr_p[i]', and end at the position 'Xtrain_csr_p[i+1]'.
    - Xtest_csr_p[m+1]
        The test (hold-out) data, in CSR format. This one denotes the index pointer.
        Items in the training and test set for each user should not intersect.
    - Xtest_csr_i[nnz_test]
        The test (hold-out) data, in CSR format. This one denotes the non-zero item indices.
    - Xtest_csr[nnz_test]
        The test (hold-out) data, in CSR format. This one denotes the confidence scores (values) associated to
        each corresponding entry in 'Xtest_csr_i'.
        This is only used for NDCG calculation, otherwise one can pass 'NULL' here,
    - k_metrics
        Number of items in the ranking for which the metrics are to be calculated (e.g. as in 'P@K').
        Users which have fewer rankeable items than this will have their metrics filled with NAN.
    - cumulative
        Whether to calculate the metrics cumulatively (e.g. [P@1, P@2, P@3] if passing 'k_metrics=3')
        for all values of 'k' up to 'k_metrics', or only for the desired 'k' given by 'k_metrics'
        (e.g. only P@3 if passing 'k_metrics=3').
    - break_ties_with_noise
        Whether to add a small amount of noise ~Uniform(-1e-10, 1e-10) in order to break ties
        at random, in case there are any ties in the ranking. This is not recommended unless
        one expects ties (can happen if e.g. some factors are set to all-zeros for some items),
        as it has the potential to slightly alter the ranking.
    - p_at_k[m or m*k_metrics] (out)
        Array where to output the calculated P@K (precision-at-K) metric(s) for each user.
        If passing 'cumulative=true', must be a row-major array in which users are rows and
        the 'k_metrics' positions are columns.
        The P@K metric for a given 'K' denotes the proportion of items among the top-K recommended
        (after excluding those that were already in the training data) that can be found in the test
        set for that user.
        If passing 'NULL', this metric will not be calculated.
    - tp_at_k[m or m*k_metrics] (out)
        Array where to output the calculated "truncated" P@K metric(s) for each users.
        Same format as for 'p_at_k'.
        This metric is similar to P@K, but will divide the sum of hits by the minimum between
        'k_metrics' and the number of items instead.
    - r_at_k[m or m*k_metrics] (out)
        Array where to output the calculated R@K (recall-at-K) metric(s) for each user.
        Same format as for 'p_at_k'.
        This metric is similar to P@K, but will divide the sum of hits by the number of test
        items instead. It signals the number of test items that have been retrieved within
        top-K recommended.
    - ap_at_k[m or m*k_metrics] (out)
        Array where to output the calculated AP@K (average precision-at-k) metric(s) for each user.
        Same format as for 'p_at_k'.
        The AP@K metric is defined as:
            AP@K = (1/n_test_items) sum[i..k](P@i * (r_i in test_items))
        Note that many authors define AP@K differently, such as dividing by the minum between
        'k_metrics' and the number of test items, or calculating the mean for P@1..P@K, perhaps
        stopping earlier if all the test items have already been retrieved, or using other
        variations of either formula.
    - tap_at_k[m or m*k_metrics]
        Array where to output the calculated TAP@K (truncated average precision-at-k) metric(s)
        for each user.
        Same format as for 'p_at_k'.
        This is the similar as AP@K, but will instead divide by the minimum between 'k_metrics'
        and the number of test items.
    - ndcg_at_k[m or m*k_metrics] (out)
        Array where to output the calculated NDCG@K (normalized discounted cumulative gain) metric(s) for each user.
        Same format as for 'p_at_k'.
        The DCG@K metric (discounted cumulative gain) is calculated as:
          sum[i=1..K](confidence[i] / log2(i+1))
        Then, this metric is standardized by dividing it by the maximum achievable DCG@K given the data for
        that user, thus obtaining the NDCG@K metric.
        The calculation of NDCG assumes that all values are non-negative, but the implementation here
        can accommodate the presence of a small fraction of negative values.
        If passing 'NULL', this metric will not be calculated.
    - hit_at_k[m or m*k_metrics] (out)
        Array where to otuput the calculated Hit@K for each users.
        Same format as for 'p_at_k'.
        This metric is binary (0/1, but can be NAN for some users too), and indicates whether any
        of the top-K recommendationded items turned out to be among the test items for that user.
        The mean of this across users is typically called "Hit Rate".
    - rr_at_k[m or m*k_metrics] (out)
        Array where to output the calculated RR@K (reciprocal rank at K).
        Same format as for 'p_at_k'.
        This metric denotes the inverse of the rank of the first item in the top-K recommended
        that is among the items in the test set for that user, with cases in which no items
        intersect having a value of zero (e.g. if the first top-K recommendation that intersects
        with the test data is the one ranked second, RR for that user is 1/2).
        The mean of this across users is typically called "Mean Reciprocal Rank" or MRR.
    - roc_auc[m] (out)
        Array where to output the calculated ROC-AUC curve for each user (area under the
        receiver-operating characteristing curve).
        This metric is calculated from the full ranking of items, thus being independent
        of 'k_metrics'.
        If passing 'NULL', this metric will not be calculated.
    - pr_auc[m] (out)
        Array where to output the calculated PR-AUC curce for each users (area under the
        precision-recall curve).
        Same format as 'roc_auc'.
        It uses the rectangular method (which is fast but not so precise), thus the formula
        is equivalent to AP@K with K=N. Some authors call this the "average precison" instead.
        If passing 'NULL', this metric will not be calculated.
    - consider_cold_start
        Whether to calculate metrics in situations in which some user has test data but no positive
        (non-zero) entries in the training data. If passing 'false' and such cases are encountered,
        the metrics will be set to NAN.
    - min_items_pool
        Minimum number of items (sum of positive and negative items) that a user must have in order to
        calculate metrics for that user. If a given user does not meet the threshold, the metrics
        will be set to NAN.
    - min_pos_test
        Minimum number of positive entries (non-zero entries in the test set) that users need to have in
        order to calculate metrics for that user. If a given user does not meet the threshold, the metrics
        will be set to NAN.
    - nthreads
        Number of parallel threads to use. Parallelization is done at the user level, so passing
        more threads than there are users will not result in a speed up. Be aware that, the more
        threads that are used, the higher the memory consumption.
    - seed
        Seed used for random number generation. Only used when passing 'break_ties_with_noise=true'.
*/
template <class real_t>
void calc_metrics
(
    const real_t *restrict A, const size_t lda, const real_t *restrict B, const size_t ldb,
    const int32_t m, const int32_t n, const int32_t k,
    const int32_t *restrict Xtrain_csr_p, const int32_t *restrict Xtrain_csr_i,
    const int32_t *restrict Xtest_csr_p, int32_t *restrict Xtest_csr_i, const real_t *restrict Xtest_csr,
    const int32_t k_metrics,
    const bool cumulative,
    const bool break_ties_with_noise,
    real_t *restrict p_at_k,
    real_t *restrict tp_at_k,
    real_t *restrict r_at_k,
    real_t *restrict ap_at_k,
    real_t *restrict tap_at_k,
    real_t *restrict ndcg_at_k,
    real_t *restrict hit_at_k,
    real_t *restrict rr_at_k,
    real_t *restrict roc_auc,
    real_t *restrict pr_auc,
    const bool consider_cold_start,
    int32_t min_items_pool,
    int32_t min_pos_test,
    int32_t nthreads,
    uint64_t seed
)
{
    #ifndef _OPENMP
    nthreads = 1;
    #endif
    nthreads = std::max(nthreads, 1);
    min_items_pool = std::max(min_items_pool, k_metrics);
    min_items_pool = std::max(min_items_pool, 2);
    min_pos_test = std::min(min_pos_test, 1);
    const size_t size_buffers = (size_t)n * (size_t)nthreads;
    std::unique_ptr<real_t[]> pred_buffer(new real_t[size_buffers]);
    std::unique_ptr<int[]> ind_buffer(new int[size_buffers]);
    std::unique_ptr<bool[]> test_bool_buffer(new bool[(roc_auc || pr_auc)? size_buffers : 0]);

    real_t *restrict pred_thread = nullptr;
    int32_t *restrict ind_thread = nullptr;
    bool *restrict test_bool_thread = nullptr;
    const real_t *restrict A_thread = nullptr;

    #if defined(__GNUC__) && (__GNUC__ >= 5)
    #   pragma GCC diagnostic push
    #   pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    #endif
    real_t *restrict p_at_k_user = nullptr;
    real_t *restrict tp_at_k_user = nullptr;
    real_t *restrict r_at_k_user = nullptr;
    real_t *restrict ap_at_k_user = nullptr;
    real_t *restrict tap_at_k_user = nullptr;
    real_t *restrict ndcg_at_k_user = nullptr;
    real_t *restrict hit_at_k_user = nullptr;
    real_t *restrict rr_at_k_user = nullptr;
    #if defined(__GNUC__) && (__GNUC__ >= 5)
    #   pragma GCC diagnostic pop
    #endif

    const bool calc_top_metrics = p_at_k || tp_at_k ||  r_at_k || ap_at_k || tap_at_k || ndcg_at_k;
    const bool only_first_pos = (hit_at_k || rr_at_k)
                                    &&
                                !p_at_k && !tp_at_k && !r_at_k && !ap_at_k &&
                                !tap_at_k && !ndcg_at_k;

    SignalSwitcher ss;

    #pragma omp parallel for schedule(dynamic) num_threads(nthreads) \
            private(pred_thread, ind_thread, test_bool_thread, A_thread, \
                    p_at_k_user, tp_at_k_user, r_at_k_user, ap_at_k_user, tap_at_k_user, \
                    ndcg_at_k_user, hit_at_k_user, rr_at_k_user) \
            shared(p_at_k, tp_at_k, r_at_k, ap_at_k, tap_at_k, ndcg_at_k, hit_at_k, rr_at_k, roc_auc, pr_auc, \
                   A, B, \
                   Xtrain_csr_p, Xtrain_csr_i, Xtest_csr_p, Xtest_csr_i, Xtest_csr, \
                   min_pos_test, min_items_pool, seed, \
                   pred_buffer, ind_buffer, test_bool_buffer)
    for (int32_t user = 0; user < m; user++)
    {
        if (
                Xtest_csr_p[user] >= Xtest_csr_p[user+1] ||
                ((
                    (Xtrain_csr_p[user+1] - Xtrain_csr_p[user]) +
                    (Xtest_csr_p[user+1] - Xtest_csr_p[user])
                ) >= n && !ndcg_at_k) || /* <- NDCG can be calculated with only-positive entries */
                (n - (Xtrain_csr_p[user+1] - Xtrain_csr_p[user])) < min_items_pool ||
                (!consider_cold_start && Xtrain_csr_p[user] == Xtrain_csr_p[user+1]) ||
                (Xtest_csr_p[user+1] - Xtest_csr_p[user] < min_pos_test)
            )
        {
            set_as_NAN:
            if (!cumulative) {
                if (p_at_k) p_at_k[user] = NAN_;
                if (tp_at_k) tp_at_k[user] = NAN_;
                if (r_at_k) r_at_k[user] = NAN_;
                if (ap_at_k) ap_at_k[user] = NAN_;
                if (tap_at_k) tap_at_k[user] = NAN_;
                if (ndcg_at_k) ndcg_at_k[user] = NAN_;
                if (hit_at_k) hit_at_k[user] = NAN_;
                if (rr_at_k) rr_at_k[user] = NAN_;
            }

            else {
                size_t st = (size_t)user*(size_t)k_metrics;
                size_t end = (size_t)(user+1)*(size_t)k_metrics;
                if (p_at_k) std::fill(p_at_k + st, p_at_k + end, NAN_);
                if (tp_at_k) std::fill(tp_at_k + st, tp_at_k + end, NAN_);
                if (r_at_k) std::fill(r_at_k + st, r_at_k + end, NAN_);
                if (ap_at_k) std::fill(ap_at_k + st, ap_at_k + end, NAN_);
                if (tap_at_k) std::fill(tap_at_k + st, tap_at_k + end, NAN_);
                if (ndcg_at_k) std::fill(ndcg_at_k + st, ndcg_at_k + end, NAN_);
                if (hit_at_k) std::fill(hit_at_k + st, hit_at_k + end, NAN_);
                if (rr_at_k) std::fill(rr_at_k + st, rr_at_k + end, NAN_);
            }
            if (roc_auc) roc_auc[user] = NAN_;
            if (pr_auc) pr_auc[user] = NAN_;
            continue;
        }

        const bool only_ndcg = (
                                (Xtrain_csr_p[user+1] - Xtrain_csr_p[user]) +
                                (Xtest_csr_p[user+1] - Xtest_csr_p[user])
                            ) >= n;
        const bool k_leq_n = (n - (Xtrain_csr_p[user+1] - Xtrain_csr_p[user])) <= k_metrics;

        if (k_leq_n && !roc_auc && !pr_auc && !ap_at_k && !tap_at_k && !rr_at_k)
            goto set_as_NAN;

        if (interrupt_switch)
            continue;

        ind_thread = ind_buffer.get() + omp_get_thread_num()*n;
        std::iota(ind_thread, ind_thread + n, (int32_t)0);
        int32_t move_to = n;
        for (int32_t ix = Xtrain_csr_p[user+1]-1; ix >= Xtrain_csr_p[user]; ix--)
            std::swap(ind_thread[Xtrain_csr_i[ix]], ind_thread[--move_to]);
        if (move_to < n)
            std::sort(ind_thread, ind_thread + move_to);
        
        A_thread = A + (size_t)user*lda;
        pred_thread = pred_buffer.get() + omp_get_thread_num()*n;
        if ((uint64_t)n*(uint64_t)ldb < INT32_MAX)
        {
            int32_t ldb_int = ldb;
            for (int32_t ix = 0; ix < move_to; ix++)
                pred_thread[ind_thread[ix]] = dot1(A_thread, B + ind_thread[ix]*ldb_int, k);
        }
        
        else
        {
            for (int32_t ix = 0; ix < move_to; ix++)
                pred_thread[ind_thread[ix]] = dot1(A_thread, B + (size_t)ind_thread[ix]*ldb, k);
        }

        if (break_ties_with_noise)
        {
            /* Before adding noise and sorting, determine whether the entries are valid */
            for (int32_t ix = 0; ix < move_to; ix++)
                if (std::isnan(pred_thread[ind_thread[ix]])) goto set_as_NAN;
            real_t pred_min = std::numeric_limits<real_t>::max();
            real_t pred_max = std::numeric_limits<real_t>::lowest();
            for (int32_t ix = 0; ix < move_to; ix++) {
                pred_max = (pred_max < pred_thread[ind_thread[ix]])? pred_thread[ind_thread[ix]] : pred_max;
                pred_min = (pred_min > pred_thread[ind_thread[ix]])? pred_thread[ind_thread[ix]] : pred_min;
            }
            if (pred_max == pred_min) goto set_as_NAN;
            if (std::isinf(pred_max) || std::isinf(pred_min)) goto set_as_NAN;

            std::mt19937 rng_user(seed + (uint64_t)user);
            std::uniform_real_distribution<real_t> runif((real_t)(-1e-12), (real_t)1e-12);
            for (int32_t ix = 0; ix < move_to; ix++)
                pred_thread[ind_thread[ix]] += runif(rng_user);
        }

        if ((!roc_auc || only_ndcg) && k_metrics < move_to) {
            std::partial_sort(ind_thread, ind_thread + k_metrics, ind_thread + move_to,
                              [&pred_thread](const int32_t i, const int32_t j)
                              {return pred_thread[i] > pred_thread[j];});
            if (!break_ties_with_noise)
            {
                real_t pred_max = pred_thread[ind_thread[0]];
                real_t pred_min = pred_thread[ind_thread[k_metrics-1]];
                if (std::isnan(pred_max) || std::isnan(pred_min)) goto set_as_NAN;
                if (std::isinf(pred_max) || std::isinf(pred_min)) goto set_as_NAN;
                if (pred_max == pred_min) goto set_as_NAN;
            }
        }

        else {
            std::sort(ind_thread, ind_thread + move_to,
                      [&pred_thread](const int32_t i, const int32_t j)
                      {return pred_thread[i] > pred_thread[j];});
            if (!break_ties_with_noise)
            {
                real_t pred_max = pred_thread[ind_thread[0]];
                real_t pred_min = pred_thread[ind_thread[move_to-1]];
                if (std::isnan(pred_max) || std::isnan(pred_min)) goto set_as_NAN;
                if (std::isinf(pred_max) || std::isinf(pred_min)) goto set_as_NAN;
                if (pred_max == pred_min) goto set_as_NAN;
            }
        }

        bool has_bool_mask = false;
        if (!only_ndcg && (roc_auc || pr_auc))
        {
            has_bool_mask = true;
            test_bool_thread = test_bool_buffer.get() + (size_t)n*(size_t)omp_get_thread_num();
            std::fill(test_bool_thread, test_bool_thread + n, false);
            for (int32_t ix = Xtest_csr_p[user]; ix < Xtest_csr_p[user+1]; ix++)
                test_bool_thread[Xtest_csr_i[ix]] = true;
        }

        if (cumulative)
        {
            size_t st = (size_t)user*(size_t)k_metrics;
            p_at_k_user = p_at_k + st;
            tp_at_k_user = tp_at_k + st;
            r_at_k_user = r_at_k + st;
            ap_at_k_user = ap_at_k + st;
            tap_at_k_user = tap_at_k + st;
            ndcg_at_k_user = ndcg_at_k + st;
            hit_at_k_user = hit_at_k + st;
            rr_at_k_user = rr_at_k + st;
        }


        int32_t *restrict user_istart = Xtest_csr_i + Xtest_csr_p[user];
        int32_t *restrict user_iend = Xtest_csr_i + Xtest_csr_p[user+1];
        const real_t *restrict user_vstart = Xtest_csr? (Xtest_csr + Xtest_csr_p[user]) : nullptr;
        const int32_t user_st = *user_istart;
        const int32_t user_end = Xtest_csr_i[Xtest_csr_p[user+1]-1];
        uint64_t npos = user_iend - user_istart;
        uint64_t nneg = move_to - npos;

        bool did_short_loop = false;

        int32_t hits = 0;
        double avg_p = 0; /* <- TODO: maybe use long double */
        double dcg = 0;
        int32_t min_rank = std::numeric_limits<int32_t>::max();
        int32_t *res;

        if (calc_top_metrics && (!k_leq_n || ap_at_k || tap_at_k || rr_at_k || ndcg_at_k))
        {
            did_short_loop = true;

            if (ndcg_at_k || !has_bool_mask)
            {
                for (int32_t ix = 0; ix < std::min(k_metrics, move_to); ix++)
                {
                    if (ind_thread[ix] >= user_st && ind_thread[ix] <= user_end)
                    {
                        res = std::lower_bound(user_istart, user_iend, ind_thread[ix]);
                        if ((res != user_iend) && (*res == ind_thread[ix]))
                        {
                            hits++;
                            avg_p += hits / (double)(ix+1);
                            dcg += user_vstart? ((double)user_vstart[res - user_istart] / std::log2(ix+2)) : 0.;
                            min_rank = std::min(min_rank, ix);
                        }
                    }

                    if (cumulative)
                    {
                        if (p_at_k) p_at_k_user[ix] = hits / (double)(ix+1);
                        if (tp_at_k) tp_at_k_user[ix] = hits / (double)std::min(ix+1, (int32_t)npos);
                        if (r_at_k) r_at_k_user[ix] = hits / (double)npos;
                        if (ap_at_k) ap_at_k_user[ix] = avg_p / (double)npos;
                        if (tap_at_k) tap_at_k_user[ix] = avg_p / (double)std::min(ix+1, (int32_t)npos);
                        if (ndcg_at_k) ndcg_at_k_user[ix] = dcg;
                        if (hit_at_k) hit_at_k_user[ix] = hits > 0;
                        if (rr_at_k) rr_at_k_user[ix] = hits? ((double)1 / (double)(min_rank+1)) : 0.;
                    }

                    if (!cumulative && hits >= move_to)
                        break;
                    if (only_first_pos && hits)
                    {
                        if (cumulative && ix != std::min(k_metrics, move_to) - 1)
                        {
                            if (hit_at_k)
                                std::fill(hit_at_k_user + ix + 1,
                                          hit_at_k_user + k_metrics,
                                          hit_at_k_user[ix]);
                            if (rr_at_k)
                                std::fill(rr_at_k_user + ix + 1,
                                          rr_at_k_user + k_metrics,
                                          rr_at_k_user[ix]);
                        }
                        break;
                    }
                }
            }

            else
            {
                for (int32_t ix = 0; ix < std::min(k_metrics, move_to); ix++)
                {
                    if (test_bool_thread[ind_thread[ix]])
                    {
                        hits++;
                        avg_p += hits / (double)(ix+1);
                        min_rank = std::min(min_rank, ix);
                    }

                    if (cumulative)
                    {
                        if (p_at_k) p_at_k_user[ix] = hits / (double)(ix+1);
                        if (tp_at_k) tp_at_k_user[ix] = hits / (double)std::min(ix+1, (int32_t)npos);
                        if (r_at_k) r_at_k_user[ix] = hits / (double)npos;
                        if (ap_at_k) ap_at_k_user[ix] = avg_p / (double)npos;
                        if (tap_at_k) tap_at_k_user[ix] = avg_p / (double)std::min(ix+1, (int32_t)npos);
                        if (hit_at_k) hit_at_k_user[ix] = hits > 0;
                        if (rr_at_k) rr_at_k_user[ix] = hits? ((double)1 / (double)(min_rank+1)) : 0.;
                    }

                    if (!cumulative && hits >= move_to)
                        break;
                    if (only_first_pos && hits)
                    {
                        if (cumulative && ix != std::min(k_metrics, move_to) - 1)
                        {
                            if (hit_at_k)
                                std::fill(hit_at_k_user + ix + 1,
                                          hit_at_k_user + k_metrics,
                                          hit_at_k_user[ix]);
                            if (rr_at_k)
                                std::fill(rr_at_k_user + ix + 1,
                                          rr_at_k_user + k_metrics,
                                          rr_at_k_user[ix]);
                        }
                        break;
                    }
                }
            }

            if (!cumulative)
            {
                if (p_at_k) p_at_k[user] = (double)hits / (double)k_metrics;
                if (tp_at_k) tp_at_k[user] = (double)hits / (double)std::min(k_metrics, (int32_t)npos);
                if (r_at_k) r_at_k[user] = (double)hits / (double)npos;
                if (ap_at_k) ap_at_k[user] = avg_p / (double)npos;
                if (tap_at_k) tap_at_k[user] = avg_p / (double)std::min(k_metrics, (int32_t)npos);
                if (hit_at_k) hit_at_k[user] = hits > 0;
                if (rr_at_k) rr_at_k[user] = hits? (1. / (double)(min_rank+1)) : 0.;
            }

            else
            {
                if (k_metrics > move_to)
                {
                    if (p_at_k) std::fill(p_at_k_user + move_to, p_at_k_user + k_metrics, NAN_);
                    if (tp_at_k) std::fill(tp_at_k_user + move_to, tp_at_k_user + k_metrics, NAN_);
                    if (r_at_k) std::fill(r_at_k_user + move_to, r_at_k_user + k_metrics, NAN_);
                    if (hit_at_k) std::fill(hit_at_k_user + move_to, hit_at_k_user + k_metrics, NAN_);

                    if (ap_at_k)
                    {
                        std::fill(ap_at_k_user + move_to,
                                  ap_at_k_user + k_metrics,
                                  ap_at_k_user[move_to-1]);
                    }

                    if (tap_at_k)
                    {
                        std::fill(tap_at_k_user + move_to,
                                  tap_at_k_user + k_metrics,
                                  tap_at_k_user[move_to-1]);
                    }

                    if (rr_at_k)
                    {
                        std::fill(rr_at_k_user + move_to,
                                  rr_at_k_user + k_metrics,
                                  rr_at_k_user[move_to-1]);
                    }

                    if (ndcg_at_k)
                    {
                        std::fill(ndcg_at_k_user + move_to,
                                  ndcg_at_k_user + k_metrics,
                                  ndcg_at_k_user[move_to-1]);
                    }
                }
            }
        }

        if (k_leq_n)
        {
            if (!cumulative) {
                if (p_at_k) p_at_k[user] = NAN_;
                if (tp_at_k) tp_at_k[user] = NAN_;
                if (r_at_k) r_at_k[user] = NAN_;
                if (hit_at_k) hit_at_k[user] = NAN_;
            }

            else if (!did_short_loop) {
                if (p_at_k) std::fill_n(p_at_k_user, k_metrics, NAN_);
                if (tp_at_k) std::fill_n(tp_at_k_user, k_metrics, NAN_);
                if (r_at_k) std::fill_n(r_at_k_user, k_metrics, NAN_);
                if (hit_at_k) std::fill_n(hit_at_k_user, k_metrics, NAN_);
            }
        }

        else if (only_ndcg)
        {
            if (!cumulative) {
                if (p_at_k) p_at_k[user] = NAN_;
                if (tp_at_k) tp_at_k[user] = NAN_;
                if (r_at_k) r_at_k[user] = NAN_;
                if (ap_at_k) ap_at_k[user] = NAN_;
                if (tap_at_k) tap_at_k[user] = NAN_;
                if (hit_at_k) hit_at_k[user] = NAN_;
                if (rr_at_k) rr_at_k[user] = NAN_;
            }

            else {
                if (p_at_k) std::fill_n(p_at_k_user, k_metrics, NAN_);
                if (tp_at_k) std::fill_n(tp_at_k_user, k_metrics, NAN_);
                if (r_at_k) std::fill_n(r_at_k_user, k_metrics, NAN_);
                if (ap_at_k) std::fill_n(ap_at_k_user, k_metrics, NAN_);
                if (tap_at_k) std::fill_n(tap_at_k_user, k_metrics, NAN_);
                if (hit_at_k) std::fill_n(hit_at_k_user, k_metrics, NAN_);
                if (rr_at_k) std::fill_n(rr_at_k_user, k_metrics, NAN_);
            }
        }


        /* Note: this will never overflow when using int32 and storing in uint64, but overflow
           could happen with wider types and getting this metric correct then would imply switching
           to a different method, such as the trapezoidal approximation. That's why these types are
           hard-coded instead of accepting generic 'int'. */
        hits = 0;
        avg_p = 0;
        uint64_t sum_ranks_pos = 0;

        #if defined(__GNUC__) && (__GNUC__ >= 5)
        #   pragma GCC diagnostic push
        #   pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
        #endif
        if (only_ndcg)
        {
            if (roc_auc) roc_auc[user] = NAN_;
            if (pr_auc) pr_auc[user] = NAN_;
        }

        else if (roc_auc && pr_auc)
        {
            for (int32_t ix = 0; ix < move_to; ix++)
            {
                if (test_bool_thread[ind_thread[ix]])
                {
                    sum_ranks_pos += (uint64_t)(ix+1);
                    hits++;
                    avg_p += (double)hits / (double)(ix+1);
                    if (hits == (int32_t)npos) break;
                }
            }
            roc_auc[user] = 1. - (long double)(sum_ranks_pos - (npos * (npos + 1)) / 2)
                                 / (long double)(npos * nneg);
            pr_auc[user] = avg_p / (double)npos;
        }

        else if (roc_auc)
        {
            for (int32_t ix = 0; ix < move_to; ix++)
            {
                if (test_bool_thread[ind_thread[ix]])
                {
                    sum_ranks_pos += (uint64_t)(ix+1);
                    hits++;
                    if (hits == (int32_t)npos) break;
                }
            }
            roc_auc[user] = 1. - (long double)(sum_ranks_pos - (npos * (npos + 1)) / 2)
                                 / (long double)(npos * nneg);
        }

        else if (pr_auc)
        {
            /* TODO: can save some computations as the top-K are already done */
            // if (calc_top_metrics)
            // {

            // }

            // else
            {
                for (int32_t ix = 0; ix < move_to; ix++)
                {
                    if (test_bool_thread[ind_thread[ix]])
                    {
                        hits++;
                        avg_p += (double)hits / (double)(ix+1);
                        if (hits == (int32_t)npos) break;
                    }
                }
                roc_auc[user] = 1. - (long double)(sum_ranks_pos - (npos * (npos + 1)) / 2)
                                     / (long double)(npos * nneg);
                pr_auc[user] = avg_p / (double)npos;
            }
        }
        #if defined(__GNUC__) && (__GNUC__ >= 5)
        #   pragma GCC diagnostic pop
        #endif


        if (ndcg_at_k)
        {
            std::iota(ind_thread, ind_thread + npos, 0);
            std::partial_sort(ind_thread, ind_thread + std::min(k_metrics, (int32_t)npos), ind_thread + npos,
                              [&user_vstart](const int32_t i, const int32_t j)
                              {return user_vstart[i] > user_vstart[j];});

            real_t vmax = user_vstart[ind_thread[0]];
            real_t vmin = user_vstart[ind_thread[std::min(k_metrics, (int32_t)npos) - 1]];
            if (std::isnan(vmax) || std::isinf(vmax) ||
                std::isnan(vmin) || std::isinf(vmin) ||
                vmax <= 0)
            {
                if (!cumulative)
                    ndcg_at_k[user] = NAN_;
                else
                    std::fill_n(ndcg_at_k_user, k_metrics, NAN_);
                continue;
            }


            double val;
            double idcg = 0;
            real_t last_val = user_vstart[ind_thread[std::min(k_metrics, (int32_t)npos) - 1]];
            if (!cumulative)
            {
                if (std::isnan(last_val) || std::isinf(last_val)) {
                    ndcg_at_k[user] = NAN_;
                    continue;
                }

                else if (last_val >= 0) {
                    for (int32_t ix = 0; ix < std::min(k_metrics, (int32_t)npos); ix++)
                        idcg += (double)user_vstart[ind_thread[ix]] / std::log2(ix+2);
                    ndcg_at_k[user] = dcg / idcg;
                }

                else {
                    for (int32_t ix = 0; ix < std::min(k_metrics, (int32_t)npos); ix++) {
                        val = user_vstart[ind_thread[ix]];
                        if (val <= 0) break;
                        idcg += val / std::log2(ix+2);
                    }
                    ndcg_at_k[user] = dcg / idcg;
                }
            }

            else
            {
                #if defined(__GNUC__) && (__GNUC__ >= 5)
                #   pragma GCC diagnostic push
                #   pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
                #endif
                if (!std::isnan(last_val) && last_val >= 0)
                {
                    for (int32_t ix = 0; ix < std::min(k_metrics, (int32_t)npos); ix++)
                    {
                        idcg += (double)user_vstart[ind_thread[ix]] / std::log2(ix+2);
                        ndcg_at_k_user[ix] /= idcg;
                    }
                }

                else
                {
                    int32_t lim = std::min(k_metrics, (int32_t)npos);
                    int32_t ix;
                    for (ix = 0; ix < lim; ix++)
                    {
                        val = user_vstart[ind_thread[ix]];
                        if (std::isnan(val) || val < 0) break;
                        idcg += val / std::log2(ix+2);
                        ndcg_at_k_user[ix] /= idcg;
                    }
                    if (std::isnan(val))
                        std::fill(ndcg_at_k_user + ix,
                                  ndcg_at_k_user + lim,
                                  NAN_);
                    else if (val < 0)
                        for (; ix < lim; ix++)
                            ndcg_at_k_user[ix] /= idcg;
                }

                if (npos < (uint64_t)k_metrics)
                {
                    std::fill(ndcg_at_k_user + npos,
                              ndcg_at_k_user + std::min(k_metrics, move_to),
                              ndcg_at_k_user[npos-1]);
                }
                #if defined(__GNUC__) && (__GNUC__ >= 5)
                #   pragma GCC diagnostic pop
                #endif
            }
        }
    }

    ss.check_interrupt_switch();
}

/*  Create a per-user train-test split of data, for all users in the data
    
    Parameters
    ==========
    - X_csr_p[m+1]
        Matrix of user-item interactions, in CSR format. This one denotes
        the index pointer. See the documentation of 'calc_metrics' for more
        details on the format.
    - X_csr_i[nnz]
        Matrix of user-item interactions, in CSR format. This one denotes
        the non-zero indices.
    - X_csr[nnz]
        Matrix of user-item interactions, in CSR format. This one denotes
        the values of the non-zero entries.
    - m
        Numer of users (rows in 'X').
    - n
        Number of items (columns in 'X').
    - Xtrain_csr_p (out, dim:m+1)
        Matrix of user-item interactions that is set as training data,
        in CSR format. This one denotes the index pointer.
        The contents of the object passed here will be overwritten and the
        object will be resized as needed.
    - Xtrain_csr_i (out, dim:nnz_train)
        Matrix of user-item interactions that is set as training data,
        in CSR format. This one denotes the non-zero indices.
    - Xtrain_csr (out, dim:nnz_train)
        Matrix of user-item interactions that is set as training data,
        in CSR format. This one denotes the non-zero values.
    - Xtest_csr_p (out, dim:m+1)
        Matrix of user-item interactions that is set as testing data,
        in CSR format. This one denotes the index pointer.
    - Xtest_csr_i (out, dim:nnz_test)
        Matrix of user-item interactions that is set as testing data,
        in CSR format. This one denotes the non-zero indices.
    - Xtest_csr (out, dim:nnz_test)
        Matrix of user-item interactions that is set as testing data,
        in CSR format. This one denotes the non-zero values.
    - test_fraction
        Target fraction of the data to set for test. Should be a number between
        zero and one (non-inclusive). Note that the actual number of test entries
        for each user will be determined as 'round(n_entries*test_fraction)', thus
        in a long-tailed distribution (typical for recommender systems), the actual
        fraction that will be obtained is likely to be slightly lower than what
        is passed here.
    - seed
        Seed to use for random number generation.
 */
template <class real_t>
void split_data_selected_users
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const real_t *restrict X_csr,
    const int32_t m, const int32_t n,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<real_t> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<real_t> &Xtest_csr,
    const double test_fraction,
    uint64_t seed
)
{
    if (!m)
        return;
    if (m < 0 || n < 0)
        throw std::runtime_error("Passed negative dimensions.\n");

    Xtrain_csr_p.resize(m+1);
    Xtest_csr_p.resize(m+1);
    Xtrain_csr_p[0] = 0;
    Xtest_csr_p[0] = 0;

    for (int32_t user = 0; user < m; user++)
        Xtest_csr_p[user+1] = Xtest_csr_p[user] + (int32_t)std::round((X_csr_p[user+1] - X_csr_p[user]) * test_fraction);
    for (int32_t user = 0; user < m; user++)
        Xtrain_csr_p[user+1] = X_csr_p[user+1] - Xtest_csr_p[user+1];

    int32_t nnz_test = Xtest_csr_p[m];
    int32_t nnz_train = X_csr_p[m] - nnz_test;
    Xtrain_csr_i.resize(nnz_train);
    Xtrain_csr.resize(nnz_train);
    Xtest_csr_i.resize(nnz_test);
    Xtest_csr.resize(nnz_test);

    std::mt19937 rng(seed);
    std::unique_ptr<int32_t[]> indices(new int32_t[n]);
    std::unique_ptr<int32_t[]> argsorted(new int32_t[n]);
    int32_t *restrict user_i;
    real_t *restrict user_v;
    const int32_t *restrict orig_i;
    const real_t *restrict orig_v;
    int32_t nnz_this;
    int32_t target_nnz;
    for (int32_t user = 0; user < m; user++)
    {
        nnz_this = X_csr_p[user+1] - X_csr_p[user];
        if (!nnz_this) continue;
        target_nnz = Xtest_csr_p[user+1] - Xtest_csr_p[user];

        if (!target_nnz)
        {
            std::copy(X_csr_i + X_csr_p[user], X_csr_i + X_csr_p[user+1], Xtrain_csr_i.data() + Xtrain_csr_p[user]);
            std::copy(X_csr + X_csr_p[user], X_csr + X_csr_p[user+1], Xtrain_csr.data() + Xtrain_csr_p[user]);
            continue;
        }

        else if (!(nnz_this - target_nnz))
        {
            std::copy(X_csr_i + X_csr_p[user], X_csr_i + X_csr_p[user+1], Xtest_csr_i.data() + Xtest_csr_p[user]);
            std::copy(X_csr + X_csr_p[user], X_csr + X_csr_p[user+1], Xtest_csr.data() + Xtest_csr_p[user]);
            continue;
        }

        std::iota(indices.get(), indices.get() + nnz_this, (int32_t)0);
        std::shuffle(indices.get(), indices.get() + nnz_this, rng);

        orig_i = X_csr_i + X_csr_p[user];
        orig_v = X_csr + X_csr_p[user];
        std::sort(indices.get(), indices.get() + target_nnz,
                  [&orig_i](const int32_t a, const int32_t b)
                  {return orig_i[a] < orig_i[b];});

        user_i = Xtest_csr_i.data() + Xtest_csr_p[user];
        user_v = Xtest_csr.data() + Xtest_csr_p[user];
        for (int32_t ix = 0; ix < target_nnz; ix++) {
            user_i[ix] = orig_i[indices[ix]];
            user_v[ix] = orig_v[indices[ix]];
        }

        std::sort(indices.get() + target_nnz, indices.get() + nnz_this,
                  [&orig_i](const int32_t a, const int32_t b)
                  {return orig_i[a] < orig_i[b];});
        user_i = Xtrain_csr_i.data() + Xtrain_csr_p[user] - target_nnz;
        user_v = Xtrain_csr.data() + Xtrain_csr_p[user] - target_nnz;
        for (int32_t ix = target_nnz; ix < nnz_this; ix++) {
            user_i[ix] = orig_i[indices[ix]];
            user_v[ix] = orig_v[indices[ix]];
        }
    }
}

/*  Select test users from implicit-feedback data and create train-test split for them

    This function will first draw a random sample of users who meet some desired minimum
    criteria to set as testing users, and will then create a train-test split for each
    of these selected users, outputting three matrices:
        - Training data for test users.
        - Test data for test users.
        - Remaining of the data (training data for non-test users).

    This is intended for situations in which the data is going to be fitted to the
    'remaining' data, then factors from the already-fitted model determined for the
    test users given their trainin data, and evaluated in the test data. For cases
    in which there will not be such separation between old/new user factors, one
    can use instead the function 'split_data_joined_users'.

    Parameters
    ==========
    - X_csr_p[m+1]
        Matrix of user-item interactions, in CSR format. This one denotes
        the index pointer. See the documentation of 'calc_metrics' for more
        details on the format.
    - X_csr_i[nnz]
        Matrix of user-item interactions, in CSR format. This one denotes
        the non-zero indices.
    - X_csr[nnz]
        Matrix of user-item interactions, in CSR format. This one denotes
        the values of the non-zero entries.
    - m
        Numer of users (rows in 'X').
    - n
        Number of items (columns in 'X').
    - users_test (out, dim:<=n_users_test)
        IDs of the users that are selected as test. The numeration will start
        at zero and will be in sorted order (ascending). The remaining data
        (users that are not selected as test) will lack these users, while
        remaining in the same order as it was before with these users excluded
        (e.g. if test users are [0,3], the remainder will have users [1,2,4],
        in that order).
        The contents of this object will be cleared and its size will be adjusted
        accordingly.
    - Xrem_csr_p (out, dim:>=m+1-n_users_test)
        Data for users which are not selected for test, in CSR format.
        This one denotes the index pointer.
    - Xrem_csr_i (out, dim:nnz_rem)
        Data for users which are not selected for test, in CSR format.
        This one denotes the indices of the non-zero entries.
    - Xrem_csr (out, dim:nnz_rem)
        Data for users which are not selected for test, in CSR format.
        This one denotes the values associated to 'Xrem_csr_i'.
    - Xtrain_csr_p (out, dim:<=n_users_test+1)
        Training data for users which are selected for test, in CSR format.
        This one denotes the index pointer.
    - Xtrain_csr_i (out, dim:nnz_train)
        Training data for users which are selected for test, in CSR format.
        This one denotes the indices of the non-zero entries.
    - Xtrain_csr (out, dim:nnz_train)
        Training data for users which are selected for test, in CSR format.
        This one denotes the values associated to 'Xtrain_csr_i'.
    - Xtest_csr_p (out, dim:<=n_users_test+1)
        Testing data for users which are selected for test, in CSR format.
        This one denotes the index pointer.
    - Xtest_csr_i (out, dim:nnz_test)
        Testing data for users which are selected for test, in CSR format.
        This one denotes the indices of the non-zero entries.
    - Xtest_csr (out, dim:nnz_test)
        Testing data for users which are selected for test, in CSR format.
        This one denotes the values associated to 'Xtest_csr_i'.
    - n_users_test
        Number of users to set as test. If there are not enough users meeting the minimum
        desired criteria, the actual number of test users will be smaller.
    - test_fraction
        Target fraction of the data to set for test for each user. Should be a number
        between zero and one (non-inclusive). Note that the actual number of test entries
        for each user will be determined as 'round(n_entries*test_fraction)', thus
        in a long-tailed distribution (typical for recommender systems), the actual
        fraction that will be obtained is likely to be slightly lower than what
        is passed here.
    - consider_cold_start
        Whether to still set users as eligible for test in situations in which some user would
        have test data but no positive (non-zero) entries in the training data.
        This will only happen when passing 'test_fraction>=0.5'.
    - min_items_pool
        Minimum number of items (sum of positive and negative items) that a user must have in order to
        be eligible as test user.
    - min_pos_test
        Minimum number of positive entries (non-zero entries in the test set) that users would need
        to have in order to be eligible as test user.
    - seed
        Seed to use for random number generation.
*/
template <class real_t>
void split_data_separate_users
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const real_t *restrict X_csr,
    int32_t m, int32_t n,
    std::vector<int32_t> &users_test,
    std::vector<int32_t> &Xrem_csr_p,
    std::vector<int32_t> &Xrem_csr_i,
    std::vector<real_t> &Xrem_csr,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<real_t> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<real_t> &Xtest_csr,
    const int32_t n_users_test,
    const double test_fraction,
    const bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    uint64_t seed
)
{
    if (n_users_test > m)
        throw std::runtime_error("Target number of test users is larger than available users.\n");
    if (min_items_pool >= n)
        throw std::runtime_error("Selected minimum number of items is larger than total number of items.\n");

    std::mt19937 rng(seed);
    std::unique_ptr<int32_t[]> user_ids(new int32_t[m]);
    std::iota(user_ids.get(), user_ids.get() + m, (int32_t)0);
    std::shuffle(user_ids.get(), user_ids.get() + m, rng);

    int32_t nnz_this;
    int32_t nnz_test;
    int32_t end = m;
    int32_t n_taken = 0;
    do
    {
        nnz_this = X_csr_p[user_ids[n_taken]+1] - X_csr_p[user_ids[n_taken]];
        if (!nnz_this)
        {
            skip_user:
            std::swap(user_ids[n_taken], user_ids[--end]);
            continue;
        }

        nnz_test = std::round(nnz_this * test_fraction);
        if (nnz_test < min_pos_test)
            goto skip_user;

        if (n - (nnz_this - nnz_test) < min_items_pool)
            goto skip_user;

        if (!consider_cold_start && nnz_test == nnz_this)
            goto skip_user;

        if (nnz_this + 1 >= n)
            goto skip_user;

        n_taken++;
    }

    while (n_taken < n_users_test && n_taken < end);
    if (!n_taken)
        throw std::runtime_error("No users satisfy criteria for test inclusion.\n");

    
    std::sort(user_ids.get(), user_ids.get() + n_taken);
    users_test.assign(user_ids.get(), user_ids.get() + n_taken);
    std::sort(user_ids.get() + n_taken, user_ids.get() + m);

    std::unique_ptr<int32_t[]> X_sel_p(new int32_t[n_taken+1]);
    X_sel_p[0] = 0;

    for (int32_t user = 0; user < n_taken; user++)
        X_sel_p[user+1] = X_sel_p[user] + (X_csr_p[user_ids[user]+1] - X_csr_p[user_ids[user]]);
    std::unique_ptr<int32_t[]> X_sel_i(new int32_t[X_sel_p[n_taken]]);
    std::unique_ptr<real_t[]> X_sel(new real_t[X_sel_p[n_taken]]);

    for (int32_t user = 0; user < n_taken; user++)
        std::copy(X_csr_i + X_csr_p[user_ids[user]],
                  X_csr_i + X_csr_p[user_ids[user]+1],
                  X_sel_i.get() + X_sel_p[user]);
    for (int32_t user = 0; user < n_taken; user++)
        std::copy(X_csr + X_csr_p[user_ids[user]],
                  X_csr + X_csr_p[user_ids[user]+1],
                  X_sel.get() + X_sel_p[user]);

    split_data_selected_users(
        X_sel_p.get(),
        X_sel_i.get(),
        X_sel.get(),
        n_taken, n,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        test_fraction,
        seed
    );

    Xrem_csr_p.resize(m-n_taken+1);
    Xrem_csr_p[0] = 0;
    for (int32_t user = n_taken; user < m; user++)
        Xrem_csr_p[user-n_taken+1] = Xrem_csr_p[user-n_taken] + (X_csr_p[user_ids[user]+1] - X_csr_p[user_ids[user]]);
    Xrem_csr_i.resize(Xrem_csr_p.back());
    Xrem_csr.resize(Xrem_csr_p.back());

    int32_t *restrict user_ids_ = user_ids.get() + n_taken;
    for (int32_t user = 0; user < m-n_taken; user++)
        std::copy(X_csr_i + X_csr_p[user_ids_[user]],
                  X_csr_i + X_csr_p[user_ids_[user]+1],
                  Xrem_csr_i.data() + Xrem_csr_p[user]);
    for (int32_t user = 0; user < m-n_taken; user++)
        std::copy(X_csr + X_csr_p[user_ids_[user]],
                  X_csr + X_csr_p[user_ids_[user]+1],
                  Xrem_csr.data() + Xrem_csr_p[user]);
}

template <class real_t>
void concat_csr_matrices
(
    std::vector<int32_t> &X1_csr_p,
    std::vector<int32_t> &X1_csr_i,
    std::vector<real_t> &X1_csr,
    std::vector<int32_t> &X2_csr_p,
    std::vector<int32_t> &X2_csr_i,
    std::vector<real_t> &X2_csr,
    std::vector<int32_t> &Xout_csr_p,
    std::vector<int32_t> &Xout_csr_i,
    std::vector<real_t> &Xout_csr
)
{
    Xout_csr_p.resize(X1_csr_p.size() + X2_csr_p.size() - 1);
    std::copy(X1_csr_p.begin(), X1_csr_p.end(), Xout_csr_p.begin());
    int32_t st_second = X1_csr_p.size();
    int32_t offset = X1_csr_p.back();
    X1_csr_p.clear(); X1_csr_p.shrink_to_fit();
    for (int32_t row = 1; row < (int32_t)X2_csr_p.size(); row++)
        Xout_csr_p[st_second + row - 1] = offset + X2_csr_p[row];
    X2_csr_p.clear(); X2_csr_p.shrink_to_fit();

    Xout_csr_i.resize(Xout_csr_p.back());
    Xout_csr.resize(Xout_csr_p.back());

    std::copy(X1_csr_i.begin(), X1_csr_i.end(), Xout_csr_i.begin());
    X1_csr_i.clear(); X1_csr_i.shrink_to_fit();
    std::copy(X2_csr_i.begin(), X2_csr_i.end(), Xout_csr_i.begin() + offset);
    X2_csr_i.clear(); X2_csr_i.shrink_to_fit();

    std::copy(X1_csr.begin(), X1_csr.end(), Xout_csr.begin());
    X1_csr.clear(); X1_csr.shrink_to_fit();
    std::copy(X2_csr.begin(), X2_csr.end(), Xout_csr.begin() + offset);
    X2_csr.clear(); X2_csr.shrink_to_fit();
}

/*  Select test users from implicit-feedback data and create train-test split for them
    
    Similar to 'split_data_separate_users', but this one will instead create a split
    with only two sets: train-test, with the train set containing as the first rows
    the data for the users that were selected for test, and the test data containing
    only the rows corresponding to test users.

    This is intended for situations in which the factors will not be determined
    separately for the test users after the model is already fit, but will rather
    be obtained from the fitted model matrices directly.

    See the documentation for 'split_data_separate_users' for more details.

    Parameters
    ==========
    - X_csr_p[m+1]
        Matrix of user-item interactions, in CSR format. This one denotes
        the index pointer. See the documentation of 'calc_metrics' for more
        details on the format.
    - X_csr_i[nnz]
        Matrix of user-item interactions, in CSR format. This one denotes
        the non-zero indices.
    - X_csr[nnz]
        Matrix of user-item interactions, in CSR format. This one denotes
        the values of the non-zero entries.
    - m
        Numer of users (rows in 'X').
    - n
        Number of items (columns in 'X').
    - users_test
        IDs of the users that are selected as test. The numeration will start
        at zero and will be in sorted order (ascending). The trainin data will
        still have these users as the first rows, while
        remaining in the same order as it was before with these users switched to the top
        (e.g. if test users are [0,3], the training data will have users [0,3,1,2,4],
        in that order).
        The contents of this object will be cleared and its size will be adjusted
        accordingly.
    - Xtrain_csr_p (out, dim:m+1)
        Data for users which are not selected for test, followed by the remainder, in CSR format.
        This one denotes the index pointer.
    - Xtrain_csr_i (out, dim:nnz-nnz_test)
        Data for users which are not selected for test, followed by the remainder, in CSR format.
        This one denotes the indices of the non-zero entries.
    - Xtrain_csr (out, dim:nnz-nnz_test)
        Data for users which are not selected for test, followed by the remainder, in CSR format.
        This one denotes the values associated to 'Xrem_csr_i'.
    - Xtest_csr_p (out, dim:<=n_users_test+1)
        Testing data for users which are selected for test, in CSR format.
        This one denotes the index pointer.
    - Xtest_csr_i (out, dim:nnz_test)
        Testing data for users which are selected for test, in CSR format.
        This one denotes the indices of the non-zero entries.
    - Xtest_csr (out, dim:nnz_test)
        Testing data for users which are selected for test, in CSR format.
        This one denotes the values associated to 'Xtest_csr_i'.
    - n_users_test
        Number of users to set as test. If there are not enough users meeting the minimum
        desired criteria, the actual number of test users will be smaller.
    - test_fraction
        Target fraction of the data to set for test for each user. Should be a number
        between zero and one (non-inclusive). Note that the actual number of test entries
        for each user will be determined as 'round(n_entries*test_fraction)', thus
        in a long-tailed distribution (typical for recommender systems), the actual
        fraction that will be obtained is likely to be slightly lower than what
        is passed here.
    - consider_cold_start
        Whether to still set users as eligible for test in situations in which some user would
        have test data but no positive (non-zero) entries in the training data.
        This will only happen when passing 'test_fraction>=0.5'.
    - min_items_pool
        Minimum number of items (sum of positive and negative items) that a user must have in order to
        be eligible as test user.
    - min_pos_test
        Minimum number of positive entries (non-zero entries in the test set) that users would need
        to have in order to be eligible as test user.
    - seed
        Seed to use for random number generation.
*/
template <class real_t>
void split_data_joined_users
(
    const int32_t *restrict X_csr_p,
    const int32_t *restrict X_csr_i,
    const real_t *restrict X_csr,
    int32_t m, int32_t n,
    std::vector<int32_t> &users_test,
    std::vector<int32_t> &Xtrain_csr_p,
    std::vector<int32_t> &Xtrain_csr_i,
    std::vector<real_t> &Xtrain_csr,
    std::vector<int32_t> &Xtest_csr_p,
    std::vector<int32_t> &Xtest_csr_i,
    std::vector<real_t> &Xtest_csr,
    const int32_t n_users_test,
    const double test_fraction,
    const bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    uint64_t seed
)
{
    std::vector<int32_t> Xrem_csr_p;
    std::vector<int32_t> Xrem_csr_i;
    std::vector<real_t> Xrem_csr;

    std::vector<int32_t> Xtemp_csr_p;
    std::vector<int32_t> Xtemp_csr_i;
    std::vector<real_t> Xtemp_csr;

    split_data_separate_users(
        X_csr_p,
        X_csr_i,
        X_csr,
        m, n,
        users_test,
        Xrem_csr_p,
        Xrem_csr_i,
        Xrem_csr,
        Xtemp_csr_p,
        Xtemp_csr_i,
        Xtemp_csr,
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

    concat_csr_matrices(
        Xtemp_csr_p,
        Xtemp_csr_i,
        Xtemp_csr,
        Xrem_csr_p,
        Xrem_csr_i,
        Xrem_csr,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr
    );
}
