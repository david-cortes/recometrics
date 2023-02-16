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
#ifdef _FOR_R

#include "recometrics.hpp"
#include <type_traits>
#include <Rcpp.h>
#include <Rcpp/unwindProtect.h>
// [[Rcpp::plugins(cpp11)]]
extern "C" {
#include <Rinternals.h>
}

SEXP convert_IntVecToRcpp(void *data)
{
    return Rcpp::IntegerVector(((std::vector<int32_t>*)data)->begin(),
                               ((std::vector<int32_t>*)data)->end());
}

SEXP convert_NumVecToRcpp(void *data)
{
    return Rcpp::NumericVector(((std::vector<double>*)data)->begin(),
                               ((std::vector<double>*)data)->end());
}


template <class real_t, class RcppVector>
real_t* NUMERIC_PTR(RcppVector Rvec)
{
    if (!Rvec.size())
        return (real_t*)nullptr;
    if (std::is_same<real_t, double>::value)
        return (real_t*)REAL(Rvec);
    else
        return (real_t*)INTEGER(Rvec);
}

template <class real_t, class RcppNumericVector, class RcppNumericMatrix>
Rcpp::List calc_metrics
(
    RcppNumericMatrix A,
    RcppNumericMatrix B,
    Rcpp::IntegerVector Xtrain_csr_p,
    Rcpp::IntegerVector Xtrain_csr_i,
    Rcpp::IntegerVector Xtest_csr_p,
    Rcpp::IntegerVector Xtest_csr_i,
    Rcpp::NumericVector Xtest_csr,
    bool calc_p_at_k = true,
    bool calc_tp_at_k = false,
    bool calc_r_at_k = false,
    bool calc_ap_at_k = true,
    bool calc_tap_at_k = false,
    bool calc_ndcg_at_k = true,
    bool calc_hit_at_k = false,
    bool calc_rr_at_k = false,
    bool calc_roc_auc = false,
    bool calc_pr_auc = false,
    int k_metrics = 10,
    bool break_ties_with_noise = true,
    int min_pos_test = 1,
    int min_items_pool = 2,
    bool consider_cold_start = 0,
    bool cumulative = 0,
    int nthreads = 1,
    uint64_t seed = 1
)
{
    int m = A.ncol();
    int n = B.ncol();
    int k = A.nrow();

    int32_t *ptr_Xtrain_csr_p = nullptr;
    int32_t *ptr_Xtrain_csr_i = nullptr;
    int32_t *ptr_Xtest_csr_p = nullptr;
    int32_t *ptr_Xtest_csr_i = nullptr;
    std::unique_ptr<int32_t[]> cp_Xtrain_csr_p;
    std::unique_ptr<int32_t[]> cp_Xtrain_csr_i;
    std::unique_ptr<int32_t[]> cp_Xtest_csr_p;
    std::unique_ptr<int32_t[]> cp_Xtest_csr_i;

    std::unique_ptr<real_t[]> cp_Xtest_csr;
    real_t *ptr_Xtest_csr = nullptr;

    RcppNumericMatrix p_at_k_mat;
    RcppNumericMatrix tp_at_k_mat;
    RcppNumericMatrix r_at_k_mat;
    RcppNumericMatrix ap_at_k_mat;
    RcppNumericMatrix tap_at_k_mat;
    RcppNumericMatrix ndcg_at_k_mat;
    RcppNumericMatrix hit_at_k_mat;
    RcppNumericMatrix rr_at_k_mat;
    
    RcppNumericVector p_at_k_vec;
    RcppNumericVector tp_at_k_vec;
    RcppNumericVector r_at_k_vec;
    RcppNumericVector ap_at_k_vec;
    RcppNumericVector tap_at_k_vec;
    RcppNumericVector ndcg_at_k_vec;
    RcppNumericVector hit_at_k_vec;
    RcppNumericVector rr_at_k_vec;
    RcppNumericVector roc_auc;
    RcppNumericVector pr_auc;

    real_t *p_at_k_ptr = nullptr;
    real_t *tp_at_k_ptr = nullptr;
    real_t *r_at_k_ptr = nullptr;
    real_t *ap_at_k_ptr = nullptr;
    real_t *tap_at_k_ptr = nullptr;
    real_t *ndcg_at_k_ptr = nullptr;
    real_t *hit_at_k_ptr = nullptr;
    real_t *rr_at_k_ptr = nullptr;

    if (cumulative)
    {
        if (calc_p_at_k) {
            p_at_k_mat = RcppNumericMatrix(k_metrics, m);
            p_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericMatrix>(p_at_k_mat);
        }
        if (calc_tp_at_k) {
            tp_at_k_mat = RcppNumericMatrix(k_metrics, m);
            tp_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericMatrix>(tp_at_k_mat);
        }
        if (calc_r_at_k) {
            r_at_k_mat = RcppNumericMatrix(k_metrics, m);
            r_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericMatrix>(r_at_k_mat);
        }
        if (calc_ap_at_k) {
            ap_at_k_mat = RcppNumericMatrix(k_metrics, m);
            ap_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericMatrix>(ap_at_k_mat);
        }
        if (calc_tap_at_k) {
            tap_at_k_mat = RcppNumericMatrix(k_metrics, m);
            tap_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericMatrix>(tap_at_k_mat);
        }
        if (calc_ndcg_at_k) {
            ndcg_at_k_mat = RcppNumericMatrix(k_metrics, m);
            ndcg_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericMatrix>(ndcg_at_k_mat);
        }
        if (calc_hit_at_k) {
            hit_at_k_mat = RcppNumericMatrix(k_metrics, m);
            hit_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericMatrix>(hit_at_k_mat);
        }
        if (calc_rr_at_k) {
            rr_at_k_mat = RcppNumericMatrix(k_metrics, m);
            rr_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericMatrix>(rr_at_k_mat);
        }
    }

    else
    {
        if (calc_p_at_k) {
            p_at_k_vec = RcppNumericVector(m);
            p_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericVector>(p_at_k_vec);
        }
        if (calc_tp_at_k) {
            tp_at_k_vec = RcppNumericVector(m);
            tp_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericVector>(tp_at_k_vec);
        }
        if (calc_r_at_k) {
            r_at_k_vec = RcppNumericVector(m);
            r_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericVector>(r_at_k_vec);
        }
        if (calc_ap_at_k) {
            ap_at_k_vec = RcppNumericVector(m);
            ap_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericVector>(ap_at_k_vec);
        }
        if (calc_tap_at_k) {
            tap_at_k_vec = RcppNumericVector(m);
            tap_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericVector>(tap_at_k_vec);
        }
        if (calc_ndcg_at_k) {
            ndcg_at_k_vec = RcppNumericVector(m);
            ndcg_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericVector>(ndcg_at_k_vec);
        }
        if (calc_hit_at_k) {
            hit_at_k_vec = RcppNumericVector(m);
            hit_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericVector>(hit_at_k_vec);
        }
        if (calc_rr_at_k) {
            rr_at_k_vec = RcppNumericVector(m);
            rr_at_k_ptr = NUMERIC_PTR<real_t, RcppNumericVector>(rr_at_k_vec);
        }
    }

    if (calc_roc_auc)
        roc_auc = RcppNumericVector(m);
    if (calc_pr_auc)
        pr_auc = RcppNumericVector(m);

    if (std::is_same<int, int32_t>::value) {
        ptr_Xtrain_csr_p = (int32_t*)INTEGER(Xtrain_csr_p);
        ptr_Xtrain_csr_i = (int32_t*)INTEGER(Xtrain_csr_i);
        ptr_Xtest_csr_p = (int32_t*)INTEGER(Xtest_csr_p);
        ptr_Xtest_csr_i = (int32_t*)INTEGER(Xtest_csr_i);
    }

    else {
        if (sizeof(int) > sizeof(int32_t))
            Rcpp::Rcerr << "Platform has non-standard integer size, results might suffer from integer overflow." << std::endl;
        cp_Xtrain_csr_p = std::unique_ptr<int32_t[]>(new int32_t[Xtrain_csr_p.size()]);
        cp_Xtrain_csr_i = std::unique_ptr<int32_t[]>(new int32_t[Xtrain_csr_i.size()]);
        cp_Xtest_csr_p = std::unique_ptr<int32_t[]>(new int32_t[Xtest_csr_p.size()]);
        cp_Xtest_csr_i = std::unique_ptr<int32_t[]>(new int32_t[Xtest_csr_i.size()]);

        std::copy(Xtrain_csr_p.begin(), Xtrain_csr_p.end(), cp_Xtrain_csr_p.get());
        std::copy(Xtrain_csr_i.begin(), Xtrain_csr_i.end(), cp_Xtrain_csr_i.get());
        std::copy(Xtest_csr_p.begin(), Xtest_csr_p.end(), cp_Xtest_csr_p.get());
        std::copy(Xtest_csr_i.begin(), Xtest_csr_i.end(), cp_Xtest_csr_i.get());

        ptr_Xtrain_csr_p = cp_Xtrain_csr_p.get();
        ptr_Xtrain_csr_i = cp_Xtrain_csr_i.get();
        ptr_Xtest_csr_p = cp_Xtest_csr_p.get();
        ptr_Xtest_csr_i = cp_Xtest_csr_i.get();
    }

    if (std::is_same<real_t, double>::value) {
        ptr_Xtest_csr = (real_t*)REAL(Xtest_csr);
    }

    else if (calc_ndcg_at_k) {
        cp_Xtest_csr = std::unique_ptr<real_t[]>(new real_t[Xtest_csr.size()]);
        std::copy(Xtest_csr.begin(), Xtest_csr.end(), cp_Xtest_csr.get());
        ptr_Xtest_csr = cp_Xtest_csr.get();
    }


    calc_metrics<real_t>(
        NUMERIC_PTR<real_t, RcppNumericMatrix>(A), k,
        NUMERIC_PTR<real_t, RcppNumericMatrix>(B), k,
        m, n, k,
        ptr_Xtrain_csr_p, ptr_Xtrain_csr_i,
        ptr_Xtest_csr_p, ptr_Xtest_csr_i, ptr_Xtest_csr,
        k_metrics,
        cumulative,
        break_ties_with_noise,
        p_at_k_ptr,
        tp_at_k_ptr,
        r_at_k_ptr,
        ap_at_k_ptr,
        tap_at_k_ptr,
        ndcg_at_k_ptr,
        hit_at_k_ptr,
        rr_at_k_ptr,
        NUMERIC_PTR<real_t, RcppNumericVector>(roc_auc),
        NUMERIC_PTR<real_t, RcppNumericVector>(pr_auc),
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        nthreads,
        seed
    );

    Rcpp::List out;
    if (calc_p_at_k) out["p_at_k"] = cumulative? p_at_k_mat : p_at_k_vec;
    if (calc_tp_at_k) out["tp_at_k"] = cumulative? tp_at_k_mat : tp_at_k_vec;
    if (calc_r_at_k) out["r_at_k"] = cumulative? r_at_k_mat : r_at_k_vec;
    if (calc_ap_at_k) out["ap_at_k"] = cumulative? ap_at_k_mat : ap_at_k_vec;
    if (calc_tap_at_k) out["tap_at_k"] = cumulative? tap_at_k_mat : tap_at_k_vec;
    if (calc_ndcg_at_k) out["ndcg_at_k"] = cumulative? ndcg_at_k_mat : ndcg_at_k_vec;
    if (calc_hit_at_k) out["hit_at_k"] = cumulative? hit_at_k_mat : hit_at_k_vec;
    if (calc_rr_at_k) out["rr_at_k"] = cumulative? rr_at_k_mat : rr_at_k_vec;
    if (calc_roc_auc) out["roc_auc"] = roc_auc;
    if (calc_pr_auc) out["pr_auc"] = pr_auc;
    out["k"] = k_metrics;

    return out;
}

#endif

// [[Rcpp::export(rng = false)]]
Rcpp::List calc_metrics_double
(
    Rcpp::NumericMatrix A,
    Rcpp::NumericMatrix B,
    Rcpp::IntegerVector Xtrain_csr_p,
    Rcpp::IntegerVector Xtrain_csr_i,
    Rcpp::IntegerVector Xtest_csr_p,
    Rcpp::IntegerVector Xtest_csr_i,
    Rcpp::NumericVector Xtest_csr,
    bool calc_p_at_k = true,
    bool calc_tp_at_k = false,
    bool calc_r_at_k = false,
    bool calc_ap_at_k = true,
    bool calc_tap_at_k = false,
    bool calc_ndcg_at_k = true,
    bool calc_hit_at_k = false,
    bool calc_rr_at_k = false,
    bool calc_roc_auc = false,
    bool calc_pr_auc = false,
    int k_metrics = 10,
    bool break_ties_with_noise = true,
    int min_pos_test = 1,
    int min_items_pool = 2,
    bool consider_cold_start = 0,
    bool cumulative = 0,
    int nthreads = 1,
    uint64_t seed = 1
)
{
    return calc_metrics<double, Rcpp::NumericVector, Rcpp::NumericMatrix>(
        A,
        B,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        calc_p_at_k,
        calc_tp_at_k,
        calc_r_at_k,
        calc_ap_at_k,
        calc_tap_at_k,
        calc_ndcg_at_k,
        calc_hit_at_k,
        calc_rr_at_k,
        calc_roc_auc,
        calc_pr_auc,
        k_metrics,
        break_ties_with_noise,
        min_pos_test,
        min_items_pool,
        consider_cold_start,
        cumulative,
        nthreads,
        seed
    );
}

// [[Rcpp::export(rng = false)]]
Rcpp::List calc_metrics_float
(
    Rcpp::IntegerMatrix A,
    Rcpp::IntegerMatrix B,
    Rcpp::IntegerVector Xtrain_csr_p,
    Rcpp::IntegerVector Xtrain_csr_i,
    Rcpp::IntegerVector Xtest_csr_p,
    Rcpp::IntegerVector Xtest_csr_i,
    Rcpp::NumericVector Xtest_csr,
    bool calc_p_at_k = true,
    bool calc_tp_at_k = false,
    bool calc_r_at_k = false,
    bool calc_ap_at_k = true,
    bool calc_tap_at_k = false,
    bool calc_ndcg_at_k = true,
    bool calc_hit_at_k = false,
    bool calc_rr_at_k = false,
    bool calc_roc_auc = false,
    bool calc_pr_auc = false,
    int k_metrics = 10,
    bool break_ties_with_noise = true,
    int min_pos_test = 1,
    int min_items_pool = 2,
    bool consider_cold_start = 0,
    bool cumulative = 0,
    int nthreads = 1,
    uint64_t seed = 1
)
{
    return calc_metrics<float, Rcpp::IntegerVector, Rcpp::IntegerMatrix>(
        A,
        B,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        calc_p_at_k,
        calc_tp_at_k,
        calc_r_at_k,
        calc_ap_at_k,
        calc_tap_at_k,
        calc_ndcg_at_k,
        calc_hit_at_k,
        calc_rr_at_k,
        calc_roc_auc,
        calc_pr_auc,
        k_metrics,
        break_ties_with_noise,
        min_pos_test,
        min_items_pool,
        consider_cold_start,
        cumulative,
        nthreads,
        seed
    );
}

// [[Rcpp::export(rng = false)]]
Rcpp::List split_csr_selected_users
(
    Rcpp::IntegerVector X_csr_p,
    Rcpp::IntegerVector X_csr_i,
    Rcpp::NumericVector X_csr,
    int ncols,
    const double test_fraction,
    uint64_t seed = 1
)
{
    std::vector<int32_t> Xtrain_csr_p;
    std::vector<int32_t> Xtrain_csr_i;
    std::vector<double> Xtrain_csr;

    std::vector<int32_t> Xtest_csr_p;
    std::vector<int32_t> Xtest_csr_i;
    std::vector<double> Xtest_csr;

    std::unique_ptr<int32_t[]> cp_X_csr_p;
    std::unique_ptr<int32_t[]> cp_X_csr_i;
    int32_t *ptr_X_csr_p = nullptr;
    int32_t *ptr_X_csr_i = nullptr;

    if (std::is_same<int, int32_t>::value) {
        ptr_X_csr_p = (int32_t*)INTEGER(X_csr_p);
        ptr_X_csr_i = (int32_t*)INTEGER(X_csr_i);
    }

    else {
        cp_X_csr_p = std::unique_ptr<int32_t[]>(new int32_t[X_csr_p.size()]);
        cp_X_csr_i = std::unique_ptr<int32_t[]>(new int32_t[X_csr_i.size()]);

        std::copy(X_csr_p.begin(), X_csr_p.end(), cp_X_csr_p.get());
        std::copy(X_csr_i.begin(), X_csr_i.end(), cp_X_csr_i.get());

        ptr_X_csr_p = cp_X_csr_p.get();
        ptr_X_csr_i = cp_X_csr_i.get();
    }
    
    split_data_selected_users<double>(
        ptr_X_csr_p,
        ptr_X_csr_i,
        REAL(X_csr),
        X_csr_p.size()-1, ncols,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        test_fraction,
        seed
    );

    return Rcpp::List::create(
        Rcpp::_["Xtrain_csr_p"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&Xtrain_csr_p),
        Rcpp::_["Xtrain_csr_i"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&Xtrain_csr_i),
        Rcpp::_["Xtrain_csr"] = Rcpp::unwindProtect(convert_NumVecToRcpp, (void*)&Xtrain_csr),
        Rcpp::_["Xtest_csr_p"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&Xtest_csr_p),
        Rcpp::_["Xtest_csr_i"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&Xtest_csr_i),
        Rcpp::_["Xtest_csr"] = Rcpp::unwindProtect(convert_NumVecToRcpp, (void*)&Xtest_csr)
    );
}

// [[Rcpp::export(rng = false)]]
Rcpp::List split_csr_separated_users
(
    Rcpp::IntegerVector X_csr_p,
    Rcpp::IntegerVector X_csr_i,
    Rcpp::NumericVector X_csr,
    const int32_t ncols,
    const int32_t n_users_test,
    const double test_fraction,
    const bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    const bool separated,
    uint64_t seed = 1
)
{
    std::vector<int32_t> Xtrain_csr_p;
    std::vector<int32_t> Xtrain_csr_i;
    std::vector<double> Xtrain_csr;

    std::vector<int32_t> Xtest_csr_p;
    std::vector<int32_t> Xtest_csr_i;
    std::vector<double> Xtest_csr;

    std::vector<int32_t> Xrem_csr_p;
    std::vector<int32_t> Xrem_csr_i;
    std::vector<double> Xrem_csr;

    std::vector<int32_t> users_test;

    std::unique_ptr<int32_t[]> cp_X_csr_p;
    std::unique_ptr<int32_t[]> cp_X_csr_i;
    int32_t *ptr_X_csr_p = nullptr;
    int32_t *ptr_X_csr_i = nullptr;

    if (std::is_same<int, int32_t>::value) {
        ptr_X_csr_p = (int32_t*)INTEGER(X_csr_p);
        ptr_X_csr_i = (int32_t*)INTEGER(X_csr_i);
    }

    else {
        cp_X_csr_p = std::unique_ptr<int32_t[]>(new int32_t[X_csr_p.size()]);
        cp_X_csr_i = std::unique_ptr<int32_t[]>(new int32_t[X_csr_i.size()]);

        std::copy(X_csr_p.begin(), X_csr_p.end(), cp_X_csr_p.get());
        std::copy(X_csr_i.begin(), X_csr_i.end(), cp_X_csr_i.get());

        ptr_X_csr_p = cp_X_csr_p.get();
        ptr_X_csr_i = cp_X_csr_i.get();
    }

    if (separated) {
        split_data_separate_users<double>(
            ptr_X_csr_p,
            ptr_X_csr_i,
            REAL(X_csr),
            X_csr_p.size()-1, ncols,
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

        for (auto &el : users_test)
            el += 1;

        return Rcpp::List::create(
            Rcpp::_["Xtrain_csr_p"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xtrain_csr_p),
            Rcpp::_["Xtrain_csr_i"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xtrain_csr_i),
            Rcpp::_["Xtrain_csr"] = Rcpp::unwindProtect(convert_NumVecToRcpp, &Xtrain_csr),
            Rcpp::_["Xtest_csr_p"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xtest_csr_p),
            Rcpp::_["Xtest_csr_i"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xtest_csr_i),
            Rcpp::_["Xtest_csr"] = Rcpp::unwindProtect(convert_NumVecToRcpp, &Xtest_csr),
            Rcpp::_["Xrem_csr_p"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xrem_csr_p),
            Rcpp::_["Xrem_csr_i"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xrem_csr_i),
            Rcpp::_["Xrem_csr"] = Rcpp::unwindProtect(convert_NumVecToRcpp, &Xrem_csr),
            Rcpp::_["users_test"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &users_test)
        );
    }

    else {
        split_data_joined_users<double>(
            ptr_X_csr_p,
            ptr_X_csr_i,
            REAL(X_csr),
            X_csr_p.size()-1, ncols,
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

        for (auto &el : users_test)
            el += 1;

        return Rcpp::List::create(
            Rcpp::_["Xtrain_csr_p"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xtrain_csr_p),
            Rcpp::_["Xtrain_csr_i"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xtrain_csr_i),
            Rcpp::_["Xtrain_csr"] = Rcpp::unwindProtect(convert_NumVecToRcpp, &Xtrain_csr),
            Rcpp::_["Xtest_csr_p"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xtest_csr_p),
            Rcpp::_["Xtest_csr_i"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &Xtest_csr_i),
            Rcpp::_["Xtest_csr"] = Rcpp::unwindProtect(convert_NumVecToRcpp, &Xtest_csr),
            Rcpp::_["users_test"] = Rcpp::unwindProtect(convert_IntVecToRcpp, &users_test)
        );
    }
}

// [[Rcpp::export(rng = false)]]
void C_NAN_to_R_NA(SEXP vec)
{
    auto n = Rf_xlength(vec);
    double *x = REAL(vec);
    for (size_t ix = 0; ix < (size_t)n; ix++)
        x[ix] = std::isnan(x[ix])? NA_REAL : x[ix];
}

// [[Rcpp::export(rng = false)]]
bool R_has_openmp()
{
    #ifdef _OPENMP
    return true;
    #else
    return false;
    #endif
}

/* Workaround in case it somehow tries and fails to link to non-R BLAS */
#if defined(__APPLE__) || defined(_WIN32)
float sdot_(const int *n, const float *dx, const int *incx, const float *dy, const int *incy)
{
    double res = 0.;
    for (int ix = 0; ix < *n; ix++) res = std::fma(dx[ix], dy[ix], res);
    return res;
}
#endif
