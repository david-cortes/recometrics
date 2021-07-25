import numpy as np
cimport numpy as np
from libcpp cimport bool as c_bool
from libcpp.vector cimport vector
from libc.string cimport memcpy
from libc.stdint cimport int32_t, uint64_t
from scipy.sparse import csr_matrix
import ctypes

### TODO: here cython fails to compile the functions using fused types or templates,
### and depending on how it's done (fused/templates), might produce something compilable
### with gcc/clang but not with msvc. Find out why it fails and avoid this duplication
### of code. Perhaps it has to do with this issue:
### https://github.com/cython/cython/issues/3968

ctypedef fused generic_t:
    int32_t
    double
    float

ctypedef fused real_t:
    double
    float

cdef extern from "recometrics_signatures.hpp":
    c_bool get_has_openmp() nogil except +

    void calc_metrics_double(
        const double *A, const size_t lda, const double *B, const size_t ldb,
        const int32_t m, const int32_t n, const int32_t k,
        const int32_t *Xtrain_csr_p, const int32_t *Xtrain_csr_i,
        const int32_t *Xtest_csr_p, const int32_t *Xtest_csr_i, const double *Xtest_csr,
        const int32_t k_metrics,
        const c_bool cumulative,
        const c_bool break_ties_with_noise,
        double *p_at_k,
        double *tp_at_k,
        double *r_at_k,
        double *ap_at_k,
        double *tap_at_k,
        double *ndcg_at_k,
        double *hit_at_k,
        double *rr_at_k,
        double *roc_auc,
        double *pr_auc,
        const c_bool consider_cold_start,
        int32_t min_items_pool,
        int32_t min_pos_test,
        int32_t nthreads,
        uint64_t seed
    ) nogil except +

    void split_data_selected_users_double(
        const int32_t *X_csr_p,
        const int32_t *X_csr_i,
        const double *X_csr,
        const int32_t m, const int32_t n,
        vector[int32_t] &Xtrain_csr_p,
        vector[int32_t] &Xtrain_csr_i,
        vector[double] &Xtrain_csr,
        vector[int32_t] &Xtest_csr_p,
        vector[int32_t] &Xtest_csr_i,
        vector[double] &Xtest_csr,
        const double test_fraction,
        uint64_t seed
    ) nogil except +

    void split_data_separate_users_double(
        const int32_t *X_csr_p,
        const int32_t *X_csr_i,
        const double *X_csr,
        int32_t m, int32_t n,
        vector[int32_t] &users_test,
        vector[int32_t] &Xrem_csr_p,
        vector[int32_t] &Xrem_csr_i,
        vector[double] &Xrem_csr,
        vector[int32_t] &Xtrain_csr_p,
        vector[int32_t] &Xtrain_csr_i,
        vector[double] &Xtrain_csr,
        vector[int32_t] &Xtest_csr_p,
        vector[int32_t] &Xtest_csr_i,
        vector[double] &Xtest_csr,
        const int32_t n_users_test,
        const double test_fraction,
        const c_bool consider_cold_start,
        const int32_t min_items_pool,
        const int32_t min_pos_test,
        uint64_t seed
    ) nogil except +

    void split_data_joined_users_double(
        const int32_t *X_csr_p,
        const int32_t *X_csr_i,
        const double *X_csr,
        int32_t m, int32_t n,
        vector[int32_t] &users_test,
        vector[int32_t] &Xtrain_csr_p,
        vector[int32_t] &Xtrain_csr_i,
        vector[double] &Xtrain_csr,
        vector[int32_t] &Xtest_csr_p,
        vector[int32_t] &Xtest_csr_i,
        vector[double] &Xtest_csr,
        const int32_t n_users_test,
        const double test_fraction,
        const c_bool consider_cold_start,
        const int32_t min_items_pool,
        const int32_t min_pos_test,
        uint64_t seed
    ) nogil except +

    void calc_metrics_float(
        const float *A, const size_t lda, const float *B, const size_t ldb,
        const int32_t m, const int32_t n, const int32_t k,
        const int32_t *Xtrain_csr_p, const int32_t *Xtrain_csr_i,
        const int32_t *Xtest_csr_p, const int32_t *Xtest_csr_i, const float *Xtest_csr,
        const int32_t k_metrics,
        const c_bool cumulative,
        const c_bool break_ties_with_noise,
        float *p_at_k,
        float *tp_at_k,
        float *r_at_k,
        float *ap_at_k,
        float *tap_at_k,
        float *ndcg_at_k,
        float *hit_at_k,
        float *rr_at_k,
        float *roc_auc,
        float *pr_auc,
        const c_bool consider_cold_start,
        int32_t min_items_pool,
        int32_t min_pos_test,
        int32_t nthreads,
        uint64_t seed
    ) nogil except +

    void split_data_selected_users_float(
        const int32_t *X_csr_p,
        const int32_t *X_csr_i,
        const float *X_csr,
        const int32_t m, const int32_t n,
        vector[int32_t] &Xtrain_csr_p,
        vector[int32_t] &Xtrain_csr_i,
        vector[float] &Xtrain_csr,
        vector[int32_t] &Xtest_csr_p,
        vector[int32_t] &Xtest_csr_i,
        vector[float] &Xtest_csr,
        const double test_fraction,
        uint64_t seed
    ) nogil except +

    void split_data_separate_users_float(
        const int32_t *X_csr_p,
        const int32_t *X_csr_i,
        const float *X_csr,
        int32_t m, int32_t n,
        vector[int32_t] &users_test,
        vector[int32_t] &Xrem_csr_p,
        vector[int32_t] &Xrem_csr_i,
        vector[float] &Xrem_csr,
        vector[int32_t] &Xtrain_csr_p,
        vector[int32_t] &Xtrain_csr_i,
        vector[float] &Xtrain_csr,
        vector[int32_t] &Xtest_csr_p,
        vector[int32_t] &Xtest_csr_i,
        vector[float] &Xtest_csr,
        const int32_t n_users_test,
        const double test_fraction,
        const c_bool consider_cold_start,
        const int32_t min_items_pool,
        const int32_t min_pos_test,
        uint64_t seed
    ) nogil except +

    void split_data_joined_users_float(
        const int32_t *X_csr_p,
        const int32_t *X_csr_i,
        const float *X_csr,
        int32_t m, int32_t n,
        vector[int32_t] &users_test,
        vector[int32_t] &Xtrain_csr_p,
        vector[int32_t] &Xtrain_csr_i,
        vector[float] &Xtrain_csr,
        vector[int32_t] &Xtest_csr_p,
        vector[int32_t] &Xtest_csr_i,
        vector[float] &Xtest_csr,
        const int32_t n_users_test,
        const double test_fraction,
        const c_bool consider_cold_start,
        const int32_t min_items_pool,
        const int32_t min_pos_test,
        uint64_t seed
    ) nogil except +

def _get_has_openmp():
    return get_has_openmp()

cdef double* get_ptr_double(np.ndarray[double, ndim=1] a):
    if a.shape[0]:
        return &a[0]
    else:
        return <double*>NULL

cdef float* get_ptr_float(np.ndarray[float, ndim=1] a):
    if a.shape[0]:
        return &a[0]
    else:
        return <float*>NULL

cdef int32_t* get_ptr_int(np.ndarray[int32_t, ndim=1] a):
    if a.shape[0]:
        return &a[0]
    else:
        return <int32_t*>NULL

cdef tuple cy_calc_metrics_double(
    np.ndarray[double, ndim=2] A, size_t lda,
    np.ndarray[double, ndim=2] B, size_t ldb,
    Xcsr_train, Xcsr_test,
    int32_t k_metrics = 10,
    c_bool precision = True,
    c_bool trunc_precision = False,
    c_bool recall = False,
    c_bool average_precision = True,
    c_bool trunc_average_precision = False,
    c_bool ndcg = True,
    c_bool hit = False,
    c_bool rr = False,
    c_bool roc_auc = False,
    c_bool pr_auc = False,
    c_bool break_ties_with_noise = 1,
    int32_t min_pos_test = 1,
    int32_t min_items_pool = 2,
    c_bool consider_cold_start = 0,
    c_bool cumulative = 0,
    int32_t nthreads = 1,
    uint64_t seed = 1
):
    
    cdef np.ndarray[int32_t, ndim=1] Xtrain_csr_p = Xcsr_train.indptr
    cdef np.ndarray[int32_t, ndim=1] Xtrain_csr_i = Xcsr_train.indices

    cdef np.ndarray[int32_t, ndim=1] Xtest_csr_p = Xcsr_test.indptr
    cdef np.ndarray[int32_t, ndim=1] Xtest_csr_i = Xcsr_test.indices
    cdef np.ndarray[double, ndim=1] Xtest_csr = Xcsr_test.data

    cdef int32_t m = A.shape[0]
    cdef int32_t n = B.shape[0]
    cdef int32_t k = A.shape[1]

    if (A.shape[0] != Xcsr_train.shape[0]):
        raise ValueError("User shapes do not match.")
    if (B.shape[0] != Xcsr_train.shape[1]):
        raise ValueError("Item shapes do not match.")
    if (A.shape[1] != B.shape[1]):
        raise ValueError("Factors have non-matching shapes.")
    if (Xcsr_train.shape[0] != Xcsr_test.shape[0]) or (Xcsr_train.shape[1] != Xcsr_test.shape[1]):
        raise ValueError("Train-test shapes do not match.")

    cdef size_t size_arr = m if not cumulative else m*k_metrics
    cdef np.ndarray[double, ndim=1] p_at_k = np.empty(size_arr if precision else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] tp_at_k = np.empty(size_arr if trunc_precision else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] r_at_k = np.empty(size_arr if recall else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] ap_at_k = np.empty(size_arr if average_precision else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] tap_at_k = np.empty(size_arr if trunc_average_precision else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] ndcg_at_k = np.empty(size_arr if ndcg else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] hit_at_k = np.empty(size_arr if hit else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] rr_at_k = np.empty(size_arr if rr else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] roc_auc_ = np.empty(m if roc_auc else 0, dtype=ctypes.c_double)
    cdef np.ndarray[double, ndim=1] pr_auc_ = np.empty(m if pr_auc else 0, dtype=ctypes.c_double)

    calc_metrics_double(
        &A[0,0], lda, &B[0,0], ldb, m, n, k,
        get_ptr_int(Xtrain_csr_p), get_ptr_int(Xtrain_csr_i),
        get_ptr_int(Xtest_csr_p), get_ptr_int(Xtest_csr_i), &Xtest_csr[0],
        k_metrics,
        cumulative,
        break_ties_with_noise,
        get_ptr_double(p_at_k),
        get_ptr_double(tp_at_k),
        get_ptr_double(r_at_k),
        get_ptr_double(ap_at_k),
        get_ptr_double(tap_at_k),
        get_ptr_double(ndcg_at_k),
        get_ptr_double(hit_at_k),
        get_ptr_double(rr_at_k),
        get_ptr_double(roc_auc_),
        get_ptr_double(pr_auc_),
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        nthreads,
        seed
    )

    cdef tuple size_shape = (m,k_metrics)
    cdef tuple size_empty = (<int>0,<int>0)

    if not cumulative:
        return p_at_k, tp_at_k, r_at_k, ap_at_k, tap_at_k, ndcg_at_k, hit_at_k, rr_at_k, roc_auc_, pr_auc_
    else:
        return (
            p_at_k.reshape(size_shape if precision else size_empty),
            tp_at_k.reshape(size_shape if trunc_precision else size_empty),
            r_at_k.reshape(size_shape if recall else size_empty),
            ap_at_k.reshape(size_shape if average_precision else size_empty),
            tap_at_k.reshape(size_shape if trunc_average_precision else size_empty),
            ndcg_at_k.reshape(size_shape if ndcg else size_empty),
            hit_at_k.reshape(size_shape if hit else size_empty),
            rr_at_k.reshape(size_shape if rr else size_empty),
            roc_auc_,
            pr_auc_
        )

cdef tuple cy_calc_metrics_float(
    np.ndarray[float, ndim=2] A, size_t lda,
    np.ndarray[float, ndim=2] B, size_t ldb,
    Xcsr_train, Xcsr_test,
    int32_t k_metrics = 10,
    c_bool precision = True,
    c_bool trunc_precision = False,
    c_bool recall = False,
    c_bool average_precision = True,
    c_bool trunc_average_precision = False,
    c_bool ndcg = True,
    c_bool hit = False,
    c_bool rr = False,
    c_bool roc_auc = False,
    c_bool pr_auc = False,
    c_bool break_ties_with_noise = 1,
    int32_t min_pos_test = 1,
    int32_t min_items_pool = 2,
    c_bool consider_cold_start = 0,
    c_bool cumulative = 0,
    int32_t nthreads = 1,
    uint64_t seed = 1
):
    
    cdef np.ndarray[int32_t, ndim=1] Xtrain_csr_p = Xcsr_train.indptr
    cdef np.ndarray[int32_t, ndim=1] Xtrain_csr_i = Xcsr_train.indices

    cdef np.ndarray[int32_t, ndim=1] Xtest_csr_p = Xcsr_test.indptr
    cdef np.ndarray[int32_t, ndim=1] Xtest_csr_i = Xcsr_test.indices
    cdef np.ndarray[float, ndim=1] Xtest_csr = Xcsr_test.data

    cdef int32_t m = A.shape[0]
    cdef int32_t n = B.shape[0]
    cdef int32_t k = A.shape[1]

    if (A.shape[0] != Xcsr_train.shape[0]):
        raise ValueError("User shapes do not match.")
    if (B.shape[0] != Xcsr_train.shape[1]):
        raise ValueError("Item shapes do not match.")
    if (A.shape[1] != B.shape[1]):
        raise ValueError("Factors have non-matching shapes.")
    if (Xcsr_train.shape[0] != Xcsr_test.shape[0]) or (Xcsr_train.shape[1] != Xcsr_test.shape[1]):
        raise ValueError("Train-test shapes do not match.")

    cdef size_t size_arr = m if not cumulative else m*k_metrics
    cdef np.ndarray[float, ndim=1] p_at_k = np.empty(size_arr if precision else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] tp_at_k = np.empty(size_arr if trunc_precision else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] r_at_k = np.empty(size_arr if recall else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] ap_at_k = np.empty(size_arr if average_precision else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] tap_at_k = np.empty(size_arr if trunc_average_precision else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] ndcg_at_k = np.empty(size_arr if ndcg else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] hit_at_k = np.empty(size_arr if hit else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] rr_at_k = np.empty(size_arr if rr else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] roc_auc_ = np.empty(m if roc_auc else 0, dtype=ctypes.c_float)
    cdef np.ndarray[float, ndim=1] pr_auc_ = np.empty(m if pr_auc else 0, dtype=ctypes.c_float)

    calc_metrics_float(
        &A[0,0], lda, &B[0,0], ldb, m, n, k,
        get_ptr_int(Xtrain_csr_p), get_ptr_int(Xtrain_csr_i),
        get_ptr_int(Xtest_csr_p), get_ptr_int(Xtest_csr_i), &Xtest_csr[0],
        k_metrics,
        cumulative,
        break_ties_with_noise,
        get_ptr_float(p_at_k),
        get_ptr_float(tp_at_k),
        get_ptr_float(r_at_k),
        get_ptr_float(ap_at_k),
        get_ptr_float(tap_at_k),
        get_ptr_float(ndcg_at_k),
        get_ptr_float(hit_at_k),
        get_ptr_float(rr_at_k),
        get_ptr_float(roc_auc_),
        get_ptr_float(pr_auc_),
        consider_cold_start,
        min_items_pool,
        min_pos_test,
        nthreads,
        seed
    )

    
    cdef tuple size_shape = (m,k_metrics)
    cdef tuple size_empty = (<int>0,<int>0)

    if not cumulative:
        return p_at_k, tp_at_k, r_at_k, ap_at_k, tap_at_k, ndcg_at_k, hit_at_k, rr_at_k, roc_auc_, pr_auc_
    else:
        return (
            p_at_k.reshape(size_shape if precision else size_empty),
            tp_at_k.reshape(size_shape if trunc_precision else size_empty),
            r_at_k.reshape(size_shape if recall else size_empty),
            ap_at_k.reshape(size_shape if average_precision else size_empty),
            tap_at_k.reshape(size_shape if trunc_average_precision else size_empty),
            ndcg_at_k.reshape(size_shape if ndcg else size_empty),
            hit_at_k.reshape(size_shape if hit else size_empty),
            rr_at_k.reshape(size_shape if rr else size_empty),
            roc_auc_,
            pr_auc_
        )

def calc_reco_metrics(
    A, size_t lda,
    B, size_t ldb,
    Xcsr_train, Xcsr_test,
    int32_t k_metrics = 10,
    c_bool precision = True,
    c_bool trunc_precision = False,
    c_bool recall = False,
    c_bool average_precision = True,
    c_bool trunc_average_precision = False,
    c_bool ndcg = True,
    c_bool hit = False,
    c_bool rr = False,
    c_bool roc_auc = False,
    c_bool pr_auc = False,
    c_bool break_ties_with_noise = 1,
    int32_t min_pos_test = 1,
    int32_t min_items_pool = 2,
    c_bool consider_cold_start = 0,
    c_bool cumulative = 0,
    int32_t nthreads = 1,
    uint64_t seed = 1
):
    if A.dtype == ctypes.c_float:
        return cy_calc_metrics_float(
            A, lda,
            B, ldb,
            Xcsr_train, Xcsr_test,
            k_metrics,
            precision,
            trunc_precision,
            recall,
            average_precision,
            trunc_average_precision,
            ndcg,
            hit,
            rr,
            roc_auc,
            pr_auc,
            break_ties_with_noise,
            min_pos_test,
            min_items_pool,
            consider_cold_start,
            cumulative,
            nthreads,
            seed
        )
    else:
        return cy_calc_metrics_double(
            A, lda,
            B, ldb,
            Xcsr_train, Xcsr_test,
            k_metrics,
            precision,
            trunc_precision,
            recall,
            average_precision,
            trunc_average_precision,
            ndcg,
            hit,
            rr,
            roc_auc,
            pr_auc,
            break_ties_with_noise,
            min_pos_test,
            min_items_pool,
            consider_cold_start,
            cumulative,
            nthreads,
            seed
        )

cdef np.ndarray[generic_t, ndim=1] cppvec_to_numpy(
    vector[generic_t] inp
):
    cdef np.ndarray[generic_t, ndim=1] out = np.empty(inp.size(),
        dtype=np.int32 if (generic_t is int32_t) else (ctypes.c_double if generic_t is double else (ctypes.c_float))
    )
    if not inp.size():
        return out
    memcpy(&out[0], inp.data(), inp.size() * <size_t>sizeof(generic_t))
    inp.clear()
    return out

cdef object cpp_csr_to_scipy(
    vector[int32_t] Xcsr_p,
    vector[int32_t] Xcsr_i,
    vector[generic_t] Xcsr,
    int32_t n
):
    cdef np.ndarray[int32_t, ndim=1] Xcsr_p_ = cppvec_to_numpy[int32_t](Xcsr_p)
    cdef np.ndarray[int32_t, ndim=1] Xcsr_i_ = cppvec_to_numpy[int32_t](Xcsr_i)
    cdef np.ndarray[generic_t, ndim=1] Xcsr_ = cppvec_to_numpy[generic_t](Xcsr)
    cdef int m = Xcsr_p_.shape[0] - 1
    return csr_matrix((Xcsr_, Xcsr_i_, Xcsr_p_), shape=(m, n))

cdef tuple split_csr_selected_users_double(
    X_csr_scipy,
    const double test_fraction,
    uint64_t seed = 1
):
    cdef np.ndarray[int32_t, ndim=1] X_csr_p = X_csr_scipy.indptr
    cdef np.ndarray[int32_t, ndim=1] X_csr_i = X_csr_scipy.indices
    cdef np.ndarray[double, ndim=1] X_csr = X_csr_scipy.data
    cdef int32_t m = X_csr_scipy.shape[0]
    cdef int32_t n = X_csr_scipy.shape[1]

    cdef vector[int32_t] Xtrain_csr_p
    cdef vector[int32_t] Xtrain_csr_i
    cdef vector[double] Xtrain_csr
    cdef vector[int32_t] Xtest_csr_p
    cdef vector[int32_t] Xtest_csr_i
    cdef vector[double] Xtest_csr

    cdef const double* tempp = &X_csr[0] if X_csr.shape[0] else NULL
    
    split_data_selected_users_double(
        get_ptr_int(X_csr_p),
        get_ptr_int(X_csr_i),
        tempp,
        m, n,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        test_fraction,
        seed
    )

    Xtrain_out = cpp_csr_to_scipy[double](Xtrain_csr_p, Xtrain_csr_i, Xtrain_csr, n)
    Xtest_out = cpp_csr_to_scipy[double](Xtest_csr_p, Xtest_csr_i, Xtest_csr, n)
    return Xtrain_out, Xtest_out

cdef tuple split_csr_selected_users_float(
    X_csr_scipy,
    const double test_fraction,
    uint64_t seed = 1
):
    cdef np.ndarray[int32_t, ndim=1] X_csr_p = X_csr_scipy.indptr
    cdef np.ndarray[int32_t, ndim=1] X_csr_i = X_csr_scipy.indices
    cdef np.ndarray[float, ndim=1] X_csr = X_csr_scipy.data
    cdef int32_t m = X_csr_scipy.shape[0]
    cdef int32_t n = X_csr_scipy.shape[1]

    cdef vector[int32_t] Xtrain_csr_p
    cdef vector[int32_t] Xtrain_csr_i
    cdef vector[float] Xtrain_csr
    cdef vector[int32_t] Xtest_csr_p
    cdef vector[int32_t] Xtest_csr_i
    cdef vector[float] Xtest_csr

    cdef const float* tempp = &X_csr[0]
    
    split_data_selected_users_float(
        get_ptr_int(X_csr_p),
        get_ptr_int(X_csr_i),
        tempp,
        m, n,
        Xtrain_csr_p,
        Xtrain_csr_i,
        Xtrain_csr,
        Xtest_csr_p,
        Xtest_csr_i,
        Xtest_csr,
        test_fraction,
        seed
    )

    Xtrain_out = cpp_csr_to_scipy[float](Xtrain_csr_p, Xtrain_csr_i, Xtrain_csr, n)
    Xtest_out = cpp_csr_to_scipy[float](Xtest_csr_p, Xtest_csr_i, Xtest_csr, n)
    return Xtrain_out, Xtest_out

def split_csr_selected_users(
    X_csr_scipy,
    const double test_fraction = 0.2,
    uint64_t seed = 1
):
    if X_csr_scipy.dtype == ctypes.c_float:
        return split_csr_selected_users_float(
            X_csr_scipy,
            test_fraction,
            seed
        )
    else:
        return split_csr_selected_users_double(
            X_csr_scipy,
            test_fraction,
            seed
        )

cdef tuple split_csr_separated_users_double(
    X_csr_scipy,
    const int32_t n_users_test,
    const double test_fraction,
    const c_bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    const c_bool separated,
    uint64_t seed
):
    cdef np.ndarray[int32_t, ndim=1] X_csr_p = X_csr_scipy.indptr
    cdef np.ndarray[int32_t, ndim=1] X_csr_i = X_csr_scipy.indices
    cdef np.ndarray[double, ndim=1] X_csr = X_csr_scipy.data
    cdef int32_t m = X_csr_scipy.shape[0]
    cdef int32_t n = X_csr_scipy.shape[1]

    cdef vector[int32_t] Xtrain_csr_p
    cdef vector[int32_t] Xtrain_csr_i
    cdef vector[double] Xtrain_csr
    cdef vector[int32_t] Xtest_csr_p
    cdef vector[int32_t] Xtest_csr_i
    cdef vector[double] Xtest_csr
    cdef vector[int32_t] Xrem_csr_p
    cdef vector[int32_t] Xrem_csr_i
    cdef vector[double] Xrem_csr

    cdef vector[int32_t] users_test
    cdef np.ndarray[int32_t, ndim=1] users_test_

    if separated:
        split_data_separate_users_double(
            get_ptr_int(X_csr_p),
            get_ptr_int(X_csr_i),
            &X_csr[0] if X_csr.shape[0] else NULL,
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
        )

        Xtrain_out = cpp_csr_to_scipy[double](Xtrain_csr_p, Xtrain_csr_i, Xtrain_csr, n)
        Xtest_out = cpp_csr_to_scipy[double](Xtest_csr_p, Xtest_csr_i, Xtest_csr, n)
        Xrem_out = cpp_csr_to_scipy[double](Xrem_csr_p, Xrem_csr_i, Xrem_csr, n)
        users_test_ = cppvec_to_numpy[int32_t](users_test)
        return Xrem_out, Xtrain_out, Xtest_out, users_test_

    else:
        split_data_joined_users_double(
            get_ptr_int(X_csr_p),
            get_ptr_int(X_csr_i),
            &X_csr[0] if X_csr.shape[0] else NULL,
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
        )

        Xtrain_out = cpp_csr_to_scipy[double](Xtrain_csr_p, Xtrain_csr_i, Xtrain_csr, n)
        Xtest_out = cpp_csr_to_scipy[double](Xtest_csr_p, Xtest_csr_i, Xtest_csr, n)
        users_test_ = cppvec_to_numpy[int32_t](users_test)
        return Xtrain_out, Xtest_out, users_test_

cdef tuple split_csr_separated_users_float(
    X_csr_scipy,
    const int32_t n_users_test,
    const double test_fraction,
    const c_bool consider_cold_start,
    const int32_t min_items_pool,
    const int32_t min_pos_test,
    const c_bool separated,
    uint64_t seed
):
    cdef np.ndarray[int32_t, ndim=1] X_csr_p = X_csr_scipy.indptr
    cdef np.ndarray[int32_t, ndim=1] X_csr_i = X_csr_scipy.indices
    cdef np.ndarray[float, ndim=1] X_csr = X_csr_scipy.data
    cdef int32_t m = X_csr_scipy.shape[0]
    cdef int32_t n = X_csr_scipy.shape[1]

    cdef vector[int32_t] Xtrain_csr_p
    cdef vector[int32_t] Xtrain_csr_i
    cdef vector[float] Xtrain_csr
    cdef vector[int32_t] Xtest_csr_p
    cdef vector[int32_t] Xtest_csr_i
    cdef vector[float] Xtest_csr
    cdef vector[int32_t] Xrem_csr_p
    cdef vector[int32_t] Xrem_csr_i
    cdef vector[float] Xrem_csr

    cdef vector[int32_t] users_test
    cdef np.ndarray[int32_t, ndim=1] users_test_

    if separated:
        split_data_separate_users_float(
            get_ptr_int(X_csr_p),
            get_ptr_int(X_csr_i),
            &X_csr[0] if X_csr.shape[0] else NULL,
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
        )

        Xtrain_out = cpp_csr_to_scipy[float](Xtrain_csr_p, Xtrain_csr_i, Xtrain_csr, n)
        Xtest_out = cpp_csr_to_scipy[float](Xtest_csr_p, Xtest_csr_i, Xtest_csr, n)
        Xrem_out = cpp_csr_to_scipy[float](Xrem_csr_p, Xrem_csr_i, Xrem_csr, n)
        users_test_ = cppvec_to_numpy[int32_t](users_test)
        return Xrem_out, Xtrain_out, Xtest_out, users_test_

    else:
        split_data_joined_users_float(
            get_ptr_int(X_csr_p),
            get_ptr_int(X_csr_i),
            &X_csr[0] if X_csr.shape[0] else NULL,
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
        )

        Xtrain_out = cpp_csr_to_scipy[float](Xtrain_csr_p, Xtrain_csr_i, Xtrain_csr, n)
        Xtest_out = cpp_csr_to_scipy[float](Xtest_csr_p, Xtest_csr_i, Xtest_csr, n)
        users_test_ = cppvec_to_numpy[int32_t](users_test)
        return Xtrain_out, Xtest_out, users_test_

def split_csr_separated_users(
    X_csr_scipy,
    const int32_t n_users_test,
    const double test_fraction = 0.2,
    const c_bool consider_cold_start = 0,
    const int32_t min_items_pool = 10,
    const int32_t min_pos_test = 1,
    const c_bool separated = True,
    uint64_t seed = 1
):
    if X_csr_scipy.dtype == ctypes.c_float:
        return split_csr_separated_users_float(
            X_csr_scipy,
            n_users_test,
            test_fraction,
            consider_cold_start,
            min_items_pool,
            min_pos_test,
            separated,
            seed
        )
    else:
        return split_csr_separated_users_double(
            X_csr_scipy,
            n_users_test,
            test_fraction,
            consider_cold_start,
            min_items_pool,
            min_pos_test,
            separated,
            seed
        )
