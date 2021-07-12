import numpy as np, pandas as pd
from . import cpp_funs
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix
from warnings import warn
import re
import multiprocessing
import ctypes

__all__ = ["calc_reco_metrics", "split_reco_train_test"]

def _as_row_major(X):
    if X.flags["C_CONTIGUOUS"]:
        return X, X.shape[1]
    if X.strides[1] != X.dtype.itemsize:
        return np.ascontiguousarray(X, dtype=X.dtype), X.shape[1]
    return X, int(X.strides[0] / X.itemsize)

def _cast_to_dtype(X, use_float):
    if not use_float:
        if X.dtype != ctypes.c_double:
            X = X.astype(ctypes.c_double)
    else:
        if X.dtype != ctypes.c_float:
            X = X.astype(ctypes.c_float)
    return X

def _cast_indices_to_int32(X):
    if (X.indptr.dtype != np.int32) or (X.indices.dtype != np.int32):
        X = X.copy()
        X.indptr = X.indptr.astype(np.int32)
        X.indices = X.indices.astype(np.int32)
    return X


def calc_reco_metrics(
    X_train, X_test,
    A, B,
    k = 5,
    item_biases = None,
    as_df = True,
    precision = True,
    trunc_precision = False,
    recall = False,
    average_precision = True,
    trunc_average_precision = False,
    ndcg = True,
    hit = False,
    rr = False,
    roc_auc = False,
    pr_auc = False,
    all_metrics = False,
    rename_k = True,
    break_ties_with_noise = True,
    min_pos_test = 1,
    min_items_pool = 2,
    consider_cold_start = True,
    cumulative = False,
    nthreads = -1,
    seed = 1
):
    """
    Calculate Recommendation Quality Metrics

    Calculates recommendation quality metrics for implicit-feedback
    recommender systems (fit to user-item interactions data such as "number of
    times that a user played each song in a music service") that are based on
    low-rank matrix factorization or for which predicted scores can be reduced to
    a dot product between user and item factors/components.
    
    These metrics are calculated on a per-user basis, by producing a ranking of the
    items according to model predictions (in descending order), ignoring the items
    that are in the training data for each user. The items that were not consumed
    by the user (not present in ``X_train`` and not present in ``X_test``) are considered
    "negative" entries, while the items in ``X_test`` are considered "positive" entries,
    and the items present in ``X_train`` are ignored for these calculations.
    
    The metrics that can be calculated by this function are:

    ``P@K`` ("precision-at-k")
        Denotes the proportion of items among the top-K
        recommended (after excluding those that were already in the training data)
        that can be found in the test set for that user:

        :math:`P@K = \\frac{1}{k} \\sum_{i=1}^k r_i \\in \\mathcal{T}`
    
        This is perhaps the most intuitive and straightforward metric, but it can
        present a lot of variation between users and does not take into account
        aspects such as the number of available test items or the specific ranks at
        which they are shown.

    ``TP@K`` (truncated precision-at-k)
        A truncated or standardized version
        of the precision metric, which will divide instead by the minimum between
        ``k`` and the number of test items:
        
        :math:`TP@K = \\frac{1}{\\min\\{k, \\mathcal{T}\\}} \\sum_{i=1}^k r_i \\in \\mathcal{T}`
        
        **Note:** many papers and libraries call this the "P@K" instead. The
        "truncated" prefix is a non-standard nomenclature introduced here to
        differentiate it from the P@K metric that is calculated by this and
        other libraries.

    ``R@K`` ("recall-at-k")
        Proportion of the test items that are retrieved
        in the top-K recommended list. Calculation is the same as precision, but the
        division is by the number of test items instead of ``k``:
        
        :math:`R@K = \\frac{1}{|\\mathcal{T}|} \\sum_{i=1}^k r_i \\in \\mathcal{T}`

    ``AP@K`` ("average precision-at-k")
        Precision and recall look at all the items
        in the top-K equally, whereas one might want to take into account also the ranking
        within this top-K list, for which this metric comes in handy.
        "Average Precision" tries to reflect the precisions that would be obtained at
        different recalls:
        
        :math:`AP@K = \\frac{1}{|\\mathcal{T}|} \\sum_{i=1}^k (r_i \\in \\mathcal{T}) \\cdot P@i`
        
        This is a metric which to some degree considers precision, recall, and rank within
        top-K. Intuitively, it tries to approximate the are under a precision-recall
        tradeoff curve.
        
        The average of this metric across users is known as "Mean Average Precision"
        or "MAP@K".
        
        **IMPORTANT:** many authors define AP@K differently, such as dividing by
        the minimum between ``k`` and the number of test items instead, or as the average
        for P@1..P@K (either as-is or stopping after already retrieving all the test
        items) - here, the second version is offered as different metric instead.
        This metric is likely to be a source of mismatches when comparing against
        other libraries due to all the different defintions used by different authors.

    ``TAP@K`` (truncated average precision-at-k)
        A truncated version of the
        AP@K metric, which will instead divide it by the minimum between ``k`` and the
        number of test items.
        
        Many other papers and libraries call this the "average precision" instead.

    ``NDCG@K`` (normalized discounted cumulative gain at K)
        A ranking metric
        calculated by first determining the following:
        
        :math:`\\sum_{i=1}^k \\frac{C_i}{log_2(i+1)}`
        
        Where :math:`C_i` denotes the confidence score for an item (taken as the value
        in ``X_test`` for that item), with :math:`i` being the item ranked at a given position
        for a given user according to the model. This metric is then standardized by
        dividing by the maximum achievable such score given the test data.
        
        Unlike the other metrics:

            - It looks not only at the presence or absence of items, but also at their
              confidence score.

            - It can handle data which contains "dislikes" in the form of negative
              values (see caveats below).
        
        If there are only non-negative values in ``X_test``, this metric will be bounded
        between zero and one.

        A note about negative values: the NDCG metric assumes that all the values are
        non-negative. This implementation however can accommodate situations in which
        a very small fraction of the items have negative values, in which case:
        (a) it will standardize the metric by dividing by a number which does not
        consider the negative values in its sum; (b) it will be set to ``nan`` if there
        are no positive values. Be aware however that NDCG loses some of its desirable
        properties in the presence of negative values.

    ``Hit@K`` ("hit-at-k")
        Indicates whether any of the top-K recommended items
        can be found in the test set for that user. The average across users is typically
        referred to as the "Hit Rate".
        
        This is a binary metric (it is either zero or one, but can also be ``nan`` when
        it is not possible to calculate, just like the other metrics).

    ``RR@K`` ("reciprocal-rank-at-k")
        Inverse rank (one divided by the rank)
        of the first item among the top-K recommended that is in the test data.
        The average across users is typically referred to as the "Mean Reciprocal Rank"
        or MRR.

        If none of the top-K recommended items are in the test data, will be set to zero.

    ``ROC-AUC`` (area under the receiver-operating characteristic curve)
        See the
        `Wikipedia entry <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
        for details. This metric considers the full ranking of items
        rather than just the top-K. It is bounded between zero and one, with a value of
        0.5 corresponding to a random order and a value of 1 corresponding to a perfect
        ordering (i.e. every single positive item has a higher predicted score than every
        single negative item).
        
        Be aware that there are different ways of calculating AUC, with some methods
        having higher precision than others. This implementation uses a fast formula
        which implies dividing two large numbers, and as such might not be as precise
        to the same number of decimals as the trapezoidal method used by e.g. scikit-learn.

    ``PR-AUC`` (area under the precision-recall curve)
        While ROC AUC provides an
        overview of the overall ranking, one is typically only interested in how well it
        retrieves test items within top ranks, and for this the area under the
        precision-recall curve can do a better job at judging rankings, albeit the metric
        itself is not standardized, and under the worst possible ranking, it does not
        evaluate to zero.
        
        The metric is calculated using the fast but not-so-precise rectangular method,
        whose formula corresponds to the AP@K metric with K=N. Some papers and libraries
        call this the average of this metric the "MAP" or "Mean Average Precision" instead
        (without the "@K").

    Metrics can be calculated for a given value of ``k`` (e.g. "P@3"), or for
    values ranging from 1 to ``k`` (e.g. ["P@1", "P@2", "P@3"]).

    This package does **NOT** cover other more specialized metrics. One might
    also want to look at other indicators such as:
        - Metrics that look at the rareness of the items recommended
          (to evaluate so-called "serendipity").
        - Metrics that look at "discoverability".
        - Metrics that take into account the diversity of the ranked lists.
    

    Metrics for a given user will be set to ``nan`` in the following
    situations:
        - All the rankeable items have the exact same predicted score.
        - One or more of the predicted scores evaluates to ``nan``.
        - There are only negative entries (no non-zero entries in the test data).
        - The number of available items to rank (between positive and negative) is
          smaller than the requested ``k``, and the metric is not affected by the exact order
          within the top-K items (i.e. precision, recall, hit, will be ``nan`` if there's
          ``k`` or fewer items after discarding those from the training data).
        - There are inconsistencies in the data (e.g. number of entries being greater
          than the number of columns in "X", meaning the matrices do not constitute valid CSR).
        - A user does not meet the minimum criteria set by the configurable parameters
          for this function.
        - There are only positive entries (i.e. the user already consumed all the items).
          In this case, "NDCG@K" will still be calculated, while the rest will be set
          to ``nan``.
    
    The NDCG@K metric with ``cumulative=True`` will have lower decimal precision
    than with ``cumulative=False`` when using ``np.float32`` inputs - this is extremely
    unlikely to be noticeable in typical datasets and small ``k``, but for large ``k``
    and large (absolute) values in ``X_test``, it might make a difference after a
    couple of decimal points.

    Parameters
    ----------
    X_train : CSR(m, n) or None
        Training data for user-item interactions, with users denoting rows,
        items denoting columns, and values corresponding to confidence scores.
        Entries in ``X_train`` and ``X_test`` for each user should not intersect (that is,
        if an item is the training data as a non-missing entry, it should not be in
        the test data as non-missing, and vice versa).
        
        Should be passed as a sparse matrix in CSR format
        from SciPy. Items not consumed by the user should not
        be present in this matrix.
        
        Alternatively, if there is no training data, can pass ``None``, in which case it
        will look only at the test data.
        
        This matrix and ``X_test`` are not meant to contain negative values, and if
        ``X_test`` does contain any, it will still be assumed for all metrics other than
        NDCG that such items are deemed better for the user than the missing/zero-valued
        items (that is, implicit feedback is not meant to signal dislikes).
    X_test : CSR(m, n)
        Test data for user-item interactions. Same format as ``X_train``.
    A : array(m, p) or None
        The user factors, as a NumPy array. If the number of users is 'm' and the number of
        factors is 'p', should have dimension '(m, p)'.
        If both ``A`` and ``B`` are passed with ``np.float32`` dtype, will do the
        calculations in single precision (which is faster and uses less memory) and
        output the  calculated metrics in that same precision, otherwise will do them
        in ``np.float64`` precision.
        
        It is assumed that the model score for a given item ``j`` for user ``i`` is
        calculated as the inner product or dot product between the corresponding vectors
        :math:`\\mathbf{a}_i \\cdot \\mathbf{b}_j`
        (rows ``i`` and ``j`` of ``A`` and ``B``, respectively),
        with higher scores meaning that the item is deemed better for
        that user, and the top-K recommendations produced by ranking these scores in
        descending order.
        
        Alternatively, for evaluation of non-personalized models, can pass ``None`` here
        and for ``B``, in which case ``item_biases`` must be passed.
    B : array(n, p) or None
        The item factors, in the same format as ``A``.
    k : int
        The number of top recommendations to consider for the metrics (as
        in "precision-at-k" or "P@K").
    item_biases : None or array(n,)
        Optional item biases/intercepts (fixed base score that is
        added to the predictions of each item). If present, it will append them to ``B``
        as an extra factor while adding a factor of all-ones to ``A``.

        Alternatively, for non-personalized models which have only item-by-item scores,
        can pass ``None`` for ``A`` and ``B`` while passing only ``item_biases``.
    as_df : bool
        Whether to output the result as a pandas DataFrame. If passing ``False``,
        the results will be returned as a dict of NumPy arrays - either 1d or 2d depending on
        what is passed for ``cumulative``.
    precision : bool
        Whether to calculate precision metrics or not.
    trunc_precision : bool
        Whether to calculate truncated precision metrics or not.
        Note that this is output as a separate metric from "precision" and they are not
        mutually exclusive options.
    recall : bool
        Whether to calculate recall metrics or not.
    average_precision : bool
        Whether to calculate average precision metrics or not.
    trunc_average_precision : bool
        Whether to calculate truncated average
        precision metrics or not. Note that this is output as a separate metric from
        "average_precision" and they are not mutually exclusive options.
    ndcg : bool
        Whether to calculate NDCG (normalized discounted cumulative gain)
        metrics or not.
    hit : bool
        Whether to calculate Hit metrics or not.
    rr : bool
        Whether to calculate RR (reciprocal rank) metrics or not.
    roc_auc : bool
        Whether to calculate ROC-AUC (area under the ROC curve) metrics or not.
    pr_auc : bool
        Whether to calculate PR-AUC (area under the PR curve) metrics or not.
    all_metrics : bool
        Passing ``True`` here is equivalent to passing ``True`` to all the
        calculable metrics.
    rename_k : bool
        If passing ``as_df=True`` and ``cumulative=False``, whether to rename
        the 'k' in the resulting column names to the actual value of 'k' that was used
        (e.g. "P@K" -> "P@5").
    break_ties_with_noise : bool
        Whether to add a small amount of noise
        :math:`\\sim \\text{Uniform}(-10^{-12}, 10^{-12})` in order to break ties
        at random, in case there are any ties in the ranking. This is not recommended unless
        one expects ties (can happen if e.g. some factors are set to all-zeros for some items),
        as it has the potential to slightly alter the ranking.
    min_pos_test : int
        Minimum number of positive entries
        (non-zero entries in the test set) that users need to have in
        order to calculate metrics for that user.
        If a given user does not meet the threshold, the metrics
        will be set to ``nan``.
    min_items_pool : int
        Minimum number of items (sum of positive and negative items)
        that a user must have in order to
        calculate metrics for that user. If a given user does not meet the threshold,
        the metrics will be set to ``nan``.
    consider_cold_start : bool
        Whether to calculate metrics in situations in
        which some user has test data but no positive
        (non-zero) entries in the training data. If passing ``False`` and such cases are
        encountered, the metrics will be set to ``nan``.

        Will be automatically set to ``True`` when passing ``None`` for ``X_train``.
    cumulative : bool
        Whether to calculate the metrics cumulatively
        (e.g. [P@1, P@2, P@3] if passing ``k=3``)
        for all values up to ``k``, or only for a single desired ``k``
        (e.g. only P@3 if passing ``k=3``).
    nthreads : int
        Number of parallel threads to use.
        Parallelization is done at the user level, so passing
        more threads than there are users will not result in a speed up. Be aware that, the more
        threads that are used, the higher the memory consumption.

        If passing a negative integer, the total number of threads to use will be determined
        in the same way as joblib:
            nthreads_use = available_threads + 1 + nthreads
    seed : int, RandomState, or Generator
        Seed used for random number generation. Only used when passing
        ``break_ties_with_noise=True``.

        Can be passed as an integer, or as a NumPy ``RandomState`` or
        ``Generator`` object, which will be used to draw a single random
        integer.

    Returns
    -------
    metrics : DataFrame or dict
        Will return the calculated metrics on a per-user basis (each user
        corresponding to a row):
            - If passing ``as_df=True`` (the default), the result will be a DataFrame,
              with the columns named according to the metric they represent (e.g. "P@3",
              see below for the other names that they can take). Depending on the value
              passed for ``rename_k``, the column names might end in "K" or in the number
              that was passed for ``k`` (e.g "P@3" or "P@K").

              If passing ``cumulative=True``, they will have names ranging from 1 to ``k``.
            - If passing ``as_df=False``, the result will be a list with entries named
              according to each metric, with ``k`` as letter rather than number (``P@K``,
              ``TP@K``, ``R@K``, ``AP@K``, ``TAP@K``, ``NDCG@K``, ``Hit@K``, ``RR@K``,
              ``ROC_AUC``, ``PR_AUC``), plus an additional entry with the actual ``k``.

              The values under each entry will be 1d arrays if passing
              ``cumulative=False``, or 2d arrays (users corresponding to rows) if passing
              ``cumulative=True``.
        
        The "ROC-AUC" and "PR-AUC" metrics will be named just "ROC_AUC" and "PR_AUC",
        since they are calculated for the full ranked predictions without stopping at ``k``.
    """
    if all_metrics:
        precision = True
        trunc_precision = True
        recall = True
        average_precision = True
        trunc_average_precision = True
        ndcg = True
        hit = True
        rr = True
        roc_auc = True
        pr_auc = True

    if (item_biases is not None) and (isinstance(item_biases, pd.Series)):
        item_biases = item_biases.to_numpy()

    if (A is None) != (B is None):
        raise ValueError("'A' and 'B' must either be passed together or passed as 'None' together.")
    if (A is None):
        if (item_biases is None):
            raise ValueError("Must pass item biases if not passing factors.")
        A = np.ones((X_test.shape[0],1), dtype=ctypes.c_double, order="C")
        B = np.ascontiguousarray(item_biases, dtype=ctypes.c_double).reshape((-1,1))
        item_biases = None

    assert isinstance(A, np.ndarray)
    assert isinstance(B, np.ndarray)
    assert issparse(X_test)

    if X_test.shape[0] >= np.iinfo(np.int32).max:
        raise ValueError("Number of test user is larger than maximum supported.")
    if X_test.shape[1] >= np.iinfo(np.int32).max:
        raise ValueError("Number of items is larger than maximum supported.")
    if not X_test.data.shape[0]:
        raise ValueError("'X_test' is empty.")

    if len(A.shape) != 2:
        raise ValueError("'A' must be a 2-dimensional array.")
    if len(B.shape) != 2:
        raise ValueError("'B' must be a 2-dimensional array.")
    if A.shape[1] != B.shape[1]:
        raise ValueError("'A' and 'B' must have the same number of columns.")
    if (not A.shape[0]) or (not A.shape[1]) or (not B.shape[1]) or (not X_test.shape[0]) or (not X_test.shape[1]):
        raise ValueError("Input matrices cannot be empty.")
    if A.shape[0] < X_test.shape[0]:
        raise ValueError("Number of users in 'A' and 'X_test' does not match.")
    if B.shape[0] < X_test.shape[1]:
        raise ValueError("Number of items in 'B' and 'X_test' does not match.")

    if (A.shape[0] > X_test.shape[0]):
        warn("'A' has more users than 'X_test'.")
        A = A[:X_test.shape[0], :]
    if (B.shape[0] > X_test.shape[1]):
        warn("'B' has more items than 'X_test'.")
        B = B[:X_test.shape[1], :]

    use_float = (A.dtype == ctypes.c_float) and (B.dtype == ctypes.c_float)

    if X_train is None:
        X_train = csr_matrix(X_test.shape, dtype=ctypes.c_double if not use_float else ctypes.c_float)
        consider_cold_start = True

    assert issparse(X_train)
    assert X_train.shape[1] == X_test.shape[1]
    if (X_train.shape[0] < X_test.shape[0]):
        raise ValueError("'X_train' and 'X_test' should have the same number of rows.")
    elif (X_train.shape[0] > X_test.shape[0]):
        warn("'X_train' mas more rows than 'X_test'.")

    as_df = bool(as_df)
    break_ties_with_noise = bool(break_ties_with_noise)
    consider_cold_start = bool(consider_cold_start)
    cumulative = bool(cumulative)
    rename_k = bool(rename_k)
    
    precision = bool(precision)
    trunc_precision = bool(trunc_precision)
    recall = bool(recall)
    average_precision = bool(average_precision)
    trunc_average_precision = bool(trunc_average_precision)
    ndcg = bool(ndcg)
    hit = bool(hit)
    rr = bool(rr)
    roc_auc = bool(roc_auc)
    pr_auc = bool(pr_auc)
    
    if (not precision) and (not average_precision) and (not ndcg) and (not hit) and (not rr) and (not roc_auc):
        raise ValueError("Must pass at least one metric to calculate.")

    if isinstance(seed, np.random.RandomState):
        seed = int(seed.randint(np.iinfo(np.int32).max))
    elif isinstance(seed, np.random.Generator):
        seed = int(seed.integers(np.iinfo(np.int32).max))

    nthreads = int(nthreads)
    seed = int(seed)
    k = int(k)
    min_pos_test = int(min_pos_test)
    min_items_pool = int(min_items_pool)

    assert seed >= 1
    assert k >= 1
    assert min_pos_test >= 1
    assert min_items_pool >= 1

    if nthreads is None:
        nthreads = 1
    elif nthreads < 0:
        nthreads = multiprocessing.cpu_count() + 1 + nthreads
    assert nthreads > 0

    if k > X_test.shape[1]:
        raise ValueError("'k' should be smaller than the number of items.")

    if item_biases is not None:
        assert isinstance(item_biases, np.ndarray)
        if (len(item_biases.shape) > 2):
            raise ValueError("'item_biases' should be a 1-d array.")
        if (len(item_biases.shape) != 1):
            item_biases = item_biases.reshape(-1)
        if (not item_biases.shape[0]):
            raise ValueError("'item_biases' is empty.")
        if item_biases.shape[0] < X_test.shape[1]:
            raise ValueError("Number of items in 'item_biases' must match with 'X_test'.")
        if item_biases.shape[0] > X_test.shape[1]:
            item_biases = item_biases[:X_test.shape[1]]
            warn("'item_biases' has more items than 'X_test'.")
        
        item_biases = _cast_to_dtype(item_biases, use_float)
        A = np.c_[A, np.ones((A.shape[0], 1), dtype=ctypes.c_double if not use_float else ctypes.c_float)]
        B = np.c_[B, item_biases.reshape((-1,1))]
        item_biases = None

    if not isspmatrix_csr(X_train):
        X_train = csr_matrix(X_train)
    else:
        X_train.sort_indices()

    if not isspmatrix_csr(X_test):
        X_test = csr_matrix(X_test)
    else:
        if (not roc_auc) and (not pr_auc):
            X_test.sort_indices()
    
    X_train = _cast_indices_to_int32(X_train)
    X_test = _cast_to_dtype(X_test, use_float)
    X_test = _cast_indices_to_int32(X_test)
    A = _cast_to_dtype(A, use_float)
    A, lda = _as_row_major(A)
    B = _cast_to_dtype(B, use_float)
    B, ldb = _as_row_major(B)

    p_at_k, tp_at_k, r_at_k, ap_at_k, tap_at_k, ndcg_at_k, \
    hit_at_k, rr_at_k, roc_auc, pr_auc = \
        cpp_funs.calc_reco_metrics(
            A, lda,
            B, ldb,
            X_train, X_test,
            k_metrics = k,
            precision = precision,
            trunc_precision = trunc_precision,
            recall = recall,
            average_precision = average_precision,
            trunc_average_precision = trunc_average_precision,
            ndcg = ndcg,
            hit = hit,
            rr = rr,
            roc_auc = roc_auc,
            pr_auc = pr_auc,
            break_ties_with_noise = break_ties_with_noise,
            min_pos_test = min_pos_test,
            min_items_pool = min_items_pool,
            consider_cold_start = consider_cold_start,
            cumulative = cumulative,
            nthreads = nthreads,
            seed = seed
        )

    out = dict()
    if p_at_k.shape[0]:
        out["P@K"] = p_at_k
    if tp_at_k.shape[0]:
        out["TP@K"] = tp_at_k
    if r_at_k.shape[0]:
        out["R@K"] = r_at_k
    if ap_at_k.shape[0]:
        out["AP@K"] = ap_at_k
    if tap_at_k.shape[0]:
        out["TAP@K"] = tap_at_k
    if ndcg_at_k.shape[0]:
        out["NDCG@K"] = ndcg_at_k
    if hit_at_k.shape[0]:
        out["Hit@K"] = hit_at_k
    if rr_at_k.shape[0]:
        out["RR@K"] = rr_at_k
    if roc_auc.shape[0]:
        out["ROC_AUC"] = roc_auc
    if pr_auc.shape[0]:
        out["PR_AUC"] = pr_auc

    if not as_df:
        out["K"] = k

    if as_df:
        if not cumulative:
            out = pd.DataFrame(out)
            if rename_k:
                out.columns = out.columns.str.replace("@K$", "@"+str(k), regex=True)
        else:
            out = [
                pd.DataFrame(v, columns=[re.sub("(@)K$", r"\1", k)+(str(i+1) if bool(re.search("@K$", k)) else "")
                                         for i in range(v.shape[1])])
                for k,v in out.items()
            ]
            out = pd.concat(out, axis=1)

    return out

def split_reco_train_test(
    X,
    split_type = "separated",
    users_test_fraction = 0.1,
    max_test_users = 10000,
    items_test_fraction = 0.3,
    min_items_pool = 2,
    min_pos_test = 1,
    consider_cold_start = False,
    seed = 1
):
    """
    Create Train-Test Splits of Implicit-Feedback Data

    Creates train-test splits of implicit-feedback data
    (recorded user-item interactions) for fitting and evaluating models for
    recommender systems.

    These splits choose "test users" and "items for a given user" separately,
    offering three modes of splitting the data:

        * Creating training and testing sets for each user in the data (when
          passing ``split_type='all'``).

          This is meant for cases in which the number of users is small or the users to
          test have already been selected (e.g. one typically does not want to create
          a train-test split which would leave one item for the user in the training data
          and zero in the test set, or would want to have other minimum criteria for the
          test set to be usable). Typically, one would want to take only a sub-sample
          of users for evaluation purposes, as calculating recommendation quality metrics
          can take a very long time.

        * Selecting a sub-set of users for testing, for which training and testing
          data splits will be generated, while leaving the remainder of users with all
          the data for model fitting (when passing ``split_type='separated'``).

          This is meant to be used for fitting a model to the remainder
          of the data, then generating latent factors (assuming a low-rank matrix factorization
          model) or top-K recommendations for the test users given their training data,
          and evaluating these recommendations on the test data for each user (which can be
          done through the function ``calc_reco_metrics``).

        * Selecting a sub-set of users for testing as above, but adding those users to
          the training data, in which case they will be the first rows (when passing
          ``split_type='joined'``).

          This is meant to be used for fitting a model to all such training data, and
          then evaluating the produced user factors or top-K recommendations for the test
          users against the test data.

          It is recommended to use the "separated" mode, as it
          is more reflective of real scenarios, but some models or libraries do not have the
          capabilities for producing factors/recommendations for users which where not in the
          training data, and this option then comes in handy.

    Parameters
    ----------
    X : CSR(m, n)
        The implicit feedback data to split into training-testing-remainder
        for evaluating recommender systems. Should be passed as a sparse CSR matrix from
        SciPy, or will be converted to CSR if it isn't. Users should correspond to rows, items
        to columns, and non-zero values to observed user-item interactions.

        Note that the indices of this matrix will be sorted in-place if it is passed as CSR.
        The data must be of type ``np.float32`` or ``np.float64`` - will be converted to
        ``np.float64`` if it is of a different type.
    split_type : str
        Type of data split to generate. Allowed values are:
        "all", "separated", "joined" (see the function description above for more details).
    users_test_fraction : float or None
        Target fraction of the users to set as test (see the
        function documentation for more details). If the number represented by this fraction
        exceeds the number set by ``max_test_users``, then the actual number will be set to
        ``max_test_users``. Note however that the end result might end up containing fewer
        users if there are not enough users in the data meeting the minimum desired
        criteria.

        If passing ``None``, will not take a fraction, but will instead take the number that
        is passed for ``max_test_users``.

        Ignored when passing ``split_type='all'``.
    max_test_users : float or None
        Maximum number of users to set as test. Note that this will
        only be applied for choosing the minimum between this and
        'ncol(X)*users_test_fraction', while the actual number might end up being
        lower if there are not enough users meeting the desired minimum conditions.

        If passing ``None`` for ``users_test_fraction``, will interpret this as the number
        of test users to take.

        Ignored when passing ``split_type='all'``.
    items_test_fraction : float
        Target fraction of the data (items) to set for test
        for each user. Should be a number between zero and one (non-inclusive).
        The actual number of test entries for each user will be determined as
        "round(n_entries_user*items_test_fraction)", thus in a long-tailed distribution
        (typical for recommender systems), the actual fraction that will be obtained is
        likely to be slightly lower than what is passed here.

        Note that items are sampled independently for each user, thus the items that are
        in the test set for some users might be in the training set for different users.
    min_items_pool : int
        Minimum number of items (sum of positive and negative items)
        that a user must have in order to be eligible as test user.
    min_pos_test : int
        Minimum number of positive entries (non-zero entries in
        the test set) that users would need to have in order to be eligible as test user.
    consider_cold_start : bool
        Whether to still set users as eligible for test in
        situations in which some user would have test data but no positive (non-zero)
        entries in the training data. This will only happen when passing
        ``test_fraction>=0.5``.
    seed : int
        Seed to use for random number generation.

    Returns
    -------
    split : tuple
        Will return a tuple with two to four elements depending on the requested split
        type:
            * If passing ``split_type='all'``, will output 2 elements: "X_train" and "X_test",
              both of which will be sparse CSR matrices
              with the same number of rows and columns as the ``X`` that was passed as input.
            * If passing ``split_type='separated'``, will output 4 elements: "X_train" and "X_test"
              as above (but with a number of rows corresponding to the number of selected test
              users instead), plus another CSR matrix "X_rem" which will contain the
              data for the remainder of the users (those which were not selected for testing and
              on which the recommendation model is meant to be fitted), and finally "users_test"
              which will be an integer vector containing the indices of the users/rows in ``X``
              which were selected for testing. The selected test users will be in sorted order,
              and the remaining data will remain in sorted order with the test users removed
              (e.g. if there's 5 users, with the second and fifth selected for testing, then
              "X_train" and "X_test" will contain rows [2,5] of ``X``, while "X_rem" will contain
              rows [1,3,4]).
            * If passing ``split_type='joined'``, will output 3 elements: same as above but will
              not contain "X_rem" - instead, "X_train" will be the concatenation of
              "X_train" and "X_rem",
              with "X_train" coming first (e.g. if there's 5 users, with the second and fifth
              selected for testing, then "X_test" will contain rows [2,5] of "X", while
              "X_train" will contain rows [2,5,1,3,4], in that order).
    """
    if (max_test_users is None) or (max_test_users == 0):
        max_test_users = X.shape[0]
    
    assert max_test_users > 0
    assert seed >= 0
    assert min_pos_test >= 0
    assert min_items_pool >= 0
    
    max_test_users = int(max_test_users)
    seed = int(seed)
    min_pos_test = int(min_pos_test)
    min_items_pool = int(min_items_pool)
    
    if users_test_fraction is not None:
        assert (users_test_fraction > 0) and (users_test_fraction < 1)
        users_test_fraction = float(users_test_fraction)

    assert (items_test_fraction > 0) and (items_test_fraction < 1)
    items_test_fraction = float(items_test_fraction)

    assert split_type in ("all", "separated", "joined")

    consider_cold_start = bool(consider_cold_start)


    if min_pos_test >= X.shape[1]:
        raise ValueError("'min_pos_test' must be smaller than the number of columns in 'X'.")

    if min_items_pool >= X.shape[1]:
        raise ValueError("'min_items_pool' must be smaller than the number of columns in 'X'.")

    if split_type != "all":
        if X.shape[0] < 2:
            raise ValueError("'X' has less than 2 rows.")
        if users_test_fraction is not None:
            n_users_take = X.shape[0] * users_test_fraction
            if n_users_take < 1:
                warn("Desired fraction of test users implies <1, will select 1 user.")
                n_users_take = 1
            n_users_take = round(n_users_take)
            n_users_take = min(n_users_take, max_test_users)
        else:
            if max_test_users > X.shape[0]:
                warn("'max_test_users' is larger than number of users. Will take all.")
            n_users_take = min(max_test_users, X.shape[0])

    if not isspmatrix_csr(X):
        X = csr_matrix(X)
    else:
        X.sort_indices()
    if (not X.shape[0]) or (not X.shape[1]):
        raise ValueError("'X' cannot be empty.")
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float64)
    if not X.data.shape[0]:
        raise ValueError("'X' contains no non-zero entries.")
    X = _cast_indices_to_int32(X)

    if split_type == "all":
        return cpp_funs.split_csr_selected_users(X, items_test_fraction, seed)
    elif split_type == "separated":
        return cpp_funs.split_csr_separated_users(
            X,
            n_users_take,
            items_test_fraction,
            consider_cold_start,
            min_items_pool,
            min_pos_test,
            True,
            seed
        )
    elif split_type == "joined":
        return cpp_funs.split_csr_separated_users(
            X,
            n_users_take,
            items_test_fraction,
            consider_cold_start,
            min_items_pool,
            min_pos_test,
            False,
            seed
        )
    else:
        raise ValueError("Unexpected error.")
