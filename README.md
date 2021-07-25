# RecoMetrics

Library-agnostic evaluation framework for implicit-feedback recommender systems that are based on low-rank matrix factorization models or latent embeddings. Calculates per-user metrics based on the ranking of items produced by the model, using efficient multi-threaded routines. Also provides functions for generating train-test splits of data. Writen in C++ with interfaces for Python and R.

** *

For a longer introduction, see:

* [IPython notebook](http://nbviewer.jupyter.org/github/david-cortes/recometrics/blob/master/examples/recometrics_example.ipynb).
* [R Vignette](http://htmlpreview.github.io/?https://github.com/david-cortes/recometrics/blob/master/examples/Evaluating_recommender_systems.html).

# Evaluating implicit-feedback recommendations

When evaluating recommender systems built from implicit-feedback data (e.g. numer of times that a user played each song in a music service), one usually wants to evaluate the quality of the produced recommendations according to how well they rank the available pool of items for each user.

This is done by setting aside some fraction of the data for testing purposes, then building a model on the remainder of the data, and producing top-K recommended lists for each user that exclude the items already consumed by him/her in this non-held-out data. The recommended lists are evaluated according to how they rank the held-out items in comparison to the items that the user did not consume, using classification or ranking metrics such as precision, NDCG, AUC, among others. The held-out items are considered to be "positive" entries, while the items which were not consumed by the user are considered "negative" entries.

Compared to metrics used for explicit-feedback recommendations such as RMSE, metrics for implicit-feedback are much slower to compute (might well be slower than fitting the model itself), since they require generating a ranking of a large number of items for each user separately and iterating over ranked lists. Many libraries for recommender systems provide their own functionality for automatically setting aside some data and evaluating these metrics while fitting models, but there are some issues with such approach:

* They can only evaluate models created with the same library, thus not allowing comparisons between libraries.
* The methodologies are oftentimes not comparable across libraries (e.g. they differ in how they would discard users with few data or what would they output in edge cases).
* Results are sometimes not possible to reproduce exactly on the outside (e.g. the library outputs only the metrics, but not the exact data split that was used).
* Oftentimes, such evaluations are done in pure Python+NumPy, either using a single core, or sharing model matrices and data across processes by serializing them, which results in very slow calculations. What's more, sometimes different metrics are calculated separately, requiring to re-generate the ranking for each metric.


This library, in contrast:

* Takes as input the model matrices and train-test data (as CSR matrices), thus allowing to work with any recommendation model in which the predicted scores are created from an inner product between user and item factors, regardless of library. Example libraries with these type of models: [implicit](https://github.com/benfred/implicit/), [libmf](https://github.com/cjlin1/libmf), [lightfm](https://github.com/lyst/lightfm), [spotlight](https://github.com/maciejkula/spotlight), [cmfrec](https://github.com/david-cortes/cmfrec), [rsparse](https://github.com/rexyai/rsparse), [lenskit](https://github.com/lenskit/lkpy), among many others.
* Allows specifying criteria for filtering users to evaluate based on required amount of data (e.g. minimum number of positive test entries, minimum size of items pool to rank, whether cold-start recommendations are accepted, among others).
* Outputs `NaN` when a metric is not calculable instead of silently filling with zeros or ones (e.g. if the user has no positive entries or no negative entries in the test data), and makes logical checks for invalid cases such as all predictions having the same values or having NAs.
* Can calculate different metrics (e.g. AP@K, NDCG@K) in the same pass, without having to re-rank the items for each user, and allowing to generate the metrics for many values of `K` at the same time (e.g. NDCG@1, NDCG@2, ..., NDCG@10, instead of just NDCG@10).
* Provides the calculation on a per-user basis, not just in aggregate, allowing further filters and post-hoc comparisons.
* Can be used to generate the train-test split separately, setting configurable minimum criteria for the test users, desired size of the test data, and sampling users and items independently, thus allowing faster calculations with sub-sampled users.
* Uses multi-threaded computations with a shared-memory model, SIMD CPU instructions (can use `float32` and `float64`), and efficient search procedures, thus running much faster than pure-Python or pure-R software.


# Supported metrics

* P@K ("precision-at-k"): this is the proportion of the top-K recommended items (excluding those that were in the training data) that are present in the test data of a given user. Can also produce a standardized or "truncated" version which will divide by the minimum between K and the number of test items.

* R@K ("recall-at-k"): this is the proportion of the test items that are found among the top-K recommended, thus accounting for the fact that some users have more test data than others and thus it's easier to find test items for them.

* AP@K ("average-precision-at-k"): this is conceptually a metric which looks at precision, recall, and rank, by calculating precisions at different recall levels. See the [Wikipedia entry](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision) for more information. Also offers a "truncated" version like for P@K. The average of this metric across users is typically called "MAP@K" or "Mean Average Precision".

* NDCG@K (normalized discounted cumulative gain): this is a ranking metric that takes into account not only the presence of recommended items in the test set, but also their confidence score (according to the data), discounting this score according to the ranking of the item in the top-K list. Entries not present in the test data are assumed to have a score of zero. See the [Wikipedia entry](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) for more details.

* Hit@K: indicates whether at least one of the top-K recommended items was in the test data (the by-user average is typically called "Hit Rate").

* RR@K (reciprocal rank): inverse rank (one divided by the rank) of the first item among the top-K recommended that is in the test data (the by-user average is typically called "Mean Reciprocal Rank" or MRR).

* ROC AUC (are under the receiver-operating characteristic curve). See the [Wikipedia entry](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve) for more details. This metric evaluates the full ranking rather than just top-K.

* PR AUC (are under the precision-recall curve): just like ROC-AUC, it evaluates the full ranking, but it is a lot more sensitive about what happens at the top of the ranks, providing a perhaps more helpful picture than ROC-AUC. It is calculated using the fast-but-not-so-precise rectangular method, whose formula corresponds to the AP@K metric with K=N.

This package does **NOT** deal with other more specialized metrics evaluating e.g. "serendipity", "discoverability", diversity of recommendations, etc.

# Installation

* Python:

```pip install recometrics```

Or if that fails:
```
pip install --no-use-pep517 recometrics
```

** *

**Note for macOS users:** on macOS, the Python version of this package might compile **without** multi-threading capabilities. In order to enable multi-threading support, first install OpenMP:
```
brew install libomp
```
And then reinstall this package: `pip install --force-reinstall recometrics`.


* R:

```r
install.packages("recometrics")
```

For better performance, it's recommended to compile the package from source with extra optimizations `-O3` and `-march=native` - in Linux, this can be done by creating a file `~/.R/Makevars` containing this line: `CXX11FLAGS += -O3 -march=native` (plus an empty line at the end) (this file should be created before installing `recometrics`).

* C++:

Library is a self-contained templated header file (`src/recometrics.hpp`). Can be copied into other projects and used by `#include "..."`'ing it.


# Documentation

* Python: documentation is available at [ReadTheDocs](http://recometrics.readthedocs.io).

* R: documentation is internally available on [CRAN](https://cran.r-project.org/web/packages/recometrics/index.html).

* C++: documentation is available in the header file [src/recometrics.hpp](https://github.com/david-cortes/recometrics/blob/master/src/recometrics.hpp).

# Examples

Applied examples with public data and different libraries for fitting models:

* [Python notebook](http://nbviewer.jupyter.org/github/david-cortes/recometrics/blob/master/examples/recometrics_example.ipynb) (LastFM-360K dataset, using libraries `implicit` and `lightfm`).

* [R vignette](https://cran.r-project.org/web/packages/recometrics/vignettes/Evaluating_recommender_systems.html) (MovieLens100K dataset, using library `cmfrec`).

# Sample usage

* Python:

```python
import numpy as np
from scipy.sparse import csr_matrix, random as sprandom
import recometrics

### User-item interactions (e.g. number of video views)
n_users = 100
n_items = 50
rng = np.random.default_rng(seed=123)
X = sprandom(n_users, n_items, density=0.2,
             data_rvs=lambda n: rng.integers(1, 100, n),
             format="csr")

### Creating a fit + train-test split
X_fit, X_train, X_test, test_users = \
    recometrics.split_reco_train_test(
        X, split_type="separated",
        users_test_fraction=0.1,
        items_test_fraction=0.3,
        min_items_pool=10, min_pos_test=2,
        seed=123
    )

### Model would be fit to 'X_fit' (non-test users)
### e.g. model = Model(...).fit(X_fit)
latent_dim = 5
Item_Factors = rng.standard_normal((n_items, latent_dim))

### Then it would produce user factors for 'X_train'
### (users to which the model was not fit)
User_Factors = rng.standard_normal((X_train.shape[0], latent_dim))

### And then the metrics would be calculated
df_metrics_by_user = \
    recometrics.calc_reco_metrics(
        X_train, X_test,
        User_Factors, Item_Factors,
        k=5,
        precision=True,
        average_precision=True,
        ndcg=True,
        nthreads=-1
    )
df_metrics_by_user.head(3)
```
```
   P@5      AP@5    NDCG@5
0  0.0  0.000000  0.000000
1  0.0  0.000000  0.000000
2  0.2  0.333333  0.610062
```

* R:

```r
library(Matrix)
library(recometrics)

### User-item interactions (e.g. number of video views)
n_users <- 100
n_items <- 50
set.seed(123)
X <- rsparsematrix(n_users, n_items, density=0.2, repr="R",
                   rand.x=function(n) sample(100, n, replace=TRUE))

### Creating a fit + train-test split
temp <- create.reco.train.test(
    X, split_type="separated",
    users_test_fraction=0.1,
    items_test_fraction=0.3,
    min_items_pool=10, min_pos_test=2,
    seed=1
)
X_train <- temp$X_train
X_test <- temp$X_test
X_fit <- temp$X_rem
rm(temp)

### Model would be fit to 'X_fit' (non-test users)
### e.g. model <- reco_model(X_fit, ...)
latent_dim <- 5
Item_Factors <- matrix(rnorm(n_items*latent_dim), ncol=n_items)

### Then it would produce user factors for 'X_train'
### (users to which the model was not fit)
User_Factors <- matrix(rnorm(nrow(X_train)*latent_dim), ncol=nrow(X_train))

### And then the metrics would be calculated
df_metrics_by_user <- calc.reco.metrics(
    X_train, X_test,
    User_Factors, Item_Factors,
    k=5,
    precision=TRUE,
    average_precision=TRUE,
    ndcg=TRUE,
    nthreads=parallel::detectCores()
)
head(df_metrics_by_user, 3)
```
```
  p_at_5 ap_at_5 ndcg_at_5
1    0.0   0.000 0.0000000
2    0.2   0.125 0.3047166
3    0.2   0.125 0.1813742
```
