library("testthat")
library("Matrix")
library("MatrixExtra")
library("recometrics")
context("Calculating AUC")

test_that("Random ROC-AUC", {
    m <- 100
    n <- 20
    k <- 3
    set.seed(1)
    A <- float::flrnorm(k, m)
    B <- float::flrnorm(k, n)
    X_test <- Matrix::rsparsematrix(m, n, .1)
    X_train <- MatrixExtra::as.csr.matrix(matrix(0, nrow=m, ncol=n))
    r <- calc.reco.metrics(X_train, X_test, A, B,
                           precision=FALSE, average_precision=FALSE,
                           ndcg=FALSE, roc_auc=TRUE, nthreads=1)
    expect_equal(mean(as.numeric(r$roc_auc), na.rm=TRUE), 0.5, tolerance=0.03)
})

test_that("Perfect ROC-AUC and PR-AUC", {
    n <- 100
    npos <- 20
    nneg <- n - npos
    k <- 2
    set.seed(1)
    pos <- sample(n, npos, replace=FALSE)
    X_test <- as.csr.matrix(sparseVector(i=pos, x=1, length=n))
    X_train <- as.csr.matrix(matrix(0, nrow=1, ncol=n))
    A <- matrix(1, nrow=k, ncol=1)
    B <- matrix(-100, nrow=k, ncol=n)
    B[, pos] <- 100
    B <- B + matrix(rnorm(nrow(B)*ncol(B)), nrow=nrow(B), ncol=ncol(B))
    r <- calc.reco.metrics(X_train, X_test, A, B, k=10,
                           precision=FALSE, average_precision=FALSE,
                           ndcg=FALSE, roc_auc=TRUE, pr_auc=TRUE,
                           as_df=FALSE, nthreads=1)
    expect_equal(r$roc_auc, 1)
    expect_equal(r$pr_auc, 1)
})

test_that("Zero ROC-AUC", {
    n <- 100
    npos <- 20
    nneg <- n - npos
    k <- 2
    set.seed(1)
    pos <- sample(n, npos, replace=FALSE)
    X_test <- as.csr.matrix(sparseVector(i=pos, x=1, length=n))
    X_train <- as.csr.matrix(matrix(0, nrow=1, ncol=n))
    A <- matrix(1, nrow=k, ncol=1)
    B <- matrix(100, nrow=k, ncol=n)
    B[, pos] <- -100
    B <- B + matrix(rnorm(nrow(B)*ncol(B)), nrow=nrow(B), ncol=ncol(B))
    r <- calc.reco.metrics(X_train, X_test, A, B, k=10,
                           precision=FALSE, average_precision=FALSE,
                           ndcg=FALSE, roc_auc=TRUE,
                           as_df=FALSE, nthreads=1)
    expect_equal(r$roc_auc, 0)
})
