library("testthat")
library("Matrix")
library("MatrixExtra")
library("recometrics")
context("Calculating NDCG")

test_that("Invalid cases", {
    X_train <- as.csr.matrix(matrix(0, ncol=10))
    set.seed(1)
    X_test <- rsparsematrix(1, 10, .4)
    
    set.seed(1)
    A <- matrix(1, nrow=5)
    
    check.na.result <- function(B) {
        res <- calc.reco.metrics(X_train, X_test, A, B, k=5L,
                                 precision=FALSE, average_precision=FALSE,
                                 ndcg=TRUE, nthreads=1)
        expect_true(is.na(res$ndcg_at_5))
    }
    
    check.na.result(matrix(0, nrow=5, ncol=10))
    check.na.result(matrix(1, nrow=5, ncol=10))
    check.na.result(matrix(NA_real_, nrow=5, ncol=10))
    check.na.result(matrix(Inf, nrow=5, ncol=10))
    
    set.seed(1)
    B <- matrix(rnorm(5*10), nrow=5, ncol=10)
    B[1,1] <- NA_real_
    B[3,3] <- NA_real_
    check.na.result(B)
    B[1,1] <- -Inf
    B[3,3] <- Inf
    check.na.result(B)
})

test_that("Some negative values", {
    X_train <- as.csr.matrix(matrix(0, ncol=10))
    X_test <- new("dgRMatrix",
                  p=c(0L, 5L),
                  j=c(3L, 4L, 6L, 8L, 10L) - 1L,
                  x=c(1,  2,  -3, 4,  5),
                  Dim=c(1L, 10L))
    A <- matrix(1, nrow=1)
    B <- matrix(c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0), nrow=1)
    
    res <- calc.reco.metrics(X_train, X_test, A, B, k=5L,
                             precision=FALSE, average_precision=FALSE,
                             ndcg=TRUE, nthreads=1)
    expect_true(is.na(res$ndcg_at_5))
    
    B <- matrix(c(0, 0, 3, 4, 0, 6, 0, 8, 9, 10), nrow=1)
    res <- calc.reco.metrics(X_train, X_test, A, B, k=5L,
                             precision=FALSE, average_precision=FALSE,
                             ndcg=TRUE, nthreads=1)
    expect_true(res > 0)
    
    B <- matrix(c(0, 0, 3, 4, 0, 600, 0, 8, 9, 10), nrow=1)
    X_test@x[3L] <- -300
    res <- calc.reco.metrics(X_train, X_test, A, B, k=5L,
                             precision=FALSE, average_precision=FALSE,
                             ndcg=TRUE, nthreads=1)
    expect_true(res < 0)
    
    
    B <- matrix(c(0, 0, 3, 4, 0, -6, 0, 8, 9, 10), nrow=1)
    res <- calc.reco.metrics(X_train, X_test, A, B, k=5L,
                             precision=FALSE, average_precision=FALSE,
                             ndcg=TRUE, nthreads=1)
    expect_true(res > 0)
})

test_that("All negative values", {
    X_train <- as.csr.matrix(matrix(0, ncol=10))
    set.seed(1)
    X_test <- rsparsematrix(1, 10, .4)
    X_test@x <- -abs(X_test@x)
    
    set.seed(1)
    A <- matrix(rnorm(5), nrow=5)
    B <- matrix(rnorm(5*10), nrow=5)
    
    res <- calc.reco.metrics(X_train, X_test, A, B,
                             precision=FALSE, average_precision=FALSE,
                             ndcg=TRUE, nthreads=1)
    expect_equal(nrow(res), 1L)
    expect_true(is.na(res$ndcg_at_5))
})

test_that("All zeros", {
    X_train <- as.csr.matrix(matrix(0, ncol=10))
    set.seed(1)
    X_test <- rsparsematrix(1, 10, .4)
    X_test@x <- numeric(length(X_test))
    
    set.seed(1)
    A <- matrix(rnorm(5), nrow=5)
    B <- matrix(rnorm(5*10), nrow=5)
    
    res <- calc.reco.metrics(X_train, X_test, A, B,
                             precision=FALSE, average_precision=FALSE,
                             ndcg=TRUE, nthreads=1)
    expect_equal(nrow(res), 1L)
    expect_true(is.na(res$ndcg_at_5))
})

test_that("Fewer items than k", {
    X_train <- as.csr.matrix(matrix(0, ncol=10))
    X_test <- new("dgRMatrix",
                  p=c(0L, 3L),
                  j=c(3L, 4L, 8L) - 1L,
                  x=c(1,  2,  3),
                  Dim=c(1L, 10L))
    A <- matrix(1, nrow=1)
    B <- matrix(c(0, 0, 3, 4, 0, 0, 0, 0, 2, 1), nrow=1)
    
    res5 <- calc.reco.metrics(X_train, X_test, A, B, k=5L,
                              precision=FALSE, average_precision=FALSE,
                              ndcg=TRUE, nthreads=1)
    res3 <- calc.reco.metrics(X_train, X_test, A, B, k=3L,
                              precision=FALSE, average_precision=FALSE,
                              ndcg=TRUE, nthreads=1)
    expect_equal(res3[,1], res5[,1])
})
