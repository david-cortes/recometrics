library("testthat")
library("Matrix")
library("MatrixExtra")
library("recometrics")
context("Creating train-test splits")

test_that("Split all users", {
    m <- 1e3
    n <- 1e4
    set.seed(123)
    X <- Matrix::rsparsematrix(m, n, .01)
    
    r <- create.reco.train.test(X, split_type="all", users_test_fraction=.2)
    X_train <- r$X_train
    X_test <- r$X_test
    d <- abs((X_train + X_test) - X)
    expect_equal(sum(d@x), 0)
    
    n <- 3L
    set.seed(123)
    X <- Matrix::rsparsematrix(m, n, .01)
    
    r <- create.reco.train.test(X, split_type="all", users_test_fraction=.2)
    X_train <- r$X_train
    X_test <- r$X_test
    d <- abs((X_train + X_test) - X)
    expect_equal(sum(d@x), 0)
    expect_equal(nrow(X_train), nrow(X_test))
    expect_equal(ncol(X_train), ncol(X_test))
    expect_equal(ncol(X_train), ncol(X))
    expect_equal(nrow(X_train), nrow(X))
})

test_that("Split separate users", {
    m <- 1e3
    n <- 1e4
    set.seed(123)
    X <- Matrix::rsparsematrix(m, n, .01)
    
    r <- create.reco.train.test(X, split_type="separated", users_test_fraction=.1)
    X_train <- r$X_train
    X_test <- r$X_test
    users_test <- r$users_test
    d <- abs((X_train + X_test) - X[users_test, ])
    expect_equal(sum(d@x), 0)
    expect_equal(nrow(X_train), nrow(X_test))
    expect_equal(ncol(X_train), ncol(X_test))
    expect_equal(ncol(X_train), ncol(X))
    expect_equal(nrow(X_train), m*.1)
})

test_that("Split joined users", {
    m <- 1e3
    n <- 1e4
    set.seed(123)
    X <- Matrix::rsparsematrix(m, n, .01)
    
    r <- create.reco.train.test(X, split_type="joined", users_test_fraction=.1)
    X_test <- r$X_test
    X_train <- r$X_train[1:nrow(X_test), ]
    users_test <- r$users_test
    d <- abs((X_train + X_test) - X[users_test, ])
    expect_equal(sum(d@x), 0)
    expect_equal(nrow(X_train), nrow(X_test))
    expect_equal(ncol(X_train), ncol(X_test))
    expect_equal(ncol(X_train), ncol(X))
    expect_equal(nrow(r$X_train), nrow(X))
    expect_equal(nrow(X_test), m*.1)
})

test_that("Split with a fixed number of users", {
    m <- 10
    n <- 9
    set.seed(123)
    X <- Matrix::rsparsematrix(m, n, .5)
    r <- create.reco.train.test(X, users_test_fraction=NULL, max_test_users=2)
    X_test <- r$X_test
    X_train <- r$X_train
    X_rem <- r$X_rem
    users_test <- r$users_test
    expect_equal(length(users_test), 2L)
    expect_equal(nrow(X_test), 2L)
    expect_equal(nrow(X_train), 2L)
    expect_equal(nrow(X_rem), 8L)
})

test_that("Throw error on splits", {
    m <- 1e3
    n <- 3
    set.seed(1)
    X <- Matrix::rsparsematrix(m, n, .01)
    expect_error({r <- create.reco.train.test(X, min_pos_test=2)})
})
