# Extracted from test-tensor_ops.R:73

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "deepr", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
x <- Tensor$new(c(-1, 2, -3, 4), requires_grad = TRUE)
y <- relu(x)
