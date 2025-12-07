# Extracted from test-tensor_ops.R:66

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "deepr", path = "..")
attach(test_env, warn.conflicts = FALSE)

# test -------------------------------------------------------------------------
x <- Tensor$new(c(1, 2))
y <- Tensor$new(c(1, 2, 3))
expect_error(x + y, "Shape Mismatch")
