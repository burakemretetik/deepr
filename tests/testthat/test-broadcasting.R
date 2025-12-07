test_that("Broadcasting Addition works (Bias Addition)", {
  # 1. Setup
  # Matrix (3 rows, 2 cols)
  mat <- Tensor$new(matrix(c(1, 2, 3, 4, 5, 6), nrow=3, ncol=2), requires_grad = TRUE)

  # Bias Vector (1 row, 2 cols)
  vec <- Tensor$new(matrix(c(10, 20), nrow=1, ncol=2), requires_grad = TRUE)

  # 2. Forward: Matrix + Vector
  # This used to crash. Now it should implicitly broadcast.
  # Row 1 adds [10, 20]
  # Row 2 adds [10, 20]
  # Row 3 adds [10, 20]
  z <- mat + vec

  # Check Forward Shape
  expect_equal(z$shape, c(3, 2))

  # Check Forward Values (Row 1: 1+10=11, 4+20=24)
  expect_equal(z$data[1, 1], 11)
  expect_equal(z$data[1, 2], 24)

  # 3. Backward
  # We feed a gradient of all 1s
  grad_seed <- matrix(1, nrow=3, ncol=2)
  grad_seed <- matrix(1, nrow=3, ncol=2)
  z$backward(grad_seed)

  # 4. Check Gradients
  # The Matrix gradient should just be 1s (Standard addition rule)
  expect_equal(as.numeric(mat$grad), rep(1, 6))

  # CRITICAL CHECK: The Vector gradient
  # Since the vector was added to 3 rows, it receives 3 gradients of 1.0.
  # So its gradient should be 3.0.
  expect_equal(vec$grad[1, 1], 3)
  expect_equal(vec$grad[1, 2], 3)
})

test_that("Broadcasting Mismatch still throws error", {
  mat <- Tensor$new(matrix(0, nrow=3, ncol=2))
  bad_vec <- Tensor$new(matrix(0, nrow=1, ncol=5)) # 5 cols vs 2 cols

  expect_error(mat + bad_vec, "Shape mismatch")
})
