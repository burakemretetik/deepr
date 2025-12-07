test_that("Tensor Addition works (Forward & Backward)", {
  # Setup
  x <- Tensor$new(c(1, 2), requires_grad = TRUE)
  y <- Tensor$new(c(3, 4), requires_grad = TRUE)

  # Forward
  z <- x + y
  expect_equal(z$data, matrix(c(4, 6), ncol=1))

  # Backward
  z$backward()

  # Gradients should be 1.0 for addition
  expect_equal(x$grad, matrix(c(1, 1), ncol=1))
  expect_equal(y$grad, matrix(c(1, 1), ncol=1))
})

test_that("Tensor Multiplication works (Forward & Backward)", {
  # Setup: z = x * y
  # x = [3], y = [4] -> z = 12
  x <- Tensor$new(c(3), requires_grad = TRUE)
  y <- Tensor$new(c(4), requires_grad = TRUE)

  # Forward
  z <- x * y
  expect_equal(z$data[1,1], 12)

  # Backward
  # dz/dx = y (4), dz/dy = x (3)
  z$backward()

  expect_equal(x$grad[1,1], 4)
  expect_equal(y$grad[1,1], 3)
})

test_that("Matrix Multiplication works (Forward & Backward)", {
  # Setup: 1x3 matrix multiplied by 3x1 vector
  # x = [1, 2, 3]
  # w = [0.5, 0.5, 0.5]T
  x_data <- matrix(c(1, 2, 3), nrow=1)
  w_data <- matrix(c(0.5, 0.5, 0.5), nrow=3)

  x <- Tensor$new(x_data, requires_grad = TRUE)
  w <- Tensor$new(w_data, requires_grad = TRUE)

  # Forward
  z <- x %*% w
  # 1*0.5 + 2*0.5 + 3*0.5 = 0.5 + 1.0 + 1.5 = 3.0
  expect_equal(z$data[1,1], 3.0)

  # Backward
  z$backward()

  # Grad X should be W^T -> [0.5, 0.5, 0.5]
  expect_equal(as.numeric(x$grad), c(0.5, 0.5, 0.5))

  # Grad W should be X^T -> [1, 2, 3]
  expect_equal(as.numeric(w$grad), c(1, 2, 3))
})

test_that("Shape Mismatch throws error", {
  x <- Tensor$new(c(1, 2))      # Shape (2,1)
  y <- Tensor$new(c(1, 2, 3))   # Shape (3,1)

  # Should fail immediately due to our safety check
  expect_error(x + y, "Shape mismatch")
})
test_that("ReLU works", {
  # Setup: Mixed positive and negative values
  x <- Tensor$new(c(-1, 2, -3, 4), requires_grad = TRUE)

  # Forward
  y <- relu(x)
  # -1 -> 0, 2 -> 2, -3 -> 0, 4 -> 4
  expect_equal(as.numeric(y$data), c(0, 2, 0, 4))

  # Backward
  y$backward()

  # Gradient should be 1 where input was >0, and 0 elsewhere
  expect_equal(as.numeric(x$grad), c(0, 1, 0, 1))
})

test_that("ReLU works", {
  # Setup: Mixed positive and negative values
  x <- Tensor$new(c(-1, 2, -3, 4), requires_grad = TRUE)

  # Forward
  y <- relu(x)
  # -1 -> 0, 2 -> 2, -3 -> 0, 4 -> 4
  expect_equal(as.numeric(y$data), c(0, 2, 0, 4))

  # Backward
  y$backward()

  # Gradient should be 1 where input was >0, and 0 elsewhere
  expect_equal(as.numeric(x$grad), c(0, 1, 0, 1))
})
