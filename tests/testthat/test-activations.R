test_that("ReLU works (Forward & Backward)", {
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

test_that("Sigmoid works (Forward & Backward)", {
  # Setup: Input 0
  # Sigmoid(0) = 0.5
  # Sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
  x <- Tensor$new(c(0), requires_grad = TRUE)

  # Forward
  s <- sigmoid(x)
  expect_equal(s$data[1,1], 0.5)

  # Backward
  s$backward()
  expect_equal(x$grad[1,1], 0.25)

  # Large positive number (should be close to 1, grad close to 0)
  x2 <- Tensor$new(c(10), requires_grad = TRUE)
  s2 <- sigmoid(x2)
  expect_lt(abs(s2$data[1,1] - 1), 1e-4) # Should be ~0.9999

  s2$backward()
  expect_lt(x2$grad[1,1], 1e-4) # Grad should be tiny
})

test_that("Tanh works (Forward & Backward)", {
  # Setup: Input 0
  # Tanh(0) = 0
  # Tanh'(0) = 1 - 0^2 = 1
  x <- Tensor$new(c(0), requires_grad = TRUE)

  # Forward
  t <- tanh_act(x)
  expect_equal(t$data[1,1], 0)

  # Backward
  t$backward()
  expect_equal(x$grad[1,1], 1)

  # Large negative number (should be close to -1, grad close to 0)
  x2 <- Tensor$new(c(-10), requires_grad = TRUE)
  t2 <- tanh_act(x2)
  expect_lt(abs(t2$data[1,1] - (-1)), 1e-4) # Should be ~ -0.9999

  t2$backward()
  expect_lt(x2$grad[1,1], 1e-4) # Grad should be tiny
})
