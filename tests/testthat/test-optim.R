test_that("Adam optimizer updates parameters", {
  # Setup: Simple quadratic bowl y = x^2
  # Start at x = 10
  x <- Tensor$new(matrix(c(10)), requires_grad = TRUE)
  optim <- Adam$new(list(x), lr = 1.0)

  # Before step
  expect_equal(x$data[1,1], 10)

  # Step 1
  optim$zero_grad()
  loss <- x * x
  loss$backward()
  optim$step()

  # After step 1, x should have moved towards 0
  # Value should be less than 10
  expect_lt(x$data[1,1], 10)

  # Check that state was created
  expect_false(is.null(optim$state[[x$id]]))
  expect_true(optim$t == 1)
})
