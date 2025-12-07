test_that("Linear Layer initialization works", {
  layer <- Linear$new(3, 2)

  # Check Weights shape: (3 inputs, 2 outputs)
  expect_equal(layer$W$shape, c(3, 2))

  # Check Bias shape: (1, 2)
  expect_equal(layer$b$shape, c(1, 2))

  # Check gradients are enabled
  expect_true(layer$W$requires_grad)
  expect_true(layer$b$requires_grad)
})

test_that("Linear Layer forward/backward works with Batches", {
  # 1. Setup Layer (3 inputs -> 2 outputs)
  layer <- Linear$new(3, 2)

  # 2. Create Batch Input (Batch size of 10)
  input <- Tensor$new(matrix(rnorm(30), nrow=10, ncol=3), requires_grad = FALSE)

  # 3. Forward
  output <- layer$forward(input)

  # Check Output Shape: Should be (10, 2)
  expect_equal(output$shape, c(10, 2))

  # 4. Backward
  # Pass a gradient of all 1s
  grad_seed <- matrix(1, nrow=10, ncol=2)
  output$backward(grad_seed)

  # Check Bias Gradient
  # Since we summed 10 rows of gradients (1.0 each), the bias grad should be 10.0
  expect_equal(layer$b$grad[1,1], 10)
})
