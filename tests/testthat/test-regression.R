test_that("Integration: Model learns Linear Regression", {
  # --- 1. The True Function ---
  # y = 2*x1 - 3*x2 + 1 (Bias)
  true_w <- c(2, -3)
  true_b <- 1

  # --- 2. Generate Data ---
  # Batch size 20, 2 features
  set.seed(42) # For reproducibility
  X_raw <- matrix(rnorm(40), nrow=20, ncol=2)

  # Calculate True Y
  Y_raw <- (X_raw[,1] * true_w[1]) + (X_raw[,2] * true_w[2]) + true_b
  Y_raw <- matrix(Y_raw, ncol=1) # Ensure shape (20, 1)

  # Wrap in Tensors
  x <- Tensor$new(X_raw, requires_grad = FALSE)
  y <- Tensor$new(Y_raw, requires_grad = FALSE)

  # --- 3. Define Model ---
  # Linear layer: 2 inputs -> 1 output
  # We use 'kaiming' init (though for linear regression simple random is fine)
  model <- Linear$new(2, 1, init_type = "kaiming")

  # --- 4. Setup Training ---
  # Use a slightly higher learning rate for this simple problem
  optim <- SGD$new(model$parameters(), lr = 0.1)
  criterion <- MSELoss$new()

  initial_loss <- criterion$forward(model$forward(x), y)$data[1,1]

  # --- 5. Training Loop ---
  # It should converge very quickly (linear problem)
  for (epoch in 1:50) {
    optim$zero_grad()

    pred <- model$forward(x)
    loss <- criterion$forward(pred, y)

    loss$backward()
    optim$step()
  }

  final_loss <- loss$data[1,1]

  # --- 6. Assertions ---

  # A) Loss should be near zero (it's a perfect linear problem)
  expect_lt(final_loss, 0.01)
  expect_lt(final_loss, initial_loss)

  # B) Weights should be close to [2, -3]
  learned_w <- as.numeric(model$W$data)
  expect_lt(abs(learned_w[1] - 2), 0.1) # Check x1 weight
  expect_lt(abs(learned_w[2] - (-3)), 0.1) # Check x2 weight

  # C) Bias should be close to 1
  learned_b <- as.numeric(model$b$data)
  expect_lt(abs(learned_b - 1), 0.1)
})
