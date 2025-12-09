test_that("Softmax works (Forward & Backward)", {
  # 1. Setup: Two equal inputs [0, 0]
  # Exp(0) = 1. Sum = 2. Probs should be [0.5, 0.5]
  x <- Tensor$new(matrix(c(0, 0), nrow=1), requires_grad = TRUE)

  # Forward
  probs <- softmax(x)

  expect_equal(probs$data[1,1], 0.5)
  expect_equal(probs$data[1,2], 0.5)
  expect_equal(sum(probs$data), 1.0)

  # Backward
  # If we pull the first probability up, the gradient should reflect that
  probs$backward(matrix(c(1, 0), nrow=1))

  # Math check: Softmax gradient is complex, but for equal inputs [0,0]
  # and grad_output [1,0], grad_input should be [0.25, -0.25]
  # (S_i * (1 - S_i) for i=j, -S_i * S_j for i!=j)
  expect_equal(x$grad[1,1], 0.25)
  expect_equal(x$grad[1,2], -0.25)
})

test_that("CrossEntropyLoss works", {
  # 1. Setup
  # Prediction: 90% sure it's Class A (0.9, 0.1)
  pred <- Tensor$new(matrix(c(0.9, 0.1), nrow=1), requires_grad = TRUE)
  # Target: It IS Class A (One-Hot: 1, 0)
  target <- Tensor$new(matrix(c(1, 0), nrow=1), requires_grad = FALSE)

  # 2. Loss Calculation
  # Loss = - (1 * log(0.9) + 0 * log(0.1)) = -log(0.9)
  criterion <- CrossEntropyLoss$new()
  loss <- criterion$forward(pred, target)

  # Expected: -log(0.9) approx 0.10536
  expected_val <- -log(0.9)
  expect_lt(abs(loss$data[1,1] - expected_val), 1e-5)

  # 3. Backward
  loss$backward()

  # Gradient should flow back to 'pred'
  # d(-log(p))/dp = -1/p
  # So grad for p1 should be -1/0.9 approx -1.11 (scaled by batch size 1)
  expect_lt(abs(pred$grad[1,1] - (-1/0.9)), 1e-4)
})

test_that("Integration: Model learns to classify", {
  # A tiny 3-class classification problem
  # We want the model to learn that Input [1,1] -> Class 1 [1,0,0]

  # 1. Data
  x <- Tensor$new(matrix(c(1, 1), nrow=1), requires_grad = FALSE)
  y <- Tensor$new(matrix(c(1, 0, 0), nrow=1), requires_grad = FALSE) # Class 1

  # 2. Model (2 inputs -> 3 classes)
  # Use Kaiming init
  l1 <- Linear$new(2, 3, init_type = "kaiming")

  # 3. Optim & Loss
  optim <- SGD$new(l1$parameters(), lr = 0.1)
  crit <- CrossEntropyLoss$new()

  initial_loss <- crit$forward(softmax(l1$forward(x)), y)$data[1,1]

  # 4. Train for 10 steps
  for (i in 1:10) {
    optim$zero_grad()

    logits <- l1$forward(x)
    probs <- softmax(logits)
    loss <- crit$forward(probs, y)

    loss$backward()
    optim$step()
  }

  final_loss <- loss$data[1,1]

  # Loss should decrease
  expect_lt(final_loss, initial_loss)

  # Prediction for Class 1 should be the highest
  final_probs <- softmax(l1$forward(x))$data
  expect_gt(final_probs[1,1], final_probs[1,2])
  expect_gt(final_probs[1,1], final_probs[1,3])
})
