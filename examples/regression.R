# examples/regression.R
library(deepr)

# --- 1. Generate Synthetic Data ---
# Function: y = 3x + 2 + noise
set.seed(42)
N <- 100
X_raw <- matrix(runif(N, min = -10, max = 10), nrow = N, ncol = 1)
Y_raw <- 3 * X_raw + 2 + rnorm(N, sd = 2) # Add some noise

# Convert to Tensors
x <- Tensor$new(X_raw, requires_grad = FALSE)
y <- Tensor$new(Y_raw, requires_grad = FALSE)

# --- 2. Define the Model (Resetting it) ---
# We need in_features = 1 because X_raw has 1 column
model <- Sequential$new(
  Linear$new(in_features = 1, out_features = 1, init_type = "kaiming")
)

# Re-initialize the optimizer to track the NEW model parameters
optim <- SGD$new(model$parameters(), lr = 0.001)

# --- 3. Optimizer & Loss ---
# SGD is perfect for simple regression
optim <- SGD$new(model$parameters(), lr = 0.001)
loss_fn <- MSELoss$new()

# --- 4. Training Loop ---
cat("Training Regression Model...\n")
epochs <- 1000

for (i in 1:epochs) {
  # A) Forward Pass
  preds <- model$forward(x)
  loss <- loss_fn$forward(preds, y)

  # B) Backward Pass
  optim$zero_grad()
  loss$backward()

  # C) Update
  optim$step()

  if (i %% 100 == 0) {
    cat(sprintf("Epoch %d | Loss: %.4f\n", i, loss$data[1]))
  }
}

# --- 5. Inspect Results ---
w_learned <- model$layers[[1]]$W$data[1]
b_learned <- model$layers[[1]]$b$data[1]

cat("\n--- Final Results ---\n")
cat(sprintf("True Function:    y = 3.00x + 2.00\n"))
cat(sprintf("Learned Function: y = %.2fx + %.2f\n", w_learned, b_learned))

# Optional: Plot if running in RStudio
plot(X_raw, Y_raw, main="Deepr Regression", col="blue", pch=19)
lines(X_raw, preds$data, col="red", lwd=2)
legend("topleft", legend=c("Actual", "Predicted"), col=c("blue", "red"), pch=c(19, NA), lwd=c(NA, 2))
