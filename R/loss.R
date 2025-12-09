#' @title MSE Loss
#' @description Mean Squared Error Loss
#' @importFrom R6 R6Class
#' @export
MSELoss <- R6::R6Class("MSELoss",
 public = list(
   forward = function(pred, target) {
     # MSE = mean( (pred - target)^2 )
     diff <- pred - target
     sq_error <- diff * diff
     return(mean.Tensor(sq_error))
   },

   `__call__` = function(pred, target) {
     self$forward(pred, target)
   }
 )
)

#' @title Cross Entropy Loss
#' @description Computes -sum(target * log(pred)).
#' Expects 'pred' to be probabilities (output of Softmax) and 'target' to be One-Hot.
#' @importFrom R6 R6Class
#' @export
CrossEntropyLoss <- R6::R6Class("CrossEntropyLoss",
  public = list(
    eps = 1e-12, # Small epsilon to prevent log(0)

    forward = function(pred, target) {
      # Clip predictions to prevent log(0) -> NaN
      # We manually clamp the data
      p_safe <- pred
      p_safe$data[p_safe$data < self$eps] <- self$eps
      p_safe$data[p_safe$data > (1 - self$eps)] <- (1 - self$eps)

      # Loss = - sum(target * log(p))
      # We rely on our existing operations!

      log_p <- log(p_safe)
      weighted_log <- target * log_p
      neg_sum <- -tensor_sum(weighted_log) # Sum all errors

      # Usually we want Mean over the batch
      batch_size <- nrow(pred$data)

      # Create a scalar tensor for batch size to divide
      # (Manually creating tensor to allow division)
      b_tensor <- Tensor$new(matrix(batch_size), requires_grad=FALSE)

      return(neg_sum / b_tensor)
    },

    `__call__` = function(pred, target) {
      self$forward(pred, target)
    }
  )
)
