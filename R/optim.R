#' @title Abstract Optimizer Class
#' @importFrom R6 R6Class
#' @export
Optimizer <- R6::R6Class("Optimizer",
   public = list(
     parameters = list(),
     lr = 0.01,
     state = list(), # Memory storage for optimizers like Adam
     t = 0,          # Time step counter

     #' @description Initialize optimizer
     #' @param params List of Tensors to optimize.
     #' @param lr Learning rate.
     initialize = function(params, lr = 0.01) {
       self$parameters <- params
       self$lr <- lr
       self$state <- list()
       self$t <- 0
     },

     #' @description Zero out gradients for all parameters
     zero_grad = function() {
       for (p in self$parameters) {
         p$zero_grad()
       }
     },

     #' @description Update parameters (Abstract)
     step = function() {
       stop("Method 'step' must be implemented by subclass.")
     }
   )
)

#' @title Stochastic Gradient Descent (SGD)
#' @inherit Optimizer
#' @export
SGD <- R6::R6Class("SGD",
  inherit = Optimizer,
  public = list(
    step = function() {
      for (p in self$parameters) {
        if (p$requires_grad) {
           # Standard SGD update: Data = Data - (LR * Grad)
           p$data <- p$data - (self$lr * p$grad)
        }
      }
    }
  )
)

#' @title Adam Optimizer
#' @description Adaptive Moment Estimation.
#' @inherit Optimizer
#' @export
Adam <- R6::R6Class("Adam",
  inherit = Optimizer,
  public = list(
    beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-8,

    #' @description Initialize Adam
    initialize = function(params, lr = 0.001, beta1=0.9, beta2=0.999, eps=1e-8) {
      super$initialize(params, lr)
      self$beta1 <- beta1
      self$beta2 <- beta2
      self$eps <- eps
    },

    step = function() {
      self$t <- self$t + 1

      for (p in self$parameters) {
        if (p$requires_grad) {

          # 1. Initialize State if missing
          # We use the Tensor's unique ID as the key
          id <- p$id
          if (is.null(self$state[[id]])) {
            self$state[[id]] <- list(
              m = matrix(0, nrow=p$shape[1], ncol=p$shape[2]), # Momentum
              v = matrix(0, nrow=p$shape[1], ncol=p$shape[2])  # Velocity
            )
          }

          # 2. Retrieve state
          m <- self$state[[id]]$m
          v <- self$state[[id]]$v
          g <- p$grad

          # 3. Adam Math
          # Update biased first moment estimate
          m <- self$beta1 * m + (1 - self$beta1) * g

          # Update biased second raw moment estimate
          v <- self$beta2 * v + (1 - self$beta2) * (g * g)

          # Compute bias-corrected first moment estimate
          m_hat <- m / (1 - (self$beta1 ^ self$t))

          # Compute bias-corrected second raw moment estimate
          v_hat <- v / (1 - (self$beta2 ^ self$t))

          # 4. Update Parameters
          # p = p - lr * m_hat / (sqrt(v_hat) + eps)
          p$data <- p$data - self$lr * m_hat / (sqrt(v_hat) + self$eps)

          # 5. Save state back
          self$state[[id]]$m <- m
          self$state[[id]]$v <- v
        }
      }
    }
  )
)
