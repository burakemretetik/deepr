#' @title Abstract Optimizer Class
#' @importFrom R6 R6Class
#' @export
Optimizer <- R6::R6Class("Optimizer",
 public = list(
   parameters = list(),
   lr = 0.01,

   #' @description Initialize optimizer
   #' @param params List of Tensors to optimize.
   #' @param lr Learning rate.
   initialize = function(params, lr = 0.01) {
     self$parameters <- params
     self$lr <- lr
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
#' @description Updates parameters using: p = p - lr * grad
#' @inherit Optimizer
#' @export
SGD <- R6::R6Class("SGD",
 inherit = Optimizer,
 public = list(
   step = function() {
     for (p in self$parameters) {
       if (p$requires_grad) {
         # Standard SGD update: Data = Data - (LR * Grad)
         # Note: We modify data in-place without creating a new graph node
         p$data <- p$data - (self$lr * p$grad)
       }
     }
   }
 )
)
