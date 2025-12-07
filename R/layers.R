#' @title Abstract Layer Class
#' @description Base class for all neural network layers.
#' @importFrom R6 R6Class
#' @export
Layer <- R6::R6Class("Layer",
  public = list(
    #' @description Retrieve parameters (weights/biases)
    parameters = function() {
     return(list())
    },

    #' @description Forward pass (Abstract)
    forward = function(x) {
     stop("Method 'forward' must be implemented by the subclass.")
    },

    #' @description Call the layer like a function
    `__call__` = function(x) {
     self$forward(x)
    }
  )
)

#' @title Linear (Dense) Layer
#' @description A fully connected layer: Output = Input %*% W + b
#' @inherit Layer
#' @export
Linear <- R6::R6Class("Linear",
  inherit = Layer,
  public = list(
    W = NULL, # Weights
    b = NULL, # Bias

    #' @description Initialize weights and biases
    #' @param in_features Integer. Input dimension.
    #' @param out_features Integer. Output dimension.
    initialize = function(in_features, out_features) {

      # 1. Initialize Weights (Xavier-like initialization)
      # Random values scaled by 0.1 to prevent exploding gradients
      w_data <- matrix(
        rnorm(in_features * out_features) * 0.1,
        nrow = in_features,
        ncol = out_features
      )
      self$W <- Tensor$new(data = w_data, requires_grad = TRUE)

      # 2. Initialize Bias (Zeros)
      # Shape is (1, out_features) for broadcasting
      b_data <- matrix(0, nrow = 1, ncol = out_features)
      self$b <- Tensor$new(data = b_data, requires_grad = TRUE)
    },

    #' @description Return the parameters for the optimizer
    parameters = function() {
      return(list(self$W, self$b))
    },

    #' @description Forward pass
    forward = function(x) {
      # 1. Matrix Multiplication
      out <- x %*% self$W

      # 2. Add Bias (Broadcasting handles this automatically now!)
      return(out + self$b)
    }
  )
)

#' @title ReLU Layer
#' @description Wraps the relu() function into a Layer object.
#' @inherit Layer
#' @export
ReLU <- R6::R6Class("ReLU",
                    inherit = Layer,
                    public = list(
                      initialize = function() {},

                      forward = function(x) {
                        # Calls the deepr::relu function we wrote in ops.R
                        return(relu(x))
                      }
                    )
)

#' @title Sequential Container
#' @description Stacks layers sequentially.
#' @inherit Layer
#' @export
Sequential <- R6::R6Class("Sequential",
  inherit = Layer,
  public = list(
    layers = list(),

    #' @description Initialize with a list of layers
    initialize = function(...) {
      self$layers <- list(...)
    },

    #' @description Sequential Forward Pass
    forward = function(x) {
      out <- x
      for (layer in self$layers) {
        out <- layer$forward(out)
      }
      return(out)
    },

    #' @description Gather parameters from ALL layers
    parameters = function() {
      params <- list()
      for (layer in self$layers) {
        params <- c(params, layer$parameters())
      }
      return(params)
    }
  )
)
