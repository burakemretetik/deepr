#' @title Tensor Class
#' @description A class to represent a Tensor (multi-dimensional array) with autograd support.
#' @importFrom R6 R6Class
#' @export
Tensor <- R6::R6Class("Tensor",
  public = list(
    data = NULL,
    grad = NULL,
    requires_grad = FALSE,
    shape = NULL,

    #' @description Create a new Tensor
    #' @param data A numeric vector or matrix.
    #' @param requires_grad Logical. Should gradients be computed for this tensor?
    initialize = function(data, requires_grad = FALSE) {

      # For V1, let's force everything to be a matrix for simplicity
      if (is.vector(data)) {
        data <- matrix(data, ncol = 1)
      } else if (!is.matrix(data)) {
        stop("For V1, data must be a vector or a matrix.")
      }

      self$data <- data
      self$shape <- dim(data)
      self$requires_grad <- requires_grad

      # Initialize gradient as zero if required
      if (self$requires_grad) {
        self$zero_grad()
      }
    },

    #' @description Zero out the gradients
    zero_grad = function() {
      self$grad <- matrix(0, nrow = self$shape[1], ncol = self$shape[2])
    },

    #' @description Print the tensor nicely
    print = function() {
      cat("<Tensor>\n")
      print(self$data)
      if (self$requires_grad) {
        cat("   [Requires Grad: TRUE]\n")
      }
    }
  )
)
