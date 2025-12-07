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

    # Graph tracking
    id = NULL,          # Unique ID for topological sort
    creation_op = NULL, # Name of operation (add, mul, mm)
    creators = NULL,    # List of parent Tensors
    aux_data = NULL,

    #' @description Create a new Tensor
    initialize = function(data, requires_grad = FALSE, creators = NULL, creation_op = NULL, aux_data = NULL) {

      # Ensure data is a matrix for V1 simplicity
      if (is.vector(data)) {
        data <- matrix(data, ncol = 1)
      } else if (!is.matrix(data)) {
        stop("For V1, data must be a vector or a matrix.")
      }

      self$data <- data
      self$shape <- dim(data)
      self$requires_grad <- requires_grad

      # Graph History
      self$creators <- creators
      self$creation_op <- creation_op
      self$aux_data <- aux_data   # <--- THIS IS THE MISSING LINE
      self$id <- paste0(Sys.time(), "_", runif(1))

      if (self$requires_grad) {
        self$zero_grad()
      }
    },

    #' @description Zero out the gradients
    zero_grad = function() {
      self$grad <- matrix(0, nrow = self$shape[1], ncol = self$shape[2])
    },

    #' @description Trigger backpropagation
    backward = function(grad = NULL) {
      if (!self$requires_grad) stop("Called backward() on a Tensor that doesn't require gradients.")

      # 1. Seed the gradient (default to 1.0)
      if (is.null(grad)) {
        self$grad <- matrix(1, nrow = self$shape[1], ncol = self$shape[2])
      } else {
        self$grad <- grad
      }

      # 2. Topological Sort
      topo <- list()
      visited <- new.env()

      build_topo <- function(node) {
        if (exists(node$id, envir = visited)) return()
        assign(node$id, TRUE, envir = visited)

        if (!is.null(node$creators)) {
          for (parent in node$creators) {
            build_topo(parent)
          }
        }
        topo <<- c(topo, list(node))
      }

      build_topo(self)

      # 3. Process Graph in Reverse
      for (node in rev(topo)) {
        if (!is.null(node$creation_op)) {

          inputs <- node$creators
          grad_output <- node$grad

          # --- Dispatcher ---

          # A) Addition
          if (node$creation_op == "add") {

            # Function to handle gradient accumulation with unbroadcasting
            accumulate_grad <- function(input_tensor, grad) {
              if (input_tensor$requires_grad) {
                current_grad <- grad

                # Check for broadcast: If input is (1, M) but grad is (N, M)
                if (input_tensor$shape[1] == 1 && nrow(current_grad) > 1) {
                  # We must SUM the gradients across the batch (dim 1)
                  # colSums returns vector, convert back to (1, M)
                  current_grad <- matrix(colSums(current_grad), nrow=1)
                }

                input_tensor$grad <- input_tensor$grad + current_grad
              }
            }

            accumulate_grad(inputs[[1]], grad_output)
            accumulate_grad(inputs[[2]], grad_output)
          }

          # B) Multiplication (Element-wise)
          if (node$creation_op == "mul") {
            # Swap parents: grad_x = grad_z * y
            if (inputs[[1]]$requires_grad) inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output * inputs[[2]]$data)
            if (inputs[[2]]$requires_grad) inputs[[2]]$grad <- inputs[[2]]$grad + (grad_output * inputs[[1]]$data)
          }

          # C) Matrix Multiplication
          if (node$creation_op == "mm") {
            # Z = X %*% Y
            # grad_X = grad_Z %*% t(Y)
            if (inputs[[1]]$requires_grad) {
              inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output %*% t(inputs[[2]]$data))
            }
            # grad_Y = t(X) %*% grad_Z
            if (inputs[[2]]$requires_grad) {
              inputs[[2]]$grad <- inputs[[2]]$grad + (t(inputs[[1]]$data) %*% grad_output)
            }
          }

          # D) ReLU
          if (node$creation_op == "relu") {
            # Gradient is passed through only if input was > 0
            # We retrieved the 'mask' we saved earlier in aux_data
            if (inputs[[1]]$requires_grad) {
              mask <- node$aux_data
              inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output * mask)
            }
          }

          # E) Sum
          if (node$creation_op == "sum") {
            if (inputs[[1]]$requires_grad) {
              orig_shape <- node$aux_data$orig_shape
              dim_sum <- node$aux_data$dim

              # 1. Expand scalar gradient back to original matrix size
              # If we summed everything (scalar output)
              if (is.null(dim_sum)) {
                # Gradient is 1x1, we need to repeat it to fill 'orig_shape'
                # grad_input = grad_output * ones(orig_shape)
                expanded_grad <- matrix(grad_output[1], nrow=orig_shape[1], ncol=orig_shape[2])
                inputs[[1]]$grad <- inputs[[1]]$grad + expanded_grad
              }
              # 2. Expand along dimension
              else if (dim_sum == 1) {
                # We did colSums. Output is (1, cols). Input was (rows, cols).
                # We need to repeat the row vector 'rows' times.
                # R repeats vectors column-wise by default, so we transpose logic slightly or use matrix mult
                grad_vec <- as.numeric(grad_output)
                # Create matrix by repeating row
                expanded_grad <- matrix(grad_vec, nrow=orig_shape[1], ncol=orig_shape[2], byrow=TRUE)
                inputs[[1]]$grad <- inputs[[1]]$grad + expanded_grad
              }
              else if (dim_sum == 2) {
                # We did rowSums. Output is (rows, 1). Input was (rows, cols).
                grad_vec <- as.numeric(grad_output)
                expanded_grad <- matrix(grad_vec, nrow=orig_shape[1], ncol=orig_shape[2], byrow=FALSE)
                inputs[[1]]$grad <- inputs[[1]]$grad + expanded_grad
              }
            }
          }

          # F) Negation (-x)
          if (node$creation_op == "neg") {
            if (inputs[[1]]$requires_grad) {
              inputs[[1]]$grad <- inputs[[1]]$grad - grad_output
            }
          }

          # G) Division (x / y)
          if (node$creation_op == "div") {
            x <- inputs[[1]]
            y <- inputs[[2]]

            # d(x/y)/dx = 1/y * grad
            if (x$requires_grad) {
              inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output / y$data)
            }

            # d(x/y)/dy = -x/y^2 * grad
            if (y$requires_grad) {
              inputs[[2]]$grad <- inputs[[2]]$grad + (-x$data / (y$data^2) * grad_output)
            }
          }

          # H) Log (log(x))
          if (node$creation_op == "log") {
            # d(log(x))/dx = 1/x * grad
            if (inputs[[1]]$requires_grad) {
              inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output / inputs[[1]]$data)
            }
          }

          # I) Exp (exp(x))
          if (node$creation_op == "exp") {
            # d(exp(x))/dx = exp(x) * grad = output * grad
            if (inputs[[1]]$requires_grad) {
              # We can use the node's own data (output) since exp(x) = output
              inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output * node$data)
            }
          }

          # J) Mean (mean(x))
          if (node$creation_op == "mean") {
            if (inputs[[1]]$requires_grad) {
              n <- node$aux_data
              inputs[[1]]$grad <- inputs[[1]]$grad + (as.numeric(grad_output) / n)
            }
          }

          # K) Subtraction (x - y)
          if (node$creation_op == "sub") {

            # Helper for unbroadcasting (copy this if you don't have it globally)
            accumulate_grad <- function(input_tensor, grad) {
              if (input_tensor$requires_grad) {
                curr <- grad
                if (input_tensor$shape[1] == 1 && nrow(curr) > 1) {
                  curr <- matrix(colSums(curr), nrow=1)
                }
                input_tensor$grad <- input_tensor$grad + curr
              }
            }

            # 1. Gradient for X (Positive)
            accumulate_grad(inputs[[1]], grad_output)

            # 2. Gradient for Y (Negative)
            accumulate_grad(inputs[[2]], -grad_output)
          }
        }
      }
    },

    #' @description Print the tensor nicely
    print = function() {
      cat("<Tensor>\n")
      print(self$data)
      if (self$requires_grad) {
        cat("   [Requires Grad: TRUE]\n")
        if (!is.null(self$creation_op)) {
          cat(paste("   [Created by:", self$creation_op, "]\n"))
        }
      }
    }
  )
)
