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
    aux_data = NULL,    # Remembers which data points are passed ReLU. (T/F)

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

          # --- Helper Function: Un-Broadcasting ---
          # Defined once, used by both Add and Sub
          accumulate_grad <- function(input_tensor, grad) {
            if (input_tensor$requires_grad) {
              current_grad <- grad
              # Check for broadcast: If input is (1, M) but grad is (N, M)
              if (input_tensor$shape[1] == 1 && nrow(current_grad) > 1) {
                # Sum gradients across the batch (dim 1)
                current_grad <- matrix(colSums(current_grad), nrow=1)
              }
              input_tensor$grad <- input_tensor$grad + current_grad
            }
          }

          # --- Dispatcher ---

          # Addition
          if (node$creation_op == "add") {
            accumulate_grad(inputs[[1]], grad_output)
            accumulate_grad(inputs[[2]], grad_output)
          }

          # Subtraction (x - y)
          if (node$creation_op == "sub") {
            # 1. Gradient for X (Positive)
            accumulate_grad(inputs[[1]], grad_output)
            # 2. Gradient for Y (Negative)
            accumulate_grad(inputs[[2]], -grad_output)
          }

          # Sum
          if (node$creation_op == "sum") {
            if (inputs[[1]]$requires_grad) {
              orig_shape <- node$aux_data$orig_shape
              dim_sum <- node$aux_data$dim

              if (is.null(dim_sum)) {
                # Expand scalar gradient back to original matrix size
                expanded_grad <- matrix(grad_output[1], nrow=orig_shape[1], ncol=orig_shape[2])
                inputs[[1]]$grad <- inputs[[1]]$grad + expanded_grad
              } else if (dim_sum == 1) {
                # colSums -> Expand rows
                grad_vec <- as.numeric(grad_output)
                expanded_grad <- matrix(grad_vec, nrow=orig_shape[1], ncol=orig_shape[2], byrow=TRUE)
                inputs[[1]]$grad <- inputs[[1]]$grad + expanded_grad
              } else if (dim_sum == 2) {
                # rowSums -> Expand cols
                grad_vec <- as.numeric(grad_output)
                expanded_grad <- matrix(grad_vec, nrow=orig_shape[1], ncol=orig_shape[2], byrow=FALSE)
                inputs[[1]]$grad <- inputs[[1]]$grad + expanded_grad
              }
            }
          }

          # Negation (-x)
          if (node$creation_op == "neg") {
            if (inputs[[1]]$requires_grad) {
              inputs[[1]]$grad <- inputs[[1]]$grad - grad_output
            }
          }

          # Division (x / y)
          if (node$creation_op == "div") {
            x <- inputs[[1]]
            y <- inputs[[2]]
            if (x$requires_grad) inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output / y$data)
            if (y$requires_grad) inputs[[2]]$grad <- inputs[[2]]$grad + (-x$data / (y$data^2) * grad_output)
          }

          # Log (log(x))
          if (node$creation_op == "log") {
            if (inputs[[1]]$requires_grad) {
              inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output / inputs[[1]]$data)
            }
          }

          # Exp (exp(x))
          if (node$creation_op == "exp") {
            if (inputs[[1]]$requires_grad) {
              inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output * node$data)
            }
          }

          # Mean (mean(x))
          if (node$creation_op == "mean") {
            if (inputs[[1]]$requires_grad) {
              n <- node$aux_data
              inputs[[1]]$grad <- inputs[[1]]$grad + (as.numeric(grad_output) / n)
            }
          }

          # Multiplication (Element-wise)
          if (node$creation_op == "mul") {
            if (inputs[[1]]$requires_grad) inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output * inputs[[2]]$data)
            if (inputs[[2]]$requires_grad) inputs[[2]]$grad <- inputs[[2]]$grad + (grad_output * inputs[[1]]$data)
          }

          # Matrix Multiplication
          if (node$creation_op == "mm") {
            if (inputs[[1]]$requires_grad) inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output %*% t(inputs[[2]]$data))
            if (inputs[[2]]$requires_grad) inputs[[2]]$grad <- inputs[[2]]$grad + (t(inputs[[1]]$data) %*% grad_output)
          }

          # Activation: ReLU
          if (node$creation_op == "relu") {
            if (inputs[[1]]$requires_grad) {
              mask <- node$aux_data
              inputs[[1]]$grad <- inputs[[1]]$grad + (grad_output * mask)
            }
          }

          # Activation: Sigmoid
          if (node$creation_op == "sigmoid") {
            if (inputs[[1]]$requires_grad) {
              output <- node$data
              d_sigmoid <- output * (1 - output)
              inputs[[1]]$grad <- inputs[[1]]$grad + (d_sigmoid * grad_output)
            }
          }

          # Activation: Tanh
          if (node$creation_op == "tanh") {
            if (inputs[[1]]$requires_grad) {
              output <- node$data
              d_tanh <- 1 - (output^2)
              inputs[[1]]$grad <- inputs[[1]]$grad + (d_tanh * grad_output)
            }
          }

          # Softmax
          if (node$creation_op == "softmax") {
            if (inputs[[1]]$requires_grad) {
              # Gradient of Softmax is complex: S * (grad - sum(S * grad))
              S <- node$data
              G <- grad_output
              # 1. S * G (Element-wise)
              SG <- S * G
              # 2. Sum(S * G) for each row
              sum_SG <- rowSums(SG)
              # 3. Subtract sum from G (Broadcasting subtraction)
              G_minus_sum <- sweep(G, 1, sum_SG, "-")
              # 4. Multiply by S
              grad_input <- S * G_minus_sum
              inputs[[1]]$grad <- inputs[[1]]$grad + grad_input
            }
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
