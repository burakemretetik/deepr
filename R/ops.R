#' @title Tensor Addition
#' @export
`+.Tensor` <- function(e1, e2) {
  if (!inherits(e1, "Tensor") || !inherits(e2, "Tensor")) stop("V1: Tensor + Tensor only")

  # 1. Shape Check (Allowed: Exact Match OR Broadcastable)
  d1 <- dim(e1$data)
  d2 <- dim(e2$data)

  # Allow strictly: Exact Match OR (Rows mismatch but cols match AND one has 1 row)
  # This covers the Bias addition: (Batch, Features) + (1, Features)
  allow_broadcast <- FALSE

  if (all(d1 == d2)) {
    # Exact match, all good
  } else if (d1[2] == d2[2]) {
    # Columns match. Check rows.
    if (d1[1] == 1 || d2[1] == 1) {
      allow_broadcast <- TRUE
    } else {
      stop("Shape mismatch: Rows don't match and neither is 1.")
    }
  } else {
    stop(paste0("Shape mismatch: ", paste(d1, collapse="x"), " vs ", paste(d2, collapse="x")))
  }

  # 2. The Math
  # R handles (N, M) + (1, M) by recycling elements column-wise.
  # THIS IS DANGEROUS. (1, M) fills column 1, then column 2...
  # We want it to fill Row 1, Row 2...

  if (allow_broadcast) {
    # If e1 is (1, M) and e2 is (N, M), we need to handle R's recycling.
    # sweep() is safer than + for broadcasting rows
    if (d1[1] == 1 && d2[1] > 1) {
      # Add e1 vector to every row of e2
      newdata <- sweep(e2$data, 2, as.numeric(e1$data), "+")
    } else if (d2[1] == 1 && d1[1] > 1) {
      # Add e2 vector to every row of e1
      newdata <- sweep(e1$data, 2, as.numeric(e2$data), "+")
    } else {
      newdata <- e1$data + e2$data # Fallback
    }
  } else {
    newdata <- e1$data + e2$data
  }

  req_grad <- e1$requires_grad || e2$requires_grad

  Tensor$new(newdata, req_grad, list(e1, e2), "add")
}

#' @title Tensor Subtraction & Negation
#' @export
`-.Tensor` <- function(e1, e2) {
  # Handle Unary Negation (-x)
  if (missing(e2)) {
    ret <- Tensor$new(
      data = -e1$data,
      requires_grad = e1$requires_grad,
      creators = list(e1),
      creation_op = "neg"
    )
    return(ret)
  }

  # Handle Binary Subtraction (x - y)
  if (!inherits(e1, "Tensor") || !inherits(e2, "Tensor")) stop("V1: Tensor - Tensor only")

  # Shape Check
  if (!all(dim(e1$data) == dim(e2$data))) {
     # (You can leave your broadcasting logic here if you added it)
  }

  newdata <- e1$data - e2$data
  req_grad <- e1$requires_grad || e2$requires_grad

  Tensor$new(newdata, req_grad, list(e1, e2), "sub")
}

#' @title Tensor Division
#' @export
`/.Tensor` <- function(e1, e2) {
  if (!inherits(e1, "Tensor") || !inherits(e2, "Tensor")) stop("V1: Tensor / Tensor only")

  # For V1: Strict Shape Check
  if (!all(dim(e1$data) == dim(e2$data))) stop("Shape Mismatch in Division")

  newdata <- e1$data / e2$data
  req_grad <- e1$requires_grad || e2$requires_grad

  Tensor$new(newdata, req_grad, list(e1, e2), "div")
}

#' @title Tensor Logarithm (Natural Log)
#' @export
log.Tensor <- function(x, base = exp(1)) {
  # V1: Only natural log for simplicity (base e)
  newdata <- log(x$data)

  Tensor$new(
    data = newdata,
    requires_grad = x$requires_grad,
    creators = list(x),
    creation_op = "log"
  )
}

#' @title Tensor Exponential
#' @export
exp.Tensor <- function(x) {
  newdata <- exp(x$data)

  Tensor$new(
    data = newdata,
    requires_grad = x$requires_grad,
    creators = list(x),
    creation_op = "exp"
  )
}

#' @title Tensor Multiplication (Element-wise)
#' @export
`*.Tensor` <- function(e1, e2) {
  if (!inherits(e1, "Tensor") || !inherits(e2, "Tensor")) stop("V1: Multiplication only supports Tensor * Tensor")

  # Safety Check: Strict Shape Matching
  if (!all(dim(e1$data) == dim(e2$data))) {
    stop("Shape Mismatch: V1 requires identical shapes for element-wise multiplication.")
  }

  newdata <- e1$data * e2$data
  req_grad <- e1$requires_grad || e2$requires_grad

  Tensor$new(newdata, req_grad, list(e1, e2), "mul")
}

#' @title Matrix Multiplication
#' @export
`%*%.Tensor` <- function(e1, e2) {
  if (!inherits(e1, "Tensor") || !inherits(e2, "Tensor")) stop("V1: MatMul only supports Tensor %*% Tensor")

  # Standard Matrix Mul Rule: Cols of A must match Rows of B
  if (ncol(e1$data) != nrow(e2$data)) {
    stop(paste0("Dimension Mismatch in %*%: ", ncol(e1$data), " vs ", nrow(e2$data)))
  }

  newdata <- e1$data %*% e2$data
  req_grad <- e1$requires_grad || e2$requires_grad

  Tensor$new(newdata, req_grad, list(e1, e2), "mm")
}

#' @title ReLU Activation
#' @description Rectified Linear Unit.
#' @param x Tensor.
#' @export
relu <- function(x) {
  if (!inherits(x, "Tensor")) stop("ReLU requires a Tensor")

  # 1. Forward: max(0, x)
  # We use logical indexing for speed
  mask <- x$data > 0
  newdata <- x$data * mask # Zeroes out negatives

  ret <- Tensor$new(
    data = newdata,
    requires_grad = x$requires_grad,
    creators = list(x),
    creation_op = "relu"
  )

  # Store the mask inside the tensor for the backward pass
  # (This is a common optimization: we remember which values were active)
  ret$aux_data <- mask

  return(ret)
}

#' @title Tensor Summation
#' @description Sums all elements or sums along a dimension.
#' @param x Tensor.
#' @param dim Integer. The dimension to sum over (1 for rows, 2 for cols). If NULL, sums all.
#' @export
tensor_sum <- function(x, dim = NULL) {
  if (!inherits(x, "Tensor")) stop("tensor_sum requires a Tensor")

  # 1. Forward Logic
  if (is.null(dim)) {
    # Sum everything -> scalar
    newdata <- sum(x$data)
    # For scalar result, keep it as 1x1 matrix for consistency
    newdata <- matrix(newdata, 1, 1)
  } else {
    # Sum over dimension
    # if dim=1 (sum rows) -> colSums
    # if dim=2 (sum cols) -> rowSums
    if (dim == 1) {
      newdata <- colSums(x$data)
      # colSums returns a vector, convert to row matrix (1, cols)
      newdata <- matrix(newdata, nrow = 1)
    } else if (dim == 2) {
      newdata <- rowSums(x$data)
      # rowSums returns vector, convert to col matrix (rows, 1)
      newdata <- matrix(newdata, ncol = 1)
    } else {
      stop("V1 only supports dim=1 or dim=2")
    }
  }

  ret <- Tensor$new(
    data = newdata,
    requires_grad = x$requires_grad,
    creators = list(x),
    creation_op = "sum"
  )

  # Store the dimension we summed over for backward pass
  ret$aux_data <- list(orig_shape = x$shape, dim = dim)

  return(ret)
}

#' @title Tensor Mean
#' @export
mean.Tensor <- function(x) {
  val <- mean(x$data)
  Tensor$new(data = matrix(val, 1, 1), requires_grad = x$requires_grad,
             creators = list(x), creation_op = "mean", aux_data = length(x$data))
}

#' @title Sigmoid Activation
#' @export
sigmoid <- function(x) {
  if (!inherits(x, "Tensor")) stop("sigmoid requires a Tensor")
  if (!inherits(x, "Tensor")) stop("sigmoid requires a Tensor")

  # Formula: 1 / (1 + exp(-x))
  val <- 1 / (1 + exp(-x$data))

  Tensor$new(
    data = val,
    requires_grad = x$requires_grad,
    creators = list(x),
    creation_op = "sigmoid"
  )
}

#' @title Tanh Activation
#' @export
tanh_act <- function(x) {
  if (!inherits(x, "Tensor")) stop("tanh requires a Tensor")

  # Standard tanh function
  val <- tanh(x$data)

  Tensor$new(
    data = val,
    requires_grad = x$requires_grad,
    creators = list(x),
    creation_op = "tanh"
  )
}

#' @title Softmax Activation (Stable)
#' @description Computes exp(x) / sum(exp(x)) row-wise.
#' @export
#'
softmax <- function(x) {
  if (!inherits(x, "Tensor")) stop("softmax requires a Tensor")

  # 1. Numerical Stability Trick: x_safe = x - max(x)
  # We do this row-wise using 'apply'

  # Shift values to prevent exp() explosion
  row_maxes <- apply(x$data, 1, max)
  x_safe <- x$data - row_maxes

  # 2. Exponentiate
  exps <- exp(x_safe)

  # 3. Normalize
  row_sums <- rowSums(exps)
  probs <- exps / row_sums

  Tensor$new(
    data = probs,
    requires_grad = x$requires_grad,
    creators = list(x),
    creation_op = "softmax"
  )
}
