#' @title Kaiming Uniform Initialization
#' @description He Initialization (good for ReLU).
#' @param fan_in Integer. Number of input features.
#' @param fan_out Integer. Number of output features.
#' @export
kaiming_uniform <- function(fan_in, fan_out) {
  # Bound is sqrt(6 / fan_in)
  limit <- sqrt(6 / fan_in)
  matrix(runif(fan_in * fan_out, min = -limit, max = limit),
         nrow = fan_in, ncol = fan_out)
}

#' @title Xavier Uniform Initialization
#' @description Glorot Initialization (good for Sigmoid/Tanh).
#' @param fan_in Integer. Number of input features.
#' @param fan_out Integer. Number of output features.
#' @export
xavier_uniform <- function(fan_in, fan_out) {
  # Bound is sqrt(6 / (fan_in + fan_out))
  limit <- sqrt(6 / (fan_in + fan_out))
  matrix(runif(fan_in * fan_out, min = -limit, max = limit),
         nrow = fan_in, ncol = fan_out)
}
