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
