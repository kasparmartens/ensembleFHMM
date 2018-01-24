generate_Y_gaussian = function(X, w, h = 50, sd = 1){
  n = ncol(X)
  Y = rep(0, n)
  Ymean = rep(0, n)
  for(t in 1:n){
    Ymean[t] = h * sum(X[, t] * w)
    Y[t] = rnorm(1, Ymean[t], sd)
  }
  list(Ymean = Ymean, Y = Y, X = X, w = w, h = h, sd = sd)
}
