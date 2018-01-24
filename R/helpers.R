#' @export
create_binary_seq = function(n, lst){
  x = rep(0L, n)
  for(i in 1:length(lst)){
    x[lst[[i]]] = 1L
  }
  x
}

#' @export
r_binmat = function(K, n, p){
  matrix(rbinom(K*n, 1, p), K, n)
}

#' @export
convert_X_to_x = function(X, K, n){
  x = rep(0, n)
  for(t in 1:n){
    temp = 0
    for(i in 1:K){
      if(X[i, t] == 1) temp = temp + 2**(i-1)
    }
    x[t] = temp
  }
  x
}

