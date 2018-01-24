#' @useDynLib ensembleFHMM
#' @importFrom Rcpp sourceCpp

FHMM_init <- function(y, X_init, w_init, h_mu, sigma2_init, alpha = 1.0, HB_sampling = TRUE, radius = 1, nrows_gibbs = 1L, inv_temperature = 1.0, h_sd = 10){
  N <- length(y)
  K <- nrow(X_init)
  all_combs <- combn(0:(K-1), ifelse(nrows_gibbs == K, 1, K-nrows_gibbs))
  transition_probs <- rep(0.01, K)
  
  chain <- Chain_Factorial$new(y, K, as.integer(2**K), N, alpha)
  chain$activate_sampling(HB_sampling, radius, nrows_gibbs, all_combs)
  chain$set_temperature(inv_temperature)
  x_init <- as.integer(convert_X_to_x(X_init, K, N))
  chain$initialise_pars(w_init, transition_probs, x_init, h_mu, h_sd, sigma2_init)
  chain
}

FHMM_VB_init <- function(y, X_init, w_init, h_mu, sigma2_init, alpha = 1.0, HB_sampling = TRUE, radius = 1, nrows_gibbs = 1L, inv_temperature = 1.0, h_sd = 10){
  N <- length(y)
  K <- nrow(X_init)
  all_combs <- combn(0:(K-1), ifelse(nrows_gibbs == K, 1, K-nrows_gibbs))
  transition_probs <- rep(0.01, K)
  
  chain <- Chain_Factorial$new(y, K, as.integer(2**K), N, alpha)
  chain$activate_variational(X_init)
  x_init <- as.integer(convert_X_to_x(X_init, K, N))
  chain$initialise_pars(w_init, transition_probs, x_init, h_mu, h_sd, sigma2_init)
  chain
}

FHMM_ensemble_init <- function(y, X_init, w_init, h_mu, sigma2_init, alpha = 1.0, HB_sampling = TRUE, radius = 1, nrows_gibbs = 1L, inv_temperatures = 1.0, h_sd = 10, variational_inference = FALSE){
  N <- length(y)
  K <- nrow(X_init)
  all_combs <- combn(0:(K-1), ifelse(nrows_gibbs == K, 1, K-nrows_gibbs))
  transition_probs <- rep(0.001, K)
  
  n_chains <- length(inv_temperatures)
  ensemble <- Ensemble_Factorial$new(y, n_chains, K, as.integer(2**K), N, alpha)
  ensemble$activate_sampling(HB_sampling, radius, nrows_gibbs, all_combs)
  ensemble$set_temperatures(inv_temperatures)
  x_init <- as.integer(convert_X_to_x(X_init, K, N))
  ensemble$initialise_pars(w_init, transition_probs, x_init, h_mu, h_sd, sigma2_init)
  ensemble
}
