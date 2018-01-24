# ensembleFHMM
R package for ensemble MCMC for Factorial Hidden Markov Models

This R package accompanies our paper on "Augmented Ensemble MCMC with applications to Factorial HMMs" [available on arxiv](https://arxiv.org/abs/1703.08520). 

It contains implementation for fitting Factorial Hidden Markov Models using C++ backend and exposing this to R via Rcpp modules. 

Example usage:

```r
# install package
devtools::install_github("kasparmartens/ensembleFHMM")

# load package
library(ensembleFHMM)

# initialise the X matrix (with K rows and N columns)
K <- 3
N <- 10
X <- matrix(1L, K, N)
# generate toy data y_t ~ N(h * \sum_k w_k X_{k, t}, sd)
data <- generate_Y_gaussian(X, w = c(0.2, 0.3, 0.5), h = 15, sd = 1)

# initialise ensemble of chains for sampling
X_init <- X
w_init <- gtools::rdirichlet(1, rep(1, K+1))
y <- data$Y
ensemble <- FHMM_ensemble_init(y, X_init, w_init, h_mu = 15, sigma2_init = 1, inv_temperatures = c(1.0, 0.2), HB_sampling = TRUE)
# one iteration of forward-filtering-backward-sampling
ensemble$update_x()
# augmented crossover exchange move
ensemble$do_crossover()

# get information about the first chain
chain <- ensemble$get_chain(0L)
chain$get_X()

```
