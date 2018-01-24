#include "Chain.h"

Chain::Chain(int k_, int n_){
  k = k_;
  n = n_;
  is_tempered = false;
  inv_temperature = 1.0;
  a0 = 0.1;
  b0 = 0.1;
  x = IntegerVector(n);
  switching_prob = NumericVector(n-1);
  pi = NumericVector(k);
  A = NumericMatrix(k, k);
  A_pars = NumericMatrix(k, k);
  A_gamma = NumericMatrix(k, k);
  emission_probs = NumericMatrix(k, n);
  // loglik_marginal = 0.0;
  // loglik_cond = 0.0;
  possible_values = seq_len(k)-1;
  marginal_distr = NumericMatrix(k, n);

  List PP(n), QQ(n);
  for(int t=0; t<n; t++){
    PP[t] = NumericMatrix(k, k);
    QQ[t] = NumericMatrix(k, k);
  }
  P = ListOf<NumericMatrix>(PP);
  Q = ListOf<NumericMatrix>(QQ);
}

// double Chain::calculate_loglik_marginal(){
//   return marginal_loglikelihood(pi, A, emission_probs, 1.0, k, n);
// }

void Chain::scale_marginals(int max_iter, int burnin){
  arma::mat out(marginal_distr.begin(), k, n, false);
  out /= (float) (max_iter - burnin);
}
