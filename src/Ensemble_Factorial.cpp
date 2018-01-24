#include "Ensemble_Factorial.h"

using namespace Rcpp;

void Ensemble_Factorial::do_crossover(){
  // select chains [i] and [j]
  int i = sample_int(n_chains-1);
  int j = i+1;
  // int i = 0;
  // int j = 1;
  // which rows of X will be included in the crossover
  // IntegerVector which_rows = sample_helper(K, nrows_crossover);
  IntegerVector which_rows = seq_len(K) - 1;
  // uniform crossover
  uniform_crossover(i, j, which_rows);
  
  chains[i].update_mu_for_all_states();
  chains[j].update_mu_for_all_states();

  // now consider all possible crossover points
  NumericVector log_probs(2*n);
  // arma::vec log_probs(2*n);
  double log_cumprod = 0.0;
  for(int t=0; t<n; t++){
    log_cumprod += crossover_likelihood(i, j, t, which_rows, nrows_crossover);
    log_probs[t] = log_cumprod;
  }
  for(int t=0; t<n; t++){
    log_cumprod += crossover_likelihood(i, j, t, which_rows, nrows_crossover);
    log_probs[t+n] = log_cumprod;
  }
  arma::vec arma_log_probs(log_probs.begin(), 2*n, false);
  NumericVector probs = exp(log_probs - arma_log_probs.max());
  // pick one of the crossovers and accept this move
  nonuniform_crossover(probs, i, j, which_rows);
}

void Ensemble_Factorial::nonuniform_crossover(NumericVector probs, int i, int j, IntegerVector which_rows){
  int t0 = sample_int(2*n, probs);
  if(t0 < n){
    // printf("t_end = %d\n", t0);
    crossover_end = t0;
    crossover_mat(chains[i].get_X(), chains[j].get_X(), t0, which_rows);
  } else{
    // printf("t_end = %d (flipped)\n", t0-n);
    crossover_end = t0-n;
    crossover_flipped = 1 - crossover_flipped;
    crossover2_mat(chains[i].get_X(), chains[j].get_X(), t0-n, n, which_rows);
  }
  // update x correpondingly
  chains[i].convert_X_to_x();
  chains[j].convert_X_to_x();
}

double Ensemble_Factorial::crossover_likelihood(int i, int j, int t, IntegerVector which_rows, int m){
  double log_denom = chains[i].pointwise_loglik_with_transitions(t) + chains[j].pointwise_loglik_with_transitions(t);

  // crossover
  crossover_one_column(chains[i].get_X(), chains[j].get_X(), t, which_rows, m);
  chains[i].convert_X_to_x(t);
  chains[j].convert_X_to_x(t);

  double log_num = chains[i].pointwise_loglik_with_transitions(t) + chains[j].pointwise_loglik_with_transitions(t);

  return log_num - log_denom;
}

void Ensemble_Factorial::uniform_crossover(int i, int j, IntegerVector which_rows){
  int t0 = sample_int(n);
  // printf("t0 = %d  ", t0);
  crossover_start = t0;
  // flip a coin
  if(R::runif(0, 1) < 0.5){
    crossover_flipped = 0;
    crossover_mat(chains[i].get_X(), chains[j].get_X(), t0, which_rows);
  } else{
    crossover_flipped = 1;
    crossover2_mat(chains[i].get_X(), chains[j].get_X(), t0, n, which_rows);
  }
  // update x correpondingly
  chains[i].convert_X_to_x();
  chains[j].convert_X_to_x();
}

void Ensemble_Factorial::swap_X(){
  Rcpp::RNGScope tmp;
  
  crossover_start = 0;
  crossover_end = 0;
  crossover_flipped = 0;
  // select chains [i] and [j]
  int i = sample_int(n_chains-1);
  int j = i+1;

  double accept_prob = MH_acceptance_prob_swap_x(chains[i].get_x(), chains[i].get_pi(), chains[i].get_A(), chains[i].get_emission_probs(),
                                                 chains[j].get_x(), chains[j].get_pi(), chains[j].get_A(), chains[j].get_emission_probs(),
                                                 n);
  // printf("swap X: accept prob %f\n", accept_prob);
  if(R::runif(0, 1) < accept_prob){
    printf("swap X accepted\n");
    crossover_start = 0;
    crossover_end = n-1;
    std::swap(chains[i].get_x(), chains[j].get_x());
    std::swap(chains[i].get_X(), chains[j].get_X());
  }
}

void Ensemble_Factorial::helper_row_swap(int i, int j, int k0, int start, int end){
  int temp;
  for(int t=start; t<end; t++){
    temp = chains[i].get_X()(k0, t);
    chains[i].get_X()(k0, t) = chains[j].get_X()(k0, t);
    chains[j].get_X()(k0, t) = temp;
  }
  chains[i].convert_X_to_x();
  chains[j].convert_X_to_x();
}

void Ensemble_Factorial::swap_one_row_X(){
  crossover_start = 0;
  crossover_end = 0;
  crossover_flipped = 0;
  // select chains [i] and [j]
  int i = sample_int(n_chains-1);
  int j = i+1;

  double ll_x_before = loglikelihood_x(chains[i].get_x(), chains[i].get_pi(), chains[i].get_A(), n) + loglikelihood_x(chains[j].get_x(), chains[j].get_pi(), chains[j].get_A(), n);
  double ll_y_before = loglikelihood(chains[i].get_x(), chains[i].get_emission_probs(), n);

  // pick a row and swap row k0
  int k0 = sample_int(K);
  helper_row_swap(i, j, k0, 0, n);

  double ll_x_after = loglikelihood_x(chains[i].get_x(), chains[i].get_pi(), chains[i].get_A(), n) + loglikelihood_x(chains[j].get_x(), chains[j].get_pi(), chains[j].get_A(), n);
  double ll_y_after = loglikelihood(chains[i].get_x(), chains[i].get_emission_probs(), n);
  double logratio_x = ll_x_after - ll_x_before;
  double logratio_y = ll_y_after - ll_y_before;
  double accept_prob = exp(logratio_x + logratio_y);

  // printf("swap one row of X: accept prob %f\n", accept_prob);

  if(R::runif(0, 1) < accept_prob){
    printf("swap one row of X: accepted\n");
    crossover_start = 0;
    crossover_end = n-1;
  } else{
    // printf("swap one row of X: rejected\n");
    helper_row_swap(i, j, k0, 0, n);
  }
}

void Ensemble_Factorial::random_crossover_one_row_X(){
  crossover_start = 0;
  crossover_end = 0;
  crossover_flipped = 0;
  // select chains [i] and [j]
  int i = sample_int(n_chains-1);
  int j = i+1;

  double ll_x_before = loglikelihood_x(chains[i].get_x(), chains[i].get_pi(), chains[i].get_A(), n) + loglikelihood_x(chains[j].get_x(), chains[j].get_pi(), chains[j].get_A(), n);
  double ll_y_before = loglikelihood(chains[i].get_x(), chains[i].get_emission_probs(), n);

  // pick a row and swap row k0
  int k0 = sample_int(K);
  int t0 = sample_int(n-1);
  helper_row_swap(i, j, k0, t0, n);

  double ll_x_after = loglikelihood_x(chains[i].get_x(), chains[i].get_pi(), chains[i].get_A(), n) + loglikelihood_x(chains[j].get_x(), chains[j].get_pi(), chains[j].get_A(), n);
  double ll_y_after = loglikelihood(chains[i].get_x(), chains[i].get_emission_probs(), n);
  double logratio_x = ll_x_after - ll_x_before;
  double logratio_y = ll_y_after - ll_y_before;
  double accept_prob = exp(logratio_x + logratio_y);

  // printf("swap one row of X: accept prob %f\n", accept_prob);

  if(R::runif(0, 1) < accept_prob){
    // printf("swap one row of X: accepted\n");
    crossover_start = t0;
    crossover_end = n-1;
  } else{
    // printf("swap one row of X: rejected\n");
    helper_row_swap(i, j, k0, t0, n);
  }
}

void Ensemble_Factorial::random_crossover_X(){
  Rcpp::RNGScope tmp;
  
  crossover_start = 0;
  crossover_end = 0;
  crossover_flipped = 0;
  // select chains [i] and [j]
  int i = sample_int(n_chains-1);
  int j = i+1;

  double ll_x_before = loglikelihood_x(chains[i].get_x(), chains[i].get_pi(), chains[i].get_A(), n) + loglikelihood_x(chains[j].get_x(), chains[j].get_pi(), chains[j].get_A(), n);
  double ll_y_before = loglikelihood(chains[i].get_x(), chains[i].get_emission_probs(), n);

  int t0 = sample_int(n-1);
  for(int k0=0; k0<K; k0++){
    helper_row_swap(i, j, k0, t0, n);
  }

  double ll_x_after = loglikelihood_x(chains[i].get_x(), chains[i].get_pi(), chains[i].get_A(), n) + loglikelihood_x(chains[j].get_x(), chains[j].get_pi(), chains[j].get_A(), n);
  double ll_y_after = loglikelihood(chains[i].get_x(), chains[i].get_emission_probs(), n);
  double logratio_x = ll_x_after - ll_x_before;
  double logratio_y = ll_y_after - ll_y_before;
  double accept_prob = exp(logratio_x + logratio_y);

  // printf("swap one row of X: accept prob %f\n", accept_prob);

  if(R::runif(0, 1) < accept_prob){
    crossover_start = t0;
    crossover_end = n-1;
  } else{
    for(int k0=0; k0<K; k0++){
      helper_row_swap(i, j, k0, t0, n);
    }
  }
}

// void Ensemble_Factorial::update_mu_mod(NumericVector y){
//   // assuming there are two chains only
//   IntegerMatrix X0 = clone(chains[0].get_X());
//   IntegerVector x0 = clone(chains[0].get_x());
//   IntegerMatrix X1 = clone(chains[1].get_X());
//   IntegerVector x1 = clone(chains[1].get_x());
// 
//   if(R::runif(0, 1) < 0.5){
//     chains[0].update_mu_mod(y, x0, X0);
//     chains[1].update_mu_mod(y, x1, X1);
//   } else{
//     chains[1].update_mu_mod(y, x0, X0);
//     chains[0].update_mu_mod(y, x1, X1);
//   }
// }
