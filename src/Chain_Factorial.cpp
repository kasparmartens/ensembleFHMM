#include "Chain_Factorial.h"

void Chain_Factorial::update_X_variational(arma::rowvec yy){
  for(int kk=0; kk<K; kk++){
    update_X_variational_single_row(yy, kk);
  }
}

void Chain_Factorial::update_X_variational_single_row(arma::rowvec yy, int which_row){
  arma::rowvec y_tilde(n);
  arma::rowvec y_pred(n);
  y_pred.zeros();
  for(int t=0; t<n; t++){
    double pred=0;
    for(int m=0; m<K; m++){
      if(m != which_row){
        y_pred[t] += w[m] * theta(m, t);
      }
    }
    // add intercept
    y_pred[t] += w[K];
  }
  y_tilde = yy - h_mu * y_pred;
  arma::mat emission_probs_local(2, n);
  // arma::rowvec log_h_k = 1 / sigma2 * h_mu * w[which_row] * y_tilde - 0.5 / sigma2 * pow(h_mu * w[which_row], 2);
  arma::rowvec log_h_k(n);
  for(int t=0; t<n; t++){
    double m = y_pred[t];
    double s2 = sigma2 + m*m*h_sigma2;
    log_h_k[t] = 1 / s2 * h_mu * w[which_row] * y_tilde[t] - 0.5 / s2 * pow(h_mu * w[which_row], 2); 
  }
  emission_probs_local.row(0).fill(exp(0.0 - log_h_k.max()));
  emission_probs_local.row(1) = exp(log_h_k - log_h_k.max());
  
  rho = transition_probs[which_row];
  arma::mat AA(2, 2);
  AA.fill(rho);
  AA.diag(1.0 - rho);
  arma::vec pii(2);
  pii.fill(0.5);
  theta.row(which_row) = forward_backward_mod(pii, AA, emission_probs_local, 2, n);
}

// void Chain_Factorial::update_sigma2(){
//   Rcpp::RNGScope tmp;
//   
//   y_pred = calculate_mean_for_all_t(X, w, h, K, n);
//   double ss=0.0;
//   for(int t=0; t<n; t++){
//     ss += pow(y[t] - y_pred[t], 2);
//   }
//   double a = a_sigma + 0.5 * inv_temperature * n;
//   double b = b_sigma + 0.5 * inv_temperature * ss;
//   sigma2 = 1.0 / R::rgamma(a, 1.0 / b);
//   sigma = sqrt(sigma2);
//   
//   printf("updated sigma2 = %1.3f\n", sigma2);
//   
// }

void Chain_Factorial::update_sigma2(double sd){
  Rcpp::RNGScope tmp;
  
  double sigma2_proposed = random_walk_log_scale(sigma2, sd);
  double logp_current = calculate_posterior_prob(w, h, alpha0, sigma2);
  double logp_proposed = calculate_posterior_prob(w, h, alpha0, sigma2_proposed);
  
  if(R::runif(0, 1) < exp(logp_proposed - logp_current)){
    sigma2 = sigma2_proposed;
    sigma = sqrt(sigma2);
    printf("updated sigma2 = %1.3f\n", sigma2);
  }
}

void Chain_Factorial::update_h(double sd){
  Rcpp::RNGScope tmp;
  
  double h_proposed = h + R::rnorm(0, sd);
  if(h_proposed < 0)
    return;
  
  double logp_current = calculate_posterior_prob(w, h, alpha0, sigma2);
  double logp_proposed = calculate_posterior_prob(w, h_proposed, alpha0, sigma2);
  
  if(R::runif(0, 1) < exp(logp_proposed - logp_current)){
    h = h_proposed;
    printf("accepted h = %1.2f\n", h);
  }
}

void Chain_Factorial::update_w(double sd){
  
  Rcpp::RNGScope tmp;
  
  // now update w
  
  double logp_current = calculate_posterior_prob(w, h, alpha0, sigma2);
  
  // propose a new alpha
  // double alpha_proposed = random_walk_log_scale(alpha0, 0.1);
  // propose a new w
  NumericVector w_unnorm_proposed = RWMH(u, K+1, sd);
  NumericVector w_proposed = w_unnorm_proposed / sum(w_unnorm_proposed);
  double logp_proposed = calculate_posterior_prob(w_proposed, h, alpha0, sigma2);
  
  // printf("proposed logp = %1.3f, current logp = %1.3f \n", logprob_proposed, logprob_current);
  
  // accept or reject
  if(R::runif(0, 1) < exp(logp_proposed - logp_current)){
    u = clone(w_unnorm_proposed);
    w = clone(w_proposed);
    
    // for(int i=0; i<K; i++){
    //   printf("w_prev[%d] = %1.3f, w_new = %1.3f\n", i, w[i], w_proposed[i]);
    // }
    // printf("w_prev[%d] = %1.3f, w_new = %1.3f\n", K, w[K], w_proposed[K]);
  }
  
}


double Chain_Factorial::get_marginal_loglik(NumericVector w_new){
  // NumericMatrix emission_probs_proposed = calculate_emission_probs(y, w_new, inv_temperature, h, sigma, df, k, K, n, mapping);
  // // NumericMatrix emission_probs_proposed = calculate_emission_probs_gaussian(y, w_new, inv_temperature, h, sigma, k, K, n, mapping);
  // // draw u ~ HB around current x
  // IntegerVector uu = return_sample_within_hamming_ball(x, n, hamming_balls);
  // FHMM_forward_step(pi, A, emission_probs_proposed, P_FHMM, loglik_marginal, k_restricted, n, uu, hamming_balls);
  // // now we have loglik_marginal value
  // double a0 = 1.0, b0 = 1.0;
  // double logprior = ddirichlet(w_new, alpha0, K);
  // logprior += R::dgamma(alpha0, a0, 1.0/b0, true) + log(alpha0);
  // for(int k=0; k<K+1; k++){
  //   logprior += mylog(abs(w_new[k] - w_new[k]*w_new[k]));
  // }
  // 
  // return loglik_marginal + logprior;
}

double Chain_Factorial::get_marginal_loglik_HB(NumericVector w_new, IntegerVector uu, double h_new){
  // NumericMatrix emission_probs_proposed = calculate_emission_probs(y, w_new, inv_temperature, h_new, sigma, df, k, K, n, mapping);
  // // NumericMatrix emission_probs_proposed = calculate_emission_probs_gaussian(y, w_new, inv_temperature, h, sigma, k, K, n, mapping);
  // // draw u ~ HB around current x
  // FHMM_forward_step(pi, A, emission_probs_proposed, P_FHMM, loglik_marginal, k_restricted, n, uu, hamming_balls);
  // // now we have loglik_marginal value
  // double a0 = 1.0, b0 = 1.0;
  // double logprior = ddirichlet(w_new, alpha0, K+1);
  // logprior += R::dgamma(alpha0, a0, 1.0/b0, true) + log(alpha0);
  // logprior += R::dnorm4(h_new, h_mu, h_sd, true);
  // for(int k=0; k<K+1; k++){
  //   logprior += mylog(abs(w_new[k] - w_new[k]*w_new[k]));
  // }
  // return loglik_marginal + logprior;
}

void Chain_Factorial::update_w_marginal(double sd){
  Rcpp::RNGScope tmp;
  
  IntegerVector uu = return_sample_within_hamming_ball(x, n, hamming_balls);
  
  double logp_current = get_marginal_loglik_HB(w, uu, h);
  NumericVector w_unnorm_proposed = RWMH(u, K+1, sd);
  NumericVector w_proposed = w_unnorm_proposed / sum(w_unnorm_proposed);
  // double h_proposed = h + R::rnorm(0, 3.0);
  double logp_proposed = get_marginal_loglik_HB(w_proposed, uu, h);
  if(R::runif(0, 1) < exp(logp_proposed - logp_current)){
    
    for(int i=0; i<K; i++){
      printf("w_prev[%d] = %1.3f, w_new = %1.3f\n", i, w[i], w_proposed[i]);
    }
    printf("w_prev[%d] = %1.3f, w_new = %1.3f\n", K, w[K], w_proposed[K]);
    
    u = clone(w_unnorm_proposed);
    w = clone(w_proposed);

  } else{
    // printf("#######REJECT######## %g %g %g\n", exp(logp_proposed - logp_current), logp_proposed, logp_current);
    // for(int i=0; i<K; i++){
    //   printf("w_prev[%d] = %1.3f, w_new = %1.3f\n", i, w[i], w_proposed[i]);
    // }
    // printf("w_prev[%d] = %1.3f, w_new = %1.3f\n", K, w[K], w_proposed[K]);
  }
}

void Chain_Factorial::update_h_marginal(double sd){
  Rcpp::RNGScope tmp;
  
  IntegerVector uu = return_sample_within_hamming_ball(x, n, hamming_balls);
  
  double logp_current = get_marginal_loglik_HB(w, uu, h);
  double h_proposed = h + R::rnorm(0, sd);
  double logp_proposed = get_marginal_loglik_HB(w, uu, h_proposed);
  if(R::runif(0, 1) < exp(logp_proposed - logp_current)){
    h = h_proposed;
  }
}

double Chain_Factorial::calculate_posterior_prob(NumericVector w_proposed, double h_proposed, double alpha_proposed, double sigma2_proposed){
  double a0 = 1.0, b0 = 1.0;
  // p(alpha) * p(w | alpha)
  double logprob = ddirichlet(w_proposed, alpha_proposed, K+1);
  logprob += R::dgamma(alpha_proposed, a0, 1.0/b0, true) + log(alpha_proposed);
  for(int k=0; k<K+1; k++){
    logprob += mylog(abs(w_proposed[k] - w_proposed[k]*w_proposed[k]));
  }
  // p(y | \sum_k h*w_k*X_k)
  y_pred_temp = calculate_mean_for_all_t(X, w_proposed, h_proposed, K, n);
  double sigma_proposed = sqrt(sigma2_proposed);
  logprob += inv_temperature * vec_t_density(y, y_pred_temp, sigma_proposed, df, n);
  // p(sigma2)
  logprob += R::dgamma(1/sigma2_proposed, a_sigma, 1.0/b_sigma, true) - log(sigma2_proposed);
  return logprob;
}


void Chain_Factorial::initialise_pars(NumericVector w_, NumericVector transition_probs_, IntegerVector x_, double h_mu_, double h_sigma2_, double sigma2_){
  h_mu = h_mu_;
  h_sigma2 = h_sigma2_;
  h = h_mu_;
  sigma2 = sigma2_;
  sigma = sqrt(sigma2);

  // draw pi from the prior
  initialise_const_vec(pi, 1.0 / K, K);

  u = NumericVector(K+1);
  w = NumericVector(K+1);
  for(int k=0; k<K+1; k++){
    u(k) = w_(k);
  }
  w = u / sum(u);

  update_mu_for_all_states();

  transition_probs = clone(transition_probs_);// [[Rcpp::export]]
  double inv_temp_prior = 1.0;
  if(abs(inv_temperature - 1.0) > 1e-16){
    inv_temp_prior = 0.75;
  } 
  FHMM_update_A(transition_probs, A, mapping, inv_temp_prior);
  for(int t=0; t<n; t++){
    x[t] = x_[t];
 }
  convert_x_to_X();

}
// void Chain_Factorial::initialise_pars(NumericVector w_, NumericVector transition_probs_, IntegerVector x_, NumericVector h_values_, NumericVector h_prob_, double sigma2_){
//   h_values = h_values_;
//   h_prob = h_prob_;
//   
//   sigma2 = sigma2_;
//   sigma = sqrt(sigma2);
//   
//   // draw pi from the prior
//   initialise_const_vec(pi, 1.0 / K, K);
//   
//   u = NumericVector(K+1);
//   w = NumericVector(K+1);
//   for(int k=0; k<K+1; k++){
//     u(k) = w_(k);
//   }
//   w = u / sum(u);
//   
//   update_mu_for_all_states();
//   
//   transition_probs = clone(transition_probs_);// [[Rcpp::export]]
//   double inv_temp_prior = 1.0;
//   FHMM_update_A(transition_probs, A, mapping, inv_temp_prior);
//   for(int t=0; t<n; t++){
//     x[t] = x_[t];
//   } 
//   convert_x_to_X();
//   
// }

void Chain_Factorial::update_mu_for_all_states(){
  for(int j=0; j<k; j++){
    double temp = 0.0;
    for(int i=0; i<K; i++){
      if(mapping(i, j) == 1){
        temp += w(i);
      }
    }
    temp += w[K];
    mu_all[j] = temp;
    
    marginal_mean[j] = h_mu * mu_all[j];
    marginal_sd[j] = sqrt(sigma2 + mu_all[j]*mu_all[j]*h_sigma2);
  }
}

void Chain_Factorial::update_emission_probs(){

  // update mu_all and compute marginal_mean and marginal_sd vectors
  update_mu_for_all_states();

  for(int t=0; t<n; t++){
    for(int i=0; i<k; i++){
      double loglik = R::dnorm4(y[t], marginal_mean[i], marginal_sd[i], true);
      // double loglik = my_t_density(y[t], marginal_mean[i], marginal_sd[i], df);
      // double loglik = R::dnorm4(y[t], mu_all[i], sigma, true);
      // double loglik = my_t_density(y[t], mu_all[i], sigma, df);
      emission_probs(i, t) = exp(inv_temperature * loglik);
    }
  }
}
// void Chain_Factorial::update_emission_probs(){
//   
//   update_mu_for_all_states();
//   
//   for(int t=0; t<n; t++){
//     for(int i=0; i<k; i++){
//       double likelihood = 0.0;
//       for(int j=0; j<h_values.size(); j++){
//         // loglik += log(h_prob[j]) + R::dnorm4(y[t], h_values[j]*mu_all[i], sigma, true);
//         double logp_j = log(h_prob[j]) + my_t_density(y[t], h_values[j]*mu_all[i], sigma, df);
//         likelihood += exp(logp_j);
//       }
//       // double loglik = R::dnorm4(y[t], mu_all[i], sigma, true);
//       // double loglik = my_t_density(y[t], mu_all[i], sigma, df);
//       emission_probs(i, t) = likelihood; //pow(likelihood, inv_temperature);
//     }
//   }
// }


void Chain_Factorial::update_A(){
  Rcpp::RNGScope tmp;
  double inv_temp_prior = 1.0;
  
  IntegerVector counts = FHMM_count_transitions(X);
  int total = (n-1);
  for(int i=0; i<K; i++){
    transition_probs[i] = R::rbeta(1 + inv_temp_prior*counts[i], 100 + inv_temp_prior*(total - counts[i]));
  }
  FHMM_update_A(transition_probs, A, mapping, inv_temp_prior);
}

void Chain_Factorial::update_x_BlockGibbs(){
  Rcpp::RNGScope tmp;
  
  if(nrows_gibbs == K){
    // forward step
    FHMM_forward_step(pi, A, emission_probs, P_FHMM, loglik_marginal, k_restricted, n, x, restricted_space);
    
    // now backward sampling
    FHMM_backward_sampling(x, P_FHMM, k_restricted, n, restricted_space);
    
  } else{
    for(int i=0; i<all_combinations.ncol(); i++){
      // forward step
      FHMM_forward_step(pi, A, emission_probs, P_FHMM, loglik_marginal, k_restricted, n, x, restr_space[i]);
      
      // now backward sampling
      FHMM_backward_sampling(x, P_FHMM, k_restricted, n, restr_space[i]);
    }
    
  }
  
  convert_x_to_X();
}

void Chain_Factorial::update_x_HammingBall(){
  Rcpp::RNGScope tmp;
  
  // Hamming ball sampling: sample u_t and overwrite x_t
  sample_within_hamming_ball(x, n, hamming_balls);
  
  // forward step
  FHMM_forward_step(pi, A, emission_probs, P_FHMM, loglik_marginal, k_restricted, n, x, hamming_balls);
  
  // now backward sampling
  FHMM_backward_sampling(x, P_FHMM, k_restricted, n, hamming_balls);
  
  // conditional loglikelihood
  // loglik_cond = loglikelihood(x, emission_probs, n) + loglikelihood_x(x, pi, A, n) + ddirichlet(w, alpha0, K);
  
  convert_x_to_X();
}

void Chain_Factorial::convert_X_to_x(){
  for(int t=0; t<n; t++){
    convert_X_to_x(t);
  }
}

void Chain_Factorial::convert_X_to_x(int t){
  int state = 0;
  for(int i=0; i<K; i++){
    if(X(i, t) == 1){
      state += myPow(2, i);
    }
  }
  x[t] = state;
}

double Chain_Factorial::pointwise_loglik(int t){
  
  // assume that update_mu_for_all_states() has already been done
  int i = x[t];
  return inv_temperature * R::dnorm4(y[t], marginal_mean[i], marginal_sd[i], true);
}

double Chain_Factorial::pointwise_loglik_with_transitions(int t){
  
  double loglik = pointwise_loglik(t);
  
  if(t == 0){
    return mylog(A(x[t], x[t+1])) + loglik;
  } else if(t == n-1){
    return mylog(A(x[t-1], x[t])) + loglik;
  } else {
    return mylog(A(x[t-1], x[t])) + mylog(A(x[t], x[t+1])) + loglik;
  }
}

double Chain_Factorial::get_loglik_cond(){
  // loglik_cond = calculate_posterior_prob(w, h, alpha0, sigma2);
  loglik_cond = 0.0;
  for(int t=0; t<n; t++){
    loglik_cond += mylog(emission_probs(x[t], t));
  }
  loglik_cond += loglikelihood_x(x, pi, A, n);
  return loglik_cond;
}
