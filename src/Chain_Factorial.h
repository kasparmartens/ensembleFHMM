#ifndef CHAIN_FACTORIAL_H
#define CHAIN_FACTORIAL_H

#include "global.h"
#include "Chain.h"

using namespace Rcpp;

class Chain_Factorial : public Chain {
  double loglik_marginal, loglik_cond;
  double a_sigma, b_sigma, rho;
  NumericVector mu, transition_probs;
  int K, k_restricted;
  IntegerMatrix X;
  IntegerMatrix mapping, hamming_balls, restricted_space, all_combinations;
  ListOf<NumericMatrix> P_FHMM, Q_FHMM;
  int nrows_gibbs;
  double h;
  bool HB_sampling;
  //NumericVector w, w_unnorm;
  NumericVector y_pred, y_pred_temp;
  NumericVector w, u;
  double alpha0;
  NumericVector mu_all;
  int nrow_Y;
  List restr_space;
  
  arma::mat theta;
  double sigma, sigma2;
  double h_mu, h_sigma2;
  double df;
  
  NumericVector y;
  NumericVector h_values, h_prob;
  NumericVector marginal_mean;
  NumericVector marginal_sd;
  
public:
  Chain_Factorial(NumericVector y_, int K_, int k, int n, double alpha0_) : Chain(k, n){
    a_sigma = 1.0;
    b_sigma = 0.1;
    rho = 0.01;
    df = 5.0;
    K = K_;
    A = NumericMatrix(k, k);
    //mu_all = NumericVector(k);
    transition_probs = NumericVector(K);
    mapping = decimal_to_binary_mapping(K);
    X = IntegerMatrix(K, n);
    
    y = y_;
    
    y_pred = NumericVector(n);
    mu_all = NumericVector(k);
    marginal_mean = NumericVector(k);
    marginal_sd = NumericVector(k);
    //w_unnorm = NumericMatrix(K);
    //w = NumericMatrix(K);
    alpha0 = alpha0_;
    
  }
  
  Chain_Factorial(){
    
  }
  
  void initialise_pars(NumericVector w, NumericVector transition_probs_, IntegerVector x_, double h_mu, double h_sd, double sigma2);
  // void initialise_pars(NumericVector w, NumericVector transition_probs_, IntegerVector x_, NumericVector h_mu, NumericVector h_sd, double sigma2);
  
  void activate_variational(arma::mat theta_){
    theta = theta_;
  }
  
  void activate_sampling(bool HB_sampling_, int radius, int nrows_gibbs_, IntegerMatrix all_combinations_){
    
    HB_sampling = HB_sampling_;
    nrows_gibbs = nrows_gibbs_;
    all_combinations = all_combinations_;
    
    if(HB_sampling){
      // hamming ball sampling
      hamming_balls = construct_all_hamming_balls(radius, mapping);
      k_restricted = hamming_balls.nrow();
    } else{
      // block gibbs sampling
      k_restricted = myPow(2, nrows_gibbs);
      if(nrows_gibbs == K){
        restricted_space = IntegerMatrix(k_restricted, k_restricted);
        for(int i=0; i<k_restricted; i++){
          restricted_space(_, i) = seq_len(k_restricted)-1;
        }
      } else {
        restr_space = List(all_combinations.ncol());
        for(int i=0; i<all_combinations.ncol(); i++){
          IntegerVector which_rows_fixed = all_combinations(_, i);
          restr_space[i] = construct_all_restricted_space(k_restricted, which_rows_fixed, mapping);
        }
      }
    }
    
    List PP(n), QQ(n);
    for(int t=0; t<n; t++){
      PP[t] = NumericMatrix(k_restricted, k_restricted);
      QQ[t] = NumericMatrix(k_restricted, k_restricted);
    }
    P_FHMM = ListOf<NumericMatrix>(PP);
    Q_FHMM = ListOf<NumericMatrix>(QQ);
    
  }
  
  IntegerMatrix& get_X(){
    return X;
  }
  
  double get_h(){
    return h;
  }
  
  arma::mat get_theta(){
    return theta;
  }
  
  IntegerMatrix get_Xcopy(){
    return clone(X);
  }
  
  NumericVector get_w(){
    return clone(w);
  }
  
  NumericVector get_mu(){
    return mu;
  }
  
  NumericVector get_y_pred(){
    y_pred = calculate_mean_for_all_t(X, w, h, K, n);
    return y_pred;
  }
  
  double get_marginal_loglik(NumericVector w_new);
  double get_marginal_loglik_HB(NumericVector w_new, IntegerVector uu, double h);
  
  void update_w_marginal(double sd);
  void update_h_marginal(double sd);
  
  void update_X_variational(arma::rowvec y);
  void update_X_variational_single_row(arma::rowvec y, int kk);
  
  void convert_x_to_X(){
    for(int t=0; t<n; t++){
      convert_x_to_X(t);
    }
  }
  
  void convert_x_to_X(int t){
    X(_, t) = mapping(_, x[t]);
  }
  
  void convert_X_to_x();
  
  void convert_X_to_x(int t);
  
  void update_A();
  
  
  void update_emission_probs();
  
  void update_mu_for_all_states();
  
  void update_sigma2(double sd);
  
  void update_w(double sd);
  
  void update_h(double sd);
  
  double calculate_posterior_prob(NumericVector w_proposed, double h_proposed, double alpha_proposed, double sigma2_proposed);
  
  void update_mu_mod(NumericVector y, IntegerVector xx, IntegerMatrix XX);
  
  void update_x(){
    update_emission_probs();
    if(HB_sampling){
      update_x_HammingBall();
    } else{
      update_x_BlockGibbs();
    }
  }
  
  void update_x_BlockGibbs();
  
  void update_x_HammingBall();
  
  
  double get_loglik(){
    return loglik_marginal;
  }
  
  double get_loglik_cond();
  
  double pointwise_loglik(int t);
  
  double pointwise_loglik_with_transitions(int t);
};

#endif
