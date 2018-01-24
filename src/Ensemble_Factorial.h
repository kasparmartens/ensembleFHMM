#ifndef ENSEMBLE_FACTORIAL_H
#define ENSEMBLE_FACTORIAL_H

#include "Chain_Factorial.h"

using namespace Rcpp;

class Ensemble_Factorial{
public:
  int n_chains, K, k, n;
  bool do_parallel_tempering;
  std::vector<Chain_Factorial> chains;
  int crossover_start, crossover_end, crossover_flipped;
  int nrows_crossover;


  Ensemble_Factorial(NumericVector y, int n_chains_, int K_, int k_, int n_, double alpha_):
  chains(std::vector<Chain_Factorial>()) {
    do_parallel_tempering = false;
    n_chains = n_chains_;
    K = K_;
    k = k_;
    n = n_;
    nrows_crossover = K;
    for(int i=0; i<n_chains; i++){
      chains.push_back(Chain_Factorial(y, K_, k_, n_, alpha_));
    }
    crossover_start = 0;
    crossover_end = 0;
    crossover_flipped = 0;
  }
  
  void activate_sampling(bool HB_sampling_, int radius, int nrows_gibbs_, IntegerMatrix all_combinations_){
    for(int i=0; i<n_chains; i++){
      chains[i].activate_sampling(HB_sampling_, radius, nrows_gibbs_, all_combinations_);
    }
  }

  NumericVector get_crossovers(){
    return NumericVector::create(crossover_start, crossover_end, crossover_flipped);
  }

  void set_temperatures(NumericVector temperatures){
    do_parallel_tempering = true;
    for(int i=0; i<n_chains; i++){
      chains[i].set_temperature(temperatures[i]);
    }
  }

  void initialise_pars(NumericVector w, NumericVector transition_probs, IntegerVector x, double h_mu, double h_sd, double sigma2_){
    for(int i=0; i<n_chains; i++){
      chains[i].initialise_pars(w, transition_probs, x, h_mu, h_sd, sigma2_);
    }
  }

  Chain_Factorial get_chain(int i){
    return chains[i];
  }

  // void initialise_pars(NumericVector transition_probs, IntegerVector x){
  //   for(int i=0; i<n_chains; i++){
  //     chains[i].initialise_pars(transition_probs, x);
  //   }
  // }

  void update_emission_probs(){
    for(int i=0; i<n_chains; i++){
      chains[i].update_emission_probs();
    }
  }

  void update_A(){
    for(int i=0; i<n_chains; i++){
      chains[i].update_A();
    }
  }

  void update_w(double sd){
    for(int i=0; i<n_chains; i++){
      chains[i].update_w(sd);
    }
  }
  
  void update_w_marginal(double sd){
    for(int i=0; i<n_chains; i++){
      chains[i].update_w_marginal(sd);
    }
  }
  
  void update_h_marginal(double sd){
    for(int i=0; i<n_chains; i++){
      chains[i].update_h_marginal(sd);
    }
  }
  
  void update_sigma2(double sd){
    for(int i=0; i<n_chains; i++){
      chains[i].update_sigma2(sd);
    }
  }
  
  void update_h(double sd){
    for(int i=0; i<n_chains; i++){
      chains[i].update_h(sd);
    }
  }

  void update_mu_mod();

  void update_x(){
    for(int i=0; i<n_chains; i++){
      chains[i].update_x();
    }
  }

  void scale_marginals(int max_iter, int burnin);

  void uniform_crossover(int i, int j, IntegerVector which_rows);

  void nonuniform_crossover(NumericVector probs, int i, int j, IntegerVector which_rows);

  // just for FHMM
  double crossover_likelihood(int i, int j, int t, IntegerVector which_rows, int m);

  void do_crossover();

  void swap_X();

  void swap_one_row_X();

  void random_crossover_X();

  void random_crossover_one_row_X();

  void helper_row_swap(int i, int j, int k0, int start, int end);

  ListOf<NumericMatrix> get_copy_of_marginals(IntegerVector& which_chains);

};

#endif
