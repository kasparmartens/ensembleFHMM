#ifndef CHAIN_H
#define CHAIN_H

#include "global.h"

using namespace Rcpp;

class Chain {
  protected:
    int k, n;
    NumericVector pi, switching_prob;
    NumericMatrix A, A_pars, A_gamma, marginal_distr, emission_probs, emission_probs_tempered;
    IntegerVector x;
    double a0, b0, inv_temperature;
    bool estimate_marginals, is_tempered;
    IntegerVector possible_values;
    ListOf<NumericMatrix> P, Q;

  public:
    Chain();

    Chain(int K, int N);

    IntegerVector& get_x(){
      return x;
    }

    NumericMatrix& get_emission_probs(){
      return emission_probs;
    }

    NumericMatrix& get_emission_probs_tempered(){
      return emission_probs_tempered;
    }

    NumericMatrix& get_A(){
      return A;
    }

    NumericVector& get_pi(){
      return pi;
    }

    double get_inv_temperature(){
      return inv_temperature;
    }

    void set_temperature(double a){
      is_tempered = true;
      inv_temperature = a;
    }

    NumericMatrix get_marginals(){
      return marginal_distr;
    }

    void scale_marginals(int max_iter, int burnin);

    // double calculate_loglik_marginal();
};

#endif
