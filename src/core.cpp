// [[Rcpp::depends(RcppArmadillo)]]

#include "Ensemble_Factorial.h"
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <Rcpp/Benchmark/Timer.h>
using namespace Rcpp;
using namespace std;


void compute_P(NumericMatrix PP, double& loglik, NumericVector pi, NumericMatrix A, NumericVector b, int k){
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      PP(r, s) = pi[r] * A(r, s) * b[s];
    }
  }
  double sum = normalise_mat(PP, k, k);
  loglik += log(sum);
}

void compute_P0(NumericMatrix PP, double& loglik, NumericVector pi, NumericVector b, int k){
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      PP(r, s) = pi[r] * b[s];
    }
  }
  double sum = normalise_mat(PP, k, k);
  loglik += log(sum);
}

void compute_Q(NumericMatrix QQ, NumericMatrix PP, NumericVector pi_backward, NumericVector pi_forward, int k){
  for(int s=0; s<k; s++){
    if(pi_forward[s]>0){
      for(int r=0; r<k; r++){
        QQ(r, s) = PP(r, s) * pi_backward[s] / pi_forward[s];
      }
    }
  }
}

void update_mu(NumericVector mu, NumericVector sigma2, NumericVector n_k, NumericVector cluster_sums, double rho, double inv_temp, int k){
  double var, mean;
  for(int i=0; i<k; i++){
    var = 1.0 / (rho + inv_temp * n_k[i] / sigma2[i]);
    mean = inv_temp * var / sigma2[i] * cluster_sums[i];
    mu[i] = R::rnorm(mean, sqrt(var));
  }
}

void update_sigma(NumericVector sigma2, NumericVector n_k, NumericVector ss, double a0, double b0, double inv_temp, int k){
  double sigma2inv, a, b;
  for(int i=0; i<k; i++){
    a = a0 + 0.5 * inv_temp * n_k[i];
    b = b0 + 0.5 * inv_temp * ss[i];
    sigma2inv = R::rgamma(a, 1.0 / b);
    sigma2[i] = 1.0 / sigma2inv;
  }
}

void update_pars_gaussian(NumericVector& y, IntegerVector& x, NumericVector& mu, NumericVector& sigma2, double rho, double inv_temp, double a0, double b0, int k, int n){
  NumericVector n_k(k), cluster_sums(k), cluster_means(k);
  int index;
  // the number of elements in each component and their sums
  for(int t=0; t<n; t++){
    index = x[t];
    n_k[index] += 1;
    cluster_sums[index] += y[t];
  }
  // mean for each component
  for(int i=0; i<k; i++){
    cluster_means[i] = cluster_sums[i] / n_k[i];
  }
  // sum of squares for each component
  NumericVector ss(k);
  for(int t=0; t<n; t++){
    index = x[t];
    ss[index] += pow(y[t] - cluster_means[index], 2);
  }
  // draw sigma from its posterior
  update_sigma(sigma2, n_k, ss, a0, b0, inv_temp, k);
  // draw mu from its posterior mu|sigma2
  update_mu(mu, sigma2, n_k, cluster_sums, rho, inv_temp, k);
}

void update_marginal_distr(ListOf<NumericMatrix> Q, NumericMatrix res, int k, int n){
  arma::mat out(res.begin(), k, n, false);
  for(int t=1; t<=n-1; t++){
    // calculate rowsums of Q[t]
    arma::mat B(Q[t].begin(), k, k, false);
    arma::colvec rowsums = sum(B, 1);
    // assign rowsums to res(_, t-1)
    out.col(t-1) += rowsums;
  }
  // calculate colsums of Q[n-1]
  arma::mat B(Q[n-1].begin(), k, k, false);
  arma::rowvec colsums = sum(B, 0);
  arma::colvec temp = arma::vec(colsums.begin(), k, false, false);
  out.col(n-1) += temp;
}

//' @export
// [[Rcpp::export]]
void forward_step(NumericVector pi, NumericMatrix A, NumericMatrix emission_probs, ListOf<NumericMatrix>& P, double& loglik, int k, int n){
  NumericVector b, colsums(k);
  b = emission_probs(_, 0);
  compute_P0(P[0], loglik, pi, b, k);
  loglik = 0.0;
  for(int t=1; t<n; t++){
    colsums = calculate_colsums(P[t-1], k, k);
    b = emission_probs(_, t);
    compute_P(P[t], loglik, colsums, A, b, k);
  }
}

// FHMM functions

void FHMM_compute_P(NumericMatrix PP, double& loglik, NumericVector pi, NumericMatrix A, NumericVector b, int k,
                    IntegerVector which_states1, IntegerVector which_states2){
  // here k is length(which_states) or equivalently the ncol/nrow of PP
  int i, j;
  for(int s=0; s<k; s++){
    j = which_states2[s];
    for(int r=0; r<k; r++){
      i = which_states1[r];
      //printf("pi[%d] * A(%d, %d) * b[%d]\n", r, i, j, j);
      //printf("pi %e * A %e * b %e \n", pi[r], A(i, j), b[j]);
      PP(r, s) = pi[r] * A(i, j) * b[j] + 1.0e-16;
    }
  }
  double sum = normalise_mat(PP, k, k);
  loglik += log(sum);
}

void FHMM_compute_P0(NumericMatrix PP, double& loglik, NumericVector pi, NumericVector b, int k,
                     IntegerVector which_states){
  // here k is length(which_states) or equivalently the ncol/nrow of PP
  int j;
  for(int s=0; s<k; s++){
    j = which_states[s];
    for(int r=0; r<k; r++){
      PP(r, s) = pi[r] * b[j] + 1.0e-16;
    }
  }
  double sum = normalise_mat(PP, k, k);
  loglik += log(sum);
}

void FHMM_forward_step(NumericVector pi, NumericMatrix A, NumericMatrix emission_probs, ListOf<NumericMatrix>& P, double& loglik, int k, int n,
                       IntegerVector& x, IntegerMatrix all_hamming_balls){
  // here k == nrow(all_hamming_balls)
  NumericVector b, colsums(k);
  b = emission_probs(_, 0);
  FHMM_compute_P0(P[0], loglik, pi, b, k, all_hamming_balls(_, x[0]));
  loglik = 0.0;
  for(int t=1; t<n; t++){
    colsums = calculate_colsums(P[t-1], k, k);
    b = emission_probs(_, t);
    FHMM_compute_P(P[t], loglik, colsums, A, b, k, all_hamming_balls(_, x[t-1]), all_hamming_balls(_, x[t]));
  }
}

void FHMM_backward_sampling(IntegerVector& x, ListOf<NumericMatrix>& P, int k, int n, IntegerMatrix all_hamming_balls){
  NumericVector prob(k);
  NumericMatrix PP;
  IntegerVector x_temp(n);
  IntegerVector possible_values = seq_len(k)-1;
  prob = calculate_colsums(P[n-1], k, k);
  x_temp[n-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  for(int t=n-1; t>0; t--){
    prob = P[t](_, x_temp[t]);
    x_temp[t-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  }
  for(int t=0; t<n; t++){
    x[t] = all_hamming_balls(x_temp[t], x[t]);
  }
}

int mywhich(IntegerVector x, int y, int K){
  int index = 0;
  for(int k=0; k<K; k++){
    if(x[k] == y){
      index = k;
    }
  }
  return index;
}

double FHMM_backward_prob(IntegerVector& x, ListOf<NumericMatrix>& P, int k, int n, IntegerMatrix all_hamming_balls){
  double logprob = 0.0;
  int i, j;
  for(int t=n-1; t>0; t--){
    // find the encoding for x[t] and then: i <- which(temp == x[t])
    i = mywhich(all_hamming_balls(_, x[t]), x[t], k);

    // same for x[t-1]
    j = mywhich(all_hamming_balls(_, x[t-1]), x[t-1], k);

    logprob += mylog(P[t](i, j));
  }
  return logprob;
}

void FHMM_update_A(NumericVector transition_probs, NumericMatrix A, IntegerMatrix mapping, double inv_temperature){
  int m = A.ncol();
  int length = mapping.nrow();
  for(int j=0; j<m; j++){
    for(int i=0; i<m; i++){
      // temp value for A(i, j)
      double temp = 1.0;
      for(int k=0; k<length; k++){
        if(mapping(k, i) == mapping(k, j)){
          temp *= (1-transition_probs[k]);
        } else{
          temp *= transition_probs[k];
        }
      }
      A(i, j) = pow(temp, inv_temperature);
      //dist = hamming_distance(mapping(_, i), mapping(_, j));
      //A(i, j) = pow(transition_prob, dist) * pow(1-transition_prob, length-dist);
    }
  }
}

IntegerVector FHMM_count_transitions(IntegerMatrix X){
  IntegerVector counts(X.nrow());
  for(int t=1; t<X.ncol(); t++){
    for(int i=0; i<X.nrow(); i++){
      if(X(i, t-1) != X(i, t)){
        counts[i] += 1;
      }
    }
  }
  return counts;
}

void sample_within_hamming_ball(IntegerVector& x, int n, IntegerMatrix hamming_balls){
  IntegerVector possible_values;
  for(int t=0; t<n; t++){
    // select u[t] uniformly within the hamming ball centered at x[t] (and overwrite it)
    // printf("prev: %d", x[t]);
    possible_values = hamming_balls(_, x[t]);
    x[t] = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
    // printf(", after: %d\n", x[t]);
  }
}

IntegerVector return_sample_within_hamming_ball(IntegerVector& x, int n, IntegerMatrix hamming_balls){
  IntegerVector possible_values;
  IntegerVector u(n);
  for(int t=0; t<n; t++){
    // select u[t] uniformly within the hamming ball centered at x[t] (and overwrite it)
    possible_values = hamming_balls(_, x[t]);
    u[t] = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
  }
  return u;
}

NumericMatrix emission_probs_mat_gaussian(NumericVector y, NumericVector mu, NumericVector sigma2, int k, int n){
  NumericMatrix out(k, n);
  for(int t=0; t<n; t++){
    for(int i=0; i<k; i++){
      out(i, t) = R::dnorm4(y[t], mu[i], sqrt(sigma2[i]), false);
    }
  }
  out = out / max(out);
  return out;
}

NumericMatrix emission_probs_mat_discrete(IntegerVector y, NumericMatrix B, int k, int n){
  NumericMatrix out(k, n);
  for(int t=0; t<n; t++){
    out(_, t) = B(_, y[t]);
  }
  return out;
}

NumericMatrix temper_emission_probs(NumericMatrix mat, double inv_temperature, int k, int n){
  NumericMatrix out(k, n);
  for(int t=0; t<n; t++){
    for(int i=0; i<k; i++){
      out(i, t) = pow(mat(i, t), inv_temperature);
    }
  }
  out = out / max(out);
  return out;
}

NumericMatrix calculate_emission_probs(NumericVector y, NumericVector w, double inv_temperature, double h, double sigma, double df, int k, int K, int n, IntegerMatrix mapping){
  NumericVector mu_proposed(k);
  for(int j=0; j<k; j++){
    double temp = 0.0;
    for(int i=0; i<K; i++){
      if(mapping(i, j) == 1){
        temp += w(i);
      }
    }
    temp += w[K];
    mu_proposed(j) = h*temp;
  }
  NumericMatrix emission_probs(k, n);
  for(int t=0; t<n; t++){
    for(int i=0; i<k; i++){
      double loglik = my_t_density(y[t], mu_proposed[i], sigma, df);
      // Poisson
      // double loglik = R::dpois(y[t], h*mu_proposed[i]+1e-16, true);
      // Normal
      // double loglik = R::dnorm(y[t], h*mu_proposed[i]+1e-16, sd, true);
      emission_probs(i, t) = exp(inv_temperature * loglik);
    }
  }
  return emission_probs;
}

NumericMatrix calculate_emission_probs_gaussian(NumericVector y, NumericVector w, double inv_temperature, double h, double sd, int k, int K, int n, IntegerMatrix mapping){
  NumericVector mu_proposed(k);
  for(int j=0; j<k; j++){
    double temp = 0.0;
    for(int i=0; i<K; i++){
      if(mapping(i, j) == 1){
        temp += w(i);
      }
    }
    mu_proposed(j) = temp;
  }
  NumericMatrix emission_probs(k, n);
  for(int t=0; t<n; t++){
    for(int i=0; i<k; i++){
      // Poisson
      // double loglik = R::dpois(y[t], h*mu_proposed[i]+1e-16, true);
      // Normal
      double loglik = R::dnorm4(y[t], h*mu_proposed[i]+1e-16, sd, true);
      emission_probs(i, t) = exp(inv_temperature * loglik);
    }
  }
  return emission_probs;
}

// NumericVector gaussian_emission_probs(double y, NumericVector mu, NumericVector sigma, int k){
//   NumericVector out(k);
//   for(int i=0; i<k; i++)
//     out[i] = R::dnorm4(y, mu[i], sigma[i], false);
//   return out;
// }

//' @export
// [[Rcpp::export]]
void backward_sampling(IntegerVector& x, ListOf<NumericMatrix>& P, IntegerVector possible_values, int k, int n){
  NumericVector prob(k);
  NumericMatrix PP;
  prob = calculate_colsums(P[n-1], k, k);
  x[n-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  for(int t=n-1; t>0; t--){
    prob = P[t](_, x[t]);
    x[t-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  }
}



//' @export
// [[Rcpp::export]]
void backward_step(ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, int k, int n){
  NumericVector q_forward(k), q_backward(k);
  Q[n-1] = P[n-1];
  for(int t=n-2; t>=0; t--){
    q_forward = calculate_colsums(P[t], k, k);
    q_backward = calculate_rowsums(Q[t+1], k, k);
    print(q_backward);
    compute_Q(Q[t], P[t], q_backward, q_forward, k);
  }
}


arma::vec normalise_vec(arma::vec x){
  double sum = accu(x);
  x /= sum;
  return x;
}

//' @export
// [[Rcpp::export]]
arma::rowvec forward_backward_mod(arma::vec pi, arma::mat A, arma::mat emission_probs, int k, int n){
  // forward
  arma::mat alpha(k, n);
  alpha.col(0) = normalise_vec(emission_probs.col(0) % pi);
  for(int t=1; t<n; t++){
    alpha.col(t) = normalise_vec(emission_probs.col(t) % (A * alpha.col(t-1)));
  }
  // print(wrap(alpha));
  // backward
  arma::mat beta(k, n);
  beta.col(n-1).ones();
  for(int t=n-1; t>0; t--){
    beta.col(t-1) = normalise_vec(A * (emission_probs.col(t) % beta.col(t)));
  }
  // print(wrap(beta));
  // gamma
  arma::mat gamma(k, n);
  for(int t=0; t<n; t++){
    gamma.col(t) = normalise_vec(alpha.col(t) % beta.col(t));
  }
  arma::rowvec out = gamma.row(1);
  return out;
}

//' @export
// [[Rcpp::export]]
NumericMatrix backward_step_mod(ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, int k, int n){
  NumericMatrix out(k, n);
  NumericVector q_forward(k), q_backward(k);
  Q[n-1] = P[n-1];
  out(_, n-1) = calculate_colsums(P[n-1], k, k);
  for(int t=n-2; t>=0; t--){
    q_forward = calculate_colsums(P[t], k, k);
    q_backward = calculate_rowsums(Q[t+1], k, k);
    out(_, t) = q_backward;
    compute_Q(Q[t], P[t], q_backward, q_forward, k);
  }
  return out;
}

void switching_probabilities(ListOf<NumericMatrix>& Q, NumericVector res, int k, int n){
  for(int t=n-1; t>0; t--){
    res[t-1] = calculate_nondiagonal_sum(Q[t], k);
  }
}

void rdirichlet_vec(NumericVector a, NumericVector res, int k){
  NumericVector temp(k);
  double sum = 0.0;
  for(int i=0; i<k; i++){
    temp[i] = R::rgamma(a[i], 1.0) + 1.0e-16;
    sum += temp[i];
  }
  for(int i=0; i<k; i++){
    res[i] = temp[i] / sum;
  }
}

void rdirichlet_mat(NumericMatrix A, NumericMatrix res, int k, int s){
  NumericVector temp(s);
  for(int i=0; i<k; i++){
    double sum = 0.0;
    for(int j=0; j<s; j++){
      temp[j] = R::rgamma(A(i, j), 1.0) + 1.0e-16;
      sum += temp[j];
    }
    for(int j=0; j<s; j++){
      res(i, j) = temp[j] / sum;
    }
  }
}

void rdirichlet_mat(NumericMatrix A, NumericMatrix res, NumericMatrix Y, double alpha, int k, int s){
  NumericVector temp(s);
  for(int i=0; i<k; i++){
    double sum = 0.0;
    for(int j=0; j<s; j++){
      Y(i, j) = R::rgamma(A(i, j) + alpha, 1.0) + 1.0e-16;
      sum += Y(i, j);
    }
    for(int j=0; j<s; j++){
      res(i, j) = Y(i, j) / sum;
    }
  }
}

void transition_mat_update0(NumericVector pi, const IntegerVector & x, double alpha, int k){
  NumericVector pi_pars(k);
  initialise_const_vec(pi_pars, alpha, k);
  pi_pars[x[0]-1] += 1;
  rdirichlet_vec(pi_pars, pi, k);
}

double random_walk_log_scale(double current_value, double sd){
  double proposal = log(current_value) + R::rnorm(0, sd);
  return exp(proposal);
}

double calculate_logprob(double alpha, NumericMatrix A, NumericMatrix A_pars, double a0, double b0, int k, int s){
  double logprob = 0.0;
  // for each row i of transition matrix
  logprob = R::dgamma(alpha, a0, 1.0/b0, true) + log(alpha);
  for(int i=0; i<k; i++){
    logprob += Rf_lgammafn(s*alpha) - s*Rf_lgammafn(alpha);
    for(int j=0; j<s; j++){
      //logprob += R::dgamma(Y(i, j), alpha + A_pars(i, j), 1.0, true);
      logprob += (alpha-1)*log(A(i, j));
    }
  }
  return logprob;
}


void update_alpha(double& alpha, NumericMatrix Y, NumericMatrix A_pars, double a0, double b0, int k){
  double logprob_current = calculate_logprob(alpha, Y, A_pars, a0, b0, k, k);
  // propose new alpha
  double alpha_proposed = random_walk_log_scale(alpha, 0.3);
  double logprob_proposed = calculate_logprob(alpha_proposed, Y, A_pars, a0, b0, k, k);
  // accept or reject
  if(R::runif(0, 1) < exp(logprob_proposed - logprob_current)){
    alpha = alpha_proposed;
    //printf("new alpha: %f\n", alpha);
  }
}

void gamma_mat_to_dirichlet(NumericMatrix out, NumericMatrix& Y, int k, int s){
  for(int i=0; i<k; i++){
    double sum = 0;
    for(int j=0; j<s; j++){
      sum += Y(i, j);
    }
    for(int j=0; j<s; j++){
      out(i, j) = Y(i, j) / sum;
    }
  }
}


// void transition_mat_update1(NumericMatrix A, const IntegerVector & x, double alpha, int k, int n){
//   NumericMatrix A_pars(k, k), AA(A);
//   initialise_const_mat(A_pars, alpha, k, k);
//   // add 1 to diagonal
//   for(int i=0; i<k; i++)
//     A_pars(i, i) += 1.0;
//   // add transition counts
//   for(int t=0; t<(n-1); t++){
//     A_pars(x[t], x[t+1]) += 1;
//   }
//   rdirichlet_mat(A_pars, AA, k, k);
// }

void transition_mat_update1(NumericMatrix A, NumericMatrix A_pars, const IntegerVector & x, NumericMatrix Y, double alpha, int k, int n){
  initialise_const_mat(A_pars, 0.0, k, k);
  // add 1 to diagonal
  for(int i=0; i<k; i++)
    A_pars(i, i) += 1.0;
  // add transition counts
  for(int t=0; t<(n-1); t++){
    A_pars(x[t], x[t+1]) += 1.0;
  }
  rdirichlet_mat(A_pars, A, Y, alpha, k, k);
}

void transition_mat_update2(NumericMatrix B, const IntegerVector & x, IntegerVector y, double alpha, int k, int s, int n){
  NumericMatrix B_pars(k, s);
  initialise_const_mat(B_pars, alpha, k, s);
  for(int t=0; t<n; t++){
    B_pars(x[t], y[t]) += 1.0;
  }
  rdirichlet_mat(B_pars, B, k, s);
}

void transition_mat_update3(NumericMatrix B, const IntegerVector & x, IntegerVector y, double alpha, int k, int s, int n, double inv_temperature){
  NumericMatrix B_pars(k, s);
  initialise_const_mat(B_pars, alpha, k, s);
  for(int t=0; t<n; t++){
    B_pars(x[t], y[t]) += inv_temperature;
  }
  rdirichlet_mat(B_pars, B, k, s);
}

double loglikelihood(IntegerVector& x, NumericMatrix& emission_probs, int n){
  double loglik = 0.0;
  for(int t=0; t<n; t++){
    loglik += mylog(emission_probs(x[t], t));
  }
  return loglik;
}

// double loglikelihood(IntegerVector& y, IntegerVector& x, NumericMatrix& B, int n){
//   double loglik = 0.0;
//   for(int t=0; t<n; t++){
//     loglik += log(B(x[t], y[t]));
//   }
//   return loglik;
// }

double loglikelihood_x(IntegerVector& x, NumericVector&pi, NumericMatrix& A, int n){
  double loglik = pi[x[0]];
  for(int t=1; t<n; t++){
    loglik += log(A(x[t-1], x[t]));
  }
  return loglik;
}

double loglikelihood_X(IntegerMatrix& X, NumericVector transition_probs){
  int K = X.nrow();
  int n = X.ncol();
  IntegerVector counts = FHMM_count_transitions(X);
  int counts1, counts2;
  double loglik = 0.0;
  for(int k=0; k<K; k++){
    counts1 = counts[k];
    counts2 = (n-1) - counts1;
    loglik = counts1 * log(transition_probs[k]) + counts2 * log(1 - transition_probs[k]);
  }
  return loglik;
}

double marginal_loglikelihood(NumericVector pi, NumericMatrix A, NumericMatrix emission_probs, double inv_temp, int k, int n){
  double loglik = 0.0;
  NumericMatrix PP(k, k);
  NumericVector b;

  NumericMatrix emission_probs_tempered = temper_emission_probs(emission_probs, inv_temp, k, n);

  for(int t=0; t<n; t++){
    b = emission_probs_tempered(_, t);
    for(int s=0; s<k; s++){
      for(int r=0; r<k; r++){
        if(t==0){
          PP(r, s) = pi[r] * b[s];
        }
        else{
          PP(r, s) = pi[r] * A(r, s) * b[s];
        }
      }
    }
    loglik += log(normalise_mat(PP, k, k));
  }
  return loglik;
}

double MH_acceptance_prob_swap_everything(IntegerVector& x1, NumericMatrix& emission_probs1, IntegerVector& x2, NumericMatrix& emission_probs2,
                                          double inv_temp1, double inv_temp2, int n){
  // here, emission_probs are already tempered. Need to "untemper" first
  double loglik1 = 1.0/inv_temp1 * loglikelihood(x1, emission_probs1, n);
  double loglik2 = 1.0/inv_temp2 * loglikelihood(x2, emission_probs2, n);
  double ratio = exp(-(inv_temp1 - inv_temp2)*(loglik1 - loglik2));
  return ratio;
}

double MH_acceptance_prob_swap_pars(NumericVector& pi1, NumericMatrix& A1, NumericMatrix& emission_probs1,
                                    NumericVector& pi2, NumericMatrix& A2, NumericMatrix& emission_probs2,
                                    double inv_temp1, double inv_temp2, int k, int n){
  double ll_12 = marginal_loglikelihood(pi1, A1, emission_probs1, inv_temp2, k, n);
  double ll_21 = marginal_loglikelihood(pi2, A2, emission_probs2, inv_temp1, k, n);
  double ll_11 = marginal_loglikelihood(pi1, A1, emission_probs1, inv_temp1, k, n);
  double ll_22 = marginal_loglikelihood(pi2, A2, emission_probs2, inv_temp2, k, n);
  double ratio = exp(ll_12 + ll_21 - ll_11 - ll_22);
  return ratio;
}

// double MH_acceptance_prob_swap_pars(double marginal_loglik1, double marginal_loglik2, double inv_temp1, double inv_temp2){
//   double ratio = exp(-(inv_temp1 - inv_temp2)*(marginal_loglik1 - marginal_loglik2));
//   return ratio;
// }

double MH_acceptance_prob_swap_x(IntegerVector& x1, NumericVector& pi1, NumericMatrix& A1, NumericMatrix& emission_probs1,
                                 IntegerVector& x2, NumericVector& pi2, NumericMatrix& A2, NumericMatrix& emission_probs2,
                                 int n){
  double logratio_x = loglikelihood_x(x1, pi2, A2, n) + loglikelihood_x(x2, pi1, A1, n) - loglikelihood_x(x1, pi1, A1, n) - loglikelihood_x(x2, pi2, A2, n);
  double logratio_y = loglikelihood(x1, emission_probs2, n) + loglikelihood(x2, emission_probs1, n) - loglikelihood(x2, emission_probs2, n) - loglikelihood(x1, emission_probs1, n);
  double ratio = exp(logratio_x + logratio_y);
  return ratio;
}

void initialise_transition_matrices(NumericVector pi, NumericMatrix A, NumericMatrix B, int k, int s){
  initialise_const_vec(pi, 1.0/k, k);
  initialise_const_mat(B, 1.0/s, k, s);
  // initialise A
  initialise_const_mat(A, 0.5*1.0/k, k, k);
  for(int i=0; i<k; i++)
    A(i, i) += 0.5;
}

//' @export
// [[Rcpp::export]]
List forward_backward_fast(NumericVector pi, NumericMatrix A, NumericMatrix B, IntegerVector y, int k, int n, bool marginal_distr){
  List PP(n), QQ(n);
  for(int t=0; t<n; t++){
    PP[t] = NumericMatrix(k, k);
    QQ[t] = NumericMatrix(k, k);
  }
  ListOf<NumericMatrix> P(PP), Q(QQ);
  double loglik=0.0;

  NumericMatrix emission_probs = emission_probs_mat_discrete(y, B, k, n);
  forward_step(pi, A, emission_probs, P, loglik, k, n);
  // now backward sampling
  IntegerVector x(n);
  IntegerVector possible_values = seq_len(k)-1;
  backward_sampling(x, P, possible_values, k, n);
  // and backward recursion to obtain marginal distributions
  if(marginal_distr) backward_step(P, Q, k, n);

  IntegerVector xx = as<IntegerVector>(wrap(x));
  xx.attr("dim") = R_NilValue;
  return List::create(Rcpp::Named("x_draw") = xx,
                      Rcpp::Named("P") = P,
                      Rcpp::Named("Q") = Q,
                      Rcpp::Named("log_posterior") = loglik);
}

void save_current_iteration(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& trace_switching_prob,
                            IntegerVector x, NumericVector pi, NumericMatrix A, NumericMatrix B, double& loglik, NumericVector switching_prob,
                            int index){
  IntegerVector xx(x.begin(), x.end());
  trace_x[index] = clone(xx);
  trace_pi[index] = clone(pi);
  trace_A[index] = clone(A);
  trace_B[index] = clone(B);
  log_posterior[index] = clone(wrap(loglik));
  trace_switching_prob[index] = clone(switching_prob);
}

// List gibbs_sampling_fast_with_starting_vals(NumericVector pi0, NumericMatrix A0, NumericMatrix B0, IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr, bool is_fixed_B){
//   NumericVector pi(clone(pi0));
//   NumericMatrix A(clone(A0)), B(clone(B0));
//   List PP(n), QQ(n);
//   for(int t=0; t<n; t++){
//     PP[t] = NumericMatrix(k, k);
//     QQ[t] = NumericMatrix(k, k);
//   }
//   ListOf<NumericMatrix> P(PP), Q(QQ);
//   IntegerVector x(n);
//
//   int trace_length, index;
//   trace_length = (max_iter - burnin + (thin - 1)) / thin;
//   List trace_x(trace_length), trace_pi(trace_length), trace_A(trace_length), trace_B(trace_length), trace_switching_prob(trace_length), log_posterior(trace_length);
//   double loglik;
//   IntegerVector possible_values = seq_len(k)-1;
//   NumericVector switching_prob(n-1);
//   NumericMatrix marginal_distr_res(k, n);
//   NumericMatrix emission_probs(k, n);
//
//   for(int iter = 1; iter <= max_iter; iter++){
//     // forward step
//     emission_probs = emission_probs_mat_discrete(y, B, k, n);
//     forward_step(pi, A, emission_probs, P, loglik, k, n);
//     // now backward sampling and nonstochastic backward step
//     backward_sampling(x, P, possible_values, k, n);
//     if(marginal_distr){
//       backward_step(P, Q, k, n);
//       switching_probabilities(Q, switching_prob, k, n);
//       update_marginal_distr(Q, marginal_distr_res, k, n);
//     }
//
//     transition_mat_update0(pi, x, alpha, k);
//     transition_mat_update1(A, x, alpha, k, n);
//     if(!is_fixed_B) transition_mat_update2(B, x, y, alpha, k, s, n);
//
//     if((iter > burnin) && ((iter-1) % thin == 0)){
//       index = (iter - burnin - 1)/thin;
//       save_current_iteration(trace_x, trace_pi, trace_A, trace_B, log_posterior, trace_switching_prob,
//                              x, pi, A, B, loglik, switching_prob, index);
//     }
//     if(iter % 1000 == 0) printf("iter %d\n", iter);
//   }
//   // scale marginal distribution estimates
//   arma::mat out(marginal_distr_res.begin(), k, n, false);
//   out /= (float) (max_iter - burnin);
//
//   return List::create(Rcpp::Named("trace_x") = trace_x,
//                       Rcpp::Named("trace_pi") = trace_pi,
//                       Rcpp::Named("trace_A") = trace_A,
//                       Rcpp::Named("trace_B") = trace_B,
//                       Rcpp::Named("log_posterior") = log_posterior,
//                       Rcpp::Named("switching_prob") = trace_switching_prob,
//                       Rcpp::Named("marginal_distr") = marginal_distr_res);
// }

//
// List gibbs_sampling_fast(IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr, bool is_fixed_B){
//   NumericVector pi(k);
//   NumericMatrix A(k, k), B(k, s);
//   initialise_transition_matrices(pi, A, B, k, s);
//   return gibbs_sampling_fast_with_starting_vals(pi, A, B, y, alpha, k, s, n, max_iter, burnin, thin, marginal_distr, is_fixed_B);
// }

void initialise_mat_list(List& mat_list, int n, int k, int s){
  for(int t=0; t<n; t++){
    mat_list[t] = NumericMatrix(k, s);
  }
}

// crossover of (x, y) at point t, resulting in subsequences
// (Cpp-indexing) 0:t and (t+1):(n)

//' @export
// [[Rcpp::export]]
void crossover(IntegerVector& x, IntegerVector& y, int t){
  int temp;
  for(int i=0; i<=t; i++){
    temp = y[i];
    y[i] = x[i];
    x[i] = temp;
  }
}

//' @export
// [[Rcpp::export]]
void crossover2(IntegerVector& x, IntegerVector& y, int t, int n){
  int temp;
  for(int i=t+1; i<n; i++){
    temp = y[i];
    y[i] = x[i];
    x[i] = temp;
  }
}

//' @export
// [[Rcpp::export]]
void crossover_one_element(IntegerVector& x, IntegerVector& y, int t){
  int temp = y[t];
  y[t] = x[t];
  x[t] = temp;
}

//' @export
// [[Rcpp::export]]
void crossover_mat(IntegerMatrix X, IntegerMatrix Y, int t, IntegerVector which_rows){
  int m = which_rows.size();
  for(int i=0; i<=t; i++){
    crossover_one_column(X, Y, i, which_rows, m);
  }
}

void crossover2_mat(IntegerMatrix X, IntegerMatrix Y, int t, int n, IntegerVector which_rows){
  int m = which_rows.size();
  for(int i=t+1; i<n; i++){
    crossover_one_column(X, Y, i, which_rows, m);
  }
}

void crossover_one_column(IntegerMatrix X, IntegerMatrix Y, int t, IntegerVector which_rows, int m){
  int index, temp;
  for(int k=0; k<m; k++){
    index = which_rows[k];
    temp = Y(index, t);
    Y(index, t) = X(index, t);
    X(index, t) = temp;
  }
}

double crossover_likelihood(const IntegerVector& x, const IntegerVector& y, int t, int n, NumericMatrix Ax, NumericMatrix Ay){
  if((t == 0) || (t==n)){
    return 1.0;
  } else{
    double num = Ax(y[t-1], x[t]) * Ay(x[t-1], y[t]);
    double denom = Ax(x[t-1], x[t]) * Ay(y[t-1], y[t]) + 1.0e-15;
    return num / denom;
  }
}



// void uniform_crossover(IntegerVector& x, IntegerVector& y, int n){
//   IntegerVector possible_values = seq_len(n);
//   int m = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
//   crossover(x, y, m);
// }

// void nonuniform_crossover(IntegerVector& x, IntegerVector& y, NumericVector& probs, int n){
//   IntegerVector possible_values = seq_len(n);
//   int m = as<int>(RcppArmadillo::sample(possible_values, 1, false, probs));
//   crossover(x, y, m);
// }
//
// void nonuniform_crossover2(IntegerVector& x, IntegerVector& y, NumericVector& probs, int n){
//   IntegerVector possible_values = seq_len(2*n);
//   int m = as<int>(RcppArmadillo::sample(possible_values, 1, false, probs));
//   if(m <= n){
//     //printf("normal crossover, m = %d", m);
//     crossover(x, y, m);
//   } else{
//     //printf("flipped crossover, m-n = %d", m-n);
//     crossover(y, x, m-n);
//   }
// }


IntegerVector sample_helper(int n_chains, int n){
  IntegerVector possible_values = seq_len(n_chains)-1;
  IntegerVector out = RcppArmadillo::sample(possible_values, n, false, NumericVector::create());
  return out;
}

int sample_int(int n){
  IntegerVector possible_values = seq_len(n)-1;
  int out = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
  return out;
}

int sample_int(int n, NumericVector probs){
  IntegerVector possible_values = seq_len(n)-1;
  int out = as<int>(RcppArmadillo::sample(possible_values, 1, false, probs));
  return out;
}

void scale_marginal_distr(NumericMatrix marginal_distr_res, int k, int n, int max_iter, int burnin){
  arma::mat out(marginal_distr_res.begin(), k, n, false);
  out /= (float) (max_iter - burnin);
}

void helper_binary_matrix(IntegerMatrix& A, int nrows, int ncols, int i, int start_col, int end_col){
  int middle_col = start_col + (end_col - start_col)/2;
  for(int j=start_col; j<middle_col; j++){
    A(i, j) = 0;
  }
  for(int j=middle_col; j<end_col; j++){
    A(i, j) = 1;
  }
  if(i > 0){
    helper_binary_matrix(A, nrows, ncols, i-1, start_col, middle_col);
    helper_binary_matrix(A, nrows, ncols, i-1, middle_col, end_col);
  }
}

//' @export
// [[Rcpp::export]]
IntegerMatrix decimal_to_binary_mapping(int K){
  int ncols = pow(2, K);
  int nrows = K;
  IntegerMatrix out(nrows, ncols);
  helper_binary_matrix(out, nrows, ncols, K-1, 0, ncols);
  return out;
}

// IntegerMatrix calculate_hamming_dist(IntegerMatrix& mapping){
//   int K = mapping.nrow();
//   int ncols = mapping.ncol();
//   IntegerMatrix dist(ncols, ncols);
//   // for each pair of configurations calculate hamming distance
//   for(int j=0; j<ncols; j++){
//     for(int jj=0; jj<ncols; jj++){
//       for(int i=0; i<K; i++){
//         if(mapping(i, j) != mapping(i, jj)){
//           dist(j, jj) += 1;
//         }
//       }
//     }
//   }
//   return dist;
// }

int hamming_distance(IntegerVector x, IntegerVector y){
  int dist = 0;
  for(int i=0; i<x.size(); i++){
    if(x[i] != y[i]) dist += 1;
  }
  return dist;
}

IntegerVector logical_to_ind(LogicalVector x, int n){
  int length = sum(x);
  IntegerVector out(length);
  int counter = 0;
  for(int i=0; i<n; i++){
    if(x[i]){
      out[counter] = i;
      counter += 1;
    }
  }
  return out;
}

//' @export
// [[Rcpp::export]]
IntegerVector hamming_ball(int index, int radius, IntegerMatrix& mapping){
  int n_states = mapping.ncol();
  LogicalVector boolean(n_states);
  // center of the ball
  IntegerVector x = mapping(_, index);
  // find all elements witihn B(x, r)
  for(int i=0; i<n_states; i++){
    if(hamming_distance(x, mapping(_, i)) <= radius){
      boolean[i] = true;
    }
  }
  IntegerVector out = logical_to_ind(boolean, n_states);
  return out;
}

//' @export
// [[Rcpp::export]]
IntegerMatrix construct_all_hamming_balls(int radius, IntegerMatrix& mapping){
  IntegerVector x = hamming_ball(0, radius, mapping);
  int n_elements_inside_ball = x.size();
  int n_states = mapping.ncol();
  IntegerMatrix out(n_elements_inside_ball, n_states);
  for(int i=0; i<n_states; i++){
    out(_, i) = hamming_ball(i, radius, mapping);
  }
  return out;
}


// helpers for block gibbs sampling

bool subset_rows_match(IntegerVector x, IntegerVector y, IntegerVector which_rows){
  bool match = true;
  for(int i=0; i<which_rows.size(); i++){
    if(x[which_rows[i]] != y[which_rows[i]]){
      match = false;
    }
  }
  return match;
}

//' @export
// [[Rcpp::export]]
IntegerVector construct_restricted_space(int x_t, IntegerVector which_rows_fixed, IntegerMatrix mapping){
  int n_states = mapping.ncol();
  LogicalVector boolean(n_states);
  for(int i=0; i<n_states; i++){
    // check whether mapping(which_rows_fixed, i) matches mapping(which_rows_fixed, x_t)
    boolean[i] = subset_rows_match(mapping(_, i), mapping(_, x_t), which_rows_fixed);
  }
  IntegerVector out = logical_to_ind(boolean, n_states);
  return out;
}

//' @export
// [[Rcpp::export]]
IntegerMatrix construct_all_restricted_space(int k_restricted, IntegerVector which_rows_fixed, IntegerMatrix mapping){
  int n_states = mapping.ncol();
  IntegerMatrix out(k_restricted, n_states);
  for(int i=0; i<n_states; i++){
    out(_, i) = construct_restricted_space(i, which_rows_fixed, mapping);
  }
  return out;
}

// List ensemble_FHMM(int n_chains, NumericVector Y, List w, NumericVector transition_probs, double alpha,
//                    int K, int k, int n, double h, int radius,
//                    int max_iter, int burnin, int thin,
//                    bool estimate_marginals, bool parallel_tempering,
//                    bool crossovers, bool swap_all, bool swap_row, bool random_crossover, bool random_crossover_row,
//                    NumericVector temperatures, int swap_type, int swaps_burnin, int swaps_freq,
//                    IntegerVector which_chains, IntegerVector subsequence, IntegerVector x,
//                    int nrows_crossover, bool HB_sampling, int nrows_gibbs, IntegerMatrix all_combs,
//                    bool update_pars, bool update_X, bool alternative_update){
// 
//   // initialise ensemble of n_chains
//   Ensemble_Factorial ensemble(n_chains, K, k, n, alpha, h, 1.0);
// 
//   ensemble.set_temperatures(temperatures);
// 
//   ensemble.initialise_pars(w[0], transition_probs, x);
// 
//   // if(update_pars){
//   //   for(int i=0; i<100; i++){
//   //     ensemble.update_mu(Y);
//   //   }
//   // }
//   ensemble.update_emission_probs(Y);
// 
//   int index;
//   int n_chains_out = which_chains.size();
//   int trace_length = (max_iter - burnin + (thin - 1)) / thin;
//   int list_length = n_chains_out * trace_length;
//   List tr_x(list_length), tr_X(list_length), tr_Ypred(list_length), tr_A(list_length), tr_mu(list_length), tr_sigma2(list_length), tr_alpha(list_length), tr_switching_prob(list_length), tr_loglik(list_length), tr_loglik_cond(list_length);
//   List tr_crossovers(trace_length);
// 
//   Timer timer;
//   nanotime_t t0, t1;
//   t0 = timer.now();
//   for(int iter = 1; iter <= max_iter; iter++){
// 
//     if(update_X){
//       ensemble.update_x();
//       // ensemble.update_A();
//     }
//     if(update_pars){
//       ensemble.update_mu(Y);
//     }
//     ensemble.update_emission_probs(Y);
// 
//     if(swap_all && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
//       ensemble.swap_X();
//     }
// 
//     if(swap_row && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
//       ensemble.swap_one_row_X();
//     }
// 
//     if(random_crossover && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
//       ensemble.random_crossover_X();
//     }
// 
//     if(random_crossover_row && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
//       ensemble.random_crossover_one_row_X();
//     }
// 
//     if(crossovers && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
//       ensemble.do_crossover();
//     }
// 
//     if(update_X){
//       // ensemble.update_A();
//     }
//     if(update_pars){
//       ensemble.update_mu(Y);
//     }
//     // ensemble.update_emission_probs(Y);
// 
//     if((iter > burnin) && ((iter-1) % thin == 0)){
//       index = (iter - burnin - 1)/thin;
//       ensemble.copy_values_to_trace(which_chains, tr_x, tr_X, tr_Ypred, tr_A, tr_mu, tr_sigma2, tr_alpha, tr_loglik, tr_loglik_cond, tr_switching_prob, index, subsequence);
//       if((iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
//         if(crossovers || swap_all || swap_row || random_crossover || random_crossover_row){
//           tr_crossovers[index] = ensemble.get_crossovers();
//         }
//       }
//     }
//     if(iter % 1000 == 0) printf("iter %d\n", iter);
//   }
// 
//   //ensemble.scale_marginals(max_iter, burnin);
//   //ListOf<NumericMatrix> tr_marginal_distr = ensemble.get_copy_of_marginals(which_chains);
// 
//   t1 = timer.now();
//   return List::create(Rcpp::Named("trace_x") = tr_x,
//                       Rcpp::Named("trace_X") = tr_X,
//                       Rcpp::Named("trace_Ypred") = tr_Ypred,
//                       Rcpp::Named("trace_A") = tr_A,
//                       Rcpp::Named("trace_mu") = tr_mu,
//                       Rcpp::Named("trace_sigma2") = tr_sigma2,
//                       Rcpp::Named("trace_alpha") = tr_alpha,
//                       Rcpp::Named("log_posterior") = tr_loglik,
//                       Rcpp::Named("log_posterior_cond") = tr_loglik_cond,
//                       Rcpp::Named("switching_prob") = tr_switching_prob,
//                       //Rcpp::Named("marginal_distr") = tr_marginal_distr,
//                       //Rcpp::Named("acceptance_ratio") = ensemble.get_acceptance_ratio(),
//                       Rcpp::Named("timer") = t1-t0,
//                       Rcpp::Named("crossovers") = tr_crossovers);
// 
// }

IntegerVector convert_X_to_x(IntegerMatrix X){
  int n = X.ncol();
  int K = X.nrow();
  IntegerVector x(n);
  for(int t=0; t<n; t++){
    int state = 0;
    for(int i=0; i<K; i++){
      if(X(i, t) == 1){
        state += myPow(2, i);
      }
    }
    x[t] = state;
  }
  return x;
}


// double calculate_posterior_prob_mod_numerator(NumericVector y, IntegerMatrix X, NumericVector w, NumericVector transition_probs, int h, double alpha0, int K, int n){
//   // log of p(theta) * p(X) * p(y | X, theta)
//   NumericVector lambdas = calculate_mean_for_all_t(X, w, h, K, n);
//   double num = calculate_posterior_prob(y, lambdas, w, alpha0, K, n) + loglikelihood_X(X, transition_probs);
//   // log of p(X | y, theta)
//   // calculated elsewhere
//   return num;
// }

NumericVector calculate_mean_for_all_t(IntegerMatrix X, NumericVector w, double h, int K, int n){
  NumericVector lambda(n);
  NumericVector h_times_w = h * w;
  for(int t=0; t<n; t++){
    for(int k=0; k<K; k++){
      if(X(k, t) != 0){
        lambda[t] += h_times_w[k];
      }
    }
    lambda[t] += h_times_w[K];
  }
  return lambda;
}

NumericVector RWMH(NumericVector x, int K, double sd){
  NumericVector out(K);
  for(int k=0; k<K; k++){
    out[k] = random_walk_log_scale(x[k], sd);
    // out[k] = R::rgamma(1.0/K, 1.0);
  }
  return out;
}

IntegerVector convert_X_to_x(IntegerMatrix X, int K, int n){
  IntegerVector x(n);
  for(int t=0; t<n; t++){
    // convert_X_to_x(t);
    int state = 0;
    for(int i=0; i<K; i++){
      if(X(i, t) == 1){
        state += myPow(2, i);
      }
    }
    x[t] = state;
  }
  return x;
}

IntegerMatrix convert_x_to_X(IntegerVector x, IntegerMatrix mapping, int K, int n){
  IntegerMatrix X(K, n);
  for(int t=0; t<n; t++){
    // convert_X_to_x(t);
    X(_, t) = mapping(_, x[t]);
  }
  return X;
}
