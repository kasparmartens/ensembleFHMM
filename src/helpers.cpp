#include "global.h"

using namespace Rcpp;

void initialise_const_vec(NumericVector pi, double alpha, int length){
  for(int i=0; i<length; i++){
    pi[i] = alpha;
  }
}

void initialise_const_mat(NumericMatrix A, double alpha, int nrow, int ncol){
  for(int i=0; i<nrow; i++){
    for(int j=0; j<ncol; j++){
      A(i, j) = alpha;
    }
  }
}

double calculate_nondiagonal_sum(NumericMatrix mat, int k){
  double sum=0;
  for(int j=0; j<k; j++){
    for(int i=0; i<k; i++){
      if(i != j) sum += mat(i, j);
    }
  }
  return sum;
}

NumericVector calculate_colsums(NumericMatrix A, int m, int n){
  arma::mat B(A.begin(), m, n, false);
  arma::rowvec colsums = sum(B, 0);
  NumericVector out(colsums.begin(), colsums.end());
  return out;
}

NumericVector calculate_rowsums(NumericMatrix A, int m, int n){
  arma::mat B(A.begin(), m, n, false);
  arma::colvec rowsums = sum(B, 1);
  NumericVector out(rowsums.begin(), rowsums.end());
  return out;
}


double normalise_mat(NumericMatrix A, int m, int n){
  // divide all elements of A by the sum of A
  arma::mat B(A.begin(), m, n, false);
  double sum = accu(B);
  B /= sum;
  return sum;
}


IntegerMatrix hamming_distance(NumericMatrix X, int n, int m){
  IntegerMatrix dist(n, n);
  int temp;
  for(int j=0; j<(n-1); j++){
    for(int i=j+1; i<n; i++){
      temp = 0;
      for(int t=0; t<m; t++){
        if(X(i, t) != X(j, t)){
          temp += 1;
        }
      }
      dist(i, j) = temp;
    }
  }
  return dist;
}

int myPow(int x, int p) {
  if (p == 0) return 1;
  if (p == 1) return x;
  return x * myPow(x, p-1);
}

double ddirichlet(NumericVector x, double alpha, int K){
  double logprob = Rf_lgammafn(K*alpha) - K*Rf_lgammafn(alpha);
  for(int k=0; k<K; k++){
    logprob += (alpha-1) * mylog(x[k]);
  }
  return logprob;
}

double mylog(double x){
  return log(x + 1.0e-16);
}

void fit_linear_model(IntegerMatrix XX, NumericVector yy, int n, int p, NumericVector mu) {
  arma::mat X_trans = Rcpp::as<arma::mat>(XX);
  arma::mat X = X_trans.t();
  arma::colvec y(yy.begin(), n, false);

  arma::colvec coef = arma::solve(X, y);    // fit model y ~ X
  arma::colvec res  = y - X*coef;           // residuals

  // std.errors of coefficients
  double s2 = std::inner_product(res.begin(), res.end(), res.begin(), 0.0)/(n - p);

  arma::colvec std_err = arma::sqrt(s2 * arma::diagvec(arma::pinv(arma::trans(X)*X)));

  for(int i=0; i<p; i++){
    mu[i] = coef[i] + R::rnorm(0, std_err[i]);
  }
}


void fit_Bayesian_linear_model(IntegerMatrix XX, NumericVector yy, int n, int p, NumericVector mu, double& sigma) {
  arma::mat X_trans = Rcpp::as<arma::mat>(XX);
  arma::mat X = X_trans.t();
  arma::colvec y(yy.begin(), n, false);

  arma::colvec rho(p);
  rho.fill(0.01);
  arma::mat V_0_inv = diagmat(rho);
  arma::mat V_N_inv = V_0_inv + arma::trans(X)*X;
  arma::mat V_N = inv(V_N_inv);

  arma::colvec w_N = V_N * arma::trans(X) * y;

  double a_0 = 0.1;
  double b_0 = 0.1;
  double a_N = a_0 + 0.5*n;
  double ss = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
  double b_N = b_0 + 0.5*(ss - as_scalar(arma::trans(w_N) * V_N_inv * w_N));

  sigma = 1 / sqrt(R::rgamma(a_N, 1.0/b_N));
  arma::colvec std_err = arma::sqrt(diagvec(V_N));

  for(int i=0; i<p; i++){
    mu[i] = R::rnorm(w_N[i], sigma * std_err[i]);
  }
}

double my_t_density(double y, double mu, double sigma, double df){
  double temp = (y - mu) / sigma;
  // return Rf_lgammafn(0.5*(df+1)) - Rf_lgammafn(0.5*df) - log(df) - 1.14473 - log(sigma) + -0.5*(df+1) * log(1 + 1/df * temp*temp);
  return -log(sigma) + -0.5*(df+1) * log(1 + 1/df * temp*temp);
}

double vec_t_density(NumericVector y, NumericVector mu, double sigma, double df, int n){
  double logp = n * (Rf_lgammafn(0.5*(df+1)) - Rf_lgammafn(0.5*df) - 0.5*log(df) - 0.5*log(3.141593) - log(sigma));
  double one_over_df = 1.0 / df;
  for(int t=0; t<n; t++){
    double temp = (y[t] - mu[t]) / sigma;
    logp -= 0.5*(df+1) * log(1.0 + one_over_df * temp*temp);
    // logp += R::dnorm4(y[t], mu[t], sigma, true);
  }
  return logp;
}
