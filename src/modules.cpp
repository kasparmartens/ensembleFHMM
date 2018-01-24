#include "global.h"
#include "Chain.h"
#include "Chain_Factorial.h"
#include "Ensemble_Factorial.h"

RCPP_EXPOSED_CLASS(Chain)
RCPP_EXPOSED_CLASS(Chain_Factorial)

  RCPP_MODULE(module_Chain){

    // exposing the class in R as "Chain"
    class_<Chain>("Chain")

      // constructor
      .constructor<int, int>("Create new component")
         // .constructor<int, int, int, double, double, int, int, bool, int, IntegerMatrix>("Create new component and initialise")


      // .field("N", &Ensemble_Factorial::N)
      // .field("K", &Ensemble_Factorial::K)

      .method("get_x", &Chain::get_x)
      .method("set_temperature", &Chain::set_temperature)
      ;

    class_<Chain_Factorial>("Chain_Factorial")

      .derives<Chain>("Chain")
      // constructor
      // .constructor<IntegerVector, double, int, bool, int, IntegerMatrix>("Create new component and initialise")
      .constructor<NumericVector, int, int, int, double>("Create new component and initialise")

      .method("initialise_pars", &Chain_Factorial::initialise_pars, "Initialise w, transition_probs, x")
      .method("update_x", &Chain_Factorial::update_x)
      .method("update_A", &Chain_Factorial::update_A)
      .method("update_w", &Chain_Factorial::update_w)
      .method("update_w_marginal", &Chain_Factorial::update_w_marginal)
      .method("update_h_marginal", &Chain_Factorial::update_h_marginal)
      .method("update_h", &Chain_Factorial::update_h)
      .method("update_sigma2", &Chain_Factorial::update_sigma2)
      .method("update_emission_probs", &Chain_Factorial::update_emission_probs)
      .method("get_X", &Chain_Factorial::get_Xcopy)
      .method("get_theta", &Chain_Factorial::get_theta)
      .method("get_w", &Chain_Factorial::get_w)
      .method("get_h", &Chain_Factorial::get_h)
      .method("get_mu", &Chain_Factorial::get_mu)
      .method("get_y_pred", &Chain_Factorial::get_y_pred)
      .method("get_marginal_loglik", &Chain_Factorial::get_marginal_loglik)
      .method("get_loglik_cond", &Chain_Factorial::get_loglik_cond)
      .method("update_w_marginal", &Chain_Factorial::update_w_marginal)
      .method("activate_variational", &Chain_Factorial::activate_variational)
      .method("activate_sampling", &Chain_Factorial::activate_sampling)
      .method("update_X_variational", &Chain_Factorial::update_X_variational)
      .method("update_X_variational_single_row", &Chain_Factorial::update_X_variational_single_row)
      ;

    class_<Ensemble_Factorial>("Ensemble_Factorial")

      // constructor
      // .constructor<int, int, double, bool>("Create new component")
      .constructor<NumericVector, int, int, int, int, double>("Create new component and initialise")


      .field("K", &Ensemble_Factorial::K)

      .method("get_chain", &Ensemble_Factorial::get_chain, "Returns chain with a given index")

      .method("set_temperatures", &Ensemble_Factorial::set_temperatures, "Set temperature values for all chains")
      .method("initialise_pars", &Ensemble_Factorial::initialise_pars, "Initialise w_list, transition_probs, x")
      .method("activate_sampling", &Ensemble_Factorial::activate_sampling, "Activate sampling")
      .method("update_emission_probs", &Ensemble_Factorial::update_emission_probs, "Update emission_probs matrix")
      .method("update_x", &Ensemble_Factorial::update_x, "Update x and X (using either HB or block Gibbs)")
      .method("update_w", &Ensemble_Factorial::update_w, "Update mu")
      .method("update_w_marginal", &Ensemble_Factorial::update_w_marginal)
      .method("update_h_marginal", &Ensemble_Factorial::update_h_marginal)
      .method("update_A", &Ensemble_Factorial::update_A)
      .method("update_h", &Ensemble_Factorial::update_h)
      .method("update_sigma2", &Ensemble_Factorial::update_sigma2)

      .method("do_crossover", &Ensemble_Factorial::do_crossover, "Crossover")
      .method("swap_X", &Ensemble_Factorial::swap_X, "Swap X")
      .method("random_crossover_X", &Ensemble_Factorial::random_crossover_X, "Random crossover")
      ;
  }
