//
// Created by busygin on 12/24/15.
//

#ifndef BART_COMPUTE_BART_H
#define BART_COMPUTE_BART_H


#include <cstddef>
#include "info.h"
#include "tree.h"


class compute_bart {
public:
   compute_bart() {}

   void set_mcmc_params(double pb_=0.5, double alpha_=0.95, double beta_=0.5, double tau_=1.0, double sigma_=1.0) {
      mcmc_params.init(pb_, alpha_, beta_, tau_, sigma_);
   }

   void set_run_params(size_t nd_=1000, double lambda_=1.0, size_t burn_=100, size_t m_=200, size_t nc_=100, double nu_=3.0, double kfac_=2.0) {
      run_params.init(nd_, lambda_, burn_, m_, nc_, nu_, kfac_);
   }

   void train(size_t n, size_t p, double *x, double *y, size_t n1=0, double *x1=nullptr, double *y1=nullptr);
   void predict(size_t n, size_t p, double *x, double *y);

   pinfo mcmc_params;
   rinfo run_params;

   xinfo xi;
   std::vector<tree> t;
};


#endif //BART_COMPUTE_BART_H
