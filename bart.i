%module bart
%{
/* Includes the header in the wrapper code */
#include "compute_bart.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%apply (size_t DIM1, size_t DIM2, double* IN_ARRAY1, int DIM1) {(size_t n, size_t p, double *x, double *y, size_t n1, double *x1, double *y1)};

class compute_bart {
public:
   compute_bart() {}

   void set_mcmc_params(double pb_=0.5, double alpha_=0.95, double beta_=2.0, double tau_=1.0, double sigma_=1.0) {
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
