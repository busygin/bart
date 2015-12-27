%module bart
%{
#define SWIG_FILE_WITH_INIT
#include "compute_bart.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

%numpy_typemaps(double, NPY_DOUBLE, size_t)

%apply (size_t DIM1, size_t DIM2, double* IN_ARRAY2) {(size_t n, size_t p, double *x)}
%apply (size_t DIM1, double* INPLACE_ARRAY1) {(size_t n, double *y)};


class compute_bart {
public:
   compute_bart() {}

   void set_insample_matrix(size_t n, size_t p, double *x) {
      insample_data.n = n; insample_data.p = p; insample_data.x = x;
   }

   void set_insample_target(size_t n, double *y) { insample_data.y = y; }

   void set_outsample_matrix(size_t n, size_t p, double *x) {
      outsample_data.n = n; outsample_data.p = p; outsample_data.x = x;
   }

   void set_outsample_target(size_t n, double *y) { outsample_data.y = y; }

   void set_mcmc_params(double pb_=0.5, double alpha_=0.95, double beta_=2.0, double tau_=1.0, double sigma_=1.0) {
      mcmc_params.init(pb_, alpha_, beta_, tau_, sigma_);
   }

   void set_run_params(size_t nd_=1000, double lambda_=1.0, size_t burn_=100, size_t m_=200, size_t nc_=100, int nu_=3, double kfac_=2.0) {
      run_params.init(nd_, lambda_, burn_, m_, nc_, nu_, kfac_);
   }

   void fit();
   void predict();

   dinfo insample_data;
   dinfo outsample_data;

   pinfo mcmc_params;
   rinfo run_params;

   xinfo xi;
   std::vector<tree> t;
};
