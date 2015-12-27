#ifndef GUARD_info_h
#define GUARD_info_h

//data
struct dinfo {
   dinfo(size_t p_=0, size_t n_=0, double *x_=nullptr, double *y_=nullptr) { init(p_, n_, x_, y_); }

   void init(size_t p_, size_t n_, double *x_, double *y_) {
      p = p_; n = n_; x = x_; y = y_;
   }

   size_t p;  //number of vars
   size_t n;  //number of observations
   double *x; // jth var of ith obs is *(x + p*i+j)
   double *y; // ith y is *(y+i) or y[i]
};

//prior and mcmc
struct pinfo {
   pinfo(double pb_=0.5, double alpha_=0.95, double beta_=2.0, double tau_=1.0, double sigma_=1.0) {
      init(pb_, alpha_, beta_, tau_, sigma_);
   }

   void init(double pb_, double alpha_, double beta_, double tau_, double sigma_) {
      pb = pb_;
      alpha = alpha_;
      beta = beta_;
      tau = tau_;
      sigma = sigma_;
   }

//mcmc info
   double pb;  //prob of birth
//prior info
   double alpha;
   double beta;
   double tau;
//sigma
   double sigma;
};

// run info
struct rinfo {
   rinfo(size_t nd_=1000, double lambda_=1.0, size_t burn_=100, size_t m_=200, size_t nc_=100, int nu_=3, double kfac_=2.0, bool regression_=true) {
      init(nd_, lambda_, burn_, m_, nc_, nu_, kfac_, regression_);
   }

   void init(size_t nd_, double lambda_, size_t burn_, size_t m_, size_t nc_, int nu_, double kfac_, bool regression_) {
      nd = nd_;
      lambda = lambda_;
      burn = burn_;
      m = m_;
      nc = nc_;
      nu = nu_;
      kfac = kfac_;
      regression = regression_;
   }

   size_t nd;
   double lambda;
   size_t burn;
   size_t m;
   size_t nc;
   int nu;
   double kfac;

   bool regression;
};

//sufficient statistics for 1 node
struct sinfo {
   sinfo(size_t n_=0, double sy_=0.0, double sy2_=0.0) { init(n_, sy_, sy2_); }

   void init(size_t n_, double sy_, double sy2_) {
      n = n_;
      sy = sy_;
      sy2 = sy2_;
   }

   size_t n;
   double sy;
   double sy2;
};

#endif
