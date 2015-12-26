//
// Created by busygin on 12/24/15.
//

#include <cstring>
#include "compute_bart.h"
#include "funs.h"


void compute_bart::fit() {
   //random number generation
   uint seed=99;
   RNG gen(seed); //this one random number generator is used in all draws

   //y stats
   double miny = INFINITY; //use range of y to calibrate prior for bottom node mu's
   double maxy = -INFINITY;
   sinfo allys;       //sufficient stats for all of y, use to initialize the bart trees.
   for(size_t i=0; i<insample_data.n; ++i) {
      double ytemp = insample_data.y[i];
      if(ytemp<miny) miny=ytemp;
      if(ytemp>maxy) maxy=ytemp;
      allys.sy += ytemp; // sum of y
      allys.sy2 += ytemp*ytemp; // sum of y^2
   }
   allys.n = insample_data.n;
   double ybar = allys.sy/allys.n; //sample mean
   double shat = sqrt((allys.sy2-allys.n*ybar*ybar)/(allys.n-1)); //sample standard deviation

   double* y1temp(nullptr);
   if (outsample_data.n) {
      memset(outsample_data.y, 0, outsample_data.n*sizeof(double));
      y1temp = new double[outsample_data.n];
   }

   mcmc_params.tau = (maxy-miny)/(2*run_params.kfac*sqrt((double)run_params.m));
   mcmc_params.sigma = shat;

   //x cutpoints
   makexinfo(insample_data.p, insample_data.n, insample_data.x, xi, run_params.nc);

   //trees
   t.resize(run_params.m);
   for(size_t i=0;i<run_params.m;++i) t[i].setm(ybar/run_params.m); //if you sum the fit over the trees you get the fit.

   //dinfo
   double* allfit = new double[insample_data.n]; //sum of fit of all trees
   for(size_t i=0;i<insample_data.n;++i) allfit[i]=ybar;
   double* r = new double[insample_data.n]; //y-(allfit-ftemp) = y-allfit+ftemp
   double* ftemp = new double[insample_data.n]; //fit of current tree
   dinfo di(insample_data.p, insample_data.n, insample_data.x, r); //the y for each draw will be the residual

   //storage for output
   //in sample fit
   double* pmean = new double[insample_data.n]; //posterior mean of in-sample fit, sum draws,then divide
   for(size_t i=0;i<insample_data.n;++i) pmean[i]=0.0;

   //for sigma draw
   double rss;  //residual sum of squares
   double restemp; //a residual
   double ats; //place for average tree size
   double anb; //place for average number of bottom nodes

   //mcmc
   for(size_t i=0;i<(run_params.nd+run_params.burn);++i) {
      //if(i%100==0) cout << "i: " << i << endl;
      //draw trees
      for(size_t j=0;j<run_params.m;++j) {
         ::fit(t[j],xi,di,ftemp);
         for(size_t k=0;k<insample_data.n;++k) {
            allfit[k] = allfit[k]-ftemp[k];
            r[k] = insample_data.y[k]-allfit[k];
         }
         bd(t[j],xi,di,mcmc_params,gen);
         drmu(t[j],xi,di,mcmc_params,gen);
         ::fit(t[j],xi,di,ftemp);
         for(size_t k=0;k<insample_data.n;++k) allfit[k] += ftemp[k];
      }
      //draw sigma
      rss=0.0;
      for(size_t k=0;k<insample_data.n;++k) {restemp=insample_data.y[k]-allfit[k]; rss += restemp*restemp;}
      mcmc_params.sigma = sqrt((run_params.nu*run_params.lambda + rss)/gen.chi_square(run_params.nu+insample_data.n));
      ats = 0.0; anb=0.0;
      for(size_t k=0;k<run_params.m;++k) {
         ats += t[k].treesize();
         anb += t[k].nbots();
      }
      if(i>=run_params.burn) {
         for(size_t k=0;k<insample_data.n;++k) pmean[k] += allfit[k];
         if (outsample_data.n) {
            for (tree &ti : t) {
               ::fit(ti, xi, outsample_data, y1temp);
               for (size_t k = 0; k < outsample_data.n; ++k) outsample_data.y[k] += y1temp[k];
            }
         }
      }
   }

   for(size_t k=0;k<insample_data.n;++k) pmean[k] /= run_params.nd;

   delete[] pmean;
   delete[] ftemp;
   delete[] r;
   delete[] allfit;
   if (outsample_data.n) {
      for (size_t i = 0; i < outsample_data.n; ++i) outsample_data.y[i] /= run_params.nd;
      delete[] y1temp;
   }
}

void compute_bart::predict() {
   memset(outsample_data.y, 0, outsample_data.n*sizeof(double));
   double* y1 = new double[outsample_data.n];
   for (tree& ti : t) {
      ::fit(ti, xi, outsample_data, y1);
      for(size_t i=0; i<outsample_data.n; ++i) outsample_data.y[i] += y1[i];
   }
   delete[] y1;
}
