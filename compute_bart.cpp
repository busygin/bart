//
// Created by busygin on 12/24/15.
//

#include <string.h>
#include "compute_bart.h"
#include "funs.h"


void compute_bart::train(size_t n, size_t p, double *x, double *y, size_t n1, double *x1, double *y1) {
   //random number generation
   uint seed=99;
   RNG gen(seed); //this one random number generator is used in all draws

   //y stats
   double miny = INFINITY; //use range of y to calibrate prior for bottom node mu's
   double maxy = -INFINITY;
   sinfo allys;       //sufficient stats for all of y, use to initialize the bart trees.
   for(size_t i=0; i<n; ++i) {
      double ytemp = y[i];
      if(ytemp<miny) miny=ytemp;
      if(ytemp>maxy) maxy=ytemp;
      allys.sy += ytemp; // sum of y
      allys.sy2 += ytemp*ytemp; // sum of y^2
   }
   allys.n = n;
   double ybar = allys.sy/allys.n; //sample mean
   double shat = sqrt((allys.sy2-allys.n*ybar*ybar)/(allys.n-1)); //sample standard deviation

   double *y1temp;
   dinfo dip(p, n1, x1);
   if (n1) {
      y1temp = new double[n1];
      memset(y1, 0, n1*sizeof(double));
   }

   mcmc_params.tau = (maxy-miny)/(2*run_params.kfac*sqrt((double)run_params.m));
   mcmc_params.sigma = shat;

   //x cutpoints
   makexinfo(p, n, x, xi, run_params.nc);

   //trees
   t.resize(run_params.m);
   for(size_t i=0;i<run_params.m;++i) t[i].setm(ybar/run_params.m); //if you sum the fit over the trees you get the fit.

   //dinfo
   double* allfit = new double[n]; //sum of fit of all trees
   for(size_t i=0;i<n;++i) allfit[i]=ybar;
   double* r = new double[n]; //y-(allfit-ftemp) = y-allfit+ftemp
   double* ftemp = new double[n]; //fit of current tree
   dinfo di;
   di.n=n; di.p=p; di.x = x; di.y=r; //the y for each draw will be the residual

   //storage for output
   //in sample fit
   double* pmean = new double[n]; //posterior mean of in-sample fit, sum draws,then divide
   for(size_t i=0;i<n;++i) pmean[i]=0.0;

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
         fit(t[j],xi,di,ftemp);
         for(size_t k=0;k<n;++k) {
            allfit[k] = allfit[k]-ftemp[k];
            r[k] = y[k]-allfit[k];
         }
         bd(t[j],xi,di,mcmc_params,gen);
         drmu(t[j],xi,di,mcmc_params,gen);
         fit(t[j],xi,di,ftemp);
         for(size_t k=0;k<n;++k) allfit[k] += ftemp[k];
      }
      //draw sigma
      rss=0.0;
      for(size_t k=0;k<n;++k) {restemp=y[k]-allfit[k]; rss += restemp*restemp;}
      mcmc_params.sigma = sqrt((run_params.nu*run_params.lambda + rss)/gen.chi_square(run_params.nu+n));
      ats = 0.0; anb=0.0;
      for(size_t k=0;k<run_params.m;++k) {
         ats += t[k].treesize();
         anb += t[k].nbots();
      }
      if(i>=run_params.burn) {
         for(size_t k=0;k<n;++k) pmean[k] += allfit[k];
         if (n1) {
            for (tree &ti : t) {
               fit(ti, xi, dip, y1temp);
               for (size_t k = 0; k < n1; ++k) y1[k] += y1temp[k];
            }
         }
      }
   }

   for(size_t k=0;k<n;++k) pmean[k] /= run_params.nd;

   delete[] pmean;
   delete[] ftemp;
   delete[] r;
   delete[] allfit;
   if (n1) {
      for (size_t i = 0; i < n1; ++i) y1[i] /= run_params.nd;
      delete[] y1temp;
   }
}

void compute_bart::predict(size_t n, size_t p, double *x, double *y) {
   dinfo di(p, n, x);
   double *y1 = new double[n];
   memset(y, 0, n*sizeof(double));
   for (tree& ti : t) {
      fit(ti, xi, di, y1);
      for(size_t i=0; i<n; ++i) y[i] += y1[i];
   }
   delete[] y1;
}
