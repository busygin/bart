//
// Created by busygin on 12/24/15.
//

#include "compute_bart.h"
#include "tree.h"
#include "funs.h"


//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc)
{
   double xinc;

   //compute min and max for each x
   std::vector<double> minx(p,INFINITY);
   std::vector<double> maxx(p,-INFINITY);
   double xx;
   for(size_t i=0;i<p;i++) {
      for(size_t j=0;j<n;j++) {
         xx = *(x+p*j+i);
         if(xx < minx[i]) minx[i]=xx;
         if(xx > maxx[i]) maxx[i]=xx;
      }
   }
   //make grid of nc cutpoints between min and max for each x.
   xi.resize(p);
   for(size_t i=0;i<p;i++) {
      xinc = (maxx[i]-minx[i])/(nc+1.0);
      xi[i].resize(nc);
      for(size_t j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
   }
}


void compute_bart::train() {
   //random number generation
   uint seed=99;
   RNG gen(seed); //this one random number generator is used in all draws

   //y stats
   double miny = INFINITY; //use range of y to calibrate prior for bottom node mu's
   double maxy = -INFINITY;
   sinfo allys;       //sufficient stats for all of y, use to initialize the bart trees.
   for(size_t i=0; i<data.n; ++i) {
      double ytemp = data.y[i];
      if(ytemp<miny) miny=ytemp;
      if(ytemp>maxy) maxy=ytemp;
      allys.sy += ytemp; // sum of y
      allys.sy2 += ytemp*ytemp; // sum of y^2
   }
   allys.n = data.n;
   double ybar = allys.sy/allys.n; //sample mean
   double shat = sqrt((allys.sy2-allys.n*ybar*ybar)/(allys.n-1)); //sample standard deviation

   mcmc_params.tau = (maxy-miny)/(2*run_params.kfac*sqrt((double)run_params.m));
   mcmc_params.sigma = shat;

   //x cutpoints
   xinfo xi;
   makexinfo(data.p, data.n, data.x, xi, run_params.nc);

   //trees
   tree* t = new tree[run_params.m];
   for(size_t i=0;i<run_params.m;++i) t[i].setm(ybar/run_params.m); //if you sum the fit over the trees you get the fit.

   //dinfo
   double* allfit = new double[data.n]; //sum of fit of all trees
   for(size_t i=0;i<data.n;++i) allfit[i]=ybar;
   double* r = new double[data.n]; //y-(allfit-ftemp) = y-allfit+ftemp
   double* ftemp = new double[data.n]; //fit of current tree
   dinfo di;
   di.n=data.n; di.p=data.p; di.x = data.x; di.y=r; //the y for each draw will be the residual

   //storage for output
   //in sample fit
   double* pmean = new double[data.n]; //posterior mean of in-sample fit, sum draws,then divide
   for(size_t i=0;i<data.n;++i) pmean[i]=0.0;

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
         for(size_t k=0;k<data.n;++k) {
            allfit[k] = allfit[k]-ftemp[k];
            r[k] = data.y[k]-allfit[k];
         }
         bd(t[j],xi,di,mcmc_params,gen);
         drmu(t[j],xi,di,mcmc_params,gen);
         fit(t[j],xi,di,ftemp);
         for(size_t k=0;k<data.n;++k) allfit[k] += ftemp[k];
      }
      //draw sigma
      rss=0.0;
      for(size_t k=0;k<data.n;++k) {restemp=data.y[k]-allfit[k]; rss += restemp*restemp;}
      mcmc_params.sigma = sqrt((run_params.nu*run_params.lambda + rss)/gen.chi_square(run_params.nu+data.n));
      ats = 0.0; anb=0.0;
      for(size_t k=0;k<run_params.m;++k) {
         ats += t[k].treesize();
         anb += t[k].nbots();
      }
      if(i>=run_params.burn) {
         for(size_t k=0;k<data.n;++k) pmean[k] += allfit[k];
      }
   }

   for(size_t k=0;k<data.n;++k) pmean[k] /= (double)run_params.nd;


   delete[] pmean;
   delete[] ftemp;
   delete[] r;
   delete[] allfit;
   delete[] t;
}
