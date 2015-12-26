//
// Created by busygin on 12/25/15.
//


#include "compute_bart.h"


int main(int argc, char **argv) {
   std::ifstream fy("y1.txt", std::ifstream::binary);
   size_t n=0;
   double tmp;
   while(fy>>tmp) ++n;  // compute number of items
   double* y = new double[n];
   fy.clear();
   fy.seekg(0);
   n=0;
   while(fy>>y[n]) ++n;  // compute number of items
   fy.close();
   std::cout << n << " insample items\n";

   std::ifstream fx("x1.txt", std::ifstream::binary);
   size_t np=0;
   while(fx>>tmp) ++np;  // compute number of items
   size_t p = np/n;
   std::cout << p << " features\n";
   double* x = new double[np];
   fx.clear();
   fx.seekg(0);
   np=0;
   while(fx>>x[np]) ++np;  // compute number of items
   fx.close();

   compute_bart bart_regressor;
   bart_regressor.set_insample_matrix(n, p, x);
   bart_regressor.set_insample_target(n, y);
   double* y1 = new double[n];
   bart_regressor.set_outsample_matrix(n, p, x);
   bart_regressor.set_outsample_target(n, y1);
   bart_regressor.fit();
   // bart_regressor.predict(n, p, x, y);

   for(size_t i=0;i<n;++i) std::cout << y1[i] << ' ';
   std::cout << std::endl;

   delete[] y1;
   delete[] x;
   delete[] y;

   return 0;
}