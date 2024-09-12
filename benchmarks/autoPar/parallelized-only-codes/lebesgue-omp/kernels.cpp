#include <math.h>
#include <omp.h>
#define max(a,b) (a) > (b) ? (a) : (b)
#include <omp.h> 

double lebesgue_function(int n,double x[],int nfun,double xfun[])
{
  double lmax = 0.0;
  double *linterp = (double *)(malloc((n * nfun) * sizeof(double )));
{
    for (int j = 0; j <= nfun - 1; j += 1) {
      
#pragma omp parallel for
      for (int i = 0; i <= n - 1; i += 1) {
        linterp[i * nfun + j] = 1.0;
      }
      
#pragma omp parallel for private (i2)
      for (int i1 = 0; i1 <= n - 1; i1 += 1) {
        for (int i2 = 0; i2 <= n - 1; i2 += 1) {
          if (i1 != i2) 
            linterp[i1 * nfun + j] = linterp[i1 * nfun + j] * (xfun[j] - x[i2]) / (x[i1] - x[i2]);
        }
      }
      double t = 0.0;
      for (int i = 0; i <= n - 1; i += 1) {
        t += fabs(linterp[i * nfun + j]);
      }
      lmax = (lmax > t?lmax : t);
    }
  }
  free(linterp);
  return lmax;
}
