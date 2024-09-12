#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <omp.h>
#define max(a,b) ((a<b)?b:a)
#define min(a,b) ((a<b)?a:b)
#include <omp.h> 
const int WSIZE = 12000;
// Walker
const int NSIZE = 2003;
// Values
const int MSIZE = NSIZE * 3 + 3;
// Gradient vectors
const int OSIZE = NSIZE * 9 + 9;
// Hessian Matrices 
const int NSIZE_round = (NSIZE % 16)?NSIZE + 16 - NSIZE % 16 : NSIZE;
const size_t SSIZE = ((size_t )NSIZE_round) * 48 * 48 * 48;
//Coefs size 

void eval_abc(const float *Af,float tx,float *a)
{
  a[0] = ((Af[0] * tx + Af[1]) * tx + Af[2]) * tx + Af[3];
  a[1] = ((Af[4] * tx + Af[5]) * tx + Af[6]) * tx + Af[7];
  a[2] = ((Af[8] * tx + Af[9]) * tx + Af[10]) * tx + Af[11];
  a[3] = ((Af[12] * tx + Af[13]) * tx + Af[14]) * tx + Af[15];
}

inline static void eval_UBspline_3d_s_vgh(const float *coefs_init,const intptr_t xs,const intptr_t ys,const intptr_t zs,float *vals,float *grads,float *hess,const float *a,const float *b,const float *c,const float *da,const float *db,const float *dc,const float *d2a,const float *d2b,const float *d2c,const float dxInv,const float dyInv,const float dzInv)
{
  float h[9];
  float v0 = 0.0f;
  
#pragma omp parallel for
  for (int i = 0; i <= 8; i += 1) {
    h[i] = 0.0f;
  }
  for (int i = 0; i <= 3; i += 1) {
    for (int j = 0; j <= 3; j += 1) {
      float pre20 = d2a[i] * b[j];
      float pre10 = da[i] * b[j];
      float pre00 = a[i] * b[j];
      float pre11 = da[i] * db[j];
      float pre01 = a[i] * db[j];
      float pre02 = a[i] * d2b[j];
      const float *coefs = coefs_init + i * xs + j * ys;
      float sum0 = c[0] * coefs[0] + c[1] * coefs[zs] + c[2] * coefs[zs * 2] + c[3] * coefs[zs * 3];
      float sum1 = dc[0] * coefs[0] + dc[1] * coefs[zs] + dc[2] * coefs[zs * 2] + dc[3] * coefs[zs * 3];
      float sum2 = d2c[0] * coefs[0] + d2c[1] * coefs[zs] + d2c[2] * coefs[zs * 2] + d2c[3] * coefs[zs * 3];
      h[0] += pre20 * sum0;
      h[1] += pre11 * sum0;
      h[2] += pre10 * sum1;
      h[4] += pre02 * sum0;
      h[5] += pre01 * sum1;
      h[8] += pre00 * sum2;
      h[3] += pre10 * sum0;
      h[6] += pre01 * sum0;
      h[7] += pre00 * sum1;
      v0 += pre00 * sum0;
    }
  }
  vals[0] = v0;
  grads[0] = h[3] * dxInv;
  grads[1] = h[6] * dyInv;
  grads[2] = h[7] * dzInv;
  hess[0] = h[0] * dxInv * dxInv;
  hess[1] = h[1] * dxInv * dyInv;
  hess[2] = h[2] * dxInv * dzInv;
  hess[3] = h[1] * dxInv * dyInv;
// Copy hessian elements into lower half of 3x3 matrix
  hess[4] = h[4] * dyInv * dyInv;
  hess[5] = h[5] * dyInv * dzInv;
  hess[6] = h[2] * dxInv * dzInv;
// Copy hessian elements into lower half of 3x3 matrix
  hess[7] = h[5] * dyInv * dzInv;
//Copy hessian elements into lower half of 3x3 matrix
  hess[8] = h[8] * dzInv * dzInv;
}

int main(int argc,char **argv)
{
  float *Af = (float *)(malloc(sizeof(float ) * 16));
  float *dAf = (float *)(malloc(sizeof(float ) * 16));
  float *d2Af = (float *)(malloc(sizeof(float ) * 16));
  Af[0] = (- 0.166667);
  Af[1] = 0.500000;
  Af[2] = (- 0.500000);
  Af[3] = 0.166667;
  Af[4] = 0.500000;
  Af[5] = (- 1.000000);
  Af[6] = 0.000000;
  Af[7] = 0.666667;
  Af[8] = (- 0.500000);
  Af[9] = 0.500000;
  Af[10] = 0.500000;
  Af[11] = 0.166667;
  Af[12] = 0.166667;
  Af[13] = 0.000000;
  Af[14] = 0.000000;
  Af[15] = 0.000000;
  dAf[0] = 0.000000;
  d2Af[0] = 0.000000;
  dAf[1] = (- 0.500000);
  d2Af[1] = 0.000000;
  dAf[2] = 1.000000;
  d2Af[2] = (- 1.000000);
  dAf[3] = (- 0.500000);
  d2Af[3] = 1.000000;
  dAf[4] = 0.000000;
  d2Af[4] = 0.000000;
  dAf[5] = 1.500000;
  d2Af[5] = 0.000000;
  dAf[6] = (- 2.000000);
  d2Af[6] = 3.000000;
  dAf[7] = 0.000000;
  d2Af[7] = (- 2.000000);
  dAf[8] = 0.000000;
  d2Af[8] = 0.000000;
  dAf[9] = (- 1.500000);
  d2Af[9] = 0.000000;
  dAf[10] = 1.000000;
  d2Af[10] = (- 3.00000);
  dAf[11] = 0.500000;
  d2Af[11] = 1.000000;
  dAf[12] = 0.000000;
  d2Af[12] = 0.000000;
  dAf[13] = 0.500000;
  d2Af[13] = 0.000000;
  dAf[14] = 0.000000;
  d2Af[14] = 1.000000;
  dAf[15] = 0.000000;
  d2Af[15] = 0.000000;
  float x = 0.822387;
  float y = 0.989919;
  float z = 0.104573;
  float *walkers_vals = (float *)(malloc(sizeof(float ) * WSIZE * NSIZE));
  float *walkers_grads = (float *)(malloc(sizeof(float ) * WSIZE * MSIZE));
  float *walkers_hess = (float *)(malloc(sizeof(float ) * WSIZE * OSIZE));
  float *walkers_x = (float *)(malloc(sizeof(float ) * WSIZE));
  float *walkers_y = (float *)(malloc(sizeof(float ) * WSIZE));
  float *walkers_z = (float *)(malloc(sizeof(float ) * WSIZE));
  
#pragma omp parallel for firstprivate (x,y,z)
  for (int i = 0; i <= WSIZE - 1; i += 1) {
    walkers_x[i] = (x + i * 1.0 / WSIZE);
    walkers_y[i] = (y + i * 1.0 / WSIZE);
    walkers_z[i] = (z + i * 1.0 / WSIZE);
  }
  float *spline_coefs = (float *)(malloc(sizeof(float ) * SSIZE));
  for (size_t i = 0; i <= SSIZE - 1; i += 1) {
    spline_coefs[i] = (sqrt(0.22 + i * 1.0) * sin(i * 1.0));
  }
  int spline_num_splines = NSIZE;
  int spline_x_grid_start = 0;
  int spline_y_grid_start = 0;
  int spline_z_grid_start = 0;
  int spline_x_grid_num = 45;
  int spline_y_grid_num = 45;
  int spline_z_grid_num = 45;
  int spline_x_stride = NSIZE_round * 48 * 48;
  int spline_y_stride = NSIZE_round * 48;
  int spline_z_stride = NSIZE_round;
  int spline_x_grid_delta_inv = 45;
  int spline_y_grid_delta_inv = 45;
  int spline_z_grid_delta_inv = 45;
  float a[4];
  float b[4];
  float c[4];
  float da[4];
  float db[4];
  float dc[4];
  float d2a[4];
  float d2b[4];
  float d2c[4];
{
    double total_time = 0.0;
    for (int i = 0; i <= WSIZE - 1; i += 1) {
      float x = walkers_x[i];
      float y = walkers_y[i];
      float z = walkers_z[i];
      float *vals = &walkers_vals[i * NSIZE];
      float *grads = &walkers_grads[i * MSIZE];
      float *hess = &walkers_hess[i * OSIZE];
      float ux = x * spline_x_grid_delta_inv;
      float uy = y * spline_y_grid_delta_inv;
      float uz = z * spline_z_grid_delta_inv;
      float ipartx;
      float iparty;
      float ipartz;
      float tx;
      float ty;
      float tz;
      intptr_t xs = spline_x_stride;
      intptr_t ys = spline_y_stride;
      intptr_t zs = spline_z_stride;
      x -= spline_x_grid_start;
      y -= spline_y_grid_start;
      z -= spline_z_grid_start;
      ipartx = ((int )ux);
      tx = ux - ipartx;
      int ix = ((0 < ((int )ipartx)?((int )ipartx) : 0)) < spline_x_grid_num - 1?((0 < ((int )ipartx)?((int )ipartx) : 0)) : spline_x_grid_num - 1;
      iparty = ((int )uy);
      ty = uy - iparty;
      int iy = ((0 < ((int )iparty)?((int )iparty) : 0)) < spline_y_grid_num - 1?((0 < ((int )iparty)?((int )iparty) : 0)) : spline_y_grid_num - 1;
      ipartz = ((int )uz);
      tz = uz - ipartz;
      int iz = ((0 < ((int )ipartz)?((int )ipartz) : 0)) < spline_z_grid_num - 1?((0 < ((int )ipartz)?((int )ipartz) : 0)) : spline_z_grid_num - 1;
      eval_abc(Af,tx,&a[0]);
      eval_abc(Af,ty,&b[0]);
      eval_abc(Af,tz,&c[0]);
      eval_abc(dAf,tx,&da[0]);
      eval_abc(dAf,ty,&db[0]);
      eval_abc(dAf,tz,&dc[0]);
      eval_abc(d2Af,tx,&d2a[0]);
      eval_abc(d2Af,ty,&d2b[0]);
      eval_abc(d2Af,tz,&d2c[0]);
      auto start = std::chrono::_V2::steady_clock::now();
      for (int n = 0; n <= spline_num_splines - 1; n += 1) {
        eval_UBspline_3d_s_vgh((spline_coefs + ix * xs + iy * ys + iz * zs + n),xs,ys,zs,vals + n,grads + n * 3,hess + n * 9,a,b,c,da,db,dc,d2a,d2b,d2c,spline_x_grid_delta_inv,spline_y_grid_delta_inv,spline_z_grid_delta_inv);
      }
      auto end = std::chrono::_V2::steady_clock::now();
      auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
      total_time += time;
    }
    printf("Total kernel execution time %lf (s)\n",total_time * 1e-9);
  }
// collect results for the first walker
  float resVal = 0.0;
  float resGrad = 0.0;
  float resHess = 0.0;
  
#pragma omp parallel for reduction (+:resVal) firstprivate (NSIZE)
  for (int i = 0; i <= NSIZE - 1; i += 1) {
    resVal = resVal + walkers_vals[i];
  }
  
#pragma omp parallel for reduction (+:resGrad) firstprivate (MSIZE)
  for (int i = 0; i <= MSIZE - 1; i += 1) {
    resGrad = resGrad + walkers_grads[i];
  }
  
#pragma omp parallel for reduction (+:resHess) firstprivate (OSIZE)
  for (int i = 0; i <= OSIZE - 1; i += 1) {
    resHess = resHess + walkers_hess[i];
  }
  printf("walkers[0]->collect([resVal resGrad resHess]) = [%e %e %e]\n",resVal,resGrad,resHess);
  free(Af);
  free(dAf);
  free(d2Af);
  free(walkers_vals);
  free(walkers_grads);
  free(walkers_hess);
  free(walkers_x);
  free(walkers_y);
  free(walkers_z);
  free(spline_coefs);
  return 0;
}
