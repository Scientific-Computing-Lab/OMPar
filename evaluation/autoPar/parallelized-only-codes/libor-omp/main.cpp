//////////////////////////////////////////////////////////////////
//                                                              //
// This software was written by Mike Giles in 2007 based on     //
// C code written by Zhao and Glasserman at Columbia University //
//                                                              //
// It is copyright University of Oxford, and provided under     //
// the terms of the BSD3 license:                               //
// https://opensource.org/licenses/BSD-3-Clause                 //
//                                                              //
// It is provided along with an informal report on              //
// https://people.maths.ox.ac.uk/~gilesm/cuda_old.html          //
//                                                              //
// Note: this was written for CUDA 1.0 and optimised for        //
// execution on an NVIDIA 8800 GTX GPU                          //
//                                                              //
// Mike Giles, 29 April 2021                                    //
//                                                              //
//////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
// parameters for device execution
#define BLOCK_SIZE 64
#define GRID_SIZE 1500
// parameters for LIBOR calculation
#define NN 80
#define NMAT 40
#define L2_SIZE 3280 //NN*(NMAT+1)
#define NOPT 15
#define NPATH 96000
// Monte Carlo LIBOR path calculation
#include <omp.h> 

void path_calc(float *L,const float *z,const float *lambda,const float delta,const int Nmat,const int N)
{
  int i;
  int n;
  float sqez;
  float lam;
  float con1;
  float v;
  float vrat;
  for (n = 0; n <= Nmat - 1; n += 1) {
    sqez = std::sqrt(delta) * z[n];
    v = 0.f;
    for (i = n + 1; i <= N - 1; i += 1) {
      lam = lambda[i - n - 1];
      con1 = delta * lam;
      v += con1 * L[i] / (1.f + delta * L[i]);
      vrat = expf(con1 * v + lam * (sqez - 0.5f * con1));
      L[i] = L[i] * vrat;
    }
  }
}
// forward path calculation storing data
// for subsequent reverse path calculation

void path_calc_b1(float *L,const float *z,float *L2,const float *lambda,const float delta,const int Nmat,const int N)
{
  int i;
  int n;
  float sqez;
  float lam;
  float con1;
  float v;
  float vrat;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= N - 1; i += 1) {
    L2[i] = L[i];
  }
  for (n = 0; n <= Nmat - 1; n += 1) {
    sqez = std::sqrt(delta) * z[n];
    v = 0.f;
    for (i = n + 1; i <= N - 1; i += 1) {
      lam = lambda[i - n - 1];
      con1 = delta * lam;
      v += con1 * L[i] / (1.f + delta * L[i]);
      vrat = expf(con1 * v + lam * (sqez - 0.5f * con1));
      L[i] = L[i] * vrat;
// store these values for reverse path //
      L2[i + (n + 1) * N] = L[i];
    }
  }
}
// reverse path calculation of deltas using stored data

void path_calc_b2(float *L_b,const float *z,const float *L2,const float *lambda,const float delta,const int Nmat,const int N)
{
  int i;
  int n;
  float faci;
  float v1;
  for (n = Nmat - 1; n >= 0; n += -1) {
    v1 = 0.f;
    for (i = N - 1; i >= n + 1; i += -1) {
      v1 += lambda[i - n - 1] * L2[i + (n + 1) * N] * L_b[i];
      faci = delta / (1.f + delta * L2[i + n * N]);
      L_b[i] = L_b[i] * (L2[i + (n + 1) * N] / L2[i + n * N]) + v1 * lambda[i - n - 1] * faci * faci;
    }
  }
}
// calculate the portfolio value v, and its sensitivity to L
// hand-coded reverse mode sensitivity

float portfolio_b(float *L,float *L_b,const float *lambda,const int *maturities,const float *swaprates,const float delta,const int Nmat,const int N,const int Nopt)
{
  int m;
  int n;
  float b;
  float s;
  float swapval;
  float v;
  float B[40];
  float S[40];
  float B_b[40];
  float S_b[40];
  b = 1.f;
  s = 0.f;
  for (m = 0; m <= N - Nmat - 1; m += 1) {
    n = m + Nmat;
    b = b / (1.f + delta * L[n]);
    s = s + delta * b;
    B[m] = b;
    S[m] = s;
  }
  v = 0.f;
  
#pragma omp parallel for private (m)
  for (m = 0; m <= 39; m += 1) {
    B_b[m] = 0.f;
    S_b[m] = 0.f;
  }
  for (n = 0; n <= Nopt - 1; n += 1) {
    m = maturities[n] - 1;
    swapval = B[m] + swaprates[n] * S[m] - 1.f;
    if (swapval < 0) {
      v += - 100.f * swapval;
      S_b[m] += - 100.f * swaprates[n];
      B_b[m] += - 100.f;
    }
  }
  for (m = N - Nmat - 1; m >= 0; m += -1) {
    n = m + Nmat;
    B_b[m] += delta * S_b[m];
    L_b[n] = -B_b[m] * B[m] * (delta / (1.f + delta * L[n]));
    if (m > 0) {
      S_b[m - 1] += S_b[m];
      B_b[m - 1] += B_b[m] / (1.f + delta * L[n]);
    }
  }
// apply discount
  b = 1.f;
  for (n = 0; n <= Nmat - 1; n += 1) {
    b = b / (1.f + delta * L[n]);
  }
  v = b * v;
  
#pragma omp parallel for private (n) firstprivate (delta)
  for (n = 0; n <= Nmat - 1; n += 1) {
    L_b[n] = -v * delta / (1.f + delta * L[n]);
  }
  
#pragma omp parallel for private (n) firstprivate (N,b)
  for (n = Nmat; n <= N - 1; n += 1) {
    L_b[n] = b * L_b[n];
  }
  return v;
}
// calculate the portfolio value v

float portfolio(float *L,const float *lambda,const int *maturities,const float *swaprates,const float delta,const int Nmat,const int N,const int Nopt)
{
  int n;
  int m;
  int i;
  float v;
  float b;
  float s;
  float swapval;
  float B[40];
  float S[40];
  b = 1.f;
  s = 0.f;
  for (n = Nmat; n <= N - 1; n += 1) {
    b = b / (1.f + delta * L[n]);
    s = s + delta * b;
    B[n - Nmat] = b;
    S[n - Nmat] = s;
  }
  v = 0.f;
  
#pragma omp parallel for private (m,swapval,i) reduction (+:v) firstprivate (Nopt)
  for (i = 0; i <= Nopt - 1; i += 1) {
    m = maturities[i] - 1;
    swapval = B[m] + swaprates[i] * S[m] - 1.f;
    if (swapval < 0) 
      v += - 100.f * swapval;
  }
// apply discount //
  b = 1.f;
  for (n = 0; n <= Nmat - 1; n += 1) {
    b = b / (1.f + delta * L[n]);
  }
  v = b * v;
  return v;
}

int main(int argc,char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
// 'h_' prefix - CPU (host) memory space
  float *h_v;
  float *h_Lb;
  float h_lambda[80];
  float h_delta = 0.25f;
  int h_N = 80;
  int h_Nmat = 40;
  int h_Nopt = 15;
  int i;
  int h_maturities[] = {(4), (4), (4), (8), (8), (8), (20), (20), (20), (28), (28), (28), (40), (40), (40)};
  float h_swaprates[] = {(.045f), (.05f), (.055f), (.045f), (.05f), (.055f), (.045f), (.05f), (.055f), (.045f), (.05f), (.055f), (.045f), (.05f), (.055f)};
  double v;
  double Lb;
  bool ok = true;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 79; i += 1) {
    h_lambda[i] = 0.2f;
  }
  h_v = ((float *)(malloc(sizeof(float ) * 96000)));
  h_Lb = ((float *)(malloc(sizeof(float ) * 96000)));
// Execute GPU kernel -- no Greeks
{
// Launch the device computation threads
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      for (int tid = 0; tid <= 95999; tid += 1) {
        const int threadN = 1500 * 64;
        int i;
        int path;
        float L[80];
        float z[80];
/* Monte Carlo LIBOR path calculation*/
        for (path = tid; path <= 95999; path += threadN) {
// initialise the data for current thread
          
#pragma omp parallel for private (i)
          for (i = 0; i <= h_N - 1; i += 1) {
// for real application, z should be randomly generated
            z[i] = 0.3f;
            L[i] = 0.05f;
          }
          path_calc(L,z,h_lambda,h_delta,h_Nmat,h_N);
          h_v[path] = portfolio(L,h_lambda,h_maturities,h_swaprates,h_delta,h_Nmat,h_N,h_Nopt);
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time : %f (s)\n",(time * 1e-9f / repeat));
// Read back GPU results and compute average
    v = 0.0;
    
#pragma omp parallel for private (i) reduction (+:v)
    for (i = 0; i <= 95999; i += 1) {
      v += h_v[i];
    }
    v = v / 96000;
    if (fabs(v - 224.323) > 1e-3) {
      ok = false;
      printf("Expected: 224.323 Actual %15.3f\n",v);
    }
// Execute GPU kernel -- Greeks
// Launch the device computation threads
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      for (int tid = 0; tid <= 95999; tid += 1) {
        const int threadN = 1500 * 64;
        int i;
        int path;
        float L[80];
        float L2[3280];
        float z[80];
        float *L_b = L;
/* Monte Carlo LIBOR path calculation*/
        for (path = tid; path <= 95999; path += threadN) {
// initialise the data for current thread
          
#pragma omp parallel for private (i)
          for (i = 0; i <= h_N - 1; i += 1) {
// for real application, z should be randomly generated
            z[i] = 0.3f;
            L[i] = 0.05f;
          }
          path_calc_b1(L,z,L2,h_lambda,h_delta,h_Nmat,h_N);
          h_v[path] = portfolio_b(L,L_b,h_lambda,h_maturities,h_swaprates,h_delta,h_Nmat,h_N,h_Nopt);
          path_calc_b2(L_b,z,L2,h_lambda,h_delta,h_Nmat,h_N);
          h_Lb[path] = L_b[80 - 1];
        }
      }
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time : %f (s)\n",(time * 1e-9f / repeat));
// Read back GPU results and compute average
  }
  v = 0.0;
  
#pragma omp parallel for private (i) reduction (+:v)
  for (i = 0; i <= 95999; i += 1) {
    v += h_v[i];
  }
  v = v / 96000;
  Lb = 0.0;
  
#pragma omp parallel for private (i) reduction (+:Lb)
  for (i = 0; i <= 95999; i += 1) {
    Lb += h_Lb[i];
  }
  Lb = Lb / 96000;
  if (fabs(v - 224.323) > 1e-3) {
    ok = false;
    printf("Expected: 224.323 Actual %15.3f\n",v);
  }
  if (fabs(Lb - 21.348) > 1e-3) {
    ok = false;
    printf("Expected:  21.348 Actual %15.3f\n",Lb);
  }
  free(h_v);
  free(h_Lb);
  return 0;
}
