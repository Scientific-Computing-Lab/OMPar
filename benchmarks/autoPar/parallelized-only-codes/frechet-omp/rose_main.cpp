/*
 * Compute the discrete Frechet distance between two curves specified by
 * discrete ordered points in n-dimensional space.
 *
 * Based on `DiscreteFrechetDist` by Zachary Danziger,
 * http://www.mathworks.com/matlabcentral/fileexchange/ \
 * 31922-discrete-frechet-distance
 *
 * This implementation omits the computation of the coupling sequence. Use
 * this program if you only want to get the DFD, and get it fast.
 *
 * Implements algorithm from
 * [1] T. Erepeat and H. Mannila. Computing discrete Frechet distance.
 *     Technical report 94/64, Christian Doppler Laboratory
 *
 * Copyright (c) 2016, Mikhail Pak
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h> /* sqrt, fabs, fmin, fmax */
#include <random>
#include <chrono>
/* (previously processed: ignoring self-referential macro declaration) macro name = n_d */ 
#include "norm1.h"
#include "norm2.h"
#include "norm3.h"
#include <omp.h> 

void discrete_frechet_distance(const int s,const int n_1,const int n_2,const int repeat)
{
  double *ca;
  double *c1;
  double *c2;
  int k;
/* Index for initialisation of `ca`*/
  int ca_size = ((n_1 * n_2) * sizeof(double ));
  int c1_size = ((n_1 * 10000) * sizeof(double ));
  int c2_size = ((n_2 * 10000) * sizeof(double ));
/* `ca` : Search array (refer to [1], Table 1, matrix `ca`) */
  ca = ((double *)(malloc(ca_size)));
/* `c1` and `c2` : Arrays with the 1st and 2nd curve's points respectively */
  c1 = ((double *)(malloc(c1_size)));
  c2 = ((double *)(malloc(c2_size)));
/* Initialise it with -1.0 */
  
#pragma omp parallel for private (k)
  for (k = 0; k <= n_1 * n_2 - 1; k += 1) {
    ca[k] = - 1.0;
  }
  std::mt19937 gen(19937);
  class std::uniform_real_distribution< double  > dis(- 1.0,1.0);
  for (k = 0; k <= n_1 * 10000 - 1; k += 1) {
    c1[k] = dis(gen);
  }
  for (k = 0; k <= n_2 * 10000 - 1; k += 1) {
    c2[k] = dis(gen);
  }
  auto start = std::chrono::_V2::steady_clock::now();
  if (s == 0) 
    for (k = 0; k <= repeat - 1; k += 1) {
      distance_norm1(n_1,n_2,ca,c1,c2);
    }
   else if (s == 1) 
    for (k = 0; k <= repeat - 1; k += 1) {
      distance_norm2(n_1,n_2,ca,c1,c2);
    }
   else if (s == 2) 
    for (k = 0; k <= repeat - 1; k += 1) {
      distance_norm3(n_1,n_2,ca,c1,c2);
    }
  auto end = std::chrono::_V2::steady_clock::now();
  auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
  printf("Average kernel execution time %f (s)\n",(time * 1e-9f / repeat));
  double checkSum = 0;
  
#pragma omp parallel for private (k) reduction (+:checkSum) firstprivate (n_1,n_2)
  for (k = 0; k <= n_1 * n_2 - 1; k += 1) {
    checkSum += ca[k];
  }
  printf("checkSum: %lf\n",checkSum);
/* Free memory */
  free(ca);
  free(c1);
  free(c2);
}

int main(int argc,char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <n_1> <n_2> <repeat>\n",argv[0]);
    printf("  n_1: number of points of the 1st curve");
    printf("  n_2: number of points of the 2nd curve");
    return 1;
  }
/* `n_1` and `n_2` : Number of points of the 1st and 2nd curves */
  const int n_1 = atoi(argv[1]);
  const int n_2 = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  for (int i = 0; i <= 2; i += 1) {
    discrete_frechet_distance(i,n_1,n_2,repeat);
  }
  return 0;
}
