#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <omp.h>
// threads per block
#define BS 256
#include "kernel.h"
#include <omp.h> 

double LCG_random_double(uint64_t *seed)
{
  const unsigned long m = 9223372036854775808ULL;
// 2^63
  const unsigned long a = 2806196910506780709ULL;
  const unsigned long c = 1ULL;
   *seed = (a *  *seed + c) % m;
  return ((double )( *seed)) / ((double )m);
}

int main(int argc,char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n",argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int m = (n + 256 - 1) / 256;
// number of groups
  int *nlist = (int *)(malloc(sizeof(int ) * n));
  int *family = (int *)(malloc(sizeof(int ) * m));
  int *n_neigh = (int *)(malloc(sizeof(int ) * m));
  double *damage = (double *)(malloc(sizeof(double ) * m));
  unsigned long seed = 123;
  for (int i = 0; i <= n - 1; i += 1) {
    nlist[i] = (LCG_random_double(&seed) > 0.5?1 : - 1);
  }
  for (int i = 0; i <= m - 1; i += 1) {
    int s = 0;
    
#pragma omp parallel for reduction (+:s)
    for (int j = 0; j <= 255; j += 1) {
      s += (nlist[i * 256 + j] != - 1?1 : 0);
    }
// non-zero values
    family[i] = ((s + 1) + s * LCG_random_double(&seed));
  }
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      damage_of_node(n,nlist,family,n_neigh,damage);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time %f (s)\n",(time * 1e-9f / repeat));
  }
  double sum = 0.0;
  
#pragma omp parallel for reduction (+:sum) firstprivate (m)
  for (int i = 0; i <= m - 1; i += 1) {
    sum += damage[i];
  }
  printf("Checksum: total damage = %lf\n",sum);
  free(nlist);
  free(family);
  free(n_neigh);
  free(damage);
  return 0;
}
