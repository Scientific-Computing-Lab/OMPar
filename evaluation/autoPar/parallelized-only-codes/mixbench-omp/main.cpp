/**
 * This file is the modified read-only mixbench GPU micro-benchmark suite.
 *
 **/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f
#include <omp.h> 

void benchmark_func(float *cd,int grid_dim,int block_dim,int compute_iterations)
{
{
{
      const unsigned int blockSize = block_dim;
      const int stride = blockSize;
      int idx = ((omp_get_team_num()) * blockSize * 8 + (omp_get_thread_num()));
      const int big_stride = ((omp_get_num_teams()) * blockSize * 8);
      float tmps[8];
      for (int k = 0; k <= 3; k += 1) {
        
#pragma omp parallel for private (i) firstprivate (compute_iterations,stride)
        for (int j = 0; j <= 7; j += 1) {
// Load elements (memory intensive part)
          tmps[j] = cd[idx + j * stride + k * big_stride];
// Perform computations (compute intensive part)
          for (int i = 0; i <= compute_iterations - 1; i += 1) {
            tmps[j] = tmps[j] * tmps[j] + ((float )0.1f);
          }
        }
// Multiply add reduction
        float sum = 0;
        
#pragma omp parallel for reduction (+:sum)
        for (int j = 0; j <= 7; j += 2) {
          sum += tmps[j] * tmps[j + 1];
        }
        for (int j = 0; j <= 7; j += 1) {
          cd[idx + k * big_stride] = sum;
        }
      }
    }
  }
}

void mixbenchGPU(long size,int repeat)
{
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n",benchtype);
  float *cd = (float *)(malloc(size * sizeof(float )));
  
#pragma omp parallel for
  for (int i = 0; ((long )i) <= size - 1; i += 1) {
    cd[i] = 0;
  }
  const long reduced_grid_size = size / 8 / 128;
  const int block_dim = 256;
  const int grid_dim = (reduced_grid_size / block_dim);
{
// warmup
    for (int i = 0; i <= repeat - 1; i += 1) {
      benchmark_func(cd,grid_dim,block_dim,i);
    }
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      benchmark_func(cd,grid_dim,block_dim,i);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Total kernel execution time: %f (s)\n",(time * 1e-9f));
  }
// verification
  bool ok = true;
  for (int i = 0; ((long )i) <= size - 1; i += 1) {
    if (cd[i] != 0) {
      if (fabsf(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n",i,cd[i]);
        break; 
      }
    }
  }
  printf("%s\n",(ok?"PASS" : "FAIL"));
  free(cd);
}

int main(int argc,char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  unsigned int datasize = ((8 * 1024 * 1024) * sizeof(float ));
  printf("Buffer size: %dMB\n",datasize / (1024 * 1024));
  mixbenchGPU((8 * 1024 * 1024),repeat);
  return 0;
}
