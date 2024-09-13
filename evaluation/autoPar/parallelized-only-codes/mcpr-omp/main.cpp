#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include "kernels.h"
// transpose
#include <omp.h> 

double *t(const double *idata,const int width,const int height)
{
  double *odata = (double *)(malloc(sizeof(double ) * width * height));
  for (int yIndex = 0; yIndex <= height - 1; yIndex += 1) {
    for (int xIndex = 0; xIndex <= width - 1; xIndex += 1) {
      int index_in = xIndex + width * yIndex;
      int index_out = yIndex + height * xIndex;
      odata[index_out] = idata[index_in];
    }
  }
  return odata;
}

int main(int argc,char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <path to filename> <repeat>\n",argv[0]);
    return 1;
  }
  char *filename = argv[1];
  const int repeat = atoi(argv[2]);
// n and K should match the dimension of the dataset in the csv file
  const int n = 26280;
  const int K = 21;
  const int M = 10000;
  FILE *fp = fopen(filename,"r");
  if (fp == 0L) {
    printf("Error: failed to open file alphas.csv. Exit\n");
    return 1;
  }
  int alphas_size = n * K;
// n rows and K cols
  int alphas_size_byte = ((n * K) * sizeof(double ));
  int rands_size = M * K;
// M rows and K cols
  int rands_size_byte = ((M * K) * sizeof(double ));
  double *alphas;
  double *rands;
  double *probs;
  alphas = ((double *)(malloc(alphas_size_byte)));
  rands = ((double *)(malloc(rands_size_byte)));
  probs = ((double *)(malloc(alphas_size_byte)));
// load the csv file 
  for (int i = 0; i <= alphas_size - 1; i += 1) {
    fscanf(fp,"%lf",&alphas[i]);
  }
  fclose(fp);
// normal distribution (mean: 0 and var: 1)
  std::mt19937 gen(19937);
  class std::normal_distribution< double  > norm_dist(0.0,1.0);
  for (int i = 0; i <= rands_size - 1; i += 1) {
    rands[i] = norm_dist(gen);
  }
{
// kernel 1
    int threads_per_block = 192;
    int num_blocks = (ceil(1.0 * n / threads_per_block));
    memset(probs,0,alphas_size_byte);
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      compute_probs(alphas,rands,probs,n,K,M,threads_per_block,num_blocks);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
    double s = 0.0;
    
#pragma omp parallel for reduction (+:s)
    for (int i = 0; i <= alphas_size - 1; i += 1) {
      s += probs[i];
    }
    printf("compute_probs: checksum = %lf\n",s);
// kernel 2
    double *t_rands = t(rands,K,M);
    double *t_alphas = t(alphas,K,n);
    memcpy(rands,t_rands,rands_size_byte);
    memcpy(alphas,t_alphas,alphas_size_byte);
    memset(probs,0,alphas_size_byte);
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      compute_probs_unitStrides(alphas,rands,probs,n,K,M,threads_per_block,num_blocks);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
    s = 0.0;
    
#pragma omp parallel for reduction (+:s)
    for (int i = 0; i <= alphas_size - 1; i += 1) {
      s += probs[i];
    }
    printf("compute_probs_unitStrides: checksum = %lf\n",s);
// kernel 3
    threads_per_block = 96;
    num_blocks = (ceil(1.0 * n / threads_per_block));
    memset(probs,0,alphas_size_byte);
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      compute_probs_unitStrides_sharedMem(alphas,rands,probs,n,K,M,threads_per_block,num_blocks);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
    s = 0.0;
    
#pragma omp parallel for reduction (+:s) firstprivate (alphas_size)
    for (int i = 0; i <= alphas_size - 1; i += 1) {
      s += probs[i];
    }
    printf("compute_probs_unitStrides_sharedMem: checksum = %lf\n",s);
    free(t_alphas);
    free(t_rands);
  }
// free memory 
  free(alphas);
  free(rands);
  free(probs);
  return 0;
}
