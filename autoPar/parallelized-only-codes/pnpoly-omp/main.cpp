#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <random>
#include <chrono>
#include <omp.h>
#define VERTICES 600
#define BLOCK_SIZE_X 256
#include <omp.h> 
typedef struct {
float x;
float y;}float2;
#include "kernel.h"

int main(int argc,char *argv[])
{
  if (argc != 2) {
    printf("Usage: ./%s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int nPoints = 2e7;
  const int vertices = 600;
  std::default_random_engine rng(123);
  class std::normal_distribution< float  > distribution(0,1);
  float2 *point = (float2 *)(malloc(sizeof(float2 ) * nPoints));
  for (int i = 0; i <= nPoints - 1; i += 1) {
    point[i] . x = distribution(rng);
    point[i] . y = distribution(rng);
  }
  float2 *vertex = (float2 *)(malloc(vertices * sizeof(float2 )));
  for (int i = 0; i <= vertices - 1; i += 1) {
    float t = ((distribution(rng) * 2.f) * 3.14159265358979323846);
    vertex[i] . x = cosf(t);
    vertex[i] . y = sinf(t);
  }
// kernel results
  int *bitmap_ref = (int *)(malloc(nPoints * sizeof(int )));
  int *bitmap_opt = (int *)(malloc(nPoints * sizeof(int )));
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      pnpoly_base(bitmap_ref,point,vertex,nPoints);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pnpoly_base): %f (s)\n",(time * 1e-9f / repeat));
// performance tuning with tile sizes
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      ::pnpoly_opt< 1 > (bitmap_opt,point,vertex,nPoints);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pnpoly_opt<1>): %f (s)\n",(time * 1e-9f / repeat));
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      ::pnpoly_opt< 2 > (bitmap_opt,point,vertex,nPoints);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pnpoly_opt<2>): %f (s)\n",(time * 1e-9f / repeat));
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      ::pnpoly_opt< 4 > (bitmap_opt,point,vertex,nPoints);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pnpoly_opt<4>): %f (s)\n",(time * 1e-9f / repeat));
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      ::pnpoly_opt< 8 > (bitmap_opt,point,vertex,nPoints);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pnpoly_opt<8>): %f (s)\n",(time * 1e-9f / repeat));
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      ::pnpoly_opt< 16 > (bitmap_opt,point,vertex,nPoints);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pnpoly_opt<16>): %f (s)\n",(time * 1e-9f / repeat));
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      ::pnpoly_opt< 32 > (bitmap_opt,point,vertex,nPoints);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pnpoly_opt<32>): %f (s)\n",(time * 1e-9f / repeat));
    start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      ::pnpoly_opt< 64 > (bitmap_opt,point,vertex,nPoints);
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pnpoly_opt<64>): %f (s)\n",(time * 1e-9f / repeat));
  }
// compare against reference kernel for verification
  int error = memcmp(bitmap_opt,bitmap_ref,nPoints * sizeof(int ));
// double check
  int checksum = 0;
  
#pragma omp parallel for reduction (+:checksum) firstprivate (nPoints)
  for (int i = 0; i <= nPoints - 1; i += 1) {
    checksum += bitmap_opt[i];
  }
  printf("Checksum: %d\n",checksum);
  printf("%s\n",(error?"FAIL" : "PASS"));
  free(vertex);
  free(point);
  free(bitmap_ref);
  free(bitmap_opt);
  return 0;
}
