#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <omp.h> 

void sum(const int teams,const int blocks,const float *array,const int N,unsigned int *count,volatile float *result)
{
{
    bool isLastBlockDone;
    float partialSum;
{
// Each block sums a subset of the input array.
      unsigned int bid = (omp_get_team_num());
      unsigned int num_blocks = teams;
      unsigned int block_size = blocks;
      unsigned int lid = (omp_get_thread_num());
      unsigned int gid = bid * block_size + lid;
      if (lid == 0) 
        partialSum = 0;
      if (gid < N) {
        partialSum += array[gid];
      }
      if (lid == 0) {
// Thread 0 of each block stores the partial sum
// to global memory. The compiler will use 
// a store operation that bypasses the L1 cache
// since the "result" variable is declared as
// volatile. This ensures that the threads of
// the last block will read the correct partial
// sums computed by all other blocks.
        result[bid] = partialSum;
// Thread 0 signals that it is done.
        unsigned int value;
        value = ( *count)++;
// Thread 0 determines if its block is the last
// block to be done.
        isLastBlockDone = value == num_blocks - 1;
      }
// Synchronize to make sure that each thread reads
// the correct value of isLastBlockDone.
      if (isLastBlockDone) {
// The last block sums the partial sums
// stored in result[0 .. num_blocks-1]
        if (lid == 0) 
          partialSum = 0;
        
#pragma omp parallel for reduction (+:partialSum) firstprivate (num_blocks,block_size)
        for (int i = lid; ((unsigned int )i) <= num_blocks - 1; i += block_size) {
          partialSum += result[i];
        }
        if (lid == 0) {
// Thread 0 of last block stores the total sum
// to global memory and resets the count
// varialble, so that the next kernel call
// works properly.
          result[0] = partialSum;
           *count = 0;
        }
      }
    }
  }
}

int main(int argc,char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int blocks = 256;
  const int grids = (N + blocks - 1) / blocks;
  float *h_array = (float *)(malloc(N * sizeof(float )));
  float *h_result = (float *)(malloc(grids * sizeof(float )));
  unsigned int *h_count = (unsigned int *)(malloc(sizeof(unsigned int )));
  h_count[0] = 0;
  bool ok = true;
  double time = 0.0;
  
#pragma omp parallel for
  for (int i = 0; i <= N - 1; i += 1) {
    h_array[i] = - 1.f;
  }
{
    for (int n = 0; n <= repeat - 1; n += 1) {
//#pragma omp target update to (h_array[0:N])
      auto start = std::chrono::_V2::steady_clock::now();
      sum(grids,blocks,h_array,N,h_count,h_result);
      auto end = std::chrono::_V2::steady_clock::now();
      time += (std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count());
      if (h_result[0] != - 1.f * N) {
        ok = false;
        break; 
      }
    }
  }
  if (ok) 
    printf("Average kernel execution time: %f (ms)\n",time * 1e-6f / repeat);
  free(h_array);
  free(h_count);
  free(h_result);
  printf("%s\n",(ok?"PASS" : "FAIL"));
  return 0;
}
