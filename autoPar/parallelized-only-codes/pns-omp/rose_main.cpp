/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
/*
// Place and Transition are implicitly included in the code
// as the grid is a fixed one
typedef struct {
        float mark;
} Place;
typedef struct {
        int from1, from2;
        int to1, to2;
} Transition;
// this starts from row 0 and col 0
P(r,c)    -> T(r,c)   -> P(r,c+1)  ->
  |            |            |
 \/           \/           \/
T(r+1,c-1)-> P(r+1,c) -> T(r+1,c)  ->
  |            |            |
 \/           \/           \/
P(r+2,c)  -> T(r+2,c) -> P(r+2,c+1)->
  |            |            |
 \/           \/           \/
T(r+3,c-1)-> P(r+3,c) -> T(r+3,c)->
  |            |            |
 \/           \/           \/
*/
#include "rand_gen.cpp"
#include "petri_kernel.cpp"
#include <omp.h> 
static int N;
static int S;
static int T;
static int NSQUARE2;
uint32 host_mt[624];
void PetrinetOnDevice(long long &ktime);
void compute_statistics();
float results[4];
float *h_vars;
int *h_maxs;

long long get_time()
{
  struct timeval tv;
  gettimeofday(&tv,0L);
  return (tv . tv_sec * 1000000 + tv . tv_usec);
}

int main(int argc,char **argv)
{
  if (argc < 4) {
    printf("Usage: %s N S T\n",argv[0]);
    printf("N: the place-transition grid is 2nX2n\nS: the maximum steps in a trajectory\nT: number of trajectories\n");
    return - 1;
  }
  N = atoi(argv[1]);
  if (N < 1) 
    return - 1;
  S = atoi(argv[2]);
  if (S < 1) 
    return - 1;
  T = atoi(argv[3]);
  if (T < 1) 
    return - 1;
  NSQUARE2 = N * (N + N);
// T >= block_num
  h_vars = ((float *)(malloc(T * sizeof(float ))));
  h_maxs = ((int *)(malloc(T * sizeof(int ))));
// compute the simulation on the GPU
  long long ktime = 0;
  auto start = get_time();
  PetrinetOnDevice(ktime);
  auto end = get_time();
  printf("Total device execution time: %.2f s\n",((end - start) / 1e6f));
  compute_statistics();
  free(h_vars);
  free(h_maxs);
  printf("petri N=%d S=%d T=%d\n",N,S,T);
  printf("mean_vars: %f    var_vars: %f\n",results[0],results[1]);
  printf("mean_maxs: %f    var_maxs: %f\n",results[2],results[3]);
  return 0;
}

void compute_statistics()
{
  float sum = 0;
  float sum_vars = 0;
  float sum_max = 0;
  float sum_max_vars = 0;
  int i;
  
#pragma omp parallel for private (i) reduction (+:sum,sum_vars,sum_max,sum_max_vars)
  for (i = 0; i <= T - 1; i += 1) {
    sum += h_vars[i];
    sum_vars += h_vars[i] * h_vars[i];
    sum_max += h_maxs[i];
    sum_max_vars += (h_maxs[i] * h_maxs[i]);
  }
  results[0] = sum / T;
  results[1] = sum_vars / T - results[0] * results[0];
  results[2] = sum_max / T;
  results[3] = sum_max_vars / T - results[2] * results[2];
}

void PetrinetOnDevice(long long &time)
{
// Allocate memory
  int i;
  int unit_size = (NSQUARE2 * (sizeof(int ) + sizeof(char )) + sizeof(float ) + sizeof(int ));
  int block_num = 750000000 / unit_size;
  printf("Number of thread blocks: %d\n",block_num);
  const int places_size_byte = ((unit_size - sizeof(float ) - sizeof(int )) * block_num / sizeof(int ));
  const int places_size_word = (places_size_byte / sizeof(int ));
  int *g_places = (int *)(malloc(places_size_byte));
  float *g_vars = (float *)(malloc(block_num * sizeof(float )));
  int *g_maxs = (int *)(malloc(block_num * sizeof(int )));
  int *p_hmaxs = h_maxs;
  float *p_hvars = h_vars;
{
// Launch the device computation threads!
    for (i = 0; i <= T - block_num - 1; i += block_num) {
      auto start = get_time();
{
        uint32 mt[624];
{
          PetrinetKernel(mt,g_places,g_vars,g_maxs,N,S,5489 * (i + 1));
        }
      }
      auto end = get_time();
      time += end - start;
      memcpy(p_hmaxs,g_maxs,block_num * sizeof(int ));
      memcpy(p_hvars,g_vars,block_num * sizeof(float ));
      p_hmaxs += block_num;
      p_hvars += block_num;
    }
    auto start = get_time();
{
      uint32 mt[624];
{
        PetrinetKernel(mt,g_places,g_vars,g_maxs,N,S,5489 * (i + 1));
      }
    }
    auto end = get_time();
    time += end - start;
    memcpy(p_hmaxs,g_maxs,(T - i) * sizeof(int ));
    memcpy(p_hvars,g_vars,(T - i) * sizeof(float ));
  }
  free(g_places);
  free(g_vars);
  free(g_maxs);
}
