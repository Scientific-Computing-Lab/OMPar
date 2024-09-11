#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "rand_helpers.h"
#include "constants.h"
#define mul24(a,b) (a)*(b)
#include <omp.h> 

void Hadamard4x4a(float &p,float &q,float &r,float &s)
{
  float t = (p + q + r + s) / 2;
  p = p - t;
  q = q - t;
  r = t - r;
  s = t - s;
}

void Hadamard4x4b(float &p,float &q,float &r,float &s)
{
  float t = (p + q + r + s) / 2;
  p = t - p;
  q = t - q;
  r = r - t;
  s = s - t;
}

int main(int argc,char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
// host buffers
  float *globalPool = (float *)(malloc((4 * WALLACE_TOTAL_POOL_SIZE)));
  
#pragma omp parallel for firstprivate (WALLACE_TOTAL_POOL_SIZE)
  for (unsigned int i = 0; i <= WALLACE_TOTAL_POOL_SIZE - 1; i += 1) {
    float x = (RandN());
    globalPool[i] = x;
  }
  float *rngChi2Corrections = (float *)(malloc((4 * WALLACE_CHI2_COUNT)));
  for (unsigned int i = 0; i <= WALLACE_CHI2_COUNT - 1; i += 1) {
    rngChi2Corrections[i] = (MakeChi2Scale(WALLACE_TOTAL_POOL_SIZE));
  }
  float *randomNumbers = (float *)(malloc((4 * WALLACE_OUTPUT_SIZE)));
  float *chi2Corrections = rngChi2Corrections;
  const unsigned int m_seed = 1;
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {{
        float pool[2049];
{
          const unsigned int lcg_a = 241;
          const unsigned int lcg_c = 59;
          const unsigned int lcg_m = 256;
          const unsigned int mod_mask = lcg_m - 1;
          const unsigned int lid = (omp_get_thread_num());
          const unsigned int gid = (omp_get_team_num());
          const unsigned int offset = WALLACE_POOL_SIZE * gid;
          
#pragma omp parallel for
          for (unsigned int i = 0; i <= ((unsigned int )8) - 1; i += 1) {
            pool[lid + WALLACE_NUM_THREADS * i] = globalPool[offset + lid + WALLACE_NUM_THREADS * i];
          }
          unsigned int t_seed = m_seed;
// Loop generating generatedRandomNumberPools repeatedly
          for (unsigned int loop = 0; loop <= WALLACE_NUM_OUTPUTS_PER_RUN - 1; loop += 1) {
            t_seed = 1664525U * t_seed + 1013904223U & 0xFFFFFFFF;
            unsigned int intermediate_address = loop * (8 * WALLACE_TOTAL_NUM_THREADS) + 8 * WALLACE_NUM_THREADS * gid + lid;
            if (lid == 0) 
              pool[WALLACE_CHI2_OFFSET] = chi2Corrections[gid * WALLACE_NUM_OUTPUTS_PER_RUN + loop];
            float chi2CorrAndScale = pool[WALLACE_CHI2_OFFSET];
            
#pragma omp parallel for firstprivate (intermediate_address,chi2CorrAndScale)
            for (unsigned int i = 0; i <= ((unsigned int )8) - 1; i += 1) {
              randomNumbers[intermediate_address + i * WALLACE_NUM_THREADS] = pool[i * WALLACE_NUM_THREADS + lid] * chi2CorrAndScale;
            }
            float rin0_0;
            float rin1_0;
            float rin2_0;
            float rin3_0;
            float rin0_1;
            float rin1_1;
            float rin2_1;
            float rin3_1;
            for (unsigned int i = 0; i <= WALLACE_NUM_POOL_PASSES - 1; i += 1) {
              unsigned int seed = t_seed + lid & mod_mask;
              seed = seed * lcg_a + lcg_c & mod_mask;
              rin0_0 = pool[seed << 3];
              seed = seed * lcg_a + lcg_c & mod_mask;
              rin1_0 = pool[(seed << 3) + 1];
              seed = seed * lcg_a + lcg_c & mod_mask;
              rin2_0 = pool[(seed << 3) + 2];
              seed = seed * lcg_a + lcg_c & mod_mask;
              rin3_0 = pool[(seed << 3) + 3];
              seed = seed * lcg_a + lcg_c & mod_mask;
              rin0_1 = pool[(seed << 3) + 4];
              seed = seed * lcg_a + lcg_c & mod_mask;
              rin1_1 = pool[(seed << 3) + 5];
              seed = seed * lcg_a + lcg_c & mod_mask;
              rin2_1 = pool[(seed << 3) + 6];
              seed = seed * lcg_a + lcg_c & mod_mask;
              rin3_1 = pool[(seed << 3) + 7];
              Hadamard4x4a(rin0_0,rin1_0,rin2_0,rin3_0);
              pool[0 * WALLACE_NUM_THREADS + lid] = rin0_0;
              pool[1 * WALLACE_NUM_THREADS + lid] = rin1_0;
              pool[2 * WALLACE_NUM_THREADS + lid] = rin2_0;
              pool[3 * WALLACE_NUM_THREADS + lid] = rin3_0;
              Hadamard4x4b(rin0_1,rin1_1,rin2_1,rin3_1);
              pool[4 * WALLACE_NUM_THREADS + lid] = rin0_1;
              pool[5 * WALLACE_NUM_THREADS + lid] = rin1_1;
              pool[6 * WALLACE_NUM_THREADS + lid] = rin2_1;
              pool[7 * WALLACE_NUM_THREADS + lid] = rin3_1;
            }
          }
          
#pragma omp parallel for firstprivate (WALLACE_NUM_THREADS,lid,offset)
          for (unsigned int i = 0; i <= ((unsigned int )8) - 1; i += 1) {
            globalPool[offset + lid + WALLACE_NUM_THREADS * i] = pool[lid + WALLACE_NUM_THREADS * i];
          }
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
    #ifdef DEBUG
// random numbers are different for each i iteration 
    #endif
  }
  free(rngChi2Corrections);
  free(randomNumbers);
  free(globalPool);
  return 0;
}
