/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "binomialOptions.h"
#include "realtype.h"
#define max(a, b) ((a) < (b) ? (b) : (a))
//Preprocessed input option data
#include <omp.h> 
typedef struct {
real S;
real X;
real vDt;
real puByDf;
real pdByDf;}__TOptionData;
// Overloaded shortcut functions for different precision modes
#ifndef DOUBLE_PRECISION

inline float expiryCallValue(float S,float X,float vDt,int i)
{
  float d = S * expf(vDt * (2.0f * i - 2048)) - X;
  return d > 0.0F?d : 0.0F;
}
#else
#endif
// GPU kernel
#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif
// Host-side interface to GPU binomialOptions

extern "C" void binomialOptionsGPU(real *callValue,TOptionData *optionData,int optN,int numIterations)
{
  __TOptionData d_OptionData[1024];
  
#pragma omp parallel for firstprivate (optN)
  for (int i = 0; i <= optN - 1; i += 1) {
    const real T = optionData[i] . T;
    const real R = optionData[i] . R;
    const real V = optionData[i] . V;
    const real dt = T / ((real )2048);
    const real vDt = V * std::sqrt(dt);
    const real rDt = R * dt;
//Per-step interest and discount factors
    const real If = std::exp(rDt);
    const real Df = std::exp(-rDt);
//Values and pseudoprobabilities of upward and downward moves
    const real u = std::exp(vDt);
    const real d = std::exp(-vDt);
    const real pu = (If - d) / (u - d);
    const real pd = ((real )1.0) - pu;
    const real puByDf = pu * Df;
    const real pdByDf = pd * Df;
    d_OptionData[i] . S = ((real )optionData[i] . S);
    d_OptionData[i] . X = ((real )optionData[i] . X);
    d_OptionData[i] . vDt = ((real )vDt);
    d_OptionData[i] . puByDf = ((real )puByDf);
    d_OptionData[i] . pdByDf = ((real )pdByDf);
  }
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= numIterations - 1; i += 1) {{
        real call_exchange[129];
{
          const int tid = omp_get_thread_num();
          const int bid = omp_get_team_num();
          const real S = d_OptionData[bid] . S;
          const real X = d_OptionData[bid] . X;
          const real vDt = d_OptionData[bid] . vDt;
          const real puByDf = d_OptionData[bid] . puByDf;
          const real pdByDf = d_OptionData[bid] . pdByDf;
          real call[17];
          for (int i = 0; i <= 15; i += 1) {
            call[i] = expiryCallValue(S,X,vDt,tid * (2048 / 128) + i);
          }
          if (tid == 0) 
            call_exchange[128] = expiryCallValue(S,X,vDt,2048);
          int final_it = 0 < tid * (2048 / 128) - 1?tid * (2048 / 128) - 1 : 0;
          for (int i = 2048; i >= 1; i += -1) {
            call_exchange[tid] = call[0];
            call[2048 / 128] = call_exchange[tid + 1];
            if (i > final_it) {
              for (int j = 0; j <= 15; j += 1) {
                call[j] = puByDf * call[j + 1] + pdByDf * call[j];
              }
            }
          }
          if (tid == 0) {
            callValue[bid] = call[0];
          }
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time : %f (us)\n",(time * 1e-3f / numIterations));
  }
}
