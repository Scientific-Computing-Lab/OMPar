/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
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
#include "conv.h"
#include <omp.h> 

int main(int argc,char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <image width> <image height> <repeat>\n",argv[0]);
    return 1;
  }
  const unsigned int imageW = (atoi(argv[1]));
  const unsigned int imageH = (atoi(argv[2]));
  const int numIterations = atoi(argv[3]);
  float *h_Kernel = (float *)(malloc((2 * 8 + 1) * sizeof(float )));
  float *h_Input = (float *)(malloc((imageW * imageH) * sizeof(float )));
  float *h_Buffer = (float *)(malloc((imageW * imageH) * sizeof(float )));
  float *h_OutputCPU = (float *)(malloc((imageW * imageH) * sizeof(float )));
  float *h_OutputGPU = (float *)(malloc((imageW * imageH) * sizeof(float )));
  srand(2009);
  for (unsigned int i = 0; i <= ((unsigned int )17) - 1; i += 1) {
    h_Kernel[i] = ((float )(rand() % 16));
  }
  for (unsigned int i = 0; i <= imageW * imageH - 1; i += 1) {
    h_Input[i] = ((float )(rand() % 16));
  }
{
//Just a single run or a warmup iteration
    convolutionRows(h_Buffer,h_Input,h_Kernel,imageW,imageH,imageW);
    convolutionColumns(h_OutputGPU,h_Buffer,h_Kernel,imageW,imageH,imageW);
    auto start = std::chrono::_V2::steady_clock::now();
    for (int iter = 0; iter <= numIterations - 1; iter += 1) {
      convolutionRows(h_Buffer,h_Input,h_Kernel,imageW,imageH,imageW);
      convolutionColumns(h_OutputGPU,h_Buffer,h_Kernel,imageW,imageH,imageW);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time %f (s)\n",(time * 1e-9f / numIterations));
  }
  printf("Comparing against Host/C++ computation...\n");
  convolutionRowHost(h_Buffer,h_Input,h_Kernel,imageW,imageH,8);
  convolutionColumnHost(h_OutputCPU,h_Buffer,h_Kernel,imageW,imageH,8);
  double sum = 0;
  double delta = 0;
  double L2norm;
  
#pragma omp parallel for reduction (+:sum,delta) firstprivate (imageW,imageH)
  for (unsigned int i = 0; i <= imageW * imageH - 1; i += 1) {
    delta += ((h_OutputCPU[i] - h_OutputGPU[i]) * (h_OutputCPU[i] - h_OutputGPU[i]));
    sum += (h_OutputCPU[i] * h_OutputCPU[i]);
  }
  L2norm = sqrt(delta / sum);
  printf("Relative L2 norm: %.3e\n\n",L2norm);
  free(h_OutputGPU);
  free(h_OutputCPU);
  free(h_Buffer);
  free(h_Input);
  free(h_Kernel);
  printf("%s\n",(L2norm < 1e-6?"PASS" : "FAIL"));
  return 0;
}
