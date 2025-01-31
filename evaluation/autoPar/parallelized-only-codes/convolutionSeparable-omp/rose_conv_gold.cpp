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
#include <omp.h> 

void convolutionRowHost(float *h_Dst,float *h_Src,float *h_Kernel,int imageW,int imageH,int kernelR)
{
  for (int y = 0; y <= imageH - 1; y += 1) {
    for (int x = 0; x <= imageW - 1; x += 1) {
      double sum = 0;
      
#pragma omp parallel for reduction (+:sum)
      for (int k = -kernelR; k <= kernelR; k += 1) {
        int d = x + k;
        if (d >= 0 && d < imageW) 
          sum += (h_Src[y * imageW + d] * h_Kernel[kernelR - k]);
      }
      h_Dst[y * imageW + x] = ((float )sum);
    }
  }
}

void convolutionColumnHost(float *h_Dst,float *h_Src,float *h_Kernel,int imageW,int imageH,int kernelR)
{
  for (int y = 0; y <= imageH - 1; y += 1) {
    for (int x = 0; x <= imageW - 1; x += 1) {
      double sum = 0;
      
#pragma omp parallel for reduction (+:sum)
      for (int k = -kernelR; k <= kernelR; k += 1) {
        int d = y + k;
        if (d >= 0 && d < imageH) 
          sum += (h_Src[d * imageW + x] * h_Kernel[kernelR - k]);
      }
      h_Dst[y * imageW + x] = ((float )sum);
    }
  }
}
