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
/*
 * Walsh transforms belong to a class of generalized Fourier transformations.
 * They have applications in various fields of electrical engineering
 * and numeric theory. In this sample we demonstrate efficient implementation
 * of naturally-ordered Walsh transform
 * (also known as Walsh-Hadamard or Hadamard transform) in CUDA and its
 * particular application to dyadic convolution computation.
 * Refer to excellent Jorg Arndt's "Algorithms for Programmers" textbook
 * http://www.jjj.de/fxt/fxtbook.pdf (Chapter 22)
 *
 * Victor Podlozhnyuk (vpodlozhnyuk@nvidia.com)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#define ELEMENTARY_LOG2SIZE 11
////////////////////////////////////////////////////////////////////////////////
// Reference CPU FWT
////////////////////////////////////////////////////////////////////////////////
#include <omp.h> 
extern "C" void fwtCPU(float *h_Output,float *h_Input,int log2N);
extern "C" void slowWTcpu(float *h_Output,float *h_Input,int log2N);
extern "C" void dyadicConvolutionCPU(float *h_Result,float *h_Data,float *h_Kernel,int log2dataN,int log2kernelN);
////////////////////////////////////////////////////////////////////////////////
// GPU FWT
////////////////////////////////////////////////////////////////////////////////
//#include "kernels.cpp"
////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
const int log2Data = 12;
const int dataN = 1 << log2Data;
const int DATA_SIZE = (dataN * sizeof(float ));
const int log2Kernel = 7;
const int kernelN = 1 << log2Kernel;
const int KERNEL_SIZE = (kernelN * sizeof(float ));
//const double NOPS = 3.0 * (double)dataN * (double)log2Data / 2.0;
////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{
  double delta;
  double ref;
  double sum_delta2;
  double sum_ref2;
  double L2norm;
  int i;
  printf("Data length: %i; kernel length: %i\n",dataN,kernelN);
  printf("Initializing data...\n");
  float *h_Kernel = (float *)(malloc(KERNEL_SIZE));
  float *h_Data = (float *)(malloc(DATA_SIZE));
  float *h_ResultCPU = (float *)(malloc(DATA_SIZE));
  srand(123);
  for (i = 0; i <= kernelN - 1; i += 1) {
    h_Kernel[i] = ((float )(rand())) / ((float )2147483647);
  }
  for (i = 0; i <= dataN - 1; i += 1) {
    h_Data[i] = ((float )(rand())) / ((float )2147483647);
  }
  printf("Running GPU dyadic convolution using Fast Walsh Transform...\n");
  float *d_Kernel = (float *)(malloc(DATA_SIZE));
  float *d_Data = (float *)(malloc(DATA_SIZE));
{
    for (i = 0; i <= 0; i += 1) {
      memset(d_Kernel,0,DATA_SIZE);
      memcpy(d_Kernel,h_Kernel,KERNEL_SIZE);
      memcpy(d_Data,h_Data,DATA_SIZE);
//  fwtBatchGPU(d_Data, 1, log2Data);
      int log2N = log2Data;
      int N = 1 << log2N;
      int M = 1;
      for (; log2N >= 11 + 1; ((log2N -= 2 , N >>= 2) , M <<= 2)) {
        for (int pos = 0; pos <= N - 1; pos += 1) {
          const int stride = N / 4;
          int lo = pos & stride - 1;
          int i0 = (pos - lo << 2) + lo;
          int i1 = i0 + stride;
          int i2 = i1 + stride;
          int i3 = i2 + stride;
          float D0 = d_Data[i0];
          float D1 = d_Data[i1];
          float D2 = d_Data[i2];
          float D3 = d_Data[i3];
          float T;
          T = D0;
          D0 = D0 + D2;
          D2 = T - D2;
          T = D1;
          D1 = D1 + D3;
          D3 = T - D3;
          T = D0;
          d_Data[i0] = D0 + D1;
          d_Data[i1] = T - D1;
          T = D2;
          d_Data[i2] = D2 + D3;
          d_Data[i3] = T - D3;
        }
      }
#ifdef DEBUG
#endif
{
        float s_data[2048];
{
          int lid = omp_get_thread_num();
          int gid = omp_get_team_num();
          int gsz = omp_get_num_threads();
// Handle to thread block group
          const int N = 1 << log2N;
          const int base = gid << log2N;
          const float *d_Src = (d_Data + base);
          float *d_Dst = d_Data + base;
          
#pragma omp parallel for
          for (int pos = lid; pos <= N - 1; pos += gsz) {
            s_data[pos] = d_Src[pos];
          }
//Main radix-4 stages
          const int pos = lid;
          for (int stride = N >> 2; stride >= 0 + 1; stride >>= 2) {
            int lo = pos & stride - 1;
            int i0 = (pos - lo << 2) + lo;
            int i1 = i0 + stride;
            int i2 = i1 + stride;
            int i3 = i2 + stride;
            float D0 = s_data[i0];
            float D1 = s_data[i1];
            float D2 = s_data[i2];
            float D3 = s_data[i3];
            float T;
            T = D0;
            D0 = D0 + D2;
            D2 = T - D2;
            T = D1;
            D1 = D1 + D3;
            D3 = T - D3;
            T = D0;
            s_data[i0] = D0 + D1;
            s_data[i1] = T - D1;
            T = D2;
            s_data[i2] = D2 + D3;
            s_data[i3] = T - D3;
          }
//Do single radix-2 stage for odd power of two
          if ((log2N & 1)) {
            for (int pos = lid; pos <= N / 2 - 1; pos += gsz) {
              int i0 = pos << 1;
              int i1 = i0 + 1;
              float D0 = s_data[i0];
              float D1 = s_data[i1];
              s_data[i0] = D0 + D1;
              s_data[i1] = D0 - D1;
            }
          }
          
#pragma omp parallel for firstprivate (gsz,N)
          for (int pos = lid; pos <= N - 1; pos += gsz) {
            d_Dst[pos] = s_data[pos];
          }
        }
      }
#ifdef DEBUG
#endif
//fwtBatchGPU(d_Kernel, 1, log2N);
      log2N = log2Data;
      N = 1 << log2N;
      M = 1;
      for (; log2N >= 11 + 1; ((log2N -= 2 , N >>= 2) , M <<= 2)) {
        for (int pos = 0; pos <= N - 1; pos += 1) {
          const int stride = N / 4;
          int lo = pos & stride - 1;
          int i0 = (pos - lo << 2) + lo;
          int i1 = i0 + stride;
          int i2 = i1 + stride;
          int i3 = i2 + stride;
          float D0 = d_Kernel[i0];
          float D1 = d_Kernel[i1];
          float D2 = d_Kernel[i2];
          float D3 = d_Kernel[i3];
          float T;
          T = D0;
          D0 = D0 + D2;
          D2 = T - D2;
          T = D1;
          D1 = D1 + D3;
          D3 = T - D3;
          T = D0;
          d_Kernel[i0] = D0 + D1;
          d_Kernel[i1] = T - D1;
          T = D2;
          d_Kernel[i2] = D2 + D3;
          d_Kernel[i3] = T - D3;
        }
      }
#ifdef DEBUG
#endif
{
        float s_data[2048];
{
          int lid = omp_get_thread_num();
          int gid = omp_get_team_num();
          int gsz = omp_get_num_threads();
// Handle to thread block group
          const int N = 1 << log2N;
          const int base = gid << log2N;
          const float *d_Src = (d_Kernel + base);
          float *d_Dst = d_Kernel + base;
          
#pragma omp parallel for
          for (int pos = lid; pos <= N - 1; pos += gsz) {
            s_data[pos] = d_Src[pos];
          }
//Main radix-4 stages
          const int pos = lid;
          for (int stride = N >> 2; stride >= 0 + 1; stride >>= 2) {
            int lo = pos & stride - 1;
            int i0 = (pos - lo << 2) + lo;
            int i1 = i0 + stride;
            int i2 = i1 + stride;
            int i3 = i2 + stride;
            float D0 = s_data[i0];
            float D1 = s_data[i1];
            float D2 = s_data[i2];
            float D3 = s_data[i3];
            float T;
            T = D0;
            D0 = D0 + D2;
            D2 = T - D2;
            T = D1;
            D1 = D1 + D3;
            D3 = T - D3;
            T = D0;
            s_data[i0] = D0 + D1;
            s_data[i1] = T - D1;
            T = D2;
            s_data[i2] = D2 + D3;
            s_data[i3] = T - D3;
          }
//Do single radix-2 stage for odd power of two
          if ((log2N & 1)) {
            for (int pos = lid; pos <= N / 2 - 1; pos += gsz) {
              int i0 = pos << 1;
              int i1 = i0 + 1;
              float D0 = s_data[i0];
              float D1 = s_data[i1];
              s_data[i0] = D0 + D1;
              s_data[i1] = D0 - D1;
            }
          }
          
#pragma omp parallel for firstprivate (gsz,N)
          for (int pos = lid; pos <= N - 1; pos += gsz) {
            d_Dst[pos] = s_data[pos];
          }
        }
      }
#ifdef DEBUG
#endif
      float rcpN = 1.0f / ((float )dataN);
      
#pragma omp parallel for firstprivate (rcpN)
      for (int pos = 0; pos <= dataN - 1; pos += 1) {
        d_Data[pos] *= d_Kernel[pos] * rcpN;
      }
//  fwtBatchGPU(d_Data, 1, log2N);
      log2N = log2Data;
      N = 1 << log2N;
      M = 1;
      for (; log2N >= 11 + 1; ((log2N -= 2 , N >>= 2) , M <<= 2)) {
        for (int pos = 0; pos <= N - 1; pos += 1) {
          const int stride = N / 4;
          int lo = pos & stride - 1;
          int i0 = (pos - lo << 2) + lo;
          int i1 = i0 + stride;
          int i2 = i1 + stride;
          int i3 = i2 + stride;
          float D0 = d_Data[i0];
          float D1 = d_Data[i1];
          float D2 = d_Data[i2];
          float D3 = d_Data[i3];
          float T;
          T = D0;
          D0 = D0 + D2;
          D2 = T - D2;
          T = D1;
          D1 = D1 + D3;
          D3 = T - D3;
          T = D0;
          d_Data[i0] = D0 + D1;
          d_Data[i1] = T - D1;
          T = D2;
          d_Data[i2] = D2 + D3;
          d_Data[i3] = T - D3;
        }
      }
#ifdef DEBUG
#endif
{
        float s_data[2048];
{
          int lid = omp_get_thread_num();
          int gid = omp_get_team_num();
          int gsz = omp_get_num_threads();
// Handle to thread block group
          const int N = 1 << log2N;
          const int base = gid << log2N;
          const float *d_Src = (d_Data + base);
          float *d_Dst = d_Data + base;
          
#pragma omp parallel for
          for (int pos = lid; pos <= N - 1; pos += gsz) {
            s_data[pos] = d_Src[pos];
          }
//Main radix-4 stages
          const int pos = lid;
          for (int stride = N >> 2; stride >= 0 + 1; stride >>= 2) {
            int lo = pos & stride - 1;
            int i0 = (pos - lo << 2) + lo;
            int i1 = i0 + stride;
            int i2 = i1 + stride;
            int i3 = i2 + stride;
            float D0 = s_data[i0];
            float D1 = s_data[i1];
            float D2 = s_data[i2];
            float D3 = s_data[i3];
            float T;
            T = D0;
            D0 = D0 + D2;
            D2 = T - D2;
            T = D1;
            D1 = D1 + D3;
            D3 = T - D3;
            T = D0;
            s_data[i0] = D0 + D1;
            s_data[i1] = T - D1;
            T = D2;
            s_data[i2] = D2 + D3;
            s_data[i3] = T - D3;
          }
//Do single radix-2 stage for odd power of two
          if ((log2N & 1)) {
            for (int pos = lid; pos <= N / 2 - 1; pos += gsz) {
              int i0 = pos << 1;
              int i1 = i0 + 1;
              float D0 = s_data[i0];
              float D1 = s_data[i1];
              s_data[i0] = D0 + D1;
              s_data[i1] = D0 - D1;
            }
          }
          
#pragma omp parallel for firstprivate (gsz,N)
          for (int pos = lid; pos <= N - 1; pos += gsz) {
            d_Dst[pos] = s_data[pos];
          }
        }
      }
#ifdef DEBUG
#endif
    }
    printf("Reading back GPU results...\n");
  }
  printf("Running straightforward CPU dyadic convolution...\n");
  dyadicConvolutionCPU(h_ResultCPU,h_Data,h_Kernel,log2Data,log2Kernel);
  printf("Comparing the results...\n");
  sum_delta2 = 0;
  sum_ref2 = 0;
  
#pragma omp parallel for private (delta,ref,i) reduction (+:sum_delta2,sum_ref2) firstprivate (dataN)
  for (i = 0; i <= dataN - 1; i += 1) {
    delta = (h_ResultCPU[i] - d_Data[i]);
    ref = h_ResultCPU[i];
    sum_delta2 += delta * delta;
    sum_ref2 += ref * ref;
  }
  L2norm = sqrt(sum_delta2 / sum_ref2);
  printf("Shutting down...\n");
  free(h_ResultCPU);
  free(h_Data);
  free(h_Kernel);
  free(d_Data);
  free(d_Kernel);
  printf("L2 norm: %E\n",L2norm);
  printf((L2norm < 1e-6?"Test passed\n" : "Test failed!\n"));
}
