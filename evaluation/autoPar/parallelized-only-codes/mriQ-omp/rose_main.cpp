/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/
/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <malloc.h>
#include <chrono>
#include <omp.h>
#include "file.h"
#include "computeQ.cpp"
#include <omp.h> 

int main(int argc,char *argv[])
{
  char *inputFileName = argv[1];
  char *outputFileName = argv[2];
  int numX;
  int numK;
/* Number of X and K values */
  float *kx;
  float *ky;
  float *kz;
/* K trajectory (3D vectors) */
  float *x;
  float *y;
  float *z;
/* X coordinates (3D vectors) */
  float *phiR;
  float *phiI;
/* Phi values (complex) */
  float *phiMag;
/* Magnitude of Phi */
  float *Qr;
  float *Qi;
/* Q signal (complex) */
  struct kValues *kVals;
/* Read in data */
  inputData(inputFileName,&numK,&numX,&kx,&ky,&kz,&x,&y,&z,&phiR,&phiI);
  printf("%d pixels in output; %d samples in trajectory\n",numX,numK);
/* Create CPU data structures */
  createDataStructsCPU(numK,numX,&phiMag,&Qr,&Qi);
/* GPU section 1 (precompute PhiMag) */
  int phiMagBlocks = numK / 256;
  if ((numK % 256)) 
    phiMagBlocks++;
{
    auto start = std::chrono::_V2::steady_clock::now();
    
#pragma omp parallel for
    for (int indexK = 0; indexK <= numK - 1; indexK += 1) {
      float real = phiR[indexK];
      float imag = phiI[indexK];
      phiMag[indexK] = real * real + imag * imag;
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("computePhiMag time: %f s\n",(time * 1e-9f));
  }
  kVals = ((struct kValues *)(calloc(numK,sizeof(struct kValues ))));
  
#pragma omp parallel for
  for (int k = 0; k <= numK - 1; k += 1) {
    kVals[k] . Kx = kx[k];
    kVals[k] . Ky = ky[k];
    kVals[k] . Kz = kz[k];
    kVals[k] . PhiMag = phiMag[k];
  }
/* GPU section 2 */
  struct kValues ck[1024];
{
    
#pragma omp parallel for
    for (int i = 0; i <= numX - 1; i += 1) {
      Qr[i] = 0.f;
      Qi[i] = 0.f;
    }
    auto start = std::chrono::_V2::steady_clock::now();
    computeQ_GPU(numK,numX,x,y,z,kVals,ck,Qr,Qi);
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("computeQ time: %f s\n",(time * 1e-9f));
  }
  outputData(outputFileName,Qr,Qi,numX);
  free(phiMag);
  free(kx);
  free(ky);
  free(kz);
  free(x);
  free(y);
  free(z);
  free(phiR);
  free(phiI);
  free(kVals);
  free(Qr);
  free(Qi);
  return 0;
}
