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
///////////////////////////////////////////////////////////////////////////////
// This sample implements Mersenne Twister random number generator
// and Cartesian Box-Muller transformation on the GPU
///////////////////////////////////////////////////////////////////////////////
// standard utilities and systems includes
#include <stdio.h>
#include <math.h>
#include "MT.h"
// comment the below line if not doing Box-Muller transformation
#define DO_BOXMULLER
// Reference CPU MT and Box-Muller transformation 
#include <omp.h> 
extern "C" void initMTRef(const char *fname);
extern "C" void RandomRef(float *h_Rand,int nPerRng,unsigned int seed);
#ifdef DO_BOXMULLER
extern "C" void BoxMullerRef(float *h_Rand,int nPerRng);
#endif
#include <chrono>
using namespace std::chrono;
///////////////////////////////////////////////////////////////////////////////
//Load twister configurations
///////////////////////////////////////////////////////////////////////////////

void loadMTGPU(const char *fname,const unsigned int seed,mt_struct_stripped *h_MT,const size_t size)
{
// open the file for binary read
  FILE *fd = fopen(fname,"rb");
  if (fd == 0L) {
    printf("Failed to open file %s\n",fname);
    exit(- 1);
  }
  for (unsigned int i = 0; ((unsigned long )i) <= size - 1; i += 1) {
    fread((&h_MT[i]),sizeof(mt_struct_stripped ),1,fd);
  }
  fclose(fd);
  
#pragma omp parallel for firstprivate (seed,size)
  for (unsigned int i = 0; ((unsigned long )i) <= size - 1; i += 1) {
    h_MT[i] . seed = seed;
  }
}

void BoxMullerTrans(float *u1,float *u2)
{
  const float r = sqrtf(- 2.0f * logf( *u1));
  const float phi = 2 * 3.14159265358979f *  *u2;
   *u1 = r * cosf(phi);
   *u2 = r * sinf(phi);
}
///////////////////////////////////////////////////////////////////////////////
// Main function 
///////////////////////////////////////////////////////////////////////////////

int main(int argc,const char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  int numIterations = atoi(argv[1]);
  size_t globalWorkSize = 4096;
// 1D var for Total # of work items
  size_t localWorkSize = 128;
// 1D var for # of work items in the work group  
  const int seed = 777;
  const int nPerRng = 5860;
// # of recurrence steps, must be even if do Box-Muller transformation
  const int nRand = 4096 * nPerRng;
// Output size
  printf("Initialization: load MT parameters and init host buffers...\n");
  mt_struct_stripped *h_MT = (mt_struct_stripped *)(malloc(sizeof(mt_struct_stripped ) * 4096));
// MT para
  const char *cDatPath = "./data/MersenneTwister.dat";
  loadMTGPU(cDatPath,seed,h_MT,4096);
  const char *cRawPath = "./data/MersenneTwister.raw";
  initMTRef(cRawPath);
  float *h_RandGPU = (float *)(malloc(sizeof(float ) * nRand));
// Host buffers for GPU output
  float *h_RandCPU = (float *)(malloc(sizeof(float ) * nRand));
// Host buffers for CPU test
  printf("Allocate memory...\n");
{
    printf("Call Mersenne Twister kernel... (%d iterations)\n\n",numIterations);
    std::chrono::_V2::system_clock::time_point t1 = std::chrono::_V2::system_clock::now();
    for (int i = 0; i <= numIterations - 1; i += 1) {
      for (int globalID = 0; ((unsigned long )globalID) <= globalWorkSize - 1; globalID += 1) {
        int iState;
        int iState1;
        int iStateM;
        int iOut;
        unsigned int mti;
        unsigned int mti1;
        unsigned int mtiM;
        unsigned int x;
        unsigned int mt[19];
        unsigned int matrix_a;
        unsigned int mask_b;
        unsigned int mask_c;
//Load bit-vector Mersenne Twister parameters
        matrix_a = h_MT[globalID] . matrix_a;
        mask_b = h_MT[globalID] . mask_b;
        mask_c = h_MT[globalID] . mask_c;
//Initialize current state
        mt[0] = h_MT[globalID] . seed;
        for (iState = 1; iState <= 18; iState += 1) {
          mt[iState] = 1812433253U * (mt[iState - 1] ^ mt[iState - 1] >> 30) + iState & 0xFFFFFFFFU;
        }
        iState = 0;
        mti1 = mt[0];
        for (iOut = 0; iOut <= nPerRng - 1; iOut += 1) {
          iState1 = iState + 1;
          iStateM = iState + 9;
          if (iState1 >= 19) 
            iState1 -= 19;
          if (iStateM >= 19) 
            iStateM -= 19;
          mti = mti1;
          mti1 = mt[iState1];
          mtiM = mt[iStateM];
// MT recurrence
          x = mti & 0xFFFFFFFEU | mti1 & 0x1U;
          x = mtiM ^ x >> 1 ^ (((x & 1)?matrix_a : 0));
          mt[iState] = x;
          iState = iState1;
//Tempering transformation
          x ^= x >> 12;
          x ^= x << 7 & mask_b;
          x ^= x << 15 & mask_c;
          x ^= x >> 18;
//Convert to (0, 1] float and write to global memory
          h_RandGPU[globalID + iOut * 4096] = (((float )x) + 1.0f) / 4294967296.0f;
        }
      }
#ifdef DO_BOXMULLER 
      for (int globalID = 0; ((unsigned long )globalID) <= globalWorkSize - 1; globalID += 1) {
        for (int iOut = 0; iOut <= nPerRng - 1; iOut += 2) {
          BoxMullerTrans(&h_RandGPU[globalID + (iOut + 0) * 4096],&h_RandGPU[globalID + (iOut + 1) * 4096]);
        }
      }
#endif
    }
    std::chrono::_V2::system_clock::time_point t2 = std::chrono::_V2::system_clock::now();
    struct std::chrono::duration< double  , class std::ratio< 1 , 1L >  > time_span = std::chrono::duration_cast< class std::chrono::duration< double  , class std::ratio< 1 , 1L >  >  , int64_t  , std::nano  > ((t2-t1));
    double gpuTime = time_span . count() / ((double )numIterations);
    printf("MersenneTwister, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, Workgroup = %lu\n",((double )nRand) * 1.0E-9 / gpuTime,gpuTime,nRand,localWorkSize);
    printf("\nRead back results...\n");
  }
  printf("Compute CPU reference solution...\n");
  RandomRef(h_RandCPU,nPerRng,seed);
#ifdef DO_BOXMULLER
  BoxMullerRef(h_RandCPU,nPerRng);
#endif
  printf("Compare CPU and GPU results...\n");
  double sum_delta = 0;
  double sum_ref = 0;
  for (int i = 0; i <= 4095; i += 1) {
    for (int j = 0; j <= nPerRng - 1; j += 1) {
      double rCPU = h_RandCPU[i * nPerRng + j];
      double rGPU = h_RandGPU[i + j * 4096];
      double delta = fabs(rCPU - rGPU);
      sum_delta += delta;
      sum_ref += fabs(rCPU);
    }
  }
  double L1norm = sum_delta / sum_ref;
  printf("L1 norm: %E\n\n",L1norm);
  free(h_MT);
  free(h_RandGPU);
  free(h_RandCPU);
// finish
  printf("%s\n",(L1norm < 1e-6?"PASS" : "FAIL"));
  return 0;
}
