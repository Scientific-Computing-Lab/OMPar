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
#include <math.h>
#include "MT.h"
#include "dci.h"
static mt_struct MT[4096];
static uint32_t state[19];

extern "C" void initMTRef(const char *fname)
{
// open the file for binary read
  FILE *fd = fopen(fname,"rb");
  if (fd == 0L) {
    printf("Failed to open file %s\n",fname);
    exit(- 1);
  }
  for (int i = 0; i <= 4095; i += 1) {
//Inline structure size for compatibility,
//since pointer types are 8-byte on 64-bit systems (unused *state variable)
/* sizeof(mt_struct) */
    fread((MT + i),16 * sizeof(int ),1,fd);
  }
  fclose(fd);
}

extern "C" void RandomRef(float *h_Rand,int NPerRng,unsigned int seed)
{
  int iRng;
  int iOut;
  for (iRng = 0; iRng <= 4095; iRng += 1) {
    MT[iRng] . state = state;
    sgenrand_mt(seed,&MT[iRng]);
    for (iOut = 0; iOut <= NPerRng - 1; iOut += 1) {
      h_Rand[iRng * NPerRng + iOut] = (((float )(genrand_mt(&MT[iRng]))) + 1.0f) / 4294967296.0f;
    }
  }
}

void BoxMuller(float &u1,float &u2)
{
  float r = sqrtf(- 2.0f * logf(u1));
  float phi = 2 * 3.14159265358979f * u2;
  u1 = r * cosf(phi);
  u2 = r * sinf(phi);
}

extern "C" void BoxMullerRef(float *h_Random,int NPerRng)
{
  int i;
  for (i = 0; i <= 4096 * NPerRng - 1; i += 2) {
    BoxMuller(h_Random[i + 0],h_Random[i + 1]);
  }
}
