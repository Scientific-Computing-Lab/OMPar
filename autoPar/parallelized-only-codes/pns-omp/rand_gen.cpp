/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include "randomc.h"

#define LOWER_MASK ((1LU << MERS_R) - 1)         
#define UPPER_MASK (0xFFFFFFFF << MERS_R)        

/* 
   The following two functions implement the Mersenne Twister random 
   number generator.  The copyright notice/disclaimer, etc are related
   to this code.

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.                          

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote 
        products derived from this software without specific prior written 
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
void RandomInit(uint32 *mt, uint32 seed) 
{
  int i;
  // re-seed generator
  if(omp_get_thread_num() == 0)
    {
      mt[0]= seed & 0xffffffffUL;
      for (i=1; i < MERS_N; i++) 
	{
	  mt[i] = (1812433253UL * (mt[i-1] ^ (mt[i-1] >> 30)) + i);
  	}
    }
}

void BRandom(uint32* mt) 
{
  // generate 32 random bits
  uint32 y;
  int thdx;
  int threadIdx_x = omp_get_thread_num();

  // block size is 256
  // step 1: 0-226, MERS_N-MERS_M=227
  if (threadIdx_x<MERS_N-MERS_M) 
    {
      y = (mt[threadIdx_x] & UPPER_MASK) | (mt[threadIdx_x+1] & LOWER_MASK);
      y = mt[threadIdx_x+MERS_M] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
    }
  if (threadIdx_x<MERS_N-MERS_M) 
    {
      mt[threadIdx_x] = y;
    }
  
  // step 2: 227-453
  thdx = threadIdx_x + (MERS_N-MERS_M);
  if (threadIdx_x<MERS_N-MERS_M) 
    {
      y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
      y = mt[threadIdx_x] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
    }
  if (threadIdx_x<MERS_N-MERS_M) 
    {
      mt[thdx] = y;
    }
  
  // step 3: 454-622
  thdx += (MERS_N-MERS_M);
  if (thdx < MERS_N-1) 
    {
      y = (mt[thdx] & UPPER_MASK) | (mt[thdx+1] & LOWER_MASK);
      y = mt[threadIdx_x+(MERS_N-MERS_M)] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
    }
  if (thdx < MERS_N-1) 
    {
      mt[thdx] = y;
    }

  // step 4: 623
  if (threadIdx_x == 0) 
    {
      y = (mt[MERS_N-1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
      mt[MERS_N-1] = mt[MERS_M-1] ^ (y >> 1) ^ ( (y & 1)? MERS_A: 0);
    }

  // Tempering (May be omitted):
  y ^=  y >> MERS_U;
  y ^= (y << MERS_S) & MERS_B;
  y ^= (y << MERS_T) & MERS_C;
  y ^=  y >> MERS_L;

}


// return a random in [0, max]
// #define Random(max_plus_1)  (BRandom()% (max_plus_1))

