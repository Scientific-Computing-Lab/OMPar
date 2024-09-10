/*  Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
  All rights reserved. https://github.com/robertwgh/cuLDPC

  CUDA implementation of LDPC decoding algorithm.
Created:   10/1/2010
Revision:  08/01/2013
/4/20/2016 prepare for release on Github.
*/
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
// custom header file
#include "LDPC.h"
//===================================
// Random info data generation
//===================================
#include <omp.h> 

void info_gen(int info_bin[],long seed)
{
  srand(seed);
  int i;
// random number generation
  for (i = 0; i <= 1151; i += 1) {
    info_bin[i] = rand() % 2;
  }
}
//===================================
// BPSK modulation
//===================================

void modulation(int code[],float trans[])
{
  int i;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 2303; i += 1) {
    if (code[i] == 0) 
      trans[i] = 1.0;
     else 
      trans[i] = (- 1.0);
  }
}
//===================================
// AWGN modulation
//===================================

void awgn(float trans[],float recv[],long seed)
{
  float u1;
  float u2;
  float s;
  float noise;
  float randmum;
  int i;
  srand(seed);
  for (i = 0; i <= 2303; i += 1) {
    do {
      randmum = ((float )(rand())) / ((float )2147483647);
      u1 = randmum * 2.0f - 1.0f;
      randmum = ((float )(rand())) / ((float )2147483647);
      u2 = randmum * 2.0f - 1.0f;
      s = u1 * u1 + u2 * u2;
    }while (s >= 1);
    noise = u1 * std::sqrt(- 2.0f * std::log(s) / s);
#ifdef NONOISE
#else
    recv[i] = trans[i] + noise * sigma;
#endif 
  }
}
//===================================
// calc LLRs
//===================================

void llr_init(float llr[],float recv[])
{
  int i;
#if PRINT_MSG == 1
#endif
  float llr_rev;
#if PRINT_MSG == 1
#endif
  
#pragma omp parallel for private (llr_rev,i) firstprivate (sigma)
  for (i = 0; i <= 2303; i += 1) {
    llr_rev = recv[i] * 2 / (sigma * sigma);
// 2r/sigma^2 ;
    llr[i] = llr_rev;
#if PRINT_MSG == 1
#endif
  }
#if PRINT_MSG == 1
#endif
}
//===================================
// parity check
//===================================

int parity_check(float app[])
{
  int *hbit = (int *)(malloc((96 * 24) * sizeof(int )));
  int error = 0;
  int i;
// hard decision
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 1151; i += 1) {
    if (app[i] >= 0) 
      hbit[i] = 0;
     else 
      hbit[i] = 1;
  }
  
#pragma omp parallel for private (i) reduction (+:error)
  for (i = 0; i <= 1151; i += 1) {
    if (hbit[i] != info_bin[i]) 
      error++;
  }
//#if PRINT_MSG == 1
//  fprintf(gfp, "After %d iteration, it has error %d\n", iter, error) ; 
//#endif
  free(hbit);
  return error;
}
//===================================
// parity check
//===================================

error_result error_check(int info[],int hard_decision[])
{
  error_result this_error;
  this_error . bit_error = 0;
  this_error . frame_error = 0;
  int bit_error = 0;
  int frame_error = 0;
  int *hard_decision_t = 0;
  int *info_t = 0;
  for (int i = 0; i <= 79; i += 1) {
    bit_error = 0;
    hard_decision_t = hard_decision + i * (24 * 96);
    info_t = info + i * (12 * 96);
    
#pragma omp parallel for reduction (+:bit_error)
    for (int j = 0; j <= 1151; j += 1) {
      if (info_t[j] != hard_decision_t[j]) 
        bit_error++;
    }
    if (bit_error != 0) 
      frame_error++;
    this_error . bit_error += bit_error;
  }
  this_error . frame_error = frame_error;
  return this_error;
}
//===================================
// encoding
//===================================

void structure_encode(int s[],int code[],int h[12][24])
{
  int i;
  int j;
  int k;
  int sk;
  int jj;
  int id;
  int shift;
  int x[12][96];
  int sum_x[96];
  int p0[96];
  int p1[96];
  int p2[96];
  int p3[96];
  int p4[96];
  int p5[96];
  int p6[96];
  int p7[96];
  int p8[96];
  int p9[96];
  int p10[96];
  int pp[96];
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= 11; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= 95; j += 1) {
      x[i][j] = 0;
    }
  }
  
#pragma omp parallel for private (j)
  for (j = 0; j <= 95; j += 1) {
    sum_x[j] = 0;
  }
  
#pragma omp parallel for private (sk,jj,shift,i,j,k)
  for (i = 0; i <= 11; i += 1) {
    for (j = 0; j <= 11; j += 1) {
      shift = h[i][j];
      if (shift >= 0) {
        
#pragma omp parallel for private (sk,jj,k) firstprivate (shift)
        for (k = 0; k <= 95; k += 1) {
          sk = (k + shift) % 96;
//Circular shifting, find the position for 1 in each sub-matrix
          jj = j * 96 + sk;
// calculate the index in the info sequence
          x[i][k] = (x[i][k] + s[jj]) % 2;
// block matrix multiplication
        }
      }
    }
  }
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= 95; i += 1) {
    for (j = 0; j <= 11; j += 1) {
      sum_x[i] = (x[j][i] + sum_x[i]) % 2;
    }
  }
  id = 12 * 96;
// p0  
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p0[i] = sum_x[i];
  }
// why p0 = sum??
  shift = h[0][12];
// h0
  for (i = 0; i <= 95; i += 1) {
    if (shift != - 1) {
      j = (i + shift) % 96;
      pp[i] = p0[j];
    }
     else 
      pp[i] = p0[j];
  }
// p1
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p1[i] = (x[0][i] + pp[i]) % 2;
  }
// p2
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p2[i] = (p1[i] + x[1][i]) % 2;
  }
// p3
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p3[i] = (p2[i] + x[2][i]) % 2;
  }
// p4
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p4[i] = (p3[i] + x[3][i]) % 2;
  }
// p5
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p5[i] = (p4[i] + x[4][i]) % 2;
  }
#if MODE == WIMAX
// p6
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p6[i] = (p5[i] + x[5][i] + p0[i]) % 2;
  }
// p7
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p7[i] = (p6[i] + x[6][i]) % 2;
  }
#else
// p6
// p7
#endif
// p8
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p8[i] = (p7[i] + x[7][i]) % 2;
  }
// p9
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p9[i] = (p8[i] + x[8][i]) % 2;
  }
// p10
  for (i = 0; i <= 95; i += 1) {
    code[id++] = p10[i] = (p9[i] + x[9][i]) % 2;
  }
// p11
  for (i = 0; i <= 95; i += 1) {
//code [id++] = p11 [i] = (p10 [i] + x[10][i]) % 2 ;
    code[id++] = (p10[i] + x[10][i]) % 2;
  }
// code word
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 1151; i += 1) {
    code[i] = s[i];
  }
}
