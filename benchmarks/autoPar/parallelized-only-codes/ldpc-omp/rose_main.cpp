/*  Copyright (c) 2011-2016, Robert Wang, email: robertwgh (at) gmail.com
  All rights reserved. https://github.com/robertwgh/cuLDPC
  Implementation of LDPC decoding algorithm.
  The details of implementation can be found from the following papers:
  1. Wang, G., Wu, M., Sun, Y., & Cavallaro, J. R. (2011, June). A massively parallel implementation of QC-LDPC decoder on GPU. In Application Specific Processors (SASP), 2011 IEEE 9th Symposium on (pp. 82-85). IEEE.
  2. Wang, G., Wu, M., Yin, B., & Cavallaro, J. R. (2013, December). High throughput low latency LDPC decoding on GPU for SDR systems. In Global Conference on Signal and Information Processing (GlobalSIP), 2013 IEEE (pp. 1258-1261). IEEE.
  The current release is close to the GlobalSIP2013 paper. 
 */
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "LDPC.h"
#include "matrix.h"
#include "kernel.cpp"
#include <omp.h> 
float sigma;
int *info_bin;

int main()
{
  printf("GPU LDPC Decoder\r\nComputing...\r\n");
// For cnp kernel
#if MODE == WIMAX
  const char h_element_count1[12] = {(6), (7), (7), (6), (6), (7), (6), (6), (7), (6), (6), (6)};
  const char h_element_count2[24] = {(3), (3), (6), (3), (3), (6), (3), (6), (3), (6), (3), (6), (3), (2), (2), (2), (2), (2), (2), (2), (2), (2), (2), (2)};
#else
#endif
  h_element h_compact1[84];
// for update dt, R
  h_element h_element_temp;
// init the compact matrix
  
#pragma omp parallel for private (j)
  for (int i = 0; i <= 6; i += 1) {
    
#pragma omp parallel for
    for (int j = 0; j <= 11; j += 1) {
      h_element_temp . x = 0;
      h_element_temp . y = 0;
      h_element_temp . value = (- 1);
      h_element_temp . valid = 0;
      h_compact1[i * 12 + j] = h_element_temp;
// h[i][0-11], the same column
    }
  }
// scan the h matrix, and gengerate compact mode of h
  for (int i = 0; i <= 11; i += 1) {
    int k = 0;
    for (int j = 0; j <= 23; j += 1) {
      if (h_base[i][j] != - 1) {
        h_element_temp . x = i;
        h_element_temp . y = j;
        h_element_temp . value = h_base[i][j];
        h_element_temp . valid = 1;
        h_compact1[k * 12 + i] = h_element_temp;
        k++;
      }
    }
// printf("row %d, #element=%d\n", i, k);
  }
// h_compact2
  h_element h_compact2[288];
// for update llr
// init the compact matrix
  
#pragma omp parallel for private (j_nom_4)
  for (int i = 0; i <= 11; i += 1) {
    
#pragma omp parallel for
    for (int j = 0; j <= 23; j += 1) {
      h_element_temp . x = 0;
      h_element_temp . y = 0;
      h_element_temp . value = (- 1);
      h_element_temp . valid = 0;
      h_compact2[i * 24 + j] = h_element_temp;
    }
  }
  for (int j = 0; j <= 23; j += 1) {
    int k = 0;
    for (int i = 0; i <= 11; i += 1) {
      if (h_base[i][j] != - 1) {
// although h is transposed, the (x,y) is still (iBlkRow, iBlkCol)
        h_element_temp . x = i;
        h_element_temp . y = j;
        h_element_temp . value = h_base[i][j];
        h_element_temp . valid = 1;
        h_compact2[k * 24 + j] = h_element_temp;
        k++;
      }
    }
  }
  int wordSize_h_compact1 = 12 * 7;
  int wordSize_h_compact2 = 12 * 24;
  int memorySize_h_compact1 = (wordSize_h_compact1 * sizeof(h_element ));
  int memorySize_h_compact2 = (wordSize_h_compact2 * sizeof(h_element ));
  int memorySize_infobits = ((12 * 96) * sizeof(int ));
  int memorySize_codeword = ((24 * 96) * sizeof(int ));
  int memorySize_llr = ((24 * 96) * sizeof(float ));
  info_bin = ((int *)(malloc(memorySize_infobits)));
  int *codeword = (int *)(malloc(memorySize_codeword));
  float *trans = (float *)(malloc(memorySize_llr));
  float *recv = (float *)(malloc(memorySize_llr));
  float *llr = (float *)(malloc(memorySize_llr));
  float rate = (float )0.5f;
//////////////////////////////////////////////////////////////////////////////////
// all the variables Starting with _gpu is used in host code and for gpu computation
  int wordSize_llr = 40 * 2 * (24 * 96);
  int wordSize_dt = 40 * 2 * (96 * 12) * 24;
  int wordSize_R = 40 * 2 * (96 * 12) * 24;
  int wordSize_hard_decision = 40 * 2 * (24 * 96);
  int memorySize_infobits_gpu = 40 * 2 * memorySize_infobits;
  int memorySize_llr_gpu = (wordSize_llr * sizeof(float ));
  int memorySize_dt_gpu = (wordSize_dt * sizeof(float ));
  int memorySize_R_gpu = (wordSize_R * sizeof(float ));
  int memorySize_hard_decision_gpu = (wordSize_hard_decision * sizeof(int ));
  int *info_bin_gpu;
  float *llr_gpu;
  int *hard_decision_gpu;
  info_bin_gpu = ((int *)(malloc(memorySize_infobits_gpu)));
  hard_decision_gpu = ((int *)(malloc(memorySize_hard_decision_gpu)));
  llr_gpu = ((float *)(malloc(memorySize_llr_gpu)));
  error_result this_error;
  int total_frame_error = 0;
  int total_bit_error = 0;
  int total_codeword = 0;
  float *dev_llr = llr_gpu;
  float *dev_dt = (float *)(malloc(memorySize_dt_gpu));
  float *dev_R = (float *)(malloc(memorySize_R_gpu));
  int *dev_hard_decision = hard_decision_gpu;
  const h_element *dev_h_compact1 = h_compact1;
  const h_element *dev_h_compact2 = h_compact2;
  const char *dev_h_element_count1 = h_element_count1;
  const char *dev_h_element_count2 = h_element_count2;
  srand(69012);
{
    for (int snri = 0; snri <= 0; snri += 1) {
      float snr = snr_array[snri];
      sigma = 1.0f / std::sqrt(2.0f * rate * std::pow(10.0f,snr / 10.0f));
      total_codeword = 0;
      total_frame_error = 0;
      total_bit_error = 0;
// Adjust MIN_CODWORD in LDPC.h to reduce simulation time
      while(total_frame_error <= 2000000 && total_codeword <= 2000){
        total_codeword += 2 * 40;
        for (int i = 0; i <= 79; i += 1) {
// generate random data
          info_gen(info_bin,(rand()));
// encode the data
          structure_encode(info_bin,codeword,h_base);
// BPSK modulation
          modulation(codeword,trans);
// additive white Gaussian noise
          awgn(trans,recv,(rand()));
// LLR init
          llr_init(llr,recv);
// copy the info_bin and llr to the total memory
          memcpy((info_bin_gpu + i * (12 * 96)),info_bin,memorySize_infobits);
          memcpy((llr_gpu + i * (24 * 96)),llr,memorySize_llr);
        }
// run the kernel
        float total_time = 0.f;
        for (int j = 0; j <= 499; j += 1) {
// Transfer LLR data into device.
// kernel launch
          auto start = std::chrono::_V2::steady_clock::now();
          for (int ii = 0; ii <= 9; ii += 1) {
// run check-node processing kernel
// TODO: run a special kernel the first iteration?
            if (ii == 0) {
              ldpc_cnp_kernel_1st_iter(dev_llr,dev_dt,dev_R,dev_h_element_count1,dev_h_compact1);
            }
             else {
              ldpc_cnp_kernel(dev_llr,dev_dt,dev_R,dev_h_element_count1,dev_h_compact1);
            }
// run variable-node processing kernel
// for the last iteration we run a special
// kernel. this is because we can make a hard
// decision instead of writing back the belief
// for the value of each bit.
            if (ii < 10 - 1) {
              ldpc_vnp_kernel_normal(dev_llr,dev_dt,dev_h_element_count2,dev_h_compact2);
            }
             else {
              ldpc_vnp_kernel_last_iter(dev_llr,dev_dt,dev_hard_decision,dev_h_element_count2,dev_h_compact2);
            }
          }
          auto end = std::chrono::_V2::steady_clock::now();
          auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
          total_time += time;
// copy the decoded data from device to host
          this_error = error_check(info_bin_gpu,hard_decision_gpu);
          total_bit_error += this_error . bit_error;
          total_frame_error += this_error . frame_error;
// end of MAX-SIM
        }
        printf("\n");
        printf("Total kernel execution time: %f (s)\n",(total_time * 1e-9f));
        printf("# codewords = %d, CW=%d, MCW=%d\n",total_codeword,2,40);
        printf("total bit error = %d\n",total_bit_error);
        printf("total frame error = %d\n",total_frame_error);
        printf("BER = %1.2e, FER = %1.2e\n",(((float )total_bit_error) / total_codeword / (12 * 96)),(((float )total_frame_error) / total_codeword));
// end of the MAX frame error.
      }
// end of the snr loop
    }
  }
  free(dev_dt);
  free(dev_R);
  free(info_bin);
  free(codeword);
  free(trans);
  free(recv);
  free(llr);
  free(llr_gpu);
  free(hard_decision_gpu);
  free(info_bin_gpu);
  return 0;
}
