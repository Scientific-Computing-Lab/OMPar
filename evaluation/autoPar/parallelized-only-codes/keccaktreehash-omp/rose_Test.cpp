/*
   GPU Implementation of Keccak by Guillaume Sevestre, 2010

   This code is hereby put in the public domain.
   It is given as is, without any guarantee.

*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "KeccakF.h"
#include "KeccakTreeCPU.h"
#include "KeccakTreeGPU.h"
// choose 8 for fast execution 
#define IMAX 8 // 1600 //2400 // 1600 for high speed mesures // iteration for speed mesure loops
//debug print function
#include <omp.h> 

void print_out(tKeccakLane *h_outBuffer,int nb_threads)
{
  printf("%08x ",h_outBuffer[0]);
  printf("%08x ",h_outBuffer[1]);
  printf("%08x ",h_outBuffer[nb_threads]);
  printf("%08x ",h_outBuffer[nb_threads + 1]);
  printf("\n\n");
}

void TestCPU(int reduc)
{
  time_t t1;
  time_t t2;
  double speed1;
  int i;
  tKeccakLane *h_inBuffer;
// Host in buffer for data to be hashed
  tKeccakLane *h_outBuffer;
// Host out buffer 
  tKeccakLane Kstate[25];
//Keccak State for top node
  memset(Kstate,0,25 * sizeof(tKeccakLane ));
//init host inBuffer 
  h_inBuffer = ((tKeccakLane *)(malloc((32 * 64 * 64 * 1024))));
  memset(h_inBuffer,0,(32 * 64 * 64 * 1024));
//init host outBuffer   
  h_outBuffer = ((tKeccakLane *)(malloc((32 * 64 * 64))));
  memset(h_outBuffer,0,(32 * 64 * 64));
//***************************
//init h_inBuffer with values
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 33554431; i += 1) {
    h_inBuffer[i] = i;
  }
//CPU computation *******************************
  printf("CPU speed test started \n");
  t1 = time(0L);
  for (i = 0; i <= 8 / reduc - 1; i += 1) {
    KeccakTreeCPU(h_inBuffer,h_outBuffer);
//print_out(h_outBuffer,NB_THREADS);
    Keccak_top(Kstate,h_outBuffer,64 * 64);
  }
  t2 = time(0L);
  print_KS_256(Kstate);
  speed1 = (32 * 64 * 64 * 1024) * (8 / (reduc * 1000.)) / ((t2 - t1) + 0.01);
  printf("CPU speed : %.2f kB/s \n\n",speed1);
//free all buffer host and device
  free(h_inBuffer);
  free(h_outBuffer);
}

void TestGPU()
{
  time_t t1;
  time_t t2;
  double speed1;
  unsigned int i;
  const tKeccakLane KeccakF_RoundConstants[22] = {((tKeccakLane )0x00000001), ((tKeccakLane )0x00008082), ((tKeccakLane )0x0000808a), ((tKeccakLane )0x80008000), ((tKeccakLane )0x0000808b), ((tKeccakLane )0x80000001), ((tKeccakLane )0x80008081), ((tKeccakLane )0x00008009), ((tKeccakLane )0x0000008a), ((tKeccakLane )0x00000088), ((tKeccakLane )0x80008009), ((tKeccakLane )0x8000000a), ((tKeccakLane )0x8000808b), ((tKeccakLane )0x0000008b), ((tKeccakLane )0x00008089), ((tKeccakLane )0x00008003), ((tKeccakLane )0x00008002), ((tKeccakLane )0x00000080), ((tKeccakLane )0x0000800a), ((tKeccakLane )0x8000000a), ((tKeccakLane )0x80008081), ((tKeccakLane )0x00008080)};
  tKeccakLane *h_inBuffer;
// Host in buffer for data to be hashed
  tKeccakLane *h_outBuffer;
// Host out buffer 
  tKeccakLane Kstate[25];
//Keccak State for top node
  memset(Kstate,0,25 * sizeof(tKeccakLane ));
//init host inBuffer 
  h_inBuffer = ((tKeccakLane *)(malloc((32 * 64 * 64 * 1024))));
  memset(h_inBuffer,0,(32 * 64 * 64 * 1024));
//init host outBuffer   
  h_outBuffer = ((tKeccakLane *)(malloc((32 * 64 * 64))));
  memset(h_outBuffer,0,(32 * 64 * 64));
//***************************
//init h_inBuffer with values
  
#pragma omp parallel for private (i)
  for (i = 0; i <= ((unsigned int )33554432) - 1; i += 1) {
    h_inBuffer[i] = i;
  }
//GPU computation *******************************
  printf("GPU speed test started\n");
  t1 = time(0L);
  for (i = 0; i <= ((unsigned int )8) - 1; i += 1) {
    KeccakTreeGPU(h_inBuffer,h_outBuffer,KeccakF_RoundConstants);
//print_out(h_outBuffer,NB_THREADS*NB_THREADS_BLOCKS);
    Keccak_top(Kstate,h_outBuffer,64 * 64);
//print_KS_256(Kstate);
  }
  t2 = time(0L);
  print_KS_256(Kstate);
  speed1 = (32 * 64 * 64 * 1024) * (8 / 1000.) / ((t2 - t1) + 0.01);
  printf("GPU speed : %.2f kB/s \n\n",speed1);
//free all buffer host and device
  free(h_inBuffer);
  free(h_outBuffer);
}

void Print_Param()
{
  printf("\n");
  printf("Numbers of Threads PER BLOCK            NB_THREADS           %u \n",64);
  printf("Numbers of Threads Blocks               NB_THREADS_BLOCKS    %u \n",64);
  printf("\n");
  printf("Input block size of Keccak (in Byte)    INPUT_BLOCK_SIZE_B   %u \n",32);
  printf("Output block size of Keccak (in Byte)   OUTPUT_BLOCK_SIZE_B  %u \n",32);
  printf("\n");
  printf("NB of input blocks in by Threads        NB_INPUT_BLOCK       %u \n",1024);
  printf("\n");
}
