/*
GPU Implementation of Keccak by Guillaume Sevestre, 2010

This code is hereby put in the public domain.
It is given as is, without any guarantee.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "KeccakTreeCPU.h"
#include "KeccakF.h"
#include <omp.h> 

void KeccakTreeCPU(tKeccakLane *inBuffer,tKeccakLane *outBuffer)
{
  int thrIdx;
  int blkIdx;
  int k;
  int ind_word;
  for (blkIdx = 0; blkIdx <= 63; blkIdx += 1) 
//loop on threads blocks
{
    for (thrIdx = 0; thrIdx <= 63; thrIdx += 1) 
//loop on threads inside a threadblock
{
      tKeccakLane Kstate[25];
      memset(Kstate,0,25 * sizeof(tKeccakLane ));
      for (k = 0; k <= 1023; k += 1) {
//xor input into state
        
#pragma omp parallel for private (ind_word)
        for (ind_word = 0; ind_word <= 7; ind_word += 1) {
          Kstate[ind_word] ^= inBuffer[thrIdx + ind_word * 64 + k * 64 * 32 / 4 + blkIdx * 64 * 32 / 4 * 1024];
        }
//apply Keccak permutation
        KeccakF_CPU(Kstate);
      }
//output hash in out buffer
      
#pragma omp parallel for private (ind_word)
      for (ind_word = 0; ind_word <= 7; ind_word += 1) {
//printf("Kstate[%02u] = %08x",ind_word,Kstate[ind_word] );  
        outBuffer[thrIdx + ind_word * 64 + blkIdx * 64 * 32 / 4] = Kstate[ind_word];
      }
//end loop threads
    }
//end loop on threadsblocks
  }
}
// Implement a second stage on treehashing
// Use output of 2x OUTPUT_BLOCK_SIZE_B size to respect conditions for soundness of Treehashing
