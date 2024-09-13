/*
   GPU Implementation of Keccak by Guillaume Sevestre, 2010

   This code is hereby put in the public domain.
   It is given as is, without any guarantee.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "KeccakTreeGPU.h"
//host constants
#include <omp.h> 
tKeccakLane KeccakF_RoundConstants_h[22] = {((tKeccakLane )0x00000001), ((tKeccakLane )0x00008082), ((tKeccakLane )0x0000808a), ((tKeccakLane )0x80008000), ((tKeccakLane )0x0000808b), ((tKeccakLane )0x80000001), ((tKeccakLane )0x80008081), ((tKeccakLane )0x00008009), ((tKeccakLane )0x0000008a), ((tKeccakLane )0x00000088), ((tKeccakLane )0x80008009), ((tKeccakLane )0x8000000a), ((tKeccakLane )0x8000808b), ((tKeccakLane )0x0000008b), ((tKeccakLane )0x00008089), ((tKeccakLane )0x00008003), ((tKeccakLane )0x00008002), ((tKeccakLane )0x00000080), ((tKeccakLane )0x0000800a), ((tKeccakLane )0x8000000a), ((tKeccakLane )0x80008081), ((tKeccakLane )0x00008080)};
// Device (GPU) Keccak-f function implementation
// unrolled

void KeccakFunr(tKeccakLane *state,const tKeccakLane *KeccakF_RoundConstants)
{
  unsigned int round;
//try to avoid to many registers
  tKeccakLane BC[5];
  tKeccakLane temp;
  for (round = 0; round <= ((unsigned int )22) - 1; round += 1) {{
// Theta
      BC[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
      BC[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
      BC[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
      BC[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
      BC[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];
      temp = BC[4] ^ (BC[1] << 1 ^ BC[1] >> 32 - 1);
//x=0
      state[0] ^= temp;
      state[5] ^= temp;
      state[10] ^= temp;
      state[15] ^= temp;
      state[20] ^= temp;
      temp = BC[0] ^ (BC[2] << 1 ^ BC[2] >> 32 - 1);
//x=1
      state[1] ^= temp;
      state[6] ^= temp;
      state[11] ^= temp;
      state[16] ^= temp;
      state[21] ^= temp;
      temp = BC[1] ^ (BC[3] << 1 ^ BC[3] >> 32 - 1);
//x=2
      state[2] ^= temp;
      state[7] ^= temp;
      state[12] ^= temp;
      state[17] ^= temp;
      state[22] ^= temp;
      temp = BC[2] ^ (BC[4] << 1 ^ BC[4] >> 32 - 1);
//x=3
      state[3] ^= temp;
      state[8] ^= temp;
      state[13] ^= temp;
      state[18] ^= temp;
      state[23] ^= temp;
      temp = BC[3] ^ (BC[0] << 1 ^ BC[0] >> 32 - 1);
//x=4
      state[4] ^= temp;
      state[9] ^= temp;
      state[14] ^= temp;
      state[19] ^= temp;
      state[24] ^= temp;
//end Theta
    }
{
// Rho Pi
      temp = state[1];
      BC[0] = state[10];
      state[10] = temp << 1 ^ temp >> 32 - 1;
      temp = BC[0];
//x=0
      BC[0] = state[7];
      state[7] = temp << 3 ^ temp >> 32 - 3;
      temp = BC[0];
      BC[0] = state[11];
      state[11] = temp << 6 ^ temp >> 32 - 6;
      temp = BC[0];
      BC[0] = state[17];
      state[17] = temp << 10 ^ temp >> 32 - 10;
      temp = BC[0];
      BC[0] = state[18];
      state[18] = temp << 15 ^ temp >> 32 - 15;
      temp = BC[0];
      BC[0] = state[3];
      state[3] = temp << 21 ^ temp >> 32 - 21;
      temp = BC[0];
//x=5
      BC[0] = state[5];
      state[5] = temp << 28 ^ temp >> 32 - 28;
      temp = BC[0];
      BC[0] = state[16];
      state[16] = temp << 4 ^ temp >> 32 - 4;
      temp = BC[0];
      BC[0] = state[8];
      state[8] = temp << 13 ^ temp >> 32 - 13;
      temp = BC[0];
      BC[0] = state[21];
      state[21] = temp << 23 ^ temp >> 32 - 23;
      temp = BC[0];
      BC[0] = state[24];
      state[24] = temp << 2 ^ temp >> 32 - 2;
      temp = BC[0];
//x=10
      BC[0] = state[4];
      state[4] = temp << 14 ^ temp >> 32 - 14;
      temp = BC[0];
      BC[0] = state[15];
      state[15] = temp << 27 ^ temp >> 32 - 27;
      temp = BC[0];
      BC[0] = state[23];
      state[23] = temp << 9 ^ temp >> 32 - 9;
      temp = BC[0];
      BC[0] = state[19];
      state[19] = temp << 24 ^ temp >> 32 - 24;
      temp = BC[0];
      BC[0] = state[13];
      state[13] = temp << 8 ^ temp >> 32 - 8;
      temp = BC[0];
//x=15
      BC[0] = state[12];
      state[12] = temp << 25 ^ temp >> 32 - 25;
      temp = BC[0];
      BC[0] = state[2];
      state[2] = temp << 11 ^ temp >> 32 - 11;
      temp = BC[0];
      BC[0] = state[20];
      state[20] = temp << 30 ^ temp >> 32 - 30;
      temp = BC[0];
      BC[0] = state[14];
      state[14] = temp << 18 ^ temp >> 32 - 18;
      temp = BC[0];
      BC[0] = state[22];
      state[22] = temp << 7 ^ temp >> 32 - 7;
      temp = BC[0];
//x=20
      BC[0] = state[9];
      state[9] = temp << 29 ^ temp >> 32 - 29;
      temp = BC[0];
      BC[0] = state[6];
      state[6] = temp << 20 ^ temp >> 32 - 20;
      temp = BC[0];
      BC[0] = state[1];
      state[1] = temp << 12 ^ temp >> 32 - 12;
      temp = BC[0];
//x=23
//end Rho Pi
    }
{
//   Chi
      BC[0] = state[0];
      BC[1] = state[1];
      BC[2] = state[2];
      BC[3] = state[3];
      BC[4] = state[4];
      state[0] = BC[0] ^ ~BC[1] & BC[2];
      state[1] = BC[1] ^ ~BC[2] & BC[3];
      state[2] = BC[2] ^ ~BC[3] & BC[4];
      state[3] = BC[3] ^ ~BC[4] & BC[0];
      state[4] = BC[4] ^ ~BC[0] & BC[1];
      BC[0] = state[5];
      BC[1] = state[6];
      BC[2] = state[7];
      BC[3] = state[8];
      BC[4] = state[9];
      state[5] = BC[0] ^ ~BC[1] & BC[2];
      state[6] = BC[1] ^ ~BC[2] & BC[3];
      state[7] = BC[2] ^ ~BC[3] & BC[4];
      state[8] = BC[3] ^ ~BC[4] & BC[0];
      state[9] = BC[4] ^ ~BC[0] & BC[1];
      BC[0] = state[10];
      BC[1] = state[11];
      BC[2] = state[12];
      BC[3] = state[13];
      BC[4] = state[14];
      state[10] = BC[0] ^ ~BC[1] & BC[2];
      state[11] = BC[1] ^ ~BC[2] & BC[3];
      state[12] = BC[2] ^ ~BC[3] & BC[4];
      state[13] = BC[3] ^ ~BC[4] & BC[0];
      state[14] = BC[4] ^ ~BC[0] & BC[1];
      BC[0] = state[15];
      BC[1] = state[16];
      BC[2] = state[17];
      BC[3] = state[18];
      BC[4] = state[19];
      state[15] = BC[0] ^ ~BC[1] & BC[2];
      state[16] = BC[1] ^ ~BC[2] & BC[3];
      state[17] = BC[2] ^ ~BC[3] & BC[4];
      state[18] = BC[3] ^ ~BC[4] & BC[0];
      state[19] = BC[4] ^ ~BC[0] & BC[1];
      BC[0] = state[20];
      BC[1] = state[21];
      BC[2] = state[22];
      BC[3] = state[23];
      BC[4] = state[24];
      state[20] = BC[0] ^ ~BC[1] & BC[2];
      state[21] = BC[1] ^ ~BC[2] & BC[3];
      state[22] = BC[2] ^ ~BC[3] & BC[4];
      state[23] = BC[3] ^ ~BC[4] & BC[0];
      state[24] = BC[4] ^ ~BC[0] & BC[1];
//end Chi
    }
//   Iota
    state[0] ^= KeccakF_RoundConstants[round];
  }
}
//Host Keccak-f function (pb with using the same constants between host and device) 
//unrolled

void KeccakFunr_h(tKeccakLane *state)
{
  unsigned int round;
//try to avoid to many registers
  tKeccakLane BC[5];
  tKeccakLane temp;
  for (round = 0; round <= ((unsigned int )22) - 1; round += 1) {{
// Theta
      BC[0] = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
      BC[1] = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
      BC[2] = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
      BC[3] = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
      BC[4] = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];
      temp = BC[4] ^ (BC[1] << 1 ^ BC[1] >> 32 - 1);
//x=0
      state[0] ^= temp;
      state[5] ^= temp;
      state[10] ^= temp;
      state[15] ^= temp;
      state[20] ^= temp;
      temp = BC[0] ^ (BC[2] << 1 ^ BC[2] >> 32 - 1);
//x=1
      state[1] ^= temp;
      state[6] ^= temp;
      state[11] ^= temp;
      state[16] ^= temp;
      state[21] ^= temp;
      temp = BC[1] ^ (BC[3] << 1 ^ BC[3] >> 32 - 1);
//x=2
      state[2] ^= temp;
      state[7] ^= temp;
      state[12] ^= temp;
      state[17] ^= temp;
      state[22] ^= temp;
      temp = BC[2] ^ (BC[4] << 1 ^ BC[4] >> 32 - 1);
//x=3
      state[3] ^= temp;
      state[8] ^= temp;
      state[13] ^= temp;
      state[18] ^= temp;
      state[23] ^= temp;
      temp = BC[3] ^ (BC[0] << 1 ^ BC[0] >> 32 - 1);
//x=4
      state[4] ^= temp;
      state[9] ^= temp;
      state[14] ^= temp;
      state[19] ^= temp;
      state[24] ^= temp;
//end Theta
    }
{
// Rho Pi
      temp = state[1];
      BC[0] = state[10];
      state[10] = temp << 1 ^ temp >> 32 - 1;
      temp = BC[0];
//x=0
      BC[0] = state[7];
      state[7] = temp << 3 ^ temp >> 32 - 3;
      temp = BC[0];
      BC[0] = state[11];
      state[11] = temp << 6 ^ temp >> 32 - 6;
      temp = BC[0];
      BC[0] = state[17];
      state[17] = temp << 10 ^ temp >> 32 - 10;
      temp = BC[0];
      BC[0] = state[18];
      state[18] = temp << 15 ^ temp >> 32 - 15;
      temp = BC[0];
      BC[0] = state[3];
      state[3] = temp << 21 ^ temp >> 32 - 21;
      temp = BC[0];
//x=5
      BC[0] = state[5];
      state[5] = temp << 28 ^ temp >> 32 - 28;
      temp = BC[0];
      BC[0] = state[16];
      state[16] = temp << 4 ^ temp >> 32 - 4;
      temp = BC[0];
      BC[0] = state[8];
      state[8] = temp << 13 ^ temp >> 32 - 13;
      temp = BC[0];
      BC[0] = state[21];
      state[21] = temp << 23 ^ temp >> 32 - 23;
      temp = BC[0];
      BC[0] = state[24];
      state[24] = temp << 2 ^ temp >> 32 - 2;
      temp = BC[0];
//x=10
      BC[0] = state[4];
      state[4] = temp << 14 ^ temp >> 32 - 14;
      temp = BC[0];
      BC[0] = state[15];
      state[15] = temp << 27 ^ temp >> 32 - 27;
      temp = BC[0];
      BC[0] = state[23];
      state[23] = temp << 9 ^ temp >> 32 - 9;
      temp = BC[0];
      BC[0] = state[19];
      state[19] = temp << 24 ^ temp >> 32 - 24;
      temp = BC[0];
      BC[0] = state[13];
      state[13] = temp << 8 ^ temp >> 32 - 8;
      temp = BC[0];
//x=15
      BC[0] = state[12];
      state[12] = temp << 25 ^ temp >> 32 - 25;
      temp = BC[0];
      BC[0] = state[2];
      state[2] = temp << 11 ^ temp >> 32 - 11;
      temp = BC[0];
      BC[0] = state[20];
      state[20] = temp << 30 ^ temp >> 32 - 30;
      temp = BC[0];
      BC[0] = state[14];
      state[14] = temp << 18 ^ temp >> 32 - 18;
      temp = BC[0];
      BC[0] = state[22];
      state[22] = temp << 7 ^ temp >> 32 - 7;
      temp = BC[0];
//x=20
      BC[0] = state[9];
      state[9] = temp << 29 ^ temp >> 32 - 29;
      temp = BC[0];
      BC[0] = state[6];
      state[6] = temp << 20 ^ temp >> 32 - 20;
      temp = BC[0];
      BC[0] = state[1];
      state[1] = temp << 12 ^ temp >> 32 - 12;
      temp = BC[0];
//x=23
//end Rho Pi
    }
{
//   Chi
      BC[0] = state[0];
      BC[1] = state[1];
      BC[2] = state[2];
      BC[3] = state[3];
      BC[4] = state[4];
      state[0] = BC[0] ^ ~BC[1] & BC[2];
      state[1] = BC[1] ^ ~BC[2] & BC[3];
      state[2] = BC[2] ^ ~BC[3] & BC[4];
      state[3] = BC[3] ^ ~BC[4] & BC[0];
      state[4] = BC[4] ^ ~BC[0] & BC[1];
      BC[0] = state[5];
      BC[1] = state[6];
      BC[2] = state[7];
      BC[3] = state[8];
      BC[4] = state[9];
      state[5] = BC[0] ^ ~BC[1] & BC[2];
      state[6] = BC[1] ^ ~BC[2] & BC[3];
      state[7] = BC[2] ^ ~BC[3] & BC[4];
      state[8] = BC[3] ^ ~BC[4] & BC[0];
      state[9] = BC[4] ^ ~BC[0] & BC[1];
      BC[0] = state[10];
      BC[1] = state[11];
      BC[2] = state[12];
      BC[3] = state[13];
      BC[4] = state[14];
      state[10] = BC[0] ^ ~BC[1] & BC[2];
      state[11] = BC[1] ^ ~BC[2] & BC[3];
      state[12] = BC[2] ^ ~BC[3] & BC[4];
      state[13] = BC[3] ^ ~BC[4] & BC[0];
      state[14] = BC[4] ^ ~BC[0] & BC[1];
      BC[0] = state[15];
      BC[1] = state[16];
      BC[2] = state[17];
      BC[3] = state[18];
      BC[4] = state[19];
      state[15] = BC[0] ^ ~BC[1] & BC[2];
      state[16] = BC[1] ^ ~BC[2] & BC[3];
      state[17] = BC[2] ^ ~BC[3] & BC[4];
      state[18] = BC[3] ^ ~BC[4] & BC[0];
      state[19] = BC[4] ^ ~BC[0] & BC[1];
      BC[0] = state[20];
      BC[1] = state[21];
      BC[2] = state[22];
      BC[3] = state[23];
      BC[4] = state[24];
      state[20] = BC[0] ^ ~BC[1] & BC[2];
      state[21] = BC[1] ^ ~BC[2] & BC[3];
      state[22] = BC[2] ^ ~BC[3] & BC[4];
      state[23] = BC[3] ^ ~BC[4] & BC[0];
      state[24] = BC[4] ^ ~BC[0] & BC[1];
//end Chi
    }
//   Iota
    state[0] ^= KeccakF_RoundConstants_h[round];
  }
}
//end unrolled
//Keccak final node hashing results of previous nodes in sequential mode

void Keccak_top_GPU(tKeccakLane *Kstate,tKeccakLane *inBuffer,int block_number)
{
  int ind_word;
  int k;
  for (k = 0; k <= block_number - 1; k += 1) {
    for (ind_word = 0; ind_word <= 7; ind_word += 1) {
      Kstate[ind_word] ^= inBuffer[ind_word + k * 32 / 4];
    }
    KeccakFunr_h(Kstate);
  }
}
//************************
//First Tree mode
//data to be hashed is in h_inBuffer
//output chaining values hashes are copied to h_outBuffer
//************************

void KeccakTreeGPU(tKeccakLane *h_inBuffer,tKeccakLane *h_outBuffer,const tKeccakLane *h_KeccakF_RoundConstants)
{
  for (int blkIdx = 0; blkIdx <= 63; blkIdx += 1) {
    for (int thrIdx = 0; thrIdx <= 63; thrIdx += 1) {
      int ind_word;
      int k;
      tKeccakLane Kstate[25];
//zeroize the state
      
#pragma omp parallel for private (ind_word)
      for (ind_word = 0; ind_word <= 24; ind_word += 1) {
        Kstate[ind_word] = 0;
      }
      for (k = 0; k <= 1023; k += 1) {
//xor input into state
        for (ind_word = 0; ind_word <= 7; ind_word += 1) {
          Kstate[ind_word] ^= h_inBuffer[thrIdx + ind_word * 64 + k * 64 * 32 / 4 + blkIdx * 64 * 32 / 4 * 1024];
        }
//apply GPU Keccak permutation
        KeccakFunr(Kstate,h_KeccakF_RoundConstants);
      }
//output hash in buffer
      for (ind_word = 0; ind_word <= 7; ind_word += 1) {
        h_outBuffer[thrIdx + ind_word * 64 + blkIdx * 64 * 32 / 4] = Kstate[ind_word];
      }
    }
  }
}
