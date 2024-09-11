/*
   GPU Implementation of Keccak by Guillaume Sevestre, 2010

   This code is hereby put in the public domain.
   It is given as is, without any guarantee.
 */
#include <stdio.h>
#include <stdlib.h>
#include "KeccakF.h"
#include "KeccakTree.h"
// 22 rounds constants
// Study constant memory best placement with Cuda (textures ?)
#include <omp.h> 
const tKeccakLane KeccakF_RoundConstantsCPU[22] = {((tKeccakLane )0x00000001), ((tKeccakLane )0x00008082), ((tKeccakLane )0x0000808a), ((tKeccakLane )0x80008000), ((tKeccakLane )0x0000808b), ((tKeccakLane )0x80000001), ((tKeccakLane )0x80008081), ((tKeccakLane )0x00008009), ((tKeccakLane )0x0000008a), ((tKeccakLane )0x00000088), ((tKeccakLane )0x80008009), ((tKeccakLane )0x8000000a), ((tKeccakLane )0x8000808b), ((tKeccakLane )0x0000008b), ((tKeccakLane )0x00008089), ((tKeccakLane )0x00008003), ((tKeccakLane )0x00008002), ((tKeccakLane )0x00000080), ((tKeccakLane )0x0000800a), ((tKeccakLane )0x8000000a), ((tKeccakLane )0x80008081), ((tKeccakLane )0x00008080)};
//INFO It could be more optimized to use unsigned char on an 8-bit CPU
const unsigned int KeccakF_RotationConstants[25] = {(1), (3), (6), (10), (15), (21), (28), (4), (13), (23), (2), (14), (27), (9), (24), (8), (25), (11), (30), (18), (7), (29), (20), (12)
// 1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14, 27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};
//INFO It could be more optimized to use unsigned char on an 8-bit CPU
const unsigned int KeccakF_PiLane[25] = {(10), (7), (11), (17), (18), (3), (5), (16), (8), (21), (24), (4), (15), (23), (19), (13), (12), (2), (20), (14), (22), (9), (6), (1)};
//INFO It could be more optimized to use unsigned char on an 8-bit CPU
const unsigned int KeccakF_Mod5[10] = {(0), (1), (2), (3), (4), (0), (1), (2), (3), (4)};

void KeccakF(tKeccakLane *state)
{
  unsigned int x;
  unsigned int y;
  unsigned int round;
//try to avoid to many registers
  tKeccakLane BC[5];
  tKeccakLane temp;
  for (round = 0; round <= ((unsigned int )22) - 1; round += 1) {
// Theta
    
#pragma omp parallel for private (x)
    for (x = 0; x <= ((unsigned int )5) - 1; x += 1) 
// derouler
{
      BC[x] = state[x] ^ state[5 + x] ^ state[10 + x] ^ state[15 + x] ^ state[20 + x];
    }
    for (x = 0; x <= ((unsigned int )5) - 1; x += 1) 
// derouler
{
      temp = BC[KeccakF_Mod5[x + 4]] ^ (BC[KeccakF_Mod5[x + 1]] << 1 ^ BC[KeccakF_Mod5[x + 1]] >> 32 - 1);
//expliciter
      
#pragma omp parallel for private (y) firstprivate (temp)
      for (y = 0; y <= ((unsigned int )25) - 1; y += 5) 
// derouler
{
        state[y + x] ^= temp;
      }
    }
// Rho Pi
    temp = state[1];
    for (x = 0; x <= ((unsigned int )24) - 1; x += 1) 
//expliciter + rotation modulo 32
{
      BC[0] = state[KeccakF_PiLane[x]];
      state[KeccakF_PiLane[x]] = temp << KeccakF_RotationConstants[x] ^ temp >> 32 - KeccakF_RotationConstants[x];
      temp = BC[0];
    }
//  Chi
    for (y = 0; y <= ((unsigned int )25) - 1; y += 5) {
      BC[0] = state[y + 0];
      BC[1] = state[y + 1];
      BC[2] = state[y + 2];
      BC[3] = state[y + 3];
      BC[4] = state[y + 4];
      
#pragma omp parallel for private (x)
      for (x = 0; x <= ((unsigned int )5) - 1; x += 1) {
        state[y + x] = BC[x] ^ ~BC[KeccakF_Mod5[x + 1]] & BC[KeccakF_Mod5[x + 2]];
      }
    }
//  Iota
    state[0] ^= KeccakF_RoundConstantsCPU[round];
  }
}

void KeccakF_CPU(tKeccakLane *state)
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
//x=0
      state[3] ^= temp;
      state[8] ^= temp;
      state[13] ^= temp;
      state[18] ^= temp;
      state[23] ^= temp;
      temp = BC[3] ^ (BC[0] << 1 ^ BC[0] >> 32 - 1);
//x=0
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
//  Chi
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
//  Iota
    state[0] ^= KeccakF_RoundConstantsCPU[round];
  }
}
// Absorb blocks in top of tree keccak hash function
// inBuffer supposed to have block_number * output_block_size of data

void Keccak_top(tKeccakLane *Kstate,tKeccakLane *inBuffer,int block_number)
{
  int ind_word;
  int k;
  for (k = 0; k <= block_number - 1; k += 1) {
    for (ind_word = 0; ind_word <= 7; ind_word += 1) {
      Kstate[ind_word] ^= inBuffer[ind_word + k * 32 / 4];
    }
    KeccakF_CPU(Kstate);
  }
}
//**************************
//Functions on Keccak state (seroize, print, compare)
//**************************

void zeroize(tKeccakLane *Kstate)
{
  int ind_word;
  
#pragma omp parallel for private (ind_word)
  for (ind_word = 0; ind_word <= 24; ind_word += 1) {
    Kstate[ind_word] = 0;
  }
}

void print_KS(tKeccakLane *Kstate)
{
  unsigned int x;
  unsigned int y;
  for (x = 0; x <= ((unsigned int )5) - 1; x += 1) {
    for (y = 0; y <= ((unsigned int )5) - 1; y += 1) {
      printf(" KS[%1u][%1u] %08x ",x,y,Kstate[x + 5 * y]);
    }
    printf("\n");
  }
  printf("\n");
}
// print first 256 bits : output of Keccak hash

void print_KS_256(tKeccakLane *Kstate)
{
  unsigned int x;
  printf("\n");
  for (x = 0; x <= ((unsigned int )8) - 1; x += 1) {
    printf("%08x ",Kstate[x]);
  }
  printf("\n");
}
// Test equality of Keccak States

int isEqual_KS(tKeccakLane *Ks1,tKeccakLane *Ks2)
{
  unsigned int x;
  unsigned int y;
  int res = 0;
  
#pragma omp parallel for private (res,x,y)
  for (x = 0; x <= ((unsigned int )5) - 1; x += 1) {
    
#pragma omp parallel for private (res,y)
    for (y = 0; y <= ((unsigned int )5) - 1; y += 1) {
      res = (Ks1[x + 5 * y] == Ks2[x + 5 * y]);
      if (res == 0) 
        return 0;
    }
  }
  return 1;
}
