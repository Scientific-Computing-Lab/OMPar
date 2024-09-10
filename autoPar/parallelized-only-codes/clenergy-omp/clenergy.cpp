/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
 * GPU accelerated coulombic potential grid test code
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "WKFUtils.h"
#define MAXATOMS 4000
#define UNROLLX       8
#define UNROLLY       1
#define BLOCKSIZEX    8
#define BLOCKSIZEY    8 
#define BLOCKSIZE    BLOCKSIZEX * BLOCKSIZEY
#include <omp.h> 

struct float4 
{
  float x;
  float y;
  float z;
  float w;
}
;

struct int3 
{
  int x;
  int y;
  int z;
}
;

int copyatoms(float *atoms,int count,float zplane,struct float4 *atominfo)
{
  if (count > 4000) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return - 1;
  }
  int i;
  for (i = 0; i <= count - 1; i += 1) {
    atominfo[i] . x = atoms[i * 4];
    atominfo[i] . y = atoms[i * 4 + 1];
    float dz = zplane - atoms[i * 4 + 2];
    atominfo[i] . z = dz * dz;
    atominfo[i] . w = atoms[i * 4 + 3];
  }
  return 0;
}

int initatoms(float **atombuf,int count,struct int3 volsize,float gridspacing)
{
  struct float4 size;
  int i;
  float *atoms;
  srand(2);
  atoms = ((float *)(malloc((count * 4) * sizeof(float ))));
   *atombuf = atoms;
// compute grid dimensions in angstroms
  size . x = gridspacing * volsize . x;
  size . y = gridspacing * volsize . y;
  size . z = gridspacing * volsize . z;
  for (i = 0; i <= count - 1; i += 1) {
    int addr = i * 4;
    atoms[addr] = (rand()) / ((float )2147483647) * size . x;
    atoms[addr + 1] = (rand()) / ((float )2147483647) * size . y;
    atoms[addr + 2] = (rand()) / ((float )2147483647) * size . z;
    atoms[addr + 3] = (((rand()) / ((float )2147483647)) * 2.0 - 1.0);
// charge
  }
  return 0;
}

int main(int argc,char **argv)
{
  float *energy = 0L;
  float *atoms = 0L;
  struct int3 volsize;
  wkf_timerhandle runtimer;
  wkf_timerhandle mastertimer;
  wkf_timerhandle copytimer;
  wkf_timerhandle hostcopytimer;
  float copytotal;
  float runtotal;
  float mastertotal;
  float hostcopytotal;
  const char *statestr = "|/-\\.";
  int state = 0;
  printf("GPU accelerated coulombic potential microbenchmark\n");
  printf("--------------------------------------------------------\n");
  printf("  Single-threaded single-device test run.\n");
// number of atoms to simulate
  int atomcount = 1000000;
// setup energy grid size
// XXX this is a large test case to clearly illustrate that even while
//     the CUDA kernel is running entirely on the GPU, the CUDA runtime
//     library is soaking up the entire host CPU for some reason.
  volsize . x = 768;
  volsize . y = 768;
  volsize . z = 1;
// set voxel spacing
  float gridspacing = 0.1f;
// setup CUDA grid and block sizes
// XXX we have to make a trade-off between the number of threads per
//     block and the resulting padding size we'll end up with since
//     each thread will do several consecutive grid cells in this version,
//     we're using up some of our available parallelism to reduce overhead.
// initialize the wall clock timers
  runtimer = wkf_timer_create();
  mastertimer = wkf_timer_create();
  copytimer = wkf_timer_create();
  hostcopytimer = wkf_timer_create();
  copytotal = 0;
  runtotal = 0;
  hostcopytotal = 0;
  printf("Grid size: %d x %d x %d\n",volsize . x,volsize . y,volsize . z);
  printf("Running kernel(atoms:%d, gridspacing %g, z %d)\n",atomcount,gridspacing,0);
// allocate and initialize atom coordinates and charges
  if ((initatoms(&atoms,atomcount,volsize,gridspacing))) 
    return - 1;
// allocate and initialize the GPU output array
  int volmem = volsize . x * volsize . y * volsize . z;
  int volmemsz = (sizeof(float ) * volmem);
  printf("Allocating %.2fMB of memory for output buffer...\n",volmemsz / (1024.0 * 1024.0));
  energy = ((float *)(malloc(volmemsz)));
  struct float4 *atominfo = (struct float4 *)(malloc(4000 * sizeof(struct float4 )));
  printf("starting run...\n");
  wkf_timer_start(mastertimer);
  int iterations = 0;
{
    
#pragma omp parallel for firstprivate (volmem)
    for (int i = 0; i <= volmem - 1; i += 1) {
      energy[i] = 0.f;
    }
    int atomstart;
    for (atomstart = 0; atomstart <= atomcount - 1; atomstart += 4000) {
      iterations++;
      int runatoms;
      int atomsremaining = atomcount - atomstart;
      if (atomsremaining > 4000) 
        runatoms = 4000;
       else 
        runatoms = atomsremaining;
      printf("%c\r",statestr[state]);
      fflush(stdout);
      state = state + 1 & 3;
// copy the atoms to the GPU
      wkf_timer_start(copytimer);
      if ((copyatoms(atoms + 4 * atomstart,runatoms,0 * gridspacing,atominfo))) 
        return - 1;
      wkf_timer_stop(copytimer);
      copytotal += wkf_timer_time(copytimer);
// RUN the kernel...
      wkf_timer_start(runtimer);
      for (unsigned int yindex = 0; yindex <= ((unsigned int )volsize . y) - 1; yindex += 1) {
        for (unsigned int xindex = 0; xindex <= ((unsigned int )(volsize . x / 8)) - 1; xindex += 1) {
          unsigned int outaddr = yindex * volsize . x + xindex;
          float coory = gridspacing * yindex;
          float coorx = gridspacing * xindex;
          float energyvalx1 = 0.0f;
          float energyvalx2 = 0.0f;
          float energyvalx3 = 0.0f;
          float energyvalx4 = 0.0f;
          float energyvalx5 = 0.0f;
          float energyvalx6 = 0.0f;
          float energyvalx7 = 0.0f;
          float energyvalx8 = 0.0f;
          float gridspacing_u = gridspacing * 8;
//
// XXX 59/8 FLOPS per atom
//
          int atomid;
          for (atomid = 0; atomid <= runatoms - 1; atomid += 1) {
            float dy = coory - atominfo[atomid] . y;
            float dyz2 = dy * dy + atominfo[atomid] . z;
            float dx1 = coorx - atominfo[atomid] . x;
            float dx2 = dx1 + gridspacing_u;
            float dx3 = dx2 + gridspacing_u;
            float dx4 = dx3 + gridspacing_u;
            float dx5 = dx4 + gridspacing_u;
            float dx6 = dx5 + gridspacing_u;
            float dx7 = dx6 + gridspacing_u;
            float dx8 = dx7 + gridspacing_u;
            energyvalx1 += atominfo[atomid] . w / sqrtf(dx1 * dx1 + dyz2);
            energyvalx2 += atominfo[atomid] . w / sqrtf(dx2 * dx2 + dyz2);
            energyvalx3 += atominfo[atomid] . w / sqrtf(dx3 * dx3 + dyz2);
            energyvalx4 += atominfo[atomid] . w / sqrtf(dx4 * dx4 + dyz2);
            energyvalx5 += atominfo[atomid] . w / sqrtf(dx5 * dx5 + dyz2);
            energyvalx6 += atominfo[atomid] . w / sqrtf(dx6 * dx6 + dyz2);
            energyvalx7 += atominfo[atomid] . w / sqrtf(dx7 * dx7 + dyz2);
            energyvalx8 += atominfo[atomid] . w / sqrtf(dx8 * dx8 + dyz2);
          }
          energy[outaddr] += energyvalx1;
          energy[outaddr + (1 * 8)] += energyvalx2;
          energy[outaddr + (2 * 8)] += energyvalx3;
          energy[outaddr + (3 * 8)] += energyvalx4;
          energy[outaddr + (4 * 8)] += energyvalx5;
          energy[outaddr + (5 * 8)] += energyvalx6;
          energy[outaddr + (6 * 8)] += energyvalx7;
          energy[outaddr + (7 * 8)] += energyvalx8;
        }
      }
      wkf_timer_stop(runtimer);
      runtotal += wkf_timer_time(runtimer);
    }
    printf("Done\n");
    wkf_timer_stop(mastertimer);
    mastertotal = (wkf_timer_time(mastertimer));
  }
// Copy the GPU output data back to the host and use/store it..
  wkf_timer_start(hostcopytimer);
  wkf_timer_stop(hostcopytimer);
  hostcopytotal = (wkf_timer_time(hostcopytimer));
  int i;
  int j;
  for (j = 0; j <= 7; j += 1) {
    for (i = 0; i <= 7; i += 1) {
      int addr = j * volsize . x + i;
      printf("[%d] %.1f ",addr,energy[addr]);
    }
    printf("\n");
  }
  printf("Final calculation required %d iterations of %d atoms\n",iterations,4000);
  printf("Copy time: %f seconds, %f per iteration\n",copytotal,(copytotal / ((float )iterations)));
  printf("Kernel time: %f seconds, %f per iteration\n",runtotal,(runtotal / ((float )iterations)));
  printf("Total time: %f seconds\n",mastertotal);
  printf("Kernel invocation rate: %f iterations per second\n",(iterations / mastertotal));
  printf("GPU to host copy bandwidth: %gMB/sec, %f seconds total\n",volmemsz / (1024.0 * 1024.0) / hostcopytotal,hostcopytotal);
  double atomevalssec = ((double )volsize . x) * volsize . y * volsize . z * atomcount / (mastertotal * 1000000000.0);
  printf("Efficiency metric, %g billion atom evals per second\n",atomevalssec);
/* 59/8 FLOPS per atom eval */
  printf("FP performance: %g GFLOPS\n",atomevalssec * (59.0 / 8.0));
  free(atoms);
  free(atominfo);
  free(energy);
  return 0;
}
