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
//Standard utilities and systems includes
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "particles.h"
#include "particles_kernels.cpp"
//Simulation parameters
#include <omp.h> 
static const size_t wgSize = 64;

static size_t uSnap(size_t a,size_t b)
{
  return a % b == 0?a : a - a % b + b;
}

void integrateSystem(struct float4 *d_Pos,struct float4 *d_Vel,const simParams_t &params,const float deltaTime,const unsigned int numParticles)
{
  size_t globalWorkSize = uSnap(numParticles,wgSize);
  for (unsigned int index = 0; index <= numParticles - 1; index += 1) {
    struct float4 pos = d_Pos[index];
    struct float4 vel = d_Vel[index];
    pos . w = 1.0f;
    vel . w = 0.0f;
//Gravity
    struct float4 g = {params . gravity . x, params . gravity . y, params . gravity . z, (0)};
    vel+=g*deltaTime;
    vel*=params . globalDamping;
//Advance pos
    pos+=vel*deltaTime;
//printf("before %d %3.f %3.f %3.f\n", index, pos.x, pos.y, pos.z);
//Collide with cube
    if (pos . x < - 1.0f + params . particleRadius) {
      pos . x = - 1.0f + params . particleRadius;
      vel . x *= params . boundaryDamping;
    }
    if (pos . x > 1.0f - params . particleRadius) {
      pos . x = 1.0f - params . particleRadius;
      vel . x *= params . boundaryDamping;
    }
    if (pos . y < - 1.0f + params . particleRadius) {
      pos . y = - 1.0f + params . particleRadius;
      vel . y *= params . boundaryDamping;
    }
    if (pos . y > 1.0f - params . particleRadius) {
      pos . y = 1.0f - params . particleRadius;
      vel . y *= params . boundaryDamping;
    }
    if (pos . z < - 1.0f + params . particleRadius) {
      pos . z = - 1.0f + params . particleRadius;
      vel . z *= params . boundaryDamping;
    }
    if (pos . z > 1.0f - params . particleRadius) {
      pos . z = 1.0f - params . particleRadius;
      vel . z *= params . boundaryDamping;
    }
//Store new position and velocity
    d_Pos[index] = pos;
    d_Vel[index] = vel;
  }
}

void calcHash(unsigned int *d_Hash,unsigned int *d_Index,struct float4 *d_Pos,const simParams_t &params,const int numParticles)
{
  size_t globalWorkSize = uSnap(numParticles,wgSize);
  
#pragma omp parallel for firstprivate (numParticles)
  for (unsigned int index = 0; index <= ((unsigned int )numParticles) - 1; index += 1) {
    struct float4 p = d_Pos[index];
//Get address in grid
    struct int4 gridPos = getGridPos(p,params);
    unsigned int gridHash = getGridHash(gridPos,params);
//Store grid hash and particle index
    d_Hash[index] = gridHash;
    d_Index[index] = index;
  }
}

void memSet(unsigned int *d_Data,unsigned int val,unsigned int N)
{
  size_t globalWorkSize = uSnap(N,wgSize);
  
#pragma omp parallel for firstprivate (val,N)
  for (unsigned int i = 0; i <= N - 1; i += 1) {
    d_Data[i] = val;
  }
}

void findCellBoundsAndReorder(unsigned int *d_CellStart,unsigned int *d_CellEnd,struct float4 *d_ReorderedPos,struct float4 *d_ReorderedVel,unsigned int *d_Hash,unsigned int *d_Index,struct float4 *d_Pos,struct float4 *d_Vel,const unsigned int numParticles,const unsigned int numCells)
{
  memSet(d_CellStart,0xFFFFFFFFU,numCells);
  size_t globalWorkSize = uSnap(numParticles,wgSize);
{
    unsigned int localHash[65];
{
      unsigned int hash;
      int lid = omp_get_thread_num();
      int index = ((omp_get_team_num()) * wgSize + lid);
//Handle case when no. of particles not multiple of block size
      if (index < numParticles) {
        hash = d_Hash[index];
//Load hash data into local memory so that we can look 
//at neighboring particle's hash value without loading
//two hash values per thread
        localHash[lid + 1] = hash;
//First thread in block must load neighbor particle hash
        if (index > 0 && lid == 0) 
          localHash[0] = d_Hash[index - 1];
      }
      if (index < numParticles) {
//Border case
        if (index == 0) 
          d_CellStart[hash] = 0;
         else 
//Main case
{
          if (hash != localHash[lid]) 
            d_CellEnd[localHash[lid]] = d_CellStart[hash] = index;
        }
        ;
//Another border case
        if (index == numParticles - 1) 
          d_CellEnd[hash] = numParticles;
//Now use the sorted index to reorder the pos and vel arrays
        unsigned int sortedIndex = d_Index[index];
        struct float4 pos = d_Pos[sortedIndex];
        struct float4 vel = d_Vel[sortedIndex];
        d_ReorderedPos[index] = pos;
        d_ReorderedVel[index] = vel;
      }
    }
  }
}

void collide(struct float4 *d_Vel,struct float4 *d_ReorderedPos,struct float4 *d_ReorderedVel,unsigned int *d_Index,unsigned int *d_CellStart,unsigned int *d_CellEnd,const simParams_t &params,const unsigned int numParticles,const unsigned int numCells)
{
  size_t globalWorkSize = uSnap(numParticles,wgSize);
  for (unsigned int index = 0; index <= numParticles - 1; index += 1) {
    struct float4 pos = d_ReorderedPos[index];
    struct float4 vel = d_ReorderedVel[index];
    struct float4 force = {(0), (0), (0), (0)};
//Get address in grid
    struct int4 gridPos = getGridPos(pos,params);
//Accumulate surrounding cells
    for (int z = - 1; z <= 1; z += 1) {
      for (int y = - 1; y <= 1; y += 1) {
        for (int x = - 1; x <= 1; x += 1) {
//Get start particle index for this cell
          struct int4 t = {x, y, z, (0)};
          unsigned int hash = getGridHash((gridPos+t),params);
          unsigned int startI = d_CellStart[hash];
//Skip empty cell
          if (startI == 0xFFFFFFFFU) 
            continue; 
//Iterate over particles in this cell
          unsigned int endI = d_CellEnd[hash];
          for (unsigned int j = startI; j <= endI - 1; j += 1) {
            if (j == index) 
              continue; 
            struct float4 pos2 = d_ReorderedPos[j];
            struct float4 vel2 = d_ReorderedVel[j];
//Collide two spheres
            force+=collideSpheres(pos,pos2,vel,vel2,params . particleRadius,params . particleRadius,params . spring,params . damping,params . shear,params . attraction);
          }
        }
      }
    }
//Collide with cursor sphere
    force+=collideSpheres(pos,{params . colliderPos . x, params . colliderPos . y, params . colliderPos . z, (0)},vel,{(0), (0), (0), (0)},params . particleRadius,params . colliderRadius,params . spring,params . damping,params . shear,params . attraction);
//Write new velocity back to original unsorted location
    d_Vel[d_Index[index]] = vel+force;
  }
}
