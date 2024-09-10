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


inline void operator+=(float4 &a, float4 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline float4 operator+(float4 a, float4 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w};
}

inline int4 operator+(int4 a, int4 b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w};
}

inline float4 operator*(float4 a, float4 b)
{
    return {a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w};
}

inline float4 operator-(float4 a, float4 b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w};
}

inline float4 operator*(float4 a, float b)
{
    return {a.x * b, a.y * b, a.z * b,  a.w * b};
}

inline float4 operator*(float b, float4 a)
{
    return {b * a.x, b * a.y, b * a.z, b * a.w};
}

inline void operator*=(float4 &a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

////////////////////////////////////////////////////////////////////////////////
// Save particle grid cell hashes and indices
////////////////////////////////////////////////////////////////////////////////
int4 getGridPos(const float4 p, const simParams_t &params)
{
  int4 gridPos;
  gridPos.x = (int)floorf((p.x - params.worldOrigin.x) / params.cellSize.x);
  gridPos.y = (int)floorf((p.y - params.worldOrigin.y) / params.cellSize.y);
  gridPos.z = (int)floorf((p.z - params.worldOrigin.z) / params.cellSize.z);
  gridPos.w = 0;
  return gridPos;
}

//Calculate address in grid from position (clamping to edges)
unsigned int getGridHash(int4 gridPos, const simParams_t &params)
{
  //Wrap addressing, assume power-of-two grid dimensions
  gridPos.x = gridPos.x & (params.gridSize.x - 1);
  gridPos.y = gridPos.y & (params.gridSize.y - 1);
  gridPos.z = gridPos.z & (params.gridSize.z - 1);
  return UMAD( UMAD(gridPos.z, params.gridSize.y, gridPos.y), params.gridSize.x, gridPos.x );
}

////////////////////////////////////////////////////////////////////////////////
// Process collisions (calculate accelerations)
////////////////////////////////////////////////////////////////////////////////
float4 collideSpheres(
    float4 posA,
    float4 posB,
    float4 velA,
    float4 velB,
    float radiusA,
    float radiusB,
    float spring,
    float damping,
    float shear,
    float attraction)
{
  //Calculate relative position
  float4     relPos = {posB.x - posA.x, posB.y - posA.y, posB.z - posA.z, 0};
  float        dist = sqrtf(relPos.x * relPos.x + relPos.y * relPos.y + relPos.z * relPos.z);
  float collideDist = radiusA + radiusB;

  float4 force = {0, 0, 0, 0};
  if(dist < collideDist){
    float4 norm = {relPos.x / dist, relPos.y / dist, relPos.z / dist, 0};

    //Relative velocity
    float4 relVel = {velB.x - velA.x, velB.y - velA.y, velB.z - velA.z, 0};

    //Relative tangential velocity
    float relVelDotNorm = relVel.x * norm.x + relVel.y * norm.y + relVel.z * norm.z;
    float4 tanVel = {relVel.x - relVelDotNorm * norm.x, relVel.y - relVelDotNorm * norm.y, 
      relVel.z - relVelDotNorm * norm.z, 0};

    //Spring force (potential)
    float springFactor = -spring * (collideDist - dist);
    force = {
      springFactor * norm.x + damping * relVel.x + shear * tanVel.x + attraction * relPos.x,
      springFactor * norm.y + damping * relVel.y + shear * tanVel.y + attraction * relPos.y,
      springFactor * norm.z + damping * relVel.z + shear * tanVel.z + attraction * relPos.z,
      0
    };
  }

  return force;
}

