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
#include <assert.h>
#include <math.h>
#include "DCT8x8.h"
////////////////////////////////////////////////////////////////////////////////
// Straightforward general-sized (i)DCT with O(N ** 2) complexity
// so that we don't forget what we're calculating :)
////////////////////////////////////////////////////////////////////////////////
#define PI 3.14159265358979323846264338327950288f

static void DCT8(float *dst,const float *src,unsigned int ostride,unsigned int istride)
{
  float X07P = src[0 * istride] + src[7 * istride];
  float X16P = src[1 * istride] + src[6 * istride];
  float X25P = src[2 * istride] + src[5 * istride];
  float X34P = src[3 * istride] + src[4 * istride];
  float X07M = src[0 * istride] - src[7 * istride];
  float X61M = src[6 * istride] - src[1 * istride];
  float X25M = src[2 * istride] - src[5 * istride];
  float X43M = src[4 * istride] - src[3 * istride];
  float X07P34PP = X07P + X34P;
  float X07P34PM = X07P - X34P;
  float X16P25PP = X16P + X25P;
  float X16P25PM = X16P - X25P;
  dst[0 * ostride] = 0.35355339059327376220042218105242f * (X07P34PP + X16P25PP);
  dst[2 * ostride] = 0.35355339059327376220042218105242f * (1.3065629648763765278566431734272f * X07P34PM + 0.54119610014619698439972320536639f * X16P25PM);
  dst[4 * ostride] = 0.35355339059327376220042218105242f * (X07P34PP - X16P25PP);
  dst[6 * ostride] = 0.35355339059327376220042218105242f * (0.54119610014619698439972320536639f * X07P34PM - 1.3065629648763765278566431734272f * X16P25PM);
  dst[1 * ostride] = 0.35355339059327376220042218105242f * (1.3870398453221474618216191915664f * X07M - 1.1758756024193587169744671046113f * X61M + 0.78569495838710218127789736765722f * X25M - 0.27589937928294301233595756366937f * X43M);
  dst[3 * ostride] = 0.35355339059327376220042218105242f * (1.1758756024193587169744671046113f * X07M + 0.27589937928294301233595756366937f * X61M - 1.3870398453221474618216191915664f * X25M + 0.78569495838710218127789736765722f * X43M);
  dst[5 * ostride] = 0.35355339059327376220042218105242f * (0.78569495838710218127789736765722f * X07M + 1.3870398453221474618216191915664f * X61M + 0.27589937928294301233595756366937f * X25M - 1.1758756024193587169744671046113f * X43M);
  dst[7 * ostride] = 0.35355339059327376220042218105242f * (0.27589937928294301233595756366937f * X07M + 0.78569495838710218127789736765722f * X61M + 1.1758756024193587169744671046113f * X25M + 1.3870398453221474618216191915664f * X43M);
}

static void IDCT8(float *dst,const float *src,unsigned int ostride,unsigned int istride)
{
  float Y04P = src[0 * istride] + src[4 * istride];
  float Y2b6eP = 1.3065629648763765278566431734272f * src[2 * istride] + 0.54119610014619698439972320536639f * src[6 * istride];
  float Y04P2b6ePP = Y04P + Y2b6eP;
  float Y04P2b6ePM = Y04P - Y2b6eP;
  float Y7f1aP3c5dPP = 0.27589937928294301233595756366937f * src[7 * istride] + 1.3870398453221474618216191915664f * src[1 * istride] + 1.1758756024193587169744671046113f * src[3 * istride] + 0.78569495838710218127789736765722f * src[5 * istride];
  float Y7a1fM3d5cMP = 1.3870398453221474618216191915664f * src[7 * istride] - 0.27589937928294301233595756366937f * src[1 * istride] + 0.78569495838710218127789736765722f * src[3 * istride] - 1.1758756024193587169744671046113f * src[5 * istride];
  float Y04M = src[0 * istride] - src[4 * istride];
  float Y2e6bM = 0.54119610014619698439972320536639f * src[2 * istride] - 1.3065629648763765278566431734272f * src[6 * istride];
  float Y04M2e6bMP = Y04M + Y2e6bM;
  float Y04M2e6bMM = Y04M - Y2e6bM;
  float Y1c7dM3f5aPM = 1.1758756024193587169744671046113f * src[1 * istride] - 0.78569495838710218127789736765722f * src[7 * istride] - 0.27589937928294301233595756366937f * src[3 * istride] - 1.3870398453221474618216191915664f * src[5 * istride];
  float Y1d7cP3a5fMM = 0.78569495838710218127789736765722f * src[1 * istride] + 1.1758756024193587169744671046113f * src[7 * istride] - 1.3870398453221474618216191915664f * src[3 * istride] + 0.27589937928294301233595756366937f * src[5 * istride];
  dst[0 * ostride] = 0.35355339059327376220042218105242f * (Y04P2b6ePP + Y7f1aP3c5dPP);
  dst[7 * ostride] = 0.35355339059327376220042218105242f * (Y04P2b6ePP - Y7f1aP3c5dPP);
  dst[4 * ostride] = 0.35355339059327376220042218105242f * (Y04P2b6ePM + Y7a1fM3d5cMP);
  dst[3 * ostride] = 0.35355339059327376220042218105242f * (Y04P2b6ePM - Y7a1fM3d5cMP);
  dst[1 * ostride] = 0.35355339059327376220042218105242f * (Y04M2e6bMP + Y1c7dM3f5aPM);
  dst[5 * ostride] = 0.35355339059327376220042218105242f * (Y04M2e6bMM - Y1d7cP3a5fMM);
  dst[2 * ostride] = 0.35355339059327376220042218105242f * (Y04M2e6bMM + Y1d7cP3a5fMM);
  dst[6 * ostride] = 0.35355339059327376220042218105242f * (Y04M2e6bMP - Y1c7dM3f5aPM);
}

void DCT8x8CPU(float *dst,const float *src,unsigned int stride,unsigned int imageH,unsigned int imageW,int dir)
{
  ((bool )(dir == 666 || dir == 777))?((void )0) : __assert_fail("(dir == DCT_FORWARD) || (dir == DCT_INVERSE)","DCT8x8_gold.cpp",78,__PRETTY_FUNCTION__);
  for (unsigned int i = 0; i + 8 - 1 < imageH; i += 8) {
    for (unsigned int j = 0; j + 8 - 1 < imageW; j += 8) {
//process rows
      for (unsigned int k = 0; k <= ((unsigned int )8) - 1; k += 1) {
        if (dir == 666) 
          DCT8(dst + (i + k) * stride + j,src + (i + k) * stride + j,1,1);
         else 
          IDCT8(dst + (i + k) * stride + j,src + (i + k) * stride + j,1,1);
      }
//process columns
      for (unsigned int k = 0; k <= ((unsigned int )8) - 1; k += 1) {
        if (dir == 666) 
          DCT8(dst + i * stride + (j + k),(dst + i * stride + (j + k)),stride,stride);
         else 
          IDCT8(dst + i * stride + (j + k),(dst + i * stride + (j + k)),stride,stride);
      }
    }
  }
}
