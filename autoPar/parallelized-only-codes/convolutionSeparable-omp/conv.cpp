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
#include <omp.h>
#include <assert.h>
#include "conv.h"
#define ROWS_BLOCKDIM_X       16
#define COLUMNS_BLOCKDIM_X    16
#define ROWS_BLOCKDIM_Y       4
#define COLUMNS_BLOCKDIM_Y    8
#define ROWS_RESULT_STEPS     8
#define COLUMNS_RESULT_STEPS  8
#define ROWS_HALO_STEPS       1
#define COLUMNS_HALO_STEPS    1
#include <omp.h> 

void convolutionRows(float *dst,const float *src,const float *kernel,const int imageW,const int imageH,const int pitch)
{
  ((bool )(16 * 1 >= 8))?((void )0) : __assert_fail("ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS","conv.cpp",35,__PRETTY_FUNCTION__);
  ((bool )(imageW % (8 * 16) == 0))?((void )0) : __assert_fail("imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0","conv.cpp",36,__PRETTY_FUNCTION__);
  ((bool )(imageH % 4 == 0))?((void )0) : __assert_fail("imageH % ROWS_BLOCKDIM_Y == 0","conv.cpp",37,__PRETTY_FUNCTION__);
  int teamX = imageW / 8 / 16;
  int teamY = imageH / 4;
  int numTeams = teamX * teamY;
{
    float l_Data[4][160];
{
      int gidX = omp_get_team_num() % teamX;
      int gidY = omp_get_team_num() / teamX;
      int lidX = omp_get_thread_num() % 16;
      int lidY = omp_get_thread_num() / 16;
//Offset to the left halo edge
      const int baseX = (gidX * 8 - 1) * 16 + lidX;
      const int baseY = gidY * 4 + lidY;
#if 1
      const float *src_new = src + baseY * pitch + baseX;
      float *dst_new = dst + baseY * pitch + baseX;
#else
#endif
//Load main data
      
#pragma omp parallel for
      for (int i = 1; i <= 8; i += 1) {
#if 1
        l_Data[lidY][lidX + i * 16] = src_new[i * 16];
      }
#else
#endif
//Load left halo
      
#pragma omp parallel for
      for (int i = 0; i <= 0; i += 1) {
#if 1
        l_Data[lidY][lidX + i * 16] = (baseX + i * 16 >= 0?src_new[i * 16] : 0);
      }
#else
#endif
//Load right halo
      
#pragma omp parallel for firstprivate (imageW,lidX,lidY,baseX)
      for (int i = 1 + 8; i <= 9; i += 1) {
#if 1
        l_Data[lidY][lidX + i * 16] = (baseX + i * 16 < imageW?src_new[i * 16] : 0);
      }
#else
#endif
//Compute and store results
      
#pragma omp parallel for private (j)
      for (int i = 1; i <= 8; i += 1) {
        float sum = 0;
        
#pragma omp parallel for reduction (+:sum) firstprivate (lidX,lidY)
        for (int j = - 8; j <= 8; j += 1) {
          sum += kernel[8 - j] * l_Data[lidY][lidX + i * 16 + j];
        }
#if 1
        dst_new[i * 16] = sum;
#else
#endif
      }
    }
  }
}

void convolutionColumns(float *dst,const float *src,const float *kernel,const int imageW,const int imageH,const int pitch)
{
  ((bool )(8 * 1 >= 8))?((void )0) : __assert_fail("COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS","conv.cpp",111,__PRETTY_FUNCTION__);
  ((bool )(imageW % 16 == 0))?((void )0) : __assert_fail("imageW % COLUMNS_BLOCKDIM_X == 0","conv.cpp",112,__PRETTY_FUNCTION__);
  ((bool )(imageH % (8 * 8) == 0))?((void )0) : __assert_fail("imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0","conv.cpp",113,__PRETTY_FUNCTION__);
  int teamX = imageW / 16;
  int teamY = imageH / 8 / 8;
  int numTeams = teamX * teamY;
{
    float l_Data[16][81];
{
      int gidX = omp_get_team_num() % teamX;
      int gidY = omp_get_team_num() / teamX;
      int lidX = omp_get_thread_num() % 16;
      int lidY = omp_get_thread_num() / 16;
//Offset to the upper halo edge
      const int baseX = gidX * 16 + lidX;
      const int baseY = (gidY * 8 - 1) * 8 + lidY;
#if 1
      const float *src_new = src + baseY * pitch + baseX;
      float *dst_new = dst + baseY * pitch + baseX;
#else
#endif
//Load main data
      
#pragma omp parallel for
      for (int i = 1; i <= 8; i += 1) {
#if 1
        l_Data[lidX][lidY + i * 8] = src_new[i * 8 * pitch];
      }
#else
#endif
//Load upper halo
      
#pragma omp parallel for
      for (int i = 0; i <= 0; i += 1) {
#if 1
        l_Data[lidX][lidY + i * 8] = (baseY + i * 8 >= 0?src_new[i * 8 * pitch] : 0);
      }
#else
#endif
//Load lower halo
      
#pragma omp parallel for firstprivate (imageH,lidX,lidY,baseY)
      for (int i = 1 + 8; i <= 9; i += 1) {
#if 1
        l_Data[lidX][lidY + i * 8] = (baseY + i * 8 < imageH?src_new[i * 8 * pitch] : 0);
      }
#else
#endif
//Compute and store results
      
#pragma omp parallel for private (j) firstprivate (pitch)
      for (int i = 1; i <= 8; i += 1) {
        float sum = 0;
        
#pragma omp parallel for reduction (+:sum) firstprivate (lidX,lidY)
        for (int j = - 8; j <= 8; j += 1) {
          sum += kernel[8 - j] * l_Data[lidX][lidY + i * 8 + j];
        }
#if 1
        dst_new[i * 8 * pitch] = sum;
#else
#endif
      }
    }
  }
}
