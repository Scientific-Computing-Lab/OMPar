/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.10 $
 ***********************************************************************EHEADER*/
#include "headers.h"
/*--------------------------------------------------------------------------
 * hypre_GenerateLaplacian
 *--------------------------------------------------------------------------*/
#include <omp.h> 

extern "C" hypre_CSRMatrix *GenerateSeqLaplacian(int nx,int ny,int nz,double *value,hypre_Vector **rhs_ptr,hypre_Vector **x_ptr,hypre_Vector **sol_ptr)
{
  hypre_CSRMatrix *A;
  hypre_Vector *rhs;
  hypre_Vector *x;
  hypre_Vector *sol;
  double *rhs_data;
  double *x_data;
  double *sol_data;
  int *A_i;
  int *A_j;
  double *A_data;
  int ix;
  int iy;
  int iz;
  int cnt;
  int row_index;
  int i;
  int j;
  int grid_size;
  grid_size = nx * ny * nz;
  A_i = ((int *)(hypre_CAlloc(((unsigned int )(grid_size + 1)),((unsigned int )(sizeof(int ))))));
  rhs_data = ((double *)(hypre_CAlloc(((unsigned int )grid_size),((unsigned int )(sizeof(double ))))));
  x_data = ((double *)(hypre_CAlloc(((unsigned int )grid_size),((unsigned int )(sizeof(double ))))));
  sol_data = ((double *)(hypre_CAlloc(((unsigned int )grid_size),((unsigned int )(sizeof(double ))))));
  
#pragma omp parallel for private (i)
  for (i = 0; i <= grid_size - 1; i += 1) {
    x_data[i] = 0.0;
    sol_data[i] = 0.0;
    rhs_data[i] = 1.0;
  }
  cnt = 1;
  A_i[0] = 0;
  for (iz = 0; iz <= nz - 1; iz += 1) {
    for (iy = 0; iy <= ny - 1; iy += 1) {
      for (ix = 0; ix <= nx - 1; ix += 1) {
        A_i[cnt] = A_i[cnt - 1];
        A_i[cnt]++;
        if (iz) 
          A_i[cnt]++;
        if (iy) 
          A_i[cnt]++;
        if (ix) 
          A_i[cnt]++;
        if (ix + 1 < nx) 
          A_i[cnt]++;
        if (iy + 1 < ny) 
          A_i[cnt]++;
        if (iz + 1 < nz) 
          A_i[cnt]++;
        cnt++;
      }
    }
  }
  A_j = ((int *)(hypre_CAlloc(((unsigned int )A_i[grid_size]),((unsigned int )(sizeof(int ))))));
  A_data = ((double *)(hypre_CAlloc(((unsigned int )A_i[grid_size]),((unsigned int )(sizeof(double ))))));
//printf("%d\n", A_i[grid_size]);
  row_index = 0;
  cnt = 0;
  for (iz = 0; iz <= nz - 1; iz += 1) {
    for (iy = 0; iy <= ny - 1; iy += 1) {
      for (ix = 0; ix <= nx - 1; ix += 1) {
        A_j[cnt] = row_index;
        A_data[cnt++] = value[0];
        if (iz) {
          A_j[cnt] = row_index - nx * ny;
          A_data[cnt++] = value[3];
        }
        if (iy) {
          A_j[cnt] = row_index - nx;
          A_data[cnt++] = value[2];
        }
        if (ix) {
          A_j[cnt] = row_index - 1;
          A_data[cnt++] = value[1];
        }
        if (ix + 1 < nx) {
          A_j[cnt] = row_index + 1;
          A_data[cnt++] = value[1];
        }
        if (iy + 1 < ny) {
          A_j[cnt] = row_index + nx;
          A_data[cnt++] = value[2];
        }
        if (iz + 1 < nz) {
          A_j[cnt] = row_index + nx * ny;
          A_data[cnt++] = value[3];
        }
        row_index++;
      }
    }
  }
  A = hypre_CSRMatrixCreate(grid_size,grid_size,A_i[grid_size]);
  rhs = hypre_SeqVectorCreate(grid_size);
  rhs -> data = rhs_data;
  x = hypre_SeqVectorCreate(grid_size);
  x -> data = x_data;
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= grid_size - 1; i += 1) {
    for (j = A_i[i]; j <= A_i[i + 1] - 1; j += 1) {
      sol_data[i] += A_data[j];
    }
  }
  sol = hypre_SeqVectorCreate(grid_size);
  sol -> data = sol_data;
  A -> i = A_i;
  A -> j = A_j;
  A -> data = A_data;
   *rhs_ptr = rhs;
   *x_ptr = x;
   *sol_ptr = sol;
  return A;
}
