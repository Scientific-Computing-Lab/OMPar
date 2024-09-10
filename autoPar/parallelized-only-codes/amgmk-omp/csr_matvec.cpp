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
/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/
#include "headers.h"
#include <assert.h>
#ifdef _OPENMP
#include <omp.h>
#endif
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec
 *--------------------------------------------------------------------------*/
#include <omp.h> 

extern "C" int hypre_CSRMatrixMatvec(double alpha,hypre_CSRMatrix *A,hypre_Vector *x,double beta,hypre_Vector *y)
{
  double *A_data = A -> data;
  int *A_i = A -> i;
  int *A_j = A -> j;
  int num_rows = A -> num_rows;
  int num_cols = A -> num_cols;
  int *A_rownnz = A -> rownnz;
  int num_rownnz = A -> num_rownnz;
  double *x_data = x -> data;
  double *y_data = y -> data;
  int x_size = x -> size;
  int y_size = y -> size;
  int num_vectors = x -> num_vectors;
  int idxstride_y = y -> idxstride;
  int vecstride_y = y -> vecstride;
  int idxstride_x = x -> idxstride;
  int vecstride_x = x -> vecstride;
  double temp;
  double tempx;
  int i;
  int j;
  int jj;
  int m;
  double xpar = 0.7;
  int ierr = 0;
/*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
  if (!(num_vectors == y -> num_vectors)) {
    fprintf(stderr,"hypre_assert failed: %s\n","num_vectors == hypre_VectorNumVectors(y)");
    hypre_error_handler("csr_matvec.cpp",94,1);
  }
  ;
  if (num_cols != x_size) 
    ierr = 1;
  if (num_rows != y_size) 
    ierr = 2;
  if (num_cols != x_size && num_rows != y_size) 
    ierr = 3;
/*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/
  if (alpha == 0.0) {
    
#pragma omp parallel for private (i) firstprivate (beta,num_rows,num_vectors)
    for (i = 0; i <= num_rows * num_vectors - 1; i += 1) {
      y_data[i] *= beta;
    }
    return ierr;
  }
/*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/
  temp = beta / alpha;
  if (temp != 1.0) {
    if (temp == 0.0) {
      
#pragma omp parallel for private (i)
      for (i = 0; i <= num_rows * num_vectors - 1; i += 1) {
        y_data[i] = 0.0;
      }
    }
     else {
      
#pragma omp parallel for private (i) firstprivate (temp)
      for (i = 0; i <= num_rows * num_vectors - 1; i += 1) {
        y_data[i] *= temp;
      }
    }
  }
/*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/
/* use rownnz pointer to do the A*x multiplication  when num_rownnz is smaller than num_rows */
  if (num_rownnz < xpar * num_rows) {
    for (i = 0; i <= num_rownnz - 1; i += 1) {
      m = A_rownnz[i];
/*
          * for (jj = A_i[m]; jj < A_i[m+1]; jj++)
          * {
          *         j = A_j[jj];   
          *  y_data[m] += A_data[jj] * x_data[j];
          * } */
      if (num_vectors == 1) {
        tempx = y_data[m];
        
#pragma omp parallel for private (jj) reduction (+:tempx)
        for (jj = A_i[m]; jj <= A_i[m + 1] - 1; jj += 1) {
          tempx += A_data[jj] * x_data[A_j[jj]];
        }
        y_data[m] = tempx;
      }
       else {
        
#pragma omp parallel for private (tempx,j,jj) firstprivate (idxstride_y,vecstride_y,m)
        for (j = 0; j <= num_vectors - 1; j += 1) {
          tempx = y_data[j * vecstride_y + m * idxstride_y];
          
#pragma omp parallel for private (jj) reduction (+:tempx) firstprivate (idxstride_x,vecstride_x)
          for (jj = A_i[m]; jj <= A_i[m + 1] - 1; jj += 1) {
            tempx += A_data[jj] * x_data[j * vecstride_x + A_j[jj] * idxstride_x];
          }
          y_data[j * vecstride_y + m * idxstride_y] = tempx;
        }
      }
    }
  }
   else {
#ifdef _OPENMP
#endif
    for (i = 0; i <= num_rows - 1; i += 1) {
      if (num_vectors == 1) {
        temp = y_data[i];
        
#pragma omp parallel for private (jj) reduction (+:temp)
        for (jj = A_i[i]; jj <= A_i[i + 1] - 1; jj += 1) {
          temp += A_data[jj] * x_data[A_j[jj]];
        }
        y_data[i] = temp;
      }
       else {
        
#pragma omp parallel for private (temp,j,jj) firstprivate (idxstride_y,vecstride_y)
        for (j = 0; j <= num_vectors - 1; j += 1) {
          temp = y_data[j * vecstride_y + i * idxstride_y];
          
#pragma omp parallel for private (jj) reduction (+:temp) firstprivate (idxstride_x,vecstride_x)
          for (jj = A_i[i]; jj <= A_i[i + 1] - 1; jj += 1) {
            temp += A_data[jj] * x_data[j * vecstride_x + A_j[jj] * idxstride_x];
          }
          y_data[j * vecstride_y + i * idxstride_y] = temp;
        }
      }
    }
  }
/*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/
  if (alpha != 1.0) {
    
#pragma omp parallel for private (i) firstprivate (alpha,num_rows,num_vectors)
    for (i = 0; i <= num_rows * num_vectors - 1; i += 1) {
      y_data[i] *= alpha;
    }
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecT
 *
 *   Performs y <- alpha * A^T * x + beta * y
 *
 *   From Van Henson's modification of hypre_CSRMatrixMatvec.
 *--------------------------------------------------------------------------*/

extern "C" int hypre_CSRMatrixMatvecT(double alpha,hypre_CSRMatrix *A,hypre_Vector *x,double beta,hypre_Vector *y)
{
  double *A_data = A -> data;
  int *A_i = A -> i;
  int *A_j = A -> j;
  int num_rows = A -> num_rows;
  int num_cols = A -> num_cols;
  double *x_data = x -> data;
  double *y_data = y -> data;
  int x_size = x -> size;
  int y_size = y -> size;
  int num_vectors = x -> num_vectors;
  int idxstride_y = y -> idxstride;
  int vecstride_y = y -> vecstride;
  int idxstride_x = x -> idxstride;
  int vecstride_x = x -> vecstride;
  double temp;
  int i;
  int i1;
  int j;
  int jv;
  int jj;
  int ns;
  int ne;
  int size;
  int rest;
  int num_threads;
  int ierr = 0;
/*---------------------------------------------------------------------
    *  Check for size compatibility.  MatvecT returns ierr = 1 if
    *  length of X doesn't equal the number of rows of A,
    *  ierr = 2 if the length of Y doesn't equal the number of 
    *  columns of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in MatvecT, none of 
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
  if (!(num_vectors == y -> num_vectors)) {
    fprintf(stderr,"hypre_assert failed: %s\n","num_vectors == hypre_VectorNumVectors(y)");
    hypre_error_handler("csr_matvec.cpp",262,1);
  }
  ;
  if (num_rows != x_size) 
    ierr = 1;
  if (num_cols != y_size) 
    ierr = 2;
  if (num_rows != x_size && num_cols != y_size) 
    ierr = 3;
/*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/
  if (alpha == 0.0) {
    
#pragma omp parallel for private (i) firstprivate (beta,num_cols,num_vectors)
    for (i = 0; i <= num_cols * num_vectors - 1; i += 1) {
      y_data[i] *= beta;
    }
    return ierr;
  }
/*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/
  temp = beta / alpha;
  if (temp != 1.0) {
    if (temp == 0.0) {
      
#pragma omp parallel for private (i)
      for (i = 0; i <= num_cols * num_vectors - 1; i += 1) {
        y_data[i] = 0.0;
      }
    }
     else {
      
#pragma omp parallel for private (i) firstprivate (temp)
      for (i = 0; i <= num_cols * num_vectors - 1; i += 1) {
        y_data[i] *= temp;
      }
    }
  }
/*-----------------------------------------------------------------
    * y += A^T*x
    *-----------------------------------------------------------------*/
  num_threads = 1;
  if (num_threads > 1) {
    for (i1 = 0; i1 <= num_threads - 1; i1 += 1) {
      size = num_cols / num_threads;
      rest = num_cols - size * num_threads;
      if (i1 < rest) {
        ns = i1 * size + i1 - 1;
        ne = (i1 + 1) * size + i1 + 1;
      }
       else {
        ns = i1 * size + rest - 1;
        ne = (i1 + 1) * size + rest;
      }
      if (num_vectors == 1) {
        for (i = 0; i <= num_rows - 1; i += 1) {
          for (jj = A_i[i]; jj <= A_i[i + 1] - 1; jj += 1) {
            j = A_j[jj];
            if (j > ns && j < ne) 
              y_data[j] += A_data[jj] * x_data[i];
          }
        }
      }
       else {
        for (i = 0; i <= num_rows - 1; i += 1) {
          for (jv = 0; jv <= num_vectors - 1; jv += 1) {
            for (jj = A_i[i]; jj <= A_i[i + 1] - 1; jj += 1) {
              j = A_j[jj];
              if (j > ns && j < ne) 
                y_data[j * idxstride_y + jv * vecstride_y] += A_data[jj] * x_data[i * idxstride_x + jv * vecstride_x];
            }
          }
        }
      }
    }
  }
   else {
    for (i = 0; i <= num_rows - 1; i += 1) {
      if (num_vectors == 1) {
        for (jj = A_i[i]; jj <= A_i[i + 1] - 1; jj += 1) {
          j = A_j[jj];
          y_data[j] += A_data[jj] * x_data[i];
        }
      }
       else {
        for (jv = 0; jv <= num_vectors - 1; jv += 1) {
          for (jj = A_i[i]; jj <= A_i[i + 1] - 1; jj += 1) {
            j = A_j[jj];
            y_data[j * idxstride_y + jv * vecstride_y] += A_data[jj] * x_data[i * idxstride_x + jv * vecstride_x];
          }
        }
      }
    }
  }
/*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/
  if (alpha != 1.0) {
    
#pragma omp parallel for private (i) firstprivate (alpha,num_cols,num_vectors)
    for (i = 0; i <= num_cols * num_vectors - 1; i += 1) {
      y_data[i] *= alpha;
    }
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvec_FF
 *--------------------------------------------------------------------------*/

extern "C" int hypre_CSRMatrixMatvec_FF(double alpha,hypre_CSRMatrix *A,hypre_Vector *x,double beta,hypre_Vector *y,int *CF_marker_x,int *CF_marker_y,int fpt)
{
  double *A_data = A -> data;
  int *A_i = A -> i;
  int *A_j = A -> j;
  int num_rows = A -> num_rows;
  int num_cols = A -> num_cols;
  double *x_data = x -> data;
  double *y_data = y -> data;
  int x_size = x -> size;
  int y_size = y -> size;
  double temp;
  int i;
  int jj;
  int ierr = 0;
/*---------------------------------------------------------------------
    *  Check for size compatibility.  Matvec returns ierr = 1 if
    *  length of X doesn't equal the number of columns of A,
    *  ierr = 2 if the length of Y doesn't equal the number of rows
    *  of A, and ierr = 3 if both are true.
    *
    *  Because temporary vectors are often used in Matvec, none of
    *  these conditions terminates processing, and the ierr flag
    *  is informational only.
    *--------------------------------------------------------------------*/
  if (num_cols != x_size) 
    ierr = 1;
  if (num_rows != y_size) 
    ierr = 2;
  if (num_cols != x_size && num_rows != y_size) 
    ierr = 3;
/*-----------------------------------------------------------------------
    * Do (alpha == 0.0) computation - RDF: USE MACHINE EPS
    *-----------------------------------------------------------------------*/
  if (alpha == 0.0) {
    
#pragma omp parallel for private (i) firstprivate (fpt,num_rows)
    for (i = 0; i <= num_rows - 1; i += 1) {
      if (CF_marker_x[i] == fpt) 
        y_data[i] *= beta;
    }
    return ierr;
  }
/*-----------------------------------------------------------------------
    * y = (beta/alpha)*y
    *-----------------------------------------------------------------------*/
  temp = beta / alpha;
  if (temp != 1.0) {
    if (temp == 0.0) {
      
#pragma omp parallel for private (i)
      for (i = 0; i <= num_rows - 1; i += 1) {
        if (CF_marker_x[i] == fpt) 
          y_data[i] = 0.0;
      }
    }
     else {
      
#pragma omp parallel for private (i)
      for (i = 0; i <= num_rows - 1; i += 1) {
        if (CF_marker_x[i] == fpt) 
          y_data[i] *= temp;
      }
    }
  }
/*-----------------------------------------------------------------
    * y += A*x
    *-----------------------------------------------------------------*/
  
#pragma omp parallel for private (temp,i,jj) firstprivate (fpt)
  for (i = 0; i <= num_rows - 1; i += 1) {
    if (CF_marker_x[i] == fpt) {
      temp = y_data[i];
      
#pragma omp parallel for private (jj) reduction (+:temp)
      for (jj = A_i[i]; jj <= A_i[i + 1] - 1; jj += 1) {
        if (CF_marker_y[A_j[jj]] == fpt) 
          temp += A_data[jj] * x_data[A_j[jj]];
      }
      y_data[i] = temp;
    }
  }
/*-----------------------------------------------------------------
    * y = alpha*y
    *-----------------------------------------------------------------*/
  if (alpha != 1.0) {
    
#pragma omp parallel for private (i) firstprivate (fpt,num_rows)
    for (i = 0; i <= num_rows - 1; i += 1) {
      if (CF_marker_x[i] == fpt) 
        y_data[i] *= alpha;
    }
  }
  return ierr;
}
