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
 * $Revision: 2.12 $
 ***********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/
#include "headers.h"
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCreate
 *--------------------------------------------------------------------------*/
#include <omp.h> 

extern "C" hypre_CSRMatrix *hypre_CSRMatrixCreate(int num_rows,int num_cols,int num_nonzeros)
{
  hypre_CSRMatrix *matrix;
  matrix = ((hypre_CSRMatrix *)(hypre_CAlloc(((unsigned int )1),((unsigned int )(sizeof(hypre_CSRMatrix ))))));
  matrix -> data = 0L;
  matrix -> i = 0L;
  matrix -> j = 0L;
  matrix -> rownnz = 0L;
  matrix -> num_rows = num_rows;
  matrix -> num_cols = num_cols;
  matrix -> num_nonzeros = num_nonzeros;
/* set defaults */
  matrix -> owns_data = 1;
  matrix -> num_rownnz = num_rows;
  return matrix;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixDestroy
 *--------------------------------------------------------------------------*/

extern "C" int hypre_CSRMatrixDestroy(hypre_CSRMatrix *matrix)
{
  int ierr = 0;
  if (matrix) {
    (hypre_Free((char *)(matrix -> i)) , matrix -> i = 0L);
    if ((matrix -> rownnz)) 
      (hypre_Free((char *)(matrix -> rownnz)) , matrix -> rownnz = 0L);
    if ((matrix -> owns_data)) {
      (hypre_Free((char *)(matrix -> data)) , matrix -> data = 0L);
      (hypre_Free((char *)(matrix -> j)) , matrix -> j = 0L);
    }
    (hypre_Free((char *)matrix) , matrix = 0L);
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixInitialize
 *--------------------------------------------------------------------------*/

extern "C" int hypre_CSRMatrixInitialize(hypre_CSRMatrix *matrix)
{
  int num_rows = matrix -> num_rows;
  int num_nonzeros = matrix -> num_nonzeros;
/*   int  num_rownnz = hypre_CSRMatrixNumRownnz(matrix); */
  int ierr = 0;
  if (!(matrix -> data) && num_nonzeros) 
    matrix -> data = ((double *)(hypre_CAlloc(((unsigned int )num_nonzeros),((unsigned int )(sizeof(double ))))));
  if (!(matrix -> i)) 
    matrix -> i = ((int *)(hypre_CAlloc(((unsigned int )(num_rows + 1)),((unsigned int )(sizeof(int ))))));
/*   if ( ! hypre_CSRMatrixRownnz(matrix) )
      hypre_CSRMatrixRownnz(matrix)    = hypre_CTAlloc(int, num_rownnz);*/
  if (!(matrix -> j) && num_nonzeros) 
    matrix -> j = ((int *)(hypre_CAlloc(((unsigned int )num_nonzeros),((unsigned int )(sizeof(int ))))));
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetDataOwner
 *--------------------------------------------------------------------------*/

extern "C" int hypre_CSRMatrixSetDataOwner(hypre_CSRMatrix *matrix,int owns_data)
{
  int ierr = 0;
  matrix -> owns_data = owns_data;
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixSetRownnz
 *
 * function to set the substructure rownnz and num_rowsnnz inside the CSRMatrix
 * it needs the A_i substructure of CSRMatrix to find the nonzero rows.
 * It runs after the create CSR and when A_i is known..It does not check for
 * the existence of A_i or of the CSR matrix.
 *--------------------------------------------------------------------------*/

extern "C" int hypre_CSRMatrixSetRownnz(hypre_CSRMatrix *matrix)
{
  int ierr = 0;
  int num_rows = matrix -> num_rows;
  int *A_i = matrix -> i;
  int *Arownnz;
  int i;
  int adiag;
  int irownnz = 0;
  
#pragma omp parallel for private (adiag,i) reduction (+:irownnz)
  for (i = 0; i <= num_rows - 1; i += 1) {
    adiag = A_i[i + 1] - A_i[i];
    if (adiag > 0) 
      irownnz++;
  }
  matrix -> num_rownnz = irownnz;
  if (irownnz == 0 || irownnz == num_rows) {
    matrix -> rownnz = 0L;
  }
   else {
    Arownnz = ((int *)(hypre_CAlloc(((unsigned int )irownnz),((unsigned int )(sizeof(int ))))));
    irownnz = 0;
    for (i = 0; i <= num_rows - 1; i += 1) {
      adiag = A_i[i + 1] - A_i[i];
      if (adiag > 0) 
        Arownnz[irownnz++] = i;
    }
    matrix -> rownnz = Arownnz;
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixRead
 *--------------------------------------------------------------------------*/

extern "C" hypre_CSRMatrix *hypre_CSRMatrixRead(char *file_name)
{
  hypre_CSRMatrix *matrix;
  FILE *fp;
  double *matrix_data;
  int *matrix_i;
  int *matrix_j;
  int num_rows;
  int num_nonzeros;
  int max_col = 0;
  int file_base = 1;
  int j;
/*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/
  fp = fopen(file_name,"r");
  fscanf(fp,"%d",&num_rows);
  matrix_i = ((int *)(hypre_CAlloc(((unsigned int )(num_rows + 1)),((unsigned int )(sizeof(int ))))));
  for (j = 0; j <= num_rows + 1 - 1; j += 1) {
    fscanf(fp,"%d",&matrix_i[j]);
    matrix_i[j] -= file_base;
  }
  num_nonzeros = matrix_i[num_rows];
  matrix = hypre_CSRMatrixCreate(num_rows,num_rows,matrix_i[num_rows]);
  matrix -> i = matrix_i;
  hypre_CSRMatrixInitialize(matrix);
  matrix_j = matrix -> j;
  for (j = 0; j <= num_nonzeros - 1; j += 1) {
    fscanf(fp,"%d",&matrix_j[j]);
    matrix_j[j] -= file_base;
    if (matrix_j[j] > max_col) {
      max_col = matrix_j[j];
    }
  }
  matrix_data = matrix -> data;
  for (j = 0; j <= matrix_i[num_rows] - 1; j += 1) {
    fscanf(fp,"%le",&matrix_data[j]);
  }
  fclose(fp);
  matrix -> num_nonzeros = num_nonzeros;
  matrix -> num_cols = ++max_col;
  return matrix;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixPrint
 *--------------------------------------------------------------------------*/

extern "C" int hypre_CSRMatrixPrint(hypre_CSRMatrix *matrix,char *file_name)
{
  FILE *fp;
  double *matrix_data;
  int *matrix_i;
  int *matrix_j;
  int num_rows;
  int file_base = 1;
  int j;
  int ierr = 0;
/*----------------------------------------------------------
    * Print the matrix data
    *----------------------------------------------------------*/
  matrix_data = matrix -> data;
  matrix_i = matrix -> i;
  matrix_j = matrix -> j;
  num_rows = matrix -> num_rows;
  fp = fopen(file_name,"w");
  fprintf(fp,"%d\n",num_rows);
  for (j = 0; j <= num_rows; j += 1) {
    fprintf(fp,"%d\n",matrix_i[j] + file_base);
  }
  for (j = 0; j <= matrix_i[num_rows] - 1; j += 1) {
    fprintf(fp,"%d\n",matrix_j[j] + file_base);
  }
  if (matrix_data) {
    for (j = 0; j <= matrix_i[num_rows] - 1; j += 1) {
      fprintf(fp,"%.14e\n",matrix_data[j]);
    }
  }
   else {
    fprintf(fp,"Warning: No matrix data!\n");
  }
  fclose(fp);
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixCopy:
 * copys A to B, 
 * if copy_data = 0 only the structure of A is copied to B.
 * the routine does not check if the dimensions of A and B match !!! 
 *--------------------------------------------------------------------------*/

extern "C" int hypre_CSRMatrixCopy(hypre_CSRMatrix *A,hypre_CSRMatrix *B,int copy_data)
{
  int ierr = 0;
  int num_rows = A -> num_rows;
  int *A_i = A -> i;
  int *A_j = A -> j;
  double *A_data;
  int *B_i = B -> i;
  int *B_j = B -> j;
  double *B_data;
  int i;
  int j;
  for (i = 0; i <= num_rows - 1; i += 1) {
    B_i[i] = A_i[i];
    
#pragma omp parallel for private (j)
    for (j = A_i[i]; j <= A_i[i + 1] - 1; j += 1) {
      B_j[j] = A_j[j];
    }
  }
  B_i[num_rows] = A_i[num_rows];
  if (copy_data) {
    A_data = A -> data;
    B_data = B -> data;
    for (i = 0; i <= num_rows - 1; i += 1) {
      
#pragma omp parallel for private (j)
      for (j = A_i[i]; j <= A_i[i + 1] - 1; j += 1) {
        B_data[j] = A_data[j];
      }
    }
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixClone
 * Creates and returns a new copy of the argument, A.
 * Data is not copied, only structural information is reproduced.
 * Copying is a deep copy in that no pointers are copied; new arrays are
 * created where necessary.
 *--------------------------------------------------------------------------*/

extern "C" hypre_CSRMatrix *hypre_CSRMatrixClone(hypre_CSRMatrix *A)
{
  int num_rows = A -> num_rows;
  int num_cols = A -> num_cols;
  int num_nonzeros = A -> num_nonzeros;
  hypre_CSRMatrix *B = hypre_CSRMatrixCreate(num_rows,num_cols,num_nonzeros);
  int *A_i;
  int *A_j;
  int *B_i;
  int *B_j;
  int i;
  int j;
  hypre_CSRMatrixInitialize(B);
  A_i = A -> i;
  A_j = A -> j;
  B_i = B -> i;
  B_j = B -> j;
  
#pragma omp parallel for private (i) firstprivate (num_rows)
  for (i = 0; i <= num_rows + 1 - 1; i += 1) {
    B_i[i] = A_i[i];
  }
  
#pragma omp parallel for private (j) firstprivate (num_nonzeros)
  for (j = 0; j <= num_nonzeros - 1; j += 1) {
    B_j[j] = A_j[j];
  }
  B -> num_rownnz = A -> num_rownnz;
  if ((A -> rownnz)) 
    hypre_CSRMatrixSetRownnz(B);
  return B;
}
/*--------------------------------------------------------------------------
 * hypre_CSRMatrixUnion
 * Creates and returns a matrix whose elements are the union of those of A and B.
 * Data is not computed, only structural information is created.
 * A and B must have the same numbers of rows.
 * Nothing is done about Rownnz.
 *
 * If col_map_offd_A and col_map_offd_B are zero, A and B are expected to have
 * the same column indexing.  Otherwise, col_map_offd_A, col_map_offd_B should
 * be the arrays of that name from two ParCSRMatrices of which A and B are the
 * offd blocks.
 *
 * The algorithm can be expected to have reasonable efficiency only for very
 * sparse matrices (many rows, few nonzeros per row).
 * The nonzeros of a computed row are NOT necessarily in any particular order.
 *--------------------------------------------------------------------------*/

extern "C" hypre_CSRMatrix *hypre_CSRMatrixUnion(hypre_CSRMatrix *A,hypre_CSRMatrix *B,int *col_map_offd_A,int *col_map_offd_B,int **col_map_offd_C)
{
  int num_rows = A -> num_rows;
  int num_cols_A = A -> num_cols;
  int num_cols_B = B -> num_cols;
  int num_cols;
  int num_nonzeros;
  int *A_i = A -> i;
  int *A_j = A -> j;
  int *B_i = B -> i;
  int *B_j = B -> j;
  int *C_i;
  int *C_j;
  int *jC = 0L;
  int i;
  int jA;
  int jB;
  int jBg;
  int ma;
  int mb;
  int mc;
  int ma_min;
  int ma_max;
  int match;
  hypre_CSRMatrix *C;
  if (!(num_rows == B -> num_rows)) {
    fprintf(stderr,"hypre_assert failed: %s\n","num_rows == hypre_CSRMatrixNumRows(B)");
    hypre_error_handler("csr_matrix.cpp",423,1);
  }
  ;
  if (col_map_offd_B) 
    if (!col_map_offd_A) {
      fprintf(stderr,"hypre_assert failed: %s\n","col_map_offd_A");
      hypre_error_handler("csr_matrix.cpp",424,1);
    }
  ;
  if (col_map_offd_A) 
    if (!col_map_offd_B) {
      fprintf(stderr,"hypre_assert failed: %s\n","col_map_offd_B");
      hypre_error_handler("csr_matrix.cpp",425,1);
    }
  ;
/* ==== First, go through the columns of A and B to count the columns of C. */
  if (col_map_offd_A == 0) {
/* The matrices are diagonal blocks.
         Normally num_cols_A==num_cols_B, col_starts is the same, etc.
      */
    num_cols = (num_cols_A < num_cols_B?num_cols_B : num_cols_A);
  }
   else {
/* The matrices are offdiagonal blocks. */
    jC = ((int *)(hypre_CAlloc(((unsigned int )num_cols_B),((unsigned int )(sizeof(int ))))));
    num_cols = num_cols_A;
/* initialization; we'll compute the actual value */
    for (jB = 0; jB <= num_cols_B - 1; jB += 1) {
      match = 0;
      jBg = col_map_offd_B[jB];
      
#pragma omp parallel for private (ma) firstprivate (jBg)
      for (ma = 0; ma <= num_cols_A - 1; ma += 1) {
        if (col_map_offd_A[ma] == jBg) 
          match = 1;
      }
      if (match == 0) {
        jC[jB] = num_cols;
        ++num_cols;
      }
    }
  }
/* ==== If we're working on a ParCSRMatrix's offd block,
      make and load col_map_offd_C */
  if (col_map_offd_A) {
     *col_map_offd_C = ((int *)(hypre_CAlloc(((unsigned int )num_cols),((unsigned int )(sizeof(int ))))));
    
#pragma omp parallel for private (jA)
    for (jA = 0; jA <= num_cols_A - 1; jA += 1) {
      ( *col_map_offd_C)[jA] = col_map_offd_A[jA];
    }
    for (jB = 0; jB <= num_cols_B - 1; jB += 1) {
      match = 0;
      jBg = col_map_offd_B[jB];
      
#pragma omp parallel for private (ma)
      for (ma = 0; ma <= num_cols_A - 1; ma += 1) {
        if (col_map_offd_A[ma] == jBg) 
          match = 1;
      }
      if (match == 0) 
        ( *col_map_offd_C)[jC[jB]] = jBg;
    }
  }
/* ==== The first run through A and B is to count the number of nonzero elements,
      without double-counting duplicates.  Then we can create C. */
  num_nonzeros = A -> num_nonzeros;
  
#pragma omp parallel for private (jA,jB,ma_min,ma_max,match,i,mb) reduction (+:num_nonzeros)
  for (i = 0; i <= num_rows - 1; i += 1) {
    ma_min = A_i[i];
    ma_max = A_i[i + 1];
    
#pragma omp parallel for private (jA,jB,match,mb) reduction (+:num_nonzeros) firstprivate (ma_max)
    for (mb = B_i[i]; mb <= B_i[i + 1] - 1; mb += 1) {
      jB = B_j[mb];
      if (col_map_offd_B) 
        jB = col_map_offd_B[jB];
      match = 0;
      for (ma = ma_min; ma <= ma_max - 1; ma += 1) {
        jA = A_j[ma];
        if (col_map_offd_A) 
          jA = col_map_offd_A[jA];
        if (jB == jA) {
          match = 1;
          if (ma == ma_min) 
            ++ma_min;
          break; 
        }
      }
      if (match == 0) 
        ++num_nonzeros;
    }
  }
  C = hypre_CSRMatrixCreate(num_rows,num_cols,num_nonzeros);
  hypre_CSRMatrixInitialize(C);
/* ==== The second run through A and B is to pick out the column numbers
      for each row, and put them in C. */
  C_i = C -> i;
  C_i[0] = 0;
  C_j = C -> j;
  mc = 0;
  for (i = 0; i <= num_rows - 1; i += 1) {
    ma_min = A_i[i];
    ma_max = A_i[i + 1];
    for (ma = ma_min; ma <= ma_max - 1; ma += 1) {
      C_j[mc] = A_j[ma];
      ++mc;
    }
    for (mb = B_i[i]; mb <= B_i[i + 1] - 1; mb += 1) {
      jB = B_j[mb];
      if (col_map_offd_B) 
        jB = col_map_offd_B[jB];
      match = 0;
      for (ma = ma_min; ma <= ma_max - 1; ma += 1) {
        jA = A_j[ma];
        if (col_map_offd_A) 
          jA = col_map_offd_A[jA];
        if (jB == jA) {
          match = 1;
          if (ma == ma_min) 
            ++ma_min;
          break; 
        }
      }
      if (match == 0) {
        C_j[mc] = jC[B_j[mb]];
        ++mc;
      }
    }
    C_i[i + 1] = mc;
  }
  if (!(mc == num_nonzeros)) {
    fprintf(stderr,"hypre_assert failed: %s\n","mc == num_nonzeros");
    hypre_error_handler("csr_matrix.cpp",547,1);
  }
  ;
  if (jC) 
    (hypre_Free((char *)jC) , jC = 0L);
  return C;
}
