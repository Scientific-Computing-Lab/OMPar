/*BHEADER****************************************************************
 * (c) 2007   The Regents of the University of California               *
 *                                                                      *
 * See the file COPYRIGHT_and_DISCLAIMER for a complete copyright       *
 * notice and disclaimer.                                               *
 *                                                                      *
 *EHEADER****************************************************************/
//--------------
//  A micro kernel 
//--------------
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#else
#include <chrono>
#endif
#include "headers.h"
// CUDA/HIP block size or OpenCL work-group size
#define BLOCK_SIZE 256
// 
#include <omp.h> 
const int testIter = 500;
double totalWallTime = 0.0;
// 
void test_Matvec();
void test_Relax();
void test_Axpy();
//

int main(int argc,char *argv[])
{
#ifdef _OPENMP
#else
  printf("**** Warning: OpenMP is disabled ****\n");
#endif
  double del_wtime = 0.0;
#ifdef _OPENMP
#endif
  printf("\n");
  printf("//------------ \n");
  printf("// \n");
  printf("//  CORAL  AMGmk Benchmark Version 1.0 \n");
  printf("// \n");
  printf("//------------ \n");
  printf("\n testIter   = %d \n\n",testIter);
#ifdef _OPENMP
#endif
#ifdef _OPENMP
#else
  auto t0 = std::chrono::_V2::steady_clock::now();
#endif
// Matvec
  totalWallTime = 0.0;
  test_Matvec();
  printf("\n");
  printf("//------------ \n");
  printf("// \n");
  printf("//   MATVEC\n");
  printf("// \n");
  printf("//------------ \n");
  printf("\nWall time = %f seconds. \n",totalWallTime);
// Relax
  totalWallTime = 0.0;
  test_Relax();
  printf("\n");
  printf("//------------ \n");
  printf("// \n");
  printf("//   Relax\n");
  printf("// \n");
  printf("//------------ \n");
  printf("\nWall time = %f seconds. \n",totalWallTime);
// Axpy
  totalWallTime = 0.0;
  test_Axpy();
  printf("\n");
  printf("//------------ \n");
  printf("// \n");
  printf("//   Axpy\n");
  printf("// \n");
  printf("//------------ \n");
  printf("\nWall time = %f seconds. \n",totalWallTime);
#ifdef _OPENMP
#else
  auto t1 = std::chrono::_V2::steady_clock::now();
  struct std::chrono::duration< double  , class std::ratio< 1 , 1L >  > diff = (t1-t0);
  del_wtime = diff . count();
#endif
  printf("\nTotal Wall time = %f seconds. \n",del_wtime);
  return 0;
}

void test_Matvec()
{
#ifdef _OPENMP
#endif
  hypre_CSRMatrix *A;
  hypre_Vector *x;
  hypre_Vector *y;
  hypre_Vector *sol;
  int nx;
  int ny;
  int nz;
  int i;
  double *values;
  double *y_data;
  double *sol_data;
  double error;
  double diff;
  nx = 50;
/* size per proc nx*ny*nz */
  ny = 50;
  nz = 50;
  values = ((double *)(hypre_CAlloc(((unsigned int )4),((unsigned int )(sizeof(double ))))));
  values[0] = 6;
  values[1] = (- 1);
  values[2] = (- 1);
  values[3] = (- 1);
  A = GenerateSeqLaplacian(nx,ny,nz,values,&y,&x,&sol);
  hypre_SeqVectorSetConstantValues(x,1);
  hypre_SeqVectorSetConstantValues(y,0);
#ifdef _OPENMP
#else
  auto t0 = std::chrono::_V2::steady_clock::now();
#endif
  for (i = 0; i <= testIter - 1; i += 1) {
    hypre_CSRMatrixMatvec(1,A,x,0,y);
  }
#ifdef _OPENMP
#else
  auto t1 = std::chrono::_V2::steady_clock::now();
  struct std::chrono::duration< double  , class std::ratio< 1 , 1L >  > tdiff = (t1-t0);
  totalWallTime += tdiff . count();
#endif
  y_data = y -> data;
  sol_data = sol -> data;
  error = 0;
  for (i = 0; i <= nx * ny * nz - 1; i += 1) {
    diff = fabs(y_data[i] - sol_data[i]);
    if (diff > error) 
      error = diff;
  }
  if (error > 0) 
    printf(" \n Matvec: error: %e\n",error);
  (hypre_Free((char *)values) , values = 0L);
  hypre_CSRMatrixDestroy(A);
  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);
  hypre_SeqVectorDestroy(sol);
}

void test_Relax()
{
#ifdef _OPENMP
#endif
  hypre_CSRMatrix *A;
  hypre_Vector *x;
  hypre_Vector *y;
  hypre_Vector *sol;
  int nx;
  int ny;
  int nz;
  double *values;
  double diff;
  double error;
  nx = 50;
/* size per proc nx*ny*nz */
  ny = 50;
  nz = 50;
  values = ((double *)(hypre_CAlloc(((unsigned int )4),((unsigned int )(sizeof(double ))))));
  values[0] = 6;
  values[1] = (- 1);
  values[2] = (- 1);
  values[3] = (- 1);
  A = GenerateSeqLaplacian(nx,ny,nz,values,&y,&x,&sol);
  hypre_SeqVectorSetConstantValues(x,1);
  double *A_diag_data = A -> data;
  int *A_diag_i = A -> i;
  int *A_diag_j = A -> j;
  int n = A -> num_rows;
  int nonzero = A -> num_nonzeros;
  double *u_data = x -> data;
//int         u_data_size  = hypre_VectorSize(x);
  double *f_data = sol -> data;
//int         f_data_size  = hypre_VectorSize(sol);
  int grid_size = nx * ny * nz;
{
#ifdef _OPENMP
#else
    auto t0 = std::chrono::_V2::steady_clock::now();
#endif
    for (int ti = 0; ti <= testIter - 1; ti += 1) {
      for (int i = 0; i <= n - 1; i += 1) {
/*-----------------------------------------------------------
      * If diagonal is nonzero, relax point i; otherwise, skip it.
      *-----------------------------------------------------------*/
        if (A_diag_data[A_diag_i[i]] != 0.0) {
          double res = f_data[i];
          
#pragma omp parallel for reduction (-:res)
          for (int jj = A_diag_i[i] + 1; jj <= A_diag_i[i + 1] - 1; jj += 1) {
            int ii = A_diag_j[jj];
            res -= A_diag_data[jj] * u_data[ii];
          }
          u_data[i] = res / A_diag_data[A_diag_i[i]];
        }
      }
    }
#ifdef _OPENMP
#else
    auto t1 = std::chrono::_V2::steady_clock::now();
    struct std::chrono::duration< double  , class std::ratio< 1 , 1L >  > tdiff = (t1-t0);
    totalWallTime += tdiff . count();
#endif
  }
  error = 0;
  for (int i = 0; i <= nx * ny * nz - 1; i += 1) {
    diff = fabs(u_data[i] - 1);
    if (diff > error) 
      error = diff;
  }
  if (error > 0) 
    printf(" \n Relax: error: %e\n",error);
  (hypre_Free((char *)values) , values = 0L);
  hypre_CSRMatrixDestroy(A);
  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);
  hypre_SeqVectorDestroy(sol);
}

void test_Axpy()
{
#ifdef _OPENMP
#endif
  hypre_Vector *x;
  hypre_Vector *y;
  int nx;
  int i;
  double alpha = 0.5;
  double diff;
  double error;
  double *y_data;
  nx = 125000;
/* size per proc  */
  x = hypre_SeqVectorCreate(nx);
  y = hypre_SeqVectorCreate(nx);
  hypre_SeqVectorInitialize(x);
  hypre_SeqVectorInitialize(y);
  hypre_SeqVectorSetConstantValues(x,1);
  hypre_SeqVectorSetConstantValues(y,1);
#ifdef _OPENMP
#else
  auto t0 = std::chrono::_V2::steady_clock::now();
#endif
  for (i = 0; i <= testIter - 1; i += 1) {
    hypre_SeqVectorAxpy(alpha,x,y);
  }
#ifdef _OPENMP
#else
  auto t1 = std::chrono::_V2::steady_clock::now();
#endif
  y_data = y -> data;
  error = 0;
  for (i = 0; i <= nx - 1; i += 1) {
    diff = fabs(y_data[i] - 1 - 0.5 * ((double )testIter));
    if (diff > error) 
      error = diff;
  }
  if (error > 0) 
    printf(" \n Axpy: error: %e\n",error);
#ifdef _OPENMP
#else
  struct std::chrono::duration< double  , class std::ratio< 1 , 1L >  > tdiff = (t1-t0);
  totalWallTime += tdiff . count();
#endif
  hypre_SeqVectorDestroy(x);
  hypre_SeqVectorDestroy(y);
}
