#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "mv.h"
// dense matrix vector multiply
#include <omp.h> 

long mv_dense_parallel(const int repeat,const int bs,const int num_rows,const float *x,float *matrix,float *y)
{
  long time;
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      for (int i = 0; i <= num_rows - 1; i += 1) {
        float temp = 0;
        
#pragma omp parallel for reduction (+:temp)
        for (int j = 0; j <= num_rows - 1; j += 1) {
          if (matrix[i * num_rows + j] != ((float )0)) 
            temp += matrix[i * num_rows + j] * x[j];
        }
        y[i] = temp;
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
  }
  return time;
}
// sparse matrix vector multiply using the CSR format

long mv_csr_parallel(const int repeat,const int bs,const int num_rows,const float *x,const size_t nnz,float *matrix,float *y)
{
  size_t *row_indices = (size_t *)(malloc((num_rows + 1) * sizeof(size_t )));
  int *col_indices = (int *)(malloc(nnz * sizeof(int )));
  float *values = (float *)(malloc(nnz * sizeof(float )));
// initialize csr structure
  init_csr(row_indices,values,col_indices,matrix,num_rows,nnz);
  long time;
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      for (int i = 0; i <= num_rows - 1; i += 1) {
        size_t row_start = row_indices[i];
        size_t row_end = row_indices[i + 1];
        float temp = 0;
        
#pragma omp parallel for reduction (+:temp) firstprivate (row_end)
        for (size_t j = row_start; j <= row_end - 1; j += 1) {
          temp += values[j] * x[col_indices[j]];
        }
        y[i] = temp;
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
  }
  free(values);
  free(row_indices);
  free(col_indices);
  return time;
}
