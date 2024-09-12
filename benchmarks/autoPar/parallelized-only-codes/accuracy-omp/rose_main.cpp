#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"
#define NUM_THREADS 256
#include <omp.h> 

int main(int argc,char *argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n",argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ndims = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const int data_size = nrows * ndims;
  const int label_size_bytes = (nrows * sizeof(int ));
  const size_t data_size_bytes = data_size * sizeof(float );
  int *label = (int *)(malloc(label_size_bytes));
  srand(123);
  for (int i = 0; i <= nrows - 1; i += 1) {
    label[i] = rand() % ndims;
  }
  float *data = (float *)(malloc(data_size_bytes));
  std::default_random_engine g(123);
  class std::uniform_real_distribution< float  > distr(0.f,1.f);
  for (int i = 0; i <= data_size - 1; i += 1) {
    data[i] = distr(g);
  }
  int count_ref = reference(nrows,ndims,top_k,data,label);
  int count[1];
{
    for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {
      printf("Grid size is %d\n",ngrid);
      auto start = std::chrono::_V2::steady_clock::now();
      for (int i = 0; i <= repeat - 1; i += 1) {
        count[0] = 0;
        for (int row = 0; row <= nrows - 1; row += 1) {
          const int label_data = label[row];
          const float label_pred = data[row * ndims + label_data];
          int ngt = 0;
          
#pragma omp parallel for reduction (+:ngt) firstprivate (label_pred)
          for (int col = 0; col <= ndims - 1; col += 1) {
            const float pred = data[row * ndims + col];
            if (pred > label_pred || pred == label_pred && col <= label_data) {
              ++ngt;
            }
          }
          if (ngt <= top_k) {
            ++count[0];
          }
        }
      }
      auto end = std::chrono::_V2::steady_clock::now();
      auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
      printf("Average execution time of accuracy kernel: %f (us)\n",(time * 1e-3f / repeat));
      bool ok = count[0] == count_ref;
      printf("%s\n",(ok?"PASS" : "FAIL"));
// printf("Accuracy = %f\n", (float)count / nrows);
    }
  }
  free(label);
  free(data);
  return 0;
}
