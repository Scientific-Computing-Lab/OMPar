#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#define min(a,b) (a) < (b) ? (a) : (b)
#include <omp.h> 

inline float michalewicz(const float *xValues,const int dim)
{
  float result = 0;
  
#pragma omp parallel for reduction (+:result) firstprivate (dim)
  for (int i = 0; i <= dim - 1; i += 1) {
    float a = sinf(xValues[i]);
    float b = sinf((i + 1) * xValues[i] * xValues[i] / ((float )3.14159265358979323846));
    float c = powf(b,20);
// m = 10
    result += a * c;
  }
  return - 1.0f * result;
}
// https://www.sfu.ca/~ssurjano/michal.html

void Error(float value,int dim)
{
  printf("Global minima = %f\n",value);
  float trueMin = 0.0;
  if (dim == 2) 
    trueMin = (- 1.8013);
   else if (dim == 5) 
    trueMin = (- 4.687658);
   else if (dim == 10) 
    trueMin = (- 9.66015);
  printf("Error = %f\n",(fabsf(trueMin - value)));
}

int main(int argc,char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of vectors> <repeat>\n",argv[0]);
    return 1;
  }
  const size_t n = (atol(argv[1]));
  const int repeat = atoi(argv[2]);
// generate random numbers
  std::mt19937 gen(19937);
  class std::uniform_real_distribution< float  > dis(0.0,4.0);
// dimensions 
  const int dims[] = {(2), (5), (10)};
  for (int d = 0; d <= 2; d += 1) {
    const int dim = dims[d];
    const size_t size = n * dim;
    const size_t size_bytes = size * sizeof(float );
    float *values = (float *)(malloc(size_bytes));
    for (size_t i = 0; i <= size - 1; i += 1) {
      values[i] = dis(gen);
    }
    float minValue = 0;
{
      auto start = std::chrono::_V2::steady_clock::now();
      for (int i = 0; i <= repeat - 1; i += 1) {
        for (size_t j = 0; j <= n - 1; j += 1) {
          minValue = (minValue < michalewicz((values + j * dim),dim)?minValue : michalewicz((values + j * dim),dim));
        }
      }
      auto end = std::chrono::_V2::steady_clock::now();
      auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
      printf("Average execution time of kernel (dim = %d): %f (us)\n",dim,(time * 1e-3f / repeat));
    }
    Error(minValue,dim);
    free(values);
  }
  return 0;
}
