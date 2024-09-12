#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"
// limits of integration
#define A 0
#define B 15
// row size is related to accuracy
#define ROW_SIZE 17
#define EPS      1e-7
#include <omp.h> 

inline double f(double x)
{
  return exp(x) * sin(x);
}

inline unsigned int getFirstSetBitPos(int n)
{
  return (std::log2((float )(n & -n)) + 1);
}

int main(int argc,char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of work-groups> ",argv[0]);
    printf("<work-group size> <repeat>\n");
    return 1;
  }
  const int nwg = atoi(argv[1]);
  const int wgs = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  double *result = (double *)(malloc(sizeof(double ) * nwg));
  double d_sum;
  double a = 0;
  double b = 15;
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {{
        double smem[1088];
{
          int threadIdx_x = omp_get_thread_num();
          int blockIdx_x = omp_get_team_num();
          int gridDim_x = omp_get_num_teams();
          int blockDim_x = omp_get_num_threads();
          double diff = (b - a) / gridDim_x;
          double step;
          int k;
          int max_eval = 1 << 17 - 1;
          b = a + (blockIdx_x + 1) * diff;
          a += blockIdx_x * diff;
          step = (b - a) / max_eval;
          double local_col[17];
// specific to the row size
          
#pragma omp parallel for
          for (int i = 0; i <= 16; i += 1) {
            local_col[i] = 0.0;
          }
          if (!threadIdx_x) {
            k = blockDim_x;
            local_col[0] = f(a) + f(b);
          }
           else 
            k = threadIdx_x;
          for (; k <= max_eval - 1; k += blockDim_x) {
            local_col[17 - getFirstSetBitPos(k)] += 2.0 * f(a + step * k);
          }
          
#pragma omp parallel for
          for (int i = 0; i <= 16; i += 1) {
            smem[17 * threadIdx_x + i] = local_col[i];
          }
          if (threadIdx_x < 17) {
            double sum = 0.0;
            
#pragma omp parallel for reduction (+:sum) firstprivate (blockDim_x)
            for (int i = threadIdx_x; i <= blockDim_x * 17 - 1; i += 17) {
              sum += smem[i];
            }
            smem[threadIdx_x] = sum;
          }
          if (!threadIdx_x) {
            double *table = local_col;
            table[0] = smem[0];
            for (int k = 1; k <= 16; k += 1) {
              table[k] = table[k - 1] + smem[k];
            }
            for (int k = 0; k <= 16; k += 1) {
              table[k] *= (b - a) / (1 << k + 1);
            }
            for (int col = 0; col <= 15; col += 1) {
              for (int row = 17 - 1; row >= col + 1; row += -1) {
                table[row] = table[row] + (table[row] - table[row - 1]) / ((1 << 2 * col + 1) - 1);
              }
            }
            result[blockIdx_x] = table[17 - 1];
          }
        }
      }
      d_sum = 0.0;
      
#pragma omp parallel for reduction (+:d_sum)
      for (int k = 0; k <= nwg - 1; k += 1) {
        d_sum += result[k];
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
  }
// verify
  double ref_sum = reference(f,0,15,17,1e-7);
  printf("%s\n",(fabs(d_sum - ref_sum) > 1e-7?"FAIL" : "PASS"));
  free(result);
  return 0;
}
