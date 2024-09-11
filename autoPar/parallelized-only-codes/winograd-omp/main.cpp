#include <chrono>
#include <omp.h>
#include "utils.h"
#include <omp.h> 

int main(int argc,char *argv[])
{
  double start = rtclock();
  float *A = (float *)(malloc((1024 * 1024) * sizeof(float )));
  float *B_host = (float *)(malloc(((1024 - 2) * (1024 - 2)) * sizeof(float )));
  float *B = (float *)(malloc(((1024 - 2) * (1024 - 2)) * sizeof(float )));
  float *C = (float *)(malloc((4 * 4) * sizeof(float )));
  for (int i = 0; i <= 1023; i += 1) {
    for (int j = 0; j <= 1023; j += 1) {
      A[i * 1024 + j] = (rand()) / ((float )2147483647);
    }
  }
// transformed filter
  WinogradConv2D_2x2_filter_transformation(C);
  const int tile_n = (1024 - 2 + 1) / 2;
// initial problem size
  size_t globalWorkSize[2] = {((size_t )(std::ceil(((float )tile_n) / ((float )32)))) * 32, ((size_t )(std::ceil(((float )tile_n) / ((float )8)))) * 8};
  size_t localWorkSize[2] = {(32), (8)};
// adjust problem size for co-run
  size_t cpu_global_size[2];
  size_t gpu_global_size[2];
  size_t global_offset[2];
  bool pass = true;
  double co_time = 0.0;
{
// sweep over cpu workload size
    for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset += 1) {
      cpu_global_size[0] = cpu_offset * ((size_t )(std::ceil(((float )tile_n) / ((float )32)))) / 100 * 32;
      cpu_global_size[1] = globalWorkSize[1];
      gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
      gpu_global_size[1] = globalWorkSize[1];
      global_offset[0] = cpu_global_size[0];
      global_offset[1] = 0;
      const int tile_i_size = gpu_global_size[0];
      const int tile_j_size = gpu_global_size[1];
      const int offset_i = global_offset[0];
      const int offset_j = global_offset[1];
      const int thread_size = (localWorkSize[1] * localWorkSize[0]);
      bool cpu_run = false;
      bool gpu_run = false;
      if (cpu_global_size[0] > 0) {
        cpu_run = true;
      }
      if (gpu_global_size[0] > 0) {
        gpu_run = true;
      }
// co-execution of host and device
      double co_start = rtclock();
      if (gpu_run) {
        for (int tile_j = 0; tile_j <= tile_j_size - 1; tile_j += 1) {
          for (int tile_i = 0; tile_i <= tile_i_size - 1; tile_i += 1) {
// input transformation
            float input_tile[4][4];
            float tmp_tile[4][4];
            float transformed_tile[4][4];
            
#pragma omp parallel for private (j_nom_2)
            for (int i = 0; i <= 3; i += 1) {
              
#pragma omp parallel for firstprivate (offset_i,offset_j)
              for (int j = 0; j <= 3; j += 1) {
                int x = 2 * (tile_i + offset_i) + i;
                int y = 2 * (tile_j + offset_j) + j;
                if (x >= 1024 || y >= 1024) {
                  input_tile[i][j] = 0;
                  continue; 
                }
                input_tile[i][j] = A[x * 1024 + y];
              }
            }
// Bt * d
            
#pragma omp parallel for
            for (int j = 0; j <= 3; j += 1) {
              tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
              tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
              tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
              tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
            }
// d * B
            
#pragma omp parallel for
            for (int i = 0; i <= 3; i += 1) {
              transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
              transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
              transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
              transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
            }
// element-wise multiplication
            float multiplied_tile[4][4];
            
#pragma omp parallel for private (j_nom_6)
            for (int i = 0; i <= 3; i += 1) {
              
#pragma omp parallel for
              for (int j = 0; j <= 3; j += 1) {
                multiplied_tile[i][j] = transformed_tile[i][j] * C[i * 4 + j];
              }
            }
// output transformation
            float tmp_tile_1[2][4];
            float final_tile[2][2];
// At * I
            
#pragma omp parallel for
            for (int j = 0; j <= 3; j += 1) {
              tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
              tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
            }
// I * A
            
#pragma omp parallel for
            for (int i = 0; i <= 1; i += 1) {
              final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
              final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
            }
            for (int i = 0; i <= 1; i += 1) {
              for (int j = 0; j <= 1; j += 1) {
                int x = 2 * (tile_i + offset_i) + i;
                int y = 2 * (tile_j + offset_j) + j;
                if (x >= 1024 - 2 || y >= 1024 - 2) {
                  continue; 
                }
                B[x * (1024 - 2) + y] = final_tile[i][j];
              }
            }
          }
        }
      }
      if (cpu_run) {
        WinogradConv2D_2x2_omp(A,B,C,cpu_global_size);
        if (gpu_run) {
        }
         else {
        }
      }
      co_time += rtclock() - co_start;
#ifdef VERBOSE
#endif
      WinogradConv2D_2x2(A,B_host,C);
      pass &= (compareResults(B_host,B));
// sweep
    }
  }
// #pragma
  printf("%s\n",(pass?"PASS" : "FAIL"));
  free(A);
  free(B);
  free(B_host);
  free(C);
  double end = rtclock();
  printf("Co-execution time: %lf s\n",co_time);
  printf("Total time: %lf s\n",end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",100.0 * co_time / (end - start));
  return 0;
}
