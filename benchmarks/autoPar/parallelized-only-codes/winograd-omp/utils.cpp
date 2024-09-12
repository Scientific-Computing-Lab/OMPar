#include "utils.h"
// F(2x2,3x3)
#include <omp.h> 

void WinogradConv2D_2x2_filter_transformation(float *transformed_filter)
{
  float filter[3][3];
  filter[0][0] = (+0.2);
  filter[1][0] = (+0.5);
  filter[2][0] = (- 0.8);
  filter[0][1] = (- 0.3);
  filter[1][1] = (+0.6);
  filter[2][1] = (- 0.9);
  filter[0][2] = (+0.4);
  filter[1][2] = (+0.7);
  filter[2][2] = (+0.10);
// filter transformation
  float tmp_filter[4][3];
// const float G[4][3] = {
//     {1.0f, 0.0f, 0.0f},
//     {0.5f, 0.5f, 0.5f},
//     {0.5f, -0.5f, 0.5f},
//     {0.0f, 0.0f, 1.0f}
// };
// G * g
  
#pragma omp parallel for
  for (int j = 0; j <= 2; j += 1) {
    tmp_filter[0][j] = filter[0][j];
    tmp_filter[1][j] = 0.5f * filter[0][j] + 0.5f * filter[1][j] + 0.5f * filter[2][j];
    tmp_filter[2][j] = 0.5f * filter[0][j] - 0.5f * filter[1][j] + 0.5f * filter[2][j];
    tmp_filter[3][j] = filter[2][j];
  }
// g * Gt
  
#pragma omp parallel for
  for (int i = 0; i <= 3; i += 1) {
    transformed_filter[i * 4 + 0] = tmp_filter[i][0];
    transformed_filter[i * 4 + 1] = 0.5f * tmp_filter[i][0] + 0.5f * tmp_filter[i][1] + 0.5f * tmp_filter[i][2];
    transformed_filter[i * 4 + 2] = 0.5f * tmp_filter[i][0] - 0.5f * tmp_filter[i][1] + 0.5f * tmp_filter[i][2];
    transformed_filter[i * 4 + 3] = tmp_filter[i][2];
  }
}

void WinogradConv2D_2x2_omp(float *input,float *output,float *transformed_filter,size_t *cpu_global_size)
{
// DATA_TYPE trasformed_filter[4][4];
// WinogradConv2D_2x2_filter_transformation(trasformed_filter);
  int out_map_size = 1024 - 2;
  int tile_n = (out_map_size + 1) / 2;
  for (int tile_i = 0; ((unsigned long )tile_i) <= cpu_global_size[0] - 1; tile_i += 1) {
    for (int tile_j = 0; tile_j <= tile_n - 1; tile_j += 1) {
// input transformation
      float input_tile[4][4];
      float tmp_tile[4][4];
      float transformed_tile[4][4];
      
#pragma omp parallel for private (j)
      for (int i = 0; i <= 3; i += 1) {
        
#pragma omp parallel for
        for (int j = 0; j <= 3; j += 1) {
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= 1024 || y >= 1024) {
            input_tile[i][j] = 0;
            continue; 
          }
          input_tile[i][j] = input[x * 1024 + y];
        }
      }
// const float Bt[4][4] = {
//     {1.0f, 0.0f, -1.0f, 0.0f},
//     {0.0f, 1.0f, 1.0f, 0.0f},
//     {0.0f, -1.0f, 1.0f, 0.0f},
//     {0.0f, 1.0f, 0.0f, -1.0f}
// }
// Bt * d
// #pragma omp simd
      
#pragma omp parallel for
      for (int j = 0; j <= 3; j += 1) {
        tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
        tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
        tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
        tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
      }
// d * B
// #pragma omp simd
      
#pragma omp parallel for
      for (int i = 0; i <= 3; i += 1) {
        transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
        transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
        transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
      }
// element-wise multiplication
      float multiplied_tile[4][4];
      
#pragma omp parallel for private (j_nom_4)
      for (int i = 0; i <= 3; i += 1) {
// #pragma omp simd
        
#pragma omp parallel for
        for (int j = 0; j <= 3; j += 1) {
          multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
        }
      }
// output transformation
      float tmp_tile_1[2][4];
      float final_tile[2][2];
// const float At[2][4] {
//     {1.0f, 1.0f, 1.0f, 0.0f},
//     {0.0f, 1.0f, -1.0f, -1.0f}
// }
// At * I
// #pragma omp simd
      
#pragma omp parallel for
      for (int j = 0; j <= 3; j += 1) {
        tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
        tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
      }
// I * A
// #pragma omp simd
      
#pragma omp parallel for
      for (int i = 0; i <= 1; i += 1) {
        final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
        final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
      }
      for (int i = 0; i <= 1; i += 1) {
        for (int j = 0; j <= 1; j += 1) {
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= out_map_size || y >= out_map_size) {
            continue; 
          }
          output[x * out_map_size + y] = final_tile[i][j];
        }
      }
    }
// for tile_i
  }
// for tile_j
}

bool compareResults(float *B,float *B_outputFromGpu)
{
  int i;
  int j;
  int fail;
  fail = 0;
// Compare a and b
  for (i = 0; i <= 1021; i += 1) {
    for (j = 0; j <= 1021; j += 1) {
      if ((percentDiff(B[i * (1024 - 2) + j],B_outputFromGpu[i * (1024 - 2) + j])) > 1.05) {
        fail++;
      }
    }
  }
// Print results
  #ifdef VERBOSE
  #endif
  return fail == 0?true : false;
}

void WinogradConv2D_2x2(float *input,float *output,float *transformed_filter)
{
  int out_map_size = 1024 - 2;
  int tile_n = (out_map_size + 1) / 2;
  for (int tile_i = 0; tile_i <= tile_n - 1; tile_i += 1) {
    for (int tile_j = 0; tile_j <= tile_n - 1; tile_j += 1) {
// input transformation
      float input_tile[4][4];
      float tmp_tile[4][4];
      float transformed_tile[4][4];
      
#pragma omp parallel for private (j)
      for (int i = 0; i <= 3; i += 1) {
        
#pragma omp parallel for
        for (int j = 0; j <= 3; j += 1) {
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= 1024 || y >= 1024) {
            input_tile[i][j] = 0;
            continue; 
          }
          input_tile[i][j] = input[x * 1024 + y];
        }
      }
// const float Bt[4][4] = {
//     {1.0f, 0.0f, -1.0f, 0.0f},
//     {0.0f, 1.0f, 1.0f, 0.0f},
//     {0.0f, -1.0f, 1.0f, 0.0f},
//     {0.0f, 1.0f, 0.0f, -1.0f}
// }
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
      
#pragma omp parallel for private (j_nom_12)
      for (int i = 0; i <= 3; i += 1) {
        
#pragma omp parallel for
        for (int j = 0; j <= 3; j += 1) {
          multiplied_tile[i][j] = transformed_tile[i][j] * transformed_filter[i * 4 + j];
        }
      }
// output transformation
      float tmp_tile_1[2][4];
      float final_tile[2][2];
// const float At[2][4] {
//     {1.0f, 1.0f, 1.0f, 0.0f},
//     {0.0f, 1.0f, -1.0f, -1.0f}
// }
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
          int x = 2 * tile_i + i;
          int y = 2 * tile_j + j;
          if (x >= out_map_size || y >= out_map_size) {
            continue; 
          }
          output[x * out_map_size + y] = final_tile[i][j];
        }
      }
    }
// for tile_i
  }
// for tile_j
}

double rtclock()
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp,(&Tzp));
  if (stat != 0) 
    printf("Error return from gettimeofday: %d",stat);
  return Tp . tv_sec + Tp . tv_usec * 1.0e-6;
}

float absVal(float a)
{
  if (a < 0) {
    return a * (- 1);
  }
   else {
    return a;
  }
}

float percentDiff(double val1,double val2)
{
  if ((absVal(val1)) < 0.01 && (absVal(val2)) < 0.01) {
    return 0.0f;
  }
   else {
    return 100.0f * absVal(absVal((val1 - val2)) / absVal((val1 + 0.00000001f)));
  }
}
