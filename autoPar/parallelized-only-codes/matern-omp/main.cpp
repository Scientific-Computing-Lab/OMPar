#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"
//
// Assumption 
// There are many more evaluation(target) points than sources for the subsequent code. 
// Each thread block will perform the evaluation for a small chunk of the target points and all source points. 
// 
#include <omp.h> 

void matern_kernel(const int ntargets,const float l,const float *sources,const float *targets,const float *weights,float *result)
{
  for (int t = 0; t <= ntargets - 1; t += 1) {
    float sum = 0.f;
    for (int s = 0; s <= 49; s += 1) {
      float squared_diff = 0.f;
      
#pragma omp parallel for reduction (+:squared_diff)
      for (int i = 0; i <= 2; i += 1) {
        squared_diff += (sources[s * 3 + i] - targets[t * 3 + i]) * (sources[s * 3 + i] - targets[t * 3 + i]);
      }
      float diff = sqrtf(squared_diff);
      sum += (1.f + sqrtf(5.f) * diff / l + 5.f * squared_diff / (3.f * l * l)) * expf(-sqrtf(5.f) * diff / l) * weights[s];
    }
    result[t] = sum;
  }
}

void matern_kernel2(const int ntargets,const float l,const float *sources,const float *targets,const float *weights,float *result)
{
  const int teams = (ntargets + 16 - 1) / 16;
// SY is a known value less than 64
{
    float local_result[800];
    float local_targets[48];
    float local_sources[150];
    float local_weights[50];
{
      int tx = omp_get_thread_num() % 16;
      int ty = omp_get_thread_num() / 16;
      int px = omp_get_team_num() * 16 + tx;
// range [0:ntargets)
      int py = ty;
// range [0:nsources)
      if (px < ntargets && py < 50) {
        if (ty == 0) {
          
#pragma omp parallel for
          for (int i = 0; i <= 2; i += 1) {
            local_targets[tx * 3 + i] = targets[px * 3 + i];
          }
        }
        if (tx == 0) {
          
#pragma omp parallel for
          for (int i = 0; i <= 2; i += 1) {
            local_sources[ty * 3 + i] = sources[py * 3 + i];
          }
          local_weights[ty] = weights[ty];
        }
      }
      if (px < ntargets && py < 50) {
        float squared_diff = 0.f;
        
#pragma omp parallel for reduction (+:squared_diff)
        for (int i = 0; i <= 2; i += 1) {
          squared_diff += (local_targets[tx * 3 + i] - local_sources[ty * 3 + i]) * (local_targets[tx * 3 + i] - local_sources[ty * 3 + i]);
        }
        float diff = sqrtf(squared_diff);
        local_result[tx * 50 + ty] = (1.f + sqrtf(5.f) * diff / l + 5.f * squared_diff / (3.f * l * l)) * expf(-sqrtf(5.f) * diff / l) * local_weights[ty];
      }
      if (px < ntargets && py < 50) {
        if (ty == 0) {
          float res = 0.f;
          
#pragma omp parallel for reduction (+:res) firstprivate (tx)
          for (int i = 0; i <= 49; i += 1) {
            res += local_result[tx * 50 + i];
          }
          result[px] = res;
        }
      }
    }
  }
}

int main(int argc,char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n",argv[0]);
    return 1;
  }
  const int npoints = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int source_size = 50 * 3;
// (x,y,z) coordinates in a 3D grid
  const int source_size_byte = (source_size * sizeof(float ));
  const int weight_size = 50;
  const int weight_size_byte = (weight_size * sizeof(float ));
  const int ntargets = npoints * npoints * npoints;
  const int target_size = ntargets * 3;
  const int target_size_byte = (target_size * sizeof(float ));
  const int result_size = ntargets;
  const int result_size_byte = (ntargets * sizeof(float ));
  float *sources = (float *)(malloc(source_size_byte));
  float *targets = (float *)(malloc(target_size_byte));
  float *weights = (float *)(malloc(weight_size_byte));
  float *result = (float *)(malloc(result_size_byte));
  float *result_ref = (float *)(malloc(result_size_byte));
  srand(123);
  for (int i = 0; i <= source_size - 1; i += 1) {
    sources[i] = (rand()) / ((float )2147483647);
  }
  for (int i = 0; i <= weight_size - 1; i += 1) {
    weights[i] = (rand()) / ((float )2147483647);
  }
  for (int i = 0; i <= target_size - 1; i += 1) {
    targets[i] = (rand()) / ((float )2147483647);
  }
{
    float l = 0.1f;
// length scale lower bound
// quickly verify the results using a small problem size
    const int ntargets_small = 16 * 16 * 16;
    printf("------------------------------------------------------------\n");
    printf("Verifying the kernel results with the problem size (16 cube)\n");
    printf("------------------------------------------------------------\n");
    while(l <= 1e5f){
      matern_kernel_reference(50,ntargets_small,l,sources,targets,weights,result_ref);
      matern_kernel2(ntargets_small,l,sources,targets,weights,result);
      bool ok = true;
      for (int i = 0; i <= ntargets_small - 1; i += 1) {
        if (fabsf(result[i] - result_ref[i]) > 1e-3f) {
          printf("@%d actual=%f expected=%f\n",i,result[i],result_ref[i]);
          ok = false;
          break; 
        }
      }
      printf("Length scale = %.1e check = %s\n",l,(ok?"PASS" : "FAIL"));
      l = l * 10.f;
    }
    printf("--------------------------------------------------------------------\n");
    printf("Timing the kernel execution with the problem size (%d cube)\n",npoints);
    printf("--------------------------------------------------------------------\n");
    l = 0.1f;
    while(l <= 1e5f){
      printf("Warmup..\n");
      for (int i = 0; i <= repeat - 1; i += 1) {
        matern_kernel2(ntargets,l,sources,targets,weights,result);
      }
      auto start = std::chrono::_V2::steady_clock::now();
      for (int i = 0; i <= repeat - 1; i += 1) {
        matern_kernel2(ntargets,l,sources,targets,weights,result);
      }
      auto end = std::chrono::_V2::steady_clock::now();
      auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
      printf("Length scale = %.1e ",l);
      printf("Average kernel execution time: %f (us)\n",(time * 1e-3f / repeat));
      l = l * 10.f;
    }
  }
  free(sources);
  free(weights);
  free(targets);
  free(result);
  free(result_ref);
  return 0;
}
