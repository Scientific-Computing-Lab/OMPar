#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#define TILE_SIZE 5900
#define NTHREADS 256
// 1,2,3,4,5,6 -> 2,3,4,6,1,5
#include <omp.h> 
static const int d1 = 41;
static const int d2 = 13;
static const int d3 = 11;
static const int d4 = 9;
static const int d5 = 76;
static const int d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;
static int repeat = 1;
static const int shape_output[] = {(d2), (d3), (d1)};
static const int shape_input[] = {(d4), (d5), (d6)};
static const float shape_output_r[] = {((1.0 / d2)), ((1.0 / d3)), ((1.0 / d1))};
static const float shape_input_r[] = {((1.0 / d4)), ((1.0 / d5)), ((1.0 / d6))};
static const int stride_output_local[] = {(d1), (d1 * d2), (1)};
static const int stride_output_global[] = {(1), (d2), (d2 * d3 * d4 * d6)};
static const int stride_input[] = {(d2 * d3), (d2 * d3 * d4 * d6 * d1), (d2 * d3 * d4)};

void verify(double *input,double *output)
{
  int input_offset = 2 + d1 * (2 + d2 * (2 + d3 * (2 + d4 * (0 + 2 * d5))));
  int output_offset = 2 + d2 * (2 + d3 * (2 + d4 * (2 + d6 * (2 + 0 * d1))));
  bool error = false;
  for (size_t i = 0; i <= ((unsigned long )d5) - 1; i += 1) {
    if (input[input_offset + i * d1 * d2 * d3 * d4] != output[output_offset + i * d2 * d3 * d4 * d6 * d1]) {
      printf("FAIL\n");
      error = true;
      break; 
    }
  }
  if (!error) 
    printf("PASS\n");
}

int main(int argc,char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  repeat = atoi(argv[1]);
  double *input = new double [200514600];
  double *output = new double [200514600];
  
#pragma omp parallel for firstprivate (data_size)
  for (size_t i = 0; i <= ((unsigned long )data_size) - 1; i += 1) {
    input[i] = i;
  }
  const int nblocks = d4 * d5 * d6;
  const int tile_size = d1 * d2 * d3;
  const int dim_output = 3;
  const int dim_input = 3;
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (size_t i = 0; i <= ((unsigned long )repeat) - 1; i += 1) {{
        double tile[5900];
{
          for (int block_idx = omp_get_team_num(); block_idx <= nblocks - 1; block_idx += omp_get_num_teams()) {
            int it = block_idx;
            int im = 0;
            int offset1 = 0;
            for (int i = 0; i <= dim_input - 1; i += 1) {
              im = (it * shape_input_r[i]);
              offset1 += stride_input[i] * (it - im * shape_input[i]);
              it = im;
            }
            
#pragma omp parallel for
            for (int i = omp_get_thread_num(); i <= tile_size - 1; i += omp_get_num_threads()) {
              tile[i] = input[i + block_idx * tile_size];
            }
            for (int i = omp_get_thread_num(); i <= tile_size - 1; i += omp_get_num_threads()) {
              it = i;
              int offset2 = 0;
              int local_offset = 0;
              for (int j = 0; j <= dim_output - 1; j += 1) {
                im = (it * shape_output_r[j]);
                int tmp = it - im * shape_output[j];
                offset2 += stride_output_global[j] * tmp;
                local_offset += stride_output_local[j] * tmp;
                it = im;
              }
              output[offset1 + offset2] = tile[local_offset];
            }
          }
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (ms)\n",(time * 1e-6f / repeat));
  }
  verify(input,output);
  delete []input;
  delete []output;
  return 0;
}
