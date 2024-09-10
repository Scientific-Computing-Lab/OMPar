// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "backprop.h"
#include <omp.h> 

double get_time()
{
  struct timeval t;
  gettimeofday(&t,0L);
  return t . tv_sec + t . tv_usec * 1e-6;
}
unsigned int num_threads = 0;
unsigned int num_blocks = 0;
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc,char **argv)
{
  setup(argc,argv);
  return 0;
}

int bpnn_train_kernel(BPNN *net,float *eo,float *eh)
{
  int in;
  int hid;
  int out;
  float out_err;
  float hid_err;
  in = net -> input_n;
  hid = net -> hidden_n;
  out = net -> output_n;
  float *input_weights_one_dim;
  float *input_weights_prev_one_dim;
  float *partial_sum;
  float sum;
  unsigned int num_blocks = (in / 16);
  input_weights_one_dim = ((float *)(malloc(((in + 1) * (hid + 1)) * sizeof(float ))));
  input_weights_prev_one_dim = ((float *)(malloc(((in + 1) * (hid + 1)) * sizeof(float ))));
  partial_sum = ((float *)(malloc((num_blocks * 16) * sizeof(float ))));
// this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
// todo: fix mem allocation
  int m = 0;
  for (int k = 0; k <= in; k += 1) {
    for (int j = 0; j <= hid; j += 1) {
      input_weights_one_dim[m] = net -> input_weights[k][j];
      input_weights_prev_one_dim[m] = net -> input_prev_weights[k][j];
      m++;
    }
  }
  printf("Performing device offload\n");
  double offload_start = get_time();
  float *input = net -> input_units;
  float *input_weights = input_weights_one_dim;
  float *input_prev_weights = input_weights_prev_one_dim;
  float *hidden_delta = net -> hidden_delta;
{
{
      float input_node[16];
      float weight_matrix[256];
{
        int by = omp_get_team_num();
        int tx = omp_get_thread_num() % 16;
        int ty = omp_get_thread_num() / 16;
        int index = (hid + 1) * 16 * by + (hid + 1) * ty + tx + 1 + (hid + 1);
        int index_in = 16 * by + ty + 1;
        if (tx == 0) 
          input_node[ty] = input[index_in];
        weight_matrix[ty * 16 + tx] = input_weights[index];
        weight_matrix[ty * 16 + tx] = weight_matrix[ty * 16 + tx] * input_node[ty];
        for (int i = 1; i <= 16; i = i * 2) {
          int power_two = i;
          if (ty % power_two == 0) 
            weight_matrix[ty * 16 + tx] = weight_matrix[ty * 16 + tx] + weight_matrix[(ty + power_two / 2) * 16 + tx];
        }
        input_weights[index] = weight_matrix[ty * 16 + tx];
        if (tx == 0) {
          partial_sum[by * hid + ty] = weight_matrix[tx * 16 + ty];
        }
      }
    }
    for (int j = 1; j <= hid; j += 1) {
      sum = 0.0;
      
#pragma omp parallel for reduction (+:sum) firstprivate (num_blocks)
      for (int k = 0; ((unsigned int )k) <= num_blocks - 1; k += 1) {
        sum += partial_sum[k * hid + j - 1];
      }
    #ifdef DEBUG
    #endif
      sum += net -> input_weights[0][j];
      net -> hidden_units[j] = ((float )(1.0 / (1.0 + (std::exp(-sum)))));
    }
    bpnn_layerforward(net -> hidden_units,net -> output_units,net -> hidden_weights,hid,out);
    bpnn_output_error(net -> output_delta,net -> target,net -> output_units,out,&out_err);
    bpnn_hidden_error(net -> hidden_delta,hid,net -> output_delta,out,net -> hidden_weights,net -> hidden_units,&hid_err);
    bpnn_adjust_weights(net -> output_delta,out,net -> hidden_units,hid,net -> hidden_weights,net -> hidden_prev_weights);
// input_weights has been written in the first kernel, so it needs to be restored.
{
{
        int by = omp_get_team_num();
        int tx = omp_get_thread_num() % 16;
        int ty = omp_get_thread_num() / 16;
        int index = (hid + 1) * 16 * by + (hid + 1) * ty + tx + 1 + (hid + 1);
        int index_y = 16 * by + ty + 1;
        int index_x = tx + 1;
        input_weights[index] += 0.3f * hidden_delta[index_x] * input[index_y] + 0.3f * input_prev_weights[index];
        input_prev_weights[index] = 0.3f * hidden_delta[index_x] * input[index_y] + 0.3f * input_prev_weights[index];
        if (ty == 0 && by == 0) {
          input_weights[index_x] += 0.3f * hidden_delta[index_x] + 0.3f * input_prev_weights[index_x];
          input_prev_weights[index_x] = 0.3f * hidden_delta[index_x] + 0.3f * input_prev_weights[index_x];
        }
      }
    }
  }
  double offload_end = get_time();
  printf("Device offloading time = %lf(s)\n",offload_end - offload_start);
#ifdef OUTPUT
#endif
  free(input_weights_prev_one_dim);
  free(partial_sum);
  free(input_weights_one_dim);
  return 0;
}
