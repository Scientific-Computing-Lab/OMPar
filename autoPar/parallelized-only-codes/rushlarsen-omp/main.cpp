#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
#include "kernels.cpp"
#include <omp.h> 

int main(int argc,char *argv[])
{
  double t_start = 0;
  double dt = 0.02E-3;
  int num_timesteps = 1000000;
  int num_nodes = 1;
  if (argc > 1) {
    num_timesteps = atoi(argv[1]);
    printf("num_timesteps set to %d\n",num_timesteps);
    num_nodes = atoi(argv[2]);
    printf("num_nodes set to %d\n",num_nodes);
    if (num_timesteps <= 0 || num_nodes <= 0) 
      exit(1);
  }
  unsigned int num_states = NUM_STATES;
  size_t total_num_states = (num_nodes * num_states);
  size_t states_size = total_num_states * sizeof(double );
  double *states = (double *)(malloc(states_size));
  init_state_values(states,num_nodes);
  double *states2 = (double *)(malloc(states_size));
  memcpy(states2,states,states_size);
  unsigned int num_parameters = NUM_PARAMS;
  size_t total_num_parameters = (num_nodes * num_parameters);
  size_t parameters_size = total_num_parameters * sizeof(double );
  double *parameters = (double *)(malloc(parameters_size));
  init_parameters_values(parameters,num_nodes);
  double t = t_start;
  struct timespec timestamp_start;
  struct timespec timestamp_now;
  double time_elapsed;
  printf("Host: Rush Larsen (exp integrator on all gates)\n");
  for (int it = 0; it <= num_timesteps - 1; it += 1) {
    forward_rush_larsen(states,t,dt,parameters,num_nodes);
    t += dt;
  }
  printf("Device: Rush Larsen (exp integrator on all gates)\n");
{
// All nodes run the same kernel
    t = t_start;
    clock_gettime(4,&timestamp_start);
    for (int it = 0; it <= num_timesteps - 1; it += 1) {
      k_forward_rush_larsen(states2,t,dt,parameters,num_nodes);
      t += dt;
    }
    clock_gettime(4,&timestamp_now);
    time_elapsed = (timestamp_now . tv_sec - timestamp_start . tv_sec) + 1E-9 * (timestamp_now . tv_nsec - timestamp_start . tv_nsec);
    printf("Device: computed %d time steps in %g s. Time steps per second: %g\n\n",num_timesteps,time_elapsed,num_timesteps / time_elapsed);
  }
  double rmse = 0.0;
  
#pragma omp parallel for reduction (+:rmse)
  for (size_t i = 0; i <= total_num_states - 1; i += 1) {
    rmse += (states2[i] - states[i]) * (states2[i] - states[i]);
#ifdef VERBOSE
#endif
  }
  printf("RMSE = %lf\n",(sqrt(rmse / total_num_states)));
  free(states);
  free(states2);
  free(parameters);
  return 0;
}
