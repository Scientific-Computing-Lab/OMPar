#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#define idx(i,j)   (i)*y_points+(j)
#include <omp.h> 

int main(int argc,char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <dim_x> <dim_y> <nt>\n",argv[0]);
    printf("dim_x: number of grid points in the x axis\n");
    printf("dim_y: number of grid points in the y axis\n");
    printf("nt: number of time steps\n");
    exit(- 1);
  }
// Define the domain
  const int x_points = atoi(argv[1]);
  const int y_points = atoi(argv[2]);
  const int num_itrs = atoi(argv[3]);
  const double x_len = 2.0;
  const double y_len = 2.0;
  const double del_x = x_len / (x_points - 1);
  const double del_y = y_len / (y_points - 1);
  const int grid_size = (sizeof(double ) * x_points * y_points);
  double *x = (double *)(malloc(sizeof(double ) * x_points));
  double *y = (double *)(malloc(sizeof(double ) * y_points));
  double *u = (double *)(malloc(grid_size));
  double *v = (double *)(malloc(grid_size));
  double *u_new = (double *)(malloc(grid_size));
  double *v_new = (double *)(malloc(grid_size));
// store device results
  double *d_u = (double *)(malloc(grid_size));
  double *d_v = (double *)(malloc(grid_size));
// Define the parameters
  const double nu = 0.01;
  const double sigma = 0.0009;
  const double del_t = sigma * del_x * del_y / nu;
// CFL criteria
  printf("2D Burger's equation\n");
  printf("Grid dimension: x = %d y = %d\n",x_points,y_points);
  printf("Number of time steps: %d\n",num_itrs);
  
#pragma omp parallel for firstprivate (del_x)
  for (int i = 0; i <= x_points - 1; i += 1) {
    x[i] = i * del_x;
  }
  
#pragma omp parallel for firstprivate (del_y)
  for (int i = 0; i <= y_points - 1; i += 1) {
    y[i] = i * del_y;
  }
  
#pragma omp parallel for private (j) firstprivate (x_points)
  for (int i = 0; i <= y_points - 1; i += 1) {
    
#pragma omp parallel for
    for (int j = 0; j <= x_points - 1; j += 1) {
      u[i * y_points + j] = 1.0;
      v[i * y_points + j] = 1.0;
      u_new[i * y_points + j] = 1.0;
      v_new[i * y_points + j] = 1.0;
      if (x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0) {
        u[i * y_points + j] = 2.0;
        v[i * y_points + j] = 2.0;
        u_new[i * y_points + j] = 2.0;
        v_new[i * y_points + j] = 2.0;
      }
    }
  }
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int itr = 0; itr <= num_itrs - 1; itr += 1) {
      
#pragma omp parallel for private (j_nom_4)
      for (int i = 1; i <= y_points - 1 - 1; i += 1) {
        
#pragma omp parallel for firstprivate (del_x,del_y,nu,del_t)
        for (int j = 1; j <= x_points - 1 - 1; j += 1) {
          u_new[i * y_points + j] = u[i * y_points + j] + nu * del_t / (del_x * del_x) * (u[i * y_points + (j + 1)] + u[i * y_points + (j - 1)] - 2 * u[i * y_points + j]) + nu * del_t / (del_y * del_y) * (u[(i + 1) * y_points + j] + u[(i - 1) * y_points + j] - 2 * u[i * y_points + j]) - del_t / del_x * u[i * y_points + j] * (u[i * y_points + j] - u[i * y_points + (j - 1)]) - del_t / del_y * v[i * y_points + j] * (u[i * y_points + j] - u[(i - 1) * y_points + j]);
          v_new[i * y_points + j] = v[i * y_points + j] + nu * del_t / (del_x * del_x) * (v[i * y_points + (j + 1)] + v[i * y_points + (j - 1)] - 2 * v[i * y_points + j]) + nu * del_t / (del_y * del_y) * (v[(i + 1) * y_points + j] + v[(i - 1) * y_points + j] - 2 * v[i * y_points + j]) - del_t / del_x * u[i * y_points + j] * (v[i * y_points + j] - v[i * y_points + (j - 1)]) - del_t / del_y * v[i * y_points + j] * (v[i * y_points + j] - v[(i - 1) * y_points + j]);
        }
      }
// Boundary conditions
      for (int i = 0; i <= x_points - 1; i += 1) {
        u_new[0 * y_points + i] = 1.0;
        v_new[0 * y_points + i] = 1.0;
        u_new[(y_points - 1) * y_points + i] = 1.0;
        v_new[(y_points - 1) * y_points + i] = 1.0;
      }
      for (int j = 0; j <= y_points - 1; j += 1) {
        u_new[j * y_points + 0] = 1.0;
        v_new[j * y_points + 0] = 1.0;
        u_new[j * y_points + (x_points - 1)] = 1.0;
        v_new[j * y_points + (x_points - 1)] = 1.0;
      }
// Updating older values to newer ones
      
#pragma omp parallel for private (j_nom_8)
      for (int i = 0; i <= y_points - 1; i += 1) {
        
#pragma omp parallel for
        for (int j = 0; j <= x_points - 1; j += 1) {
          u[i * y_points + j] = u_new[i * y_points + j];
          v[i * y_points + j] = v_new[i * y_points + j];
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Total kernel execution time %f (s)\n",(time * 1e-9f));
  }
  memcpy(d_u,u,grid_size);
  memcpy(d_v,v,grid_size);
  printf("Serial computing for verification...\n");
// Reset velocities
  
#pragma omp parallel for private (j_nom_10)
  for (int i = 0; i <= y_points - 1; i += 1) {
    
#pragma omp parallel for
    for (int j = 0; j <= x_points - 1; j += 1) {
      u[i * y_points + j] = 1.0;
      v[i * y_points + j] = 1.0;
      u_new[i * y_points + j] = 1.0;
      v_new[i * y_points + j] = 1.0;
      if (x[j] > 0.5 && x[j] < 1.0 && y[i] > 0.5 && y[i] < 1.0) {
        u[i * y_points + j] = 2.0;
        v[i * y_points + j] = 2.0;
        u_new[i * y_points + j] = 2.0;
        v_new[i * y_points + j] = 2.0;
      }
    }
  }
  for (int itr = 0; itr <= num_itrs - 1; itr += 1) {
    
#pragma omp parallel for private (j_nom_13)
    for (int i = 1; i <= y_points - 1 - 1; i += 1) {
      
#pragma omp parallel for firstprivate (del_x,del_y,nu,del_t)
      for (int j = 1; j <= x_points - 1 - 1; j += 1) {
        u_new[i * y_points + j] = u[i * y_points + j] + nu * del_t / (del_x * del_x) * (u[i * y_points + (j + 1)] + u[i * y_points + (j - 1)] - 2 * u[i * y_points + j]) + nu * del_t / (del_y * del_y) * (u[(i + 1) * y_points + j] + u[(i - 1) * y_points + j] - 2 * u[i * y_points + j]) - del_t / del_x * u[i * y_points + j] * (u[i * y_points + j] - u[i * y_points + (j - 1)]) - del_t / del_y * v[i * y_points + j] * (u[i * y_points + j] - u[(i - 1) * y_points + j]);
        v_new[i * y_points + j] = v[i * y_points + j] + nu * del_t / (del_x * del_x) * (v[i * y_points + (j + 1)] + v[i * y_points + (j - 1)] - 2 * v[i * y_points + j]) + nu * del_t / (del_y * del_y) * (v[(i + 1) * y_points + j] + v[(i - 1) * y_points + j] - 2 * v[i * y_points + j]) - del_t / del_x * u[i * y_points + j] * (v[i * y_points + j] - v[i * y_points + (j - 1)]) - del_t / del_y * v[i * y_points + j] * (v[i * y_points + j] - v[(i - 1) * y_points + j]);
      }
    }
// Boundary conditions
    for (int i = 0; i <= x_points - 1; i += 1) {
      u_new[0 * y_points + i] = 1.0;
      v_new[0 * y_points + i] = 1.0;
      u_new[(y_points - 1) * y_points + i] = 1.0;
      v_new[(y_points - 1) * y_points + i] = 1.0;
    }
    for (int j = 0; j <= y_points - 1; j += 1) {
      u_new[j * y_points + 0] = 1.0;
      v_new[j * y_points + 0] = 1.0;
      u_new[j * y_points + (x_points - 1)] = 1.0;
      v_new[j * y_points + (x_points - 1)] = 1.0;
    }
// Updating older values to newer ones
    
#pragma omp parallel for private (j_nom_17)
    for (int i = 0; i <= y_points - 1; i += 1) {
      
#pragma omp parallel for
      for (int j = 0; j <= x_points - 1; j += 1) {
        u[i * y_points + j] = u_new[i * y_points + j];
        v[i * y_points + j] = v_new[i * y_points + j];
      }
    }
  }
  bool ok = true;
  for (int i = 0; i <= y_points - 1; i += 1) {
    for (int j = 0; j <= x_points - 1; j += 1) {
      if (fabs(d_u[i * y_points + j] - u[i * y_points + j]) > 1e-6 || fabs(d_v[i * y_points + j] - v[i * y_points + j]) > 1e-6) 
        ok = false;
    }
  }
  printf("%s\n",(ok?"PASS" : "FAIL"));
  free(x);
  free(y);
  free(u);
  free(v);
  free(d_u);
  free(d_v);
  free(u_new);
  free(v_new);
  return 0;
}
