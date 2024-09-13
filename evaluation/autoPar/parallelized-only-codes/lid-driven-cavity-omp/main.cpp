/** GPU solver for 2D lid-driven cavity problem, using finite difference method
 * \file main_gpu.cpp
 *
 * Solve the incompressible, isothermal 2D Navier–Stokes equations for a square
 * lid-driven cavity on a GPU (via CUDA), using the finite difference method.
 * To change the grid resolution, modify "NUM". In addition, the problem is controlled
 * by the Reynolds number ("Re_num").
 * 
 * Based on the methodology given in Chapter 3 of "Numerical Simulation in Fluid
 * Dynamics", by M. Griebel, T. Dornseifer, and T. Neunhoeffer. SIAM, Philadelphia,
 * PA, 1998.
 * 
 * Boundary conditions:
 * u = 0 and v = 0 at x = 0, x = L, y = 0
 * u = ustar at y = H
 * v = 0 at y = H
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
/** Problem size along one side; total number of cells is this squared */
#define NUM 512
// block size
#define BLOCK_SIZE 128
/** Double precision */
#define DOUBLE
#ifdef DOUBLE
#define Real double
#define ZERO 0.0
#define ONE 1.0
#define TWO 2.0
#define FOUR 4.0
#define SMALL 1.0e-10;
/** Reynolds number */
#include <omp.h> 
const double Re_num = 1000.0;
/** SOR relaxation parameter */
const double omega = 1.7;
/** Discretization mixture parameter (gamma) */
const double mix_param = 0.9;
/** Safety factor for time step modification */
const double tau = 0.5;
/** Body forces in x- and y- directions */
const double gx = 0.0;
const double gy = 0.0;
/** Domain size (non-dimensional) */
#define xLength 1.0
#define yLength 1.0
#else
#define Real float
// replace double functions with float versions
#undef fmin
#define fmin fminf
#undef fmax
#define fmax fmaxf
#undef fabs
#define fabs fabsf
#undef sqrt
#define sqrt sqrtf
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
#define FOUR 4.0f
#define SMALL 1.0e-10f;
/** Reynolds number */
/** SOR relaxation parameter */
/** Discretization mixture parameter (gamma) */
/** Safety factor for time step modification */
/** Body forces in x- and y- directions */
/** Domain size (non-dimensional) */
#define xLength 1.0f
#define yLength 1.0f
#endif
/** Mesh sizes */
const double dx = 1.0 / 512;
const double dy = 1.0 / 512;
/** Max macro (type safe, from GNU) */
//#define MAX(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a > _b ? _a : _b; })
/** Min macro (type safe) */
//#define MIN(a,b) ({ __typeof__ (a) _a = (a); __typeof__ (b) _b = (b); _a < _b ? _a : _b; })
// map two-dimensional indices to one-dimensional memory
/* (previously processed: ignoring self-referential macro declaration) macro name = u */ 
/* (previously processed: ignoring self-referential macro declaration) macro name = v */ 
/* (previously processed: ignoring self-referential macro declaration) macro name = F */ 
/* (previously processed: ignoring self-referential macro declaration) macro name = G */ 
/* (previously processed: ignoring self-referential macro declaration) macro name = pres_red */ 
/* (previously processed: ignoring self-referential macro declaration) macro name = pres_black */ 
///////////////////////////////////////////////////////////////////////////////

void set_BCs_host(double *u,double *v)
{
  int ind;
// loop through rows and columns
  for (ind = 0; ind <= 513; ind += 1) {
// left boundary
    u[0 * (512 + 2) + ind] = 0.0;
    v[0 * (512 + 2) + ind] = -v[1 * (512 + 2) + ind];
// right boundary
    u[512 * (512 + 2) + ind] = 0.0;
    v[(512 + 1) * (512 + 2) + ind] = -v[512 * (512 + 2) + ind];
// bottom boundary
    u[ind * (512 + 2) + 0] = -u[ind * (512 + 2) + 1];
    v[ind * (512 + 2) + 0] = 0.0;
// top boundary
    u[ind * (512 + 2) + (512 + 1)] = 2.0 - u[ind * (512 + 2) + 512];
    v[ind * (512 + 2) + 512] = 0.0;
    if (ind == 512) {
// left boundary
      u[0 * (512 + 2) + 0] = 0.0;
      v[0 * (512 + 2) + 0] = -v[1 * (512 + 2) + 0];
      u[0 * (512 + 2) + (512 + 1)] = 0.0;
      v[0 * (512 + 2) + (512 + 1)] = -v[1 * (512 + 2) + (512 + 1)];
// right boundary
      u[512 * (512 + 2) + 0] = 0.0;
      v[(512 + 1) * (512 + 2) + 0] = -v[512 * (512 + 2) + 0];
      u[512 * (512 + 2) + (512 + 1)] = 0.0;
      v[(512 + 1) * (512 + 2) + (512 + 1)] = -v[512 * (512 + 2) + (512 + 1)];
// bottom boundary
      u[0 * (512 + 2) + 0] = -u[0 * (512 + 2) + 1];
      v[0 * (512 + 2) + 0] = 0.0;
      u[(512 + 1) * (512 + 2) + 0] = -u[(512 + 1) * (512 + 2) + 1];
      v[(512 + 1) * (512 + 2) + 0] = 0.0;
// top boundary
      u[0 * (512 + 2) + (512 + 1)] = 2.0 - u[0 * (512 + 2) + 512];
      v[0 * (512 + 2) + 512] = 0.0;
      u[(512 + 1) * (512 + 2) + (512 + 1)] = 2.0 - u[(512 + 1) * (512 + 2) + 512];
      v[ind * (512 + 2) + (512 + 1)] = 0.0;
// end if
    }
  }
// end for
}
// end set_BCs_host
///////////////////////////////////////////////////////////////////////////////

int main(int argc,char *argv[])
{
// iterations for Red-Black Gauss-Seidel with SOR
  int iter = 0;
  const int it_max = 1000000;
// SOR iteration tolerance
  const double tol = 0.001;
// time range
  const double time_start = 0.0;
  const double time_end = 0.001;
//20.0;
// initial time step size
  double dt = 0.02;
  int size = (512 + 2) * (512 + 2);
  int size_pres = (512 / 2 + 2) * (512 + 2);
// arrays for pressure and velocity
  double *F;
  double *u;
  double *G;
  double *v;
  F = ((double *)(calloc(size,sizeof(double ))));
  u = ((double *)(calloc(size,sizeof(double ))));
  G = ((double *)(calloc(size,sizeof(double ))));
  v = ((double *)(calloc(size,sizeof(double ))));
  
#pragma omp parallel for firstprivate (size)
  for (int i = 0; i <= size - 1; i += 1) {
    F[i] = 0.0;
    u[i] = 0.0;
    G[i] = 0.0;
    v[i] = 0.0;
  }
// arrays for pressure
  double *pres_red;
  double *pres_black;
  pres_red = ((double *)(calloc(size_pres,sizeof(double ))));
  pres_black = ((double *)(calloc(size_pres,sizeof(double ))));
  
#pragma omp parallel for firstprivate (size_pres)
  for (int i = 0; i <= size_pres - 1; i += 1) {
    pres_red[i] = 0.0;
    pres_black[i] = 0.0;
  }
// print problem size
  printf("Problem size: %d x %d \n",512,512);
// residual variable
  double *res_arr;
  int size_res = 512 / (2 * 128) * 512;
  res_arr = ((double *)(calloc(size_res,sizeof(double ))));
// variables to store maximum velocities
  double *max_u_arr;
  double *max_v_arr;
  int size_max = size_res;
  max_u_arr = ((double *)(calloc(size_max,sizeof(double ))));
  max_v_arr = ((double *)(calloc(size_max,sizeof(double ))));
// pressure sum
  double *pres_sum;
  pres_sum = ((double *)(calloc(size_res,sizeof(double ))));
// set initial BCs
  set_BCs_host(u,v);
  double max_u = 1.0e-10;
  ;
  double max_v = 1.0e-10;
  ;
// get max velocity for initial values (including BCs)
  for (int col = 0; col <= 513; col += 1) {
    for (int row = 1; row <= 513; row += 1) {
      max_u = fmax(max_u,(fabs(u[col * (512 + 2) + row])));
    }
  }
  for (int col = 1; col <= 513; col += 1) {
    for (int row = 0; row <= 513; row += 1) {
      max_v = fmax(max_v,(fabs(v[col * (512 + 2) + row])));
    }
  }
{
    double time = time_start;
// time-step size based on grid and Reynolds number
    double dt_Re = 0.5 * Re_num / (1.0 / (dx * dx) + 1.0 / (dy * dy));
    auto start = std::chrono::_V2::steady_clock::now();
// time iteration loop
    while(time < time_end){
// calculate time step based on stability and CFL
      dt = fmin(dx / max_u,dy / max_v);
      dt = tau * fmin(dt_Re,dt);
      if (time + dt >= time_end) {
        dt = time_end - time;
      }
// calculate F and G    
//calculate_F <<<grid_F, block_F>>> (dt, u_d, v_d, F_d);
      #include "calculate_F.h"
//    calculate_G <<<grid_G, block_G>>> (dt, u_d, v_d, G_d);
      #include "calculate_G.h"
// get L2 norm of initial pressure
//sum_pressure <<<grid_pr, block_pr>>> (pres_red_d, pres_black_d, pres_sum_d);
      #include "sum_pressure.h"
//cudaMemcpy (pres_sum, pres_sum_d, size_res * sizeof(Real), cudaMemcpyDeviceToHost);
      double p0_norm = 0.0;
      
#pragma omp parallel for reduction (+:p0_norm) firstprivate (size_res)
      for (int i = 0; i <= size_res - 1; i += 1) {
        p0_norm += pres_sum[i];
      }
//printf("p_norm = %lf\n", p0_norm);
      p0_norm = sqrt(p0_norm / ((double )(512 * 512)));
      if (p0_norm < 0.0001) {
        p0_norm = 1.0;
      }
      double norm_L2;
// calculate new pressure
// red-black Gauss-Seidel with SOR iteration loop
      for (iter = 1; iter <= it_max; iter += 1) {
// set pressure boundary conditions
//set_horz_pres_BCs <<<grid_hpbc, block_hpbc>>> (pres_red_d, pres_black_d);
        #include "set_horz_pres_BCs.h"
//      set_vert_pres_BCs <<<grid_vpbc, block_hpbc>>> (pres_red_d, pres_black_d);
        #include "set_vert_pres_BCs.h"
// update red cells
//      red_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_black_d, pres_red_d);
        #include "red_kernel.h"
// update black cells
//      black_kernel <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d);
        #include "black_kernel.h"
// calculate residual values
//calc_residual <<<grid_pr, block_pr>>> (dt, F_d, G_d, pres_red_d, pres_black_d, res_d);
        #include "calc_residual.h"
// transfer residual value(s) back to CPU
//      cudaMemcpy (res, res_d, size_res * sizeof(Real), cudaMemcpyDeviceToHost);
        norm_L2 = 0.0;
        
#pragma omp parallel for reduction (+:norm_L2) firstprivate (size_res)
        for (int i = 0; i <= size_res - 1; i += 1) {
          norm_L2 += res_arr[i];
        }
//printf("norm_L2 = %lf\n", norm_L2);
// calculate residual
        norm_L2 = sqrt(norm_L2 / ((double )(512 * 512))) / p0_norm;
// if tolerance has been reached, end SOR iterations
        if (norm_L2 < tol) {
          break; 
        }
      }
// end for
      printf("Time = %f, delt = %e, iter = %i, res = %e\n",time + dt,dt,iter,norm_L2);
// calculate new velocities and transfer maximums back
//calculate_u <<<grid_pr, block_pr>>> (dt, F_d, pres_red_d, pres_black_d, u_d, max_u_d);
      #include "calculate_u.h"
//    cudaMemcpy (max_u_arr, max_u_d, size_max * sizeof(Real), cudaMemcpyDeviceToHost);
//    calculate_v <<<grid_pr, block_pr>>> (dt, G_d, pres_red_d, pres_black_d, v_d, max_v_d);
      #include "calculate_v.h"
//    cudaMemcpy (max_v_arr, max_v_d, size_max * sizeof(Real), cudaMemcpyDeviceToHost);
// get maximum u- and v- velocities
      max_v = 1.0e-10;
      ;
      max_u = 1.0e-10;
      ;
      for (int i = 0; i <= size_max - 1; i += 1) {
        double test_u = max_u_arr[i];
        max_u = fmax(max_u,test_u);
        double test_v = max_v_arr[i];
        max_v = fmax(max_v,test_v);
      }
// set velocity boundary conditions
//set_BCs <<<grid_bcs, block_bcs>>> (u_d, v_d);
      #include "set_BCs.h"
// increase time
      time += dt;
// single time step
//break;
// end while
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("\nTotal execution time of the iteration loop: %f (s)\n",(elapsed_time * 1e-9f));
  }
// transfer final temperature values back implicitly
// write data to file
  FILE *pfile;
  pfile = fopen("velocity_gpu.dat","w");
  fprintf(pfile,"#x\ty\tu\tv\n");
  if (pfile != 0L) {
    for (int row = 0; row <= 511; row += 1) {
      for (int col = 0; col <= 511; col += 1) {
        double u_ij = u[col * 512 + row];
        double u_im1j;
        if (col == 0) {
          u_im1j = 0.0;
        }
         else {
          u_im1j = u[(col - 1) * 512 + row];
        }
        u_ij = (u_ij + u_im1j) / 2.0;
        double v_ij = v[col * 512 + row];
        double v_ijm1;
        if (row == 0) {
          v_ijm1 = 0.0;
        }
         else {
          v_ijm1 = v[col * 512 + row - 1];
        }
        v_ij = (v_ij + v_ijm1) / 2.0;
        fprintf(pfile,"%f\t%f\t%f\t%f\n",(((double )col) + 0.5) * dx,(((double )row) + 0.5) * dy,u_ij,v_ij);
      }
    }
  }
  fclose(pfile);
  free(pres_red);
  free(pres_black);
  free(u);
  free(v);
  free(F);
  free(G);
  free(max_u_arr);
  free(max_v_arr);
  free(res_arr);
  free(pres_sum);
  return 0;
}
