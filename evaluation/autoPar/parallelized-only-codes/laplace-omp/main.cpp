/** GPU Laplace solver using optimized red-black Gauss–Seidel with SOR solver
 *
 * Solves Laplace's equation in 2D (e.g., heat conduction in a rectangular plate)
 * on GPU using OpenMP with the red-black Gauss–Seidel with sucessive overrelaxation
 * (SOR) that has been "optimized". This means that the red and black kernels 
 * only loop over their respective cells, instead of over all cells and skipping
 * even/odd cells. This requires separate arrays for red and black cells.
 * 
 * Boundary conditions:
 * T = 0 at x = 0, x = L, y = 0
 * T = TN at y = H
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "timer.h"
/** Problem size along one side; total number of cells is this squared */
#define NUM 512
// block size
#define BLOCK_SIZE 128
#define Real float
#define ZERO 0.0f
#define ONE 1.0f
#define TWO 2.0f
/** SOR relaxation parameter */
#include <omp.h> 
const float omega = 1.85f;
/** Function to evaluate coefficient matrix and right-hand side vector.
 * 
 * \param[in]   rowmax   number of rows
 * \param[in]   colmax   number of columns
 * \param[in]   th_cond  thermal conductivity
 * \param[in]   dx       grid size in x dimension (uniform)
 * \param[in]   dy       grid size in y dimension (uniform)
 * \param[in]   width    width of plate (z dimension)
 * \param[in]   TN       temperature at top boundary
 * \param[out]  aP       array of self coefficients
 * \param[out]  aW       array of west neighbor coefficients
 * \param[out]  aE       array of east neighbor coefficients
 * \param[out]  aS       array of south neighbor coefficients
 * \param[out]  aN       array of north neighbor coefficients
 * \param[out]  b        right-hand side array
 */

void fill_coeffs(int rowmax,int colmax,float th_cond,float dx,float dy,float width,float TN,float *aP,float *aW,float *aE,float *aS,float *aN,float *b)
{
  int col;
  int row;
  for (col = 0; col <= colmax - 1; col += 1) {
    for (row = 0; row <= rowmax - 1; row += 1) {
      int ind = col * rowmax + row;
      b[ind] = 0.0f;
      float SP = 0.0f;
      if (col == 0) {
// left BC: temp = 0
        aW[ind] = 0.0f;
        SP = - 2.0f * th_cond * width * dy / dx;
      }
       else {
        aW[ind] = th_cond * width * dy / dx;
      }
      if (col == colmax - 1) {
// right BC: temp = 0
        aE[ind] = 0.0f;
        SP = - 2.0f * th_cond * width * dy / dx;
      }
       else {
        aE[ind] = th_cond * width * dy / dx;
      }
      if (row == 0) {
// bottom BC: temp = 0
        aS[ind] = 0.0f;
        SP = - 2.0f * th_cond * width * dx / dy;
      }
       else {
        aS[ind] = th_cond * width * dx / dy;
      }
      if (row == rowmax - 1) {
// top BC: temp = TN
        aN[ind] = 0.0f;
        b[ind] = 2.0f * th_cond * width * dx * TN / dy;
        SP = - 2.0f * th_cond * width * dx / dy;
      }
       else {
        aN[ind] = th_cond * width * dx / dy;
      }
      aP[ind] = aW[ind] + aE[ind] + aS[ind] + aN[ind] - SP;
// end for row
    }
  }
// end for col
}
// end fill_coeffs
/** Main function that solves Laplace's equation in 2D (heat conduction in plate)
 * 
 * Contains iteration loop for red-black Gauss-Seidel with SOR GPU kernels
 */

int main()
{
// size of plate
  float L = 1.0;
  float H = 1.0;
  float width = 0.01;
// thermal conductivity
  float th_cond = 1.0;
// temperature at top boundary
  float TN = 1.0;
// SOR iteration tolerance
  float tol = 1.e-6;
// number of cells in x and y directions
// including unused boundary cells
  int num_rows = 512 / 2 + 2;
  int num_cols = 512 + 2;
  int size_temp = num_rows * num_cols;
  int size = 512 * 512;
// size of cells
  float dx = L / 512;
  float dy = H / 512;
// iterations for Red-Black Gauss-Seidel with SOR
  int iter;
  int it_max = 1e6;
// allocate memory
  float *aP;
  float *aW;
  float *aE;
  float *aS;
  float *aN;
  float *b;
  float *temp_red;
  float *temp_black;
// arrays of coefficients
  aP = ((float *)(calloc(size,sizeof(float ))));
  aW = ((float *)(calloc(size,sizeof(float ))));
  aE = ((float *)(calloc(size,sizeof(float ))));
  aS = ((float *)(calloc(size,sizeof(float ))));
  aN = ((float *)(calloc(size,sizeof(float ))));
// RHS
  b = ((float *)(calloc(size,sizeof(float ))));
// temperature arrays
  temp_red = ((float *)(calloc(size_temp,sizeof(float ))));
  temp_black = ((float *)(calloc(size_temp,sizeof(float ))));
// set coefficients
  fill_coeffs(512,512,th_cond,dx,dy,width,TN,aP,aW,aE,aS,aN,b);
  int i;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= size_temp - 1; i += 1) {
    temp_red[i] = 0.0f;
    temp_black[i] = 0.0f;
  }
// residual
  float *bl_norm_L2;
// one for each temperature value
  int size_norm = size_temp;
  bl_norm_L2 = ((float *)(calloc(size_norm,sizeof(float ))));
  
#pragma omp parallel for private (i) firstprivate (size_norm)
  for (i = 0; i <= size_norm - 1; i += 1) {
    bl_norm_L2[i] = 0.0f;
  }
// print problem info
  printf("Problem size: %d x %d \n",512,512);
// iteration loop
{
    StartTimer();
    for (iter = 1; iter <= it_max; iter += 1) {
      float norm_L2 = 0.0f;
      for (int row = 1; row <= 256; row += 1) {
        for (int col = 1; col <= 512; col += 1) {
          int ind_red = col * ((512 >> 1) + 2) + row;
// local (red) index
          int ind = 2 * row - (col & 1) - 1 + 512 * (col - 1);
// global index
          float temp_old = temp_red[ind_red];
          float res = b[ind] + (aW[ind] * temp_black[row + (col - 1) * ((512 >> 1) + 2)] + aE[ind] * temp_black[row + (col + 1) * ((512 >> 1) + 2)] + aS[ind] * temp_black[row - (col & 1) + col * ((512 >> 1) + 2)] + aN[ind] * temp_black[row + (col + 1 & 1) + col * ((512 >> 1) + 2)]);
          float temp_new = temp_old * (1.0f - omega) + omega * (res / aP[ind]);
          temp_red[ind_red] = temp_new;
          res = temp_new - temp_old;
          bl_norm_L2[ind_red] = res * res;
        }
      }
// add red cell contributions to residual
      
#pragma omp parallel for reduction (+:norm_L2)
      for (int i = 0; i <= size_norm - 1; i += 1) {
        norm_L2 += bl_norm_L2[i];
      }
      for (int row = 1; row <= 256; row += 1) {
        for (int col = 1; col <= 512; col += 1) {
          int ind_black = col * ((512 >> 1) + 2) + row;
// local (black) index
          int ind = 2 * row - (col + 1 & 1) - 1 + 512 * (col - 1);
// global index
          float temp_old = temp_black[ind_black];
          float res = b[ind] + (aW[ind] * temp_red[row + (col - 1) * ((512 >> 1) + 2)] + aE[ind] * temp_red[row + (col + 1) * ((512 >> 1) + 2)] + aS[ind] * temp_red[row - (col + 1 & 1) + col * ((512 >> 1) + 2)] + aN[ind] * temp_red[row + (col & 1) + col * ((512 >> 1) + 2)]);
          float temp_new = temp_old * (1.0f - omega) + omega * (res / aP[ind]);
          temp_black[ind_black] = temp_new;
          res = temp_new - temp_old;
          bl_norm_L2[ind_black] = res * res;
        }
      }
// transfer residual value(s) back to CPU and 
// add black cell contributions to residual
      
#pragma omp parallel for reduction (+:norm_L2) firstprivate (size_norm)
      for (int i = 0; i <= size_norm - 1; i += 1) {
        norm_L2 += bl_norm_L2[i];
      }
// calculate residual
      norm_L2 = std::sqrt(norm_L2 / ((float )size));
      if (iter % 1000 == 0) 
        printf("%5d, %0.6f\n",iter,norm_L2);
// if tolerance has been reached, end SOR iterations
      if (norm_L2 < tol) 
        break; 
    }
    double runtime = GetTimer();
    printf("Total time for %i iterations: %f s\n",iter,runtime / 1000.0);
  }
// print temperature data to file
  FILE *pfile;
  pfile = fopen("temperature.dat","w");
  if (pfile != 0L) {
    fprintf(pfile,"#x\ty\ttemp(K)\n");
    int row;
    int col;
    for (row = 1; row <= 512; row += 1) {
      for (col = 1; col <= 512; col += 1) {
        float x_pos = (col - 1) * dx + dx / 2;
        float y_pos = (row - 1) * dy + dy / 2;
        if ((row + col) % 2 == 0) {
// even, so red cell
          int ind = col * num_rows + (row + col % 2) / 2;
          fprintf(pfile,"%f\t%f\t%f\n",x_pos,y_pos,temp_red[ind]);
        }
         else {
// odd, so black cell
          int ind = col * num_rows + (row + (col + 1) % 2) / 2;
          fprintf(pfile,"%f\t%f\t%f\n",x_pos,y_pos,temp_black[ind]);
        }
      }
      fprintf(pfile,"\n");
    }
  }
  fclose(pfile);
  free(aP);
  free(aW);
  free(aE);
  free(aS);
  free(aN);
  free(b);
  free(temp_red);
  free(temp_black);
  free(bl_norm_L2);
  return 0;
}
