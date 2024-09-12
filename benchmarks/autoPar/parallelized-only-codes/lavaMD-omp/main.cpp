#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include "./util/timer/timer.h"
#include "./util/num/num.h"
#include "main.h"
#include <omp.h> 

int main(int argc,char *argv[])
{
// counters
  int i;
  int j;
  int k;
  int l;
  int m;
  int n;
// system memory
  ::par_str par_cpu;
  ::dim_str dim_cpu;
  ::box_str *box_cpu;
  FOUR_VECTOR *rv_cpu;
  float *qv_cpu;
  FOUR_VECTOR *fv_cpu;
  int nh;
  printf("WG size of kernel = %d \n",128);
// assing default values
  dim_cpu . arch_arg = 0;
  dim_cpu . cores_arg = 1;
  dim_cpu . boxes1d_arg = 1;
// go through arguments
  if (argc == 3) {
    for (dim_cpu . cur_arg = 1; dim_cpu . cur_arg < argc; dim_cpu . cur_arg++) {
// check if -boxes1d
      if (strcmp(argv[dim_cpu . cur_arg],"-boxes1d") == 0) {
// check if value provided
        if (argc >= dim_cpu . cur_arg + 1) {
// check if value is a number
          if (isInteger(argv[dim_cpu . cur_arg + 1]) == 1) {
            dim_cpu . boxes1d_arg = atoi(argv[dim_cpu . cur_arg + 1]);
            if (dim_cpu . boxes1d_arg < 0) {
              printf("ERROR: Wrong value to -boxes1d argument, cannot be <=0\n");
              return 0;
            }
            dim_cpu . cur_arg = dim_cpu . cur_arg + 1;
          }
           else 
// value is not a number
{
            printf("ERROR: Value to -boxes1d argument in not a number\n");
            return 0;
          }
        }
         else 
// value not provided
{
          printf("ERROR: Missing value to -boxes1d argument\n");
          return 0;
        }
      }
       else 
// unknown
{
        printf("ERROR: Unknown argument\n");
        return 0;
      }
    }
// Print configuration
    printf("Configuration used: arch = %d, cores = %d, boxes1d = %d\n",dim_cpu . arch_arg,dim_cpu . cores_arg,dim_cpu . boxes1d_arg);
  }
   else {
    printf("Provide boxes1d argument, example: -boxes1d 16");
    return 0;
  }
  par_cpu . alpha = 0.5;
// total number of boxes
  dim_cpu . number_boxes = (dim_cpu . boxes1d_arg * dim_cpu . boxes1d_arg * dim_cpu . boxes1d_arg);
// 8*8*8=512
// how many particles space has in each direction
  dim_cpu . space_elem = dim_cpu . number_boxes * 100;
//512*100=51,200
  dim_cpu . space_mem = (dim_cpu . space_elem * sizeof(FOUR_VECTOR ));
  dim_cpu . space_mem2 = (dim_cpu . space_elem * sizeof(float ));
// box array
  dim_cpu . box_mem = (dim_cpu . number_boxes * sizeof(::box_str ));
// allocate boxes
  box_cpu = ((::box_str *)(malloc(dim_cpu . box_mem)));
// initialize number of home boxes
  nh = 0;
// home boxes in z direction
  for (i = 0; i <= dim_cpu . boxes1d_arg - 1; i += 1) {
// home boxes in y direction
    for (j = 0; j <= dim_cpu . boxes1d_arg - 1; j += 1) {
// home boxes in x direction
      for (k = 0; k <= dim_cpu . boxes1d_arg - 1; k += 1) {
// current home box
        box_cpu[nh] . x = k;
        box_cpu[nh] . y = j;
        box_cpu[nh] . z = i;
        box_cpu[nh] . number = nh;
        box_cpu[nh] . offset = (nh * 100);
// initialize number of neighbor boxes
        box_cpu[nh] . nn = 0;
// neighbor boxes in z direction
        for (l = - 1; l <= 1; l += 1) {
// neighbor boxes in y direction
          for (m = - 1; m <= 1; m += 1) {
// neighbor boxes in x direction
            for (n = - 1; n <= 1; n += 1) {
// check if (this neighbor exists) and (it is not the same as home box)
              if ((i + l >= 0 && j + m >= 0 && k + n >= 0) == true && (i + l < dim_cpu . boxes1d_arg && j + m < dim_cpu . boxes1d_arg && k + n < dim_cpu . boxes1d_arg) == true && (l == 0 && m == 0 && n == 0) == false) {
// current neighbor box
                box_cpu[nh] . nei[box_cpu[nh] . nn] . x = k + n;
                box_cpu[nh] . nei[box_cpu[nh] . nn] . y = j + m;
                box_cpu[nh] . nei[box_cpu[nh] . nn] . z = i + l;
                box_cpu[nh] . nei[box_cpu[nh] . nn] . number = box_cpu[nh] . nei[box_cpu[nh] . nn] . z * dim_cpu . boxes1d_arg * dim_cpu . boxes1d_arg + box_cpu[nh] . nei[box_cpu[nh] . nn] . y * dim_cpu . boxes1d_arg + box_cpu[nh] . nei[box_cpu[nh] . nn] . x;
                box_cpu[nh] . nei[box_cpu[nh] . nn] . offset = (box_cpu[nh] . nei[box_cpu[nh] . nn] . number * 100);
// increment neighbor box
                box_cpu[nh] . nn = box_cpu[nh] . nn + 1;
              }
            }
// neighbor boxes in x direction
          }
// neighbor boxes in y direction
        }
// neighbor boxes in z direction
// increment home box
        nh = nh + 1;
// home boxes in x direction
      }
    }
// home boxes in y direction
  }
// home boxes in z direction
//====================================================================================================100
//  PARAMETERS, DISTANCE, CHARGE AND FORCE
//====================================================================================================100
// random generator seed set to random value - time in this case
  srand(2);
// input (distances)
  rv_cpu = ((FOUR_VECTOR *)(malloc(dim_cpu . space_mem)));
  
#pragma omp parallel for private (i)
  for (i = 0; ((long )i) <= dim_cpu . space_elem - 1; i = i + 1) {
    rv_cpu[i] . v = ((rand() % 10 + 1) / 10.0);
// get a number in the range 0.1 - 1.0
// rv_cpu[i].v = 0.1;      // get a number in the range 0.1 - 1.0
    rv_cpu[i] . x = ((rand() % 10 + 1) / 10.0);
// get a number in the range 0.1 - 1.0
// rv_cpu[i].x = 0.2;      // get a number in the range 0.1 - 1.0
    rv_cpu[i] . y = ((rand() % 10 + 1) / 10.0);
// get a number in the range 0.1 - 1.0
// rv_cpu[i].y = 0.3;      // get a number in the range 0.1 - 1.0
    rv_cpu[i] . z = ((rand() % 10 + 1) / 10.0);
// get a number in the range 0.1 - 1.0
// rv_cpu[i].z = 0.4;      // get a number in the range 0.1 - 1.0
  }
// input (charge)
  qv_cpu = ((float *)(malloc(dim_cpu . space_mem2)));
  
#pragma omp parallel for private (i)
  for (i = 0; ((long )i) <= dim_cpu . space_elem - 1; i = i + 1) {
    qv_cpu[i] = ((rand() % 10 + 1) / 10.0);
// get a number in the range 0.1 - 1.0
// qv_cpu[i] = 0.5;      // get a number in the range 0.1 - 1.0
  }
// output (forces)
  fv_cpu = ((FOUR_VECTOR *)(malloc(dim_cpu . space_mem)));
  
#pragma omp parallel for private (i)
  for (i = 0; ((long )i) <= dim_cpu . space_elem - 1; i = i + 1) {
    fv_cpu[i] . v = 0;
// set to 0, because kernels keeps adding to initial value
    fv_cpu[i] . x = 0;
// set to 0, because kernels keeps adding to initial value
    fv_cpu[i] . y = 0;
// set to 0, because kernels keeps adding to initial value
    fv_cpu[i] . z = 0;
// set to 0, because kernels keeps adding to initial value
  }
  long long kstart;
  long long kend;
  long long start = get_time();
// only the member number_boxes is used in the kernel
  int dim_cpu_number_boxes = dim_cpu . number_boxes;
{
    kstart = get_time();
{
      FOUR_VECTOR rA_shared[100];
      FOUR_VECTOR rB_shared[100];
      float qB_shared[100];
{
        int bx = omp_get_team_num();
        int tx = omp_get_thread_num();
        int wtx = tx;
//  DO FOR THE NUMBER OF BOXES
        if (bx < dim_cpu_number_boxes) {
//  Extract input parameters
// parameters
          float a2 = 2 * par_cpu . alpha * par_cpu . alpha;
// home box
          int first_i;
// (enable the line below only if wanting to use shared memory)
// nei box
          int pointer;
          int k = 0;
          int first_j;
          int j = 0;
// (enable the two lines below only if wanting to use shared memory)
// common
          float r2;
          float u2;
          float vij;
          float fs;
          float fxij;
          float fyij;
          float fzij;
          THREE_VECTOR d;
//  Home box
//  Setup parameters
// home box - box parameters
          first_i = box_cpu[bx] . offset;
//  Copy to shared memory
// (enable the section below only if wanting to use shared memory)
// home box - shared memory
          while(wtx < 100){
            rA_shared[wtx] = rv_cpu[first_i + wtx];
            wtx = wtx + 128;
          }
          wtx = tx;
// (enable the section below only if wanting to use shared memory)
// synchronize threads  - not needed, but just to be safe for now
//  nei box loop
// loop over nei boxes of home box
          for (k = 0; k <= 1 + box_cpu[bx] . nn - 1; k += 1) {
//----------------------------------------50
//  nei box - get pointer to the right box
//----------------------------------------50
            if (k == 0) {
              pointer = bx;
// set first box to be processed to home box
            }
             else {
              pointer = box_cpu[bx] . nei[k - 1] . number;
// remaining boxes are nei boxes
            }
//  Setup parameters
// nei box - box parameters
            first_j = box_cpu[pointer] . offset;
// (enable the section below only if wanting to use shared memory)
// nei box - shared memory
            while(wtx < 100){
              rB_shared[wtx] = rv_cpu[first_j + wtx];
              qB_shared[wtx] = qv_cpu[first_j + wtx];
              wtx = wtx + 128;
            }
            wtx = tx;
// (enable the section below only if wanting to use shared memory)
// synchronize threads because in next section each thread accesses data brought in by different threads here
//  Calculation
// loop for the number of particles in the home box
            while(wtx < 100){
// loop for the number of particles in the current nei box
              for (j = 0; j <= 99; j += 1) {
                r2 = rA_shared[wtx] . v + rB_shared[j] . v - (rA_shared[wtx] . x * rB_shared[j] . x + rA_shared[wtx] . y * rB_shared[j] . y + rA_shared[wtx] . z * rB_shared[j] . z);
                u2 = a2 * r2;
                vij = std::exp(-u2);
                fs = 2 * vij;
                d . x = rA_shared[wtx] . x - rB_shared[j] . x;
                fxij = fs * d . x;
                d . y = rA_shared[wtx] . y - rB_shared[j] . y;
                fyij = fs * d . y;
                d . z = rA_shared[wtx] . z - rB_shared[j] . z;
                fzij = fs * d . z;
                fv_cpu[first_i + wtx] . v += qB_shared[j] * vij;
                fv_cpu[first_i + wtx] . x += qB_shared[j] * fxij;
                fv_cpu[first_i + wtx] . y += qB_shared[j] * fyij;
                fv_cpu[first_i + wtx] . z += qB_shared[j] * fzij;
              }
// increment work thread index
              wtx = wtx + 128;
            }
// reset work index
            wtx = tx;
// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
          }
        }
      }
    }
    kend = get_time();
  }
  long long end = get_time();
  printf("Device offloading time:\n");
  printf("%.12f s\n",(((float )(end - start)) / 1000000));
  printf("Kernel execution time:\n");
  printf("%.12f s\n",(((float )(kend - kstart)) / 1000000));
#ifdef DEBUG
#endif
// dump results
#ifdef OUTPUT
#endif         
  free(rv_cpu);
  free(qv_cpu);
  free(fv_cpu);
  free(box_cpu);
  return 0;
}
