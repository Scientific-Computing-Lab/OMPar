#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include "common.h"
#include <omp.h> 

void stopwatch_start(stopwatch *sw)
{
  if (sw == ((void *)0)) 
    return ;
  bzero((&sw -> begin),sizeof(struct timeval ));
  bzero((&sw -> end),sizeof(struct timeval ));
  gettimeofday(&sw -> begin,(void *)0);
}

void stopwatch_stop(stopwatch *sw)
{
  if (sw == ((void *)0)) 
    return ;
  gettimeofday(&sw -> end,(void *)0);
}

double get_interval_by_sec(stopwatch *sw)
{
  if (sw == ((void *)0)) 
    return 0;
  return ((double )(sw -> end . tv_sec - sw -> begin . tv_sec)) + ((double )(sw -> end . tv_usec - sw -> begin . tv_usec)) / 1000000;
}

int get_interval_by_usec(stopwatch *sw)
{
  if (sw == ((void *)0)) 
    return 0;
  return ((sw -> end . tv_sec - sw -> begin . tv_sec) * 1000000 + (sw -> end . tv_usec - sw -> begin . tv_usec));
}

func_ret_t create_matrix_from_file(float **mp,const char *filename,int *size_p)
{
  int i;
  int j;
  int size;
  float *m;
  FILE *fp = ((void *)0);
  fp = fopen(filename,"rb");
  if (fp == ((void *)0)) {
    return RET_FAILURE;
  }
  fscanf(fp,"%d\n",&size);
  m = ((float *)(malloc(sizeof(float ) * size * size)));
  if (m == ((void *)0)) {
    fclose(fp);
    return RET_FAILURE;
  }
  for (i = 0; i <= size - 1; i += 1) {
    for (j = 0; j <= size - 1; j += 1) {
      fscanf(fp,"%f ",m + i * size + j);
    }
  }
  fclose(fp);
   *size_p = size;
   *mp = m;
  return RET_SUCCESS;
}

void matrix_multiply(float *inputa,float *inputb,float *output,int size)
{
  int i;
  int j;
  int k;
  for (i = 0; i <= size - 1; i += 1) {
    for (k = 0; k <= size - 1; k += 1) {
      for (j = 0; j <= size - 1; j += 1) {
        output[i * size + j] = inputa[i * size + k] * inputb[k * size + j];
      }
    }
  }
}

void lud_verify(float *m,float *lu,int matrix_dim)
{
  int i;
  int j;
  int k;
  float *tmp = (float *)(malloc((matrix_dim * matrix_dim) * sizeof(float )));
  
#pragma omp parallel for private (i,j,k)
  for (i = 0; i <= matrix_dim - 1; i += 1) {
    
#pragma omp parallel for private (j,k)
    for (j = 0; j <= matrix_dim - 1; j += 1) {
      float sum = 0;
      float l;
      float u;
      
#pragma omp parallel for private (l,u,k) reduction (+:sum)
      for (k = 0; k <= ((i < j?i : j)); k += 1) {
        if (i == k) 
          l = 1;
         else 
          l = lu[i * matrix_dim + k];
        u = lu[k * matrix_dim + j];
        sum += l * u;
      }
      tmp[i * matrix_dim + j] = sum;
    }
  }
/* printf(">>>>>LU<<<<<<<\n"); */
/* for (i=0; i<matrix_dim; i++){ */
/*   for (j=0; j<matrix_dim;j++){ */
/*       printf("%f ", lu[i*matrix_dim+j]); */
/*   } */
/*   printf("\n"); */
/* } */
/* printf(">>>>>result<<<<<<<\n"); */
/* for (i=0; i<matrix_dim; i++){ */
/*   for (j=0; j<matrix_dim;j++){ */
/*       printf("%f ", tmp[i*matrix_dim+j]); */
/*   } */
/*   printf("\n"); */
/* } */
/* printf(">>>>>input<<<<<<<\n"); */
/* for (i=0; i<matrix_dim; i++){ */
/*   for (j=0; j<matrix_dim;j++){ */
/*       printf("%f ", m[i*matrix_dim+j]); */
/*   } */
/*   printf("\n"); */
/* } */
  for (i = 0; i <= matrix_dim - 1; i += 1) {
    for (j = 0; j <= matrix_dim - 1; j += 1) {
      if (fabs((m[i * matrix_dim + j] - tmp[i * matrix_dim + j])) > 0.0001) 
        printf("dismatch at (%d, %d): (o)%f (n)%f\n",i,j,m[i * matrix_dim + j],tmp[i * matrix_dim + j]);
    }
  }
  free(tmp);
}

void matrix_duplicate(float *src,float **dst,int matrix_dim)
{
  int s = ((matrix_dim * matrix_dim) * sizeof(float ));
  float *p = (float *)(malloc(s));
  memcpy(p,src,s);
   *dst = p;
}

void print_matrix(float *m,int matrix_dim)
{
  int i;
  int j;
  for (i = 0; i <= matrix_dim - 1; i += 1) {
    for (j = 0; j <= matrix_dim - 1; j += 1) {
      printf("%f ",m[i * matrix_dim + j]);
    }
    printf("\n");
  }
}
// Generate well-conditioned matrix internally  by Ke Wang 2013/08/07 22:20:06

func_ret_t create_matrix(float **mp,int size)
{
  float *m;
  int i;
  int j;
  float lamda = (- 0.001);
  float coe[2 * size - 1];
  float coe_i = 0.0;
  for (i = 0; i <= size - 1; i += 1) {
    coe_i = (10 * exp((lamda * i)));
    j = size - 1 + i;
    coe[j] = coe_i;
    j = size - 1 - i;
    coe[j] = coe_i;
  }
  m = ((float *)(malloc(sizeof(float ) * size * size)));
  if (m == ((void *)0)) {
    return RET_FAILURE;
  }
  
#pragma omp parallel for private (i,j) firstprivate (size)
  for (i = 0; i <= size - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= size - 1; j += 1) {
      m[i * size + j] = coe[size - 1 - i + j];
    }
  }
   *mp = m;
  return RET_SUCCESS;
}
