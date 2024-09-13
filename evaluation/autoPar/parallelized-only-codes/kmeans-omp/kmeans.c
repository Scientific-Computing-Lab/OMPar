#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans.h"
#ifdef WIN
	#include <windows.h>
#else
	#include <pthread.h>
	#include <sys/time.h>

double gettime()
{
  struct timeval t;
  gettimeofday(&t,(void *)0);
  return t . tv_sec + t . tv_usec * 1e-6;
}
#endif

int main(int argc,char **argv)
{
  printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d \n",256,256);
  setup(argc,argv);
  return 0;
}
