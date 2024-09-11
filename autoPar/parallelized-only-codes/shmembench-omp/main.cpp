#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "shmem_kernels.h"
#define VECTOR_SIZE (1024*1024)

int main(int argc,char *argv[])
{
  printf("Shared memory bandwidth microbenchmark\n");
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
// launch kernel n times
  unsigned int datasize = ((1024 * 1024) * sizeof(double ));
  printf("Buffer sizes: %dMB\n",datasize / (1024 * 1024));
  double *c = (double *)(malloc(datasize));
  memset(c,0,sizeof(int ) * (1024 * 1024));
// benchmark execution
  shmembenchGPU(c,(1024 * 1024),n);
  free(c);
  return 0;
}
