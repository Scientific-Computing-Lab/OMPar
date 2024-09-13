/*
   Shared memory speeds up performance when we need to access data frequently. 
   Here, the 1D stencil kernel adds all its neighboring data within a radius.
   The C model is added to verify the stencil result on a GPU
*/
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#define RADIUS 7
#define BLOCK_SIZE 256
#include <omp.h> 

int main(int argc,char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n",argv[0]);
    printf("length is a multiple of %d\n",256);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  int size = length;
  int pad_size = length + 7;
// Alloc space for host copies of a, b, c and setup input values
  int *a = (int *)(malloc(pad_size * sizeof(int )));
  int *b = (int *)(malloc(size * sizeof(int )));
  
#pragma omp parallel for
  for (int i = 0; i <= length + 7 - 1; i += 1) {
    a[i] = i;
  }
  auto start = std::chrono::_V2::steady_clock::now();
  for (int i = 0; i <= repeat - 1; i += 1) {
    for (int i = 0; i <= length - 1; i = i + 256) {
      int temp[270];
      for (int j = 0; j <= 255; j += 1) {
        int gindex = i + j;
        temp[j + 7] = a[gindex];
        if (j < 7) {
          temp[j] = (gindex < 7?0 : a[gindex - 7]);
          temp[j + 7 + 256] = a[gindex + 256];
        }
      }
      
#pragma omp parallel for private (offset)
      for (int j = 0; j <= 255; j += 1) {
        int result = 0;
        
#pragma omp parallel for reduction (+:result)
        for (int offset = - 7; offset <= 7; offset += 1) {
          result += temp[j + 7 + offset];
        }
        b[i + j] = result;
      }
    }
  }
  auto end = std::chrono::_V2::steady_clock::now();
  auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
  printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
// verification
  bool ok = true;
  for (int i = 0; i <= 13; i += 1) {
    int s = 0;
    
#pragma omp parallel for reduction (+:s)
    for (int j = i_nom_4; j <= i + 14; j += 1) {
      s += (j < 7?0 : a[j] - 7);
    }
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n",i,s,b[i]);
      ok = false;
      break; 
    }
  }
  for (int i = 2 * 7; i <= length - 1; i += 1) {
    int s = 0;
    
#pragma omp parallel for reduction (+:s)
    for (int j = i_nom_6 - 7; j <= i + 7; j += 1) {
      s += a[j];
    }
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n",i,s,b[i]);
      ok = false;
      break; 
    }
  }
  printf("%s\n",(ok?"PASS" : "FAIL"));
// Cleanup
  free(a);
  free(b);
  return 0;
}
