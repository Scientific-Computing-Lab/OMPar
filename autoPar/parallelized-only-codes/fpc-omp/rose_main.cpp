#include <stdio.h>
#include <stdlib.h> 
#include <chrono>
#include <omp.h>
#include <omp.h> 
typedef unsigned long ulong;

ulong *convertBuffer2Array(char *cbuffer,unsigned int size,unsigned int step)
{
  unsigned int i;
  unsigned int j;
  ulong *values = 0L;
  posix_memalign((void **)(&values),1024,sizeof(ulong ) * size / step);
  
#pragma omp parallel for private (i)
  for (i = 0; i <= size / step - 1; i += 1) {
    values[i] = 0;
// Initialize all elements to zero.
  }
  for (i = 0; i <= size - 1; i += step) {
    for (j = 0; j <= step - 1; j += 1) {
      values[i / step] += ((ulong )((unsigned char )cbuffer[i + j])) << 8 * j;
    }
  }
  return values;
}

unsigned int my_abs(int x)
{
  unsigned int t = (x >> 31);
  return (x ^ t) - t;
}

unsigned int FPCCompress(ulong *values,unsigned int size)
{
  unsigned int compressable = 0;
  unsigned int i;
  for (i = 0; i <= size - 1; i += 1) {
// 000
    if (values[i] == 0) {
      compressable += 1;
      continue; 
    }
// 001 010
    if (my_abs((int )values[i]) <= 0xFF) {
      compressable += 1;
      continue; 
    }
// 011
    if (my_abs((int )values[i]) <= 0xFFFF) {
      compressable += 2;
      continue; 
    }
//100  
    if ((values[i] & 0xFFFF) == 0) {
      compressable += 2;
      continue; 
    }
//101
    if (my_abs((int )(values[i] & 0xFFFF)) <= 0xFF && my_abs((int )(values[i] >> 16 & 0xFFFF)) <= 0xFF) {
      compressable += 2;
      continue; 
    }
//110
    unsigned int byte0 = (values[i] & 0xFF);
    unsigned int byte1 = (values[i] >> 8 & 0xFF);
    unsigned int byte2 = (values[i] >> 16 & 0xFF);
    unsigned int byte3 = (values[i] >> 24 & 0xFF);
    if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) {
      compressable += 1;
      continue; 
    }
//111
    compressable += 4;
  }
  return compressable;
}

unsigned int f1(ulong value,bool *mask)
{
  if (value == 0) {
     *mask = 1;
  }
  return 1;
}

unsigned int f2(ulong value,bool *mask)
{
  if (my_abs((int )value) <= 0xFF) 
     *mask = 1;
  return 1;
}

unsigned int f3(ulong value,bool *mask)
{
  if (my_abs((int )value) <= 0xFFFF) 
     *mask = 1;
  return 2;
}

unsigned int f4(ulong value,bool *mask)
{
  if ((value & 0xFFFF) == 0) 
     *mask = 1;
  return 2;
}

unsigned int f5(ulong value,bool *mask)
{
  if (my_abs((int )(value & 0xFFFF)) <= 0xFF && my_abs((int )(value >> 16 & 0xFFFF)) <= 0xFF) 
     *mask = 1;
  return 2;
}

unsigned int f6(ulong value,bool *mask)
{
  unsigned int byte0 = (value & 0xFF);
  unsigned int byte1 = (value >> 8 & 0xFF);
  unsigned int byte2 = (value >> 16 & 0xFF);
  unsigned int byte3 = (value >> 24 & 0xFF);
  if (byte0 == byte1 && byte0 == byte2 && byte0 == byte3) 
     *mask = 1;
  return 1;
}

unsigned int f7(ulong value,bool *mask)
{
   *mask = 1;
  return 4;
}

void fpc(const ulong *values,unsigned int *cmp_size,const int values_size,const int wgs)
{
   *cmp_size = 0;
{
{
      unsigned int compressable;
{
        int lid = omp_get_thread_num();
        int WGS = omp_get_num_threads();
        int gid = omp_get_team_num() * WGS + lid;
        ulong value = values[gid];
        unsigned int inc;
// 000
        if (value == 0) {
          inc = 1;
        }
         else 
// 001 010
if (my_abs((int )value) <= 0xFF) {
          inc = 1;
        }
         else 
// 011
if (my_abs((int )value) <= 0xFFFF) {
          inc = 2;
        }
         else 
//100  
if ((value & 0xFFFF) == 0) {
          inc = 2;
        }
         else 
//101
if (my_abs((int )(value & 0xFFFF)) <= 0xFF && my_abs((int )(value >> 16 & 0xFFFF)) <= 0xFF) {
          inc = 2;
        }
         else 
//110
if ((value & 0xFF) == (value >> 8 & 0xFF) && (value & 0xFF) == (value >> 16 & 0xFF) && (value & 0xFF) == (value >> 24 & 0xFF)) {
          inc = 1;
        }
         else {
          inc = 4;
        }
        if (lid == 0) 
          compressable = 0;
        compressable += inc;
        if (lid == WGS - 1) {
          cmp_size[0] += compressable;
        }
      }
    }
  }
}

void fpc2(const ulong *values,unsigned int *cmp_size,const int values_size,const int wgs)
{
   *cmp_size = 0;
{
{
      unsigned int compressable;
{
        int lid = omp_get_thread_num();
        int WGS = omp_get_num_threads();
        int gid = omp_get_team_num() * WGS + lid;
        ulong value = values[gid];
        unsigned int inc;
        bool m1 = 0;
        bool m2 = 0;
        bool m3 = 0;
        bool m4 = 0;
        bool m5 = 0;
        bool m6 = 0;
        bool m7 = 0;
        unsigned int inc1 = f1(value,&m1);
        unsigned int inc2 = f2(value,&m2);
        unsigned int inc3 = f3(value,&m3);
        unsigned int inc4 = f4(value,&m4);
        unsigned int inc5 = f5(value,&m5);
        unsigned int inc6 = f6(value,&m6);
        unsigned int inc7 = f7(value,&m7);
        if (m1) 
          inc = inc1;
         else if (m2) 
          inc = inc2;
         else if (m3) 
          inc = inc3;
         else if (m4) 
          inc = inc4;
         else if (m5) 
          inc = inc5;
         else if (m6) 
          inc = inc6;
         else 
          inc = inc7;
        if (lid == 0) 
          compressable = 0;
        compressable += inc;
        if (lid == WGS - 1) {
          cmp_size[0] += compressable;
        }
      }
    }
  }
}

int main(int argc,char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <work-group size> <repeat>\n",argv[0]);
    return 1;
  }
  const int wgs = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
// create the char buffer
  const int step = 4;
  const size_t size = ((size_t )wgs) * wgs * wgs;
  char *cbuffer = (char *)(malloc(size * step));
  srand(2);
  for (int i = 0; ((unsigned long )i) <= size * ((unsigned long )step) - 1; i += 1) {
    cbuffer[i] = (0xFF << rand() % 256);
  }
  ulong *values = convertBuffer2Array(cbuffer,size,step);
  unsigned int values_size = (size / step);
// run on the host
  unsigned int cmp_size = FPCCompress(values,values_size);
// run on the device
  unsigned int cmp_size_hw;
  bool ok = true;
// warmup
  fpc(values,&cmp_size_hw,values_size,wgs);
  auto start = std::chrono::_V2::system_clock::now();
  for (int i = 0; i <= repeat - 1; i += 1) {
    fpc(values,&cmp_size_hw,values_size,wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc failed %u != %u\n",cmp_size_hw,cmp_size);
      ok = false;
      break; 
    }
  }
  auto end = std::chrono::_V2::system_clock::now();
  auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
  printf("fpc: average device offload time %f (s)\n",(time * 1e-9f / repeat));
// warmup
  fpc2(values,&cmp_size_hw,values_size,wgs);
  start = std::chrono::_V2::system_clock::now();
  for (int i = 0; i <= repeat - 1; i += 1) {
    fpc2(values,&cmp_size_hw,values_size,wgs);
    if (cmp_size_hw != cmp_size) {
      printf("fpc2 failed %u != %u\n",cmp_size_hw,cmp_size);
      ok = false;
      break; 
    }
  }
  end = std::chrono::_V2::system_clock::now();
  time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
  printf("fpc2: average device offload time %f (s)\n",(time * 1e-9f / repeat));
  printf("%s\n",(ok?"PASS" : "FAIL"));
  free(values);
  free(cbuffer);
  return 0;
}
