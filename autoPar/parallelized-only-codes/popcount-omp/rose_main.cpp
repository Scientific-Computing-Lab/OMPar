#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#define m1  0x5555555555555555
#define m2  0x3333333333333333 
#define m4  0x0f0f0f0f0f0f0f0f 
#define h01 0x0101010101010101
#define BLOCK_SIZE 256
// reference implementation
#include <omp.h> 

int popcount_ref(unsigned long x)
{
  int count;
  for (count = 0; x; count++) 
    x &= x - 1;
  return count;
}

void checkResults(const unsigned long *d,const int *r,const int length)
{
  int error = 0;
  for (int i = 0; i <= length - 1; i += 1) {
    if (popcount_ref(d[i]) != r[i]) {
      error = 1;
      break; 
    }
  }
  if (error) 
    printf("Fail\n");
   else 
    printf("Success\n");
}

int main(int argc,char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n",argv[0]);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  unsigned long *data = 0L;
  int *result = 0L;
  int s1 = posix_memalign((void **)(&data),1024,length * sizeof(unsigned long ));
  int s2 = posix_memalign((void **)(&result),1024,length * sizeof(int ));
  if (s1 != 0 || s2 != 0) {
    printf("Error: posix_memalign fails\n");
    if (s1 == 0) 
      free(data);
    if (s2 == 0) 
      free(result);
    return 1;
  }
// initialize input
  srand(2);
  for (int i = 0; i <= length - 1; i += 1) {
    unsigned long t = ((unsigned long )(rand())) << 32;
    data[i] = t | (rand());
  }
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      for (int i = 0; i <= length - 1; i += 1) {
        unsigned long x = data[i];
        x -= x >> 1 & 0x5555555555555555;
//put count of each 2 bits into those 2 bits
        x = (x & 0x3333333333333333) + (x >> 2 & 0x3333333333333333);
//put count of each 4 bits into those 4 bits 
        x = x + (x >> 4) & 0x0f0f0f0f0f0f0f0f;
//put count of each 8 bits into those 8 bits 
        x += x >> 8;
//put count of each 16 bits into their lowest 8 bits
        x += x >> 16;
//put count of each 32 bits into their lowest 8 bits
        x += x >> 32;
//put count of each 64 bits into their lowest 8 bits
        result[i] = (x & 0x7f);
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pc1): %f (us)\n",time * 1e-3 / repeat);
    checkResults(data,result,length);
//========================================================================================
    start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      for (int i = 0; i <= length - 1; i += 1) {
        unsigned long x = data[i];
        x -= x >> 1 & 0x5555555555555555;
//put count of each 2 bits into those 2 bits
        x = (x & 0x3333333333333333) + (x >> 2 & 0x3333333333333333);
//put count of each 4 bits into those 4 bits 
        x = x + (x >> 4) & 0x0f0f0f0f0f0f0f0f;
//put count of each 8 bits into those 8 bits 
        result[i] = (x * 0x0101010101010101 >> 56);
//returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ... 
      }
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pc2): %f (us)\n",time * 1e-3 / repeat);
    checkResults(data,result,length);
//========================================================================================
    start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      
#pragma omp parallel for
      for (int i = 0; i <= length - 1; i += 1) {
        char count;
        unsigned long x = data[i];
        for (count = 0; x; count++) 
          x &= x - 1;
        result[i] = count;
      }
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pc3): %f (us)\n",time * 1e-3 / repeat);
    checkResults(data,result,length);
//========================================================================================
    start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      for (int i = 0; i <= length - 1; i += 1) {
        unsigned long x = data[i];
        char cnt = 0;
        for (char i = 0; ((int )i) <= 63; i += 1) {
          cnt = (cnt + (x & 0x1));
          x = x >> 1;
        }
        result[i] = cnt;
      }
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pc4): %f (us)\n",time * 1e-3 / repeat);
    checkResults(data,result,length);
//========================================================================================
    start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      for (int i = 0; i <= length - 1; i += 1) {
        unsigned long x = data[i];
        const unsigned char a[256] = {(0), (1), (1), (2), (1), (2), (2), (3), (1), (2), (2), (3), (2), (3), (3), (4), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (4), (5), (5), (6), (5), (6), (6), (7), (5), (6), (6), (7), (6), (7), (7), (8)};
        const unsigned char b[256] = {(0), (1), (1), (2), (1), (2), (2), (3), (1), (2), (2), (3), (2), (3), (3), (4), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (4), (5), (5), (6), (5), (6), (6), (7), (5), (6), (6), (7), (6), (7), (7), (8)};
        const unsigned char c[256] = {(0), (1), (1), (2), (1), (2), (2), (3), (1), (2), (2), (3), (2), (3), (3), (4), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (4), (5), (5), (6), (5), (6), (6), (7), (5), (6), (6), (7), (6), (7), (7), (8)};
        const unsigned char d[256] = {(0), (1), (1), (2), (1), (2), (2), (3), (1), (2), (2), (3), (2), (3), (3), (4), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (1), (2), (2), (3), (2), (3), (3), (4), (2), (3), (3), (4), (3), (4), (4), (5), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (2), (3), (3), (4), (3), (4), (4), (5), (3), (4), (4), (5), (4), (5), (5), (6), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (3), (4), (4), (5), (4), (5), (5), (6), (4), (5), (5), (6), (5), (6), (6), (7), (4), (5), (5), (6), (5), (6), (6), (7), (5), (6), (6), (7), (6), (7), (7), (8)};
        unsigned char i1 = a[x & 0xFF];
        unsigned char i2 = a[x >> 8 & 0xFF];
        unsigned char i3 = b[x >> 16 & 0xFF];
        unsigned char i4 = b[x >> 24 & 0xFF];
        unsigned char i5 = c[x >> 32 & 0xFF];
        unsigned char i6 = c[x >> 40 & 0xFF];
        unsigned char i7 = d[x >> 48 & 0xFF];
        unsigned char i8 = d[x >> 56 & 0xFF];
        result[i] = i1 + i2 + (i3 + i4) + (i5 + i6) + (i7 + i8);
      }
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pc5): %f (us)\n",time * 1e-3 / repeat);
    checkResults(data,result,length);
//========================================================================================
    start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      for (int i = 0; i <= length - 1; i += 1) {
        result[i] = __builtin_popcountll(data[i]);
      }
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (pc6): %f (us)\n",time * 1e-3 / repeat);
    checkResults(data,result,length);
  }
  free(data);
  free(result);
  return 0;
}
