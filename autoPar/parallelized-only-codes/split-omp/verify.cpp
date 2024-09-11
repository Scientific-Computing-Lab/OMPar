#include <string.h>
#include <omp.h> 

bool verify(const unsigned int *sorted_keys,const unsigned int *keys,const unsigned int threads,const int N)
{
  unsigned int m1[16];
  unsigned int m2[16];
  int n = (threads * 4);
// n elements are expected to be sorted
  
#pragma omp parallel for private (j) firstprivate (n)
  for (int i = 0; i <= N - 1; i = i + n) {
    
#pragma omp parallel for
    for (int j = 0; j <= n - 1 - 1; j += 1) {
      if (sorted_keys[i + j] > sorted_keys[i + j + 1]) 
        return false;
    }
  }
  
#pragma omp parallel for
  for (int i = 0; i <= N - 1; i += 1) {
    if (sorted_keys[i] >= 16) 
      return false;
  }
  for (int i = 0; i <= N - 1; i = i + n) {
    memset(m1,0,64);
    memset(m2,0,64);
    for (int j = 0; j <= n - 1; j += 1) {
      m1[keys[i + j]]++;
      m2[sorted_keys[i + j]]++;
    }
    if ((memcmp(m1,m2,64))) 
      return false;
  }
  return true;
}
