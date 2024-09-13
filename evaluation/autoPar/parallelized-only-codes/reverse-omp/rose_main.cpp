#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <random>
#include <chrono>
#include <omp.h>
#include <omp.h> 

int main(int argc,char *argv[])
{
  if (argc != 2) {
    printf("Usage: ./%s <iterations>\n",argv[0]);
    return 1;
  }
// specify the number of test cases
  const int iteration = atoi(argv[1]);
// number of elements to reverse
  const int len = 256;
  const int elem_size = (len * sizeof(int ));
// save device result
  int test[256];
// save expected results after performing reverse operations even/odd times
  int error = 0;
  int gold_odd[256];
  int gold_even[256];
  
#pragma omp parallel for firstprivate (len)
  for (int i = 0; i <= len - 1; i += 1) {
    gold_odd[i] = len - i - 1;
    gold_even[i] = i;
  }
  std::default_random_engine generator(123);
// bound the number of reverse operations
  class std::uniform_int_distribution< int  > distribution(100,9999);
  long time = 0;
{
    for (int i = 0; i <= iteration - 1; i += 1) {
      const int count = distribution(generator);
      memcpy(test,gold_even,elem_size);
      auto start = std::chrono::_V2::steady_clock::now();
      for (int j = 0; j <= count - 1; j += 1) {{
          int s[256];
{
            int t = omp_get_thread_num();
            s[t] = test[t];
            test[t] = s[len - t - 1];
          }
        }
      }
      auto end = std::chrono::_V2::steady_clock::now();
      time += std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
      if (count % 2 == 0) 
        error = memcmp(test,gold_even,elem_size);
       else 
        error = memcmp(test,gold_odd,elem_size);
      if (error) 
        break; 
    }
  }
  printf("Total kernel execution time: %f (s)\n",(time * 1e-9f));
  printf("%s\n",(error?"FAIL" : "PASS"));
  return 0;
}
