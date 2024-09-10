#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <omp.h> 
double Fresnel_Sine_Integral(double );

void reference(const double *input,double *output,const int n)
{
  for (int i = 0; i <= n - 1; i += 1) {
    output[i] = Fresnel_Sine_Integral(input[i]);
  }
}

int main(int argc,char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
// range [0, 8], interval 1e-7
  const double interval = 1e-7;
  const int points = (int )(8.0 / interval);
  const size_t points_size = points * sizeof(double );
  double *x = (double *)(malloc(points_size));
  double *output = (double *)(malloc(points_size));
  double *h_output = (double *)(malloc(points_size));
  
#pragma omp parallel for firstprivate (interval)
  for (int i = 0; i <= points - 1; i += 1) {
    x[i] = ((double )i) * interval;
  }
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      for (int i = 0; i <= points - 1; i += 1) {
        output[i] = Fresnel_Sine_Integral(x[i]);
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time %f (s)\n",(time * 1e-9f / repeat));
  }
// verify
  reference(x,h_output,points);
  bool ok = true;
  for (int i = 0; i <= points - 1; i += 1) {
    if (fabs(h_output[i] - output[i]) > 1e-6) {
      printf("%lf %lf\n",h_output[i],output[i]);
      ok = false;
      break; 
    }
  }
  printf("%s\n",(ok?"PASS" : "FAIL"));
  free(x);
  free(output);
  free(h_output);
  return 0;
}
