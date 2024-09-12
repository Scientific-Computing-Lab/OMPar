#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#define BLOCK_SIZE 256
// A C model derived from the OpenCL kernel 
#include <omp.h> 

void softMax_cpu(const int numSlice,const int sliceSize,const float *src,float *dest)
{
  
#pragma omp parallel for private (j,j_nom_1,j_nom_2) firstprivate (numSlice,sliceSize)
  for (int i = 0; i <= numSlice - 1; i += 1) {
    float max_ = src[i * sliceSize];
    for (int j = 1; j <= sliceSize - 1; j += 1) {
      max_ = (max_ < src[i * sliceSize + j]?src[i * sliceSize + j] : max_);
    }
    float sum = 0;
    
#pragma omp parallel for reduction (+:sum) firstprivate (max_)
    for (int j = 0; j <= sliceSize - 1; j += 1) {
      float e = expf(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    
#pragma omp parallel for firstprivate (sum)
    for (int j = 0; j <= sliceSize - 1; j += 1) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

int main(int argc,char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of slices> <slice size> <repeat>\n",argv[0]);
    return 1;
  }
  int numSlice = atoi(argv[1]);
  int sliceSize = atoi(argv[2]);
  int repeat = atoi(argv[3]);
  int numElem = numSlice * sliceSize;
  float *input = (float *)(aligned_alloc(1024,sizeof(float ) * numElem));
  float *output_gpu = (float *)(aligned_alloc(1024,sizeof(float ) * numElem));
  float *output_cpu = (float *)(aligned_alloc(1024,sizeof(float ) * numElem));
  srand(2);
  for (int i = 0; i <= numSlice - 1; i += 1) {
    for (int j = 0; j <= sliceSize - 1; j += 1) {
      input[i * sliceSize + j] = (rand() % 13);
    }
  }
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      for (int i = 0; i <= numSlice - 1; i += 1) {
        float max_ = input[i * sliceSize];
        for (int j = 1; j <= sliceSize - 1; j += 1) {
          max_ = (max_ < input[i * sliceSize + j]?input[i * sliceSize + j] : max_);
        }
        float sum = 0;
        for (int j = 0; j <= sliceSize - 1; j += 1) {
          sum += expf(input[i * sliceSize + j] - max_);
        }
        for (int j = 0; j <= sliceSize - 1; j += 1) {
          output_gpu[i * sliceSize + j] = expf(input[i * sliceSize + j] - max_) / sum;
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
  }
// verification
  bool ok = true;
  softMax_cpu(numSlice,sliceSize,input,output_cpu);
  for (int i = 0; i <= numElem - 1; i += 1) {
    if ((fabsf(output_cpu[i] - output_gpu[i])) > 1e-3) {
      printf("@index %d host: %f device: %f\n",i,output_cpu[i],output_gpu[i]);
      ok = false;
      break; 
    }
  }
  printf("%s\n",(ok?"PASS" : "FAIL"));
  free(input);
  free(output_cpu);
  free(output_gpu);
  return 0;
}
