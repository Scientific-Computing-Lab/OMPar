#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <omp.h>
#define BLOCK_SIZE 256
#include <omp.h> 

void findMovingPixels(const size_t imgSize,const unsigned char *Img,const unsigned char *Img1,const unsigned char *Img2,const unsigned char *Tn,unsigned char *Mp)
// moving pixel map
{
  for (size_t i = 0; i <= imgSize - 1; i += 1) {
    if (abs(Img[i] - Img1[i]) > Tn[i] || abs(Img[i] - Img2[i]) > Tn[i]) 
      Mp[i] = 255;
     else 
      Mp[i] = 0;
  }
}
// alpha = 0.92 

void updateBackground(const size_t imgSize,const unsigned char *Img,const unsigned char *Mp,unsigned char *Bn)
{
  
#pragma omp parallel for firstprivate (imgSize)
  for (size_t i = 0; i <= imgSize - 1; i += 1) {
    if (Mp[i] == 0) 
      Bn[i] = (0.92f * Bn[i] + 0.08f * Img[i]);
  }
}
// alpha = 0.92, c = 3

void updateThreshold(const size_t imgSize,const unsigned char *Img,const unsigned char *Mp,const unsigned char *Bn,unsigned char *Tn)
{
  for (size_t i = 0; i <= imgSize - 1; i += 1) {
    if (Mp[i] == 0) {
      float th = 0.92f * Tn[i] + 0.24f * (Img[i] - Bn[i]);
      Tn[i] = (fmaxf(th,20.f));
    }
  }
}
//
// merge three kernels into a single kernel
//

void merge(const size_t imgSize,const unsigned char *Img,const unsigned char *Img1,const unsigned char *Img2,unsigned char *Tn,unsigned char *Bn)
{
  for (size_t i = 0; i <= imgSize - 1; i += 1) {
    if (abs(Img[i] - Img1[i]) <= Tn[i] && abs(Img[i] - Img2[i]) <= Tn[i]) {
// update background
      Bn[i] = (0.92f * Bn[i] + 0.08f * Img[i]);
// update threshold
      float th = 0.92f * Tn[i] + 0.24f * (Img[i] - Bn[i]);
      Tn[i] = (fmaxf(th,20.f));
    }
  }
}

int main(int argc,char *argv[])
{
  if (argc != 5) {
    printf("Usage: %s <image width> <image height> <merge> <repeat>\n",argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int merged = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  const int imgSize = width * height;
  const size_t imgSize_bytes = imgSize * sizeof(char );
  unsigned char *Img = (unsigned char *)(malloc(imgSize_bytes));
  unsigned char *Img1 = (unsigned char *)(malloc(imgSize_bytes));
  unsigned char *Img2 = (unsigned char *)(malloc(imgSize_bytes));
  unsigned char *Bn = (unsigned char *)(malloc(imgSize_bytes));
  unsigned char *Mp = (unsigned char *)(malloc(imgSize_bytes));
  unsigned char *Tn = (unsigned char *)(malloc(imgSize_bytes));
  std::mt19937 generator(123);
  class std::uniform_int_distribution< int  > distribute(0,255);
  for (int j = 0; j <= imgSize - 1; j += 1) {
    Bn[j] = (distribute(generator));
    Tn[j] = 128;
  }
  long time = 0;
{
    for (int i = 0; i <= repeat - 1; i += 1) {
      for (int j = 0; j <= imgSize - 1; j += 1) {
        Img[j] = (distribute(generator));
      }
// Time t   : Image   | Image1   | Image2
// Time t+1 : Image2  | Image    | Image1
// Time t+2 : Image1  | Image2   | Image
      unsigned char *t = Img2;
      Img2 = Img1;
      Img1 = Img;
      Img = t;
      if (i >= 2) {
        if (merged) {
          auto start = std::chrono::_V2::steady_clock::now();
          merge(imgSize,Img,Img1,Img2,Tn,Bn);
          auto end = std::chrono::_V2::steady_clock::now();
          time += std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
        }
         else {
          auto start = std::chrono::_V2::steady_clock::now();
          findMovingPixels(imgSize,Img,Img1,Img2,Tn,Mp);
          updateBackground(imgSize,Img,Mp,Bn);
          updateThreshold(imgSize,Img,Mp,Bn,Tn);
          auto end = std::chrono::_V2::steady_clock::now();
          time += std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
        }
      }
    }
    float kernel_time = repeat <= 2?0 : time * 1e-3f / (repeat - 2);
    printf("Average kernel execution time: %f (us)\n",kernel_time);
  }
// verification
  int sum = 0;
  int bin[4] = {(0), (0), (0), (0)};
  for (int j = 0; j <= imgSize - 1; j += 1) {
    sum += abs(Tn[j] - 128);
    if (Tn[j] < 64) 
      bin[0]++;
     else if (Tn[j] < 128) 
      bin[1]++;
     else if (Tn[j] < 192) 
      bin[2]++;
     else 
      bin[3]++;
  }
  sum = sum / imgSize;
  printf("Average threshold change is %d\n",sum);
  printf("Bin counts are %d %d %d %d\n",bin[0],bin[1],bin[2],bin[3]);
  free(Img);
  free(Img1);
  free(Img2);
  free(Tn);
  free(Bn);
  free(Mp);
  return 0;
}
