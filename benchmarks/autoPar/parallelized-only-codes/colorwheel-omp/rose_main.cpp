#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
// Color encoding of flow vectors
// adapted from the color circle idea described at
//   http://members.shaw.ca/quadibloc/other/colint.htm
//
// Daniel Scharstein, 4/2007
#define RY  15
#define YG  6
#define GC  4
#define CB  11
#define BM  13
#define MR  6
#define MAXCOLS  (RY + YG + GC + CB + BM + MR)
#include <omp.h> 
typedef unsigned char uchar;

void setcols(int cw[55][3],int r,int g,int b,int k)
{
  cw[k][0] = r;
  cw[k][1] = g;
  cw[k][2] = b;
}

void computeColor(float fx,float fy,uchar *pix)
{
  int cw[55][3];
// color wheel
// relative lengths of color transitions:
// these are chosen based on perceptual similarity
// (e.g. one can distinguish more shades between red and yellow 
//  than between yellow and green)
  int i;
  int k = 0;
  for (i = 0; i <= 14; i += 1) {
    setcols(cw,255,255 * i / 15,0,k++);
  }
  for (i = 0; i <= 5; i += 1) {
    setcols(cw,255 - 255 * i / 6,255,0,k++);
  }
  for (i = 0; i <= 3; i += 1) {
    setcols(cw,0,255,255 * i / 4,k++);
  }
  for (i = 0; i <= 10; i += 1) {
    setcols(cw,0,255 - 255 * i / 11,255,k++);
  }
  for (i = 0; i <= 12; i += 1) {
    setcols(cw,255 * i / 13,0,255,k++);
  }
  for (i = 0; i <= 5; i += 1) {
    setcols(cw,255,0,255 - 255 * i / 6,k++);
  }
  float rad = sqrtf(fx * fx + fy * fy);
  float a = atan2f(-fy,-fx) / ((float )3.14159265358979323846);
  float fk = (a + 1.f) / 2.f * (15 + 6 + 4 + 11 + 13 + 6 - 1);
  int k0 = (int )fk;
  int k1 = (k0 + 1) % (15 + 6 + 4 + 11 + 13 + 6);
  float f = fk - k0;
  
#pragma omp parallel for firstprivate (rad,k0,k1,f)
  for (int b = 0; b <= 2; b += 1) {
    float col0 = cw[k0][b] / 255.f;
    float col1 = cw[k1][b] / 255.f;
    float col = (1.f - f) * col0 + f * col1;
    if (rad <= 1) 
      col = 1.f - rad * (1.f - col);
     else 
// increase saturation with radius
      col *= .75f;
// out of range
    pix[2 - b] = ((int )(255.f * col));
  }
}

int main(int argc,char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <range> <size> <repeat>\n",argv[0]);
    exit(1);
  }
  const float truerange = (atof(argv[1]));
  const int size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
// make picture slightly bigger to show out-of-range coding
  float range = 1.04f * truerange;
  const int half_size = size / 2;
// create a test image showing the color encoding
  size_t imgSize = (size * size * 3);
  uchar *pix = (uchar *)(malloc(imgSize));
  uchar *res = (uchar *)(malloc(imgSize));
  memset(pix,0,imgSize);
  for (int y = 0; y <= size - 1; y += 1) {
    for (int x = 0; x <= size - 1; x += 1) {
      float fx = ((float )x) / ((float )half_size) * range - range;
      float fy = ((float )y) / ((float )half_size) * range - range;
      if (x == half_size || y == half_size) 
        continue; 
// make black coordinate axes
      size_t idx = ((y * size + x) * 3);
      computeColor(fx / truerange,fy / truerange,pix + idx);
    }
  }
  printf("Start execution on a device\n");
  uchar *d_pix = (uchar *)(malloc(imgSize));
  memset(d_pix,0,imgSize);
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      for (int y = 0; y <= size - 1; y += 1) {
        for (int x = 0; x <= size - 1; x += 1) {
          float fx = ((float )x) / ((float )half_size) * range - range;
          float fy = ((float )y) / ((float )half_size) * range - range;
          if (x != half_size && y != half_size) {
            size_t idx = ((y * size + x) * 3);
            computeColor(fx / truerange,fy / truerange,d_pix + idx);
          }
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time : %f (ms)\n",(time * 1e-6f / repeat));
  }
// verify
  int fail = memcmp(pix,d_pix,imgSize);
  if (fail) {
    int max_error = 0;
    
#pragma omp parallel for reduction (max:max_error) firstprivate (imgSize)
    for (size_t i = 0; i <= imgSize - 1; i += 1) {
      int e = abs(d_pix[i] - pix[i]);
      if (e > max_error) 
        max_error = e;
    }
    printf("Maximum error between host and device results: %d\n",max_error);
  }
   else {
    printf("%s\n","PASS");
  }
  free(d_pix);
  free(pix);
  free(res);
  return 0;
}
