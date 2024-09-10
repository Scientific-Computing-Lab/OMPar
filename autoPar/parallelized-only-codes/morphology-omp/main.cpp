#include "morphology.h"
#include <omp.h> 

void display(unsigned char *img,const int height,const int width)
{
  for (int i = 0; i <= height - 1; i += 1) {
    for (int j = 0; j <= width - 1; j += 1) {
      printf("%d ",img[i * width + j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(int argc,char *argv[])
{
  if (argc != 6) {
    printf("Usage: %s <kernel width> <kernel height> ",argv[0]);
    printf("<image width> <image height> <repeat>\n");
    return 1;
  }
  int hsize = atoi(argv[1]);
// kernel width
  int vsize = atoi(argv[2]);
// kernel height
  int width = atoi(argv[3]);
// image width
  int height = atoi(argv[4]);
// image height
  int repeat = atoi(argv[5]);
  unsigned int memSize = ((width * height) * sizeof(unsigned char ));
  unsigned char *srcImg = (unsigned char *)(malloc(memSize));
  unsigned char *tmpImg = (unsigned char *)(malloc(memSize));
  
#pragma omp parallel for private (j)
  for (int i = 0; i <= height - 1; i += 1) {
    
#pragma omp parallel for
    for (int j = 0; j <= width - 1; j += 1) {
      srcImg[i * width + j] = ((i == height / 2 - 1 && j == width / 2 - 1?255 : 0));
    }
  }
{
    double dilate_time = 0.0;
    double erode_time = 0.0;
    for (int n = 0; n <= repeat - 1; n += 1) {
      dilate_time += dilate(srcImg,tmpImg,width,height,hsize,vsize);
      erode_time += erode(srcImg,tmpImg,width,height,hsize,vsize);
    }
    printf("Average kernel execution time (dilate): %f (s)\n",dilate_time * 1e-9f / repeat);
    printf("Average kernel execution time (erode): %f (s)\n",erode_time * 1e-9f / repeat);
  }
  int s = 0;
  
#pragma omp parallel for reduction (+:s) firstprivate (memSize)
  for (unsigned int i = 0; i <= memSize - 1; i += 1) {
    s += srcImg[i];
  }
  printf("%s\n",(s == 255?"PASS" : "FAIL"));
  free(srcImg);
  free(tmpImg);
  return 0;
}
