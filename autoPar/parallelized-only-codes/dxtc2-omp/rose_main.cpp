/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
// *********************************************************************
// Demo application for realtime DXT1 compression based on the OpenCL
// DXTC sample
// *********************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <chrono>
#include <omp.h>
#include "dds.h"
#include "permutations.h"
#include "block.h"
#include "shrUtils.h"
#define ERROR_THRESHOLD 0.02f
#define NUM_THREADS     64      // Number of threads per work group.
#include <omp.h> 
typedef struct {
float x;
float y;
float z;
float w;}float4;
typedef struct {
uint x;
uint y;
uint z;
uint w;}uint4;
#include "kernel.cpp"
// Main function
// *********************************************************************

int main(int argc,char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <path to image> <path to reference image> <repeat>\n",argv[0]);
    return 1;
  }
  const char *image_path = argv[1];
  ((bool )(image_path != 0L))?((void )0) : __assert_fail("image_path != NULL","main.cpp",44,__PRETTY_FUNCTION__);
  const char *reference_image_path = argv[2];
  ((bool )(reference_image_path != 0L))?((void )0) : __assert_fail("reference_image_path != NULL","main.cpp",47,__PRETTY_FUNCTION__);
  const int numIterations = atoi(argv[3]);
  unsigned int width;
  unsigned int height;
  unsigned int *h_img = 0L;
  const float alphaTable4[4] = {(9.0f), (0.0f), (6.0f), (3.0f)};
  const float alphaTable3[4] = {(4.0f), (0.0f), (2.0f), (2.0f)};
  const int prods4[4] = {(0x090000), (0x000900), (0x040102), (0x010402)};
  const int prods3[4] = {(0x040000), (0x000400), (0x040101), (0x010401)};
// load image 
  shrLoadPPM4ub(image_path,(unsigned char **)(&h_img),&width,&height);
  ((bool )(h_img != 0L))?((void )0) : __assert_fail("h_img != NULL","main.cpp",60,__PRETTY_FUNCTION__);
  printf("Loaded '%s', %d x %d pixels\n\n",image_path,width,height);
// Convert linear image to block linear. 
  const unsigned int memSize = width * height;
  const unsigned int memSizeByte = (memSize * sizeof(unsigned int ));
  unsigned int *block_image = (unsigned int *)(malloc(memSizeByte));
// Convert linear image to block linear. 
  for (unsigned int by = 0; by <= height / ((unsigned int )4) - 1; by += 1) {
    
#pragma omp parallel for private (i)
    for (unsigned int bx = 0; bx <= width / ((unsigned int )4) - 1; bx += 1) {
      
#pragma omp parallel for
      for (int i = 0; i <= 15; i += 1) {
        const int x = i & 3;
        const int y = i / 4;
        block_image[(by * width / 4 + bx) * 16 + i] = ((unsigned int *)h_img)[(by * 4 + y) * 4 * (width / 4) + bx * 4 + x];
      }
    }
  }
// Compute permutations.
  unsigned int permutations[1024];
  computePermutations(permutations);
  const unsigned int compressedSize = width / 4 * (height / 4) * 8;
  unsigned int *h_result = (unsigned int *)(malloc(compressedSize));
{
// Determine launch configuration and run timed computation numIterations times
    int blocks = ((width + 3) / 4 * ((height + 3) / 4));
// rounds up by 1 block in each dim if %4 != 0
// Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
    unsigned int compute_units = 24;
    int blocksPerLaunch = blocks < 768 * ((int )compute_units)?blocks : 768 * ((int )compute_units);
    printf("\nRunning DXT Compression on %u x %u image...\n",width,height);
    printf("\n%u Workgroups, %u Work Items per Workgroup, %u Work Items in NDRange...\n\n",blocks,64,blocks * 64);
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= numIterations - 1; i += 1) {
      for (int j = 0; j <= blocks - 1; j += blocksPerLaunch) {
        int grid = blocksPerLaunch < blocks - j?blocksPerLaunch : blocks - j;
{
          float4 colors[16];
          float4 sums[16];
          int s_int[64];
          float s_float[96];
          unsigned int s_permutations[160];
          int xrefs[16];
{
            const int idx = omp_get_thread_num();
            loadColorBlock(block_image,colors,sums,xrefs,s_float,j);
            uint4 best = evalAllPermutations(colors,permutations,s_float,sums[0],s_permutations,alphaTable4,prods4,alphaTable3,prods3);
// Use a parallel reduction to find minimum error.
            const int minIdx = findMinError(s_float,s_int);
// Only write the result of the winner thread.
            if (idx == minIdx) {
              saveBlockDXT1(best . x,best . y,best . z,xrefs,h_result,j);
            }
          }
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time %f (s)\n",(time * 1e-9f / numIterations));
  }
// Write DDS file.
  FILE *fp = 0L;
  char output_filename[1024];
#ifdef WIN32
#else
  strcpy(output_filename,image_path);
  strcpy(output_filename + strlen(image_path) - 3,"dds");
  fp = fopen(output_filename,"wb");
#endif
  ((bool )(fp != 0L))?((void )0) : __assert_fail("fp != NULL","main.cpp",150,__PRETTY_FUNCTION__);
  struct DDSHeader header;
  header . fourcc = FOURCC_DDS;
  header . size = 124;
  header . flags = DDSD_WIDTH | DDSD_HEIGHT | DDSD_CAPS | DDSD_PIXELFORMAT | DDSD_LINEARSIZE;
  header . height = height;
  header . width = width;
  header . pitch = compressedSize;
  header . depth = 0;
  header . mipmapcount = 0;
  memset(header . reserved,0,sizeof(header . reserved));
  header . pf . size = 32;
  header . pf . flags = DDPF_FOURCC;
  header . pf . fourcc = FOURCC_DXT1;
  header . pf . bitcount = 0;
  header . pf . rmask = 0;
  header . pf . gmask = 0;
  header . pf . bmask = 0;
  header . pf . amask = 0;
  header . caps . caps1 = DDSCAPS_TEXTURE;
  header . caps . caps2 = 0;
  header . caps . caps3 = 0;
  header . caps . caps4 = 0;
  header . notused = 0;
  fwrite((&header),sizeof(struct DDSHeader ),1,fp);
  fwrite(h_result,compressedSize,1,fp);
  fclose(fp);
// Make sure the generated image matches the reference image (regression check)
  printf("\nComparing against Host/C++ computation...\n");
// read in the reference image from file
#ifdef WIN32
#else
  fp = fopen(reference_image_path,"rb");
#endif
  ((bool )(fp != 0L))?((void )0) : __assert_fail("fp != NULL","main.cpp",190,__PRETTY_FUNCTION__);
  fseek(fp,(sizeof(struct DDSHeader )),0);
  unsigned int referenceSize = width / 4 * (height / 4) * 8;
  unsigned int *reference = (unsigned int *)(malloc(referenceSize));
  fread(reference,referenceSize,1,fp);
  fclose(fp);
// compare the reference image data to the sample/generated image
  float rms = 0;
  for (unsigned int y = 0; y <= height - 1; y += 4) {
    for (unsigned int x = 0; x <= width - 1; x += 4) {
// binary comparison of data
      unsigned int referenceBlockIdx = y / 4 * (width / 4) + x / 4;
      unsigned int resultBlockIdx = y / 4 * (width / 4) + x / 4;
      int cmp = compareBlock((((struct BlockDXT1 *)h_result) + resultBlockIdx),(((struct BlockDXT1 *)reference) + referenceBlockIdx));
// log deviations, if any
      if (cmp != 0.0f) {
        compareBlock((((struct BlockDXT1 *)h_result) + resultBlockIdx),(((struct BlockDXT1 *)reference) + referenceBlockIdx));
//printf("Deviation at (%d, %d):\t%f rms\n", x/4, y/4, float(cmp)/16/3);
      }
      rms += cmp;
    }
  }
  rms /= (width * height * 3);
  printf("RMS(reference, result) = %f\n\n",rms);
// Free host memory
  free(block_image);
  free(h_result);
  free(h_img);
  free(reference);
// finish
  if (rms <= 0.02f) 
    printf("PASS\n");
   else 
    printf("FAIL\n");
  return 0;
}
