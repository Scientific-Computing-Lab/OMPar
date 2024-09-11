/* 
 * NMS Benchmarking Framework
 *
 * "Work-Efficient Parallel Non-Maximum Suppression Kernels"
 * Copyright (c) 2019 David Oro et al.
 * 
 * This program is free software: you can redistribute it and/or modify  
 * it under the terms of the GNU General Public License as published by  
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License 
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include <omp.h> 
typedef struct {
float x;
float y;
float z;
float w;}float4;
#define MAX_DETECTIONS  4096
#define N_PARTITIONS    32

void print_help()
{
  printf("\nUsage: nmstest  <detections.txt>  <output.txt>\n\n");
  printf("               detections.txt -> Input file containing the coordinates, width, and scores of detected objects\n");
  printf("               output.txt     -> Output file after performing NMS\n");
  printf("               repeat         -> Kernel execution count\n\n");
}
/* Gets the optimal X or Y dimension for a given thread block */

int get_optimal_dim(int val)
{
  int div;
  int neg;
  int cntneg;
  int cntpos;
/* We start figuring out if 'val' is divisible by 16 
     (e.g. optimal 16x16 thread block of maximum GPU occupancy */
  neg = 1;
  div = 16;
  cntneg = div;
  cntpos = div;
/* In order to guarantee the ending of this loop if 'val' is 
     a prime number, we limit the loop to 5 iterations */
  for (int i = 0; i <= 4; i += 1) {
    if (val % div == 0) 
      return div;
    if (neg) {
      cntneg--;
      div = cntneg;
      neg = 0;
    }
     else {
      cntpos++;
      div = cntpos;
      neg = 1;
    }
  }
  return 16;
}
/* Gets an upper limit for 'val' multiple of the 'mul' integer */

int get_upper_limit(int val,int mul)
{
  int cnt = mul;
/* The upper limit must be lower than
     the maximum allowed number of detections */
  while(cnt < val)
    cnt += mul;
  if (cnt > 4096) 
    cnt = 4096;
  return cnt;
}

int main(int argc,char *argv[])
{
  int x;
  int y;
  int w;
  float score;
  if (argc != 4) {
    print_help();
    return 0;
  }
/* Read input detection coordinates from the text file */
  int ndetections = 0;
  FILE *fp = fopen(argv[1],"r");
  if (!fp) {
    printf("Error: Unable to open file %s for input detection coordinates.\n",argv[1]);
    return - 1;
  }
/* Memory allocation in the host memory address space */
  float4 *points = (float4 *)(malloc(sizeof(float4 ) * 4096));
  if (!points) {
    printf("Error: Unable to allocate CPU memory.\n");
    return - 1;
  }
  memset(points,0,sizeof(float4 ) * 4096);
  while(!(feof(fp))){
    int cnt = fscanf(fp,"%d,%d,%d,%f\n",&x,&y,&w,&score);
    if (cnt != 4) {
      printf("Error: Invalid file format in line %d when reading %s\n",ndetections,argv[1]);
      return - 1;
    }
    points[ndetections] . x = ((float )x);
// x coordinate
    points[ndetections] . y = ((float )y);
// y coordinate
    points[ndetections] . z = ((float )w);
// window dimensions
    points[ndetections] . w = score;
// score
    ndetections++;
  }
  printf("Number of detections read from input file (%s): %d\n",argv[1],ndetections);
  fclose(fp);
/* CPU array for storing the detection bitmap */
  unsigned char *pointsbitmap = (unsigned char *)(malloc(sizeof(unsigned char ) * 4096));
  memset(pointsbitmap,0,sizeof(unsigned char ) * 4096);
  unsigned char *nmsbitmap = (unsigned char *)(malloc(sizeof(unsigned char ) * 4096 * 4096));
  memset(nmsbitmap,1,sizeof(unsigned char ) * 4096 * 4096);
/* We build up the non-maximum supression bitmap matrix by removing overlapping windows */
  const int repeat = atoi(argv[3]);
  const int limit = get_upper_limit(ndetections,16);
  const int threads = get_optimal_dim(limit) * get_optimal_dim(limit);
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; n <= repeat - 1; n += 1) {
      
#pragma omp parallel for private (j)
      for (int i = 0; i <= limit - 1; i += 1) {
        
#pragma omp parallel for
        for (int j = 0; j <= limit - 1; j += 1) {
          if (points[i] . w < points[j] . w) {
            float area = (points[j] . z + 1.0f) * (points[j] . z + 1.0f);
            float w = fmaxf(0.0f,fminf(points[i] . x + points[i] . z,points[j] . x + points[j] . z) - fmaxf(points[i] . x,points[j] . x) + 1.0f);
            float h = fmaxf(0.0f,fminf(points[i] . y + points[i] . z,points[j] . y + points[j] . z) - fmaxf(points[i] . y,points[j] . y) + 1.0f);
            nmsbitmap[i * 4096 + j] = (w * h / area < 0.3f && points[j] . z != 0);
          }
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (generate_nms_bitmap): %f (s)\n",(time * 1e-9f / repeat));
    start = std::chrono::_V2::steady_clock::now();
/* Then we perform a reduction for generating a point bitmap vector */
    for (int n = 0; n <= repeat - 1; n += 1) {{
        #ifdef BYTE_ATOMIC
        #else
        unsigned int s;
        #endif
{
          int bid = omp_get_team_num();
          int lid = omp_get_thread_num();
          int idx = bid * 4096 + lid;
          if (lid == 0) 
            s = 1;
          s &= ((unsigned int )nmsbitmap[idx]);
            #ifndef BYTE_ATOMIC
            #endif
          for (int i = 0; i <= 30; i += 1) {
            idx += 4096 / 32;
            s &= nmsbitmap[idx];
          }
          pointsbitmap[bid] = s;
        }
      }
    }
    end = std::chrono::_V2::steady_clock::now();
    time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time (reduce_nms_bitmap): %f (s)\n",(time * 1e-9f / repeat));
  }
  fp = fopen(argv[2],"w");
  if (!fp) {
    printf("Error: Unable to open file %s for detection outcome.\n",argv[2]);
    return - 1;
  }
  int totaldets = 0;
  for (int i = 0; i <= ndetections - 1; i += 1) {
    if (pointsbitmap[i]) {
      x = ((int )points[i] . x);
// x coordinate
      y = ((int )points[i] . y);
// y coordinate
      w = ((int )points[i] . z);
// window dimensions
      score = points[i] . w;
// score
      fprintf(fp,"%d,%d,%d,%f\n",x,y,w,score);
      totaldets++;
    }
  }
  fclose(fp);
  printf("Detections after NMS: %d\n",totaldets);
  free(points);
  free(pointsbitmap);
  free(nmsbitmap);
  return 0;
}
