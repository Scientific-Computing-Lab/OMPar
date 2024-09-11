////////////////////////////////////////////////////////////////////////////////
// Includes
////////////////////////////////////////////////////////////////////////////////
#include <fcntl.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h>
#include <stdbool.h>
#include "mergesort.h"
////////////////////////////////////////////////////////////////////////////////
// Defines
////////////////////////////////////////////////////////////////////////////////
#define BLOCKSIZE  256
#define ROW_LENGTH  BLOCKSIZE * 4
#define ROWS    4096
////////////////////////////////////////////////////////////////////////////////
// The mergesort algorithm
////////////////////////////////////////////////////////////////////////////////
// Codeplay 
//  error: no viable conversion from 'vec<typename
//        detail::vec_ops::logical_return<sizeof(float)>::type, 1>'
//              (aka 'vec<int, 1>') to 'bool'
//                b.z() = a.y() >= b.z() ? a.y() : b.z();
#include <omp.h> 

float4 sortElem(float4 r)
{
  float4 nr;
  float xt = r . x;
  float yt = r . y;
  float zt = r . z;
  float wt = r . w;
  float nr_xt = xt > yt?yt : xt;
  float nr_yt = yt > xt?yt : xt;
  float nr_zt = zt > wt?wt : zt;
  float nr_wt = wt > zt?wt : zt;
  xt = (nr_xt > nr_zt?nr_zt : nr_xt);
  yt = (nr_yt > nr_wt?nr_wt : nr_yt);
  zt = (nr_zt > nr_xt?nr_zt : nr_xt);
  wt = (nr_wt > nr_yt?nr_wt : nr_yt);
  nr . x = xt;
  nr . y = (yt > zt?zt : yt);
  nr . z = (zt > yt?zt : yt);
  nr . w = wt;
  return nr;
}

float4 getLowest(float4 a,float4 b)
{
  float ax = a . x;
  float ay = a . y;
  float az = a . z;
  float aw = a . w;
  float bx = b . x;
  float by = b . y;
  float bz = b . z;
  float bw = b . w;
  a . x = (ax < bw?ax : bw);
  a . y = (ay < bz?ay : bz);
  a . z = (az < by?az : by);
  a . w = (aw < bx?aw : bx);
  return a;
}

float4 getHighest(float4 a,float4 b)
{
  float ax = a . x;
  float ay = a . y;
  float az = a . z;
  float aw = a . w;
  float bx = b . x;
  float by = b . y;
  float bz = b . z;
  float bw = b . w;
  b . x = (aw >= bx?aw : bx);
  b . y = (az >= by?az : by);
  b . z = (ay >= bz?ay : bz);
  b . w = (ax >= bw?ax : bw);
  return b;
}

float4 *runMergeSort(int listsize,int divisions,float4 *d_origList,float4 *d_resultList,int *sizes,int *nullElements,unsigned int *origOffsets)
{
  int *startaddr = (int *)(malloc((divisions + 1) * sizeof(int )));
  int largestSize = - 1;
  startaddr[0] = 0;
  for (int i = 1; i <= divisions; i += 1) {
    startaddr[i] = startaddr[i - 1] + sizes[i - 1];
    if (sizes[i - 1] > largestSize) 
      largestSize = sizes[i - 1];
  }
  largestSize *= 4;
#ifdef MERGE_WG_SIZE_0
#else
  const int THREADS = 256;
#endif
  size_t local[] = {(THREADS), (1), (1)};
  size_t blocks = (listsize / 4 % THREADS == 0?listsize / 4 / THREADS : listsize / 4 / THREADS + 1);
  size_t global[] = {blocks * THREADS, (1), (1)};
  size_t grid[3];
//buffer<int, 1> d_constStartAddr (startaddr, (divisions+1), props);
{
    for (int i = 0; i <= listsize / 4 - 1; i += 1) {
      d_resultList[i] = sortElem(d_origList[i]);
    }
#ifdef DEBUG
#endif
//double mergePassTime = 0;
    int nrElems = 2;
    while(1){
      int floatsperthread = nrElems * 4;
//printf("FPT %d \n", floatsperthread);
      int threadsPerDiv = (int )(ceil((largestSize / ((float )floatsperthread))));
//printf("TPD %d \n",threadsPerDiv);
      int threadsNeeded = threadsPerDiv * divisions;
//printf("TN %d \n", threadsNeeded);
#ifdef MERGE_WG_SIZE_1
#else
      local[0] = 208;
#endif
      grid[0] = (threadsNeeded % local[0] == 0?threadsNeeded / local[0] : threadsNeeded / local[0] + 1);
      if (grid[0] < 8) {
        grid[0] = 8;
        local[0] = (threadsNeeded % grid[0] == 0?threadsNeeded / grid[0] : threadsNeeded / grid[0] + 1);
      }
// Swap orig/result list
      float4 *tempList = d_origList;
      d_origList = d_resultList;
      d_resultList = tempList;
// update the workload size
      global[0] = grid[0] * local[0];
#include "kernel_mergeSortPass.h"
      nrElems *= 2;
      floatsperthread = nrElems * 4;
      if (threadsPerDiv == 1) 
        break; 
    }
#ifdef DEBUG
#endif
// buffer<unsigned int, 1> finalStartAddr(origOffsets, divisions+1, props);
// buffer<int, 1> nullElems(nullElements, divisions, props);
// buffer<float,1> d_res ((float*)d_origList, listsize, props);
// buffer<float,1> d_orig ((float*)d_resultList, listsize, props);
    float *d_orig = (float *)d_origList;
    float *d_res = (float *)d_resultList;
//#pragma omp target data map(to: d_res[0:listsize]) map(from: d_orig[0:listsize])
    for (int division = 0; division <= divisions - 1; division += 1) {
      for (int idx = 0; idx <= largestSize - 1; idx += 1) {
        if (origOffsets[division] + idx < origOffsets[division + 1]) 
          d_orig[origOffsets[division] + idx] = d_res[startaddr[division] * 4 + nullElements[division] + idx];
      }
    }
  }
  free(startaddr);
  return d_origList;
}
