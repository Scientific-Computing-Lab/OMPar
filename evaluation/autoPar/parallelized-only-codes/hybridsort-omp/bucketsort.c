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
#include <omp.h>
#include "bucketsort.h"

////////////////////////////////////////////////////////////////////////////////
// Forward declarations
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints,
    float histo_width);

////////////////////////////////////////////////////////////////////////////////
// Given the input array of floats and the min and max of the distribution,
// sort the elements into float4 aligned buckets of roughly equal size
////////////////////////////////////////////////////////////////////////////////
void bucketSort(float *d_input, float *d_output, int listsize,
    int *sizes, int *nullElements, float minimum, float maximum,
    unsigned int *origOffsets)
{

  const int histosize = 1024;
  //  ////////////////////////////////////////////////////////////////////////////
  //  // First pass - Create 1024 bin histogram
  //  ////////////////////////////////////////////////////////////////////////////
  unsigned int* h_offsets = (unsigned int *) malloc(DIVISIONS * sizeof(unsigned int));
  for(int i = 0; i < DIVISIONS; i++){
    h_offsets[i] = 0;
  }
  float* pivotPoints = (float *)malloc(DIVISIONS * sizeof(float));
  int* d_indice = (int *)malloc(listsize * sizeof(int));
  float* historesult = (float *)malloc(histosize * sizeof(float));

  int blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;
  unsigned int* d_prefixoffsets = (unsigned int *)malloc(blocks*BUCKET_BLOCK_MEMORY*sizeof(int));

  size_t global = 6144;
  size_t local;

#ifdef HISTO_WG_SIZE_0
  local = HISTO_WG_SIZE_0;
#else
  local = 96;
#endif

  {
    // h_offsets is read and write
#include "kernel_histogram.h"


    for(int i=0; i<histosize; i++) 
      historesult[i] = (float)h_offsets[i];


    //  ///////////////////////////////////////////////////////////////////////////
    //  // Calculate pivot points (CPU algorithm)
    //  ///////////////////////////////////////////////////////////////////////////
    calcPivotPoints(historesult, histosize, listsize, DIVISIONS,
        minimum, maximum, pivotPoints,
        (maximum - minimum)/(float)histosize);
    //
    //  ///////////////////////////////////////////////////////////////////////////
    //  // Count the bucket sizes in new divisions
    //  ///////////////////////////////////////////////////////////////////////////

#include "kernel_bucketcount.h"

    //
    //  ///////////////////////////////////////////////////////////////////////////
    //  // Prefix scan offsets and align each division to float4 (required by
    //  // mergesort)
    //  ///////////////////////////////////////////////////////////////////////////
#ifdef BUCKET_WG_SIZE_0
    size_t localpre = BUCKET_WG_SIZE_0;
#else
    size_t localpre = 128;
#endif
    size_t globalpre = DIVISIONS;

    int size = blocks * BUCKET_BLOCK_MEMORY;

#include "kernel_bucketprefix.h"

    // copy the sizes from device to host
    origOffsets[0] = 0;
    for(int i=0; i<DIVISIONS; i++){
      origOffsets[i+1] = h_offsets[i] + origOffsets[i];
      if((h_offsets[i] % 4) != 0){
        nullElements[i] = (h_offsets[i] & ~3) + 4 - h_offsets[i];
      }
      else nullElements[i] = 0;
    }
    for(int i=0; i<DIVISIONS; i++) sizes[i] = (h_offsets[i] + nullElements[i])/4;
    for(int i=0; i<DIVISIONS; i++) {
      if((h_offsets[i] % 4) != 0)  h_offsets[i] = (h_offsets[i] & ~3) + 4;
    }
    for(int i=1; i<DIVISIONS; i++) h_offsets[i] = h_offsets[i-1] + h_offsets[i];
    for(int i=DIVISIONS - 1; i>0; i--) h_offsets[i] = h_offsets[i-1];
    h_offsets[0] = 0;

    blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1;

    // update h_offsets on the device
    // d_output is read and write
#include "kernel_bucketsort.h"
  }


  free(pivotPoints);
  free(d_indice);
  free(historesult);
  free(d_prefixoffsets);
}
////////////////////////////////////////////////////////////////////////////////
// Given a histogram of the list, figure out suitable pivotpoints that divide
// the list into approximately listsize/divisions elements each
////////////////////////////////////////////////////////////////////////////////
void calcPivotPoints(float *histogram, int histosize, int listsize,
    int divisions, float min, float max, float *pivotPoints, float histo_width)
{
  float elemsPerSlice = listsize/(float)divisions;
  float startsAt = min;
  float endsAt = min + histo_width;
  float we_need = elemsPerSlice;
  int p_idx = 0;
  for(int i=0; i<histosize; i++)
  {
    if(i == histosize - 1){
      if(!(p_idx < divisions)){
        pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      }
      break;
    }
    while(histogram[i] > we_need){
      if(!(p_idx < divisions)){
        printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions);
        exit(0);
      }
      pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
      startsAt += (we_need/histogram[i]) * histo_width;
      histogram[i] -= we_need;
      we_need = elemsPerSlice;
    }
    // grab what we can from what remains of it
    we_need -= histogram[i];

    startsAt = endsAt;
    endsAt += histo_width;
  }
  while(p_idx < divisions){
    pivotPoints[p_idx] = pivotPoints[p_idx-1];
    p_idx++;
  }
}
