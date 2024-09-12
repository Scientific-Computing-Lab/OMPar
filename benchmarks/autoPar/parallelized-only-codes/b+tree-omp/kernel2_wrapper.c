#include <stdio.h>
#include <string.h>
#include <omp.h>
#include "../common.h"                // (in directory provided here)
#include "../util/timer/timer.h"          // (in directory provided here)
#include "./kernel2_wrapper.h"      // (in directory provided here)
//========================================================================================================================================================================================================200
//  KERNEL_GPU_CUDA_WRAPPER FUNCTION
//========================================================================================================================================================================================================200

void kernel2_wrapper(knode *knodes,long knodes_elem,long knodes_mem,
// not length in byte
int order,long maxheight,int count,long *currKnode,long *offset,long *lastKnode,long *offset_2,int *start,int *end,int *recstart,int *reclength)
{
//======================================================================================================================================================150
//  CPU VARIABLES
//======================================================================================================================================================150
// findRangeK kernel
  size_t threads;
  threads = ((order < 256?order : 256));
{
    long long kernel_start = get_time();
{
{
// private thread IDs
        int thid = omp_get_thread_num();
        int bid = omp_get_team_num();
        int i;
        for (i = 0; ((long )i) <= maxheight - 1; i += 1) {
          if (knodes[currKnode[bid]] . keys[thid] <= start[bid] && knodes[currKnode[bid]] . keys[thid + 1] > start[bid]) {
// this conditional statement is inserted to avoid crush due to but in original code
// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
            if (knodes[currKnode[bid]] . indices[thid] < knodes_elem) {
              offset[bid] = knodes[currKnode[bid]] . indices[thid];
            }
          }
          if (knodes[lastKnode[bid]] . keys[thid] <= end[bid] && knodes[lastKnode[bid]] . keys[thid + 1] > end[bid]) {
// this conditional statement is inserted to avoid crush due to but in original code
// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
            if (knodes[lastKnode[bid]] . indices[thid] < knodes_elem) {
              offset_2[bid] = knodes[lastKnode[bid]] . indices[thid];
            }
          }
// set for next tree level
          if (thid == 0) {
            currKnode[bid] = offset[bid];
            lastKnode[bid] = offset_2[bid];
          }
        }
// Find the index of the starting record
        if (knodes[currKnode[bid]] . keys[thid] == start[bid]) {
          recstart[bid] = knodes[currKnode[bid]] . indices[thid];
        }
// Find the index of the ending record
        if (knodes[lastKnode[bid]] . keys[thid] == end[bid]) {
          reclength[bid] = knodes[lastKnode[bid]] . indices[thid] - recstart[bid] + 1;
        }
      }
    }
    long long kernel_end = get_time();
    printf("Kernel execution time: %f (us)\n",((float )(kernel_end - kernel_start)));
  }
#ifdef DEBUG
#endif
}
