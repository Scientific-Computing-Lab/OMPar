/* 
   FSM_GA is a GPU-accelerated implementation of a genetic algorithm
   (GA) for finding well-performing finite-state machines (FSM) for predicting
   binary sequences.
   Copyright (c) 2013, Texas State University. All rights reserved.
   Redistribution and use in source and binary forms, with or without modification,
   are permitted for academic, research, experimental, or personal use provided
   that the following conditions are met:
 * Redistributions of source code must retain the above copyright notice,
 this list of conditions, and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
 this list of conditions, and the following disclaimer in the documentation
 and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from this
 software without specific prior written permission.
 For all other uses, please contact the Office for Commercialization and Industry
 Relations at Texas State University <http://www.txstate.edu/ocir/>.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
Authors: Martin Burtscher
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>
#include "parameters.h"
#include "kernels.h"
#include <omp.h> 

int main(int argc,char *argv[])
{
  if (argc != 2) {
    fprintf(stderr,"usage: %s trace_length\n",argv[0]);
    exit(- 1);
  }
  int length = atoi(argv[1]);
  ((bool )(sizeof(unsigned short ) == 2))?((void )0) : __assert_fail("sizeof(unsigned short) == 2","main.cpp",52,__PRETTY_FUNCTION__);
  ((bool )(0 < length))?((void )0) : __assert_fail("0 < length","main.cpp",53,__PRETTY_FUNCTION__);
  ((bool )((8 & 8 - 1) == 0))?((void )0) : __assert_fail("(FSMSIZE & (FSMSIZE - 1)) == 0","main.cpp",54,__PRETTY_FUNCTION__);
  ((bool )((32768 & 32768 - 1) == 0))?((void )0) : __assert_fail("(TABSIZE & (TABSIZE - 1)) == 0","main.cpp",55,__PRETTY_FUNCTION__);
  ((bool )(0 < 8 && 8 <= 256))?((void )0) : __assert_fail("(0 < FSMSIZE) && (FSMSIZE <= 256)","main.cpp",56,__PRETTY_FUNCTION__);
  ((bool )(0 < 32768 && 32768 <= 32768))?((void )0) : __assert_fail("(0 < TABSIZE) && (TABSIZE <= 32768)","main.cpp",57,__PRETTY_FUNCTION__);
  ((bool )(0 < 1024))?((void )0) : __assert_fail("0 < POPCNT","main.cpp",58,__PRETTY_FUNCTION__);
  ((bool )(0 < 256 && 256 <= 1024))?((void )0) : __assert_fail("(0 < POPSIZE) && (POPSIZE <= 1024)","main.cpp",59,__PRETTY_FUNCTION__);
  ((bool )(0 < 1))?((void )0) : __assert_fail("0 < CUTOFF","main.cpp",60,__PRETTY_FUNCTION__);
  int i;
  int j;
  int d;
  int s;
  int bit;
  int pc;
  int misses;
  int besthits;
  int generations;
  unsigned short *data;
  unsigned char state[32768];
  unsigned char fsm[16];
  int best[19];
  int trans[8][2];
  double runtime;
  struct timeval starttime;
  struct timeval endtime;
  data = ((unsigned short *)(malloc(sizeof(unsigned short ) * length)));
  srand(123);
  for (int i = 0; i <= length - 1; i += 1) {
    data[i] = (rand());
  }
  printf("%d\t#kernel execution times\n",10);
  printf("%d\t#fsm size\n",8);
  printf("%d\t#entries\n",length);
  printf("%d\t#tab size\n",32768);
  printf("%d\t#blocks\n",1024);
  printf("%d\t#threads\n",256);
  printf("%d\t#cutoff\n",1);
  unsigned int *rndstate = (unsigned int *)(malloc((1024 * 256) * sizeof(unsigned int )));
  unsigned char *bfsm = (unsigned char *)(malloc((1024 * 8 * 2) * sizeof(unsigned char )));
  unsigned char *same = (unsigned char *)(malloc(1024 * sizeof(unsigned char )));
  int *smax = (int *)(malloc(1024 * sizeof(int )));
  int *sbest = (int *)(malloc(1024 * sizeof(int )));
  int *oldmax = (int *)(malloc(1024 * sizeof(int )));
{
    gettimeofday(&starttime,0L);
    for (int i = 0; i <= 9; i += 1) {
      
#pragma omp parallel for
      for (int i = 0; i <= 18; i += 1) {
        best[i] = 0;
      }
      FSMKernel(length,data,best,rndstate,bfsm,same,smax,sbest,oldmax);
      MaxKernel(best,bfsm);
    }
    gettimeofday(&endtime,0L);
    runtime = endtime . tv_sec + endtime . tv_usec / 1000000.0 - starttime . tv_sec - starttime . tv_usec / 1000000.0;
    printf("%.6lf\t#runtime [s]\n",runtime / 10);
  }
  besthits = best[1];
  generations = best[2];
  printf("%.6lf\t#throughput [Gtr/s]\n",0.000000001 * 256 * generations * length / (runtime / 10));
// evaluate saturating up/down counter
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 7; i += 1) {
    fsm[i * 2 + 0] = (i - 1);
    fsm[i * 2 + 1] = (i + 1);
  }
  fsm[0] = 0;
  fsm[(8 - 1) * 2 + 1] = (8 - 1);
  memset(state,0,32768);
  misses = 0;
  for (i = 0; i <= length - 1; i += 1) {
    d = ((int )data[i]);
    pc = d >> 1 & 32768 - 1;
    bit = d & 1;
    s = ((int )state[pc]);
    misses += bit ^ (s + s) / 8 & 1;
    state[pc] = fsm[s + s + bit];
  }
  printf("%d\t#sudcnt hits\n",length - misses);
  printf("%d\t#GAfsm hits\n",besthits);
  printf("%.3lf%%\t#sudcnt hits\n",100.0 * (length - misses) / length);
  printf("%.3lf%%\t#GAfsm hits\n\n",100.0 * besthits / length);
// verify result and count transitions
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= 7; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= 1; j += 1) {
      trans[i][j] = 0;
    }
  }
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 15; i += 1) {
    fsm[i] = best[i + 3];
  }
  memset(state,0,32768);
  misses = 0;
  for (i = 0; i <= length - 1; i += 1) {
    d = ((int )data[i]);
    pc = d >> 1 & 32768 - 1;
    bit = d & 1;
    s = ((int )state[pc]);
    trans[s][bit]++;
    misses += bit ^ s & 1;
    state[pc] = ((unsigned char )fsm[s + s + bit]);
  }
  bool ok = length - misses == besthits;
  printf("%s\n",(ok?"PASS" : "FAIL"));
#ifdef DEBUG
// print FSM state assignment in R's ncol format
#endif
  free(data);
  free(rndstate);
  free(bfsm);
  free(same);
  free(smax);
  free(sbest);
  free(oldmax);
  return 0;
}
