/*
 *  Extension to the interleaving example in CUDA Programming by Shane Cook
 */
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#define NUM_ELEMENTS 4096
#define COUNT 4096       // accumulation count
// an interleaved type
#include <omp.h> 
typedef struct {
unsigned int s0;
unsigned int s1;
unsigned int s2;
unsigned int s3;
unsigned int s4;
unsigned int s5;
unsigned int s6;
unsigned int s7;
unsigned int s8;
unsigned int s9;
unsigned int sa;
unsigned int sb;
unsigned int sc;
unsigned int sd;
unsigned int se;
unsigned int sf;}INTERLEAVED_T;
// Define an array type based on the interleaved structure
typedef INTERLEAVED_T INTERLEAVED_ARRAY_T[4096];
// Alternative - structure of arrays
typedef unsigned int ARRAY_MEMBER_T[4096];
typedef struct {
ARRAY_MEMBER_T s0;
ARRAY_MEMBER_T s1;
ARRAY_MEMBER_T s2;
ARRAY_MEMBER_T s3;
ARRAY_MEMBER_T s4;
ARRAY_MEMBER_T s5;
ARRAY_MEMBER_T s6;
ARRAY_MEMBER_T s7;
ARRAY_MEMBER_T s8;
ARRAY_MEMBER_T s9;
ARRAY_MEMBER_T sa;
ARRAY_MEMBER_T sb;
ARRAY_MEMBER_T sc;
ARRAY_MEMBER_T sd;
ARRAY_MEMBER_T se;
ARRAY_MEMBER_T sf;}NON_INTERLEAVED_T;
// data initialization and verification
#include "util.cpp"

void add_test_interleaved(INTERLEAVED_T *const h_dst,const INTERLEAVED_T *const h_src,const unsigned int repeat,const unsigned int num_elements)
{
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; ((unsigned int )n) <= repeat - 1; n += 1) {
      
#pragma omp parallel for private (i)
      for (unsigned int tid = 0; tid <= num_elements - 1; tid += 1) {
        for (unsigned int i = 0; i <= ((unsigned int )4096) - 1; i += 1) {
          h_dst[tid] . s0 += h_src[tid] . s0;
          h_dst[tid] . s1 += h_src[tid] . s1;
          h_dst[tid] . s2 += h_src[tid] . s2;
          h_dst[tid] . s3 += h_src[tid] . s3;
          h_dst[tid] . s4 += h_src[tid] . s4;
          h_dst[tid] . s5 += h_src[tid] . s5;
          h_dst[tid] . s6 += h_src[tid] . s6;
          h_dst[tid] . s7 += h_src[tid] . s7;
          h_dst[tid] . s8 += h_src[tid] . s8;
          h_dst[tid] . s9 += h_src[tid] . s9;
          h_dst[tid] . sa += h_src[tid] . sa;
          h_dst[tid] . sb += h_src[tid] . sb;
          h_dst[tid] . sc += h_src[tid] . sc;
          h_dst[tid] . sd += h_src[tid] . sd;
          h_dst[tid] . se += h_src[tid] . se;
          h_dst[tid] . sf += h_src[tid] . sf;
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel (interleaved) execution time %f (s)\n",(time * 1e-9f / repeat));
  }
}

void add_test_non_interleaved(NON_INTERLEAVED_T *const h_dst,const NON_INTERLEAVED_T *const h_src,const unsigned int repeat,const unsigned int num_elements)
{
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int n = 0; ((unsigned int )n) <= repeat - 1; n += 1) {
      
#pragma omp parallel for private (i)
      for (unsigned int tid = 0; tid <= num_elements - 1; tid += 1) {
        for (unsigned int i = 0; i <= ((unsigned int )4096) - 1; i += 1) {
          h_dst -> s0[tid] += h_src -> s0[tid];
          h_dst -> s1[tid] += h_src -> s1[tid];
          h_dst -> s2[tid] += h_src -> s2[tid];
          h_dst -> s3[tid] += h_src -> s3[tid];
          h_dst -> s4[tid] += h_src -> s4[tid];
          h_dst -> s5[tid] += h_src -> s5[tid];
          h_dst -> s6[tid] += h_src -> s6[tid];
          h_dst -> s7[tid] += h_src -> s7[tid];
          h_dst -> s8[tid] += h_src -> s8[tid];
          h_dst -> s9[tid] += h_src -> s9[tid];
          h_dst -> sa[tid] += h_src -> sa[tid];
          h_dst -> sb[tid] += h_src -> sb[tid];
          h_dst -> sc[tid] += h_src -> sc[tid];
          h_dst -> sd[tid] += h_src -> sd[tid];
          h_dst -> se[tid] += h_src -> se[tid];
          h_dst -> sf[tid] += h_src -> sf[tid];
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel (non-interleaved) execution time %f (s)\n",(time * 1e-9f / repeat));
  }
}

int main(int argc,char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  NON_INTERLEAVED_T non_interleaved_src;
  NON_INTERLEAVED_T non_interleaved_dst;
  INTERLEAVED_ARRAY_T interleaved_src;
  INTERLEAVED_ARRAY_T interleaved_dst;
  initialize(interleaved_src,interleaved_dst,non_interleaved_src,non_interleaved_dst,4096);
  add_test_non_interleaved(&non_interleaved_dst,(&non_interleaved_src),repeat,4096);
  add_test_interleaved(interleaved_dst,interleaved_src,repeat,4096);
  verify(interleaved_dst,non_interleaved_dst,4096);
  return 0;
}
