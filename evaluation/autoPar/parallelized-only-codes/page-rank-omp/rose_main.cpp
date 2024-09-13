/*
 ** The MIT License (MIT)
 **
 ** Copyright (c) 2014, Erick Lavoie, Faiz Khan, Sujay Kathrotia, Vincent
 ** Foley-Bourgon, Laurie Hendren
 **
 ** Permission is hereby granted, free of charge, to any person obtaining a copy
 **of this software and associated documentation files (the "Software"), to deal
 ** in the Software without restriction, including without limitation the rights
 ** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 ** copies of the Software, and to permit persons to whom the Software is
 ** furnished to do so, subject to the following conditions:
 **
 ** The above copyright notice and this permission notice shall be included in all
 ** copies or substantial portions of the Software.
 **
 ** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 ** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 ** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 ** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 ** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 ** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 ** SOFTWARE.
 **
 **/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <chrono>
#define D_FACTOR (0.85f)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
// default values 
#include <omp.h> 
const int max_iter = 1000;
const float threshold = 1e-16f;
// generates an array of random pages and their links

int *random_pages(int n,unsigned int *noutlinks,int divisor)
{
  int i;
  int j;
  int k;
  int *pages = (int *)(malloc(sizeof(int ) * n * n));
// matrix 1 means link from j->i
  if (divisor <= 0) {
    fprintf(stderr,"ERROR: Invalid divisor '%d' for random initialization, divisor should be greater or equal to 1\n",divisor);
    exit(1);
  }
  for (i = 0; i <= n - 1; i += 1) {
    noutlinks[i] = 0;
    for (j = 0; j <= n - 1; j += 1) {
      if (i != j && abs((rand())) % divisor == 0) {
        pages[i * n + j] = 1;
        noutlinks[i] += 1;
      }
    }
// the case with no outlinks is avoided
    if (noutlinks[i] == 0) {
      do {
        k = abs((rand())) % n;
      }while (k == i);
      pages[i * n + k] = 1;
      noutlinks[i] = 1;
    }
  }
  return pages;
}

void init_array(float *a,int n,float val)
{
  int i;
  
#pragma omp parallel for private (i) firstprivate (n,val)
  for (i = 0; i <= n - 1; i += 1) {
    a[i] = val;
  }
}

void usage(char *argv[])
{
  fprintf(stderr,"Usage: %s [-n number of pages] [-i max iterations] [-t threshold] [-q divsor for zero density]\n",argv[0]);
}
static struct option size_opts[] = {
/* name, has_tag, flag, val*/
/* Need explicit braces: is this where we insert the class name? */ {("number of pages"), (1), (0L), ('n')}, /* Need explicit braces: is this where we insert the class name? */ {("max number of iterations"), (1), (0L), ('i')}, /* Need explicit braces: is this where we insert the class name? */ {("minimum threshold"), (1), (0L), ('t')}, /* Need explicit braces: is this where we insert the class name? */ {("divisor for zero density"), (1), (0L), ('q')}, /* Need explicit braces: is this where we insert the class name? */ {(0), (0), (0)}};

float maximum_dif(float *difs,int n)
{
  int i;
  float max = 0.0f;
  for (i = 0; i <= n - 1; i += 1) {
    max = (difs[i] > max?difs[i] : max);
  }
  return max;
}

int main(int argc,char *argv[])
{
  int *pages;
  float *maps;
  float *page_ranks;
  unsigned int *noutlinks;
  int t;
  float max_diff;
  int i = 0;
  int j;
  int n = 1000;
  int iter = max_iter;
  float thresh = threshold;
  int divisor = 2;
  int nb_links = 0;
  int opt;
  int opt_index = 0;
  while((opt = getopt_long(argc,argv,"::n:i:t:q:",size_opts,&opt_index)) != - 1){
    switch(opt){
      case 'n':
      n = atoi(optarg);
      break; 
      case 'i':
      iter = atoi(optarg);
      break; 
      case 't':
      thresh = (atof(optarg));
      break; 
      case 'q':
      divisor = atoi(optarg);
      break; 
      default:
      usage(argv);
      exit(1);
    }
  }
  page_ranks = ((float *)(malloc(sizeof(float ) * n)));
  maps = ((float *)(malloc(sizeof(float ) * n * n)));
  noutlinks = ((unsigned int *)(malloc(sizeof(unsigned int ) * n)));
  max_diff = 99.0f;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= n - 1; i += 1) {
    noutlinks[i] = 0;
  }
  pages = random_pages(n,noutlinks,divisor);
  init_array(page_ranks,n,1.0f / ((float )n));
  nb_links = 0;
  
#pragma omp parallel for private (nb_links,i,j) reduction (+:nb_links)
  for (i = 0; i <= n - 1; i += 1) {
    
#pragma omp parallel for private (j) reduction (+:nb_links)
    for (j = 0; j <= n - 1; j += 1) {
      nb_links += pages[i * n + j];
    }
  }
  float *diffs;
  diffs = ((float *)(malloc(sizeof(float ) * n)));
  
#pragma omp parallel for private (i)
  for (i = 0; i <= n - 1; i += 1) {
    diffs[i] = 0.0f;
  }
  size_t block_size = (n < 256?n : 256);
  double ktime = 0.0;
{
    for (t = 1; t <= iter && max_diff >= thresh; ++t) {
      auto start = std::chrono::_V2::system_clock::now();
      
#pragma omp parallel for private (j_nom_2)
      for (int i = 0; i <= n - 1; i += 1) {
        float outbound_rank = page_ranks[i] / ((float )noutlinks[i]);
        
#pragma omp parallel for firstprivate (outbound_rank)
        for (int j = 0; j <= n - 1; j += 1) {
          maps[i * n + j] = pages[i * n + j] * outbound_rank;
        }
      }
      for (int j = 0; j <= n - 1; j += 1) {
        float new_rank;
        float old_rank;
        old_rank = page_ranks[j];
        new_rank = 0.0f;
        
#pragma omp parallel for reduction (+:new_rank)
        for (int i = 0; i <= n - 1; i += 1) {
          new_rank += maps[i * n + j];
        }
        new_rank = (1.f - 0.85f) / n + 0.85f * new_rank;
        diffs[j] = fmaxf((fabsf(new_rank - old_rank)),diffs[j]);
        page_ranks[j] = new_rank;
      }
      auto end = std::chrono::_V2::system_clock::now();
      ktime += std::chrono::duration_cast< class std::chrono::duration< double  , class std::ratio< 1 , 1L >  >  , int64_t  , std::nano  > ((end-start)) . count();
      max_diff = maximum_dif(diffs,n);
      
#pragma omp parallel for
      for (int i = 0; i <= n - 1; i += 1) {
        diffs[i] = 0.f;
      }
    }
    fprintf(stderr,"Max difference %f is reached at iteration %d\n",max_diff,t);
    printf("\"Options\": \"-n %d -i %d -t %f\". Total kernel execution time: %lf (s)\n",n,iter,thresh,ktime);
  }
  free(pages);
  free(maps);
  free(page_ranks);
  free(noutlinks);
  free(diffs);
  return 0;
}
