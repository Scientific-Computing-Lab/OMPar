#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#define  bidx  omp_get_team_num()
#define  tidx  omp_get_thread_num()
#include "modP.h"
#include <omp.h> 

void intt_3_64k_modcrt(const uint32 numTeams,uint32 *dst,const uint64 *src)
{
{
    uint64 buffer[512];
{
      register uint64 samples[8];
      register uint64 s8[8];
      register uint32 fmem;
      register uint32 tmem;
      register uint32 fbuf;
      register uint32 tbuf;
      fmem = (omp_get_team_num() << 9 | (omp_get_thread_num() & 0x3E) << 3 | omp_get_thread_num() & 0x1);
      tbuf = (omp_get_thread_num() << 3);
      fbuf = ((omp_get_thread_num() & 0x38) << 3 | omp_get_thread_num() & 0x7);
      tmem = (omp_get_team_num() << 9 | (omp_get_thread_num() & 0x38) << 3 | omp_get_thread_num() & 0x7);
      for (int i = 0; i <= 7; i += 1) {
        samples[i] = src[fmem | (i << 1)];
      }
      ntt8(samples);
      for (int i = 0; i <= 7; i += 1) {
        buffer[tbuf | i] = _ls_modP(samples[i],((omp_get_thread_num() & 0x1) << 2) * i * 3);
      }
      for (int i = 0; i <= 7; i += 1) {
        samples[i] = buffer[fbuf | (i << 3)];
      }
      for (int i = 0; i <= 3; i += 1) {
        s8[2 * i] = _add_modP(samples[2 * i],samples[2 * i + 1]);
        s8[2 * i + 1] = _sub_modP(samples[2 * i],samples[2 * i + 1]);
      }
      for (int i = 0; i <= 7; i += 1) {
        dst[((tmem | (i << 3)) & 0xf) << 12 | (tmem | (i << 3)) >> 4] = ((uint32 )(_mul_modP(s8[i],18446462594437939201UL,0xffffffff00000001UL)));
      }
    }
  }
}

int main(int argc,char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int nttLen = 64 * 1024;
  uint64 *ntt = (uint64 *)(malloc(nttLen * sizeof(uint64 )));
  uint32 *res = (uint32 *)(malloc(nttLen * sizeof(uint32 )));
  srand(123);
  for (int i = 0; i <= nttLen - 1; i += 1) {
    uint64 hi = (rand());
    uint64 lo = (rand());
    ntt[i] = hi << 32 | lo;
  }
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      intt_3_64k_modcrt((nttLen / 512),res,ntt);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (us)\n",(time * 1e-3f / repeat));
  }
  uint64 checksum = 0;
  
#pragma omp parallel for reduction (+:checksum) firstprivate (nttLen)
  for (int i = 0; i <= nttLen - 1; i += 1) {
    checksum += res[i];
  }
  printf("Checksum: %lu\n",checksum);
  free(ntt);
  free(res);
  return 0;
}
