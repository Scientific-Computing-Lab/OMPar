#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include <omp.h> 
typedef unsigned long long u64Int;
typedef long long s64Int;
/* Random number generator */
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
#define NUPDATE (4 * TableSize)

u64Int HPCC_starts(s64Int n)
{
  int i;
  int j;
  u64Int m2[64];
  u64Int temp;
  u64Int ran;
  while(n < 0)
    n += 1317624576693539401L;
  while(n > 1317624576693539401L)
    n -= 1317624576693539401L;
  if (n == 0) 
    return 0x1;
  temp = 0x1;
  for (i = 0; i <= 63; i += 1) {
    m2[i] = temp;
    temp = temp << 1 ^ ((((s64Int )temp) < 0?0x0000000000000007UL : 0));
    temp = temp << 1 ^ ((((s64Int )temp) < 0?0x0000000000000007UL : 0));
  }
  for (i = 62; i >= 0; i += -1) {
    if ((n >> i & 1)) 
      break; 
  }
  ran = 0x2;
  while(i > 0){
    temp = 0;
    for (j = 0; j <= 63; j += 1) {
      if ((ran >> j & 1)) 
        temp ^= m2[j];
    }
    ran = temp;
    i -= 1;
    if ((n >> i & 1)) 
      ran = ran << 1 ^ ((((s64Int )ran) < 0?0x0000000000000007UL : 0));
  }
  return ran;
}

int main(int argc,char **argv)
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n",argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  int failure;
  u64Int i;
  u64Int temp;
  double totalMem;
  u64Int *Table = 0L;
  u64Int logTableSize;
  u64Int TableSize;
/* calculate local memory per node for the update table */
  totalMem = (1024 * 1024 * 512);
  totalMem /= (sizeof(u64Int ));
/* calculate the size of update array (must be a power of 2) */
  for (((totalMem *= 0.5 , logTableSize = 0) , TableSize = 1); totalMem >= 1.0; ((totalMem *= 0.5 , logTableSize++) , TableSize <<= 1)) 
    ;
/* EMPTY */
  printf("Table size = %llu\n",TableSize);
  posix_memalign((void **)(&Table),1024,(TableSize * (sizeof(u64Int ))));
  if (!Table) {
    fprintf(stderr,"Failed to allocate memory for the update table %llu\n",TableSize);
    return 1;
  }
/* Print parameters for run */
  fprintf(stdout,"Main table size   = 2^%llu = %llu words\n",logTableSize,TableSize);
  fprintf(stdout,"Number of updates = %llu\n",4 * TableSize);
  u64Int ran[128];
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
/* Initialize main table */
      
#pragma omp parallel for private (i_nom_1)
      for (i = 0; ((unsigned long long )i) <= TableSize - 1; i += 1) {
        Table[i] = i;
      }
      for (int j = 0; j <= 127; j += 1) {
        ran[j] = HPCC_starts((4 * TableSize / 128 * j));
      }
      for (int j = 0; j <= 127; j += 1) {
        for (u64Int i = 0; i <= ((unsigned long long )4) * TableSize / ((unsigned long long )128) - 1; i += 1) {
          ran[j] = ran[j] << 1 ^ ((((s64Int )ran[j]) < 0?0x0000000000000007UL : 0));
          Table[ran[j] & TableSize - 1] ^= ran[j];
        }
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
  }
/* validation */
  temp = 0x1;
  for (i = 0; i <= ((unsigned long long )4) * TableSize - 1; i += 1) {
    temp = temp << 1 ^ ((((s64Int )temp) < 0?0x0000000000000007UL : 0));
    Table[temp & TableSize - 1] ^= temp;
  }
  temp = 0;
  
#pragma omp parallel for private (i) reduction (+:temp)
  for (i = 0; i <= TableSize - 1; i += 1) {
    if (Table[i] != i) {
      temp++;
    }
  }
  fprintf(stdout,"Found %llu errors in %llu locations (%s).\n",temp,TableSize,(temp <= 0.01 * TableSize?"passed" : "failed"));
  if (temp <= 0.01 * TableSize) 
    failure = 0;
   else 
    failure = 1;
  free(Table);
  return failure;
}
