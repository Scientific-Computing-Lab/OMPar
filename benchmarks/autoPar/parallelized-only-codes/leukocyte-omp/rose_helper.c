#include <stdlib.h>
#include <sys/time.h>
/*****************************
 *GET_TIME
 *returns a long int representing the time
 *****************************/

long long get_time()
{
  struct timeval tv;
  gettimeofday(&tv,(void *)0);
  return (tv . tv_sec * 1000000 + tv . tv_usec);
}
