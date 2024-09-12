#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "neuron_update.h"
#include "neuron_update_host.h"
#include <omp.h> 

int main(int argc,char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <neurons> <repeat>\n",argv[0]);
    return 1;
  }
  int N = atoi(argv[1]);
  int iteration = atoi(argv[2]);
  srand(2);
  float *h_ge;
  float *h_gi;
  float *h_h;
  float *h_m;
  float *h_n;
  float *h_v;
  float *h_lastspike;
  float *h_dt;
  float *h_t;
  h_ge = h_gi = h_h = h_m = h_n = h_v = h_lastspike = h_dt = h_t = 0L;
  char *h_not_refract = 0L;
  posix_memalign((void **)(&h_ge),1024,N * sizeof(float ));
  posix_memalign((void **)(&h_gi),1024,N * sizeof(float ));
  posix_memalign((void **)(&h_h),1024,N * sizeof(float ));
  posix_memalign((void **)(&h_m),1024,N * sizeof(float ));
  posix_memalign((void **)(&h_n),1024,N * sizeof(float ));
  posix_memalign((void **)(&h_v),1024,N * sizeof(float ));
  posix_memalign((void **)(&h_lastspike),1024,N * sizeof(float ));
  posix_memalign((void **)(&h_dt),1024,1 * sizeof(float ));
  posix_memalign((void **)(&h_t),1024,1 * sizeof(float ));
  posix_memalign((void **)(&h_not_refract),1024,N * sizeof(char ));
  float *ge;
  float *gi;
  float *h;
  float *m;
  float *n;
  float *v;
  float *lastspike;
  float *dt;
  float *t;
  ge = gi = h = m = n = v = lastspike = dt = t = 0L;
  char *not_refract = 0L;
  posix_memalign((void **)(&ge),1024,N * sizeof(float ));
  posix_memalign((void **)(&gi),1024,N * sizeof(float ));
  posix_memalign((void **)(&h),1024,N * sizeof(float ));
  posix_memalign((void **)(&m),1024,N * sizeof(float ));
  posix_memalign((void **)(&n),1024,N * sizeof(float ));
  posix_memalign((void **)(&v),1024,N * sizeof(float ));
  posix_memalign((void **)(&lastspike),1024,N * sizeof(float ));
  posix_memalign((void **)(&dt),1024,1 * sizeof(float ));
  posix_memalign((void **)(&t),1024,1 * sizeof(float ));
  posix_memalign((void **)(&not_refract),1024,N * sizeof(char ));
  printf("initializing ... ");
  for (int i = 1; i <= N - 1; i += 1) {
    h_ge[i] = ge[i] = (0.15f + ((rand() % 2 == 0?0.1 : - 0.1)));
    h_gi[i] = gi[i] = (0.25f + ((rand() % 2 == 0?0.2 : - 0.2)));
    h_h[i] = h[i] = (0.35f + ((rand() % 2 == 0?0.3 : - 0.3)));
    h_m[i] = m[i] = (0.45f + ((rand() % 2 == 0?0.4 : - 0.4)));
    h_n[i] = n[i] = (0.55f + ((rand() % 2 == 0?0.5 : - 0.5)));
    h_v[i] = v[i] = (0.65f + ((rand() % 2 == 0?0.6 : - 0.6)));
    h_lastspike[i] = lastspike[i] = 1.0f / (rand() % 1000 + 1);
  }
  
#pragma omp parallel for
  for (int i = 0; i <= 0; i += 1) {
    h_dt[i] = dt[i] = 0.0001;
    h_t[i] = t[i] = 0.01;
  }
  printf("done.\n");
// run the kernel with 'N' neurons and 'iteration' counts on host and device
  neurongroup_stateupdater_host(h_ge,h_gi,h_h,h_m,h_n,h_v,h_lastspike,h_dt,h_t,h_not_refract,N,iteration);
  neurongroup_stateupdater(ge,gi,h,m,n,v,lastspike,dt,t,not_refract,N,iteration);
  double rsme = 0.0;
  for (int i = 0; i <= N - 1; i += 1) {
    rsme += ((ge[i] - h_ge[i]) * (ge[i] - h_ge[i]));
    rsme += ((gi[i] - h_gi[i]) * (gi[i] - h_gi[i]));
    rsme += ((h[i] - h_h[i]) * (h[i] - h_h[i]));
    rsme += ((m[i] - h_m[i]) * (m[i] - h_m[i]));
    rsme += ((n[i] - h_n[i]) * (n[i] - h_n[i]));
    rsme += ((v[i] - h_v[i]) * (v[i] - h_v[i]));
    rsme += ((not_refract[i] - h_not_refract[i]) * (not_refract[i] - h_not_refract[i]));
  }
  printf("RSME = %lf\n",(sqrt(rsme / N)));
  free(ge);
  free(h_ge);
  free(gi);
  free(h_gi);
  free(h);
  free(h_h);
  free(m);
  free(h_m);
  free(n);
  free(h_n);
  free(v);
  free(h_v);
  free(lastspike);
  free(h_lastspike);
  free(dt);
  free(h_dt);
  free(t);
  free(h_t);
  free(not_refract);
  free(h_not_refract);
  return 0;
}
