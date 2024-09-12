#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
extern int layer_size;

void load(BPNN *net)
//BPNN *net;
{
  float *units;
  int nr;
  int i;
  int k;
  nr = layer_size;
  units = net -> input_units;
  k = 1;
  for (i = 0; i <= nr - 1; i += 1) {
    units[k] = ((float )(rand())) / 2147483647;
    k++;
  }
}
