/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
/* genmtrand.c */
/* Copyright (C) 2001-2006 Makoto Matsumoto and Takuji Nishimura.  */
/* This library is free software; you can redistribute it and/or   */
/* modify it under the terms of the GNU Library General Public     */
/* License as published by the Free Software Foundation; either    */
/* version 2 of the License, or (at your option) any later         */
/* version.                                                        */
/* This library is distributed in the hope that it will be useful, */
/* but WITHOUT ANY WARRANTY; without even the implied warranty of  */
/* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.            */
/* See the GNU Library General Public License for more details.    */
/* You should have received a copy of the GNU Library General      */
/* Public License along with this library; if not, write to the    */
/* Free Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA   */
/* 02111-1307  USA                                                 */
/*
 * Victor Podlozhnyuk @ 05/13/2007:
 * 1) Fixed sgenrand_mt():
 *    - Fixed loop indexing: 'i' variable was off by one.
 *    - apply wmask right on the state element initialization instead
 *      of separate loop, which could produce machine-dependent results(wrong).
 * 2) Slightly reformatted sources to be included into CUDA SDK.
 */
#include <stdio.h>
#include <stdlib.h>
#include "dci.h"
#define SHIFT1 18

extern "C" void sgenrand_mt(uint32_t seed,mt_struct *mts)
{
  int i;
  mts -> state[0] = seed & mts -> wmask;
  for (i = 1; i <= mts -> nn - 1; i += 1) {
    mts -> state[i] = ((uint32_t )1812433253) * (mts -> state[i - 1] ^ mts -> state[i - 1] >> 30) + i & mts -> wmask;
/* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
/* In the previous versions, MSBs of the seed affect   */
/* only MSBs of the array mt[].                        */
  }
  mts -> i = mts -> nn;
}

extern "C" uint32_t genrand_mt(mt_struct *mts)
{
  uint32_t *st;
  uint32_t uuu;
  uint32_t lll;
  uint32_t aa;
  uint32_t x;
  int k;
  int n;
  int m;
  int lim;
  if (mts -> i >= mts -> nn) {
    n = mts -> nn;
    m = mts -> mm;
    aa = mts -> aaa;
    st = mts -> state;
    uuu = mts -> umask;
    lll = mts -> lmask;
    lim = n - m;
    for (k = 0; k <= lim - 1; k += 1) {
      x = st[k] & uuu | st[k + 1] & lll;
      st[k] = st[k + m] ^ x >> 1 ^ (((x & 1U)?aa : 0U));
    }
    lim = n - 1;
    for (; k <= lim - 1; k += 1) {
      x = st[k] & uuu | st[k + 1] & lll;
      st[k] = st[k + m - n] ^ x >> 1 ^ (((x & 1U)?aa : 0U));
    }
    x = st[n - 1] & uuu | st[0] & lll;
    st[n - 1] = st[m - 1] ^ x >> 1 ^ (((x & 1U)?aa : 0U));
    mts -> i = 0;
  }
  x = mts -> state[mts -> i];
  mts -> i += 1;
  x ^= x >> mts -> shift0;
  x ^= x << mts -> shiftB & mts -> maskB;
  x ^= x << mts -> shiftC & mts -> maskC;
  x ^= x >> mts -> shift1;
  return x;
}
