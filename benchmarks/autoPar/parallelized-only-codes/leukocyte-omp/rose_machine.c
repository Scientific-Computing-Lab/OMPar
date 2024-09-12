/**************************************************************************
**
** Copyright (C) 1993 David E. Stewart & Zbigniew Leyk, all rights reserved.
**
**			     Meschach Library
** 
** This Meschach Library is provided "as is" without any express 
** or implied warranty of any kind with respect to this software. 
** In particular the authors shall not be liable for any direct, 
** indirect, special, incidental or consequential damages arising 
** in any way from use of the software.
** 
** Everyone is granted permission to copy, modify and redistribute this
** Meschach Library, provided:
**  1.  All copies contain this copyright notice.
**  2.  All modified copies shall carry a notice stating who
**      made the last modification and the date of such modification.
**  3.  No charge is made for this software or works derived from it.  
**      This clause shall not be construed as constraining other software
**      distributed on the same medium as this software, nor is a
**      distribution fee considered a charge.
**
***************************************************************************/
/*
  This file contains basic routines which are used by the functions
  in meschach.a etc.
  These are the routines that should be modified in order to take
  full advantage of specialised architectures (pipelining, vector
  processors etc).
  */
#include <omp.h> 
static char *rcsid = "$Id: machine.c,v 1.4 1994/01/13 05:28:56 des Exp $";
#include	"machine.h"
/* __ip__ -- inner product */
#ifndef ANSI_C
#else

double __ip__(const double *dp1,const double *dp2,int len)
#endif
{
#ifdef VUNROLL
#endif
  int i;
  double sum;
  sum = 0.0;
#ifdef VUNROLL
#endif
  
#pragma omp parallel for private (sum,i) reduction (+:sum)
  for (i = 0; i <= len - 1; i += 1) {
    sum += dp1[i] * dp2[i];
  }
  return sum;
}
/* __mltadd__ -- scalar multiply and add c.f. v_mltadd() */
#ifndef ANSI_C
#else

void __mltadd__(double *dp1,const double *dp2,double s,int len)
#endif
{
  int i;
#ifdef VUNROLL
#endif
  
#pragma omp parallel for private (i)
  for (i = 0; i <= len - 1; i += 1) {
    dp1[i] += s * dp2[i];
  }
}
/* __smlt__ scalar multiply array c.f. sv_mlt() */
#ifndef ANSI_C
#else

void __smlt__(const double *dp,double s,double *out,int len)
#endif
{
  int i;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= len - 1; i += 1) {
    out[i] = s * dp[i];
  }
}
/* __add__ -- add arrays c.f. v_add() */
#ifndef ANSI_C
#else

void __add__(const double *dp1,const double *dp2,double *out,int len)
#endif
{
  int i;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= len - 1; i += 1) {
    out[i] = dp1[i] + dp2[i];
  }
}
/* __sub__ -- subtract arrays c.f. v_sub() */
#ifndef ANSI_C
#else

void __sub__(const double *dp1,const double *dp2,double *out,int len)
#endif
{
  int i;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= len - 1; i += 1) {
    out[i] = dp1[i] - dp2[i];
  }
}
/* __zero__ -- zeros an array of floating point numbers */
#ifndef ANSI_C
#else

void __zero__(double *dp,int len)
#endif
{
#ifdef CHAR0ISDBL0
/* if a floating point zero is equivalent to a string of nulls */
  memset(((char *)dp),'\0',len * sizeof(double ));
#else
/* else, need to zero the array entry by entry */
#endif
}
