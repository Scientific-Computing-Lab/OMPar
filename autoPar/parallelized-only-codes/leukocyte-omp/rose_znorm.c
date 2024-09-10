/**************************************************************************
**
** Copyright (C) 1993 David E. Steward & Zbigniew Leyk, all rights reserved.
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
	A collection of functions for computing norms: scaled and unscaled
	Complex version
*/
#include <omp.h> 
static char rcsid[] = "$Id: znorm.c,v 1.1 1994/01/13 04:21:31 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
/* _zv_norm1 -- computes (scaled) 1-norms of vectors */

double _zv_norm1(ZVEC *x,VEC *scale)
{
  int i;
  int dim;
  double s;
  double sum;
  if (x == ((ZVEC *)((void *)0))) 
    ev_err("znorm.c",8,46,"_zv_norm1",0);
  dim = (x -> dim);
  sum = 0.0;
  if (scale == ((VEC *)((void *)0))) 
    for (i = 0; i <= dim - 1; i += 1) {
      sum += zabs(x -> ve[i]);
    }
   else if (scale -> dim < dim) 
    ev_err("znorm.c",1,54,"_zv_norm1",0);
   else 
    for (i = 0; i <= dim - 1; i += 1) {
      s = scale -> ve[i];
      sum += (s == 0.0?zabs(x -> ve[i]) : zabs(x -> ve[i]) / fabs(s));
    }
  return sum;
}
/* square -- returns x^2 */
/******************************
double	square(x)
double	x;
{	return x*x;	}
******************************/
#define	square(x)	((x)*(x))
/* _zv_norm2 -- computes (scaled) 2-norm (Euclidean norm) of vectors */

double _zv_norm2(ZVEC *x,VEC *scale)
{
  int i;
  int dim;
  double s;
  double sum;
  if (x == ((ZVEC *)((void *)0))) 
    ev_err("znorm.c",8,81,"_zv_norm2",0);
  dim = (x -> dim);
  sum = 0.0;
  if (scale == ((VEC *)((void *)0))) {
    
#pragma omp parallel for private (i) reduction (+:sum) firstprivate (dim)
    for (i = 0; i <= dim - 1; i += 1) {
      sum += x -> ve[i] . re * x -> ve[i] . re + x -> ve[i] . im * x -> ve[i] . im;
    }
  }
   else if (scale -> dim < dim) 
    ev_err("znorm.c",1,89,"_v_norm2",0);
   else {
    
#pragma omp parallel for private (s,i) reduction (+:sum) firstprivate (dim)
    for (i = 0; i <= dim - 1; i += 1) {
      s = scale -> ve[i];
      sum += (s == 0.0?x -> ve[i] . re * x -> ve[i] . re + x -> ve[i] . im * x -> ve[i] . im : (x -> ve[i] . re * x -> ve[i] . re + x -> ve[i] . im * x -> ve[i] . im) / (s * s));
    }
  }
  return sqrt(sum);
}
#define	max(a,b)	((a) > (b) ? (a) : (b))
/* _zv_norm_inf -- computes (scaled) infinity-norm (supremum norm) of vectors */

double _zv_norm_inf(ZVEC *x,VEC *scale)
{
  int i;
  int dim;
  double s;
  double maxval;
  double tmp;
  if (x == ((ZVEC *)((void *)0))) 
    ev_err("znorm.c",8,110,"_zv_norm_inf",0);
  dim = (x -> dim);
  maxval = 0.0;
  if (scale == ((VEC *)((void *)0))) 
    for (i = 0; i <= dim - 1; i += 1) {
      tmp = zabs(x -> ve[i]);
      maxval = (maxval > tmp?maxval : tmp);
    }
   else if (scale -> dim < dim) 
    ev_err("znorm.c",1,121,"_zv_norm_inf",0);
   else 
    for (i = 0; i <= dim - 1; i += 1) {
      s = scale -> ve[i];
      tmp = (s == 0.0?zabs(x -> ve[i]) : zabs(x -> ve[i]) / fabs(s));
      maxval = (maxval > tmp?maxval : tmp);
    }
  return maxval;
}
/* zm_norm1 -- compute matrix 1-norm -- unscaled
	-- complex version */

double zm_norm1(ZMAT *A)
{
  int i;
  int j;
  int m;
  int n;
  double maxval;
  double sum;
  if (A == ((ZMAT *)((void *)0))) 
    ev_err("znorm.c",8,141,"zm_norm1",0);
  m = (A -> m);
  n = (A -> n);
  maxval = 0.0;
  for (j = 0; j <= n - 1; j += 1) {
    sum = 0.0;
    for (i = 0; i <= m - 1; i += 1) {
      sum += zabs(A -> me[i][j]);
    }
    maxval = (maxval > sum?maxval : sum);
  }
  return maxval;
}
/* zm_norm_inf -- compute matrix infinity-norm -- unscaled
	-- complex version */

double zm_norm_inf(ZMAT *A)
{
  int i;
  int j;
  int m;
  int n;
  double maxval;
  double sum;
  if (A == ((ZMAT *)((void *)0))) 
    ev_err("znorm.c",8,165,"zm_norm_inf",0);
  m = (A -> m);
  n = (A -> n);
  maxval = 0.0;
  for (i = 0; i <= m - 1; i += 1) {
    sum = 0.0;
    for (j = 0; j <= n - 1; j += 1) {
      sum += zabs(A -> me[i][j]);
    }
    maxval = (maxval > sum?maxval : sum);
  }
  return maxval;
}
/* zm_norm_frob -- compute matrix frobenius-norm -- unscaled */

double zm_norm_frob(ZMAT *A)
{
  int i;
  int j;
  int m;
  int n;
  double sum;
  if (A == ((ZMAT *)((void *)0))) 
    ev_err("znorm.c",8,188,"zm_norm_frob",0);
  m = (A -> m);
  n = (A -> n);
  sum = 0.0;
  
#pragma omp parallel for private (i,j) reduction (+:sum) firstprivate (m,n)
  for (i = 0; i <= m - 1; i += 1) {
    
#pragma omp parallel for private (j) reduction (+:sum)
    for (j = 0; j <= n - 1; j += 1) {
      sum += A -> me[i][j] . re * A -> me[i][j] . re + A -> me[i][j] . im * A -> me[i][j] . im;
    }
  }
  return sqrt(sum);
}
