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
	This is a file of routines for zero-ing, and initialising
	vectors, matrices and permutations.
	This is to be included in the matrix.a library
*/
#include <omp.h> 
static char rcsid[] = "$Id: init.c,v 1.6 1994/01/13 05:36:58 des Exp $";
#include	<stdio.h>
#include	"matrix.h"
/* v_zero -- zero the vector x */
#ifndef ANSI_C
#else

VEC *v_zero(VEC *x)
#endif
{
  if (x == ((VEC *)((void *)0))) 
    ev_err("init.c",8,47,"v_zero",0);
  __zero__(x -> ve,(x -> dim));
/* for ( i = 0; i < x->dim; i++ )
		x->ve[i] = 0.0; */
  return x;
}
/* iv_zero -- zero the vector ix */
#ifndef ANSI_C
#else

IVEC *iv_zero(IVEC *ix)
#endif
{
  int i;
  if (ix == ((IVEC *)((void *)0))) 
    ev_err("init.c",8,68,"iv_zero",0);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= ix -> dim - 1; i += 1) {
    ix -> ive[i] = 0;
  }
  return ix;
}
/* m_zero -- zero the matrix A */
#ifndef ANSI_C
#else

MAT *m_zero(MAT *A)
#endif
{
  int i;
  int A_m;
  int A_n;
  double **A_me;
  if (A == ((MAT *)((void *)0))) 
    ev_err("init.c",8,89,"m_zero",0);
  A_m = (A -> m);
  A_n = (A -> n);
  A_me = A -> me;
  for (i = 0; i <= A_m - 1; i += 1) {
    __zero__(A_me[i],A_n);
  }
/* for ( j = 0; j < A_n; j++ )
			A_me[i][j] = 0.0; */
  return A;
}
/* mat_id -- set A to being closest to identity matrix as possible
	-- i.e. A[i][j] == 1 if i == j and 0 otherwise */
#ifndef ANSI_C
#else

MAT *m_ident(MAT *A)
#endif
{
  int i;
  int size;
  if (A == ((MAT *)((void *)0))) 
    ev_err("init.c",8,112,"m_ident",0);
  m_zero(A);
  size = ((A -> m > A -> n?A -> n : A -> m));
  
#pragma omp parallel for private (i) firstprivate (size)
  for (i = 0; i <= size - 1; i += 1) {
    A -> me[i][i] = 1.0;
  }
  return A;
}
/* px_ident -- set px to identity permutation */
#ifndef ANSI_C
#else

PERM *px_ident(PERM *px)
#endif
{
  int i;
  int px_size;
  unsigned int *px_pe;
  if (px == ((PERM *)((void *)0))) 
    ev_err("init.c",8,134,"px_ident",0);
  px_size = (px -> size);
  px_pe = px -> pe;
  
#pragma omp parallel for private (i) firstprivate (px_size)
  for (i = 0; i <= px_size - 1; i += 1) {
    px_pe[i] = i;
  }
  return px;
}
/* Pseudo random number generator data structures */
/* Knuth's lagged Fibonacci-based generator: See "Seminumerical Algorithms:
   The Art of Computer Programming" sections 3.2-3.3 */
#ifdef ANSI_C
#ifndef LONG_MAX
#include	<limits.h>
#endif
#endif
#ifdef LONG_MAX
#define MODULUS	LONG_MAX
#else
#define MODULUS	1000000000L	/* assuming long's at least 32 bits long */
#endif
#define MZ	0L
static long mrand_list[56];
static int started = 0;
static int inext = 0;
static int inextp = 31;
/* mrand -- pseudo-random number generator */
#ifdef ANSI_C

double mrand()
#else
#endif
{
  long lval;
  static double factor = 1.0 / ((double )0x7fffffffffffffffL);
  if (!started) 
    smrand(3127);
  inext = (inext >= 54?0 : inext + 1);
  inextp = (inextp >= 54?0 : inextp + 1);
  lval = mrand_list[inext] - mrand_list[inextp];
  if (lval < 0L) 
    lval += 0x7fffffffffffffffL;
  mrand_list[inext] = lval;
  return ((double )lval) * factor;
}
/* mrandlist -- fills the array a[] with len random numbers */
#ifndef ANSI_C
#else

void mrandlist(double a[],int len)
#endif
{
  int i;
  long lval;
  static double factor = 1.0 / ((double )0x7fffffffffffffffL);
  if (!started) 
    smrand(3127);
  for (i = 0; i <= len - 1; i += 1) {
    inext = (inext >= 54?0 : inext + 1);
    inextp = (inextp >= 54?0 : inextp + 1);
    lval = mrand_list[inext] - mrand_list[inextp];
    if (lval < 0L) 
      lval += 0x7fffffffffffffffL;
    mrand_list[inext] = lval;
    a[i] = ((double )lval) * factor;
  }
}
/* smrand -- set seed for mrand() */
#ifndef ANSI_C
#else

void smrand(int seed)
#endif
{
  int i;
  mrand_list[0] = (123413 * seed) % 0x7fffffffffffffffL;
  for (i = 1; i <= 54; i += 1) {
    mrand_list[i] = 123413 * mrand_list[i - 1] % 0x7fffffffffffffffL;
  }
  started = 1;
/* run mrand() through the list sufficient times to
       thoroughly randomise the array */
  for (i = 0; i <= 3024; i += 1) {
    mrand();
  }
}
#undef MODULUS
#undef MZ
#undef FAC
/* v_rand -- initialises x to be a random vector, components
	independently & uniformly ditributed between 0 and 1 */
#ifndef ANSI_C
#else

VEC *v_rand(VEC *x)
#endif
{
/* int	i; */
  if (!x) 
    ev_err("init.c",8,257,"v_rand",0);
/* for ( i = 0; i < x->dim; i++ ) */
/* x->ve[i] = rand()/((Real)MAX_RAND); */
/* x->ve[i] = mrand(); */
  mrandlist(x -> ve,(x -> dim));
  return x;
}
/* m_rand -- initialises A to be a random vector, components
	independently & uniformly distributed between 0 and 1 */
#ifndef ANSI_C
#else

MAT *m_rand(MAT *A)
#endif
{
  int i;
/* , j */
  if (!A) 
    ev_err("init.c",8,279,"m_rand",0);
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
/* for ( j = 0; j < A->n; j++ ) */
/* A->me[i][j] = rand()/((Real)MAX_RAND); */
/* A->me[i][j] = mrand(); */
    mrandlist(A -> me[i],(A -> n));
  }
  return A;
}
/* v_ones -- fills x with one's */
#ifndef ANSI_C
#else

VEC *v_ones(VEC *x)
#endif
{
  int i;
  if (!x) 
    ev_err("init.c",8,301,"v_ones",0);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= x -> dim - 1; i += 1) {
    x -> ve[i] = 1.0;
  }
  return x;
}
/* m_ones -- fills matrix with one's */
#ifndef ANSI_C
#else

MAT *m_ones(MAT *A)
#endif
{
  int i;
  int j;
  if (!A) 
    ev_err("init.c",8,320,"m_ones",0);
  
#pragma omp parallel for private (i,j)
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; ((unsigned int )j) <= A -> n - 1; j += 1) {
      A -> me[i][j] = 1.0;
    }
  }
  return A;
}
/* v_count -- initialises x so that x->ve[i] == i */
#ifndef ANSI_C
#else

VEC *v_count(VEC *x)
#endif
{
  int i;
  if (!x) 
    ev_err("init.c",8,340,"v_count",0);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= x -> dim - 1; i += 1) {
    x -> ve[i] = ((double )i);
  }
  return x;
}
