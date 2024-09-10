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
	Matrix factorisation routines to work with the other matrix files.
*/
#include <omp.h> 
static char rcsid[] = "$Id: bkpfacto.c,v 1.7 1994/01/13 05:45:50 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"
#define	btos(x)	((x) ? "TRUE" : "FALSE")
/* Most matrix factorisation routines are in-situ unless otherwise specified */
#define alpha	0.6403882032022076 /* = (1+sqrt(17))/8 */
/* sqr -- returns square of x -- utility function */

double sqr(double x)
{
  return x * x;
}
/* interchange -- a row/column swap routine */

static void interchange(MAT *A,int i,int j)
//MAT	*A;	/* assumed != NULL & also SQUARE */
//int	i, j;	/* assumed in range */
{
  double **A_me;
  double tmp;
  int k;
  int n;
  A_me = A -> me;
  n = (A -> n);
  if (i == j) 
    return ;
  if (i > j) {
    k = i;
    i = j;
    j = k;
  }
  
#pragma omp parallel for private (tmp,k)
  for (k = 0; k <= i - 1; k += 1) {
/* tmp = A_me[k][i]; */
    tmp = A -> me[k][i];
/* A_me[k][i] = A_me[k][j]; */
    A -> me[k][i] = A -> me[k][j];
/* A_me[k][j] = tmp; */
    A -> me[k][j] = tmp;
  }
  
#pragma omp parallel for private (tmp,k) firstprivate (n)
  for (k = j + 1; k <= n - 1; k += 1) {
/* tmp = A_me[j][k]; */
    tmp = A -> me[j][k];
/* A_me[j][k] = A_me[i][k]; */
    A -> me[j][k] = A -> me[i][k];
/* A_me[i][k] = tmp; */
    A -> me[i][k] = tmp;
  }
  for (k = i + 1; k <= j - 1; k += 1) {
/* tmp = A_me[k][j]; */
    tmp = A -> me[k][j];
/* A_me[k][j] = A_me[i][k]; */
    A -> me[k][j] = A -> me[i][k];
/* A_me[i][k] = tmp; */
    A -> me[i][k] = tmp;
  }
/* tmp = A_me[i][i]; */
  tmp = A -> me[i][i];
/* A_me[i][i] = A_me[j][j]; */
  A -> me[i][i] = A -> me[j][j];
/* A_me[j][j] = tmp; */
  A -> me[j][j] = tmp;
}
/* BKPfactor -- Bunch-Kaufman-Parlett factorisation of A in-situ
	-- A is factored into the form P'AP = MDM' where 
	P is a permutation matrix, M lower triangular and D is block
	diagonal with blocks of size 1 or 2
	-- P is stored in pivot; blocks[i]==i iff D[i][i] is a block */
#ifndef ANSI_C
#else

MAT *BKPfactor(MAT *A,PERM *pivot,PERM *blocks)
#endif
{
  int i;
  int j;
  int k;
  int n;
  int onebyone;
  int r;
  double **A_me;
  double aii;
  double aip1;
  double aip1i;
  double lambda;
  double sigma;
  double tmp;
  double det;
  double s;
  double t;
  if (!A || !pivot || !blocks) 
    ev_err("bkpfacto.c",8,114,"BKPfactor",0);
  if (A -> m != A -> n) 
    ev_err("bkpfacto.c",9,116,"BKPfactor",0);
  if (A -> m != pivot -> size || pivot -> size != blocks -> size) 
    ev_err("bkpfacto.c",1,118,"BKPfactor",0);
  n = (A -> n);
  A_me = A -> me;
  px_ident(pivot);
  px_ident(blocks);
  for (i = 0; i <= n - 1; i = (onebyone?i + 1 : i + 2)) {
/* printf("# Stage: %d\n",i); */
    aii = fabs(A -> me[i][i]);
    lambda = 0.0;
    r = (i + 1 < n?i + 1 : i);
    for (k = i + 1; k <= n - 1; k += 1) {
      tmp = fabs(A -> me[i][k]);
      if (tmp >= lambda) {
        lambda = tmp;
        r = k;
      }
    }
/* printf("# lambda = %g, r = %d\n", lambda, r); */
/* printf("# |A[%d][%d]| = %g\n",r,r,fabs(m_entry(A,r,r))); */
/* determine if 1x1 or 2x2 block, and do pivoting if needed */
    if (aii >= 0.6403882032022076 * lambda) {
      onebyone = 1;
      goto dopivot;
    }
/* compute sigma */
    sigma = 0.0;
    for (k = i; k <= n - 1; k += 1) {
      if (k == r) 
        continue; 
      tmp = (k > r?fabs(A -> me[r][k]) : fabs(A -> me[k][r]));
      if (tmp > sigma) 
        sigma = tmp;
    }
    if (aii * sigma >= 0.6403882032022076 * sqr(lambda)) 
      onebyone = 1;
     else if (fabs(A -> me[r][r]) >= 0.6403882032022076 * sigma) {
/* printf("# Swapping rows/cols %d and %d\n",i,r); */
      interchange(A,i,r);
      px_transp(pivot,i,r);
      onebyone = 1;
    }
     else {
/* printf("# Swapping rows/cols %d and %d\n",i+1,r); */
      interchange(A,i + 1,r);
      px_transp(pivot,(i + 1),r);
      px_transp(blocks,i,(i + 1));
      onebyone = 0;
    }
/* printf("onebyone = %s\n",btos(onebyone)); */
/* printf("# Matrix so far (@checkpoint A) =\n"); */
/* m_output(A); */
/* printf("# pivot =\n");	px_output(pivot); */
/* printf("# blocks =\n");	px_output(blocks); */
    dopivot:
    if (onebyone) {
/* do one by one block */
      if (A -> me[i][i] != 0.0) {
        aii = A -> me[i][i];
        for (j = i + 1; j <= n - 1; j += 1) {
          tmp = A -> me[i][j] / aii;
          
#pragma omp parallel for private (k)
          for (k = j; k <= n - 1; k += 1) {
            A -> me[j][k] -= tmp * A -> me[i][k];
          }
          A -> me[i][j] = tmp;
        }
      }
    }
     else 
/* onebyone == FALSE */
{
/* do two by two block */
      det = A -> me[i][i] * A -> me[i + 1][i + 1] - sqr(A -> me[i][i + 1]);
/* Must have det < 0 */
/* printf("# det = %g\n",det); */
      aip1i = A -> me[i][i + 1] / det;
      aii = A -> me[i][i] / det;
      aip1 = A -> me[i + 1][i + 1] / det;
      for (j = i + 2; j <= n - 1; j += 1) {
        s = -aip1i * A -> me[i + 1][j] + aip1 * A -> me[i][j];
        t = -aip1i * A -> me[i][j] + aii * A -> me[i + 1][j];
        
#pragma omp parallel for private (k)
        for (k = j; k <= n - 1; k += 1) {
          A -> me[j][k] -= A -> me[i][k] * s + A -> me[i + 1][k] * t;
        }
        A -> me[i][j] = s;
        A -> me[i + 1][j] = t;
      }
    }
/* printf("# Matrix so far (@checkpoint B) =\n"); */
/* m_output(A); */
/* printf("# pivot =\n");	px_output(pivot); */
/* printf("# blocks =\n");	px_output(blocks); */
  }
/* set lower triangular half */
  
#pragma omp parallel for private (i,j)
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
    for (j = 0; j <= i - 1; j += 1) {
      A -> me[i][j] = A -> me[j][i];
    }
  }
  return A;
}
/* BKPsolve -- solves A.x = b where A has been factored a la BKPfactor()
	-- returns x, which is created if NULL */
#ifndef ANSI_C
#else

VEC *BKPsolve(const MAT *A,PERM *pivot,const PERM *block,const VEC *b,VEC *x)
#endif
{
  static VEC *tmp = (VEC *)((void *)0);
/* dummy storage needed */
  int i;
  int j;
  int n;
  int onebyone;
  double **A_me;
  double a11;
  double a12;
  double a22;
  double b1;
  double b2;
  double det;
  double sum;
  double *tmp_ve;
  double tmp_diag;
  if (!A || !pivot || !block || !b) 
    ev_err("bkpfacto.c",8,245,"BKPsolve",0);
  if (A -> m != A -> n) 
    ev_err("bkpfacto.c",9,247,"BKPsolve",0);
  n = (A -> n);
  if (b -> dim != n || pivot -> size != n || block -> size != n) 
    ev_err("bkpfacto.c",1,250,"BKPsolve",0);
  x = v_resize(x,n);
  tmp = v_resize(tmp,n);
  mem_stat_reg_list((void **)(&tmp),3,0,"bkpfacto.c",253);
  A_me = A -> me;
  tmp_ve = tmp -> ve;
  px_vec(pivot,b,tmp);
/* solve for lower triangular part */
  for (i = 0; i <= n - 1; i += 1) {
    sum = tmp -> ve[i];
    if (block -> pe[i] < i) {
      
#pragma omp parallel for private (j) reduction (-:sum)
      for (j = 0; j <= i - 1 - 1; j += 1) {
        sum -= A -> me[i][j] * tmp -> ve[j];
      }
    }
     else {
      
#pragma omp parallel for private (j) reduction (-:sum)
      for (j = 0; j <= i - 1; j += 1) {
        sum -= A -> me[i][j] * tmp -> ve[j];
      }
    }
    tmp -> ve[i] = sum;
  }
/* printf("# BKPsolve: solving L part: tmp =\n");	v_output(tmp); */
/* solve for diagonal part */
  for (i = 0; i <= n - 1; i = (onebyone?i + 1 : i + 2)) {
    onebyone = block -> pe[i] == i;
    if (onebyone) {
      tmp_diag = A -> me[i][i];
      if (tmp_diag == 0.0) 
        ev_err("bkpfacto.c",4,279,"BKPsolve",0);
/* tmp_ve[i] /= tmp_diag; */
      tmp -> ve[i] = tmp -> ve[i] / tmp_diag;
    }
     else {
      a11 = A -> me[i][i];
      a22 = A -> me[i + 1][i + 1];
      a12 = A -> me[i + 1][i];
      b1 = tmp -> ve[i];
      b2 = tmp -> ve[i + 1];
      det = a11 * a22 - a12 * a12;
/* < 0 : see BKPfactor() */
      if (det == 0.0) 
        ev_err("bkpfacto.c",4,291,"BKPsolve",0);
      det = 1 / det;
      tmp -> ve[i] = det * (a22 * b1 - a12 * b2);
      tmp -> ve[i + 1] = det * (a11 * b2 - a12 * b1);
    }
  }
/* printf("# BKPsolve: solving D part: tmp =\n");	v_output(tmp); */
/* solve for transpose of lower traingular part */
  for (i = n - 1; i >= 0; i += -1) {
/* use symmetry of factored form to get stride 1 */
    sum = tmp -> ve[i];
    if (block -> pe[i] > i) {
      
#pragma omp parallel for private (j) reduction (-:sum)
      for (j = i + 2; j <= n - 1; j += 1) {
        sum -= A -> me[i][j] * tmp -> ve[j];
      }
    }
     else {
      
#pragma omp parallel for private (j) reduction (-:sum)
      for (j = i + 1; j <= n - 1; j += 1) {
        sum -= A -> me[i][j] * tmp -> ve[j];
      }
    }
    tmp -> ve[i] = sum;
  }
/* printf("# BKPsolve: solving L^T part: tmp =\n");v_output(tmp); */
/* and do final permutation */
  x = pxinv_vec(pivot,tmp,x);
#ifdef THREADSAFE
#endif
  return x;
}
