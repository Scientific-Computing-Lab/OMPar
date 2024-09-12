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
	File containing routines for symmetric eigenvalue problems
*/
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"
#include <omp.h> 
static char rcsid[] = "$Id: symmeig.c,v 1.6 1995/03/27 15:45:55 des Exp $";
#define	SQRT2	1.4142135623730949
#define	sgn(x)	( (x) >= 0 ? 1 : -1 )
/* trieig -- finds eigenvalues of symmetric tridiagonal matrices
	-- matrix represented by a pair of vectors a (diag entries)
		and b (sub- & super-diag entries)
	-- eigenvalues in a on return */
#ifndef ANSI_C
#else

VEC *trieig(VEC *a,VEC *b,MAT *Q)
#endif
{
  int i;
  int i_min;
  int i_max;
  int n;
  int split;
  double *a_ve;
  double *b_ve;
  double b_sqr;
  double bk;
  double ak1;
  double bk1;
  double ak2;
  double bk2;
  double z;
  double c;
  double c2;
  double cs;
  double s;
  double s2;
  double d;
  double mu;
  if (!a || !b) 
    ev_err("symmeig.c",8,62,"trieig",0);
  if (a -> dim != b -> dim + 1 || Q && Q -> m != a -> dim) 
    ev_err("symmeig.c",1,64,"trieig",0);
  if (Q && Q -> m != Q -> n) 
    ev_err("symmeig.c",9,66,"trieig",0);
  n = (a -> dim);
  a_ve = a -> ve;
  b_ve = b -> ve;
  i_min = 0;
  while(i_min < n)
/* outer while loop */
{
/* find i_max to suit;
			submatrix i_min..i_max should be irreducible */
    i_max = n - 1;
    for (i = i_min; i <= n - 1 - 1; i += 1) {
      if (b_ve[i] == 0.0) {
        i_max = i;
        break; 
      }
    }
    if (i_max <= i_min) {
/* printf("# i_min = %d, i_max = %d\n",i_min,i_max); */
      i_min = i_max + 1;
      continue; 
/* outer while loop */
    }
/* printf("# i_min = %d, i_max = %d\n",i_min,i_max); */
/* repeatedly perform QR method until matrix splits */
    split = 0;
    while(!split)
/* inner while loop */
{
/* find Wilkinson shift */
      d = (a_ve[i_max - 1] - a_ve[i_max]) / 2;
      b_sqr = b_ve[i_max - 1] * b_ve[i_max - 1];
      mu = a_ve[i_max] - b_sqr / (d + ((d >= 0?1 : - 1)) * sqrt(d * d + b_sqr));
/* printf("# Wilkinson shift = %g\n",mu); */
/* initial Givens' rotation */
      givens(a_ve[i_min] - mu,b_ve[i_min],&c,&s);
      s = -s;
/* printf("# c = %g, s = %g\n",c,s); */
      if (fabs(c) < 1.4142135623730949) {
        c2 = c * c;
        s2 = 1 - c2;
      }
       else {
        s2 = s * s;
        c2 = 1 - s2;
      }
      cs = c * s;
      ak1 = c2 * a_ve[i_min] + s2 * a_ve[i_min + 1] - 2 * cs * b_ve[i_min];
      bk1 = cs * (a_ve[i_min] - a_ve[i_min + 1]) + (c2 - s2) * b_ve[i_min];
      ak2 = s2 * a_ve[i_min] + c2 * a_ve[i_min + 1] + 2 * cs * b_ve[i_min];
      bk2 = (i_min < i_max - 1?c * b_ve[i_min + 1] : 0.0);
      z = (i_min < i_max - 1?-s * b_ve[i_min + 1] : 0.0);
      a_ve[i_min] = ak1;
      a_ve[i_min + 1] = ak2;
      b_ve[i_min] = bk1;
      if (i_min < i_max - 1) 
        b_ve[i_min + 1] = bk2;
      if (Q) 
        rot_cols(Q,i_min,(i_min + 1),c,-s,Q);
/* printf("# z = %g\n",z); */
/* printf("# a [temp1] =\n");	v_output(a); */
/* printf("# b [temp1] =\n");	v_output(b); */
      for (i = i_min + 1; i <= i_max - 1; i += 1) {
/* get Givens' rotation for sub-block -- k == i-1 */
        givens(b_ve[i - 1],z,&c,&s);
        s = -s;
/* printf("# c = %g, s = %g\n",c,s); */
/* perform Givens' rotation on sub-block */
        if (fabs(c) < 1.4142135623730949) {
          c2 = c * c;
          s2 = 1 - c2;
        }
         else {
          s2 = s * s;
          c2 = 1 - s2;
        }
        cs = c * s;
        bk = c * b_ve[i - 1] - s * z;
        ak1 = c2 * a_ve[i] + s2 * a_ve[i + 1] - 2 * cs * b_ve[i];
        bk1 = cs * (a_ve[i] - a_ve[i + 1]) + (c2 - s2) * b_ve[i];
        ak2 = s2 * a_ve[i] + c2 * a_ve[i + 1] + 2 * cs * b_ve[i];
        bk2 = (i + 1 < i_max?c * b_ve[i + 1] : 0.0);
        z = (i + 1 < i_max?-s * b_ve[i + 1] : 0.0);
        a_ve[i] = ak1;
        a_ve[i + 1] = ak2;
        b_ve[i] = bk1;
        if (i < i_max - 1) 
          b_ve[i + 1] = bk2;
        if (i > i_min) 
          b_ve[i - 1] = bk;
        if (Q) 
          rot_cols(Q,i,(i + 1),c,-s,Q);
/* printf("# a [temp2] =\n");	v_output(a); */
/* printf("# b [temp2] =\n");	v_output(b); */
      }
/* test to see if matrix should be split */
      for (i = i_min; i <= i_max - 1; i += 1) {
        if (fabs(b_ve[i]) < ((double )2.22044604925031308084726333618164062e-16L) * (fabs(a_ve[i]) + fabs(a_ve[i + 1]))) {
          b_ve[i] = 0.0;
          split = 1;
        }
/* printf("# a =\n");	v_output(a); */
/* printf("# b =\n");	v_output(b); */
      }
    }
  }
  return a;
}
/* symmeig -- computes eigenvalues of a dense symmetric matrix
	-- A **must** be symmetric on entry
	-- eigenvalues stored in out
	-- Q contains orthogonal matrix of eigenvectors
	-- returns vector of eigenvalues */
#ifndef ANSI_C
#else

VEC *symmeig(const MAT *A,MAT *Q,VEC *out)
#endif
{
  int i;
  static MAT *tmp = (MAT *)((void *)0);
  static VEC *b = (VEC *)((void *)0);
  static VEC *diag = (VEC *)((void *)0);
  static VEC *beta = (VEC *)((void *)0);
  if (!A) 
    ev_err("symmeig.c",8,190,"symmeig",0);
  if (A -> m != A -> n) 
    ev_err("symmeig.c",9,192,"symmeig",0);
  if (!out || out -> dim != A -> m) 
    out = v_resize(out,(A -> m));
  tmp = m_resize(tmp,(A -> m),(A -> n));
  tmp = _m_copy(A,tmp,0,0);
  b = v_resize(b,(A -> m - 1));
  diag = v_resize(diag,((unsigned int )(A -> m)));
  beta = v_resize(beta,((unsigned int )(A -> m)));
  mem_stat_reg_list((void **)(&tmp),0,0,"symmeig.c",201);
  mem_stat_reg_list((void **)(&b),3,0,"symmeig.c",202);
  mem_stat_reg_list((void **)(&diag),3,0,"symmeig.c",203);
  mem_stat_reg_list((void **)(&beta),3,0,"symmeig.c",204);
  Hfactor(tmp,diag,beta);
  if (Q) 
    makeHQ(tmp,diag,beta,Q);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= A -> m - ((unsigned int )1) - 1; i += 1) {
    out -> ve[i] = tmp -> me[i][i];
    b -> ve[i] = tmp -> me[i][i + 1];
  }
  out -> ve[i] = tmp -> me[i][i];
  trieig(out,b,Q);
#ifdef	THREADSAFE
#endif
  return out;
}
