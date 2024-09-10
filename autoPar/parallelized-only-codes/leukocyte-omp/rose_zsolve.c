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
	Complex case
*/
static char rcsid[] = "$Id: zsolve.c,v 1.1 1994/01/13 04:20:33 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include        "zmatrix2.h"
#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0 )
/* Most matrix factorisation routines are in-situ unless otherwise specified */
/* zUsolve -- back substitution with optional over-riding diagonal
		-- can be in-situ but doesn't need to be */

ZVEC *zUsolve(ZMAT *matrix,ZVEC *b,ZVEC *out,double diag)
{
  unsigned int dim;
/* , j */
  int i;
  int i_lim;
  complex **mat_ent;
  complex *mat_row;
  complex *b_ent;
  complex *out_ent;
  complex *out_col;
  complex sum;
  if (matrix == ((ZMAT *)((void *)0)) || b == ((ZVEC *)((void *)0))) 
    ev_err("zsolve.c",8,56,"zUsolve",0);
  dim = (matrix -> m > matrix -> n?matrix -> n : matrix -> m);
  if (b -> dim < dim) 
    ev_err("zsolve.c",1,59,"zUsolve",0);
  if (out == ((ZVEC *)((void *)0)) || out -> dim < dim) 
    out = zv_resize(out,(matrix -> n));
  mat_ent = matrix -> me;
  b_ent = b -> ve;
  out_ent = out -> ve;
  for (i = (dim - 1); i >= 0; i += -1) {
    if (!(b_ent[i] . re == 0.0 && b_ent[i] . im == 0.0)) 
      break; 
     else 
      out_ent[i] . re = out_ent[i] . im = 0.0;
  }
  i_lim = i;
  for (i = i_lim; i >= 0; i += -1) {
    sum = b_ent[i];
    mat_row = &mat_ent[i][i + 1];
    out_col = &out_ent[i + 1];
    sum = zsub(sum,(__zip__(mat_row,out_col,i_lim - i,0)));
/******************************************************
	  for ( j=i+1; j<=i_lim; j++ )
	  sum -= mat_ent[i][j]*out_ent[j];
	  sum -= (*mat_row++)*(*out_col++);
	******************************************************/
    if (diag == 0.0) {
      if (mat_ent[i][i] . re == 0.0 && mat_ent[i][i] . im == 0.0) 
        ev_err("zsolve.c",4,85,"zUsolve",0);
       else 
/* out_ent[i] = sum/mat_ent[i][i]; */
        out_ent[i] = zdiv(sum,mat_ent[i][i]);
    }
     else {
/* out_ent[i] = sum/diag; */
      out_ent[i] . re = sum . re / diag;
      out_ent[i] . im = sum . im / diag;
    }
  }
  return out;
}
/* zLsolve -- forward elimination with (optional) default diagonal value */

ZVEC *zLsolve(ZMAT *matrix,ZVEC *b,ZVEC *out,double diag)
{
  unsigned int dim;
  unsigned int i;
  unsigned int i_lim;
/* , j */
  complex **mat_ent;
  complex *mat_row;
  complex *b_ent;
  complex *out_ent;
  complex *out_col;
  complex sum;
  if (matrix == ((ZMAT *)((void *)0)) || b == ((ZVEC *)((void *)0))) 
    ev_err("zsolve.c",8,112,"zLsolve",0);
  dim = (matrix -> m > matrix -> n?matrix -> n : matrix -> m);
  if (b -> dim < dim) 
    ev_err("zsolve.c",1,115,"zLsolve",0);
  if (out == ((ZVEC *)((void *)0)) || out -> dim < dim) 
    out = zv_resize(out,(matrix -> n));
  mat_ent = matrix -> me;
  b_ent = b -> ve;
  out_ent = out -> ve;
  for (i = 0; i <= dim - 1; i += 1) {
    if (!(b_ent[i] . re == 0.0 && b_ent[i] . im == 0.0)) 
      break; 
     else 
      out_ent[i] . re = out_ent[i] . im = 0.0;
  }
  i_lim = i;
  for (i = i_lim; i <= dim - 1; i += 1) {
    sum = b_ent[i];
    mat_row = &mat_ent[i][i_lim];
    out_col = &out_ent[i_lim];
    sum = zsub(sum,(__zip__(mat_row,out_col,(int )(i - i_lim),0)));
/*****************************************************
	  for ( j=i_lim; j<i; j++ )
	  sum -= mat_ent[i][j]*out_ent[j];
	  sum -= (*mat_row++)*(*out_col++);
	******************************************************/
    if (diag == 0.0) {
      if (mat_ent[i][i] . re == 0.0 && mat_ent[i][i] . im == 0.0) 
        ev_err("zsolve.c",4,141,"zLsolve",0);
       else 
        out_ent[i] = zdiv(sum,mat_ent[i][i]);
    }
     else {
      out_ent[i] . re = sum . re / diag;
      out_ent[i] . im = sum . im / diag;
    }
  }
  return out;
}
/* zUAsolve -- forward elimination with (optional) default diagonal value
		using UPPER triangular part of matrix */

ZVEC *zUAsolve(ZMAT *U,ZVEC *b,ZVEC *out,double diag)
{
  unsigned int dim;
  unsigned int i;
  unsigned int i_lim;
/* , j */
  complex **U_me;
  complex *b_ve;
  complex *out_ve;
  complex tmp;
  double invdiag;
  if (!U || !b) 
    ev_err("zsolve.c",8,169,"zUAsolve",0);
  dim = (U -> m > U -> n?U -> n : U -> m);
  if (b -> dim < dim) 
    ev_err("zsolve.c",1,172,"zUAsolve",0);
  out = zv_resize(out,(U -> n));
  U_me = U -> me;
  b_ve = b -> ve;
  out_ve = out -> ve;
  for (i = 0; i <= dim - 1; i += 1) {
    if (!(b_ve[i] . re == 0.0 && b_ve[i] . im == 0.0)) 
      break; 
     else 
      out_ve[i] . re = out_ve[i] . im = 0.0;
  }
  i_lim = i;
  if (b != out) {
    __zzero__(out_ve,(out -> dim));
/* MEM_COPY(&(b_ve[i_lim]),&(out_ve[i_lim]),
	   (dim-i_lim)*sizeof(complex)); */
    memmove(((char *)(&out_ve[i_lim])),((char *)(&b_ve[i_lim])),((unsigned int )(dim - i_lim)) * sizeof(complex ));
  }
  if (diag == 0.0) {
    for (; i <= dim - 1; i += 1) {
      tmp = zconj(U_me[i][i]);
      if (tmp . re == 0.0 && tmp . im == 0.0) 
        ev_err("zsolve.c",4,196,"zUAsolve",0);
/* out_ve[i] /= tmp; */
      out_ve[i] = zdiv(out_ve[i],tmp);
      tmp . re = -out_ve[i] . re;
      tmp . im = -out_ve[i] . im;
      __zmltadd__(&out_ve[i + 1],(&U_me[i][i + 1]),tmp,(dim - i - 1),1);
    }
  }
   else {
    invdiag = 1.0 / diag;
    for (; i <= dim - 1; i += 1) {
      out_ve[i] . re *= invdiag;
      out_ve[i] . im *= invdiag;
      tmp . re = -out_ve[i] . re;
      tmp . im = -out_ve[i] . im;
      __zmltadd__(&out_ve[i + 1],(&U_me[i][i + 1]),tmp,(dim - i - 1),1);
    }
  }
  return out;
}
/* zDsolve -- solves Dx=b where D is the diagonal of A -- may be in-situ */

ZVEC *zDsolve(ZMAT *A,ZVEC *b,ZVEC *x)
{
  unsigned int dim;
  unsigned int i;
  if (!A || !b) 
    ev_err("zsolve.c",8,228,"zDsolve",0);
  dim = (A -> m > A -> n?A -> n : A -> m);
  if (b -> dim < dim) 
    ev_err("zsolve.c",1,231,"zDsolve",0);
  x = zv_resize(x,(A -> n));
  dim = b -> dim;
  for (i = 0; i <= dim - 1; i += 1) {
    if (A -> me[i][i] . re == 0.0 && A -> me[i][i] . im == 0.0) 
      ev_err("zsolve.c",4,237,"zDsolve",0);
     else 
      x -> ve[i] = zdiv(b -> ve[i],A -> me[i][i]);
  }
  return x;
}
/* zLAsolve -- back substitution with optional over-riding diagonal
		using the LOWER triangular part of matrix
		-- can be in-situ but doesn't need to be */

ZVEC *zLAsolve(ZMAT *L,ZVEC *b,ZVEC *out,double diag)
{
  unsigned int dim;
  int i;
  int i_lim;
  complex **L_me;
  complex *b_ve;
  complex *out_ve;
  complex tmp;
  double invdiag;
  if (!L || !b) 
    ev_err("zsolve.c",8,259,"zLAsolve",0);
  dim = (L -> m > L -> n?L -> n : L -> m);
  if (b -> dim < dim) 
    ev_err("zsolve.c",1,262,"zLAsolve",0);
  out = zv_resize(out,(L -> n));
  L_me = L -> me;
  b_ve = b -> ve;
  out_ve = out -> ve;
  for (i = (dim - 1); i >= 0; i += -1) {
    if (!(b_ve[i] . re == 0.0 && b_ve[i] . im == 0.0)) 
      break; 
  }
  i_lim = i;
  if (b != out) {
    __zzero__(out_ve,(out -> dim));
/* MEM_COPY(b_ve,out_ve,(i_lim+1)*sizeof(complex)); */
    memmove(((char *)out_ve),((char *)b_ve),((unsigned int )(i_lim + 1)) * sizeof(complex ));
  }
  if (diag == 0.0) {
    for (; i >= 0; i += -1) {
      tmp = zconj(L_me[i][i]);
      if (tmp . re == 0.0 && tmp . im == 0.0) 
        ev_err("zsolve.c",4,284,"zLAsolve",0);
      out_ve[i] = zdiv(out_ve[i],tmp);
      tmp . re = -out_ve[i] . re;
      tmp . im = -out_ve[i] . im;
      __zmltadd__(out_ve,L_me[i],tmp,i,1);
    }
  }
   else {
    invdiag = 1.0 / diag;
    for (; i >= 0; i += -1) {
      out_ve[i] . re *= invdiag;
      out_ve[i] . im *= invdiag;
      tmp . re = -out_ve[i] . re;
      tmp . im = -out_ve[i] . im;
      __zmltadd__(out_ve,L_me[i],tmp,i,1);
    }
  }
  return out;
}
