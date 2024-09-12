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
/* solve.c 1.2 11/25/87 */
static char rcsid[] = "$Id: solve.c,v 1.3 1994/01/13 05:29:57 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include        "matrix2.h"
/* Most matrix factorisation routines are in-situ unless otherwise specified */
/* Usolve -- back substitution with optional over-riding diagonal
		-- can be in-situ but doesn't need to be */
#ifndef ANSI_C
#else

VEC *Usolve(const MAT *matrix,const VEC *b,VEC *out,double diag)
#endif
{
  unsigned int dim;
/* , j */
  int i;
  int i_lim;
  double **mat_ent;
  double *mat_row;
  double *b_ent;
  double *out_ent;
  double *out_col;
  double sum;
  double tiny;
  if (matrix == ((MAT *)((void *)0)) || b == ((VEC *)((void *)0))) 
    ev_err("solve.c",8,60,"Usolve",0);
  dim = (matrix -> m > matrix -> n?matrix -> n : matrix -> m);
  if (b -> dim < dim) 
    ev_err("solve.c",1,63,"Usolve",0);
  if (out == ((VEC *)((void *)0)) || out -> dim < dim) 
    out = v_resize(out,(matrix -> n));
  mat_ent = matrix -> me;
  b_ent = b -> ve;
  out_ent = out -> ve;
  tiny = 10.0 / __builtin_huge_val();
  for (i = (dim - 1); i >= 0; i += -1) {
    if (b_ent[i] != 0.0) 
      break; 
     else 
      out_ent[i] = 0.0;
  }
  i_lim = i;
  for (; i >= 0; i += -1) {
    sum = b_ent[i];
    mat_row = &mat_ent[i][i + 1];
    out_col = &out_ent[i + 1];
    sum -= __ip__(mat_row,out_col,i_lim - i);
/******************************************************
		for ( j=i+1; j<=i_lim; j++ )
			sum -= mat_ent[i][j]*out_ent[j];
			sum -= (*mat_row++)*(*out_col++);
		******************************************************/
    if (diag == 0.0) {
      if (fabs(mat_ent[i][i]) <= tiny * fabs(sum)) 
        ev_err("solve.c",4,91,"Usolve",0);
       else 
        out_ent[i] = sum / mat_ent[i][i];
    }
     else 
      out_ent[i] = sum / diag;
  }
  return out;
}
/* Lsolve -- forward elimination with (optional) default diagonal value */
#ifndef ANSI_C
#else

VEC *Lsolve(const MAT *matrix,const VEC *b,VEC *out,double diag)
#endif
{
  unsigned int dim;
  unsigned int i;
  unsigned int i_lim;
/* , j */
  double **mat_ent;
  double *mat_row;
  double *b_ent;
  double *out_ent;
  double *out_col;
  double sum;
  double tiny;
  if (matrix == ((MAT *)((void *)0)) || b == ((VEC *)((void *)0))) 
    ev_err("solve.c",8,116,"Lsolve",0);
  dim = (matrix -> m > matrix -> n?matrix -> n : matrix -> m);
  if (b -> dim < dim) 
    ev_err("solve.c",1,119,"Lsolve",0);
  if (out == ((VEC *)((void *)0)) || out -> dim < dim) 
    out = v_resize(out,(matrix -> n));
  mat_ent = matrix -> me;
  b_ent = b -> ve;
  out_ent = out -> ve;
  for (i = 0; i <= dim - 1; i += 1) {
    if (b_ent[i] != 0.0) 
      break; 
     else 
      out_ent[i] = 0.0;
  }
  i_lim = i;
  tiny = 10.0 / __builtin_huge_val();
  for (; i <= dim - 1; i += 1) {
    sum = b_ent[i];
    mat_row = &mat_ent[i][i_lim];
    out_col = &out_ent[i_lim];
    sum -= __ip__(mat_row,out_col,(int )(i - i_lim));
/*****************************************************
		for ( j=i_lim; j<i; j++ )
			sum -= mat_ent[i][j]*out_ent[j];
			sum -= (*mat_row++)*(*out_col++);
		******************************************************/
    if (diag == 0.0) {
      if (fabs(mat_ent[i][i]) <= tiny * fabs(sum)) 
        ev_err("solve.c",4,147,"Lsolve",0);
       else 
        out_ent[i] = sum / mat_ent[i][i];
    }
     else 
      out_ent[i] = sum / diag;
  }
  return out;
}
/* UTsolve -- forward elimination with (optional) default diagonal value
		using UPPER triangular part of matrix */
#ifndef ANSI_C
#else

VEC *UTsolve(const MAT *U,const VEC *b,VEC *out,double diag)
#endif
{
  unsigned int dim;
  unsigned int i;
  unsigned int i_lim;
  double **U_me;
  double *b_ve;
  double *out_ve;
  double tmp;
  double invdiag;
  double tiny;
  if (!U || !b) 
    ev_err("solve.c",8,174,"UTsolve",0);
  dim = (U -> m > U -> n?U -> n : U -> m);
  if (b -> dim < dim) 
    ev_err("solve.c",1,177,"UTsolve",0);
  out = v_resize(out,(U -> n));
  U_me = U -> me;
  b_ve = b -> ve;
  out_ve = out -> ve;
  tiny = 10.0 / __builtin_huge_val();
  for (i = 0; i <= dim - 1; i += 1) {
    if (b_ve[i] != 0.0) 
      break; 
     else 
      out_ve[i] = 0.0;
  }
  i_lim = i;
  if (b != out) {
    __zero__(out_ve,(out -> dim));
    memmove((&out_ve[i_lim]),(&b_ve[i_lim]),(dim - i_lim) * sizeof(double ));
  }
  if (diag == 0.0) {
    for (; i <= dim - 1; i += 1) {
      tmp = U_me[i][i];
      if (fabs(tmp) <= tiny * fabs(out_ve[i])) 
        ev_err("solve.c",4,201,"UTsolve",0);
      out_ve[i] /= tmp;
      __mltadd__(&out_ve[i + 1],(&U_me[i][i + 1]),-out_ve[i],(dim - i - 1));
    }
  }
   else {
    invdiag = 1.0 / diag;
    for (; i <= dim - 1; i += 1) {
      out_ve[i] *= invdiag;
      __mltadd__(&out_ve[i + 1],(&U_me[i][i + 1]),-out_ve[i],(dim - i - 1));
    }
  }
  return out;
}
/* Dsolve -- solves Dx=b where D is the diagonal of A -- may be in-situ */
#ifndef ANSI_C
#else

VEC *Dsolve(const MAT *A,const VEC *b,VEC *x)
#endif
{
  unsigned int dim;
  unsigned int i;
  double tiny;
  if (!A || !b) 
    ev_err("solve.c",8,231,"Dsolve",0);
  dim = (A -> m > A -> n?A -> n : A -> m);
  if (b -> dim < dim) 
    ev_err("solve.c",1,234,"Dsolve",0);
  x = v_resize(x,(A -> n));
  tiny = 10.0 / __builtin_huge_val();
  dim = b -> dim;
  for (i = 0; i <= dim - 1; i += 1) {
    if (fabs(A -> me[i][i]) <= tiny * fabs(b -> ve[i])) 
      ev_err("solve.c",4,242,"Dsolve",0);
     else 
      x -> ve[i] = b -> ve[i] / A -> me[i][i];
  }
  return x;
}
/* LTsolve -- back substitution with optional over-riding diagonal
		using the LOWER triangular part of matrix
		-- can be in-situ but doesn't need to be */
#ifndef ANSI_C
#else

VEC *LTsolve(const MAT *L,const VEC *b,VEC *out,double diag)
#endif
{
  unsigned int dim;
  int i;
  int i_lim;
  double **L_me;
  double *b_ve;
  double *out_ve;
  double tmp;
  double invdiag;
  double tiny;
  if (!L || !b) 
    ev_err("solve.c",8,266,"LTsolve",0);
  dim = (L -> m > L -> n?L -> n : L -> m);
  if (b -> dim < dim) 
    ev_err("solve.c",1,269,"LTsolve",0);
  out = v_resize(out,(L -> n));
  L_me = L -> me;
  b_ve = b -> ve;
  out_ve = out -> ve;
  tiny = 10.0 / __builtin_huge_val();
  for (i = (dim - 1); i >= 0; i += -1) {
    if (b_ve[i] != 0.0) 
      break; 
  }
  i_lim = i;
  if (b != out) {
    __zero__(out_ve,(out -> dim));
    memmove(out_ve,b_ve,(i_lim + 1) * sizeof(double ));
  }
  if (diag == 0.0) {
    for (; i >= 0; i += -1) {
      tmp = L_me[i][i];
      if (fabs(tmp) <= tiny * fabs(out_ve[i])) 
        ev_err("solve.c",4,292,"LTsolve",0);
      out_ve[i] /= tmp;
      __mltadd__(out_ve,L_me[i],-out_ve[i],i);
    }
  }
   else {
    invdiag = 1.0 / diag;
    for (; i >= 0; i += -1) {
      out_ve[i] *= invdiag;
      __mltadd__(out_ve,L_me[i],-out_ve[i],i);
    }
  }
  return out;
}
