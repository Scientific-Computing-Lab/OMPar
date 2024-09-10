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
#include	<stdio.h>
#include	"zmatrix.h"
#include <omp.h> 
static char rcsid[] = "$Id: zmatop.c,v 1.2 1995/03/27 15:49:03 des Exp $";
#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)
/* zm_add -- matrix addition -- may be in-situ */

ZMAT *zm_add(ZMAT *mat1,ZMAT *mat2,ZMAT *out)
{
  unsigned int m;
  unsigned int n;
  unsigned int i;
  if (mat1 == ((ZMAT *)((void *)0)) || mat2 == ((ZMAT *)((void *)0))) 
    ev_err("zmatop.c",8,42,"zm_add",0);
  if (mat1 -> m != mat2 -> m || mat1 -> n != mat2 -> n) 
    ev_err("zmatop.c",1,44,"zm_add",0);
  if (out == ((ZMAT *)((void *)0)) || out -> m != mat1 -> m || out -> n != mat1 -> n) 
    out = zm_resize(out,(mat1 -> m),(mat1 -> n));
  m = mat1 -> m;
  n = mat1 -> n;
  for (i = 0; i <= m - 1; i += 1) {
    __zadd__(mat1 -> me[i],mat2 -> me[i],out -> me[i],(int )n);
/**************************************************
	  for ( j=0; j<n; j++ )
	  out->me[i][j] = mat1->me[i][j]+mat2->me[i][j];
	  **************************************************/
  }
  return out;
}
/* zm_sub -- matrix subtraction -- may be in-situ */

ZMAT *zm_sub(ZMAT *mat1,ZMAT *mat2,ZMAT *out)
{
  unsigned int m;
  unsigned int n;
  unsigned int i;
  if (mat1 == ((ZMAT *)((void *)0)) || mat2 == ((ZMAT *)((void *)0))) 
    ev_err("zmatop.c",8,66,"zm_sub",0);
  if (mat1 -> m != mat2 -> m || mat1 -> n != mat2 -> n) 
    ev_err("zmatop.c",1,68,"zm_sub",0);
  if (out == ((ZMAT *)((void *)0)) || out -> m != mat1 -> m || out -> n != mat1 -> n) 
    out = zm_resize(out,(mat1 -> m),(mat1 -> n));
  m = mat1 -> m;
  n = mat1 -> n;
  for (i = 0; i <= m - 1; i += 1) {
    __zsub__(mat1 -> me[i],mat2 -> me[i],out -> me[i],(int )n);
/**************************************************
	  for ( j=0; j<n; j++ )
	  out->me[i][j] = mat1->me[i][j]-mat2->me[i][j];
	**************************************************/
  }
  return out;
}
/*
  Note: In the following routines, "adjoint" means complex conjugate
  transpose:
  A* = conjugate(A^T)
  */
/* zm_mlt -- matrix-matrix multiplication */

ZMAT *zm_mlt(ZMAT *A,ZMAT *B,ZMAT *OUT)
{
  unsigned int i;
/* j, */
  unsigned int k;
  unsigned int m;
  unsigned int n;
  unsigned int p;
  complex **A_v;
  complex **B_v;
/*, *B_row, *OUT_row, sum, tmp */
  if (A == ((ZMAT *)((void *)0)) || B == ((ZMAT *)((void *)0))) 
    ev_err("zmatop.c",8,97,"zm_mlt",0);
  if (A -> n != B -> m) 
    ev_err("zmatop.c",1,99,"zm_mlt",0);
  if (A == OUT || B == OUT) 
    ev_err("zmatop.c",12,101,"zm_mlt",0);
  m = A -> m;
  n = A -> n;
  p = B -> n;
  A_v = A -> me;
  B_v = B -> me;
  if (OUT == ((ZMAT *)((void *)0)) || OUT -> m != A -> m || OUT -> n != B -> n) 
    OUT = zm_resize(OUT,(A -> m),(B -> n));
/****************************************************************
      for ( i=0; i<m; i++ )
      for  ( j=0; j<p; j++ )
      {
      sum = 0.0;
      for ( k=0; k<n; k++ )
      sum += A_v[i][k]*B_v[k][j];
      OUT->me[i][j] = sum;
      }
    ****************************************************************/
  zm_zero(OUT);
  for (i = 0; i <= m - 1; i += 1) {
    for (k = 0; k <= n - 1; k += 1) {
      if (!(A_v[i][k] . re == 0.0 && A_v[i][k] . im == 0.0)) 
        __zmltadd__(OUT -> me[i],B_v[k],A_v[i][k],(int )p,0);
/**************************************************
	      B_row = B_v[k];	OUT_row = OUT->me[i];
	      for ( j=0; j<p; j++ )
	      (*OUT_row++) += tmp*(*B_row++);
	    **************************************************/
    }
  }
  return OUT;
}
/* zmma_mlt -- matrix-matrix adjoint multiplication
   -- A.B* is returned, and stored in OUT */

ZMAT *zmma_mlt(ZMAT *A,ZMAT *B,ZMAT *OUT)
{
  int i;
  int j;
  int limit;
/* complex	*A_row, *B_row, sum; */
  if (!A || !B) 
    ev_err("zmatop.c",8,142,"zmma_mlt",0);
  if (A == OUT || B == OUT) 
    ev_err("zmatop.c",12,144,"zmma_mlt",0);
  if (A -> n != B -> n) 
    ev_err("zmatop.c",1,146,"zmma_mlt",0);
  if (!OUT || OUT -> m != A -> m || OUT -> n != B -> m) 
    OUT = zm_resize(OUT,(A -> m),(B -> m));
  limit = (A -> n);
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
    for (j = 0; ((unsigned int )j) <= B -> m - 1; j += 1) {
      OUT -> me[i][j] = __zip__(B -> me[j],A -> me[i],(int )limit,1);
/**************************************************
	      sum = 0.0;
	      A_row = A->me[i];
	      B_row = B->me[j];
	      for ( k = 0; k < limit; k++ )
	      sum += (*A_row++)*(*B_row++);
	      OUT->me[i][j] = sum;
	      **************************************************/
    }
  }
  return OUT;
}
/* zmam_mlt -- matrix adjoint-matrix multiplication
   -- A*.B is returned, result stored in OUT */

ZMAT *zmam_mlt(ZMAT *A,ZMAT *B,ZMAT *OUT)
{
  int i;
  int k;
  int limit;
/* complex	*B_row, *OUT_row, multiplier; */
  complex tmp;
  if (!A || !B) 
    ev_err("zmatop.c",8,177,"zmam_mlt",0);
  if (A == OUT || B == OUT) 
    ev_err("zmatop.c",12,179,"zmam_mlt",0);
  if (A -> m != B -> m) 
    ev_err("zmatop.c",1,181,"zmam_mlt",0);
  if (!OUT || OUT -> m != A -> n || OUT -> n != B -> n) 
    OUT = zm_resize(OUT,(A -> n),(B -> n));
  limit = (B -> n);
  zm_zero(OUT);
  for (k = 0; ((unsigned int )k) <= A -> m - 1; k += 1) {
    for (i = 0; ((unsigned int )i) <= A -> n - 1; i += 1) {
      tmp . re = A -> me[k][i] . re;
      tmp . im = -A -> me[k][i] . im;
      if (!(tmp . re == 0.0 && tmp . im == 0.0)) 
        __zmltadd__(OUT -> me[i],B -> me[k],tmp,(int )limit,0);
    }
  }
  return OUT;
}
/* zmv_mlt -- matrix-vector multiplication 
   -- Note: b is treated as a column vector */

ZVEC *zmv_mlt(ZMAT *A,ZVEC *b,ZVEC *out)
{
  unsigned int i;
  unsigned int m;
  unsigned int n;
  complex **A_v;
  complex *b_v;
/*, *A_row */
/* register complex	sum; */
  if (A == ((ZMAT *)((void *)0)) || b == ((ZVEC *)((void *)0))) 
    ev_err("zmatop.c",8,208,"zmv_mlt",0);
  if (A -> n != b -> dim) 
    ev_err("zmatop.c",1,210,"zmv_mlt",0);
  if (b == out) 
    ev_err("zmatop.c",12,212,"zmv_mlt",0);
  if (out == ((ZVEC *)((void *)0)) || out -> dim != A -> m) 
    out = zv_resize(out,(A -> m));
  m = A -> m;
  n = A -> n;
  A_v = A -> me;
  b_v = b -> ve;
  for (i = 0; i <= m - 1; i += 1) {
/* for ( j=0; j<n; j++ )
	   sum += A_v[i][j]*b_v[j]; */
    out -> ve[i] = __zip__(A_v[i],b_v,(int )n,0);
/**************************************************
	  A_row = A_v[i];		b_v = b->ve;
	  for ( j=0; j<n; j++ )
	  sum += (*A_row++)*(*b_v++);
	  out->ve[i] = sum;
	**************************************************/
  }
  return out;
}
/* zsm_mlt -- scalar-matrix multiply -- may be in-situ */

ZMAT *zsm_mlt(complex scalar,ZMAT *matrix,ZMAT *out)
{
  unsigned int m;
  unsigned int n;
  unsigned int i;
  if (matrix == ((ZMAT *)((void *)0))) 
    ev_err("zmatop.c",8,240,"zsm_mlt",0);
  if (out == ((ZMAT *)((void *)0)) || out -> m != matrix -> m || out -> n != matrix -> n) 
    out = zm_resize(out,(matrix -> m),(matrix -> n));
  m = matrix -> m;
  n = matrix -> n;
  for (i = 0; i <= m - 1; i += 1) {
    __zmlt__(matrix -> me[i],scalar,out -> me[i],(int )n);
  }
/**************************************************
      for ( j=0; j<n; j++ )
      out->me[i][j] = scalar*matrix->me[i][j];
      **************************************************/
  return out;
}
/* zvm_mlt -- vector adjoint-matrix multiplication */

ZVEC *zvm_mlt(ZMAT *A,ZVEC *b,ZVEC *out)
{
  unsigned int j;
  unsigned int m;
  unsigned int n;
/* complex	sum,**A_v,*b_v; */
  if (A == ((ZMAT *)((void *)0)) || b == ((ZVEC *)((void *)0))) 
    ev_err("zmatop.c",8,260,"zvm_mlt",0);
  if (A -> m != b -> dim) 
    ev_err("zmatop.c",1,262,"zvm_mlt",0);
  if (b == out) 
    ev_err("zmatop.c",12,264,"zvm_mlt",0);
  if (out == ((ZVEC *)((void *)0)) || out -> dim != A -> n) 
    out = zv_resize(out,(A -> n));
  m = A -> m;
  n = A -> n;
  zv_zero(out);
  for (j = 0; j <= m - 1; j += 1) {
    if (b -> ve[j] . re != 0.0 || b -> ve[j] . im != 0.0) 
      __zmltadd__(out -> ve,A -> me[j],b -> ve[j],(int )n,1);
  }
/**************************************************
      A_v = A->me;		b_v = b->ve;
      for ( j=0; j<n; j++ )
      {
      sum = 0.0;
      for ( i=0; i<m; i++ )
      sum += b_v[i]*A_v[i][j];
      out->ve[j] = sum;
      }
      **************************************************/
  return out;
}
/* zm_adjoint -- adjoint matrix */

ZMAT *zm_adjoint(ZMAT *in,ZMAT *out)
{
  int i;
  int j;
  int in_situ;
  complex tmp;
  if (in == ((ZMAT *)((void *)0))) 
    ev_err("zmatop.c",8,296,"zm_adjoint",0);
  if (in == out && in -> n != in -> m) 
    ev_err("zmatop.c",11,298,"zm_adjoint",0);
  in_situ = in == out;
  if (out == ((ZMAT *)((void *)0)) || out -> m != in -> n || out -> n != in -> m) 
    out = zm_resize(out,(in -> n),(in -> m));
  if (!in_situ) {
    
#pragma omp parallel for private (i,j)
    for (i = 0; ((unsigned int )i) <= in -> m - 1; i += 1) {
      
#pragma omp parallel for private (j)
      for (j = 0; ((unsigned int )j) <= in -> n - 1; j += 1) {
        out -> me[j][i] . re = in -> me[i][j] . re;
        out -> me[j][i] . im = -in -> me[i][j] . im;
      }
    }
  }
   else {
    
#pragma omp parallel for private (i,j)
    for (i = 0; ((unsigned int )i) <= in -> m - 1; i += 1) {
      for (j = 0; j <= i - 1; j += 1) {
        tmp . re = in -> me[i][j] . re;
        tmp . im = in -> me[i][j] . im;
        in -> me[i][j] . re = in -> me[j][i] . re;
        in -> me[i][j] . im = -in -> me[j][i] . im;
        in -> me[j][i] . re = tmp . re;
        in -> me[j][i] . im = -tmp . im;
      }
      in -> me[i][i] . im = -in -> me[i][i] . im;
    }
  }
  return out;
}
/* zswap_rows -- swaps rows i and j of matrix A upto column lim */

ZMAT *zswap_rows(ZMAT *A,int i,int j,int lo,int hi)
{
  int k;
  complex **A_me;
  complex tmp;
  if (!A) 
    ev_err("zmatop.c",8,339,"swap_rows",0);
  if (i < 0 || j < 0 || i >= A -> m || j >= A -> m) 
    ev_err("zmatop.c",1,341,"swap_rows",0);
  lo = (0 > lo?0 : lo);
  hi = ((hi > A -> n - 1?A -> n - 1 : hi));
  A_me = A -> me;
  for (k = lo; k <= hi; k += 1) {
    tmp = A_me[k][i];
    A_me[k][i] = A_me[k][j];
    A_me[k][j] = tmp;
  }
  return A;
}
/* zswap_cols -- swap columns i and j of matrix A upto row lim */

ZMAT *zswap_cols(ZMAT *A,int i,int j,int lo,int hi)
{
  int k;
  complex **A_me;
  complex tmp;
  if (!A) 
    ev_err("zmatop.c",8,362,"swap_cols",0);
  if (i < 0 || j < 0 || i >= A -> n || j >= A -> n) 
    ev_err("zmatop.c",1,364,"swap_cols",0);
  lo = (0 > lo?0 : lo);
  hi = ((hi > A -> m - 1?A -> m - 1 : hi));
  A_me = A -> me;
  for (k = lo; k <= hi; k += 1) {
    tmp = A_me[i][k];
    A_me[i][k] = A_me[j][k];
    A_me[j][k] = tmp;
  }
  return A;
}
/* mz_mltadd -- matrix-scalar multiply and add
   -- may be in situ
   -- returns out == A1 + s*A2 */

ZMAT *mz_mltadd(ZMAT *A1,ZMAT *A2,complex s,ZMAT *out)
{
/* register complex	*A1_e, *A2_e, *out_e; */
/* register int	j; */
  int i;
  int m;
  int n;
  if (!A1 || !A2) 
    ev_err("zmatop.c",8,388,"mz_mltadd",0);
  if (A1 -> m != A2 -> m || A1 -> n != A2 -> n) 
    ev_err("zmatop.c",1,390,"mz_mltadd",0);
  if (out != A1 && out != A2) 
    out = zm_resize(out,(A1 -> m),(A1 -> n));
  if (s . re == 0.0 && s . im == 0.0) 
    return _zm_copy(A1,out,0,0);
  if (s . re == 1.0 && s . im == 0.0) 
    return zm_add(A1,A2,out);
  out = _zm_copy(A1,out,0,0);
  m = (A1 -> m);
  n = (A1 -> n);
  for (i = 0; i <= m - 1; i += 1) {
    __zmltadd__(out -> me[i],A2 -> me[i],s,(int )n,0);
/**************************************************
	  A1_e = A1->me[i];
	  A2_e = A2->me[i];
	  out_e = out->me[i];
	  for ( j = 0; j < n; j++ )
	  out_e[j] = A1_e[j] + s*A2_e[j];
	  **************************************************/
  }
  return out;
}
/* zmv_mltadd -- matrix-vector multiply and add
   -- may not be in situ
   -- returns out == v1 + alpha*A*v2 */

ZVEC *zmv_mltadd(ZVEC *v1,ZVEC *v2,ZMAT *A,complex alpha,ZVEC *out)
{
/* register	int	j; */
  int i;
  int m;
  int n;
  complex tmp;
  complex *v2_ve;
  complex *out_ve;
  if (!v1 || !v2 || !A) 
    ev_err("zmatop.c",8,428,"zmv_mltadd",0);
  if (out == v2) 
    ev_err("zmatop.c",12,430,"zmv_mltadd",0);
  if (v1 -> dim != A -> m || v2 -> dim != A -> n) 
    ev_err("zmatop.c",1,432,"zmv_mltadd",0);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      out = _zv_copy(v1,out,0);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("zmatop.c",_err_num,434,"zmv_mltadd",0);
    }
  }
  ;
  v2_ve = v2 -> ve;
  out_ve = out -> ve;
  m = (A -> m);
  n = (A -> n);
  if (alpha . re == 0.0 && alpha . im == 0.0) 
    return out;
  for (i = 0; i <= m - 1; i += 1) {
    tmp = __zip__(A -> me[i],v2_ve,(int )n,0);
    out_ve[i] . re += alpha . re * tmp . re - alpha . im * tmp . im;
    out_ve[i] . im += alpha . re * tmp . im + alpha . im * tmp . re;
/**************************************************
	  A_e = A->me[i];
	  sum = 0.0;
	  for ( j = 0; j < n; j++ )
	  sum += A_e[j]*v2_ve[j];
	  out_ve[i] = v1->ve[i] + alpha*sum;
	  **************************************************/
  }
  return out;
}
/* zvm_mltadd -- vector-matrix multiply and add a la zvm_mlt()
   -- may not be in situ
   -- returns out == v1 + v2*.A */

ZVEC *zvm_mltadd(ZVEC *v1,ZVEC *v2,ZMAT *A,complex alpha,ZVEC *out)
{
/* i, */
  int j;
  int m;
  int n;
  complex tmp;
/* *A_e, */
  complex *out_ve;
  if (!v1 || !v2 || !A) 
    ev_err("zmatop.c",8,468,"zvm_mltadd",0);
  if (v2 == out) 
    ev_err("zmatop.c",12,470,"zvm_mltadd",0);
  if (v1 -> dim != A -> n || A -> m != v2 -> dim) 
    ev_err("zmatop.c",1,472,"zvm_mltadd",0);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      out = _zv_copy(v1,out,0);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("zmatop.c",_err_num,474,"zvm_mltadd",0);
    }
  }
  ;
  out_ve = out -> ve;
  m = (A -> m);
  n = (A -> n);
  for (j = 0; j <= m - 1; j += 1) {
/* tmp = zmlt(v2->ve[j],alpha); */
    tmp . re = v2 -> ve[j] . re * alpha . re - v2 -> ve[j] . im * alpha . im;
    tmp . im = v2 -> ve[j] . re * alpha . im + v2 -> ve[j] . im * alpha . re;
    if (tmp . re != 0.0 || tmp . im != 0.0) 
      __zmltadd__(out_ve,A -> me[j],tmp,(int )n,1);
/**************************************************
	  A_e = A->me[j];
	  for ( i = 0; i < n; i++ )
	  out_ve[i] += A_e[i]*tmp;
	**************************************************/
  }
  return out;
}
/* zget_col -- gets a specified column of a matrix; returned as a vector */

ZVEC *zget_col(ZMAT *mat,int col,ZVEC *vec)
{
  unsigned int i;
  if (mat == ((ZMAT *)((void *)0))) 
    ev_err("zmatop.c",8,500,"zget_col",0);
  if (col < 0 || col >= mat -> n) 
    ev_err("zmatop.c",10,502,"zget_col",0);
  if (vec == ((ZVEC *)((void *)0)) || vec -> dim < mat -> m) 
    vec = zv_resize(vec,(mat -> m));
  
#pragma omp parallel for private (i) firstprivate (col)
  for (i = 0; i <= mat -> m - 1; i += 1) {
    vec -> ve[i] = mat -> me[i][col];
  }
  return vec;
}
/* zget_row -- gets a specified row of a matrix and retruns it as a vector */

ZVEC *zget_row(ZMAT *mat,int row,ZVEC *vec)
{
/* i, */
  int lim;
  if (mat == ((ZMAT *)((void *)0))) 
    ev_err("zmatop.c",8,518,"zget_row",0);
  if (row < 0 || row >= mat -> m) 
    ev_err("zmatop.c",10,520,"zget_row",0);
  if (vec == ((ZVEC *)((void *)0)) || vec -> dim < mat -> n) 
    vec = zv_resize(vec,(mat -> n));
  lim = ((mat -> n > vec -> dim?vec -> dim : mat -> n));
/* for ( i=0; i<mat->n; i++ ) */
/*     vec->ve[i] = mat->me[row][i]; */
  memmove(((char *)(vec -> ve)),((char *)mat -> me[row]),((unsigned int )lim) * sizeof(complex ));
  return vec;
}
/* zset_col -- sets column of matrix to values given in vec (in situ) */

ZMAT *zset_col(ZMAT *mat,int col,ZVEC *vec)
{
  unsigned int i;
  unsigned int lim;
  if (mat == ((ZMAT *)((void *)0)) || vec == ((ZVEC *)((void *)0))) 
    ev_err("zmatop.c",8,539,"zset_col",0);
  if (col < 0 || col >= mat -> n) 
    ev_err("zmatop.c",10,541,"zset_col",0);
  lim = (mat -> m > vec -> dim?vec -> dim : mat -> m);
  
#pragma omp parallel for private (i) firstprivate (col,lim)
  for (i = 0; i <= lim - 1; i += 1) {
    mat -> me[i][col] = vec -> ve[i];
  }
  return mat;
}
/* zset_row -- sets row of matrix to values given in vec (in situ) */

ZMAT *zset_row(ZMAT *mat,int row,ZVEC *vec)
{
/* j, */
  unsigned int lim;
  if (mat == ((ZMAT *)((void *)0)) || vec == ((ZVEC *)((void *)0))) 
    ev_err("zmatop.c",8,555,"zset_row",0);
  if (row < 0 || row >= mat -> m) 
    ev_err("zmatop.c",10,557,"zset_row",0);
  lim = (mat -> n > vec -> dim?vec -> dim : mat -> n);
/* for ( j=j0; j<lim; j++ ) */
/*     mat->me[row][j] = vec->ve[j]; */
  memmove(((char *)mat -> me[row]),((char *)(vec -> ve)),((unsigned int )lim) * sizeof(complex ));
  return mat;
}
/* zm_rand -- randomise a complex matrix; uniform in [0,1)+[0,1)*i */

ZMAT *zm_rand(ZMAT *A)
{
  int i;
  if (!A) 
    ev_err("zmatop.c",8,572,"zm_rand",0);
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
    mrandlist((double *)A -> me[i],(2 * A -> n));
  }
  return A;
}
