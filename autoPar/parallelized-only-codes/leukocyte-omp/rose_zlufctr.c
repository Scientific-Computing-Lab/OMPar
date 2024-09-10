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
	Complex version
*/
#include <omp.h> 
static char rcsid[] = "$Id: zlufctr.c,v 1.3 1996/08/20 20:07:09 stewart Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
#include        "zmatrix2.h"
#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)
/* Most matrix factorisation routines are in-situ unless otherwise specified */
/* zLUfactor -- Gaussian elimination with scaled partial pivoting
		-- Note: returns LU matrix which is A */

ZMAT *zLUfactor(ZMAT *A,PERM *pivot)
{
  unsigned int i;
  unsigned int j;
  unsigned int m;
  unsigned int n;
  int i_max;
  int k;
  int k_max;
  double dtemp;
  double max1;
  complex **A_v;
  complex *A_piv;
  complex *A_row;
  complex temp;
  static VEC *scale = (VEC *)((void *)0);
  if (A == ((ZMAT *)((void *)0)) || pivot == ((PERM *)((void *)0))) 
    ev_err("zlufctr.c",8,54,"zLUfactor",0);
  if (pivot -> size != A -> m) 
    ev_err("zlufctr.c",1,56,"zLUfactor",0);
  m = A -> m;
  n = A -> n;
  scale = v_resize(scale,(A -> m));
  mem_stat_reg_list((void **)(&scale),3,0,"zlufctr.c",59);
  A_v = A -> me;
/* initialise pivot with identity permutation */
  
#pragma omp parallel for private (i)
  for (i = 0; i <= m - 1; i += 1) {
    pivot -> pe[i] = i;
  }
/* set scale parameters */
  for (i = 0; i <= m - 1; i += 1) {
    max1 = 0.0;
    for (j = 0; j <= n - 1; j += 1) {
      dtemp = zabs(A_v[i][j]);
      max1 = (max1 > dtemp?max1 : dtemp);
    }
    scale -> ve[i] = max1;
  }
/* main loop */
  k_max = (((m > n?n : m)) - 1);
  for (k = 0; k <= k_max - 1; k += 1) {
/* find best pivot row */
    max1 = 0.0;
    i_max = - 1;
    for (i = k; i <= m - 1; i += 1) {
      if (scale -> ve[i] > 0.0) {
        dtemp = zabs(A_v[i][k]) / scale -> ve[i];
        if (dtemp > max1) {
          max1 = dtemp;
          i_max = i;
        }
      }
    }
/* if no pivot then ignore column k... */
    if (i_max == - 1) 
      continue; 
/* do we pivot ? */
    if (i_max != k) 
/* yes we do... */
{
      px_transp(pivot,i_max,k);
      for (j = 0; j <= n - 1; j += 1) {
        temp = A_v[i_max][j];
        A_v[i_max][j] = A_v[k][j];
        A_v[k][j] = temp;
      }
    }
/* row operations */
    for (i = (k + 1); i <= m - 1; i += 1) 
/* for each row do... */
{
/* Note: divide by zero should never happen */
      temp = A_v[i][k] = zdiv(A_v[i][k],A_v[k][k]);
      A_piv = &A_v[k][k + 1];
      A_row = &A_v[i][k + 1];
      temp . re = -temp . re;
      temp . im = -temp . im;
      if ((k + 1) < n) 
        __zmltadd__(A_row,A_piv,temp,(int )(n - (k + 1)),0);
/*********************************************
		  for ( j=k+1; j<n; j++ )
		  A_v[i][j] -= temp*A_v[k][j];
		  (*A_row++) -= temp*(*A_piv++);
		*********************************************/
    }
  }
#ifdef	THREADSAFE
#endif
  return A;
}
/* zLUsolve -- given an LU factorisation in A, solve Ax=b */

ZVEC *zLUsolve(ZMAT *A,PERM *pivot,ZVEC *b,ZVEC *x)
{
  if (A == ((ZMAT *)((void *)0)) || b == ((ZVEC *)((void *)0)) || pivot == ((PERM *)((void *)0))) 
    ev_err("zlufctr.c",8,141,"zLUsolve",0);
  if (A -> m != A -> n || A -> n != b -> dim) 
    ev_err("zlufctr.c",1,143,"zLUsolve",0);
  x = px_zvec(pivot,b,x);
/* x := P.b */
  zLsolve(A,x,x,1.0);
/* implicit diagonal = 1 */
  zUsolve(A,x,x,0.0);
/* explicit diagonal */
  return x;
}
/* zLUAsolve -- given an LU factorisation in A, solve A^*.x=b */

ZVEC *zLUAsolve(ZMAT *LU,PERM *pivot,ZVEC *b,ZVEC *x)
{
  if (!LU || !b || !pivot) 
    ev_err("zlufctr.c",8,160,"zLUAsolve",0);
  if (LU -> m != LU -> n || LU -> n != b -> dim) 
    ev_err("zlufctr.c",1,162,"zLUAsolve",0);
  x = _zv_copy(b,x,0);
  zUAsolve(LU,x,x,0.0);
/* explicit diagonal */
  zLAsolve(LU,x,x,1.0);
/* implicit diagonal = 1 */
  pxinv_zvec(pivot,x,x);
/* x := P^*.x */
  return x;
}
/* zm_inverse -- returns inverse of A, provided A is not too rank deficient
	-- uses LU factorisation */

ZMAT *zm_inverse(ZMAT *A,ZMAT *out)
{
  int i;
  static ZVEC *tmp = (ZVEC *)((void *)0);
  static ZVEC *tmp2 = (ZVEC *)((void *)0);
  static ZMAT *A_cp = (ZMAT *)((void *)0);
  static PERM *pivot = (PERM *)((void *)0);
  if (!A) 
    ev_err("zlufctr.c",8,183,"zm_inverse",0);
  if (A -> m != A -> n) 
    ev_err("zlufctr.c",9,185,"zm_inverse",0);
  if (!out || out -> m < A -> m || out -> n < A -> n) 
    out = zm_resize(out,(A -> m),(A -> n));
  A_cp = zm_resize(A_cp,(A -> m),(A -> n));
  A_cp = _zm_copy(A,A_cp,0,0);
  tmp = zv_resize(tmp,(A -> m));
  tmp2 = zv_resize(tmp2,(A -> m));
  pivot = px_resize(pivot,(A -> m));
  mem_stat_reg_list((void **)(&A_cp),9,0,"zlufctr.c",194);
  mem_stat_reg_list((void **)(&tmp),8,0,"zlufctr.c",195);
  mem_stat_reg_list((void **)(&tmp2),8,0,"zlufctr.c",196);
  mem_stat_reg_list((void **)(&pivot),2,0,"zlufctr.c",197);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      zLUfactor(A_cp,pivot);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("zlufctr.c",_err_num,198,"zm_inverse",0);
    }
  }
  ;
  for (i = 0; ((unsigned int )i) <= A -> n - 1; i += 1) {
    zv_zero(tmp);
    tmp -> ve[i] . re = 1.0;
    tmp -> ve[i] . im = 0.0;
{
      jmp_buf _save;
      int _err_num;
      int _old_flag;
      _old_flag = set_err_flag(2);
      memmove(_save,restart,sizeof(jmp_buf ));
      if ((_err_num = _setjmp(restart)) == 0) {
        zLUsolve(A_cp,pivot,tmp,tmp2);
        set_err_flag(_old_flag);
        memmove(restart,_save,sizeof(jmp_buf ));
      }
       else {
        set_err_flag(_old_flag);
        memmove(restart,_save,sizeof(jmp_buf ));
        ev_err("zlufctr.c",_err_num,204,"zm_inverse",0);
      }
    }
    ;
    zset_col(out,i,tmp2);
  }
#ifdef	THREADSAFE
#endif
  return out;
}
/* zLUcondest -- returns an estimate of the condition number of LU given the
	LU factorisation in compact form */

double zLUcondest(ZMAT *LU,PERM *pivot)
{
  static ZVEC *y = (ZVEC *)((void *)0);
  static ZVEC *z = (ZVEC *)((void *)0);
  double cond_est;
  double L_norm;
  double U_norm;
  double norm;
  double sn_inv;
  complex sum;
  int i;
  int j;
  int n;
  if (!LU || !pivot) 
    ev_err("zlufctr.c",8,228,"zLUcondest",0);
  if (LU -> m != LU -> n) 
    ev_err("zlufctr.c",9,230,"zLUcondest",0);
  if (LU -> n != pivot -> size) 
    ev_err("zlufctr.c",1,232,"zLUcondest",0);
  n = (LU -> n);
  y = zv_resize(y,n);
  z = zv_resize(z,n);
  mem_stat_reg_list((void **)(&y),8,0,"zlufctr.c",237);
  mem_stat_reg_list((void **)(&z),8,0,"zlufctr.c",238);
  cond_est = 0.0;
/* should never be returned */
  for (i = 0; i <= n - 1; i += 1) {
    sum . re = 1.0;
    sum . im = 0.0;
    for (j = 0; j <= i - 1; j += 1) {
/* sum -= LU->me[j][i]*y->ve[j]; */
      sum = zsub(sum,(zmlt(LU -> me[j][i],y -> ve[j])));
    }
/* sum -= (sum < 0.0) ? 1.0 : -1.0; */
    sn_inv = 1.0 / zabs(sum);
    sum . re += sum . re * sn_inv;
    sum . im += sum . im * sn_inv;
    if (LU -> me[i][i] . re == 0.0 && LU -> me[i][i] . im == 0.0) 
      return __builtin_huge_val();
/* y->ve[i] = sum / LU->me[i][i]; */
    y -> ve[i] = zdiv(sum,LU -> me[i][i]);
  }
  zLAsolve(LU,y,y,1.0);
  zLUsolve(LU,pivot,y,z);
/* now estimate norm of A (even though it is not directly available) */
/* actually computes ||L||_inf.||U||_inf */
  U_norm = 0.0;
  for (i = 0; i <= n - 1; i += 1) {
    norm = 0.0;
    for (j = i; j <= n - 1; j += 1) {
      norm += zabs(LU -> me[i][j]);
    }
    if (norm > U_norm) 
      U_norm = norm;
  }
  L_norm = 0.0;
  for (i = 0; i <= n - 1; i += 1) {
    norm = 1.0;
    for (j = 0; j <= i - 1; j += 1) {
      norm += zabs(LU -> me[i][j]);
    }
    if (norm > L_norm) 
      L_norm = norm;
  }
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      cond_est = U_norm * L_norm * _zv_norm_inf(z,(VEC *)((void *)0)) / _zv_norm_inf(y,(VEC *)((void *)0));
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("zlufctr.c",_err_num,284,"zLUcondest",0);
    }
  }
  ;
#ifdef	THREADSAFE
#endif
  return cond_est;
}
