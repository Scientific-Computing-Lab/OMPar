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
/* LUfactor.c 1.5 11/25/87 */
#include <omp.h> 
static char rcsid[] = "$Id: lufactor.c,v 1.10 1995/05/16 17:26:44 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"
/* Most matrix factorisation routines are in-situ unless otherwise specified */
/* LUfactor -- gaussian elimination with scaled partial pivoting
		-- Note: returns LU matrix which is A */
#ifndef ANSI_C
#else

MAT *LUfactor(MAT *A,PERM *pivot)
#endif
{
  unsigned int i;
  unsigned int j;
  unsigned int m;
  unsigned int n;
  int i_max;
  int k;
  int k_max;
  double **A_v;
  double *A_piv;
  double *A_row;
  double max1;
  double temp;
  double tiny;
  static VEC *scale = (VEC *)((void *)0);
  if (A == ((MAT *)((void *)0)) || pivot == ((PERM *)((void *)0))) 
    ev_err("lufactor.c",8,60,"LUfactor",0);
  if (pivot -> size != A -> m) 
    ev_err("lufactor.c",1,62,"LUfactor",0);
  m = A -> m;
  n = A -> n;
  scale = v_resize(scale,(A -> m));
  mem_stat_reg_list((void **)(&scale),3,0,"lufactor.c",65);
  A_v = A -> me;
  tiny = 10.0 / __builtin_huge_val();
/* initialise pivot with identity permutation */
  
#pragma omp parallel for private (i)
  for (i = 0; i <= m - 1; i += 1) {
    pivot -> pe[i] = i;
  }
/* set scale parameters */
  for (i = 0; i <= m - 1; i += 1) {
    max1 = 0.0;
    for (j = 0; j <= n - 1; j += 1) {
      temp = fabs(A_v[i][j]);
      max1 = (max1 > temp?max1 : temp);
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
      if (fabs(scale -> ve[i]) >= tiny * fabs(A_v[i][k])) {
        temp = fabs(A_v[i][k]) / scale -> ve[i];
        if (temp > max1) {
          max1 = temp;
          i_max = i;
        }
      }
    }
/* if no pivot then ignore column k... */
    if (i_max == - 1) {
/* set pivot entry A[k][k] exactly to zero,
		   rather than just "small" */
      A_v[k][k] = 0.0;
      continue; 
    }
/* do we pivot ? */
    if (i_max != k) 
/* yes we do... */
{
      px_transp(pivot,i_max,k);
      
#pragma omp parallel for private (temp,j) firstprivate (i_max)
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
      temp = A_v[i][k] = A_v[i][k] / A_v[k][k];
      A_piv = &A_v[k][k + 1];
      A_row = &A_v[i][k + 1];
      if ((k + 1) < n) 
        __mltadd__(A_row,A_piv,-temp,(int )(n - (k + 1)));
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
/* LUsolve -- given an LU factorisation in A, solve Ax=b */
#ifndef ANSI_C
#else

VEC *LUsolve(const MAT *LU,PERM *pivot,const VEC *b,VEC *x)
#endif
{
  if (!LU || !b || !pivot) 
    ev_err("lufactor.c",8,157,"LUsolve",0);
  if (LU -> m != LU -> n || LU -> n != b -> dim) 
    ev_err("lufactor.c",1,159,"LUsolve",0);
  x = v_resize(x,(b -> dim));
  px_vec(pivot,b,x);
/* x := P.b */
  Lsolve(LU,x,x,1.0);
/* implicit diagonal = 1 */
  Usolve(LU,x,x,0.0);
/* explicit diagonal */
  return x;
}
/* LUTsolve -- given an LU factorisation in A, solve A^T.x=b */
#ifndef ANSI_C
#else

VEC *LUTsolve(const MAT *LU,PERM *pivot,const VEC *b,VEC *x)
#endif
{
  if (!LU || !b || !pivot) 
    ev_err("lufactor.c",8,180,"LUTsolve",0);
  if (LU -> m != LU -> n || LU -> n != b -> dim) 
    ev_err("lufactor.c",1,182,"LUTsolve",0);
  x = _v_copy(b,x,0);
  UTsolve(LU,x,x,0.0);
/* explicit diagonal */
  LTsolve(LU,x,x,1.0);
/* implicit diagonal = 1 */
  pxinv_vec(pivot,x,x);
/* x := P^T.tmp */
  return x;
}
/* m_inverse -- returns inverse of A, provided A is not too rank deficient
	-- uses LU factorisation */
#ifndef ANSI_C
#else

MAT *m_inverse(const MAT *A,MAT *out)
#endif
{
  int i;
  static VEC *tmp = (VEC *)((void *)0);
  static VEC *tmp2 = (VEC *)((void *)0);
  static MAT *A_cp = (MAT *)((void *)0);
  static PERM *pivot = (PERM *)((void *)0);
  if (!A) 
    ev_err("lufactor.c",8,207,"m_inverse",0);
  if (A -> m != A -> n) 
    ev_err("lufactor.c",9,209,"m_inverse",0);
  if (!out || out -> m < A -> m || out -> n < A -> n) 
    out = m_resize(out,(A -> m),(A -> n));
  A_cp = m_resize(A_cp,(A -> m),(A -> n));
  A_cp = _m_copy(A,A_cp,0,0);
  tmp = v_resize(tmp,(A -> m));
  tmp2 = v_resize(tmp2,(A -> m));
  pivot = px_resize(pivot,(A -> m));
  mem_stat_reg_list((void **)(&A_cp),0,0,"lufactor.c",218);
  mem_stat_reg_list((void **)(&tmp),3,0,"lufactor.c",219);
  mem_stat_reg_list((void **)(&tmp2),3,0,"lufactor.c",220);
  mem_stat_reg_list((void **)(&pivot),2,0,"lufactor.c",221);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      LUfactor(A_cp,pivot);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("lufactor.c",_err_num,222,"m_inverse",0);
    }
  }
  ;
  for (i = 0; ((unsigned int )i) <= A -> n - 1; i += 1) {
    v_zero(tmp);
    tmp -> ve[i] = 1.0;
{
      jmp_buf _save;
      int _err_num;
      int _old_flag;
      _old_flag = set_err_flag(2);
      memmove(_save,restart,sizeof(jmp_buf ));
      if ((_err_num = _setjmp(restart)) == 0) {
        LUsolve(A_cp,pivot,tmp,tmp2);
        set_err_flag(_old_flag);
        memmove(restart,_save,sizeof(jmp_buf ));
      }
       else {
        set_err_flag(_old_flag);
        memmove(restart,_save,sizeof(jmp_buf ));
        ev_err("lufactor.c",_err_num,227,"m_inverse",0);
      }
    }
    ;
    _set_col(out,i,tmp2,0);
  }
#ifdef	THREADSAFE
#endif
  return out;
}
/* LUcondest -- returns an estimate of the condition number of LU given the
	LU factorisation in compact form */
#ifndef ANSI_C
#else

double LUcondest(const MAT *LU,PERM *pivot)
#endif
{
  static VEC *y = (VEC *)((void *)0);
  static VEC *z = (VEC *)((void *)0);
  double cond_est;
  double L_norm;
  double U_norm;
  double sum;
  double tiny;
  int i;
  int j;
  int n;
  if (!LU || !pivot) 
    ev_err("lufactor.c",8,254,"LUcondest",0);
  if (LU -> m != LU -> n) 
    ev_err("lufactor.c",9,256,"LUcondest",0);
  if (LU -> n != pivot -> size) 
    ev_err("lufactor.c",1,258,"LUcondest",0);
  tiny = 10.0 / __builtin_huge_val();
  n = (LU -> n);
  y = v_resize(y,n);
  z = v_resize(z,n);
  mem_stat_reg_list((void **)(&y),3,0,"lufactor.c",265);
  mem_stat_reg_list((void **)(&z),3,0,"lufactor.c",266);
  for (i = 0; i <= n - 1; i += 1) {
    sum = 0.0;
    
#pragma omp parallel for private (j) reduction (-:sum)
    for (j = 0; j <= i - 1; j += 1) {
      sum -= LU -> me[j][i] * y -> ve[j];
    }
    sum -= (sum < 0.0?1.0 : - 1.0);
    if (fabs(LU -> me[i][i]) <= tiny * fabs(sum)) 
      return __builtin_huge_val();
    y -> ve[i] = sum / LU -> me[i][i];
  }
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(3);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      LTsolve(LU,y,y,1.0);
      LUsolve(LU,pivot,y,z);
      ;
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else if (_err_num == 4) {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      return __builtin_huge_val();
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("lufactor.c",_err_num,283,"catch",0);
    }
  }
  ;
/* now estimate norm of A (even though it is not directly available) */
/* actually computes ||L||_inf.||U||_inf */
  U_norm = 0.0;
  for (i = 0; i <= n - 1; i += 1) {
    sum = 0.0;
    for (j = i; j <= n - 1; j += 1) {
      sum += fabs(LU -> me[i][j]);
    }
    if (sum > U_norm) 
      U_norm = sum;
  }
  L_norm = 0.0;
  for (i = 0; i <= n - 1; i += 1) {
    sum = 1.0;
    for (j = 0; j <= i - 1; j += 1) {
      sum += fabs(LU -> me[i][j]);
    }
    if (sum > L_norm) 
      L_norm = sum;
  }
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      cond_est = U_norm * L_norm * _v_norm_inf(z,((VEC *)((void *)0))) / _v_norm_inf(y,((VEC *)((void *)0)));
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("lufactor.c",_err_num,307,"LUcondest",0);
    }
  }
  ;
#ifdef	THREADSAFE
#endif
  return cond_est;
}
