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
	Arnoldi method for finding eigenvalues of large non-symmetric
		matrices
*/
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include	"matrix2.h"
#include	"sparse.h"
static char rcsid[] = "$Id: arnoldi.c,v 1.3 1994/01/13 05:45:40 des Exp $";
/* arnoldi -- an implementation of the Arnoldi method */

MAT *arnoldi(A,A_param,x0,m,h_rem,Q,H)
VEC *(*A)();
void *A_param;
VEC *x0;
int m;
double *h_rem;
MAT *Q;
MAT *H;
{
  static VEC *v = (VEC *)((void *)0);
  static VEC *u = (VEC *)((void *)0);
  static VEC *r = (VEC *)((void *)0);
  static VEC *s = (VEC *)((void *)0);
  static VEC *tmp = (VEC *)((void *)0);
  int i;
  double h_val;
  if (!A || !Q || !x0) 
    ev_err("arnoldi.c",8,53,"arnoldi",0);
  if (m <= 0) 
    ev_err("arnoldi.c",2,55,"arnoldi",0);
  if (Q -> n != x0 -> dim || Q -> m != m) 
    ev_err("arnoldi.c",1,57,"arnoldi",0);
  m_zero(Q);
  H = m_resize(H,m,m);
  m_zero(H);
  u = v_resize(u,(x0 -> dim));
  v = v_resize(v,(x0 -> dim));
  r = v_resize(r,m);
  s = v_resize(s,m);
  tmp = v_resize(tmp,(x0 -> dim));
  mem_stat_reg_list((void **)(&u),3,0,"arnoldi.c",67);
  mem_stat_reg_list((void **)(&v),3,0,"arnoldi.c",68);
  mem_stat_reg_list((void **)(&r),3,0,"arnoldi.c",69);
  mem_stat_reg_list((void **)(&s),3,0,"arnoldi.c",70);
  mem_stat_reg_list((void **)(&tmp),3,0,"arnoldi.c",71);
  sv_mlt(1.0 / _v_norm2(x0,((VEC *)((void *)0))),x0,v);
  for (i = 0; i <= m - 1; i += 1) {
    _set_row(Q,i,v,0);
    u = ( *A)(A_param,v,u);
    r = mv_mlt(Q,u,r);
    tmp = vm_mlt(Q,r,tmp);
    v_sub(u,tmp,u);
    h_val = _v_norm2(u,((VEC *)((void *)0)));
/* if u == 0 then we have an exact subspace */
    if (h_val == 0.0) {
       *h_rem = h_val;
      return H;
    }
/* iterative refinement -- ensures near orthogonality */
    do {
      s = mv_mlt(Q,u,s);
      tmp = vm_mlt(Q,s,tmp);
      v_sub(u,tmp,u);
      v_add(r,s,r);
    }while (_v_norm2(s,((VEC *)((void *)0))) > 0.1 * (h_val = _v_norm2(u,((VEC *)((void *)0)))));
/* now that u is nearly orthogonal to Q, update H */
    _set_col(H,i,r,0);
    if (i == m - 1) {
       *h_rem = h_val;
      continue; 
    }
/* H->me[i+1][i] = h_val; */
    H -> me[i + 1][i] = h_val;
    sv_mlt(1.0 / h_val,u,v);
  }
#ifdef THREADSAFE
#endif
  return H;
}
/* sp_arnoldi -- uses arnoldi() with an explicit representation of A */

MAT *sp_arnoldi(A,x0,m,h_rem,Q,H)
SPMAT *A;
VEC *x0;
int m;
double *h_rem;
MAT *Q;
MAT *H;
{
  return arnoldi(sp_mv_mlt,A,x0,m,h_rem,Q,H);
}
/* gmres -- generalised minimum residual algorithm of Saad & Schultz
		SIAM J. Sci. Stat. Comp. v.7, pp.856--869 (1986)
	-- y is overwritten with the solution */

VEC *gmres(A,A_param,m,Q,R,b,tol,x)
VEC *(*A)();
void *A_param;
int m;
MAT *Q;
MAT *R;
VEC *b;
double tol;
VEC *x;
{
  static VEC *v = (VEC *)((void *)0);
  static VEC *u = (VEC *)((void *)0);
  static VEC *r = (VEC *)((void *)0);
  static VEC *tmp = (VEC *)((void *)0);
  static VEC *rhs = (VEC *)((void *)0);
  static VEC *diag = (VEC *)((void *)0);
  static VEC *beta = (VEC *)((void *)0);
  int i;
  double h_val;
  double norm_b;
  if (!A || !Q || !b || !R) 
    ev_err("arnoldi.c",8,139,"gmres",0);
  if (m <= 0) 
    ev_err("arnoldi.c",2,141,"gmres",0);
  if (Q -> n != b -> dim || Q -> m != m) 
    ev_err("arnoldi.c",1,143,"gmres",0);
  x = _v_copy(b,x,0);
  m_zero(Q);
  R = m_resize(R,m + 1,m);
  m_zero(R);
  u = v_resize(u,(x -> dim));
  v = v_resize(v,(x -> dim));
  tmp = v_resize(tmp,(x -> dim));
  rhs = v_resize(rhs,m + 1);
  mem_stat_reg_list((void **)(&u),3,0,"arnoldi.c",153);
  mem_stat_reg_list((void **)(&v),3,0,"arnoldi.c",154);
  mem_stat_reg_list((void **)(&r),3,0,"arnoldi.c",155);
  mem_stat_reg_list((void **)(&tmp),3,0,"arnoldi.c",156);
  mem_stat_reg_list((void **)(&rhs),3,0,"arnoldi.c",157);
  norm_b = _v_norm2(x,((VEC *)((void *)0)));
  if (norm_b == 0.0) 
    ev_err("arnoldi.c",10,160,"gmres",0);
  sv_mlt(1.0 / norm_b,x,v);
  for (i = 0; i <= m - 1; i += 1) {
    _set_row(Q,i,v,0);
{
      jmp_buf _save;
      int _err_num;
      int _old_flag;
      _old_flag = set_err_flag(2);
      memmove(_save,restart,sizeof(jmp_buf ));
      if ((_err_num = _setjmp(restart)) == 0) {
        u = ( *A)(A_param,v,u);
        set_err_flag(_old_flag);
        memmove(restart,_save,sizeof(jmp_buf ));
      }
       else {
        set_err_flag(_old_flag);
        memmove(restart,_save,sizeof(jmp_buf ));
        ev_err("arnoldi.c",_err_num,166,"gmres",0);
      }
    }
    ;
    r = mv_mlt(Q,u,r);
    tmp = vm_mlt(Q,r,tmp);
    v_sub(u,tmp,u);
    h_val = _v_norm2(u,((VEC *)((void *)0)));
    _set_col(R,i,r,0);
    R -> me[i + 1][i] = h_val;
    sv_mlt(1.0 / h_val,u,v);
  }
/* use i x i submatrix of R */
  R = m_resize(R,i + 1,i);
  rhs = v_resize(rhs,i + 1);
  v_zero(rhs);
  rhs -> ve[0] = norm_b;
  tmp = v_resize(tmp,i);
  diag = v_resize(diag,i + 1);
  beta = v_resize(beta,i + 1);
  mem_stat_reg_list((void **)(&beta),3,0,"arnoldi.c",184);
  mem_stat_reg_list((void **)(&diag),3,0,"arnoldi.c",185);
/* ,beta */
  QRfactor(R,diag);
/* beta, */
  tmp = QRsolve(R,diag,rhs,tmp);
  v_resize(tmp,m);
  vm_mlt(Q,tmp,x);
#ifdef THREADSAFE
#endif
  return x;
}
