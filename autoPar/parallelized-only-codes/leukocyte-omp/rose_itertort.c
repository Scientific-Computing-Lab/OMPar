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
/*  iter_tort.c  16/09/93 */
/*
  This file contains tests for the iterative part of Meschach
*/
#include	<stdio.h>
#include	"matrix2.h"
#include	"sparse2.h"
#include	"iter.h"
#include	<math.h>
#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);
/* for iterative methods */
#if REAL == DOUBLE
#define	EPS	1e-7
#define KK	20
#elif REAL == FLOAT
#define EPS   1e-5
#define KK	8
#endif
#define ANON  513
#define ASYM  ANON   
#include <omp.h> 
static VEC *ex_sol = (VEC *)((void *)0);
/* new iter information */

void iter_mod_info(ip,nres,res,Bres)
ITER *ip;
double nres;
VEC *res;
VEC *Bres;
{
  static VEC *tmp;
  if (ip -> b == ((VEC *)((void *)0))) 
    return ;
  tmp = v_resize(tmp,(ip -> b -> dim));
  mem_stat_reg_list((void **)(&tmp),3,0,"itertort.c",68);
  if (nres >= 0.0) {
    printf(" %d. residual = %g\n",ip -> steps,nres);
  }
   else 
    printf(" %d. residual = %g (WARNING !!! should be >= 0) \n",ip -> steps,nres);
  if (ex_sol != ((VEC *)((void *)0))) 
    printf("    ||u_ex - u_approx||_2 = %g\n",(_v_norm2((v_sub((ip -> x),ex_sol,tmp)),((VEC *)((void *)0)))));
}
/* out = A^T*A*x */

VEC *norm_equ(A,x,out)
SPMAT *A;
VEC *x;
VEC *out;
{
  static VEC *tmp;
  tmp = v_resize(tmp,(x -> dim));
  mem_stat_reg_list((void **)(&tmp),3,0,"itertort.c",90);
  sp_mv_mlt(A,x,tmp);
  sp_vm_mlt(A,tmp,out);
  return out;
}
/* 
  make symmetric preconditioner for nonsymmetric matrix A;
   B = 0.5*(A+A^T) and then B is factorized using 
   incomplete Choleski factorization
*/

SPMAT *gen_sym_precond(A)
SPMAT *A;
{
  SPMAT *B;
  SPROW *row;
  int i;
  int j;
  int k;
  double val;
  B = sp_get(A -> m,A -> n,A -> row[0] . maxlen);
  for (i = 0; i <= A -> m - 1; i += 1) {
    row = &A -> row[i];
    for (j = 0; j <= row -> len - 1; j += 1) {
      k = row -> elt[j] . col;
      if (i != k) {
        val = 0.5 * (sp_get_val(A,i,k) + sp_get_val(A,k,i));
        sp_set_val(B,i,k,val);
        sp_set_val(B,k,i,val);
      }
       else {
/* i == k */
        val = sp_get_val(A,i,i);
        sp_set_val(B,i,i,val);
      }
    }
  }
  spICHfactor(B);
  return B;
}
/* Dv_mlt -- diagonal by vector multiply; the diagonal matrix is represented
		by a vector d */

VEC *Dv_mlt(d,x,out)
VEC *d;
VEC *x;
VEC *out;
{
  int i;
  if (!d || !x) 
    ev_err("itertort.c",8,141,"Dv_mlt",0);
  if (d -> dim != x -> dim) 
    ev_err("itertort.c",1,143,"Dv_mlt",0);
  out = v_resize(out,(x -> dim));
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= x -> dim - 1; i += 1) {
    out -> ve[i] = d -> ve[i] * x -> ve[i];
  }
  return out;
}
/************************************************/

void main(argc,argv)
int argc;
char *argv[];
{
  VEC *x;
  VEC *y;
  VEC *z;
  VEC *u;
  VEC *v;
  VEC *xn;
  VEC *yn;
  SPMAT *A = ((void *)0);
  SPMAT *B = ((void *)0);
  SPMAT *An = ((void *)0);
  SPMAT *Bn = ((void *)0);
  int i;
  int k;
  int kk;
  int j;
  ITER *ips;
  ITER *ips1;
  ITER *ipns;
  ITER *ipns1;
  MAT *Q;
  MAT *H;
  MAT *Q1;
  MAT *H1;
  VEC vt;
  VEC vt1;
  double hh;
  mem_info_on(1);
  printf("# Testing %s...\n","allocating sparse matrices");
  ;
  printf(" dim of A = %dx%d\n",513,513);
  A = iter_gen_sym(513,8);
  B = sp_copy(A);
  spICHfactor(B);
  u = v_get(A -> n);
  x = v_get(A -> n);
  y = v_get(A -> n);
  v = v_get(A -> n);
  v_rand(x);
  sp_mv_mlt(A,x,y);
  ex_sol = x;
  printf("# Testing %s...\n"," initialize ITER variables");
  ;
/* ips for symmetric matrices with precondition */
  ips = iter_get(A -> m,A -> n);
/*  printf(" ips:\n");
   iter_dump(stdout,ips);   */
  ips -> limit = 500;
  ips -> eps = 1e-7;
  ((ips -> Ax = ((Fun_Ax )sp_mv_mlt) , ips -> A_par = ((void *)A)) , 0);
  ((ips -> Bx = ((Fun_Ax )spCHsolve) , ips -> B_par = ((void *)B)) , 0);
  ips -> b = _v_copy(y,ips -> b,0);
  v_rand(ips -> x);
/* test of iter_resize */
  ips = iter_resize(ips,2 * A -> m,2 * A -> n);
  ips = iter_resize(ips,A -> m,A -> n);
/*  printf(" ips:\n");
   iter_dump(stdout,ips); */
/* ips1 for symmetric matrices without precondition */
  ips1 = iter_get(0,0);
/*   printf(" ips1:\n");
   iter_dump(stdout,ips1);   */
  (iter_free(ips1) , ips1 = ((ITER *)((void *)0)));
  ips1 = iter_copy2(ips,ips1);
  ((ips1 -> Bx = ((Fun_Ax )((void *)0)) , ips1 -> B_par = ((void *)((void *)0))) , 0);
  ips1 -> b = ips -> b;
  ips1 -> shared_b = 1;
/*    printf(" ips1:\n");
   iter_dump(stdout,ips1);   */
/* ipns for nonsymetric matrices with precondition */
  ipns = iter_copy(ips,(ITER *)((void *)0));
  ipns -> k = 20;
  ipns -> limit = 500;
  ipns -> info = ((void *)0);
  An = iter_gen_nonsym_posdef(513,8);
  Bn = gen_sym_precond(An);
  xn = v_get(An -> n);
  yn = v_get(An -> n);
  v_rand(xn);
  sp_mv_mlt(An,xn,yn);
  ipns -> b = _v_copy(yn,ipns -> b,0);
  ((ipns -> Ax = ((Fun_Ax )sp_mv_mlt) , ipns -> A_par = ((void *)An)) , 0);
  ((ipns -> ATx = ((Fun_Ax )sp_vm_mlt) , ipns -> AT_par = ((void *)An)) , 0);
  ((ipns -> Bx = ((Fun_Ax )spCHsolve) , ipns -> B_par = ((void *)Bn)) , 0);
/*  printf(" ipns:\n");
   iter_dump(stdout,ipns); */
/* ipns1 for nonsymmetric matrices without precondition */
  ipns1 = iter_copy2(ipns,(ITER *)((void *)0));
  ipns1 -> b = ipns -> b;
  ipns1 -> shared_b = 1;
  ((ipns1 -> Bx = ((Fun_Ax )((void *)0)) , ipns1 -> B_par = ((void *)((void *)0))) , 0);
/*   printf(" ipns1:\n");
   iter_dump(stdout,ipns1);  */
/*******  CG  ********/
  printf("# Testing %s...\n"," CG method without preconditioning");
  ;
  ips1 -> info = ((void *)0);
  mem_stat_mark(1);
  iter_cg(ips1);
  k = ips1 -> steps;
  z = ips1 -> x;
  printf(" cg: no. of iter.steps = %d\n",k);
  v_sub(z,x,u);
  printf(" (cg:) ||u_ex - u_approx||_2 = %g [EPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),1e-7);
  printf("# Testing %s...\n"," CG method with ICH preconditioning");
  ;
  ips -> info = ((void *)0);
  v_zero(ips -> x);
  iter_cg(ips);
  k = ips -> steps;
  printf(" cg: no. of iter.steps = %d\n",k);
  v_sub((ips -> x),x,u);
  printf(" (cg:) ||u_ex - u_approx||_2 = %g [EPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),1e-7);
  (v_free(v) , v = ((VEC *)((void *)0)));
  if ((v = iter_spcg(A,B,y,1e-7,(VEC *)((void *)0),1000,&k)) == ((VEC *)((void *)0))) 
    printf("Error: %s error: line %d\n","CG method with precond.: NULL solution",281);
  v_sub((ips -> x),v,u);
  if (_v_norm2(u,((VEC *)((void *)0))) >= 1e-7) {
    printf("Error: %s error: line %d\n","CG method with precond.: different solutions",285);
    printf(" diff. = %g\n",(_v_norm2(u,((VEC *)((void *)0)))));
  }
  mem_stat_free_list(1,0);
  printf(" spcg: # of iter. steps = %d\n",k);
  v_sub(v,x,u);
  printf(" (spcg:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),1e-7);
/***  CG FOR NORMAL EQUATION *****/
  printf("# Testing %s...\n","CGNE method with ICH preconditioning (nonsymmetric case)");
  ;
/* ipns->info = iter_std_info;  */
  ipns -> info = ((void *)0);
  v_zero(ipns -> x);
  mem_stat_mark(1);
  iter_cgne(ipns);
  k = ipns -> steps;
  z = ipns -> x;
  printf(" cgne: no. of iter.steps = %d\n",k);
  v_sub(z,xn,u);
  printf(" (cgne:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),1e-7);
  printf("# Testing %s...\n","CGNE method without preconditioning (nonsymmetric case)");
  ;
  v_rand(u);
  u = iter_spcgne(An,((void *)0),yn,1e-7,u,1000,&k);
  mem_stat_free_list(1,0);
  printf(" spcgne: no. of iter.steps = %d\n",k);
  v_sub(u,xn,u);
  printf(" (spcgne:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),1e-7);
/***  CGS  *****/
  printf("# Testing %s...\n","CGS method with ICH preconditioning (nonsymmetric case)");
  ;
  v_zero(ipns -> x);
/* new init guess == 0 */
  mem_stat_mark(1);
  ipns -> info = ((void *)0);
  v_rand(u);
  iter_cgs(ipns,u);
  k = ipns -> steps;
  z = ipns -> x;
  printf(" cgs: no. of iter.steps = %d\n",k);
  v_sub(z,xn,u);
  printf(" (cgs:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),1e-7);
  printf("# Testing %s...\n","CGS method without preconditioning (nonsymmetric case)");
  ;
  v_rand(u);
  v_rand(v);
  v = iter_spcgs(An,((void *)0),yn,u,1e-7,v,1000,&k);
  mem_stat_free_list(1,0);
  printf(" cgs: no. of iter.steps = %d\n",k);
  v_sub(v,xn,u);
  printf(" (cgs:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),1e-7);
/*** LSQR ***/
  printf("# Testing %s...\n","LSQR method (without preconditioning)");
  ;
  v_rand(u);
  v_free(ipns1 -> x);
  ipns1 -> x = u;
  ipns1 -> shared_x = 1;
  ipns1 -> info = ((void *)0);
  mem_stat_mark(2);
  z = iter_lsqr(ipns1);
  v_sub(xn,z,v);
  k = ipns1 -> steps;
  printf(" lsqr: # of iter. steps = %d\n",k);
  printf(" (lsqr:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),1e-7);
  v_rand(u);
  u = iter_splsqr(An,yn,1e-7,u,1000,&k);
  mem_stat_free_list(2,0);
  v_sub(xn,u,v);
  printf(" splsqr: # of iter. steps = %d\n",k);
  printf(" (splsqr:) ||u_ex - u_approx||_2 = %g [EPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),1e-7);
/***** GMRES ********/
  printf("# Testing %s...\n","GMRES method with ICH preconditioning (nonsymmetric case)");
  ;
  v_zero(ipns -> x);
/*   ipns->info = iter_std_info;  */
  ipns -> info = ((void *)0);
  mem_stat_mark(2);
  z = iter_gmres(ipns);
  v_sub(xn,z,v);
  k = ipns -> steps;
  printf(" gmres: # of iter. steps = %d\n",k);
  printf(" (gmres:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),1e-7);
  printf("# Testing %s...\n","GMRES method without preconditioning (nonsymmetric case)");
  ;
  (v_free(v) , v = ((VEC *)((void *)0)));
  v = iter_spgmres(An,((void *)0),yn,1e-7,(VEC *)((void *)0),10,1004,&k);
  mem_stat_free_list(2,0);
  v_sub(xn,v,v);
  printf(" spgmres: # of iter. steps = %d\n",k);
  printf(" (spgmres:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),1e-7);
/**** MGCR *****/
  printf("# Testing %s...\n","MGCR method with ICH preconditioning (nonsymmetric case)");
  ;
  v_zero(ipns -> x);
  mem_stat_mark(2);
  z = iter_mgcr(ipns);
  v_sub(xn,z,v);
  k = ipns -> steps;
  printf(" mgcr: # of iter. steps = %d\n",k);
  printf(" (mgcr:) ||u_ex - u_approx||_2 = %g  [EPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),1e-7);
  printf("# Testing %s...\n","MGCR method without  preconditioning (nonsymmetric case)");
  ;
  (v_free(v) , v = ((VEC *)((void *)0)));
  v = iter_spmgcr(An,((void *)0),yn,1e-7,(VEC *)((void *)0),10,1004,&k);
  mem_stat_free_list(2,0);
  v_sub(xn,v,v);
  printf(" spmgcr: # of iter. steps = %d\n",k);
  printf(" (spmgcr:) ||u_ex - u_approx||_2 = %g [EPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),1e-7);
/***** ARNOLDI METHOD ********/
  printf("# Testing %s...\n","arnoldi method");
  ;
  kk = (ipns1 -> k = 20);
  Q = m_get(kk,(x -> dim));
  Q1 = m_get(kk,(x -> dim));
  H = m_get(kk,kk);
  v_rand(u);
  ipns1 -> x = u;
  ipns1 -> shared_x = 1;
  mem_stat_mark(3);
  iter_arnoldi_iref(ipns1,&hh,Q,H);
  mem_stat_free_list(3,0);
/* check the equality:
      Q*A*Q^T = H; */
  vt . dim = vt . max_dim = x -> dim;
  vt1 . dim = vt1 . max_dim = x -> dim;
  for (j = 0; j <= kk - 1; j += 1) {
    vt . ve = Q -> me[j];
    vt1 . ve = Q1 -> me[j];
    sp_mv_mlt(An,(&vt),&vt1);
  }
  H1 = m_get(kk,kk);
  mmtr_mlt(Q,Q1,H1);
  m_sub(H,H1,H1);
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (arnoldi_iref) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
/* check Q*Q^T = I  */
  mmtr_mlt(Q,Q,H1);
  
#pragma omp parallel for private (j)
  for (j = 0; j <= kk - 1; j += 1) {
    H1 -> me[j][j] -= 1.0;
  }
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (arnoldi_iref) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
  ipns1 -> x = u;
  ipns1 -> shared_x = 1;
  mem_stat_mark(3);
  iter_arnoldi(ipns1,&hh,Q,H);
  mem_stat_free_list(3,0);
/* check the equality:
      Q*A*Q^T = H; */
  vt . dim = vt . max_dim = x -> dim;
  vt1 . dim = vt1 . max_dim = x -> dim;
  for (j = 0; j <= kk - 1; j += 1) {
    vt . ve = Q -> me[j];
    vt1 . ve = Q1 -> me[j];
    sp_mv_mlt(An,(&vt),&vt1);
  }
  mmtr_mlt(Q,Q1,H1);
  m_sub(H,H1,H1);
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (arnoldi) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
/* check Q*Q^T = I  */
  mmtr_mlt(Q,Q,H1);
  
#pragma omp parallel for private (j)
  for (j = 0; j <= kk - 1; j += 1) {
    H1 -> me[j][j] -= 1.0;
  }
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (arnoldi) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
  v_rand(u);
  mem_stat_mark(3);
  iter_sparnoldi(An,u,kk,&hh,Q,H);
  mem_stat_free_list(3,0);
/* check the equality:
      Q*A*Q^T = H; */
  vt . dim = vt . max_dim = x -> dim;
  vt1 . dim = vt1 . max_dim = x -> dim;
  for (j = 0; j <= kk - 1; j += 1) {
    vt . ve = Q -> me[j];
    vt1 . ve = Q1 -> me[j];
    sp_mv_mlt(An,(&vt),&vt1);
  }
  mmtr_mlt(Q,Q1,H1);
  m_sub(H,H1,H1);
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (sparnoldi) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
/* check Q*Q^T = I  */
  mmtr_mlt(Q,Q,H1);
  
#pragma omp parallel for private (j) firstprivate (kk)
  for (j = 0; j <= kk - 1; j += 1) {
    H1 -> me[j][j] -= 1.0;
  }
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (sparnoldi) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
/****** LANCZOS METHOD ******/
  printf("# Testing %s...\n","lanczos method");
  ;
  kk = (ipns1 -> k);
  Q = m_resize(Q,kk,(x -> dim));
  Q1 = m_resize(Q1,kk,(x -> dim));
  H = m_resize(H,kk,kk);
  ips1 -> k = kk;
  v_rand(u);
  v_free(ips1 -> x);
  ips1 -> x = u;
  ips1 -> shared_x = 1;
  mem_stat_mark(3);
  iter_lanczos(ips1,x,y,&hh,Q);
  mem_stat_free_list(3,0);
/* check the equality:
      Q*A*Q^T = H; */
  vt . dim = vt1 . dim = Q -> n;
  vt . max_dim = vt1 . max_dim = Q -> max_n;
  Q1 = m_resize(Q1,(Q -> m),(Q -> n));
  for (j = 0; ((unsigned int )j) <= Q -> m - 1; j += 1) {
    vt . ve = Q -> me[j];
    vt1 . ve = Q1 -> me[j];
    sp_mv_mlt(A,(&vt),&vt1);
  }
  H1 = m_resize(H1,(Q -> m),(Q -> m));
  H = m_resize(H,(Q -> m),(Q -> m));
  mmtr_mlt(Q,Q1,H1);
  m_zero(H);
  
#pragma omp parallel for private (j)
  for (j = 0; ((unsigned int )j) <= Q -> m - ((unsigned int )1) - 1; j += 1) {
    H -> me[j][j] = x -> ve[j];
    H -> me[j][j + 1] = H -> me[j + 1][j] = y -> ve[j];
  }
  H -> me[Q -> m - 1][Q -> m - 1] = x -> ve[Q -> m - 1];
  m_sub(H,H1,H1);
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (lanczos) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
/* check Q*Q^T = I  */
  mmtr_mlt(Q,Q,H1);
  
#pragma omp parallel for private (j)
  for (j = 0; ((unsigned int )j) <= Q -> m - 1; j += 1) {
    H1 -> me[j][j] -= 1.0;
  }
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (lanczos) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
  mem_stat_mark(3);
  v_rand(u);
  iter_splanczos(A,kk,u,x,y,&hh,Q);
  mem_stat_free_list(3,0);
/* check the equality:
      Q*A*Q^T = H; */
  vt . dim = vt1 . dim = Q -> n;
  vt . max_dim = vt1 . max_dim = Q -> max_n;
  Q1 = m_resize(Q1,(Q -> m),(Q -> n));
  for (j = 0; ((unsigned int )j) <= Q -> m - 1; j += 1) {
    vt . ve = Q -> me[j];
    vt1 . ve = Q1 -> me[j];
    sp_mv_mlt(A,(&vt),&vt1);
  }
  H1 = m_resize(H1,(Q -> m),(Q -> m));
  H = m_resize(H,(Q -> m),(Q -> m));
  mmtr_mlt(Q,Q1,H1);
  
#pragma omp parallel for private (j)
  for (j = 0; ((unsigned int )j) <= Q -> m - ((unsigned int )1) - 1; j += 1) {
    H -> me[j][j] = x -> ve[j];
    H -> me[j][j + 1] = H -> me[j + 1][j] = y -> ve[j];
  }
  H -> me[Q -> m - 1][Q -> m - 1] = x -> ve[Q -> m - 1];
  m_sub(H,H1,H1);
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (splanczos) ||Q*A*Q^T - H|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
/* check Q*Q^T = I  */
  mmtr_mlt(Q,Q,H1);
  
#pragma omp parallel for private (j)
  for (j = 0; ((unsigned int )j) <= Q -> m - 1; j += 1) {
    H1 -> me[j][j] -= 1.0;
  }
  if (m_norm_inf(H1) > ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf(" (splanczos) ||Q*Q^T - I|| = %g [cf. MACHEPS = %g]\n",(m_norm_inf(H1)),(double )2.22044604925031308084726333618164062e-16L);
/***** LANCZOS2 ****/
  printf("# Testing %s...\n","lanczos2 method");
  ;
  kk = 50;
/* # of dir. vectors */
  ips1 -> k = kk;
  v_rand(u);
  ips1 -> x = u;
  ips1 -> shared_x = 1;
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= xn -> dim - 1; i += 1) {
    xn -> ve[i] = i;
  }
  ((ips1 -> Ax = ((Fun_Ax )Dv_mlt) , ips1 -> A_par = ((void *)xn)) , 0);
  mem_stat_mark(3);
  iter_lanczos2(ips1,y,v);
  mem_stat_free_list(3,0);
  printf("# Number of steps of Lanczos algorithm = %d\n",kk);
  printf("# Exact eigenvalues are 0, 1, 2, ..., %d\n",513 - 1);
  printf("# Extreme eigenvalues should be accurate; \n");
  printf("# interior values usually are not.\n");
  printf("# approx e-vals =\n");
  v_foutput(stdout,y);
  printf("# Error in estimate of bottom e-vec (Lanczos) = %g\n",(fabs(v -> ve[0])));
  mem_stat_mark(3);
  v_rand(u);
  iter_splanczos2(A,kk,u,y,v);
  mem_stat_free_list(3,0);
/***** FINISHING *******/
  printf("# Testing %s...\n","release ITER variables");
  ;
  (m_free(Q) , Q = ((MAT *)((void *)0)));
  (m_free(Q1) , Q1 = ((MAT *)((void *)0)));
  (m_free(H) , H = ((MAT *)((void *)0)));
  (m_free(H1) , H1 = ((MAT *)((void *)0)));
  (iter_free(ipns) , ipns = ((ITER *)((void *)0)));
  (iter_free(ips) , ips = ((ITER *)((void *)0)));
  (iter_free(ipns1) , ipns1 = ((ITER *)((void *)0)));
  (iter_free(ips1) , ips1 = ((ITER *)((void *)0)));
  (sp_free(A) , A = ((SPMAT *)((void *)0)));
  (sp_free(B) , B = ((SPMAT *)((void *)0)));
  (sp_free(An) , An = ((SPMAT *)((void *)0)));
  (sp_free(Bn) , Bn = ((SPMAT *)((void *)0)));
  (v_free(x) , x = ((VEC *)((void *)0)));
  (v_free(y) , y = ((VEC *)((void *)0)));
  (v_free(u) , u = ((VEC *)((void *)0)));
  (v_free(v) , v = ((VEC *)((void *)0)));
  (v_free(xn) , xn = ((VEC *)((void *)0)));
  (v_free(yn) , yn = ((VEC *)((void *)0)));
  printf("# Done testing (%s)\n",argv[0]);
  mem_info_file(stdout,0);
}
