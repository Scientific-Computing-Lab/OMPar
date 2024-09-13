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
	This file contains tests for the sparse matrix part of Meschach
*/
#include	<stdio.h>
#include	<math.h>
#include	"matrix2.h"
#include	"sparse2.h"
#include        "iter.h"
#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);
/* for iterative methods */
#if REAL == DOUBLE
#define	EPS	1e-7
#elif REAL == FLOAT
#define EPS   1e-3
#endif
#include <omp.h> 

int chk_col_accessSPT(A)
SPMAT *A;
{
  int i;
  int j;
  int nxt_idx;
  int nxt_row;
  int scan_cnt;
  int total_cnt;
  SPROW *r;
  row_elt *e;
  if (!A) 
    ev_err("sptort.c",8,56,"chk_col_accessSPT",0);
  if (!A -> flag_col) 
    return 0;
/* scan down each column, counting the number of entries met */
  scan_cnt = 0;
  
#pragma omp parallel for private (i,nxt_idx,nxt_row,j) reduction (+:scan_cnt)
  for (j = 0; j <= A -> n - 1; j += 1) {
    i = - 1;
    nxt_idx = A -> start_idx[j];
    nxt_row = A -> start_row[j];
    while(nxt_row >= 0 && nxt_idx >= 0 && nxt_row > i){
      i = nxt_row;
      r = &A -> row[i];
      e = &r -> elt[nxt_idx];
      nxt_idx = e -> nxt_idx;
      nxt_row = e -> nxt_row;
      scan_cnt++;
    }
  }
  total_cnt = 0;
  
#pragma omp parallel for private (i) reduction (+:total_cnt)
  for (i = 0; i <= A -> m - 1; i += 1) {
    total_cnt += A -> row[i] . len;
  }
  if (total_cnt != scan_cnt) 
    return 0;
   else 
    return 1;
}

void main(argc,argv)
int argc;
char *argv[];
{
  VEC *x;
  VEC *y;
  VEC *z;
  VEC *u;
  VEC *v;
  double s1;
  double s2;
  PERM *pivot;
  SPMAT *A;
  SPMAT *B;
  SPMAT *C;
  SPMAT *B1;
  SPMAT *C1;
  SPROW *r;
  int i;
  int j;
  int k;
  int deg;
  int seed;
  int m;
  int m_old;
  int n;
  int n_old;
  mem_info_on(1);
  setbuf(stdout,(char *)((void *)0));
/* get seed if in argument list */
  if (argc == 1) 
    seed = 1111;
   else if (argc == 2 && sscanf(argv[1],"%d",&seed) == 1) 
    ;
   else {
    printf("usage: %s [seed]\n",argv[0]);
    exit(0);
  }
  srand(seed);
/* set up two random sparse matrices */
  m = 120;
  n = 100;
  deg = 8;
  printf("# Testing %s...\n","allocating sparse matrices");
  ;
  A = sp_get(m,n,deg);
  B = sp_get(m,n,deg);
  printf("# Testing %s...\n","setting and getting matrix entries");
  ;
  for (k = 0; k <= m * deg - 1; k += 1) {
    i = (rand() >> 8) % m;
    j = (rand() >> 8) % n;
    sp_set_val(A,i,j,(rand()) / ((double )((double )2147483647)));
    i = (rand() >> 8) % m;
    j = (rand() >> 8) % n;
    sp_set_val(B,i,j,(rand()) / ((double )((double )2147483647)));
  }
  for (k = 0; k <= 9; k += 1) {
    s1 = (rand()) / ((double )((double )2147483647));
    i = (rand() >> 8) % m;
    j = (rand() >> 8) % n;
    sp_set_val(A,i,j,s1);
    s2 = sp_get_val(A,i,j);
    if (fabs(s1 - s2) >= ((double )2.22044604925031308084726333618164062e-16L)) 
      break; 
  }
  if (k < 10) 
    printf("Error: %s error: line %d\n","sp_set_val()/sp_get_val()",144);
/* test copy routines */
  printf("# Testing %s...\n","copy routines");
  ;
  x = v_get(n);
  y = v_get(m);
  z = v_get(m);
/* first copy routine */
  C = sp_copy(A);
  for (k = 0; k <= 99; k += 1) {
    v_rand(x);
    sp_mv_mlt(A,x,y);
    sp_mv_mlt(C,x,z);
    if (_v_norm_inf((v_sub(y,z,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * deg * m) 
      break; 
  }
  if (k < 100) {
    printf("Error: %s error: line %d\n","sp_copy()/sp_mv_mlt()",163);
    printf("# Error in A.x (inf norm) = %g [cf MACHEPS = %g]\n",(_v_norm_inf(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* second copy routine
       -- note that A & B have different sparsity patterns */
  mem_stat_mark(1);
  sp_copy2(A,B);
  mem_stat_free_list(1,0);
  for (k = 0; k <= 9; k += 1) {
    v_rand(x);
    sp_mv_mlt(A,x,y);
    sp_mv_mlt(B,x,z);
    if (_v_norm_inf((v_sub(y,z,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * deg * m) 
      break; 
  }
  if (k < 10) {
    printf("Error: %s error: line %d\n","sp_copy2()/sp_mv_mlt()",183);
    printf("# Error in A.x (inf norm) = %g [cf MACHEPS = %g]\n",(_v_norm_inf(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* now check compacting routine */
  printf("# Testing %s...\n","compacting routine");
  ;
  sp_compact(B,0.0);
  for (k = 0; k <= 9; k += 1) {
    v_rand(x);
    sp_mv_mlt(A,x,y);
    sp_mv_mlt(B,x,z);
    if (_v_norm_inf((v_sub(y,z,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * deg * m) 
      break; 
  }
  if (k < 10) {
    printf("Error: %s error: line %d\n","sp_compact()",201);
    printf("# Error in A.x (inf norm) = %g [cf MACHEPS = %g]\n",(_v_norm_inf(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  
#pragma omp parallel for private (i)
  for (i = 0; i <= B -> m - 1; i += 1) {
    r = &B -> row[i];
    for (j = 0; j <= r -> len - 1; j += 1) {
      if (r -> elt[j] . val == 0.0) 
        break; 
    }
  }
  if (i < B -> m) {
    printf("Error: %s error: line %d\n","sp_compact()",214);
    printf("# Zero entry in compacted matrix\n");
  }
/* check column access paths */
  printf("# Testing %s...\n","resizing and access paths");
  ;
  m_old = A -> m - 1;
  n_old = A -> n - 1;
  A = sp_resize(A,A -> m + 10,A -> n + 10);
  for (k = 0; k <= 19; k += 1) {
    i = m_old + (rand() >> 8) % 10;
    j = n_old + (rand() >> 8) % 10;
    s1 = (rand()) / ((double )((double )2147483647));
    sp_set_val(A,i,j,s1);
    if (fabs(s1 - sp_get_val(A,i,j)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
      break; 
  }
  if (k < 20) 
    printf("Error: %s error: line %d\n","sp_resize()",233);
  sp_col_access(A);
  if (!chk_col_accessSPT(A)) {
    printf("Error: %s error: line %d\n","sp_col_access()",237);
  }
  sp_diag_access(A);
  for (i = 0; i <= A -> m - 1; i += 1) {
    r = &A -> row[i];
    if (r -> diag != sprow_idx(r,i)) 
      break; 
  }
  if (i < A -> m) {
    printf("Error: %s error: line %d\n","sp_diag_access()",248);
  }
/* test both sp_mv_mlt() and sp_vm_mlt() */
  x = v_resize(x,B -> n);
  y = v_resize(y,B -> m);
  u = v_get(B -> m);
  v = v_get(B -> n);
  for (k = 0; k <= 9; k += 1) {
    v_rand(x);
    v_rand(y);
    sp_mv_mlt(B,x,u);
    sp_vm_mlt(B,y,v);
    if (fabs(_in_prod(x,v,0) - _in_prod(y,u,0)) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * _v_norm2(u,((VEC *)((void *)0))) * 5) 
      break; 
  }
  if (k < 10) {
    printf("Error: %s error: line %d\n","sp_mv_mlt()/sp_vm_mlt()",268);
    printf("# Error in inner products = %g [cf MACHEPS = %g]\n",(fabs(_in_prod(x,v,0) - _in_prod(y,u,0))),(double )2.22044604925031308084726333618164062e-16L);
  }
  (sp_free(A) , A = ((SPMAT *)((void *)0)));
  (sp_free(B) , B = ((SPMAT *)((void *)0)));
  (sp_free(C) , C = ((SPMAT *)((void *)0)));
/* now test Cholesky and LU factorise and solve */
  printf("# Testing %s...\n","sparse Cholesky factorise/solve");
  ;
  A = iter_gen_sym(120,8);
  B = sp_copy(A);
  spCHfactor(A);
  x = v_resize(x,A -> m);
  y = v_resize(y,A -> m);
  v_rand(x);
  sp_mv_mlt(B,x,y);
  z = v_resize(z,A -> m);
  spCHsolve(A,y,z);
  v = v_resize(v,A -> m);
  sp_mv_mlt(B,z,v);
/* compute residual */
  v_sub(y,v,v);
  if (_v_norm2(v,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(y,((VEC *)((void *)0))) * 10) {
    printf("Error: %s error: line %d\n","spCHfactor()/spCHsolve()",294);
    printf("# Sparse Cholesky residual = %g [cf MACHEPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* compute error in solution */
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * 10) {
    printf("Error: %s error: line %d\n","spCHfactor()/spCHsolve()",302);
    printf("# Solution error = %g [cf MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* now test symbolic and incomplete factorisation */
  (sp_free(A) , A = ((SPMAT *)((void *)0)));
  A = sp_copy(B);
  mem_stat_mark(2);
  spCHsymb(A);
  mem_stat_mark(2);
  spICHfactor(A);
  spCHsolve(A,y,z);
  v = v_resize(v,A -> m);
  sp_mv_mlt(B,z,v);
/* compute residual */
  v_sub(y,v,v);
  if (_v_norm2(v,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(y,((VEC *)((void *)0))) * 5) {
    printf("Error: %s error: line %d\n","spCHsymb()/spICHfactor()",323);
    printf("# Sparse Cholesky residual = %g [cf MACHEPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* compute error in solution */
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * 10) {
    printf("Error: %s error: line %d\n","spCHsymb()/spICHfactor()",331);
    printf("# Solution error = %g [cf MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* now test sparse LU factorisation */
  printf("# Testing %s...\n","sparse LU factorise/solve");
  ;
  (sp_free(A) , A = ((SPMAT *)((void *)0)));
  (sp_free(B) , B = ((SPMAT *)((void *)0)));
  A = iter_gen_nonsym(100,100,8,1.0);
  B = sp_copy(A);
  x = v_resize(x,A -> n);
  z = v_resize(z,A -> n);
  y = v_resize(y,A -> m);
  v = v_resize(v,A -> m);
  v_rand(x);
  sp_mv_mlt(B,x,y);
  pivot = px_get(A -> m);
  mem_stat_mark(3);
  spLUfactor(A,pivot,0.1);
  spLUsolve(A,pivot,y,z);
  mem_stat_free_list(3,0);
  sp_mv_mlt(B,z,v);
/* compute residual */
  v_sub(y,v,v);
  if (_v_norm2(v,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(y,((VEC *)((void *)0))) * (A -> m)) {
    printf("Error: %s error: line %d\n","spLUfactor()/spLUsolve()",362);
    printf("# Sparse LU residual = %g [cf MACHEPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* compute error in solution */
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * 100 * (A -> m)) {
    printf("Error: %s error: line %d\n","spLUfactor()/spLUsolve()",370);
    printf("# Sparse LU solution error = %g [cf MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* now check spLUTsolve */
  mem_stat_mark(4);
  sp_vm_mlt(B,x,y);
  spLUTsolve(A,pivot,y,z);
  sp_vm_mlt(B,z,v);
  mem_stat_free_list(4,0);
/* compute residual */
  v_sub(y,v,v);
  if (_v_norm2(v,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(y,((VEC *)((void *)0))) * (A -> m)) {
    printf("Error: %s error: line %d\n","spLUTsolve()",386);
    printf("# Sparse LU residual = %g [cf MACHEPS = %g]\n",(_v_norm2(v,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* compute error in solution */
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * 100 * (A -> m)) {
    printf("Error: %s error: line %d\n","spLUTsolve()",394);
    printf("# Sparse LU solution error = %g [cf MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* algebraic operations */
  printf("# Testing %s...\n","addition,subtraction and multiplying by a number");
  ;
  (sp_free(A) , A = ((SPMAT *)((void *)0)));
  (sp_free(B) , B = ((SPMAT *)((void *)0)));
  m = 120;
  n = 120;
  deg = 5;
  A = sp_get(m,n,deg);
  B = sp_get(m,n,deg);
  C = sp_get(m,n,deg);
  C1 = sp_get(m,n,deg);
  for (k = 0; k <= m * deg - 1; k += 1) {
    i = (rand() >> 8) % m;
    j = (rand() >> 8) % n;
    sp_set_val(A,i,j,(rand()) / ((double )((double )2147483647)));
    i = (rand() >> 8) % m;
    j = (rand() >> 8) % n;
    sp_set_val(B,i,j,(rand()) / ((double )((double )2147483647)));
  }
  s1 = mrand();
  B1 = sp_copy(B);
  mem_stat_mark(1);
  sp_smlt(B,s1,C);
  sp_add(A,C,C1);
  sp_sub(C1,A,C);
  sp_smlt(C,- 1.0 / s1,C);
  sp_add(C,B1,C);
  s2 = 0.0;
  for (k = 0; k <= C -> m - 1; k += 1) {
    r = &C -> row[k];
    for (j = 0; j <= r -> len - 1; j += 1) {
      if (s2 < fabs(r -> elt[j] . val)) 
        s2 = fabs(r -> elt[j] . val);
    }
  }
  if (s2 > ((double )2.22044604925031308084726333618164062e-16L) * (A -> m)) {
    printf("Error: %s error: line %d\n","add, sub, mlt sparse matrices (args not in situ)\n",442);
    printf(" difference = %g [MACEPS = %g]\n",s2,(double )2.22044604925031308084726333618164062e-16L);
  }
  sp_mltadd(A,B1,s1,C1);
  sp_sub(C1,A,A);
  sp_smlt(A,1.0 / s1,C1);
  sp_sub(C1,B1,C1);
  mem_stat_free_list(1,0);
  s2 = 0.0;
  for (k = 0; k <= C1 -> m - 1; k += 1) {
    r = &C1 -> row[k];
    for (j = 0; j <= r -> len - 1; j += 1) {
      if (s2 < fabs(r -> elt[j] . val)) 
        s2 = fabs(r -> elt[j] . val);
    }
  }
  if (s2 > ((double )2.22044604925031308084726333618164062e-16L) * (A -> m)) {
    printf("Error: %s error: line %d\n","add, sub, mlt sparse matrices (args not in situ)\n",462);
    printf(" difference = %g [MACEPS = %g]\n",s2,(double )2.22044604925031308084726333618164062e-16L);
  }
  (v_free(x) , x = ((VEC *)((void *)0)));
  (v_free(y) , y = ((VEC *)((void *)0)));
  (v_free(z) , z = ((VEC *)((void *)0)));
  (v_free(u) , u = ((VEC *)((void *)0)));
  (v_free(v) , v = ((VEC *)((void *)0)));
  (px_free(pivot) , pivot = ((PERM *)((void *)0)));
  (sp_free(A) , A = ((SPMAT *)((void *)0)));
  (sp_free(B) , B = ((SPMAT *)((void *)0)));
  (sp_free(C) , C = ((SPMAT *)((void *)0)));
  (sp_free(B1) , B1 = ((SPMAT *)((void *)0)));
  (sp_free(C1) , C1 = ((SPMAT *)((void *)0)));
  printf("# Done testing (%s)\n",argv[0]);
  mem_info_file(stdout,0);
}
