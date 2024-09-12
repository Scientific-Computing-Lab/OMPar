/**************************************************************************
**
** Copyright (C) 1993 David E. Stewart & Zbigniew Leyk, all rights reserved.
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
	This file contains a series of tests for the Meschach matrix
	library, parts 1 and 2
*/
#include <omp.h> 
static char rcsid[] = "$Id: torture.c,v 1.6 1994/08/25 15:22:11 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"matrix2.h"
#include        "matlab.h"
#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);
static char *test_err_list[] = {("unknown error"), ("testing error messages"), ("unexpected end-of-file")
/* 0 */
/* 1 */
/* 2 */
};
#define MAX_TEST_ERR   (sizeof(test_err_list)/sizeof(char *))
/* extern	int	malloc_chain_check(); */
/* #define MEMCHK() if ( malloc_chain_check(0) ) \
{ printf("Error in malloc chain: \"%s\", line %d\n", \
	 __FILE__, __LINE__); exit(0); } */
#define	MEMCHK() 
/* cmp_perm -- returns 1 if pi1 == pi2, 0 otherwise */

int cmp_perm(pi1,pi2)
PERM *pi1;
PERM *pi2;
{
  int i;
  if (!pi1 || !pi2) 
    ev_err("torture.c",8,63,"cmp_perm",0);
  if (pi1 -> size != pi2 -> size) 
    return 0;
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= pi1 -> size - 1; i += 1) {
    if (pi1 -> pe[i] != pi2 -> pe[i]) 
      return 0;
  }
  return 1;
}
/* px_rand -- generates sort-of random permutation */

PERM *px_rand(pi)
PERM *pi;
{
  int i;
  int j;
  int k;
  if (!pi) 
    ev_err("torture.c",8,79,"px_rand",0);
  for (i = 0; ((unsigned int )i) <= ((unsigned int )3) * pi -> size - 1; i += 1) {
    j = ((rand() >> 8) % pi -> size);
    k = ((rand() >> 8) % pi -> size);
    px_transp(pi,j,k);
  }
  return pi;
}
#define	SAVE_FILE	"asx5213a.mat"
#define	MATLAB_NAME	"alpha"
char name[81] = "alpha";

int main(argc,argv)
int argc;
char *argv[];
{
  VEC *x = (VEC *)((void *)0);
  VEC *y = (VEC *)((void *)0);
  VEC *z = (VEC *)((void *)0);
  VEC *u = (VEC *)((void *)0);
  VEC *v = (VEC *)((void *)0);
  VEC *w = (VEC *)((void *)0);
  VEC *diag = (VEC *)((void *)0);
  VEC *beta = (VEC *)((void *)0);
  PERM *pi1 = (PERM *)((void *)0);
  PERM *pi2 = (PERM *)((void *)0);
  PERM *pi3 = (PERM *)((void *)0);
  PERM *pivot = (PERM *)((void *)0);
  PERM *blocks = (PERM *)((void *)0);
  MAT *A = (MAT *)((void *)0);
  MAT *B = (MAT *)((void *)0);
  MAT *C = (MAT *)((void *)0);
  MAT *D = (MAT *)((void *)0);
  MAT *Q = (MAT *)((void *)0);
  MAT *U = (MAT *)((void *)0);
  BAND *bA;
  BAND *bB;
  BAND *bC;
  double cond_est;
  double s1;
  double s2;
  double s3;
  int i;
  int j;
  int seed;
  FILE *fp;
  char *cp;
  mem_info_on(1);
  setbuf(stdout,(char *)((void *)0));
  seed = 1111;
  if (argc > 2) {
    printf("usage: %s [seed]\n",argv[0]);
    exit(0);
  }
   else if (argc == 2) 
    sscanf(argv[1],"%d",&seed);
/* set seed for rand() */
  smrand(seed);
  mem_stat_mark(1);
/* print version information */
  m_version();
  printf("# grep \"^Error\" the output for a listing of errors\n");
  printf("# Don't panic if you see \"Error\" appearing; \n");
  printf("# Also check the reported size of error\n");
  printf("# This program uses randomly generated problems and therefore\n");
  printf("# may occasionally produce ill-conditioned problems\n");
  printf("# Therefore check the size of the error compared with MACHEPS\n");
  printf("# If the error is within 1000*MACHEPS then don't worry\n");
  printf("# If you get an error of size 0.1 or larger there is \n");
  printf("# probably a bug in the code or the compilation procedure\n\n");
  printf("# seed = %d\n",seed);
  printf("# Check: MACHEPS = %g\n",(double )2.22044604925031308084726333618164062e-16L);
/* allocate, initialise, copy and resize operations */
/* VEC */
  printf("# Testing %s...\n","vector initialise, copy & resize");
  ;
  x = v_get(12);
  y = v_get(15);
  z = v_get(12);
  v_rand(x);
  v_rand(y);
  z = _v_copy(x,z,0);
  if (_v_norm2((v_sub(x,z,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","VEC copy",156);
  _v_copy(x,y,0);
  x = v_resize(x,10);
  y = v_resize(y,10);
  if (_v_norm2((v_sub(x,y,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","VEC copy/resize",161);
  x = v_resize(x,15);
  y = v_resize(y,15);
  if (_v_norm2((v_sub(x,y,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","VEC resize",165);
/* MAT */
  printf("# Testing %s...\n","matrix initialise, copy & resize");
  ;
  A = m_get(8,5);
  B = m_get(3,9);
  C = m_get(8,5);
  m_rand(A);
  m_rand(B);
  C = _m_copy(A,C,0,0);
  if (m_norm_inf((m_sub(A,C,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","MAT copy",176);
  _m_copy(A,B,0,0);
  A = m_resize(A,3,5);
  B = m_resize(B,3,5);
  if (m_norm_inf((m_sub(A,B,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","MAT copy/resize",181);
  A = m_resize(A,10,10);
  B = m_resize(B,10,10);
  if (m_norm_inf((m_sub(A,B,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","MAT resize",185);
  ;
/* PERM */
  printf("# Testing %s...\n","permutation initialise, inverting & permuting vectors");
  ;
  pi1 = px_get(15);
  pi2 = px_get(12);
  px_rand(pi1);
  v_rand(x);
  px_vec(pi1,x,z);
  y = v_resize(y,(x -> dim));
  pxinv_vec(pi1,z,y);
  if (_v_norm2((v_sub(x,y,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","PERMute vector",199);
  pi2 = px_inv(pi1,pi2);
  pi3 = px_mlt(pi1,pi2,(PERM *)((void *)0));
  for (i = 0; ((unsigned int )i) <= pi3 -> size - 1; i += 1) {
    if (pi3 -> pe[i] != i) 
      printf("Error: %s error: line %d\n","PERM inverse/multiply",204);
  }
/* testing catch() etc */
  printf("# Testing %s...\n","error handling routines");
  ;
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(3);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {{
        jmp_buf _save;
        int _err_num;
        int _old_flag;
        _old_flag = set_err_flag(3);
        memmove(_save,restart,sizeof(jmp_buf ));
        if ((_err_num = _setjmp(restart)) == 0) {
          v_add(((VEC *)((void *)0)),((VEC *)((void *)0)),(VEC *)((void *)0));
          printf("Error: %s error: line %d\n","tracecatch() failure",213);
          set_err_flag(_old_flag);
          memmove(restart,_save,sizeof(jmp_buf ));
        }
         else {
          set_err_flag(_old_flag);
          memmove(restart,_save,sizeof(jmp_buf ));
          printf("# tracecatch() caught error\n");
          ev_err("torture.c",8,213,"main",0);
        }
      }
      ;
      printf("Error: %s error: line %d\n","catch() failure",213);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else if (_err_num == 8) {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      printf("# catch() caught E_NULL error\n");
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("torture.c",_err_num,214,"catch",0);
    }
  }
  ;
/* testing attaching a new error list (error list 2) */
  printf("# Testing %s...\n","attaching error lists");
  ;
  printf("# IT IS NOT A REAL WARNING ... \n");
  err_list_attach(2,(sizeof(test_err_list) / sizeof(char *)),test_err_list,1);
  if (!err_is_list_attached(2)) 
    printf("Error: %s error: line %d\n","attaching the error list 2",222);
  ev_err("torture.c",1,223,"main",2);
  err_list_free(2);
  if (err_is_list_attached(2)) 
    printf("Error: %s error: line %d\n","detaching the error list 2",226);
/* testing inner products and v_mltadd() etc */
  printf("# Testing %s...\n","inner products and linear combinations");
  ;
  u = v_get((x -> dim));
  v_rand(u);
  v_rand(x);
  v_resize(y,(x -> dim));
  v_rand(y);
  v_mltadd(y,x,-_in_prod(x,y,0) / _in_prod(x,x,0),z);
  if (fabs((_in_prod(x,z,0))) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf("Error: %s error: line %d\n","v_mltadd()/in_prod()",237);
  s1 = -_in_prod(x,y,0) / (_v_norm2(x,((VEC *)((void *)0))) * _v_norm2(x,((VEC *)((void *)0))));
  sv_mlt(s1,x,u);
  v_add(y,u,u);
  if (_v_norm2((v_sub(u,z,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf("Error: %s error: line %d\n","sv_mlt()/v_norm2()",242);
#ifdef ANSI_C 
  v_linlist(u,x,s1,y,1.0,(VEC *)((void *)0));
  if (_v_norm2((v_sub(u,z,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf("Error: %s error: line %d\n","v_linlist()",247);
#endif
#ifdef VARARGS
  v_linlist(u,x,s1,y,1.0,(VEC *)((void *)0));
  if (_v_norm2((v_sub(u,z,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf("Error: %s error: line %d\n","v_linlist()",252);
#endif
  ;
/* vector norms */
  printf("# Testing %s...\n","vector norms");
  ;
  x = v_resize(x,12);
  v_rand(x);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= x -> dim - 1; i += 1) {
    if (x -> ve[i] >= 0.5) 
      x -> ve[i] = 1.0;
     else 
      x -> ve[i] = - 1.0;
  }
  s1 = _v_norm1(x,((VEC *)((void *)0)));
  s2 = _v_norm2(x,((VEC *)((void *)0)));
  s3 = _v_norm_inf(x,((VEC *)((void *)0)));
  if (fabs(s1 - (x -> dim)) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim) || fabs(s2 - sqrt((double )(x -> dim))) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim) || fabs(s3 - 1.0) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","v_norm1/2/_inf()",273);
/* test matrix multiply etc */
  printf("# Testing %s...\n","matrix multiply and invert");
  ;
  A = m_resize(A,10,10);
  B = m_resize(B,10,10);
  m_rand(A);
  m_inverse(A,B);
  m_mlt(A,B,C);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = C -> me[i][i] - 1.0;
  }
  if (m_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm_inf(A) * m_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","m_inverse()/m_mlt()",285);
  ;
/* ... and transposes */
  printf("# Testing %s...\n","transposes and transpose-multiplies");
  ;
  m_transp(A,A);
/* can do square matrices in situ */
  mtrm_mlt(A,B,C);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = C -> me[i][i] - 1.0;
  }
  if (m_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm_inf(A) * m_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","m_transp()/mtrm_mlt()",296);
  m_transp(A,A);
  m_transp(B,B);
  mmtr_mlt(A,B,C);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = C -> me[i][i] - 1.0;
  }
  if (m_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm_inf(A) * m_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","m_transp()/mmtr_mlt()",303);
  sm_mlt(3.71,B,B);
  mmtr_mlt(A,B,C);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = C -> me[i][i] - 3.71;
  }
  if (m_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm_inf(A) * m_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","sm_mlt()/mmtr_mlt()",309);
  m_transp(B,B);
  sm_mlt(1.0 / 3.71,B,B);
  ;
/* ... and matrix-vector multiplies */
  printf("# Testing %s...\n","matrix-vector multiplies");
  ;
  x = v_resize(x,(A -> n));
  y = v_resize(y,(A -> m));
  z = v_resize(z,(A -> m));
  u = v_resize(u,(A -> n));
  v_rand(x);
  v_rand(y);
  mv_mlt(A,x,z);
  s1 = _in_prod(y,z,0);
  vm_mlt(A,y,u);
  s2 = _in_prod(u,x,0);
  if (fabs(s1 - s2) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim) * (x -> dim)) 
    printf("Error: %s error: line %d\n","mv_mlt()/vm_mlt()",328);
  mv_mlt(B,z,u);
  if (_v_norm2((v_sub(u,x,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm_inf(A) * m_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","mv_mlt()/m_inverse()",331);
  ;
/* get/set row/col */
  printf("# Testing %s...\n","getting and setting rows and cols");
  ;
  x = v_resize(x,(A -> n));
  y = v_resize(y,(B -> m));
  x = get_row(A,3,x);
  y = get_col(B,3,y);
  if (fabs(_in_prod(x,y,0) - 1.0) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm_inf(A) * m_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","get_row()/get_col()",342);
  sv_mlt(- 1.0,x,x);
  sv_mlt(- 1.0,y,y);
  _set_row(A,3,x,0);
  _set_col(B,3,y,0);
  m_mlt(A,B,C);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = C -> me[i][i] - 1.0;
  }
  if (m_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm_inf(A) * m_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","set_row()/set_col()",351);
  ;
/* matrix norms */
  printf("# Testing %s...\n","matrix norms");
  ;
  A = m_resize(A,11,15);
  m_rand(A);
  s1 = m_norm_inf(A);
  B = m_transp(A,B);
  s2 = m_norm1(B);
  if (fabs(s1 - s2) >= ((double )2.22044604925031308084726333618164062e-16L) * (A -> m)) 
    printf("Error: %s error: line %d\n","m_norm1()/m_norm_inf()",363);
  C = mtrm_mlt(A,A,C);
  s1 = 0.0;
  for (i = 0; i < C -> m && i < C -> n; i++) 
    s1 += C -> me[i][i];
  if (fabs(sqrt(s1) - m_norm_frob(A)) >= ((double )2.22044604925031308084726333618164062e-16L) * (A -> m) * (A -> n)) 
    printf("Error: %s error: line %d\n","m_norm_frob",369);
  ;
/* permuting rows and columns */
  printf("# Testing %s...\n","permuting rows & cols");
  ;
  A = m_resize(A,11,15);
  B = m_resize(B,11,15);
  pi1 = px_resize(pi1,(A -> m));
  px_rand(pi1);
  x = v_resize(x,(A -> n));
  y = mv_mlt(A,x,y);
  px_rows(pi1,A,B);
  px_vec(pi1,y,z);
  mv_mlt(B,x,u);
  if (_v_norm2((v_sub(z,u,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * (A -> m)) 
    printf("Error: %s error: line %d\n","px_rows()",385);
  pi1 = px_resize(pi1,(A -> n));
  px_rand(pi1);
  px_cols(pi1,A,B);
  pxinv_vec(pi1,x,z);
  mv_mlt(B,z,u);
  if (_v_norm2((v_sub(y,u,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * (A -> n)) 
    printf("Error: %s error: line %d\n","px_cols()",392);
  ;
/* MATLAB save/load */
  printf("# Testing %s...\n","MATLAB save/load");
  ;
  A = m_resize(A,12,11);
  if ((fp = fopen("asx5213a.mat","w")) == ((FILE *)((void *)0))) 
    printf("Cannot perform MATLAB save/load test\n");
   else {
    m_rand(A);
    m_save(fp,A,name);
    fclose(fp);
    if ((fp = fopen("asx5213a.mat","r")) == ((FILE *)((void *)0))) 
      printf("Cannot open save file \"%s\"\n","asx5213a.mat");
     else {
      (m_free(B) , B = ((MAT *)((void *)0)));
      B = m_load(fp,&cp);
      if (strcmp(name,cp) || m_norm1((m_sub(A,B,B))) >= ((double )2.22044604925031308084726333618164062e-16L) * (A -> m)) 
        printf("Error: %s error: line %d\n","mload()/m_save()",413);
    }
  }
  ;
/* Now, onto matrix factorisations */
  A = m_resize(A,10,10);
  B = m_resize(B,(A -> m),(A -> n));
  _m_copy(A,B,0,0);
  x = v_resize(x,(A -> n));
  y = v_resize(y,(A -> m));
  z = v_resize(z,(A -> n));
  u = v_resize(u,(A -> m));
  v_rand(x);
  mv_mlt(B,x,y);
  z = _v_copy(x,z,0);
  printf("# Testing %s...\n","LU factor/solve");
  ;
  pivot = px_get((A -> m));
  LUfactor(A,pivot);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      LUsolve(A,pivot,y,x);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("torture.c",_err_num,434,"main",0);
    }
  }
  ;
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      cond_est = LUcondest(A,pivot);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("torture.c",_err_num,435,"main",0);
    }
  }
  ;
  printf("# cond(A) approx= %g\n",cond_est);
  if (_v_norm2((v_sub(x,z,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * cond_est) {
    printf("Error: %s error: line %d\n","LUfactor()/LUsolve()",439);
    printf("# LU solution error = %g [cf MACHEPS = %g]\n",(_v_norm2((v_sub(x,z,u)),((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  _v_copy(y,x,0);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      LUsolve(A,pivot,x,x);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("torture.c",_err_num,445,"main",0);
    }
  }
  ;
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      cond_est = LUcondest(A,pivot);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("torture.c",_err_num,446,"main",0);
    }
  }
  ;
  if (_v_norm2((v_sub(x,z,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * cond_est) {
    printf("Error: %s error: line %d\n","LUfactor()/LUsolve()",449);
    printf("# LU solution error = %g [cf MACHEPS = %g]\n",(_v_norm2((v_sub(x,z,u)),((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  vm_mlt(B,z,y);
  _v_copy(y,x,0);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      LUTsolve(A,pivot,x,x);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("torture.c",_err_num,456,"main",0);
    }
  }
  ;
  if (_v_norm2((v_sub(x,z,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * cond_est) {
    printf("Error: %s error: line %d\n","LUfactor()/LUTsolve()",459);
    printf("# LU solution error = %g [cf MACHEPS = %g]\n",(_v_norm2((v_sub(x,z,u)),((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* QR factorisation */
  _m_copy(B,A,0,0);
  mv_mlt(B,z,y);
  printf("# Testing %s...\n","QR factor/solve:");
  ;
  diag = v_get((A -> m));
  beta = v_get((A -> m));
  QRfactor(A,diag);
  QRsolve(A,diag,y,x);
  if (_v_norm2((v_sub(x,z,u)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * cond_est) {
    printf("Error: %s error: line %d\n","QRfactor()/QRsolve()",476);
    printf("# QR solution error = %g [cf MACHEPS = %g]\n",(_v_norm2((v_sub(x,z,u)),((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  Q = m_get((A -> m),(A -> m));
  makeQ(A,diag,Q);
  makeR(A,A);
  m_mlt(Q,A,C);
  m_sub(B,C,C);
  if (m_norm1(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm1(B)) {
    printf("Error: %s error: line %d\n","QRfactor()/makeQ()/makeR()",487);
    printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",(m_norm1(C)),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* now try with a non-square matrix */
  A = m_resize(A,15,7);
  m_rand(A);
  B = _m_copy(A,B,0,0);
  diag = v_resize(diag,(A -> n));
  beta = v_resize(beta,(A -> n));
  x = v_resize(x,(A -> n));
  y = v_resize(y,(A -> m));
  v_rand(y);
  QRfactor(A,diag);
  x = QRsolve(A,diag,y,x);
/* z is the residual vector */
  mv_mlt(B,x,z);
  v_sub(z,y,z);
/* check B^T.z = 0 */
  vm_mlt(B,z,u);
  if (_v_norm2(u,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(B) * _v_norm2(y,((VEC *)((void *)0)))) {
    printf("Error: %s error: line %d\n","QRfactor()/QRsolve()",511);
    printf("# QR solution error = %g [cf MACHEPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  Q = m_resize(Q,(A -> m),(A -> m));
  makeQ(A,diag,Q);
  makeR(A,A);
  m_mlt(Q,A,C);
  m_sub(B,C,C);
  if (m_norm1(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm1(B)) {
    printf("Error: %s error: line %d\n","QRfactor()/makeQ()/makeR()",522);
    printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",(m_norm1(C)),(double )2.22044604925031308084726333618164062e-16L);
  }
  D = m_get((A -> m),(Q -> m));
  mtrm_mlt(Q,Q,D);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= D -> m - 1; i += 1) {
    D -> me[i][i] = D -> me[i][i] - 1.0;
  }
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm_inf(Q)) {
    printf("Error: %s error: line %d\n","QRfactor()/makeQ()/makeR()",532);
    printf("# QR orthogonality error = %g [cf MACHEPS = %g]\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* QRCP factorisation */
  _m_copy(B,A,0,0);
  printf("# Testing %s...\n","QR factor/solve with column pivoting");
  ;
  pivot = px_resize(pivot,(A -> n));
  QRCPfactor(A,diag,pivot);
  z = v_resize(z,(A -> n));
  QRCPsolve(A,diag,pivot,y,z);
/* pxinv_vec(pivot,z,x); */
/* now compute residual (z) vector */
  mv_mlt(B,x,z);
  v_sub(z,y,z);
/* check B^T.z = 0 */
  vm_mlt(B,z,u);
  if (_v_norm2(u,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(B) * _v_norm2(y,((VEC *)((void *)0)))) {
    printf("Error: %s error: line %d\n","QRCPfactor()/QRsolve()",553);
    printf("# QR solution error = %g [cf MACHEPS = %g]\n",(_v_norm2(u,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  Q = m_resize(Q,(A -> m),(A -> m));
  makeQ(A,diag,Q);
  makeR(A,A);
  m_mlt(Q,A,C);
  (m_free(D) , D = ((MAT *)((void *)0)));
  D = m_get((B -> m),(B -> n));
  px_cols(pivot,C,D);
  m_sub(B,D,D);
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm1(B)) {
    printf("Error: %s error: line %d\n","QRCPfactor()/makeQ()/makeR()",568);
    printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* Cholesky and LDL^T factorisation */
/* Use these for normal equations approach */
  printf("# Testing %s...\n","Cholesky factor/solve");
  ;
  mtrm_mlt(B,B,A);
  CHfactor(A);
  u = v_resize(u,(B -> n));
  vm_mlt(B,y,u);
  z = v_resize(z,(B -> n));
  CHsolve(A,u,z);
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * 100) {
    printf("Error: %s error: line %d\n","CHfactor()/CHsolve()",587);
    printf("# Cholesky solution error = %g [cf MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* modified Cholesky factorisation should be identical with Cholesky
       factorisation provided the matrix is "sufficiently positive definite" */
  mtrm_mlt(B,B,C);
  MCHfactor(C,(double )2.22044604925031308084726333618164062e-16L);
  m_sub(A,C,C);
  if (m_norm1(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(A)) {
    printf("Error: %s error: line %d\n","MCHfactor()",598);
    printf("# Modified Cholesky error = %g [cf MACHEPS = %g]\n",(m_norm1(C)),(double )2.22044604925031308084726333618164062e-16L);
  }
/* now test the LDL^T factorisation -- using a negative def. matrix */
  mtrm_mlt(B,B,A);
  sm_mlt(- 1.0,A,A);
  _m_copy(A,C,0,0);
  LDLfactor(A);
  LDLsolve(A,u,z);
  w = v_get((A -> m));
  mv_mlt(C,z,w);
  v_sub(w,u,w);
  if (_v_norm2(w,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(u,((VEC *)((void *)0))) * m_norm1(C)) {
    printf("Error: %s error: line %d\n","LDLfactor()/LDLsolve()",613);
    printf("# LDL^T residual = %g [cf MACHEPS = %g]\n",(_v_norm2(w,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  v_add(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * _v_norm2(x,((VEC *)((void *)0))) * 100) {
    printf("Error: %s error: line %d\n","LDLfactor()/LDLsolve()",620);
    printf("# LDL^T solution error = %g [cf MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* and now the Bunch-Kaufman-Parlett method */
/* set up D to be an indefinite diagonal matrix */
  printf("# Testing %s...\n","Bunch-Kaufman-Parlett factor/solve");
  ;
  D = m_resize(D,(B -> m),(B -> m));
  m_zero(D);
  w = v_resize(w,(B -> m));
  v_rand(w);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= w -> dim - 1; i += 1) {
    if (w -> ve[i] >= 0.5) 
      D -> me[i][i] = 1.0;
     else 
      D -> me[i][i] = - 1.0;
  }
/* set A <- B^T.D.B */
  C = m_resize(C,(B -> n),(B -> n));
  C = mtrm_mlt(B,D,C);
  A = m_mlt(C,B,A);
  C = m_resize(C,(B -> n),(B -> n));
  C = _m_copy(A,C,0,0);
/* ... and use BKPfactor() */
  blocks = px_get((A -> m));
  pivot = px_resize(pivot,(A -> m));
  x = v_resize(x,(A -> m));
  y = v_resize(y,(A -> m));
  z = v_resize(z,(A -> m));
  v_rand(x);
  mv_mlt(A,x,y);
  BKPfactor(A,pivot,blocks);
  printf("# BKP pivot =\n");
  px_foutput(stdout,pivot);
  printf("# BKP blocks =\n");
  px_foutput(stdout,blocks);
  BKPsolve(A,pivot,blocks,y,z);
/* compute & check residual */
  mv_mlt(C,z,w);
  v_sub(w,y,w);
  if (_v_norm2(w,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(C) * _v_norm2(z,((VEC *)((void *)0)))) {
    printf("Error: %s error: line %d\n","BKPfactor()/BKPsolve()",663);
    printf("# BKP residual size = %g [cf MACHEPS = %g]\n",(_v_norm2(w,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* check update routines */
/* check LDLupdate() first */
  printf("# Testing %s...\n","update L.D.L^T routine");
  ;
  A = mtrm_mlt(B,B,A);
  m_resize(C,(A -> m),(A -> n));
  C = _m_copy(A,C,0,0);
  LDLfactor(A);
  s1 = 3.7;
  w = v_resize(w,(A -> m));
  v_rand(w);
  
#pragma omp parallel for private (i,j)
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; ((unsigned int )j) <= C -> n - 1; j += 1) {
      C -> me[i][j] = C -> me[i][j] + s1 * w -> ve[i] * w -> ve[j];
    }
  }
  LDLfactor(C);
  LDLupdate(A,w,s1);
/* zero out strictly upper triangular parts of A and C */
  
#pragma omp parallel for private (i,j)
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = i + 1; ((unsigned int )j) <= A -> n - 1; j += 1) {
      A -> me[i][j] = 0.0;
      C -> me[i][j] = 0.0;
    }
  }
  if (m_norm1((m_sub(A,C,C))) >= sqrt((double )2.22044604925031308084726333618164062e-16L) * m_norm1(A)) {
    printf("Error: %s error: line %d\n","LDLupdate()",692);
    printf("# LDL update matrix error = %g [cf MACHEPS = %g]\n",(m_norm1(C)),(double )2.22044604925031308084726333618164062e-16L);
  }
/* BAND MATRICES */
#define COL 40
#define UDIAG  5
#define LDIAG  2
  smrand(101);
  bA = bd_get(2,5,40);
  bB = bd_get(2,5,40);
  bC = bd_get(2,5,40);
  A = m_resize(A,40,40);
  B = m_resize(B,40,40);
  pivot = px_resize(pivot,40);
  x = v_resize(x,40);
  w = v_resize(w,40);
  z = v_resize(z,40);
  m_rand(A);
/* generate band matrix */
  mat2band(A,2,5,bA);
  band2mat(bA,A);
/* now A is banded */
  bB = bd_copy(bA,bB);
  v_rand(x);
  mv_mlt(A,x,w);
/* test of bd_mv_mlt */
  printf("# Testing %s...\n","bd_mv_mlt");
  ;
  bd_mv_mlt(bA,x,z);
  v_sub(z,w,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > _v_norm2(x,((VEC *)((void *)0))) * sqrt((double )2.22044604925031308084726333618164062e-16L)) {
    printf("Error: %s error: line %d\n","incorrect vector (bd_mv_mlt)",728);
    printf(" ||exact vector. - computed vector.|| = %g [MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
  z = _v_copy(w,z,0);
  printf("# Testing %s...\n","band LU factorization");
  ;
  bdLUfactor(bA,pivot);
/* pivot will be changed */
  bdLUsolve(bA,pivot,z,z);
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > _v_norm2(x,((VEC *)((void *)0))) * sqrt((double )2.22044604925031308084726333618164062e-16L)) {
    printf("Error: %s error: line %d\n","incorrect solution (band LU factorization)",742);
    printf(" ||exact sol. - computed sol.|| = %g [MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* solve transpose system */
  printf("# Testing %s...\n","band LU factorization for transpose system");
  ;
  m_transp(A,B);
  mv_mlt(B,x,w);
  bd_copy(bB,bA);
  bd_transp(bA,bA);
/* transposition in situ */
  bd_transp(bA,bB);
  bd_transp(bB,bB);
  bdLUfactor(bB,pivot);
  bdLUsolve(bB,pivot,w,z);
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > _v_norm2(x,((VEC *)((void *)0))) * sqrt((double )2.22044604925031308084726333618164062e-16L)) {
    printf("Error: %s error: line %d\n","incorrect solution (band transposed LU factorization)",764);
    printf(" ||exact sol. - computed sol.|| = %g [MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* Cholesky factorization */
  printf("# Testing %s...\n","band Choleski LDL' factorization");
  ;
  m_add(A,B,A);
/* symmetric matrix */
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 39; i += 1) {
/* positive definite */
    A -> me[i][i] += (2 * 2);
  }
  mat2band(A,2,2,bA);
  band2mat(bA,A);
/* corresponding matrix A */
  v_rand(x);
  mv_mlt(A,x,w);
  z = _v_copy(w,z,0);
  bdLDLfactor(bA);
  z = bdLDLsolve(bA,z,z);
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > _v_norm2(x,((VEC *)((void *)0))) * sqrt((double )2.22044604925031308084726333618164062e-16L)) {
    printf("Error: %s error: line %d\n","incorrect solution (band LDL' factorization)",789);
    printf(" ||exact sol. - computed sol.|| = %g [MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* new bandwidths */
  m_rand(A);
  bA = bd_resize(bA,5,2,40);
  bB = bd_resize(bB,5,2,40);
  mat2band(A,5,2,bA);
  band2mat(bA,A);
  bd_copy(bA,bB);
  mv_mlt(A,x,w);
  printf("# Testing %s...\n","band LU factorization (resized)");
  ;
  bdLUfactor(bA,pivot);
/* pivot will be changed */
  bdLUsolve(bA,pivot,w,z);
  v_sub(x,z,z);
  if (_v_norm2(z,((VEC *)((void *)0))) > _v_norm2(x,((VEC *)((void *)0))) * sqrt((double )2.22044604925031308084726333618164062e-16L)) {
    printf("Error: %s error: line %d\n","incorrect solution (band LU factorization)",811);
    printf(" ||exact sol. - computed sol.|| = %g [MACHEPS = %g]\n",(_v_norm2(z,((VEC *)((void *)0)))),(double )2.22044604925031308084726333618164062e-16L);
  }
/* testing transposition */
  printf("# Testing %s...\n","band matrix transposition");
  ;
  m_zero(bA -> mat);
  bd_copy(bB,bA);
  m_zero(bB -> mat);
  bd_copy(bA,bB);
  bd_transp(bB,bB);
  bd_transp(bB,bB);
  m_zero(bC -> mat);
  bd_copy(bB,bC);
  m_sub((bA -> mat),(bC -> mat),bC -> mat);
  if (m_norm_inf((bC -> mat)) > ((double )2.22044604925031308084726333618164062e-16L) * (bC -> mat -> n)) {
    printf("Error: %s error: line %d\n","band transposition",832);
    printf(" difference ||A - (A')'|| = %g\n",(m_norm_inf((bC -> mat))));
  }
  bd_free(bA);
  bd_free(bB);
  bd_free(bC);
  ;
/* now check QRupdate() routine */
  printf("# Testing %s...\n","update QR routine");
  ;
  B = m_resize(B,15,7);
  A = m_resize(A,(B -> m),(B -> n));
  _m_copy(B,A,0,0);
  diag = v_resize(diag,(A -> n));
  beta = v_resize(beta,(A -> n));
  QRfactor(A,diag);
  Q = m_resize(Q,(A -> m),(A -> m));
  makeQ(A,diag,Q);
  makeR(A,A);
  m_resize(C,(A -> m),(A -> n));
  w = v_resize(w,(A -> m));
  v = v_resize(v,(A -> n));
  u = v_resize(u,(A -> m));
  v_rand(w);
  v_rand(v);
  vm_mlt(Q,w,u);
  QRupdate(Q,A,u,v);
  m_mlt(Q,A,C);
  
#pragma omp parallel for private (i,j)
  for (i = 0; ((unsigned int )i) <= B -> m - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; ((unsigned int )j) <= B -> n - 1; j += 1) {
      B -> me[i][j] = B -> me[i][j] + w -> ve[i] * v -> ve[j];
    }
  }
  m_sub(B,C,C);
  if (m_norm1(C) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(A) * m_norm1(Q) * 2) {
    printf("Error: %s error: line %d\n","QRupdate()",870);
    printf("# Reconstruction error in QR update = %g [cf MACHEPS = %g]\n",(m_norm1(C)),(double )2.22044604925031308084726333618164062e-16L);
  }
  m_resize(D,(Q -> m),(Q -> n));
  mtrm_mlt(Q,Q,D);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= D -> m - 1; i += 1) {
    D -> me[i][i] = D -> me[i][i] - 1.0;
  }
  if (m_norm1(D) >= 10 * ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm_inf(Q)) {
    printf("Error: %s error: line %d\n","QRupdate()",880);
    printf("# QR update orthogonality error = %g [cf MACHEPS = %g]\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
/* Now check eigenvalue/SVD routines */
  printf("# Testing %s...\n","eigenvalue and SVD routines");
  ;
  A = m_resize(A,11,11);
  B = m_resize(B,(A -> m),(A -> n));
  C = m_resize(C,(A -> m),(A -> n));
  D = m_resize(D,(A -> m),(A -> n));
  Q = m_resize(Q,(A -> m),(A -> n));
  m_rand(A);
/* A <- A + A^T  for symmetric case */
  m_add(A,(m_transp(A,C)),A);
  u = v_resize(u,(A -> m));
  u = symmeig(A,Q,u);
  m_zero(B);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= B -> m - 1; i += 1) {
    B -> me[i][i] = u -> ve[i];
  }
  m_mlt(Q,B,C);
  mmtr_mlt(C,Q,D);
  m_sub(A,D,D);
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm_inf(Q) * _v_norm_inf(u,((VEC *)((void *)0))) * 3) {
    printf("Error: %s error: line %d\n","symmeig()",906);
    printf("# Reconstruction error = %g [cf MACHEPS = %g]\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  mtrm_mlt(Q,Q,D);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= D -> m - 1; i += 1) {
    D -> me[i][i] = D -> me[i][i] - 1.0;
  }
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm_inf(Q) * 3) {
    printf("Error: %s error: line %d\n","symmeig()",915);
    printf("# symmeig() orthogonality error = %g [cf MACHEPS = %g]\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* now test (real) Schur decomposition */
/* m_copy(A,B); */
  (m_free(A) , A = ((MAT *)((void *)0)));
  A = m_get(11,11);
  m_rand(A);
  B = _m_copy(A,B,0,0);
  ;
  B = schur(B,Q);
  ;
  m_mlt(Q,B,C);
  mmtr_mlt(C,Q,D);
  ;
  m_sub(A,D,D);
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm_inf(Q) * m_norm1(B) * 5) {
    printf("Error: %s error: line %d\n","schur()",939);
    printf("# Schur reconstruction error = %g [cf MACHEPS = %g]\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
/* orthogonality check */
  mmtr_mlt(Q,Q,D);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= D -> m - 1; i += 1) {
    D -> me[i][i] = D -> me[i][i] - 1.0;
  }
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm_inf(Q) * 10) {
    printf("Error: %s error: line %d\n","schur()",950);
    printf("# Schur orthogonality error = %g [cf MACHEPS = %g]\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* now test SVD */
  A = m_resize(A,11,7);
  m_rand(A);
  U = m_get((A -> n),(A -> n));
  Q = m_resize(Q,(A -> m),(A -> m));
  u = v_resize(u,(max(A -> m,A -> n)));
  svd(A,Q,U,u);
/* check reconstruction of A */
  D = m_resize(D,(A -> m),(A -> n));
  C = m_resize(C,(A -> m),(A -> n));
  m_zero(D);
  
#pragma omp parallel for private (i)
  for (i = 0; i <= min(A -> m,A -> n) - 1; i += 1) {
    D -> me[i][i] = u -> ve[i];
  }
  mtrm_mlt(Q,D,C);
  m_mlt(C,U,D);
  m_sub(A,D,D);
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(U) * m_norm_inf(Q) * m_norm1(A)) {
    printf("Error: %s error: line %d\n","svd()",975);
    printf("# SVD reconstruction error = %g [cf MACHEPS = %g]\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
/* check orthogonality of Q and U */
  D = m_resize(D,(Q -> n),(Q -> n));
  mtrm_mlt(Q,Q,D);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= D -> m - 1; i += 1) {
    D -> me[i][i] = D -> me[i][i] - 1.0;
  }
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(Q) * m_norm_inf(Q) * 5) {
    printf("Error: %s error: line %d\n","svd()",986);
    printf("# SVD orthognality error (Q) = %g [cf MACHEPS = %g\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  D = m_resize(D,(U -> n),(U -> n));
  mtrm_mlt(U,U,D);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= D -> m - 1; i += 1) {
    D -> me[i][i] = D -> me[i][i] - 1.0;
  }
  if (m_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * m_norm1(U) * m_norm_inf(U) * 5) {
    printf("Error: %s error: line %d\n","svd()",996);
    printf("# SVD orthognality error (U) = %g [cf MACHEPS = %g\n",(m_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  for (i = 0; ((unsigned int )i) <= u -> dim - 1; i += 1) {
    if (u -> ve[i] < 0 || i < u -> dim - 1 && u -> ve[i + 1] > u -> ve[i]) 
      break; 
  }
  if (i < u -> dim) {
    printf("Error: %s error: line %d\n","svd()",1006);
    printf("# SVD sorting error\n");
  }
/* test of long vectors */
  printf("# Testing %s...\n","Long vectors");
  ;
  x = v_resize(x,100000);
  y = v_resize(y,100000);
  z = v_resize(z,100000);
  v_rand(x);
  v_rand(y);
  v_mltadd(x,y,3.0,z);
  sv_mlt(1.0 / 3.0,z,z);
  v_mltadd(z,x,- 1.0 / 3.0,z);
  v_sub(z,y,x);
  if (_v_norm2(x,((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) {
    printf("Error: %s error: line %d\n","long vectors",1023);
    printf(" norm = %g\n",(_v_norm2(x,((VEC *)((void *)0)))));
  }
  mem_stat_free_list(1,0);
  ;
/**************************************************
    VEC		*x, *y, *z, *u, *v, *w;
    VEC		*diag, *beta;
    PERM	*pi1, *pi2, *pi3, *pivot, *blocks;
    MAT		*A, *B, *C, *D, *Q, *U;
    **************************************************/
  (v_free(x) , x = ((VEC *)((void *)0)));
  (v_free(y) , y = ((VEC *)((void *)0)));
  (v_free(z) , z = ((VEC *)((void *)0)));
  (v_free(u) , u = ((VEC *)((void *)0)));
  (v_free(v) , v = ((VEC *)((void *)0)));
  (v_free(w) , w = ((VEC *)((void *)0)));
  (v_free(diag) , diag = ((VEC *)((void *)0)));
  (v_free(beta) , beta = ((VEC *)((void *)0)));
  (px_free(pi1) , pi1 = ((PERM *)((void *)0)));
  (px_free(pi2) , pi2 = ((PERM *)((void *)0)));
  (px_free(pi3) , pi3 = ((PERM *)((void *)0)));
  (px_free(pivot) , pivot = ((PERM *)((void *)0)));
  (px_free(blocks) , blocks = ((PERM *)((void *)0)));
  (m_free(A) , A = ((MAT *)((void *)0)));
  (m_free(B) , B = ((MAT *)((void *)0)));
  (m_free(C) , C = ((MAT *)((void *)0)));
  (m_free(D) , D = ((MAT *)((void *)0)));
  (m_free(Q) , Q = ((MAT *)((void *)0)));
  (m_free(U) , U = ((MAT *)((void *)0)));
  ;
  printf("# Finished torture test\n");
  mem_info_file(stdout,0);
  return 0;
}
