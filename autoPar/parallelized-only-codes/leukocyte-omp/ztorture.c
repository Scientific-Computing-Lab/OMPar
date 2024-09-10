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
	This file contains a series of tests for the Meschach matrix
	library, complex routines
*/
#include <omp.h> 
static char rcsid[] = "$Id: $";
#include	<stdio.h>
#include	<math.h>
#include 	"zmatrix2.h"
#include        "matlab.h"
#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);
/* extern	int	malloc_chain_check(); */
/* #define MEMCHK() if ( malloc_chain_check(0) ) \
{ printf("Error in malloc chain: \"%s\", line %d\n", \
	 __FILE__, __LINE__); exit(0); } */
#define	MEMCHK()
#define	checkpt()	printf("At line %d in file \"%s\"\n",__LINE__,__FILE__)
/* cmp_perm -- returns 1 if pi1 == pi2, 0 otherwise */

int cmp_perm(pi1,pi2)
PERM *pi1;
PERM *pi2;
{
  int i;
  if (!pi1 || !pi2) 
    ev_err("ztorture.c",8,58,"cmp_perm",0);
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
    ev_err("ztorture.c",8,74,"px_rand",0);
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

void main(argc,argv)
int argc;
char *argv[];
{
  ZVEC *x = (ZVEC *)((void *)0);
  ZVEC *y = (ZVEC *)((void *)0);
  ZVEC *z = (ZVEC *)((void *)0);
  ZVEC *u = (ZVEC *)((void *)0);
  ZVEC *diag = (ZVEC *)((void *)0);
  PERM *pi1 = (PERM *)((void *)0);
  PERM *pi2 = (PERM *)((void *)0);
  PERM *pivot = (PERM *)((void *)0);
  ZMAT *A = (ZMAT *)((void *)0);
  ZMAT *B = (ZMAT *)((void *)0);
  ZMAT *C = (ZMAT *)((void *)0);
  ZMAT *D = (ZMAT *)((void *)0);
  ZMAT *Q = (ZMAT *)((void *)0);
  complex ONE;
  complex z1;
  complex z2;
  complex z3;
  double cond_est;
  double s1;
  double s2;
  double s3;
  int i;
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
/* print out version information */
  m_version();
  printf("# Meschach Complex numbers & vectors torture test\n\n");
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
  printf("\n");
  mem_stat_mark(1);
  printf("# Testing %s...\n","complex arithmetic & special functions");
  ;
  ONE = zmake(1.0,0.0);
  printf("# ONE = ");
  z_foutput(stdout,ONE);
  z1 . re = mrand();
  z1 . im = mrand();
  z2 . re = mrand();
  z2 . im = mrand();
  z3 = zadd(z1,z2);
  if (fabs(z1 . re + z2 . re - z3 . re) + fabs(z1 . im + z2 . im - z3 . im) > 10 * ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","zadd",150);
  z3 = zsub(z1,z2);
  if (fabs(z1 . re - z2 . re - z3 . re) + fabs(z1 . im - z2 . im - z3 . im) > 10 * ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","zadd",153);
  z3 = zmlt(z1,z2);
  if (fabs(z1 . re * z2 . re - z1 . im * z2 . im - z3 . re) + fabs(z1 . im * z2 . re + z1 . re * z2 . im - z3 . im) > 10 * ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","zmlt",157);
  s1 = zabs(z1);
  if (fabs(s1 * s1 - (z1 . re * z1 . re + z1 . im * z1 . im)) > 10 * ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","zabs",160);
  if (zabs((zsub(z1,(zmlt(z2,(zdiv(z1,z2))))))) > 10 * ((double )2.22044604925031308084726333618164062e-16L) || zabs((zsub(ONE,(zdiv(z1,(zmlt(z2,(zdiv(z1,z2))))))))) > 10 * ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","zdiv",163);
  z3 = zsqrt(z1);
  if (zabs((zsub(z1,(zmlt(z3,z3))))) > 10 * ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","zsqrt",167);
  if (zabs((zsub(z1,(zlog((zexp(z1))))))) > 10 * ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","zexp/zlog",169);
  printf("# Check: MACHEPS = %g\n",(double )2.22044604925031308084726333618164062e-16L);
/* allocate, initialise, copy and resize operations */
/* ZVEC */
  printf("# Testing %s...\n","vector initialise, copy & resize");
  ;
  x = zv_get(12);
  y = zv_get(15);
  z = zv_get(12);
  zv_rand(x);
  zv_rand(y);
  z = _zv_copy(x,z,0);
  if (_zv_norm2((zv_sub(x,z,z)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZVEC copy",183);
  _zv_copy(x,y,0);
  x = zv_resize(x,10);
  y = zv_resize(y,10);
  if (_zv_norm2((zv_sub(x,y,z)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZVEC copy/resize",188);
  x = zv_resize(x,15);
  y = zv_resize(y,15);
  if (_zv_norm2((zv_sub(x,y,z)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","VZEC resize",192);
/* ZMAT */
  printf("# Testing %s...\n","matrix initialise, copy & resize");
  ;
  A = zm_get(8,5);
  B = zm_get(3,9);
  C = zm_get(8,5);
  zm_rand(A);
  zm_rand(B);
  C = _zm_copy(A,C,0,0);
  if (zm_norm_inf((zm_sub(A,C,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZMAT copy",203);
  _zm_copy(A,B,0,0);
  A = zm_resize(A,3,5);
  B = zm_resize(B,3,5);
  if (zm_norm_inf((zm_sub(A,B,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZMAT copy/resize",208);
  A = zm_resize(A,10,10);
  B = zm_resize(B,10,10);
  if (zm_norm_inf((zm_sub(A,B,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZMAT resize",212);
  ;
/* PERM */
  printf("# Testing %s...\n","permutation initialise, inverting & permuting vectors");
  ;
  pi1 = px_get(15);
  pi2 = px_get(12);
  px_rand(pi1);
  zv_rand(x);
  px_zvec(pi1,x,z);
  y = zv_resize(y,(x -> dim));
  pxinv_zvec(pi1,z,y);
  if (_zv_norm2((zv_sub(x,y,z)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","PERMute vector",226);
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
          zv_add(((ZVEC *)((void *)0)),((ZVEC *)((void *)0)),(ZVEC *)((void *)0));
          printf("Error: %s error: line %d\n","tracecatch() failure",235);
          set_err_flag(_old_flag);
          memmove(restart,_save,sizeof(jmp_buf ));
        }
         else {
          set_err_flag(_old_flag);
          memmove(restart,_save,sizeof(jmp_buf ));
          printf("# tracecatch() caught error\n");
          ev_err("ztorture.c",8,235,"main",0);
        }
      }
      ;
      printf("Error: %s error: line %d\n","catch() failure",235);
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
      ev_err("ztorture.c",_err_num,236,"catch",0);
    }
  }
  ;
/* testing inner products and v_mltadd() etc */
  printf("# Testing %s...\n","inner products and linear combinations");
  ;
  u = zv_get((x -> dim));
  zv_rand(u);
  zv_rand(x);
  zv_resize(y,(x -> dim));
  zv_rand(y);
  zv_mltadd(y,x,(zneg((zdiv((_zin_prod(x,y,0,1)),(_zin_prod(x,x,0,1)))))),z);
  if (zabs((_zin_prod(x,z,0,1))) >= 5 * ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) {
    printf("Error: %s error: line %d\n","zv_mltadd()/zin_prod()",248);
    printf("# error norm = %g\n",(zabs((_zin_prod(x,z,0,1)))));
  }
  z1 = zneg((zdiv((_zin_prod(x,y,0,1)),(zmake(_zv_norm2(x,(VEC *)((void *)0)) * _zv_norm2(x,(VEC *)((void *)0)),0.0)))));
  zv_mlt(z1,x,u);
  zv_add(y,u,u);
  if (_zv_norm2((zv_sub(u,z,u)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) {
    printf("Error: %s error: line %d\n","zv_mlt()/zv_norm2()",257);
    printf("# error norm = %g\n",(_zv_norm2(u,(VEC *)((void *)0))));
  }
#ifdef ANSI_C
  zv_linlist(u,x,z1,y,ONE,(VEC *)((void *)0));
  if (_zv_norm2((zv_sub(u,z,u)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf("Error: %s error: line %d\n","zv_linlist()",264);
#endif
#ifdef VARARGS
  zv_linlist(u,x,z1,y,ONE,(VEC *)((void *)0));
  if (_zv_norm2((zv_sub(u,z,u)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim)) 
    printf("Error: %s error: line %d\n","zv_linlist()",269);
#endif
  ;
/* vector norms */
  printf("# Testing %s...\n","vector norms");
  ;
  x = zv_resize(x,12);
  zv_rand(x);
  for (i = 0; ((unsigned int )i) <= x -> dim - 1; i += 1) {
    if (zabs(x -> ve[i]) >= 0.7) 
      x -> ve[i] = ONE;
     else 
      x -> ve[i] = zneg(ONE);
  }
  s1 = _zv_norm1(x,(VEC *)((void *)0));
  s2 = _zv_norm2(x,(VEC *)((void *)0));
  s3 = _zv_norm_inf(x,(VEC *)((void *)0));
  if (fabs(s1 - (x -> dim)) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim) || fabs(s2 - sqrt((double )(x -> dim))) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim) || fabs(s3 - 1.0) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","zv_norm1/2/_inf()",289);
/* test matrix multiply etc */
  printf("# Testing %s...\n","matrix multiply and invert");
  ;
  A = zm_resize(A,10,10);
  B = zm_resize(B,10,10);
  zm_rand(A);
  zm_inverse(A,B);
  zm_mlt(A,B,C);
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = zsub(C -> me[i][i],ONE);
  }
  if (zm_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm_inf(A) * zm_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","zm_inverse()/zm_mlt()",301);
  ;
/* ... and adjoints */
  printf("# Testing %s...\n","adjoints and adjoint-multiplies");
  ;
  zm_adjoint(A,A);
/* can do square matrices in situ */
  zmam_mlt(A,B,C);
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = zsub(C -> me[i][i],ONE);
  }
  if (zm_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm_inf(A) * zm_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","zm_adjoint()/zmam_mlt()",312);
  zm_adjoint(A,A);
  zm_adjoint(B,B);
  zmma_mlt(A,B,C);
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = zsub(C -> me[i][i],ONE);
  }
  if (zm_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm_inf(A) * zm_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","zm_adjoint()/zmma_mlt()",319);
  zsm_mlt((zmake(3.71,2.753)),B,B);
  zmma_mlt(A,B,C);
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = zsub(C -> me[i][i],(zmake(3.71,- 2.753)));
  }
  if (zm_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm_inf(A) * zm_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","szm_mlt()/zmma_mlt()",325);
  zm_adjoint(B,B);
  zsm_mlt((zdiv(ONE,(zmake(3.71,- 2.753)))),B,B);
  ;
/* ... and matrix-vector multiplies */
  printf("# Testing %s...\n","matrix-vector multiplies");
  ;
  x = zv_resize(x,(A -> n));
  y = zv_resize(y,(A -> m));
  z = zv_resize(z,(A -> m));
  u = zv_resize(u,(A -> n));
  zv_rand(x);
  zv_rand(y);
  zmv_mlt(A,x,z);
  z1 = _zin_prod(y,z,0,1);
  zvm_mlt(A,y,u);
  z2 = _zin_prod(u,x,0,1);
  if (zabs((zsub(z1,z2))) >= ((double )2.22044604925031308084726333618164062e-16L) * (x -> dim) * (x -> dim)) {
    printf("Error: %s error: line %d\n","zmv_mlt()/zvm_mlt()",345);
    printf("# difference between inner products is %g\n",(zabs((zsub(z1,z2)))));
  }
  zmv_mlt(B,z,u);
  if (_zv_norm2((zv_sub(u,x,u)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm_inf(A) * zm_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","zmv_mlt()/zvm_mlt()",351);
  ;
/* get/set row/col */
  printf("# Testing %s...\n","getting and setting rows and cols");
  ;
  x = zv_resize(x,(A -> n));
  y = zv_resize(y,(B -> m));
  x = zget_row(A,3,x);
  y = zget_col(B,3,y);
  if (zabs((zsub((_zin_prod(x,y,0,0)),ONE))) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm_inf(A) * zm_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","zget_row()/zget_col()",363);
  zv_mlt((zmake(- 1.0,0.0)),x,x);
  zv_mlt((zmake(- 1.0,0.0)),y,y);
  zset_row(A,3,x);
  zset_col(B,3,y);
  zm_mlt(A,B,C);
  for (i = 0; ((unsigned int )i) <= C -> m - 1; i += 1) {
    C -> me[i][i] = zsub(C -> me[i][i],ONE);
  }
  if (zm_norm_inf(C) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm_inf(A) * zm_norm_inf(B) * 5) 
    printf("Error: %s error: line %d\n","zset_row()/zset_col()",372);
  ;
/* matrix norms */
  printf("# Testing %s...\n","matrix norms");
  ;
  A = zm_resize(A,11,15);
  zm_rand(A);
  s1 = zm_norm_inf(A);
  B = zm_adjoint(A,B);
  s2 = zm_norm1(B);
  if (fabs(s1 - s2) >= ((double )2.22044604925031308084726333618164062e-16L) * (A -> m)) 
    printf("Error: %s error: line %d\n","zm_norm1()/zm_norm_inf()",384);
  C = zmam_mlt(A,A,C);
  z1 . re = z1 . im = 0.0;
  for (i = 0; i < C -> m && i < C -> n; i++) 
    z1 = zadd(z1,C -> me[i][i]);
  if (fabs(sqrt(z1 . re) - zm_norm_frob(A)) >= ((double )2.22044604925031308084726333618164062e-16L) * (A -> m) * (A -> n)) 
    printf("Error: %s error: line %d\n","zm_norm_frob",390);
  ;
/* permuting rows and columns */
/******************************
    notice("permuting rows & cols");
    A = zm_resize(A,11,15);
    B = zm_resize(B,11,15);
    pi1 = px_resize(pi1,A->m);
    px_rand(pi1);
    x = zv_resize(x,A->n);
    y = zmv_mlt(A,x,y);
    px_rows(pi1,A,B);
    px_zvec(pi1,y,z);
    zmv_mlt(B,x,u);
    if ( zv_norm2(zv_sub(z,u,u)) >= MACHEPS*A->m )
	errmesg("px_rows()");
    pi1 = px_resize(pi1,A->n);
    px_rand(pi1);
    px_cols(pi1,A,B);
    pxinv_zvec(pi1,x,z);
    zmv_mlt(B,z,u);
    if ( zv_norm2(zv_sub(y,u,u)) >= MACHEPS*A->n )
	errmesg("px_cols()");
    ******************************/
  ;
/* MATLAB save/load */
  printf("# Testing %s...\n","MATLAB save/load");
  ;
  A = zm_resize(A,12,11);
  if ((fp = fopen("asx5213a.mat","w")) == ((FILE *)((void *)0))) 
    printf("Cannot perform MATLAB save/load test\n");
   else {
    zm_rand(A);
    zm_save(fp,A,name);
    fclose(fp);
    if ((fp = fopen("asx5213a.mat","r")) == ((FILE *)((void *)0))) 
      printf("Cannot open save file \"%s\"\n","asx5213a.mat");
     else {
      (zm_free(B) , B = ((ZMAT *)((void *)0)));
      B = zm_load(fp,&cp);
      if (strcmp(name,cp) || zm_norm1((zm_sub(A,B,C))) >= ((double )2.22044604925031308084726333618164062e-16L) * (A -> m)) {
        printf("Error: %s error: line %d\n","zm_load()/zm_save()",438);
        printf("# orig. name = %s, restored name = %s\n",name,cp);
        printf("# orig. A =\n");
        zm_foutput(stdout,A);
        printf("# restored A =\n");
        zm_foutput(stdout,B);
      }
    }
  }
  ;
/* Now, onto matrix factorisations */
  A = zm_resize(A,10,10);
  B = zm_resize(B,(A -> m),(A -> n));
  _zm_copy(A,B,0,0);
  x = zv_resize(x,(A -> n));
  y = zv_resize(y,(A -> m));
  z = zv_resize(z,(A -> n));
  u = zv_resize(u,(A -> m));
  zv_rand(x);
  zmv_mlt(B,x,y);
  z = _zv_copy(x,z,0);
  printf("# Testing %s...\n","LU factor/solve");
  ;
  pivot = px_get((A -> m));
  zLUfactor(A,pivot);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      zLUsolve(A,pivot,y,x);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("ztorture.c",_err_num,463,"main",0);
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
      cond_est = zLUcondest(A,pivot);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("ztorture.c",_err_num,464,"main",0);
    }
  }
  ;
  printf("# cond(A) approx= %g\n",cond_est);
  if (_zv_norm2((zv_sub(x,z,u)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * _zv_norm2(x,(VEC *)((void *)0)) * cond_est) {
    printf("Error: %s error: line %d\n","zLUfactor()/zLUsolve()",468);
    printf("# LU solution error = %g [cf MACHEPS = %g]\n",(_zv_norm2((zv_sub(x,z,u)),(VEC *)((void *)0))),(double )2.22044604925031308084726333618164062e-16L);
  }
  _zv_copy(y,x,0);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      zLUsolve(A,pivot,x,x);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("ztorture.c",_err_num,475,"main",0);
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
      cond_est = zLUcondest(A,pivot);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("ztorture.c",_err_num,476,"main",0);
    }
  }
  ;
  if (_zv_norm2((zv_sub(x,z,u)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * _zv_norm2(x,(VEC *)((void *)0)) * cond_est) {
    printf("Error: %s error: line %d\n","zLUfactor()/zLUsolve()",479);
    printf("# LU solution error = %g [cf MACHEPS = %g]\n",(_zv_norm2((zv_sub(x,z,u)),(VEC *)((void *)0))),(double )2.22044604925031308084726333618164062e-16L);
  }
  zvm_mlt(B,z,y);
  _zv_copy(y,x,0);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      zLUAsolve(A,pivot,x,x);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("ztorture.c",_err_num,486,"main",0);
    }
  }
  ;
  if (_zv_norm2((zv_sub(x,z,u)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * _zv_norm2(x,(VEC *)((void *)0)) * cond_est) {
    printf("Error: %s error: line %d\n","zLUfactor()/zLUAsolve()",489);
    printf("# LU solution error = %g [cf MACHEPS = %g]\n",(_zv_norm2((zv_sub(x,z,u)),(VEC *)((void *)0))),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* QR factorisation */
  _zm_copy(B,A,0,0);
  zmv_mlt(B,z,y);
  printf("# Testing %s...\n","QR factor/solve:");
  ;
  diag = zv_get((A -> m));
  zQRfactor(A,diag);
  zQRsolve(A,diag,y,x);
  if (_zv_norm2((zv_sub(x,z,u)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * _zv_norm2(x,(VEC *)((void *)0)) * cond_est) {
    printf("Error: %s error: line %d\n","zQRfactor()/zQRsolve()",505);
    printf("# QR solution error = %g [cf MACHEPS = %g]\n",(_zv_norm2((zv_sub(x,z,u)),(VEC *)((void *)0))),(double )2.22044604925031308084726333618164062e-16L);
  }
  printf("# QR cond(A) approx= %g\n",(zQRcondest(A)));
  Q = zm_get((A -> m),(A -> m));
  zmakeQ(A,diag,Q);
  zmakeR(A,A);
  zm_mlt(Q,A,C);
  zm_sub(B,C,C);
  if (zm_norm1(C) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm1(Q) * zm_norm1(B)) {
    printf("Error: %s error: line %d\n","zQRfactor()/zmakeQ()/zmakeR()",517);
    printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",(zm_norm1(C)),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* now try with a non-square matrix */
  A = zm_resize(A,15,7);
  zm_rand(A);
  B = _zm_copy(A,B,0,0);
  diag = zv_resize(diag,(A -> n));
  x = zv_resize(x,(A -> n));
  y = zv_resize(y,(A -> m));
  zv_rand(y);
  zQRfactor(A,diag);
  x = zQRsolve(A,diag,y,x);
/* z is the residual vector */
  zmv_mlt(B,x,z);
  zv_sub(z,y,z);
/* check B*.z = 0 */
  zvm_mlt(B,z,u);
  if (_zv_norm2(u,(VEC *)((void *)0)) >= 100 * ((double )2.22044604925031308084726333618164062e-16L) * zm_norm1(B) * _zv_norm2(y,(VEC *)((void *)0))) {
    printf("Error: %s error: line %d\n","zQRfactor()/zQRsolve()",540);
    printf("# QR solution error = %g [cf MACHEPS = %g]\n",(_zv_norm2(u,(VEC *)((void *)0))),(double )2.22044604925031308084726333618164062e-16L);
  }
  Q = zm_resize(Q,(A -> m),(A -> m));
  zmakeQ(A,diag,Q);
  zmakeR(A,A);
  zm_mlt(Q,A,C);
  zm_sub(B,C,C);
  if (zm_norm1(C) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm1(Q) * zm_norm1(B)) {
    printf("Error: %s error: line %d\n","zQRfactor()/zmakeQ()/zmakeR()",551);
    printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",(zm_norm1(C)),(double )2.22044604925031308084726333618164062e-16L);
  }
  D = zm_get((A -> m),(Q -> m));
  zmam_mlt(Q,Q,D);
  for (i = 0; ((unsigned int )i) <= D -> m - 1; i += 1) {
    D -> me[i][i] = zsub(D -> me[i][i],ONE);
  }
  if (zm_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm1(Q) * zm_norm_inf(Q)) {
    printf("Error: %s error: line %d\n","QRfactor()/makeQ()/makeR()",561);
    printf("# QR orthogonality error = %g [cf MACHEPS = %g]\n",(zm_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* QRCP factorisation */
  _zm_copy(B,A,0,0);
  printf("# Testing %s...\n","QR factor/solve with column pivoting");
  ;
  pivot = px_resize(pivot,(A -> n));
  zQRCPfactor(A,diag,pivot);
  z = zv_resize(z,(A -> n));
  zQRCPsolve(A,diag,pivot,y,z);
/* pxinv_zvec(pivot,z,x); */
/* now compute residual (z) vector */
  zmv_mlt(B,x,z);
  zv_sub(z,y,z);
/* check B^T.z = 0 */
  zvm_mlt(B,z,u);
  if (_zv_norm2(u,(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm1(B) * _zv_norm2(y,(VEC *)((void *)0))) {
    printf("Error: %s error: line %d\n","QRCPfactor()/QRsolve()",582);
    printf("# QR solution error = %g [cf MACHEPS = %g]\n",(_zv_norm2(u,(VEC *)((void *)0))),(double )2.22044604925031308084726333618164062e-16L);
  }
  Q = zm_resize(Q,(A -> m),(A -> m));
  zmakeQ(A,diag,Q);
  zmakeR(A,A);
  zm_mlt(Q,A,C);
  (zm_free(D) , D = ((ZMAT *)((void *)0)));
  D = zm_get((B -> m),(B -> n));
/******************************
    px_cols(pivot,C,D);
    zm_sub(B,D,D);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(Q)*zm_norm1(B) )
    {
	errmesg("QRCPfactor()/makeQ()/makeR()");
	printf("# QR reconstruction error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(D), MACHEPS);
    }
    ******************************/
/* Now check eigenvalue/SVD routines */
  printf("# Testing %s...\n","complex Schur routines");
  ;
  A = zm_resize(A,11,11);
  B = zm_resize(B,(A -> m),(A -> n));
  C = zm_resize(C,(A -> m),(A -> n));
  D = zm_resize(D,(A -> m),(A -> n));
  Q = zm_resize(Q,(A -> m),(A -> n));
  ;
/* now test complex Schur decomposition */
/* zm_copy(A,B); */
  (zm_free(A) , A = ((ZMAT *)((void *)0)));
  A = zm_get(11,11);
  zm_rand(A);
  B = _zm_copy(A,B,0,0);
  ;
  B = zschur(B,Q);
  printf("At line %d in file \"%s\"\n",623,"ztorture.c");
  zm_mlt(Q,B,C);
  zmma_mlt(C,Q,D);
  ;
  zm_sub(A,D,D);
  if (zm_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm1(Q) * zm_norm_inf(Q) * zm_norm1(B) * 5) {
    printf("Error: %s error: line %d\n","zschur()",631);
    printf("# Schur reconstruction error = %g [cf MACHEPS = %g]\n",(zm_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
/* orthogonality check */
  zmma_mlt(Q,Q,D);
  for (i = 0; ((unsigned int )i) <= D -> m - 1; i += 1) {
    D -> me[i][i] = zsub(D -> me[i][i],ONE);
  }
  if (zm_norm1(D) >= ((double )2.22044604925031308084726333618164062e-16L) * zm_norm1(Q) * zm_norm_inf(Q) * 10) {
    printf("Error: %s error: line %d\n","zschur()",642);
    printf("# Schur orthogonality error = %g [cf MACHEPS = %g]\n",(zm_norm1(D)),(double )2.22044604925031308084726333618164062e-16L);
  }
  ;
/* now test SVD */
/******************************
    A = zm_resize(A,11,7);
    zm_rand(A);
    U = zm_get(A->n,A->n);
    Q = zm_resize(Q,A->m,A->m);
    u = zv_resize(u,max(A->m,A->n));
    svd(A,Q,U,u);
    ******************************/
/* check reconstruction of A */
/******************************
    D = zm_resize(D,A->m,A->n);
    C = zm_resize(C,A->m,A->n);
    zm_zero(D);
    for ( i = 0; i < min(A->m,A->n); i++ )
	zm_set_val(D,i,i,v_entry(u,i));
    zmam_mlt(Q,D,C);
    zm_mlt(C,U,D);
    zm_sub(A,D,D);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(U)*zm_norm_inf(Q)*zm_norm1(A) )
    {
	errmesg("svd()");
	printf("# SVD reconstruction error = %g [cf MACHEPS = %g]\n",
	       zm_norm1(D), MACHEPS);
    }
    ******************************/
/* check orthogonality of Q and U */
/******************************
    D = zm_resize(D,Q->n,Q->n);
    zmam_mlt(Q,Q,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(Q)*zm_norm_inf(Q)*5 )
    {
	errmesg("svd()");
	printf("# SVD orthognality error (Q) = %g [cf MACHEPS = %g\n",
	       zm_norm1(D), MACHEPS);
    }
    D = zm_resize(D,U->n,U->n);
    zmam_mlt(U,U,D);
    for ( i = 0; i < D->m; i++ )
	m_set_val(D,i,i,m_entry(D,i,i)-1.0);
    if ( zm_norm1(D) >= MACHEPS*zm_norm1(U)*zm_norm_inf(U)*5 )
    {
	errmesg("svd()");
	printf("# SVD orthognality error (U) = %g [cf MACHEPS = %g\n",
	       zm_norm1(D), MACHEPS);
    }
    for ( i = 0; i < u->dim; i++ )
	if ( v_entry(u,i) < 0 || (i < u->dim-1 &&
				  v_entry(u,i+1) > v_entry(u,i)) )
	    break;
    if ( i < u->dim )
    {
	errmesg("svd()");
	printf("# SVD sorting error\n");
    }
    ******************************/
  (zv_free(x) , x = ((ZVEC *)((void *)0)));
  (zv_free(y) , y = ((ZVEC *)((void *)0)));
  (zv_free(z) , z = ((ZVEC *)((void *)0)));
  (zv_free(u) , u = ((ZVEC *)((void *)0)));
  (zv_free(diag) , diag = ((ZVEC *)((void *)0)));
  (px_free(pi1) , pi1 = ((PERM *)((void *)0)));
  (px_free(pi2) , pi2 = ((PERM *)((void *)0)));
  (px_free(pivot) , pivot = ((PERM *)((void *)0)));
  (zm_free(A) , A = ((ZMAT *)((void *)0)));
  (zm_free(B) , B = ((ZMAT *)((void *)0)));
  (zm_free(C) , C = ((ZMAT *)((void *)0)));
  (zm_free(D) , D = ((ZMAT *)((void *)0)));
  (zm_free(Q) , Q = ((ZMAT *)((void *)0)));
  mem_stat_free_list(1,0);
  ;
  printf("# Finished torture test for complex numbers/vectors/matrices\n");
  mem_info_file(stdout,0);
}
