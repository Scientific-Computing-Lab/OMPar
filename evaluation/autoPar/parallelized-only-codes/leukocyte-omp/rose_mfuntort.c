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
/* mfuntort.c,  10/11/93 */
#include <omp.h> 
static char rcsid[] = "$Id: mfuntort.c,v 1.2 1994/01/14 01:08:06 des Exp $";
#include        <stdio.h>
#include        <math.h>
#include        "matrix.h"
#include        "matrix2.h"
#define errmesg(mesg)   printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)    printf("# Testing %s...\n",mesg);
#define DIM  10

void main()
{
  MAT *A;
  MAT *B;
  MAT *C;
  MAT *OUTA;
  MAT *OUTB;
  MAT *TMP;
  MAT *exp_A_expected;
  MAT *exp_A;
  VEC *x;
  VEC *b;
  double c;
  double eps = 1e-10;
  int i;
  int j;
  int q_out;
  int j_out;
  mem_info_on(1);
  A = m_get(10,10);
  B = m_get(10,10);
  C = m_get(10,10);
  OUTA = m_get(10,10);
  OUTB = m_get(10,10);
  TMP = m_get(10,10);
  x = v_get(10);
  b = v_get(6);
  printf("# Testing %s...\n","exponent of a matrix");
  ;
  m_ident(A);
  mem_stat_mark(1);
  _m_exp(A,eps,OUTA,&q_out,&j_out);
  printf("# q_out = %d, j_out = %d\n",q_out,j_out);
  m_exp(A,eps,OUTA);
  sm_mlt((exp(1.0)),A,A);
  m_sub(OUTA,A,TMP);
  printf("# ||exp(I) - e*I|| = %g\n",(m_norm_inf(TMP)));
  m_rand(A);
  m_transp(A,TMP);
  m_add(A,TMP,A);
  B = _m_copy(A,B,0,0);
  m_exp(A,eps,OUTA);
  symmeig(B,OUTB,x);
  m_zero(TMP);
  for (i = 0; ((unsigned int )i) <= x -> dim - 1; i += 1) {
    TMP -> me[i][i] = exp(x -> ve[i]);
  }
  m_mlt(OUTB,TMP,C);
  mmtr_mlt(C,OUTB,TMP);
  m_sub(TMP,OUTA,TMP);
  printf("# ||exp(A) - Q*exp(lambda)*Q^T|| = %g\n",(m_norm_inf(TMP)));
  printf("# Testing %s...\n","polynomial of a matrix");
  ;
  m_rand(A);
  m_transp(A,TMP);
  m_add(A,TMP,A);
  B = _m_copy(A,B,0,0);
  v_rand(b);
  m_poly(A,b,OUTA);
  symmeig(B,OUTB,x);
  m_zero(TMP);
  
#pragma omp parallel for private (c,i,j)
  for (i = 0; ((unsigned int )i) <= x -> dim - 1; i += 1) {
    c = b -> ve[b -> dim - 1];
    for (j = (b -> dim - 2); j >= 0; j += -1) {
      c = c * x -> ve[i] + b -> ve[j];
    }
    TMP -> me[i][i] = c;
  }
  m_mlt(OUTB,TMP,C);
  mmtr_mlt(C,OUTB,TMP);
  m_sub(TMP,OUTA,TMP);
  printf("# ||poly(A) - Q*poly(lambda)*Q^T|| = %g\n",(m_norm_inf(TMP)));
  mem_stat_free_list(1,0);
/* Brook Milligan's test */
  (m_free(A) , A = ((MAT *)((void *)0)));
  (m_free(B) , B = ((MAT *)((void *)0)));
  (m_free(C) , C = ((MAT *)((void *)0)));
  printf("# Testing %s...\n","exponent of a nonsymmetric matrix");
  ;
  A = m_get(2,2);
  A -> me[0][0] = 1.0;
  A -> me[0][1] = 1.0;
  A -> me[1][0] = 4.0;
  A -> me[1][1] = 1.0;
  exp_A_expected = m_get(2,2);
  exp_A_expected -> me[0][0] = exp(3.0) / 2.0 + exp(- 1.0) / 2.0;
  exp_A_expected -> me[0][1] = exp(3.0) / 4.0 - exp(- 1.0) / 4.0;
  exp_A_expected -> me[1][0] = exp(3.0) - exp(- 1.0);
  exp_A_expected -> me[1][1] = exp(3.0) / 2.0 + exp(- 1.0) / 2.0;
  printf("A:\n");
  for (i = 0; i <= 1; i += 1) {
    for (j = 0; j <= 1; j += 1) {
      printf("   %15.8e",A -> me[i][j]);
    }
    printf("\n");
  }
  printf("\nexp(A) (expected):\n");
  for (i = 0; i <= 1; i += 1) {
    for (j = 0; j <= 1; j += 1) {
      printf("   %15.8e",exp_A_expected -> me[i][j]);
    }
    printf("\n");
  }
  mem_stat_mark(3);
  exp_A = m_exp(A,1e-16,((void *)0));
  mem_stat_free_list(3,0);
  printf("\nexp(A):\n");
  for (i = 0; i <= 1; i += 1) {
    for (j = 0; j <= 1; j += 1) {
      printf("   %15.8e",exp_A -> me[i][j]);
    }
    printf("\n");
  }
  printf("\nexp(A) - exp(A) (expected):\n");
  for (i = 0; i <= 1; i += 1) {
    for (j = 0; j <= 1; j += 1) {
      printf("   %15.8e",exp_A -> me[i][j] - exp_A_expected -> me[i][j]);
    }
    printf("\n");
  }
  (m_free(A) , A = ((MAT *)((void *)0)));
  (m_free(B) , B = ((MAT *)((void *)0)));
  (m_free(C) , C = ((MAT *)((void *)0)));
  (m_free(exp_A) , exp_A = ((MAT *)((void *)0)));
  (m_free(exp_A_expected) , exp_A_expected = ((MAT *)((void *)0)));
  (m_free(OUTA) , OUTA = ((MAT *)((void *)0)));
  (m_free(OUTB) , OUTB = ((MAT *)((void *)0)));
  (m_free(TMP) , TMP = ((MAT *)((void *)0)));
  (v_free(b) , b = ((VEC *)((void *)0)));
  (v_free(x) , x = ((VEC *)((void *)0)));
  mem_info_file(stdout,0);
}
