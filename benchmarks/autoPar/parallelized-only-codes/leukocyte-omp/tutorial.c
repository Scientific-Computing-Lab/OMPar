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
/* tutorial.c 10/12/1993 */
/* routines from Chapter 1 of Meschach */
static char rcsid[] = "$Id: tutorial.c,v 1.3 1994/01/16 22:53:09 des Exp $";
#include <math.h>
#include "matrix.h"
/* rk4 -- 4th order Runge--Kutta method */

double rk4(f,t,x,h)
VEC *(*f)();
double t;
VEC *x;
double h;
{
  static VEC *v1 = (VEC *)((void *)0);
  static VEC *v2 = (VEC *)((void *)0);
  static VEC *v3 = (VEC *)((void *)0);
  static VEC *v4 = (VEC *)((void *)0);
  static VEC *temp = (VEC *)((void *)0);
/* do not work with NULL initial vector */
  if (x == ((VEC *)((void *)0))) 
    ev_err("tutorial.c",8,45,"rk4",0);
/* ensure that v1, ..., v4, temp are of the correct size */
  v1 = v_resize(v1,(x -> dim));
  v2 = v_resize(v2,(x -> dim));
  v3 = v_resize(v3,(x -> dim));
  v4 = v_resize(v4,(x -> dim));
  temp = v_resize(temp,(x -> dim));
/* register workspace variables */
  mem_stat_reg_list((void **)(&v1),3,0,"tutorial.c",55);
  mem_stat_reg_list((void **)(&v2),3,0,"tutorial.c",56);
  mem_stat_reg_list((void **)(&v3),3,0,"tutorial.c",57);
  mem_stat_reg_list((void **)(&v4),3,0,"tutorial.c",58);
  mem_stat_reg_list((void **)(&temp),3,0,"tutorial.c",59);
/* end of memory allocation */
  ( *f)(t,x,v1);
/* most compilers allow: "f(t,x,v1);" */
  v_mltadd(x,v1,0.5 * h,temp);
/* temp = x+.5*h*v1 */
  ( *f)((t + 0.5 * h),temp,v2);
  v_mltadd(x,v2,0.5 * h,temp);
/* temp = x+.5*h*v2 */
  ( *f)((t + 0.5 * h),temp,v3);
  v_mltadd(x,v3,h,temp);
/* temp = x+h*v3 */
  ( *f)((t + h),temp,v4);
/* now add: v1+2*v2+2*v3+v4 */
  _v_copy(v1,temp,0);
/* temp = v1 */
  v_mltadd(temp,v2,2.0,temp);
/* temp = v1+2*v2 */
  v_mltadd(temp,v3,2.0,temp);
/* temp = v1+2*v2+2*v3 */
  v_add(temp,v4,temp);
/* temp = v1+2*v2+2*v3+v4 */
/* adjust x */
  v_mltadd(x,temp,h / 6.0,x);
/* x = x+(h/6)*temp */
  return t + h;
/* return the new time */
}
/* rk4 -- 4th order Runge-Kutta method */
/* another variant */

double rk4_var(f,t,x,h)
VEC *(*f)();
double t;
VEC *x;
double h;
{
  static VEC *v1;
  static VEC *v2;
  static VEC *v3;
  static VEC *v4;
  static VEC *temp;
/* do not work with NULL initial vector */
  if (x == ((VEC *)((void *)0))) 
    ev_err("tutorial.c",8,93,"rk4",0);
/* ensure that v1, ..., v4, temp are of the correct size */
  v_resize_vars((x -> dim),&v1,&v2,&v3,&v4,&temp,(void *)0);
/* register workspace variables */
  mem_stat_reg_vars(0,3,"tutorial.c",99,&v1,&v2,&v3,&v4,&temp,(void *)0);
/* end of memory allocation */
  ( *f)(t,x,v1);
  v_mltadd(x,v1,0.5 * h,temp);
  ( *f)((t + 0.5 * h),temp,v2);
  v_mltadd(x,v2,0.5 * h,temp);
  ( *f)((t + 0.5 * h),temp,v3);
  v_mltadd(x,v3,h,temp);
  ( *f)((t + h),temp,v4);
/* now add: temp = v1+2*v2+2*v3+v4 */
  v_linlist(temp,v1,1.0,v2,2.0,v3,2.0,v4,1.0,(VEC *)((void *)0));
/* adjust x */
  v_mltadd(x,temp,h / 6.0,x);
/* x = x+(h/6)*temp */
  return t + h;
/* return the new time */
}
/* f -- right-hand side of ODE solver */

VEC *f(t,x,out)
double t;
VEC *x;
VEC *out;
{
  if (x == ((VEC *)((void *)0)) || out == ((VEC *)((void *)0))) 
    ev_err("tutorial.c",8,123,"f",0);
  if (x -> dim != 2 || out -> dim != 2) 
    ev_err("tutorial.c",1,125,"f",0);
  out -> ve[0] = x -> ve[1];
  out -> ve[1] = -x -> ve[0];
  return out;
}

void tutor_rk4()
{
  VEC *x;
  VEC *f();
  double h;
  double t;
  double t_fin;
  double rk4();
  ((isatty((fileno(stdin)))?fprintf(stderr,"Input initial time: ") : skipjunk(stdin)) , fscanf(stdin,"%lf",&t));
  ((isatty((fileno(stdin)))?fprintf(stderr,"Input final time: ") : skipjunk(stdin)) , fscanf(stdin,"%lf",&t_fin));
  x = v_get(2);
/* this is the size needed by f() */
  isatty((fileno(stdin)))?fprintf(stderr,"Input initial state:\n") : skipjunk(stdin);
  x = v_finput(stdin,(VEC *)((void *)0));
  ((isatty((fileno(stdin)))?fprintf(stderr,"Input step size: ") : skipjunk(stdin)) , fscanf(stdin,"%lf",&h));
  printf("# At time %g, the state is\n",t);
  v_foutput(stdout,x);
  while(t < t_fin){
/* you can use t = rk4_var(f,t,x,min(h,t_fin-t)); */
    t = rk4(f,t,x,(min(h,t_fin - t)));
/* new t is returned */
    printf("# At time %g, the state is\n",t);
    v_foutput(stdout,x);
  }
}
#include "matrix2.h"

void tutor_ls()
{
  MAT *A;
  MAT *QR;
  VEC *b;
  VEC *x;
  VEC *diag;
/* read in A matrix */
  printf("Input A matrix:\n");
  A = m_finput(stdin,(MAT *)((void *)0));
/* A has whatever size is input */
  if (A -> m < A -> n) {
    printf("Need m >= n to obtain least squares fit\n");
    exit(0);
  }
  printf("# A =\n");
  m_foutput(stdout,A);
  diag = v_get((A -> m));
/* QR is to be the QR factorisation of A */
  QR = _m_copy(A,(MAT *)((void *)0),0,0);
  QRfactor(QR,diag);
/* read in b vector */
  printf("Input b vector:\n");
  b = v_get((A -> m));
  b = v_finput(stdin,b);
  printf("# b =\n");
  v_foutput(stdout,b);
/* solve for x */
  x = QRsolve(QR,diag,b,(VEC *)((void *)0));
  printf("Vector of best fit parameters is\n");
  v_foutput(stdout,x);
/* ... and work out norm of errors... */
  printf("||A*x-b|| = %g\n",(_v_norm2((v_sub((mv_mlt(A,x,(VEC *)((void *)0))),b,(VEC *)((void *)0))),((VEC *)((void *)0)))));
}
#include "iter.h"
#define N 50
#define VEC2MAT(v,m)  vm_move((v),0,(m),0,0,N,N);
#define PI 3.141592653589793116
#define index(i,j) (N*((i)-1)+(j)-1)
/* right hand side function (for generating b) */

double f1(x,y)
double x;
double y;
{
/* return 2.0*PI*PI*sin(PI*x)*sin(PI*y); */
  return exp(x * y);
}
/* discrete laplacian */

SPMAT *laplacian(A)
SPMAT *A;
{
  double h;
  int i;
  int j;
  if (!A) 
    A = sp_get(50 * 50,50 * 50,5);
  for (i = 1; i <= 50; i += 1) {
    for (j = 1; j <= 50; j += 1) {
      if (i < 50) 
        sp_set_val(A,50 * (i - 1) + j - 1,50 * (i + 1 - 1) + j - 1,- 1.0);
      if (i > 1) 
        sp_set_val(A,50 * (i - 1) + j - 1,50 * (i - 1 - 1) + j - 1,- 1.0);
      if (j < 50) 
        sp_set_val(A,50 * (i - 1) + j - 1,50 * (i - 1) + (j + 1) - 1,- 1.0);
      if (j > 1) 
        sp_set_val(A,50 * (i - 1) + j - 1,50 * (i - 1) + (j - 1) - 1,- 1.0);
      sp_set_val(A,50 * (i - 1) + j - 1,50 * (i - 1) + j - 1,4.0);
    }
  }
  return A;
}
/* generating right hand side */

VEC *rhs_lap(b)
VEC *b;
{
  double h;
  double h2;
  double x;
  double y;
  int i;
  int j;
  if (!b) 
    b = v_get(50 * 50);
  h = 1.0 / (50 + 1);
/* for a unit square */
  h2 = h * h;
  x = 0.0;
  for (i = 1; i <= 50; i += 1) {
    x += h;
    y = 0.0;
    for (j = 1; j <= 50; j += 1) {
      y += h;
      b -> ve[50 * (i - 1) + j - 1] = h2 * f1(x,y);
    }
  }
  return b;
}

void tut_lap()
{
  SPMAT *A;
  SPMAT *LLT;
  VEC *b;
  VEC *out;
  VEC *x;
  MAT *B;
  int num_steps;
  FILE *fp;
  A = sp_get(50 * 50,50 * 50,5);
  b = v_get(50 * 50);
  laplacian(A);
  LLT = sp_copy(A);
  spICHfactor(LLT);
  out = v_get(A -> m);
  x = v_get(A -> m);
  rhs_lap(b);
/* new rhs */
  iter_spcg(A,LLT,b,1e-6,out,1000,&num_steps);
  printf("Number of iterations = %d\n",num_steps);
/* save b as a MATLAB matrix */
  fp = fopen("laplace.mat","w");
/* b will be saved in laplace.mat */
  if (fp == ((void *)0)) {
    printf("Cannot open %s\n","laplace.mat");
    exit(1);
  }
/* b must be transformed to a matrix */
  B = m_get(50,50);
  vm_move(out,0,B,0,0,50,50);
  ;
  m_save(fp,B,"sol");
/* sol is an internal name in MATLAB */
}

void main()
{
  int i;
  ((isatty((fileno(stdin)))?fprintf(stderr,"Choose the problem (1=Runge-Kutta, 2=least squares,3=laplace): ") : skipjunk(stdin)) , fscanf(stdin,"%d",&i));
  switch(i){
    case 1:
    tutor_rk4();
    break; 
    case 2:
    tutor_ls();
    break; 
    case 3:
    tut_lap();
    break; 
    default:
    printf(" Wrong value of i (only 1, 2 or 3)\n\n");
    break; 
  }
}
