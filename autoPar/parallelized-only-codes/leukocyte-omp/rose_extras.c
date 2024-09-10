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
	Memory port routines: MEM_COPY and MEM_ZERO
*/
/* For BSD 4.[23] environments: using bcopy() and bzero() */
#include "machine.h"
#ifndef MEM_COPY
#endif
#ifndef MEM_ZERO
#endif
/*
	This file contains versions of something approximating the well-known
	BLAS routines in C, suitable for Meschach (hence the `m').
	These are "vanilla" implementations, at least with some consideration
	of the effects of caching and paging, and maybe some loop unrolling
	for register-rich machines
*/
/*
	Organisation of matrices: it is assumed that matrices are represented
	by Real **'s. To keep flexibility, there is also an "initial
	column" parameter j0, so that the actual elements used are
		A[0][j0],   A[0][j0+1],   ..., A[0][j0+n-1]
		A[1][j0],   A[1][j0+1],   ..., A[1][j0+n-1]
		   ..         ..          ...      ..
		A[m-1][j0], A[m-1][j0+1], ..., A[m-1][j0+n-1]
*/
#include <omp.h> 
static char rcsid[] = "$Id: extras.c,v 1.4 1995/06/08 15:13:15 des Exp $";
#include	<math.h>
#define	REGISTER_RICH	1
/* mblar-1 routines */
/* Mscale -- sets x <- alpha.x */

void Mscale(len,alpha,x)
int len;
double alpha;
double *x;
{
  register int i;
  
#pragma omp parallel for private (i) firstprivate (len,alpha)
  for (i = 0; i <= len - 1; i += 1) {
    x[i] *= alpha;
  }
}
/* Mswap -- swaps x and y */

void Mswap(len,x,y)
int len;
double *x;
double *y;
{
  register int i;
  register double tmp;
  
#pragma omp parallel for private (tmp,i) firstprivate (len)
  for (i = 0; i <= len - 1; i += 1) {
    tmp = x[i];
    x[i] = y[i];
    y[i] = tmp;
  }
}
/* Mcopy -- copies x to y */

void Mcopy(len,x,y)
int len;
double *x;
double *y;
{
  register int i;
  
#pragma omp parallel for private (i) firstprivate (len)
  for (i = 0; i <= len - 1; i += 1) {
    y[i] = x[i];
  }
}
/* Maxpy -- y <- y + alpha.x */

void Maxpy(len,alpha,x,y)
int len;
double alpha;
double *x;
double *y;
{
  register int i;
  register int len4;
/****************************************
    for ( i = 0; i < len; i++ )
	y[i] += alpha*x[i];
    ****************************************/
#ifdef REGISTER_RICH
  len4 = len / 4;
  len = len % 4;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= len4 - 1; i += 1) {
    y[4 * i] += alpha * x[4 * i];
    y[4 * i + 1] += alpha * x[4 * i + 1];
    y[4 * i + 2] += alpha * x[4 * i + 2];
    y[4 * i + 3] += alpha * x[4 * i + 3];
  }
  x += 4 * len4;
  y += 4 * len4;
#endif
  
#pragma omp parallel for private (i) firstprivate (len,alpha)
  for (i = 0; i <= len - 1; i += 1) {
    y[i] += alpha * x[i];
  }
}
/* Mdot -- returns x'.y */

double Mdot(len,x,y)
int len;
double *x;
double *y;
{
  register int i;
  register int len4;
  register double sum;
#ifndef REGISTER_RICH
#endif
#ifdef REGISTER_RICH
  register double sum0;
  register double sum1;
  register double sum2;
  register double sum3;
  sum0 = sum1 = sum2 = sum3 = 0.0;
  len4 = len / 4;
  len = len % 4;
  
#pragma omp parallel for private (i) reduction (+:sum0,sum1,sum2,sum3)
  for (i = 0; i <= len4 - 1; i += 1) {
    sum0 += x[4 * i] * y[4 * i];
    sum1 += x[4 * i + 1] * y[4 * i + 1];
    sum2 += x[4 * i + 2] * y[4 * i + 2];
    sum3 += x[4 * i + 3] * y[4 * i + 3];
  }
  sum = sum0 + sum1 + sum2 + sum3;
  x += 4 * len4;
  y += 4 * len4;
#endif
  
#pragma omp parallel for private (i) reduction (+:sum) firstprivate (len)
  for (i = 0; i <= len - 1; i += 1) {
    sum += x[i] * y[i];
  }
  return sum;
}
#ifndef ABS
#define	ABS(x)	((x) >= 0 ? (x) : -(x))
#endif
/* Mnorminf -- returns ||x||_inf */

double Mnorminf(len,x)
int len;
double *x;
{
  register int i;
  register double tmp;
  register double max_val;
  max_val = 0.0;
  
#pragma omp parallel for private (tmp,i) reduction (max:max_val) firstprivate (len)
  for (i = 0; i <= len - 1; i += 1) {
    tmp = (x[i] >= 0?x[i] : -x[i]);
    if (max_val < tmp) 
      max_val = tmp;
  }
  return max_val;
}
/* Mnorm1 -- returns ||x||_1 */

double Mnorm1(len,x)
int len;
double *x;
{
  register int i;
  register double sum;
  sum = 0.0;
  
#pragma omp parallel for private (i) reduction (+:sum) firstprivate (len)
  for (i = 0; i <= len - 1; i += 1) {
    sum += (x[i] >= 0?x[i] : -x[i]);
  }
  return sum;
}
/* Mnorm2 -- returns ||x||_2 */

double Mnorm2(len,x)
int len;
double *x;
{
  register int i;
  register double norm;
  register double invnorm;
  register double sum;
  register double tmp;
  norm = Mnorminf(len,x);
  if (norm == 0.0) 
    return 0.0;
  invnorm = 1.0 / norm;
  sum = 0.0;
  
#pragma omp parallel for private (tmp,i) reduction (+:sum) firstprivate (len)
  for (i = 0; i <= len - 1; i += 1) {
    tmp = x[i] * invnorm;
    sum += tmp * tmp;
  }
  return sum / invnorm;
}
/* mblar-2 routines */
/* Mmv -- y <- alpha.A.x + beta.y */

void Mmv(m,n,alpha,A,j0,x,beta,y)
int m;
int n;
double alpha;
double **A;
int j0;
double *x;
double beta;
double *y;
{
  register int i;
  register int j;
  register int m4;
  register int n4;
  register double sum0;
  register double sum1;
  register double sum2;
  register double sum3;
  register double tmp0;
  register double tmp1;
  register double tmp2;
  register double tmp3;
  register double *dp0;
  register double *dp1;
  register double *dp2;
  register double *dp3;
/****************************************
    for ( i = 0; i < m; i++ )
	y[i] += alpha*Mdot(n,&(A[i][j0]),x);
    ****************************************/
  m4 = n4 = 0;
#ifdef REGISTER_RICH
  m4 = m / 4;
  m = m % 4;
  n4 = n / 4;
  n = n % 4;
  for (i = 0; i <= m4 - 1; i += 1) {
    sum0 = sum1 = sum2 = sum3 = 0.0;
    dp0 = &A[4 * i][j0];
    dp1 = &A[4 * i + 1][j0];
    dp2 = &A[4 * i + 2][j0];
    dp3 = &A[4 * i + 3][j0];
    for (j = 0; j <= n4 - 1; j += 1) {
      tmp0 = x[4 * j];
      tmp1 = x[4 * j + 1];
      tmp2 = x[4 * j + 2];
      tmp3 = x[4 * j + 3];
      sum0 = sum0 + dp0[j] * tmp0 + dp0[j + 1] * tmp1 + dp0[j + 2] * tmp2 + dp0[j + 3] * tmp3;
      sum1 = sum1 + dp1[j] * tmp0 + dp1[j + 1] * tmp1 + dp1[j + 2] * tmp2 + dp1[j + 3] * tmp3;
      sum2 = sum2 + dp2[j] * tmp0 + dp2[j + 1] * tmp1 + dp2[j + 2] * tmp2 + dp2[j + 3] * tmp3;
      sum3 = sum3 + dp3[j] * tmp0 + dp3[j + 1] * tmp2 + dp3[j + 2] * tmp2 + dp3[j + 3] * tmp3;
    }
    
#pragma omp parallel for private (j) reduction (+:sum0,sum1,sum2,sum3)
    for (j = 0; j <= n - 1; j += 1) {
      sum0 += dp0[4 * n4 + j] * x[4 * n4 + j];
      sum1 += dp1[4 * n4 + j] * x[4 * n4 + j];
      sum2 += dp2[4 * n4 + j] * x[4 * n4 + j];
      sum3 += dp3[4 * n4 + j] * x[4 * n4 + j];
    }
    y[4 * i] = beta * y[4 * i] + alpha * sum0;
    y[4 * i + 1] = beta * y[4 * i + 1] + alpha * sum1;
    y[4 * i + 2] = beta * y[4 * i + 2] + alpha * sum2;
    y[4 * i + 3] = beta * y[4 * i + 3] + alpha * sum3;
  }
#endif
  for (i = 0; i <= m - 1; i += 1) {
    y[4 * m4 + i] = beta * y[i] + alpha * Mdot(4 * n4 + n,&A[4 * m4 + i][j0],x);
  }
}
/* Mvm -- y <- alpha.A^T.x + beta.y */

void Mvm(m,n,alpha,A,j0,x,beta,y)
int m;
int n;
double alpha;
double **A;
int j0;
double *x;
double beta;
double *y;
{
  register int i;
  register int j;
  register int m4;
  register int n2;
  register double *Aref;
  register double tmp;
#ifdef REGISTER_RICH
  register double *Aref0;
  register double *Aref1;
  register double tmp0;
  register double tmp1;
  register double yval0;
  register double yval1;
  register double yval2;
  register double yval3;
#endif
  if (beta != 1.0) 
    Mscale(m,beta,y);
/****************************************
    for ( j = 0; j < n; j++ )
	Maxpy(m,alpha*x[j],&(A[j][j0]),y);
    ****************************************/
  m4 = n2 = 0;
  m4 = m / 4;
  m = m % 4;
#ifdef REGISTER_RICH
  n2 = n / 2;
  n = n % 2;
  for (j = 0; j <= n2 - 1; j += 1) {
    tmp0 = alpha * x[2 * j];
    tmp1 = alpha * x[2 * j + 1];
    Aref0 = &A[2 * j][j0];
    Aref1 = &A[2 * j + 1][j0];
    
#pragma omp parallel for private (yval0,yval1,yval2,yval3,i)
    for (i = 0; i <= m4 - 1; i += 1) {
      yval0 = y[4 * i] + tmp0 * Aref0[4 * i];
      yval1 = y[4 * i + 1] + tmp0 * Aref0[4 * i + 1];
      yval2 = y[4 * i + 2] + tmp0 * Aref0[4 * i + 2];
      yval3 = y[4 * i + 3] + tmp0 * Aref0[4 * i + 3];
      y[4 * i] = yval0 + tmp1 * Aref1[4 * i];
      y[4 * i + 1] = yval1 + tmp1 * Aref1[4 * i + 1];
      y[4 * i + 2] = yval2 + tmp1 * Aref1[4 * i + 2];
      y[4 * i + 3] = yval3 + tmp1 * Aref1[4 * i + 3];
    }
    y += 4 * m4;
    Aref0 += 4 * m4;
    Aref1 += 4 * m4;
    
#pragma omp parallel for private (i) firstprivate (tmp0,tmp1)
    for (i = 0; i <= m - 1; i += 1) {
      y[i] += tmp0 * Aref0[i] + tmp1 * Aref1[i];
    }
  }
#endif
  for (j = 0; j <= n - 1; j += 1) {
    tmp = alpha * x[2 * n2 + j];
    Aref = &A[2 * n2 + j][j0];
    
#pragma omp parallel for private (i)
    for (i = 0; i <= m4 - 1; i += 1) {
      y[4 * i] += tmp * Aref[4 * i];
      y[4 * i + 1] += tmp * Aref[4 * i + 1];
      y[4 * i + 2] += tmp * Aref[4 * i + 2];
      y[4 * i + 3] += tmp * Aref[4 * i + 3];
    }
    y += 4 * m4;
    Aref += 4 * m4;
    
#pragma omp parallel for private (i) firstprivate (tmp)
    for (i = 0; i <= m - 1; i += 1) {
      y[i] += tmp * Aref[i];
    }
  }
}
/* Mupdate -- A <- A + alpha.x.y^T */

void Mupdate(m,n,alpha,x,y,A,j0)
int m;
int n;
double alpha;
double *x;
double *y;
double **A;
int j0;
{
  register int i;
  register int j;
  register int n4;
  register double *Aref;
  register double tmp;
/****************************************
    for ( i = 0; i < m; i++ )
	Maxpy(n,alpha*x[i],y,&(A[i][j0]));
    ****************************************/
  n4 = n / 4;
  n = n % 4;
  for (i = 0; i <= m - 1; i += 1) {
    tmp = alpha * x[i];
    Aref = &A[i][j0];
    
#pragma omp parallel for private (j)
    for (j = 0; j <= n4 - 1; j += 1) {
      Aref[4 * j] += tmp * y[4 * j];
      Aref[4 * j + 1] += tmp * y[4 * j + 1];
      Aref[4 * j + 2] += tmp * y[4 * j + 2];
      Aref[4 * j + 3] += tmp * y[4 * j + 3];
    }
    Aref += 4 * n4;
    y += 4 * n4;
    
#pragma omp parallel for private (j) firstprivate (tmp)
    for (j = 0; j <= n - 1; j += 1) {
      Aref[j] += tmp * y[j];
    }
  }
}
/* mblar-3 routines */
/* Mmm -- C <- C + alpha.A.B */

void Mmm(m,n,p,alpha,A,Aj0,B,Bj0,C,Cj0)
int m;
int n;
int p;
double alpha;
double **A;
int Aj0;
double **B;
int Bj0;
double **C;
int Cj0;
/* C is m x n */
{
  register int i;
  register int j;
  register int k;
/* register Real	tmp, sum; */
/****************************************
    for ( i = 0; i < m; i++ )
	for ( k = 0; k < p; k++ )
	    Maxpy(n,alpha*A[i][Aj0+k],&(B[k][Bj0]),&(C[i][Cj0]));
    ****************************************/
  for (i = 0; i <= m - 1; i += 1) {
    Mvm(p,n,alpha,B,Bj0,&A[i][Aj0],1.0,&C[i][Cj0]);
  }
}
/* Mmtrm -- C <- C + alpha.A^T.B */

void Mmtrm(m,n,p,alpha,A,Aj0,B,Bj0,C,Cj0)
int m;
int n;
int p;
double alpha;
double **A;
int Aj0;
double **B;
int Bj0;
double **C;
int Cj0;
/* C is m x n */
{
  register int i;
  register int j;
  register int k;
/****************************************
    for ( i = 0; i < m; i++ )
	for ( k = 0; k < p; k++ )
	    Maxpy(n,alpha*A[k][Aj0+i],&(B[k][Bj0]),&(C[i][Cj0]));
    ****************************************/
  for (k = 0; k <= p - 1; k += 1) {
    Mupdate(m,n,alpha,&A[k][Aj0],&B[k][Bj0],C,Cj0);
  }
}
/* Mmmtr -- C <- C + alpha.A.B^T */

void Mmmtr(m,n,p,alpha,A,Aj0,B,Bj0,C,Cj0)
int m;
int n;
int p;
double alpha;
double **A;
int Aj0;
double **B;
int Bj0;
double **C;
int Cj0;
/* C is m x n */
{
  register int i;
  register int j;
  register int k;
/****************************************
    for ( i = 0; i < m; i++ )
	for ( j = 0; j < n; j++ )
	    C[i][Cj0+j] += alpha*Mdot(p,&(A[i][Aj0]),&(B[j][Bj0]));
    ****************************************/
  for (i = 0; i <= m - 1; i += 1) {
    Mmv(n,p,alpha,B,Bj0,&A[i][Aj0],1.0,&C[i][Cj0]);
  }
}
/* Mmtrmtr -- C <- C + alpha.A^T.B^T */

void Mmtrmtr(m,n,p,alpha,A,Aj0,B,Bj0,C,Cj0)
int m;
int n;
int p;
double alpha;
double **A;
int Aj0;
double **B;
int Bj0;
double **C;
int Cj0;
/* C is m x n */
{
  register int i;
  register int j;
  register int k;
  for (i = 0; i <= m - 1; i += 1) {
    for (j = 0; j <= n - 1; j += 1) {
      for (k = 0; k <= p - 1; k += 1) {
        C[i][Cj0 + j] += A[i][Aj0 + k] * B[k][Bj0 + j];
      }
    }
  }
}
