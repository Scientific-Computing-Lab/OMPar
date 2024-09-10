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
	File containing routines for computing the Schur decomposition
	of a complex non-symmetric matrix
	See also: hessen.c
	Complex version
*/
#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
#include        "zmatrix2.h"
#include <omp.h> 
static char rcsid[] = "$Id: zschur.c,v 1.4 1995/04/07 16:28:58 des Exp $";
#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)
#define	b2s(t_or_f)	((t_or_f) ? "TRUE" : "FALSE")
/* zschur -- computes the Schur decomposition of the matrix A in situ
	-- optionally, gives Q matrix such that Q^*.A.Q is upper triangular
	-- returns upper triangular Schur matrix */

ZMAT *zschur(
//A,Q)
ZMAT *A,ZMAT *Q)
{
  int i;
  int j;
  int iter;
  int k;
  int k_min;
  int k_max;
  int k_tmp;
  int n;
  int split;
  double c;
  complex det;
  complex discrim;
  complex lambda;
  complex lambda0;
  complex lambda1;
  complex s;
  complex sum;
  complex ztmp;
  complex x;
  complex y;
/* for chasing algorithm */
  complex **A_me;
  static ZVEC *diag = (ZVEC *)((void *)0);
  if (!A) 
    ev_err("zschur.c",8,60,"zschur",0);
  if (A -> m != A -> n || Q && Q -> m != Q -> n) 
    ev_err("zschur.c",9,62,"zschur",0);
  if (Q != ((ZMAT *)((void *)0)) && Q -> m != A -> m) 
    ev_err("zschur.c",1,64,"zschur",0);
  n = (A -> n);
  diag = zv_resize(diag,(A -> n));
  mem_stat_reg_list((void **)(&diag),8,0,"zschur.c",67);
/* compute Hessenberg form */
  zHfactor(A,diag);
/* save Q if necessary, and make A explicitly Hessenberg */
  zHQunpack(A,diag,Q,A);
  k_min = 0;
  A_me = A -> me;
  while(k_min < n){
/* find k_max to suit:
	   submatrix k_min..k_max should be irreducible */
    k_max = n - 1;
    for (k = k_min; k <= k_max - 1; k += 1) {
      if (A_me[k + 1][k] . re == 0.0 && A_me[k + 1][k] . im == 0.0) {
        k_max = k;
        break; 
      }
    }
    if (k_max <= k_min) {
      k_min = k_max + 1;
      continue; 
/* outer loop */
    }
/* now have r x r block with r >= 2:
	   apply Francis QR step until block splits */
    split = 0;
    iter = 0;
    while(!split){
      complex a00;
      complex a01;
      complex a10;
      complex a11;
      iter++;
/* set up Wilkinson/Francis complex shift */
/* use the smallest eigenvalue of the bottom 2 x 2 submatrix */
      k_tmp = k_max - 1;
      a00 = A_me[k_tmp][k_tmp];
      a01 = A_me[k_tmp][k_max];
      a10 = A_me[k_max][k_tmp];
      a11 = A_me[k_max][k_max];
      ztmp . re = 0.5 * (a00 . re - a11 . re);
      ztmp . im = 0.5 * (a00 . im - a11 . im);
      discrim = zsqrt((zadd((zmlt(ztmp,ztmp)),(zmlt(a01,a10)))));
      sum . re = 0.5 * (a00 . re + a11 . re);
      sum . im = 0.5 * (a00 . im + a11 . im);
      lambda0 = zadd(sum,discrim);
      lambda1 = zsub(sum,discrim);
      det = zsub((zmlt(a00,a11)),(zmlt(a01,a10)));
      if (lambda0 . re == 0.0 && lambda0 . im == 0.0 && (lambda1 . re == 0.0 && lambda1 . im == 0.0)) {
        lambda . re = lambda . im = 0.0;
      }
       else if (zabs(lambda0) > zabs(lambda1)) 
        lambda = zdiv(det,lambda0);
       else 
        lambda = zdiv(det,lambda1);
/* perturb shift if convergence is slow */
      if (iter % 10 == 0) {
        lambda . re += iter * 0.02;
        lambda . im += iter * 0.02;
      }
/* set up Householder transformations */
      k_tmp = k_min + 1;
      x = zsub(A -> me[k_min][k_min],lambda);
      y = A -> me[k_min + 1][k_min];
/* use Givens' rotations to "chase" off-Hessenberg entry */
      for (k = k_min; k <= k_max - 1; k += 1) {
        zgivens(x,y,&c,&s);
        zrot_cols(A,k,k + 1,c,s,A);
        zrot_rows(A,k,k + 1,c,s,A);
        if (Q != ((ZMAT *)((void *)0))) 
          zrot_cols(Q,k,k + 1,c,s,Q);
/* zero things that should be zero */
        if (k > k_min) 
          A -> me[k + 1][k - 1] . re = A -> me[k + 1][k - 1] . im = 0.0;
/* get next entry to chase along sub-diagonal */
        x = A -> me[k + 1][k];
        if (k <= k_max - 2) 
          y = A -> me[k + 2][k];
         else 
          y . re = y . im = 0.0;
      }
      
#pragma omp parallel for private (k)
      for (k = k_min; k <= k_max - 2; k += 1) {
/* zero appropriate sub-diagonals */
        A -> me[k + 2][k] . re = A -> me[k + 2][k] . im = 0.0;
      }
/* test to see if matrix should split */
      for (k = k_min; k <= k_max - 1; k += 1) {
        if (zabs(A_me[k + 1][k]) < ((double )2.22044604925031308084726333618164062e-16L) * (zabs(A_me[k][k]) + zabs(A_me[k + 1][k + 1]))) {
          A_me[k + 1][k] . re = A_me[k + 1][k] . im = 0.0;
          split = 1;
        }
      }
    }
  }
/* polish up A by zeroing strictly lower triangular elements
       and small sub-diagonal elements */
  
#pragma omp parallel for private (i,j)
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= i - 1 - 1; j += 1) {
      A_me[i][j] . re = A_me[i][j] . im = 0.0;
    }
  }
  for (i = 0; ((unsigned int )i) <= A -> m - ((unsigned int )1) - 1; i += 1) {
    if (zabs(A_me[i + 1][i]) < ((double )2.22044604925031308084726333618164062e-16L) * (zabs(A_me[i][i]) + zabs(A_me[i + 1][i + 1]))) 
      A_me[i + 1][i] . re = A_me[i + 1][i] . im = 0.0;
  }
#ifdef	THREADSAFE
#endif
  return A;
}
#if 0
/* schur_vecs -- returns eigenvectors computed from the real Schur
		decomposition of a matrix
	-- T is the block upper triangular Schur matrix
	-- Q is the orthognal matrix where A = Q.T.Q^T
	-- if Q is null, the eigenvectors of T are returned
	-- X_re is the real part of the matrix of eigenvectors,
		and X_im is the imaginary part of the matrix.
	-- X_re is returned */
/* complex eigenvalue */
/* yes -- complex e-vals */
/* not correct Real Schur form */
/* solve (T-l.I)x = tmp1 */
/* printf("limit = %d\n",limit); */
/* 2 x 2 diagonal block */
/* printf("checkpoint A\n"); */
/* printf("checkpoint B\n"); */
/* printf("checkpoint C\n"); */
/* printf("checkpoint D\n"); */
/* printf("checkpoint E\n"); */
/* printf("checkpoint F\n"); */
/* printf("checkpoint G\n"); */
/* printf("checkpoint H\n"); */
/* zero vector */
#endif
