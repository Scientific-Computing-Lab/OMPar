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
/* update.c 1.3 11/25/87 */
#include <omp.h> 
static char rcsid[] = "$Id: update.c,v 1.2 1994/01/13 05:26:06 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"
/* Most matrix factorisation routines are in-situ unless otherwise specified */
/* LDLupdate -- updates a CHolesky factorisation, replacing LDL' by
	MD~M' = LDL' + alpha.w.w' Note: w is overwritten
	Ref: Gill et al Math Comp 28, p516 Algorithm C1 */
#ifndef ANSI_C
#else

MAT *LDLupdate(MAT *CHmat,VEC *w,double alpha)
#endif
{
  unsigned int i;
  unsigned int j;
  double diag;
  double new_diag;
  double beta;
  double p;
  if (CHmat == ((MAT *)((void *)0)) || w == ((VEC *)((void *)0))) 
    ev_err("update.c",8,60,"LDLupdate",0);
  if (CHmat -> m != CHmat -> n || w -> dim != CHmat -> m) 
    ev_err("update.c",1,62,"LDLupdate",0);
  for (j = 0; j <= w -> dim - 1; j += 1) {
    p = w -> ve[j];
    diag = CHmat -> me[j][j];
    new_diag = CHmat -> me[j][j] = diag + alpha * p * p;
    if (new_diag <= 0.0) 
      ev_err("update.c",5,70,"LDLupdate",0);
    beta = p * alpha / new_diag;
    alpha *= diag / new_diag;
    for (i = j + 1; i <= w -> dim - 1; i += 1) {
      w -> ve[i] -= p * CHmat -> me[i][j];
      CHmat -> me[i][j] += beta * w -> ve[i];
      CHmat -> me[j][i] = CHmat -> me[i][j];
    }
  }
  return CHmat;
}
/* QRupdate -- updates QR factorisation in expanded form (seperate matrices)
	Finds Q+, R+ s.t. Q+.R+ = Q.(R+u.v') and Q+ orthogonal, R+ upper triang
	Ref: Golub & van Loan Matrix Computations pp437-443
	-- does not update Q if it is NULL */
#ifndef ANSI_C
#else

MAT *QRupdate(MAT *Q,MAT *R,VEC *u,VEC *v)
#endif
{
  int i;
  int j;
  int k;
  double c;
  double s;
  double temp;
  if (!R || !u || !v) 
    ev_err("update.c",8,102,"QRupdate",0);
  if (Q && (Q -> m != Q -> n || R -> m != Q -> n) || u -> dim != R -> m || v -> dim != R -> n) 
    ev_err("update.c",1,105,"QRupdate",0);
/* find largest k s.t. u[k] != 0 */
  for (k = (R -> m - 1); k >= 0; k += -1) {
    if (u -> ve[k] != 0.0) 
      break; 
  }
/* transform R+u.v' to Hessenberg form */
  for (i = k - 1; i >= 0; i += -1) {
/* get Givens rotation */
    givens(u -> ve[i],u -> ve[i + 1],&c,&s);
    rot_rows(R,i,(i + 1),c,s,R);
    if (Q) 
      rot_cols(Q,i,(i + 1),c,s,Q);
    rot_vec(u,i,(i + 1),c,s,u);
  }
/* add into R */
  temp = u -> ve[0];
  
#pragma omp parallel for private (j) firstprivate (temp)
  for (j = 0; ((unsigned int )j) <= R -> n - 1; j += 1) {
    R -> me[0][j] += temp * v -> ve[j];
  }
/* transform Hessenberg to upper triangular */
  for (i = 0; i <= k - 1; i += 1) {
    givens(R -> me[i][i],R -> me[i + 1][i],&c,&s);
    rot_rows(R,i,(i + 1),c,s,R);
    if (Q) 
      rot_cols(Q,i,(i + 1),c,s,Q);
  }
  return R;
}
