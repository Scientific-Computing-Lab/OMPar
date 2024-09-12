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
  This file contains the routines needed to perform QR factorisation
  of matrices, as well as Householder transformations.
  The internal "factored form" of a matrix A is not quite standard.
  The diagonal of A is replaced by the diagonal of R -- not by the 1st non-zero
  entries of the Householder vectors. The 1st non-zero entries are held in
  the diag parameter of QRfactor(). The reason for this non-standard
  representation is that it enables direct use of the Usolve() function
  rather than requiring that  a seperate function be written just for this case.
  See, e.g., QRsolve() below for more details.
  Complex version
  
*/
#include <omp.h> 
static char rcsid[] = "$Id: zqrfctr.c,v 1.1 1994/01/13 04:21:22 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
#include	"zmatrix2.h" 
#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)
#define		sign(x)	((x) > 0.0 ? 1 : ((x) < 0.0 ? -1 : 0 ))
/* Note: The usual representation of a Householder transformation is taken
   to be:
   P = I - beta.u.u*
   where beta = 2/(u*.u) and u is called the Householder vector
   (u* is the conjugate transposed vector of u
*/
/* zQRfactor -- forms the QR factorisation of A
	-- factorisation stored in compact form as described above
	(not quite standard format) */

ZMAT *zQRfactor(
//A,diag)
ZMAT *A,ZVEC *diag)
{
  unsigned int k;
  unsigned int limit;
  double beta;
  static ZVEC *tmp1 = (ZVEC *)((void *)0);
  static ZVEC *w = (ZVEC *)((void *)0);
  if (!A || !diag) 
    ev_err("zqrfctr.c",8,73,"zQRfactor",0);
  limit = (A -> m > A -> n?A -> n : A -> m);
  if (diag -> dim < limit) 
    ev_err("zqrfctr.c",1,76,"zQRfactor",0);
  tmp1 = zv_resize(tmp1,(A -> m));
  w = zv_resize(w,(A -> n));
  mem_stat_reg_list((void **)(&tmp1),8,0,"zqrfctr.c",80);
  mem_stat_reg_list((void **)(&w),8,0,"zqrfctr.c",81);
  for (k = 0; k <= limit - 1; k += 1) {
/* get H/holder vector for the k-th column */
    zget_col(A,k,tmp1);
    zhhvec(tmp1,k,&beta,tmp1,&A -> me[k][k]);
    diag -> ve[k] = tmp1 -> ve[k];
/* apply H/holder vector to remaining columns */
{
      jmp_buf _save;
      int _err_num;
      int _old_flag;
      _old_flag = set_err_flag(2);
      memmove(_save,restart,sizeof(jmp_buf ));
      if ((_err_num = _setjmp(restart)) == 0) {
        _zhhtrcols(A,k,(k + 1),tmp1,beta,w);
        set_err_flag(_old_flag);
        memmove(restart,_save,sizeof(jmp_buf ));
      }
       else {
        set_err_flag(_old_flag);
        memmove(restart,_save,sizeof(jmp_buf ));
        ev_err("zqrfctr.c",_err_num,91,"zQRfactor",0);
      }
    }
    ;
  }
#ifdef	THREADSAFE
#endif
  return A;
}
/* zQRCPfactor -- forms the QR factorisation of A with column pivoting
   -- factorisation stored in compact form as described above
   ( not quite standard format )				*/

ZMAT *zQRCPfactor(
//A,diag,px)
ZMAT *A,ZVEC *diag,PERM *px)
{
  unsigned int i;
  unsigned int i_max;
  unsigned int j;
  unsigned int k;
  unsigned int limit;
  static ZVEC *tmp1 = (ZVEC *)((void *)0);
  static ZVEC *tmp2 = (ZVEC *)((void *)0);
  static ZVEC *w = (ZVEC *)((void *)0);
  static VEC *gamma = (VEC *)((void *)0);
  double beta;
  double maxgamma;
  double sum;
  double tmp;
  complex ztmp;
  if (!A || !diag || !px) 
    ev_err("zqrfctr.c",8,117,"QRCPfactor",0);
  limit = (A -> m > A -> n?A -> n : A -> m);
  if (diag -> dim < limit || px -> size != A -> n) 
    ev_err("zqrfctr.c",1,120,"QRCPfactor",0);
  tmp1 = zv_resize(tmp1,(A -> m));
  tmp2 = zv_resize(tmp2,(A -> m));
  gamma = v_resize(gamma,(A -> n));
  w = zv_resize(w,(A -> n));
  mem_stat_reg_list((void **)(&tmp1),8,0,"zqrfctr.c",126);
  mem_stat_reg_list((void **)(&tmp2),8,0,"zqrfctr.c",127);
  mem_stat_reg_list((void **)(&gamma),3,0,"zqrfctr.c",128);
  mem_stat_reg_list((void **)(&w),8,0,"zqrfctr.c",129);
/* initialise gamma and px */
  for (j = 0; j <= A -> n - 1; j += 1) {
    px -> pe[j] = j;
    sum = 0.0;
    for (i = 0; i <= A -> m - 1; i += 1) {
      sum += square(A -> me[i][j] . re) + square(A -> me[i][j] . im);
    }
    gamma -> ve[j] = sum;
  }
  for (k = 0; k <= limit - 1; k += 1) {
/* find "best" column to use */
    i_max = k;
    maxgamma = gamma -> ve[k];
    for (i = k + 1; i <= A -> n - 1; i += 1) {
/* Loop invariant:maxgamma=gamma[i_max]
	       >=gamma[l];l=k,...,i-1 */
      if (gamma -> ve[i] > maxgamma) {
        maxgamma = gamma -> ve[i];
        i_max = i;
      }
    }
/* swap columns if necessary */
    if (i_max != k) {
/* swap gamma values */
      tmp = gamma -> ve[k];
      gamma -> ve[k] = gamma -> ve[i_max];
      gamma -> ve[i_max] = tmp;
/* update column permutation */
      px_transp(px,k,i_max);
/* swap columns of A */
      for (i = 0; i <= A -> m - 1; i += 1) {
        ztmp = A -> me[i][k];
        A -> me[i][k] = A -> me[i][i_max];
        A -> me[i][i_max] = ztmp;
      }
    }
/* get H/holder vector for the k-th column */
    zget_col(A,k,tmp1);
/* hhvec(tmp1,k,&beta->ve[k],tmp1,&A->me[k][k]); */
    zhhvec(tmp1,k,&beta,tmp1,&A -> me[k][k]);
    diag -> ve[k] = tmp1 -> ve[k];
/* apply H/holder vector to remaining columns */
    _zhhtrcols(A,k,(k + 1),tmp1,beta,w);
/* update gamma values */
    for (j = k + 1; j <= A -> n - 1; j += 1) {
      gamma -> ve[j] -= square(A -> me[k][j] . re) + square(A -> me[k][j] . im);
    }
  }
#ifdef	THREADSAFE
#endif
  return A;
}
/* zQsolve -- solves Qx = b, Q is an orthogonal matrix stored in compact
	form a la QRfactor()
	-- may be in-situ */

//QR,diag,b,x,tmp)
ZVEC *_zQsolve(ZMAT *QR,ZVEC *diag,ZVEC *b,ZVEC *x,ZVEC *tmp)
{
  unsigned int dynamic;
  int k;
  int limit;
  double beta;
  double r_ii;
  double tmp_val;
  limit = ((QR -> m > QR -> n?QR -> n : QR -> m));
  dynamic = 0;
  if (!QR || !diag || !b) 
    ev_err("zqrfctr.c",8,208,"_zQsolve",0);
  if (diag -> dim < limit || b -> dim != QR -> m) 
    ev_err("zqrfctr.c",1,210,"_zQsolve",0);
  x = zv_resize(x,(QR -> m));
  if (tmp == ((ZVEC *)((void *)0))) 
    dynamic = 1;
  tmp = zv_resize(tmp,(QR -> m));
/* apply H/holder transforms in normal order */
  x = _zv_copy(b,x,0);
  for (k = 0; k <= limit - 1; k += 1) {
    zget_col(QR,k,tmp);
    r_ii = zabs(tmp -> ve[k]);
    tmp -> ve[k] = diag -> ve[k];
    tmp_val = r_ii * zabs(diag -> ve[k]);
    beta = (tmp_val == 0.0?0.0 : 1.0 / tmp_val);
/* hhtrvec(tmp,beta->ve[k],k,x,x); */
    zhhtrvec(tmp,beta,k,x,x);
  }
  if (dynamic) 
    (zv_free(tmp) , tmp = ((ZVEC *)((void *)0)));
  return x;
}
/* zmakeQ -- constructs orthogonal matrix from Householder vectors stored in
   compact QR form */

ZMAT *zmakeQ(
//QR,diag,Qout)
ZMAT *QR,ZVEC *diag,ZMAT *Qout)
{
  static ZVEC *tmp1 = (ZVEC *)((void *)0);
  static ZVEC *tmp2 = (ZVEC *)((void *)0);
  unsigned int i;
  unsigned int limit;
  double beta;
  double r_ii;
  double tmp_val;
  int j;
  limit = (QR -> m > QR -> n?QR -> n : QR -> m);
  if (!QR || !diag) 
    ev_err("zqrfctr.c",8,249,"zmakeQ",0);
  if (diag -> dim < limit) 
    ev_err("zqrfctr.c",1,251,"zmakeQ",0);
  Qout = zm_resize(Qout,(QR -> m),(QR -> m));
  tmp1 = zv_resize(tmp1,(QR -> m));
/* contains basis vec & columns of Q */
  tmp2 = zv_resize(tmp2,(QR -> m));
/* contains H/holder vectors */
  mem_stat_reg_list((void **)(&tmp1),8,0,"zqrfctr.c",256);
  mem_stat_reg_list((void **)(&tmp2),8,0,"zqrfctr.c",257);
  for (i = 0; i <= QR -> m - 1; i += 1) {
/* get i-th column of Q */
/* set up tmp1 as i-th basis vector */
    
#pragma omp parallel for private (j)
    for (j = 0; ((unsigned int )j) <= QR -> m - 1; j += 1) {
      tmp1 -> ve[j] . re = tmp1 -> ve[j] . im = 0.0;
    }
    tmp1 -> ve[i] . re = 1.0;
/* apply H/h transforms in reverse order */
    for (j = (limit - 1); j >= 0; j += -1) {
      zget_col(QR,j,tmp2);
      r_ii = zabs(tmp2 -> ve[j]);
      tmp2 -> ve[j] = diag -> ve[j];
      tmp_val = r_ii * zabs(diag -> ve[j]);
      beta = (tmp_val == 0.0?0.0 : 1.0 / tmp_val);
/* hhtrvec(tmp2,beta->ve[j],j,tmp1,tmp1); */
      zhhtrvec(tmp2,beta,j,tmp1,tmp1);
    }
/* insert into Q */
    zset_col(Qout,i,tmp1);
  }
#ifdef	THREADSAFE
#endif
  return Qout;
}
/* zmakeR -- constructs upper triangular matrix from QR (compact form)
	-- may be in-situ (all it does is zero the lower 1/2) */

ZMAT *zmakeR(
//QR,Rout)
ZMAT *QR,ZMAT *Rout)
{
  unsigned int i;
  unsigned int j;
  if (QR == ((ZMAT *)((void *)0))) 
    ev_err("zqrfctr.c",8,298,"zmakeR",0);
  Rout = _zm_copy(QR,Rout,0,0);
  
#pragma omp parallel for private (i)
  for (i = 1; i <= QR -> m - 1; i += 1) {
    for (j = 0; j < QR -> n && j < i; j++) 
      Rout -> me[i][j] . re = Rout -> me[i][j] . im = 0.0;
  }
  return Rout;
}
/* zQRsolve -- solves the system Q.R.x=b where Q & R are stored in compact form
   -- returns x, which is created if necessary */

ZVEC *zQRsolve(
//QR,diag,b,x)
ZMAT *QR,ZVEC *diag,ZVEC *b,ZVEC *x)
{
  int limit;
  static ZVEC *tmp = (ZVEC *)((void *)0);
  if (!QR || !diag || !b) 
    ev_err("zqrfctr.c",8,320,"zQRsolve",0);
  limit = ((QR -> m > QR -> n?QR -> n : QR -> m));
  if (diag -> dim < limit || b -> dim != QR -> m) 
    ev_err("zqrfctr.c",1,323,"zQRsolve",0);
  tmp = zv_resize(tmp,limit);
  mem_stat_reg_list((void **)(&tmp),8,0,"zqrfctr.c",325);
  x = zv_resize(x,(QR -> n));
  _zQsolve(QR,diag,b,x,tmp);
  x = zUsolve(QR,x,x,0.0);
  x = zv_resize(x,(QR -> n));
#ifdef	THREADSAFE
#endif
  return x;
}
/* zQRAsolve -- solves the system (Q.R)*.x = b
	-- Q & R are stored in compact form
	-- returns x, which is created if necessary */

ZVEC *zQRAsolve(
//QR,diag,b,x)
ZMAT *QR,ZVEC *diag,ZVEC *b,ZVEC *x)
{
  int j;
  int limit;
  double beta;
  double r_ii;
  double tmp_val;
  static ZVEC *tmp = (ZVEC *)((void *)0);
  if (!QR || !diag || !b) 
    ev_err("zqrfctr.c",8,353,"zQRAsolve",0);
  limit = ((QR -> m > QR -> n?QR -> n : QR -> m));
  if (diag -> dim < limit || b -> dim != QR -> n) 
    ev_err("zqrfctr.c",1,356,"zQRAsolve",0);
  x = zv_resize(x,(QR -> m));
  x = zUAsolve(QR,b,x,0.0);
  x = zv_resize(x,(QR -> m));
  tmp = zv_resize(tmp,(x -> dim));
  mem_stat_reg_list((void **)(&tmp),8,0,"zqrfctr.c",363);
/*  printf("zQRAsolve: tmp->dim = %d, x->dim = %d\n", tmp->dim, x->dim); */
/* apply H/h transforms in reverse order */
  for (j = limit - 1; j >= 0; j += -1) {
    zget_col(QR,j,tmp);
    tmp = zv_resize(tmp,(QR -> m));
    r_ii = zabs(tmp -> ve[j]);
    tmp -> ve[j] = diag -> ve[j];
    tmp_val = r_ii * zabs(diag -> ve[j]);
    beta = (tmp_val == 0.0?0.0 : 1.0 / tmp_val);
    zhhtrvec(tmp,beta,j,x,x);
  }
#ifdef	THREADSAFE
#endif
  return x;
}
/* zQRCPsolve -- solves A.x = b where A is factored by QRCPfactor()
   -- assumes that A is in the compact factored form */

ZVEC *zQRCPsolve(
//QR,diag,pivot,b,x)
ZMAT *QR,ZVEC *diag,PERM *pivot,ZVEC *b,ZVEC *x)
{
  if (!QR || !diag || !pivot || !b) 
    ev_err("zqrfctr.c",8,394,"zQRCPsolve",0);
  if (QR -> m > diag -> dim && QR -> n > diag -> dim || QR -> n != pivot -> size) 
    ev_err("zqrfctr.c",1,396,"zQRCPsolve",0);
  x = zQRsolve(QR,diag,b,x);
  x = pxinv_zvec(pivot,x,x);
  return x;
}
/* zUmlt -- compute out = upper_triang(U).x
	-- may be in situ */

ZVEC *zUmlt(
//U,x,out)
ZMAT *U,ZVEC *x,ZVEC *out)
{
  int i;
  int limit;
  if (U == ((ZMAT *)((void *)0)) || x == ((ZVEC *)((void *)0))) 
    ev_err("zqrfctr.c",8,414,"zUmlt",0);
  limit = ((U -> m > U -> n?U -> n : U -> m));
  if (limit != x -> dim) 
    ev_err("zqrfctr.c",1,417,"zUmlt",0);
  if (out == ((ZVEC *)((void *)0)) || out -> dim < limit) 
    out = zv_resize(out,limit);
  for (i = 0; i <= limit - 1; i += 1) {
    out -> ve[i] = __zip__((&x -> ve[i]),(&U -> me[i][i]),limit - i,0);
  }
  return out;
}
/* zUAmlt -- returns out = upper_triang(U)^T.x */

//U,x,out)
ZVEC *zUAmlt(ZMAT *U,ZVEC *x,ZVEC *out)
{
/* complex	sum; */
  complex tmp;
  int i;
  int limit;
  if (U == ((ZMAT *)((void *)0)) || x == ((ZVEC *)((void *)0))) 
    ev_err("zqrfctr.c",8,436,"zUAmlt",0);
  limit = ((U -> m > U -> n?U -> n : U -> m));
  if (out == ((ZVEC *)((void *)0)) || out -> dim < limit) 
    out = zv_resize(out,limit);
  for (i = limit - 1; i >= 0; i += -1) {
    tmp = x -> ve[i];
    out -> ve[i] . re = out -> ve[i] . im = 0.0;
    __zmltadd__(&out -> ve[i],(&U -> me[i][i]),tmp,limit - i - 1,1);
  }
  return out;
}
/* zQRcondest -- returns an estimate of the 2-norm condition number of the
		matrix factorised by QRfactor() or QRCPfactor()
	-- note that as Q does not affect the 2-norm condition number,
		it is not necessary to pass the diag, beta (or pivot) vectors
	-- generates a lower bound on the true condition number
	-- if the matrix is exactly singular, HUGE_VAL is returned
	-- note that QRcondest() is likely to be more reliable for
		matrices factored using QRCPfactor() */

double zQRcondest(
//QR)
ZMAT *QR)
{
  static ZVEC *y = (ZVEC *)((void *)0);
  double norm;
  double norm1;
  double norm2;
  double tmp1;
  double tmp2;
  complex sum;
  complex tmp;
  int i;
  int j;
  int limit;
  if (QR == ((ZMAT *)((void *)0))) 
    ev_err("zqrfctr.c",8,469,"zQRcondest",0);
  limit = ((QR -> m > QR -> n?QR -> n : QR -> m));
  
#pragma omp parallel for private (i)
  for (i = 0; i <= limit - 1; i += 1) {
/* if ( QR->me[i][i] == 0.0 ) */
    if (QR -> me[i][i] . re == 0.0 && QR -> me[i][i] . im == 0.0) 
      return __builtin_huge_val();
  }
  y = zv_resize(y,limit);
  mem_stat_reg_list((void **)(&y),8,0,"zqrfctr.c",478);
/* use the trick for getting a unit vector y with ||R.y||_inf small
       from the LU condition estimator */
  for (i = 0; i <= limit - 1; i += 1) {
    sum . re = sum . im = 0.0;
    for (j = 0; j <= i - 1; j += 1) {
/* sum -= QR->me[j][i]*y->ve[j]; */
      sum = zsub(sum,(zmlt(QR -> me[j][i],y -> ve[j])));
    }
/* sum -= (sum < 0.0) ? 1.0 : -1.0; */
    norm1 = zabs(sum);
    if (norm1 == 0.0) 
      sum . re = 1.0;
     else {
      sum . re += sum . re / norm1;
      sum . im += sum . im / norm1;
    }
/* y->ve[i] = sum / QR->me[i][i]; */
    y -> ve[i] = zdiv(sum,QR -> me[i][i]);
  }
  zUAmlt(QR,y,y);
/* now apply inverse power method to R*.R */
  for (i = 0; i <= 2; i += 1) {
    tmp1 = _zv_norm2(y,(VEC *)((void *)0));
    zv_mlt((zmake(1.0 / tmp1,0.0)),y,y);
    zUAsolve(QR,y,y,0.0);
    tmp2 = _zv_norm2(y,(VEC *)((void *)0));
    zv_mlt((zmake(1.0 / tmp2,0.0)),y,y);
    zUsolve(QR,y,y,0.0);
  }
/* now compute approximation for ||R^{-1}||_2 */
  norm1 = sqrt(tmp1) * sqrt(tmp2);
/* now use complementary approach to compute approximation to ||R||_2 */
  for (i = limit - 1; i >= 0; i += -1) {
    sum . re = sum . im = 0.0;
    for (j = i + 1; j <= limit - 1; j += 1) {
      sum = zadd(sum,(zmlt(QR -> me[i][j],y -> ve[j])));
    }
    if (QR -> me[i][i] . re == 0.0 && QR -> me[i][i] . im == 0.0) 
      return __builtin_huge_val();
    tmp = zdiv(sum,QR -> me[i][i]);
    if (tmp . re == 0.0 && tmp . im == 0.0) {
      y -> ve[i] . re = 1.0;
      y -> ve[i] . im = 0.0;
    }
     else {
      norm = zabs(tmp);
      y -> ve[i] . re = sum . re / norm;
      y -> ve[i] . im = sum . im / norm;
    }
/* y->ve[i] = (sum >= 0.0) ? 1.0 : -1.0; */
/* y->ve[i] = (QR->me[i][i] >= 0.0) ? y->ve[i] : - y->ve[i]; */
  }
/* now apply power method to R*.R */
  for (i = 0; i <= 2; i += 1) {
    tmp1 = _zv_norm2(y,(VEC *)((void *)0));
    zv_mlt((zmake(1.0 / tmp1,0.0)),y,y);
    zUmlt(QR,y,y);
    tmp2 = _zv_norm2(y,(VEC *)((void *)0));
    zv_mlt((zmake(1.0 / tmp2,0.0)),y,y);
    zUAmlt(QR,y,y);
  }
  norm2 = sqrt(tmp1) * sqrt(tmp2);
/* printf("QRcondest: norm1 = %g, norm2 = %g\n",norm1,norm2); */
#ifdef	THREADSAFE
#endif
  return norm1 * norm2;
}
