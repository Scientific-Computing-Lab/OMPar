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
  
*/
#include <omp.h> 
static char rcsid[] = "$Id: qrfactor.c,v 1.5 1994/01/13 05:35:07 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include        "matrix2.h"
#define		sign(x)	((x) > 0.0 ? 1 : ((x) < 0.0 ? -1 : 0 ))
extern VEC *Usolve();
/* See matrix2.h */
/* Note: The usual representation of a Householder transformation is taken
   to be:
   P = I - beta.u.uT
   where beta = 2/(uT.u) and u is called the Householder vector
   */
/* QRfactor -- forms the QR factorisation of A -- factorisation stored in
   compact form as described above ( not quite standard format ) */
#ifndef ANSI_C
#else

MAT *QRfactor(MAT *A,VEC *diag)
#endif
{
  unsigned int k;
  unsigned int limit;
  double beta;
  static VEC *hh = (VEC *)((void *)0);
  static VEC *w = (VEC *)((void *)0);
  if (!A || !diag) 
    ev_err("qrfactor.c",8,76,"QRfactor",0);
  limit = (A -> m > A -> n?A -> n : A -> m);
  if (diag -> dim < limit) 
    ev_err("qrfactor.c",1,79,"QRfactor",0);
  hh = v_resize(hh,(A -> m));
  w = v_resize(w,(A -> n));
  mem_stat_reg_list((void **)(&hh),3,0,"qrfactor.c",83);
  mem_stat_reg_list((void **)(&w),3,0,"qrfactor.c",84);
  for (k = 0; k <= limit - 1; k += 1) {
/* get H/holder vector for the k-th column */
    get_col(A,k,hh);
/* hhvec(hh,k,&beta->ve[k],hh,&A->me[k][k]); */
    hhvec(hh,k,&beta,hh,&A -> me[k][k]);
    diag -> ve[k] = hh -> ve[k];
/* apply H/holder vector to remaining columns */
/* hhtrcols(A,k,k+1,hh,beta->ve[k]); */
    _hhtrcols(A,k,k + 1,hh,beta,w);
  }
#ifdef	THREADSAFE
#endif
  return A;
}
/* QRCPfactor -- forms the QR factorisation of A with column pivoting
   -- factorisation stored in compact form as described above
   ( not quite standard format )				*/
#ifndef ANSI_C
#else

MAT *QRCPfactor(MAT *A,VEC *diag,PERM *px)
#endif
{
  unsigned int i;
  unsigned int i_max;
  unsigned int j;
  unsigned int k;
  unsigned int limit;
  static VEC *gamma = (VEC *)((void *)0);
  static VEC *tmp1 = (VEC *)((void *)0);
  static VEC *tmp2 = (VEC *)((void *)0);
  static VEC *w = (VEC *)((void *)0);
  double beta;
  double maxgamma;
  double sum;
  double tmp;
  if (!A || !diag || !px) 
    ev_err("qrfactor.c",8,123,"QRCPfactor",0);
  limit = (A -> m > A -> n?A -> n : A -> m);
  if (diag -> dim < limit || px -> size != A -> n) 
    ev_err("qrfactor.c",1,126,"QRCPfactor",0);
  tmp1 = v_resize(tmp1,(A -> m));
  tmp2 = v_resize(tmp2,(A -> m));
  gamma = v_resize(gamma,(A -> n));
  w = v_resize(w,(A -> n));
  mem_stat_reg_list((void **)(&tmp1),3,0,"qrfactor.c",132);
  mem_stat_reg_list((void **)(&tmp2),3,0,"qrfactor.c",133);
  mem_stat_reg_list((void **)(&gamma),3,0,"qrfactor.c",134);
  mem_stat_reg_list((void **)(&w),3,0,"qrfactor.c",135);
/* initialise gamma and px */
  for (j = 0; j <= A -> n - 1; j += 1) {
    px -> pe[j] = j;
    sum = 0.0;
    for (i = 0; i <= A -> m - 1; i += 1) {
      sum += square(A -> me[i][j]);
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
      
#pragma omp parallel for private (tmp,i) firstprivate (i_max)
      for (i = 0; i <= A -> m - 1; i += 1) {
        tmp = A -> me[i][k];
        A -> me[i][k] = A -> me[i][i_max];
        A -> me[i][i_max] = tmp;
      }
    }
/* get H/holder vector for the k-th column */
    get_col(A,k,tmp1);
/* hhvec(tmp1,k,&beta->ve[k],tmp1,&A->me[k][k]); */
    hhvec(tmp1,k,&beta,tmp1,&A -> me[k][k]);
    diag -> ve[k] = tmp1 -> ve[k];
/* apply H/holder vector to remaining columns */
/* hhtrcols(A,k,k+1,tmp1,beta->ve[k]); */
    _hhtrcols(A,k,k + 1,tmp1,beta,w);
/* update gamma values */
    for (j = k + 1; j <= A -> n - 1; j += 1) {
      gamma -> ve[j] -= square(A -> me[k][j]);
    }
  }
#ifdef	THREADSAFE
#endif
  return A;
}
/* Qsolve -- solves Qx = b, Q is an orthogonal matrix stored in compact
   form a la QRfactor() -- may be in-situ */
#ifndef ANSI_C
#else

VEC *_Qsolve(const MAT *QR,const VEC *diag,const VEC *b,VEC *x,VEC *tmp)
#endif
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
    ev_err("qrfactor.c",8,217,"_Qsolve",0);
  if (diag -> dim < limit || b -> dim != QR -> m) 
    ev_err("qrfactor.c",1,219,"_Qsolve",0);
  x = v_resize(x,(QR -> m));
  if (tmp == ((VEC *)((void *)0))) 
    dynamic = 1;
  tmp = v_resize(tmp,(QR -> m));
/* apply H/holder transforms in normal order */
  x = _v_copy(b,x,0);
  for (k = 0; k <= limit - 1; k += 1) {
    get_col(QR,k,tmp);
    r_ii = fabs(tmp -> ve[k]);
    tmp -> ve[k] = diag -> ve[k];
    tmp_val = r_ii * fabs(diag -> ve[k]);
    beta = (tmp_val == 0.0?0.0 : 1.0 / tmp_val);
/* hhtrvec(tmp,beta->ve[k],k,x,x); */
    hhtrvec(tmp,beta,k,x,x);
  }
  if (dynamic) 
    (v_free(tmp) , tmp = ((VEC *)((void *)0)));
  return x;
}
/* makeQ -- constructs orthogonal matrix from Householder vectors stored in
   compact QR form */
#ifndef ANSI_C
#else

MAT *makeQ(const MAT *QR,const VEC *diag,MAT *Qout)
#endif
{
  static VEC *tmp1 = (VEC *)((void *)0);
  static VEC *tmp2 = (VEC *)((void *)0);
  unsigned int i;
  unsigned int limit;
  double beta;
  double r_ii;
  double tmp_val;
  int j;
  limit = (QR -> m > QR -> n?QR -> n : QR -> m);
  if (!QR || !diag) 
    ev_err("qrfactor.c",8,261,"makeQ",0);
  if (diag -> dim < limit) 
    ev_err("qrfactor.c",1,263,"makeQ",0);
  if (Qout == ((MAT *)((void *)0)) || Qout -> m < QR -> m || Qout -> n < QR -> m) 
    Qout = m_get((QR -> m),(QR -> m));
  tmp1 = v_resize(tmp1,(QR -> m));
/* contains basis vec & columns of Q */
  tmp2 = v_resize(tmp2,(QR -> m));
/* contains H/holder vectors */
  mem_stat_reg_list((void **)(&tmp1),3,0,"qrfactor.c",269);
  mem_stat_reg_list((void **)(&tmp2),3,0,"qrfactor.c",270);
  for (i = 0; i <= QR -> m - 1; i += 1) {
/* get i-th column of Q */
/* set up tmp1 as i-th basis vector */
    
#pragma omp parallel for private (j)
    for (j = 0; ((unsigned int )j) <= QR -> m - 1; j += 1) {
      tmp1 -> ve[j] = 0.0;
    }
    tmp1 -> ve[i] = 1.0;
/* apply H/h transforms in reverse order */
    for (j = (limit - 1); j >= 0; j += -1) {
      get_col(QR,j,tmp2);
      r_ii = fabs(tmp2 -> ve[j]);
      tmp2 -> ve[j] = diag -> ve[j];
      tmp_val = r_ii * fabs(diag -> ve[j]);
      beta = (tmp_val == 0.0?0.0 : 1.0 / tmp_val);
/* hhtrvec(tmp2,beta->ve[j],j,tmp1,tmp1); */
      hhtrvec(tmp2,beta,j,tmp1,tmp1);
    }
/* insert into Q */
    _set_col(Qout,i,tmp1,0);
  }
#ifdef	THREADSAFE
#endif
  return Qout;
}
/* makeR -- constructs upper triangular matrix from QR (compact form)
   -- may be in-situ (all it does is zero the lower 1/2) */
#ifndef ANSI_C
#else

MAT *makeR(const MAT *QR,MAT *Rout)
#endif
{
  unsigned int i;
  unsigned int j;
  if (QR == ((MAT *)((void *)0))) 
    ev_err("qrfactor.c",8,314,"makeR",0);
  Rout = _m_copy(QR,Rout,0,0);
  
#pragma omp parallel for private (i)
  for (i = 1; i <= QR -> m - 1; i += 1) {
    for (j = 0; j < QR -> n && j < i; j++) 
      Rout -> me[i][j] = 0.0;
  }
  return Rout;
}
/* QRsolve -- solves the system Q.R.x=b where Q & R are stored in compact form
   -- returns x, which is created if necessary */
#ifndef ANSI_C
/* , *beta */
#else

VEC *QRsolve(const MAT *QR,const VEC *diag,const VEC *b,VEC *x)
#endif
{
  int limit;
  static VEC *tmp = (VEC *)((void *)0);
  if (!QR || !diag || !b) 
    ev_err("qrfactor.c",8,338,"QRsolve",0);
  limit = ((QR -> m > QR -> n?QR -> n : QR -> m));
  if (diag -> dim < limit || b -> dim != QR -> m) 
    ev_err("qrfactor.c",1,341,"QRsolve",0);
  tmp = v_resize(tmp,limit);
  mem_stat_reg_list((void **)(&tmp),3,0,"qrfactor.c",343);
  x = v_resize(x,(QR -> n));
  _Qsolve(QR,diag,b,x,tmp);
  x = Usolve(QR,x,x,0.0);
  v_resize(x,(QR -> n));
#ifdef	THREADSAFE
#endif
  return x;
}
/* QRCPsolve -- solves A.x = b where A is factored by QRCPfactor()
   -- assumes that A is in the compact factored form */
#ifndef ANSI_C
#else

VEC *QRCPsolve(const MAT *QR,const VEC *diag,PERM *pivot,const VEC *b,VEC *x)
#endif
{
  static VEC *tmp = (VEC *)((void *)0);
  if (!QR || !diag || !pivot || !b) 
    ev_err("qrfactor.c",8,373,"QRCPsolve",0);
  if (QR -> m > diag -> dim && QR -> n > diag -> dim || QR -> n != pivot -> size) 
    ev_err("qrfactor.c",1,375,"QRCPsolve",0);
  tmp = QRsolve(QR,diag,b,tmp);
  mem_stat_reg_list((void **)(&tmp),3,0,"qrfactor.c",378);
  x = pxinv_vec(pivot,tmp,x);
#ifdef	THREADSAFE
#endif
  return x;
}
/* Umlt -- compute out = upper_triang(U).x
	-- may be in situ */
#ifndef ANSI_C
#else

static VEC *Umlt(const MAT *U,const VEC *x,VEC *out)
#endif
{
  int i;
  int limit;
  if (U == ((MAT *)((void *)0)) || x == ((VEC *)((void *)0))) 
    ev_err("qrfactor.c",8,401,"Umlt",0);
  limit = ((U -> m > U -> n?U -> n : U -> m));
  if (limit != x -> dim) 
    ev_err("qrfactor.c",1,404,"Umlt",0);
  if (out == ((VEC *)((void *)0)) || out -> dim < limit) 
    out = v_resize(out,limit);
  for (i = 0; i <= limit - 1; i += 1) {
    out -> ve[i] = __ip__((&x -> ve[i]),(&U -> me[i][i]),limit - i);
  }
  return out;
}
/* UTmlt -- returns out = upper_triang(U)^T.x */
#ifndef ANSI_C
#else

static VEC *UTmlt(const MAT *U,const VEC *x,VEC *out)
#endif
{
  double sum;
  int i;
  int j;
  int limit;
  if (U == ((MAT *)((void *)0)) || x == ((VEC *)((void *)0))) 
    ev_err("qrfactor.c",8,426,"UTmlt",0);
  limit = ((U -> m > U -> n?U -> n : U -> m));
  if (out == ((VEC *)((void *)0)) || out -> dim < limit) 
    out = v_resize(out,limit);
  
#pragma omp parallel for private (sum,i,j)
  for (i = limit - 1; i >= 0; i += -1) {
    sum = 0.0;
    
#pragma omp parallel for private (j) reduction (+:sum)
    for (j = 0; j <= i; j += 1) {
      sum += U -> me[j][i] * x -> ve[j];
    }
    out -> ve[i] = sum;
  }
  return out;
}
/* QRTsolve -- solve A^T.sc = c where the QR factors of A are stored in
	compact form
	-- returns sc
	-- original due to Mike Osborne modified Wed 09th Dec 1992 */
#ifndef ANSI_C
#else

VEC *QRTsolve(const MAT *A,const VEC *diag,const VEC *c,VEC *sc)
#endif
{
  int i;
  int j;
  int k;
  int n;
  int p;
  double beta;
  double r_ii;
  double s;
  double tmp_val;
  if (!A || !diag || !c) 
    ev_err("qrfactor.c",8,457,"QRTsolve",0);
  if (diag -> dim < ((A -> m > A -> n?A -> n : A -> m))) 
    ev_err("qrfactor.c",1,459,"QRTsolve",0);
  sc = v_resize(sc,(A -> m));
  n = (sc -> dim);
  p = (c -> dim);
  if (n == p) 
    k = p - 2;
   else 
    k = p - 1;
  v_zero(sc);
  sc -> ve[0] = c -> ve[0] / A -> me[0][0];
  if (n == 1) 
    return sc;
  if (p > 1) {
    for (i = 1; i <= p - 1; i += 1) {
      s = 0.0;
      
#pragma omp parallel for private (j) reduction (+:s)
      for (j = 0; j <= i - 1; j += 1) {
        s += A -> me[j][i] * sc -> ve[j];
      }
      if (A -> me[i][i] == 0.0) 
        ev_err("qrfactor.c",4,479,"QRTsolve",0);
      sc -> ve[i] = (c -> ve[i] - s) / A -> me[i][i];
    }
  }
  for (i = k; i >= 0; i += -1) {
    s = diag -> ve[i] * sc -> ve[i];
    
#pragma omp parallel for private (j) reduction (+:s)
    for (j = i + 1; j <= n - 1; j += 1) {
      s += A -> me[j][i] * sc -> ve[j];
    }
    r_ii = fabs(A -> me[i][i]);
    tmp_val = r_ii * fabs(diag -> ve[i]);
    beta = (tmp_val == 0.0?0.0 : 1.0 / tmp_val);
    tmp_val = beta * s;
    sc -> ve[i] -= tmp_val * diag -> ve[i];
    
#pragma omp parallel for private (j) firstprivate (tmp_val)
    for (j = i + 1; j <= n - 1; j += 1) {
      sc -> ve[j] -= tmp_val * A -> me[j][i];
    }
  }
  return sc;
}
/* QRcondest -- returns an estimate of the 2-norm condition number of the
		matrix factorised by QRfactor() or QRCPfactor()
	-- note that as Q does not affect the 2-norm condition number,
		it is not necessary to pass the diag, beta (or pivot) vectors
	-- generates a lower bound on the true condition number
	-- if the matrix is exactly singular, HUGE_VAL is returned
	-- note that QRcondest() is likely to be more reliable for
		matrices factored using QRCPfactor() */
#ifndef ANSI_C
#else

double QRcondest(const MAT *QR)
#endif
{
  static VEC *y = (VEC *)((void *)0);
  double norm1;
  double norm2;
  double sum;
  double tmp1;
  double tmp2;
  int i;
  int j;
  int limit;
  if (QR == ((MAT *)((void *)0))) 
    ev_err("qrfactor.c",8,520,"QRcondest",0);
  limit = ((QR -> m > QR -> n?QR -> n : QR -> m));
  
#pragma omp parallel for private (i)
  for (i = 0; i <= limit - 1; i += 1) {
    if (QR -> me[i][i] == 0.0) 
      return __builtin_huge_val();
  }
  y = v_resize(y,limit);
  mem_stat_reg_list((void **)(&y),3,0,"qrfactor.c",528);
/* use the trick for getting a unit vector y with ||R.y||_inf small
       from the LU condition estimator */
  for (i = 0; i <= limit - 1; i += 1) {
    sum = 0.0;
    
#pragma omp parallel for private (j) reduction (-:sum)
    for (j = 0; j <= i - 1; j += 1) {
      sum -= QR -> me[j][i] * y -> ve[j];
    }
    sum -= (sum < 0.0?1.0 : - 1.0);
    y -> ve[i] = sum / QR -> me[i][i];
  }
  UTmlt(QR,y,y);
/* now apply inverse power method to R^T.R */
  for (i = 0; i <= 2; i += 1) {
    tmp1 = _v_norm2(y,((VEC *)((void *)0)));
    sv_mlt(1 / tmp1,y,y);
    UTsolve(QR,y,y,0.0);
    tmp2 = _v_norm2(y,((VEC *)((void *)0)));
    sv_mlt(1 / _v_norm2(y,((VEC *)((void *)0))),y,y);
    Usolve(QR,y,y,0.0);
  }
/* now compute approximation for ||R^{-1}||_2 */
  norm1 = sqrt(tmp1) * sqrt(tmp2);
/* now use complementary approach to compute approximation to ||R||_2 */
  for (i = limit - 1; i >= 0; i += -1) {
    sum = 0.0;
    
#pragma omp parallel for private (j) reduction (+:sum) firstprivate (limit)
    for (j = i + 1; j <= limit - 1; j += 1) {
      sum += QR -> me[i][j] * y -> ve[j];
    }
    y -> ve[i] = (sum >= 0.0?1.0 : - 1.0);
    y -> ve[i] = (QR -> me[i][i] >= 0.0?y -> ve[i] : -y -> ve[i]);
  }
/* now apply power method to R^T.R */
  for (i = 0; i <= 2; i += 1) {
    tmp1 = _v_norm2(y,((VEC *)((void *)0)));
    sv_mlt(1 / tmp1,y,y);
    Umlt(QR,y,y);
    tmp2 = _v_norm2(y,((VEC *)((void *)0)));
    sv_mlt(1 / tmp2,y,y);
    UTmlt(QR,y,y);
  }
  norm2 = sqrt(tmp1) * sqrt(tmp2);
/* printf("QRcondest: norm1 = %g, norm2 = %g\n",norm1,norm2); */
#ifdef THREADSAFE
#endif
  return norm1 * norm2;
}
