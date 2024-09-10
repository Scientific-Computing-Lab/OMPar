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
	File containing Lanczos type routines for finding eigenvalues
	of large, sparse, symmetic matrices
*/
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include	"sparse.h"
#include <omp.h> 
static char rcsid[] = "$Id: lanczos.c,v 1.4 1994/01/13 05:28:24 des Exp $";
#ifdef ANSI_C
extern VEC *trieig(VEC *,VEC *,MAT *);
#else
#endif
/* lanczos -- raw lanczos algorithm -- no re-orthogonalisation
	-- creates T matrix of size == m,
		but no larger than before beta_k == 0
	-- uses passed routine to do matrix-vector multiplies */

void lanczos(A_fn,A_params,m,x0,a,b,beta2,Q)
VEC *(*A_fn)();
void *A_params;
int m;
VEC *x0;
VEC *a;
VEC *b;
double *beta2;
MAT *Q;
/* VEC *(*A_fn)(void *A_params,VEC *in, VEC *out) */
{
  int j;
  VEC *v;
  VEC *w;
  VEC *tmp;
  double alpha;
  double beta;
  if (!A_fn || !x0 || !a || !b) 
    ev_err("lanczos.c",8,62,"lanczos",0);
  if (m <= 0) 
    ev_err("lanczos.c",2,64,"lanczos",0);
  if (Q && (Q -> m < x0 -> dim || Q -> n < m)) 
    ev_err("lanczos.c",1,66,"lanczos",0);
  a = v_resize(a,((unsigned int )m));
  b = v_resize(b,((unsigned int )(m - 1)));
  v = v_get((x0 -> dim));
  w = v_get((x0 -> dim));
  tmp = v_get((x0 -> dim));
  beta = 1.0;
/* normalise x0 as w */
  sv_mlt(1.0 / _v_norm2(x0,((VEC *)((void *)0))),x0,w);
  ( *A_fn)(A_params,w,v);
  for (j = 0; j <= m - 1; j += 1) {
/* store w in Q if Q not NULL */
    if (Q) 
      _set_col(Q,j,w,0);
    alpha = _in_prod(w,v,0);
    a -> ve[j] = alpha;
    v_mltadd(v,w,-alpha,v);
    beta = _v_norm2(v,((VEC *)((void *)0)));
    if (beta == 0.0) {
      v_resize(a,(((unsigned int )j) + 1));
      v_resize(b,((unsigned int )j));
       *beta2 = 0.0;
      if (Q) 
        Q = m_resize(Q,(Q -> m),j + 1);
      return ;
    }
    if (j < m - 1) 
      b -> ve[j] = beta;
    _v_copy(w,tmp,0);
    sv_mlt(1 / beta,v,w);
    sv_mlt(-beta,tmp,v);
    ( *A_fn)(A_params,w,tmp);
    v_add(v,tmp,v);
  }
   *beta2 = beta;
  (v_free(v) , v = ((VEC *)((void *)0)));
  (v_free(w) , w = ((VEC *)((void *)0)));
  (v_free(tmp) , tmp = ((VEC *)((void *)0)));
}
extern double frexp() __attribute__((no_throw)) ;
extern double ldexp() __attribute__((no_throw)) ;
/* product -- returns the product of a long list of numbers
	-- answer stored in mant (mantissa) and expt (exponent) */

static double product(a,offset,expt)
VEC *a;
double offset;
int *expt;
{
  double mant;
  double tmp_fctr;
  int i;
  int tmp_expt;
  if (!a) 
    ev_err("lanczos.c",8,126,"product",0);
  mant = 1.0;
   *expt = 0;
  if (offset == 0.0) 
    for (i = 0; ((unsigned int )i) <= a -> dim - 1; i += 1) {
      mant *= frexp(a -> ve[i],&tmp_expt);
       *expt += tmp_expt;
      if (!(i % 10)) {
        mant = frexp(mant,&tmp_expt);
         *expt += tmp_expt;
      }
    }
   else 
    for (i = 0; ((unsigned int )i) <= a -> dim - 1; i += 1) {
      tmp_fctr = a -> ve[i] - offset;
      tmp_fctr += (tmp_fctr > 0.0?-((double )2.22044604925031308084726333618164062e-16L) * offset : ((double )2.22044604925031308084726333618164062e-16L) * offset);
      mant *= frexp(tmp_fctr,&tmp_expt);
       *expt += tmp_expt;
      if (!(i % 10)) {
        mant = frexp(mant,&tmp_expt);
         *expt += tmp_expt;
      }
    }
  mant = frexp(mant,&tmp_expt);
   *expt += tmp_expt;
  return mant;
}
/* product2 -- returns the product of a long list of numbers
	-- answer stored in mant (mantissa) and expt (exponent) */

static double product2(a,k,expt)
VEC *a;
int k;
int *expt;
/* entry of a to leave out */
{
  double mant;
  double mu;
  double tmp_fctr;
  int i;
  int tmp_expt;
  if (!a) 
    ev_err("lanczos.c",8,173,"product2",0);
  if (k < 0 || k >= a -> dim) 
    ev_err("lanczos.c",2,175,"product2",0);
  mant = 1.0;
   *expt = 0;
  mu = a -> ve[k];
  for (i = 0; ((unsigned int )i) <= a -> dim - 1; i += 1) {
    if (i == k) 
      continue; 
    tmp_fctr = a -> ve[i] - mu;
    tmp_fctr += (tmp_fctr > 0.0?-((double )2.22044604925031308084726333618164062e-16L) * mu : ((double )2.22044604925031308084726333618164062e-16L) * mu);
    mant *= frexp(tmp_fctr,&tmp_expt);
     *expt += tmp_expt;
    if (!(i % 10)) {
      mant = frexp(mant,&tmp_expt);
       *expt += tmp_expt;
    }
  }
  mant = frexp(mant,&tmp_expt);
   *expt += tmp_expt;
  return mant;
}
/* dbl_cmp -- comparison function to pass to qsort() */

static int dbl_cmp(x,y)
double *x;
double *y;
{
  double tmp;
  tmp =  *x -  *y;
  return tmp > 0?1 : ((tmp < 0?- 1 : 0));
}
/* lanczos2 -- lanczos + error estimate for every e-val
	-- uses Cullum & Willoughby approach, Sparse Matrix Proc. 1978
	-- returns multiple e-vals where multiple e-vals may not exist
	-- returns evals vector */

VEC *lanczos2(A_fn,A_params,m,x0,evals,err_est)
VEC *(*A_fn)();
void *A_params;
int m;
VEC *x0;
VEC *evals;
VEC *err_est;
/* initial vector */
/* eigenvalue vector */
/* error estimates of eigenvalues */
{
  VEC *a;
  static VEC *b = (VEC *)((void *)0);
  static VEC *a2 = (VEC *)((void *)0);
  static VEC *b2 = (VEC *)((void *)0);
  double beta;
  double pb_mant;
  double det_mant;
  double det_mant1;
  double det_mant2;
  int i;
  int pb_expt;
  int det_expt;
  int det_expt1;
  int det_expt2;
  if (!A_fn || !x0) 
    ev_err("lanczos.c",8,228,"lanczos2",0);
  if (m <= 0) 
    ev_err("lanczos.c",10,230,"lanczos2",0);
  a = evals;
  a = v_resize(a,((unsigned int )m));
  b = v_resize(b,((unsigned int )(m - 1)));
  mem_stat_reg_list((void **)(&b),3,0,"lanczos.c",235);
  lanczos(A_fn,A_params,m,x0,a,b,&beta,(MAT *)((void *)0));
/* printf("# beta =%g\n",beta); */
  pb_mant = 0.0;
  if (err_est) {
    pb_mant = product(b,(double )0.0,&pb_expt);
/* printf("# pb_mant = %g, pb_expt = %d\n",pb_mant, pb_expt); */
  }
/* printf("# diags =\n");	out_vec(a); */
/* printf("# off diags =\n");	out_vec(b); */
  a2 = v_resize(a2,(a -> dim - 1));
  b2 = v_resize(b2,(b -> dim - 1));
  mem_stat_reg_list((void **)(&a2),3,0,"lanczos.c",251);
  mem_stat_reg_list((void **)(&b2),3,0,"lanczos.c",252);
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= a2 -> dim - ((unsigned int )1) - 1; i += 1) {
    a2 -> ve[i] = a -> ve[i + 1];
    b2 -> ve[i] = b -> ve[i + 1];
  }
  a2 -> ve[a2 -> dim - 1] = a -> ve[a2 -> dim];
  trieig(a,b,(MAT *)((void *)0));
/* sort evals as a courtesy */
  qsort((void *)(a -> ve),((int )(a -> dim)),sizeof(double ),((int (*)())dbl_cmp));
/* error estimates */
  if (err_est) {
    err_est = v_resize(err_est,((unsigned int )m));
    trieig(a2,b2,(MAT *)((void *)0));
/* printf("# a =\n");	out_vec(a); */
/* printf("# a2 =\n");	out_vec(a2); */
    for (i = 0; ((unsigned int )i) <= a -> dim - 1; i += 1) {
      det_mant1 = product2(a,i,&det_expt1);
      det_mant2 = product(a2,(double )a -> ve[i],&det_expt2);
/* printf("# det_mant1=%g, det_expt1=%d\n",
					det_mant1,det_expt1); */
/* printf("# det_mant2=%g, det_expt2=%d\n",
					det_mant2,det_expt2); */
      if (det_mant1 == 0.0) {
/* multiple e-val of T */
        err_est -> ve[i] = 0.0;
        continue; 
      }
       else if (det_mant2 == 0.0) {
        err_est -> ve[i] = __builtin_huge_val();
        continue; 
      }
      if ((det_expt1 + det_expt2) % 2) 
/* if odd... */
        det_mant = sqrt(2.0 * fabs(det_mant1 * det_mant2));
       else 
/* if even... */
        det_mant = sqrt((fabs(det_mant1 * det_mant2)));
      det_expt = (det_expt1 + det_expt2) / 2;
      err_est -> ve[i] = fabs(beta * ldexp(pb_mant / det_mant,pb_expt - det_expt));
    }
  }
#ifdef	THREADSAFE
#endif
  return a;
}
/* sp_lanczos -- version that uses sparse matrix data structure */

void sp_lanczos(A,m,x0,a,b,beta2,Q)
SPMAT *A;
int m;
VEC *x0;
VEC *a;
VEC *b;
double *beta2;
MAT *Q;
{
  lanczos(sp_mv_mlt,A,m,x0,a,b,beta2,Q);
}
/* sp_lanczos2 -- version of lanczos2() that uses sparse matrix data
					structure */

VEC *sp_lanczos2(A,m,x0,evals,err_est)
SPMAT *A;
int m;
VEC *x0;
VEC *evals;
VEC *err_est;
/* initial vector */
/* eigenvalue vector */
/* error estimates of eigenvalues */
{
  return lanczos2(sp_mv_mlt,A,m,x0,evals,err_est);
}
