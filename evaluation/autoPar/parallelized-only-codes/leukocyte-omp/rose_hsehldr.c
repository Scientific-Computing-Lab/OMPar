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
		Files for matrix computations
	Householder transformation file. Contains routines for calculating
	householder transformations, applying them to vectors and matrices
	by both row & column.
*/
/* hsehldr.c 1.3 10/8/87 */
static char rcsid[] = "$Id: hsehldr.c,v 1.2 1994/01/13 05:36:29 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"
/* hhvec -- calulates Householder vector to eliminate all entries after the
	i0 entry of the vector vec. It is returned as out. May be in-situ */
#ifndef ANSI_C
#else

VEC *hhvec(const VEC *vec,unsigned int i0,double *beta,VEC *out,double *newval)
#endif
{
  double norm;
  out = _v_copy(vec,out,i0);
  norm = sqrt((_in_prod(out,out,i0)));
  if (norm <= 0.0) {
     *beta = 0.0;
    return out;
  }
   *beta = 1.0 / (norm * (norm + fabs(out -> ve[i0])));
  if (out -> ve[i0] > 0.0) 
     *newval = -norm;
   else 
     *newval = norm;
  out -> ve[i0] -=  *newval;
  return out;
}
/* hhtrvec -- apply Householder transformation to vector 
	-- that is, out <- (I-beta.hh(i0:n).hh(i0:n)^T).in
	-- may be in-situ */
#ifndef ANSI_C
/* hh = Householder vector */
#else

VEC *hhtrvec(const VEC *hh,double beta,unsigned int i0,const VEC *in,VEC *out)
#endif
{
  double scale;
/* unsigned int	i; */
  if (hh == ((VEC *)((void *)0)) || in == ((VEC *)((void *)0))) 
    ev_err("hsehldr.c",8,92,"hhtrvec",0);
  if (in -> dim != hh -> dim) 
    ev_err("hsehldr.c",1,94,"hhtrvec",0);
  if (i0 > in -> dim) 
    ev_err("hsehldr.c",2,96,"hhtrvec",0);
  scale = beta * _in_prod(hh,in,i0);
  out = _v_copy(in,out,0);
  __mltadd__(&out -> ve[i0],(&hh -> ve[i0]),-scale,(int )(in -> dim - i0));
/************************************************************
	for ( i=i0; i<in->dim; i++ )
		out->ve[i] = in->ve[i] - scale*hh->ve[i];
	************************************************************/
  return out;
}
/* hhtrrows -- transform a matrix by a Householder vector by rows
	starting at row i0 from column j0 -- in-situ
	-- that is, M(i0:m,j0:n) <- M(i0:m,j0:n)(I-beta.hh(j0:n).hh(j0:n)^T) */
#ifndef ANSI_C
#else

MAT *hhtrrows(MAT *M,unsigned int i0,unsigned int j0,const VEC *hh,double beta)
#endif
{
  double ip;
  double scale;
  int i;
/*, j */
  if (M == ((MAT *)((void *)0)) || hh == ((VEC *)((void *)0))) 
    ev_err("hsehldr.c",8,127,"hhtrrows",0);
  if (M -> n != hh -> dim) 
    ev_err("hsehldr.c",10,129,"hhtrrows",0);
  if (i0 > M -> m || j0 > M -> n) 
    ev_err("hsehldr.c",2,131,"hhtrrows",0);
  if (beta == 0.0) 
    return M;
/* for each row ... */
  for (i = i0; ((unsigned int )i) <= M -> m - 1; i += 1) {
/* compute inner product */
    ip = __ip__((&M -> me[i][j0]),(&hh -> ve[j0]),(int )(M -> n - j0));
/**************************************************
		ip = 0.0;
		for ( j = j0; j < M->n; j++ )
			ip += M->me[i][j]*hh->ve[j];
		**************************************************/
    scale = beta * ip;
    if (scale == 0.0) 
      continue; 
/* do operation */
    __mltadd__(&M -> me[i][j0],(&hh -> ve[j0]),-scale,(int )(M -> n - j0));
/**************************************************
		for ( j = j0; j < M->n; j++ )
			M->me[i][j] -= scale*hh->ve[j];
		**************************************************/
  }
  return M;
}
/* hhtrcols -- transform a matrix by a Householder vector by columns
	starting at row i0 from column j0 
	-- that is, M(i0:m,j0:n) <- (I-beta.hh(i0:m).hh(i0:m)^T)M(i0:m,j0:n)
	-- in-situ
	-- calls _hhtrcols() with the scratch vector w
	-- Meschach internal routines should call _hhtrcols() to
	avoid excessive memory allocation/de-allocation
*/
#ifndef ANSI_C
#else

MAT *hhtrcols(MAT *M,unsigned int i0,unsigned int j0,const VEC *hh,double beta)
#endif
{
  static VEC *w = (VEC *)((void *)0);
  if (M == ((MAT *)((void *)0)) || hh == ((VEC *)((void *)0)) || w == ((VEC *)((void *)0))) 
    ev_err("hsehldr.c",8,182,"hhtrcols",0);
  if (M -> m != hh -> dim) 
    ev_err("hsehldr.c",1,184,"hhtrcols",0);
  if (i0 > M -> m || j0 > M -> n) 
    ev_err("hsehldr.c",2,186,"hhtrcols",0);
  if (!w || w -> dim < M -> n) 
    w = v_resize(w,(M -> n));
  mem_stat_reg_list((void **)(&w),3,0,"hsehldr.c",190);
  M = _hhtrcols(M,i0,j0,hh,beta,w);
#ifdef THREADSAFE
#endif
  return M;
}
/* _hhtrcols -- transform a matrix by a Householder vector by columns
	starting at row i0 from column j0 
	-- that is, M(i0:m,j0:n) <- (I-beta.hh(i0:m).hh(i0:m)^T)M(i0:m,j0:n)
	-- in-situ
	-- scratch vector w passed as argument
	-- raises error if w == NULL
*/
#ifndef ANSI_C
#else

MAT *_hhtrcols(MAT *M,unsigned int i0,unsigned int j0,const VEC *hh,double beta,VEC *w)
#endif
{
/* Real	ip, scale; */
  int i;
/*, k */
/*  STATIC	VEC	*w = VNULL; */
  if (M == ((MAT *)((void *)0)) || hh == ((VEC *)((void *)0)) || w == ((VEC *)((void *)0))) 
    ev_err("hsehldr.c",8,225,"_hhtrcols",0);
  if (M -> m != hh -> dim) 
    ev_err("hsehldr.c",1,227,"_hhtrcols",0);
  if (i0 > M -> m || j0 > M -> n) 
    ev_err("hsehldr.c",2,229,"_hhtrcols",0);
  if (beta == 0.0) 
    return M;
  if (w -> dim < M -> n) 
    w = v_resize(w,(M -> n));
/*  MEM_STAT_REG(w,TYPE_VEC); */
  v_zero(w);
  for (i = i0; ((unsigned int )i) <= M -> m - 1; i += 1) {
    if (hh -> ve[i] != 0.0) 
      __mltadd__(&w -> ve[j0],(&M -> me[i][j0]),hh -> ve[i],(int )(M -> n - j0));
  }
  for (i = i0; ((unsigned int )i) <= M -> m - 1; i += 1) {
    if (hh -> ve[i] != 0.0) 
      __mltadd__(&M -> me[i][j0],(&w -> ve[j0]),-beta * hh -> ve[i],(int )(M -> n - j0));
  }
  return M;
}
