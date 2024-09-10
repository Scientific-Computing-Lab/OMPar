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
	Complex version
*/
static char rcsid[] = "$Id: zhsehldr.c,v 1.2 1994/04/07 01:43:47 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"zmatrix.h"
#include        "zmatrix2.h"
#define	is_zero(z)	((z).re == 0.0 && (z).im == 0.0)
/* zhhvec -- calulates Householder vector to eliminate all entries after the
	i0 entry of the vector vec. It is returned as out. May be in-situ */

//vec,i0,beta,out,newval)
ZVEC *zhhvec(ZVEC *vec,int i0,double *beta,ZVEC *out,complex *newval)
{
  complex tmp;
  double norm;
  double abs_val;
  if (i0 < 0 || i0 >= vec -> dim) 
    ev_err("zhsehldr.c",2,59,"zhhvec",0);
  out = _zv_copy(vec,out,i0);
  tmp = _zin_prod(out,out,i0,1);
  if (tmp . re <= 0.0) {
     *beta = 0.0;
     *newval = out -> ve[i0];
    return out;
  }
  norm = sqrt(tmp . re);
  abs_val = zabs(out -> ve[i0]);
   *beta = 1.0 / (norm * (norm + abs_val));
  if (abs_val == 0.0) {
    newval -> re = norm;
    newval -> im = 0.0;
  }
   else {
    abs_val = -norm / abs_val;
    newval -> re = abs_val * out -> ve[i0] . re;
    newval -> im = abs_val * out -> ve[i0] . im;
  }
  abs_val = -norm / abs_val;
  out -> ve[i0] . re -= newval -> re;
  out -> ve[i0] . im -= newval -> im;
  return out;
}
/* zhhtrvec -- apply Householder transformation to vector -- may be in-situ */

ZVEC *zhhtrvec(
//hh,beta,i0,in,out)
ZVEC *hh,double beta,int i0,ZVEC *in,ZVEC *out)
{
  complex scale;
  complex tmp;
/* unsigned int	i; */
  if (hh == ((ZVEC *)((void *)0)) || in == ((ZVEC *)((void *)0))) 
    ev_err("zhsehldr.c",8,100,"zhhtrvec",0);
  if (in -> dim != hh -> dim) 
    ev_err("zhsehldr.c",1,102,"zhhtrvec",0);
  if (i0 < 0 || i0 > in -> dim) 
    ev_err("zhsehldr.c",2,104,"zhhvec",0);
  tmp = _zin_prod(hh,in,i0,1);
  scale . re = -beta * tmp . re;
  scale . im = -beta * tmp . im;
  out = _zv_copy(in,out,0);
  __zmltadd__(&out -> ve[i0],(&hh -> ve[i0]),scale,(int )(in -> dim - i0),0);
/************************************************************
	for ( i=i0; i<in->dim; i++ )
		out->ve[i] = in->ve[i] - scale*hh->ve[i];
	************************************************************/
  return out;
}
/* zhhtrrows -- transform a matrix by a Householder vector by rows
	starting at row i0 from column j0 
	-- in-situ
	-- that is, M(i0:m,j0:n) <- M(i0:m,j0:n)(I-beta.hh(j0:n).hh(j0:n)^T) */

ZMAT *zhhtrrows(
//M,i0,j0,hh,beta)
ZMAT *M,int i0,int j0,ZVEC *hh,double beta)
{
  complex ip;
  complex scale;
  int i;
/*, j */
  if (M == ((ZMAT *)((void *)0)) || hh == ((ZVEC *)((void *)0))) 
    ev_err("zhsehldr.c",8,135,"zhhtrrows",0);
  if (M -> n != hh -> dim) 
    ev_err("zhsehldr.c",10,137,"zhhtrrows",0);
  if (i0 < 0 || i0 > M -> m || j0 < 0 || j0 > M -> n) 
    ev_err("zhsehldr.c",2,139,"zhhtrrows",0);
  if (beta == 0.0) 
    return M;
/* for each row ... */
  for (i = i0; ((unsigned int )i) <= M -> m - 1; i += 1) {
/* compute inner product */
    ip = __zip__((&M -> me[i][j0]),(&hh -> ve[j0]),(int )(M -> n - j0),0);
/**************************************************
		ip = 0.0;
		for ( j = j0; j < M->n; j++ )
			ip += M->me[i][j]*hh->ve[j];
		**************************************************/
    scale . re = -beta * ip . re;
    scale . im = -beta * ip . im;
/* if ( scale == 0.0 ) */
    if (scale . re == 0.0 && scale . im == 0.0) 
      continue; 
/* do operation */
    __zmltadd__(&M -> me[i][j0],(&hh -> ve[j0]),scale,(int )(M -> n - j0),1);
/**************************************************
		for ( j = j0; j < M->n; j++ )
			M->me[i][j] -= scale*hh->ve[j];
		**************************************************/
  }
  return M;
}
/* zhhtrcols -- transform a matrix by a Householder vector by columns
	starting at row i0 from column j0 
	-- that is, M(i0:m,j0:n) <- (I-beta.hh(i0:m).hh(i0:m)^T)M(i0:m,j0:n)
	-- in-situ
	-- calls _zhhtrcols() with the scratch vector w
	-- Meschach internal routines should call _zhhtrcols() to
	avoid excessive memory allocation/de-allocation
*/

ZMAT *zhhtrcols(
//M,i0,j0,hh,beta)
ZMAT *M,int i0,int j0,ZVEC *hh,double beta)
{
/* Real	ip, scale; */
  complex scale;
  int i;
/*, k */
  static ZVEC *w = (ZVEC *)((void *)0);
  if (M == ((ZMAT *)((void *)0)) || hh == ((ZVEC *)((void *)0))) 
    ev_err("zhsehldr.c",8,192,"zhhtrcols",0);
  if (M -> m != hh -> dim) 
    ev_err("zhsehldr.c",1,194,"zhhtrcols",0);
  if (i0 < 0 || i0 > M -> m || j0 < 0 || j0 > M -> n) 
    ev_err("zhsehldr.c",2,196,"zhhtrcols",0);
  if (beta == 0.0) 
    return M;
  if (!w || w -> dim < M -> n) 
    w = zv_resize(w,(M -> n));
  mem_stat_reg_list((void **)(&w),8,0,"zhsehldr.c",202);
  M = _zhhtrcols(M,i0,j0,hh,beta,w);
#ifdef THREADSAFE
#endif
  return M;
}
/* _zhhtrcols -- transform a matrix by a Householder vector by columns
	starting at row i0 from column j0 
	-- that is, M(i0:m,j0:n) <- (I-beta.hh(i0:m).hh(i0:m)^T)M(i0:m,j0:n)
	-- in-situ
	-- scratch vector w passed as argument
	-- raises error if w == NULL */

ZMAT *_zhhtrcols(
//M,i0,j0,hh,beta,w)
ZMAT *M,int i0,int j0,ZVEC *hh,double beta,ZVEC *w)
{
/* Real	ip, scale; */
  complex scale;
  int i;
/*, k */
  if (M == ((ZMAT *)((void *)0)) || hh == ((ZVEC *)((void *)0))) 
    ev_err("zhsehldr.c",8,232,"zhhtrcols",0);
  if (M -> m != hh -> dim) 
    ev_err("zhsehldr.c",1,234,"zhhtrcols",0);
  if (i0 < 0 || i0 > M -> m || j0 < 0 || j0 > M -> n) 
    ev_err("zhsehldr.c",2,236,"zhhtrcols",0);
  if (beta == 0.0) 
    return M;
  if (w -> dim < M -> n) 
    w = zv_resize(w,(M -> n));
  zv_zero(w);
  for (i = i0; ((unsigned int )i) <= M -> m - 1; i += 1) {
/* if ( hh->ve[i] != 0.0 ) */
    if (!(hh -> ve[i] . re == 0.0 && hh -> ve[i] . im == 0.0)) 
      __zmltadd__(&w -> ve[j0],(&M -> me[i][j0]),hh -> ve[i],(int )(M -> n - j0),1);
  }
  for (i = i0; ((unsigned int )i) <= M -> m - 1; i += 1) {
/* if ( hh->ve[i] != 0.0 ) */
    if (!(hh -> ve[i] . re == 0.0 && hh -> ve[i] . im == 0.0)) {
      scale . re = -beta * hh -> ve[i] . re;
      scale . im = -beta * hh -> ve[i] . im;
      __zmltadd__(&M -> me[i][j0],(&w -> ve[j0]),scale,(int )(M -> n - j0),1);
    }
  }
  return M;
}
