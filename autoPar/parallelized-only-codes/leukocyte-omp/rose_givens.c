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
	Givens operations file. Contains routines for calculating and
	applying givens rotations for/to vectors and also to matrices by
	row and by column.
*/
/* givens.c 1.2 11/25/87 */
#include <omp.h> 
static char rcsid[] = "$Id: givens.c,v 1.3 1995/03/27 15:41:15 des Exp $";
#include	<stdio.h>
#include	<math.h>
#include	"matrix.h"
#include        "matrix2.h"
/* givens -- returns c,s parameters for Givens rotation to
		eliminate y in the vector [ x y ]' */
#ifndef ANSI_C
#else

void givens(double x,double y,double *c,double *s)
#endif
{
  double norm;
  norm = sqrt(x * x + y * y);
  if (norm == 0.0) {
     *c = 1.0;
     *s = 0.0;
  }
   else 
/* identity */
{
     *c = x / norm;
     *s = y / norm;
  }
}
/* rot_vec -- apply Givens rotation to x's i & k components */
#ifndef ANSI_C
#else

VEC *rot_vec(const VEC *x,unsigned int i,unsigned int k,double c,double s,VEC *out)
#endif
{
  double temp;
  if (x == ((VEC *)((void *)0))) 
    ev_err("givens.c",8,77,"rot_vec",0);
  if (i >= x -> dim || k >= x -> dim) 
    ev_err("givens.c",10,79,"rot_vec",0);
  out = _v_copy(x,out,0);
/* temp = c*out->ve[i] + s*out->ve[k]; */
  temp = c * out -> ve[i] + s * out -> ve[k];
/* out->ve[k] = -s*out->ve[i] + c*out->ve[k]; */
  out -> ve[k] = -s * out -> ve[i] + c * out -> ve[k];
/* out->ve[i] = temp; */
  out -> ve[i] = temp;
  return out;
}
/* rot_rows -- premultiply mat by givens rotation described by c,s */
#ifndef ANSI_C
#else

MAT *rot_rows(const MAT *mat,unsigned int i,unsigned int k,double c,double s,MAT *out)
#endif
{
  unsigned int j;
  double temp;
  if (mat == ((MAT *)((void *)0))) 
    ev_err("givens.c",8,107,"rot_rows",0);
  if (i >= mat -> m || k >= mat -> m) 
    ev_err("givens.c",10,109,"rot_rows",0);
  if (mat != out) 
    out = _m_copy(mat,(m_resize(out,(mat -> m),(mat -> n))),0,0);
  
#pragma omp parallel for private (temp,j) firstprivate (i,k,c,s)
  for (j = 0; j <= mat -> n - 1; j += 1) {
/* temp = c*out->me[i][j] + s*out->me[k][j]; */
    temp = c * out -> me[i][j] + s * out -> me[k][j];
/* out->me[k][j] = -s*out->me[i][j] + c*out->me[k][j]; */
    out -> me[k][j] = -s * out -> me[i][j] + c * out -> me[k][j];
/* out->me[i][j] = temp; */
    out -> me[i][j] = temp;
  }
  return out;
}
/* rot_cols -- postmultiply mat by givens rotation described by c,s */
#ifndef ANSI_C
#else

MAT *rot_cols(const MAT *mat,unsigned int i,unsigned int k,double c,double s,MAT *out)
#endif
{
  unsigned int j;
  double temp;
  if (mat == ((MAT *)((void *)0))) 
    ev_err("givens.c",8,141,"rot_cols",0);
  if (i >= mat -> n || k >= mat -> n) 
    ev_err("givens.c",10,143,"rot_cols",0);
  if (mat != out) 
    out = _m_copy(mat,(m_resize(out,(mat -> m),(mat -> n))),0,0);
  
#pragma omp parallel for private (temp,j) firstprivate (i,k,c,s)
  for (j = 0; j <= mat -> m - 1; j += 1) {
/* temp = c*out->me[j][i] + s*out->me[j][k]; */
    temp = c * out -> me[j][i] + s * out -> me[j][k];
/* out->me[j][k] = -s*out->me[j][i] + c*out->me[j][k]; */
    out -> me[j][k] = -s * out -> me[j][i] + c * out -> me[j][k];
/* out->me[j][i] = temp; */
    out -> me[j][i] = temp;
  }
  return out;
}
