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
static char rcsid[] = "$Id: zcopy.c,v 1.1 1994/01/13 04:28:42 des Exp $";
#include	<stdio.h>
#include	"zmatrix.h"
/* _zm_copy -- copies matrix into new area */
#ifndef ANSI_C
#else

ZMAT *_zm_copy(const ZMAT *in,ZMAT *out,int i0,int j0)
#endif
{
  unsigned int i;
/* ,j */
  if (in == ((ZMAT *)((void *)0))) 
    ev_err("zcopy.c",8,45,"_zm_copy",0);
  if (in == out) 
    return out;
  if (out == ((ZMAT *)((void *)0)) || out -> m < in -> m || out -> n < in -> n) 
    out = zm_resize(out,(in -> m),(in -> n));
  for (i = i0; i <= in -> m - 1; i += 1) {
    memmove((&out -> me[i][j0]),(&in -> me[i][j0]),(in -> n - j0) * sizeof(complex ));
  }
/* for ( j=j0; j < in->n; j++ )
			out->me[i][j] = in->me[i][j]; */
  return out;
}
/* _zv_copy -- copies vector into new area */
#ifndef ANSI_C
#else

ZVEC *_zv_copy(const ZVEC *in,ZVEC *out,int i0)
#endif
{
/* unsigned int	i,j; */
  if (in == ((ZVEC *)((void *)0))) 
    ev_err("zcopy.c",8,72,"_zv_copy",0);
  if (in == out) 
    return out;
  if (out == ((ZVEC *)((void *)0)) || out -> dim < in -> dim) 
    out = zv_resize(out,(in -> dim));
  memmove((&out -> ve[i0]),(&in -> ve[i0]),(in -> dim - i0) * sizeof(complex ));
/* for ( i=i0; i < in->dim; i++ )
		out->ve[i] = in->ve[i]; */
  return out;
}
/*
	The z._move() routines are for moving blocks of memory around
	within Meschach data structures and for re-arranging matrices,
	vectors etc.
*/
/* zm_move -- copies selected pieces of a matrix
	-- moves the m0 x n0 submatrix with top-left cor-ordinates (i0,j0)
	   to the corresponding submatrix of out with top-left co-ordinates
	   (i1,j1)
	-- out is resized (& created) if necessary */
#ifndef ANSI_C
#else

ZMAT *zm_move(const ZMAT *in,int i0,int j0,int m0,int n0,ZMAT *out,int i1,int j1)
#endif
{
  int i;
  if (!in) 
    ev_err("zcopy.c",8,109,"zm_move",0);
  if (i0 < 0 || j0 < 0 || i1 < 0 || j1 < 0 || m0 < 0 || n0 < 0 || (i0 + m0) > in -> m || (j0 + n0) > in -> n) 
    ev_err("zcopy.c",2,112,"zm_move",0);
  if (!out) 
    out = zm_resize(out,i1 + m0,j1 + n0);
   else if ((i1 + m0) > out -> m || (j1 + n0) > out -> n) 
    out = zm_resize(out,((out -> m > (i1 + m0)?out -> m : (i1 + m0))),((out -> n > (j1 + n0)?out -> n : (j1 + n0))));
  for (i = 0; i <= m0 - 1; i += 1) {
    memmove((&out -> me[i1 + i][j1]),(&in -> me[i0 + i][j0]),n0 * sizeof(complex ));
  }
  return out;
}
/* zv_move -- copies selected pieces of a vector
	-- moves the length dim0 subvector with initial index i0
	   to the corresponding subvector of out with initial index i1
	-- out is resized if necessary */
#ifndef ANSI_C
#else

ZVEC *zv_move(const ZVEC *in,int i0,int dim0,ZVEC *out,int i1)
#endif
{
  if (!in) 
    ev_err("zcopy.c",8,140,"zv_move",0);
  if (i0 < 0 || dim0 < 0 || i1 < 0 || (i0 + dim0) > in -> dim) 
    ev_err("zcopy.c",2,143,"zv_move",0);
  if (!out || (i1 + dim0) > out -> dim) 
    out = zv_resize(out,i1 + dim0);
  memmove((&out -> ve[i1]),(&in -> ve[i0]),dim0 * sizeof(complex ));
  return out;
}
/* zmv_move -- copies selected piece of matrix to a vector
	-- moves the m0 x n0 submatrix with top-left co-ordinate (i0,j0) to
	   the subvector with initial index i1 (and length m0*n0)
	-- rows are copied contiguously
	-- out is resized if necessary */
#ifndef ANSI_C
#else

ZVEC *zmv_move(const ZMAT *in,int i0,int j0,int m0,int n0,ZVEC *out,int i1)
#endif
{
  int dim1;
  int i;
  if (!in) 
    ev_err("zcopy.c",8,172,"zmv_move",0);
  if (i0 < 0 || j0 < 0 || m0 < 0 || n0 < 0 || i1 < 0 || (i0 + m0) > in -> m || (j0 + n0) > in -> n) 
    ev_err("zcopy.c",2,175,"zmv_move",0);
  dim1 = m0 * n0;
  if (!out || (i1 + dim1) > out -> dim) 
    out = zv_resize(out,i1 + dim1);
  for (i = 0; i <= m0 - 1; i += 1) {
    memmove((&out -> ve[i1 + i * n0]),(&in -> me[i0 + i][j0]),n0 * sizeof(complex ));
  }
  return out;
}
/* zvm_move -- copies selected piece of vector to a matrix
	-- moves the subvector with initial index i0 and length m1*n1 to
	   the m1 x n1 submatrix with top-left co-ordinate (i1,j1)
        -- copying is done by rows
	-- out is resized if necessary */
#ifndef ANSI_C
#else

ZMAT *zvm_move(const ZVEC *in,int i0,ZMAT *out,int i1,int j1,int m1,int n1)
#endif
{
  int dim0;
  int i;
  if (!in) 
    ev_err("zcopy.c",8,205,"zvm_move",0);
  if (i0 < 0 || i1 < 0 || j1 < 0 || m1 < 0 || n1 < 0 || (i0 + m1 * n1) > in -> dim) 
    ev_err("zcopy.c",2,208,"zvm_move",0);
  if (!out) 
    out = zm_resize(out,i1 + m1,j1 + n1);
   else 
    out = zm_resize(out,(((i1 + m1) > out -> m?(i1 + m1) : out -> m)),(((j1 + n1) > out -> n?(j1 + n1) : out -> n)));
  dim0 = m1 * n1;
  for (i = 0; i <= m1 - 1; i += 1) {
    memmove((&out -> me[i1 + i][j1]),(&in -> ve[i0 + i * n1]),n1 * sizeof(complex ));
  }
  return out;
}
