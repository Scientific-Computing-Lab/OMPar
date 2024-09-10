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
static char rcsid[] = "$Id: copy.c,v 1.2 1994/01/13 05:37:14 des Exp $";
#include	<stdio.h>
#include	"matrix.h"
/* _m_copy -- copies matrix into new area
	-- out(i0:m,j0:n) <- in(i0:m,j0:n) */
#ifndef ANSI_C
#else

MAT *_m_copy(const MAT *in,MAT *out,unsigned int i0,unsigned int j0)
#endif
{
  unsigned int i;
/* ,j */
  if (in == ((MAT *)((void *)0))) 
    ev_err("copy.c",8,47,"_m_copy",0);
  if (in == out) 
    return out;
  if (out == ((MAT *)((void *)0)) || out -> m < in -> m || out -> n < in -> n) 
    out = m_resize(out,(in -> m),(in -> n));
  for (i = i0; i <= in -> m - 1; i += 1) {
    memmove((&out -> me[i][j0]),(&in -> me[i][j0]),(in -> n - j0) * sizeof(double ));
  }
/* for ( j=j0; j < in->n; j++ )
			out->me[i][j] = in->me[i][j]; */
  return out;
}
/* _v_copy -- copies vector into new area
	-- out(i0:dim) <- in(i0:dim) */
#ifndef ANSI_C
#else

VEC *_v_copy(const VEC *in,VEC *out,unsigned int i0)
#endif
{
/* unsigned int	i,j; */
  if (in == ((VEC *)((void *)0))) 
    ev_err("copy.c",8,75,"_v_copy",0);
  if (in == out) 
    return out;
  if (out == ((VEC *)((void *)0)) || out -> dim < in -> dim) 
    out = v_resize(out,(in -> dim));
  memmove((&out -> ve[i0]),(&in -> ve[i0]),(in -> dim - i0) * sizeof(double ));
/* for ( i=i0; i < in->dim; i++ )
		out->ve[i] = in->ve[i]; */
  return out;
}
/* px_copy -- copies permutation 'in' to 'out'
	-- out is resized to in->size */
#ifndef ANSI_C
#else

PERM *px_copy(const PERM *in,PERM *out)
#endif
{
/* int	i; */
  if (in == ((PERM *)((void *)0))) 
    ev_err("copy.c",8,100,"px_copy",0);
  if (in == out) 
    return out;
  if (out == ((PERM *)((void *)0)) || out -> size != in -> size) 
    out = px_resize(out,(in -> size));
  memmove((out -> pe),(in -> pe),(in -> size) * sizeof(unsigned int ));
/* for ( i = 0; i < in->size; i++ )
		out->pe[i] = in->pe[i]; */
  return out;
}
/*
	The .._move() routines are for moving blocks of memory around
	within Meschach data structures and for re-arranging matrices,
	vectors etc.
*/
/* m_move -- copies selected pieces of a matrix
	-- moves the m0 x n0 submatrix with top-left cor-ordinates (i0,j0)
	   to the corresponding submatrix of out with top-left co-ordinates
	   (i1,j1)
	-- out is resized (& created) if necessary */
#ifndef ANSI_C
#else

MAT *m_move(const MAT *in,int i0,int j0,int m0,int n0,MAT *out,int i1,int j1)
#endif
{
  int i;
  if (!in) 
    ev_err("copy.c",8,136,"m_move",0);
  if (i0 < 0 || j0 < 0 || i1 < 0 || j1 < 0 || m0 < 0 || n0 < 0 || (i0 + m0) > in -> m || (j0 + n0) > in -> n) 
    ev_err("copy.c",2,139,"m_move",0);
  if (!out) 
    out = m_resize(out,i1 + m0,j1 + n0);
   else if ((i1 + m0) > out -> m || (j1 + n0) > out -> n) 
    out = m_resize(out,((out -> m > (i1 + m0)?out -> m : (i1 + m0))),((out -> n > (j1 + n0)?out -> n : (j1 + n0))));
  for (i = 0; i <= m0 - 1; i += 1) {
    memmove((&out -> me[i1 + i][j1]),(&in -> me[i0 + i][j0]),n0 * sizeof(double ));
  }
  return out;
}
/* v_move -- copies selected pieces of a vector
	-- moves the length dim0 subvector with initial index i0
	   to the corresponding subvector of out with initial index i1
	-- out is resized if necessary */
#ifndef ANSI_C
#else

VEC *v_move(const VEC *in,int i0,int dim0,VEC *out,int i1)
#endif
{
  if (!in) 
    ev_err("copy.c",8,167,"v_move",0);
  if (i0 < 0 || dim0 < 0 || i1 < 0 || (i0 + dim0) > in -> dim) 
    ev_err("copy.c",2,170,"v_move",0);
  if (!out || (i1 + dim0) > out -> dim) 
    out = v_resize(out,i1 + dim0);
  memmove((&out -> ve[i1]),(&in -> ve[i0]),dim0 * sizeof(double ));
  return out;
}
/* mv_move -- copies selected piece of matrix to a vector
	-- moves the m0 x n0 submatrix with top-left co-ordinate (i0,j0) to
	   the subvector with initial index i1 (and length m0*n0)
	-- rows are copied contiguously
	-- out is resized if necessary */
#ifndef ANSI_C
#else

VEC *mv_move(const MAT *in,int i0,int j0,int m0,int n0,VEC *out,int i1)
#endif
{
  int dim1;
  int i;
  if (!in) 
    ev_err("copy.c",8,198,"mv_move",0);
  if (i0 < 0 || j0 < 0 || m0 < 0 || n0 < 0 || i1 < 0 || (i0 + m0) > in -> m || (j0 + n0) > in -> n) 
    ev_err("copy.c",2,201,"mv_move",0);
  dim1 = m0 * n0;
  if (!out || (i1 + dim1) > out -> dim) 
    out = v_resize(out,i1 + dim1);
  for (i = 0; i <= m0 - 1; i += 1) {
    memmove((&out -> ve[i1 + i * n0]),(&in -> me[i0 + i][j0]),n0 * sizeof(double ));
  }
  return out;
}
/* vm_move -- copies selected piece of vector to a matrix
	
-- moves the subvector with initial index i0 and length m1*n1 to
	   the m1 x n1 submatrix with top-left co-ordinate (i1,j1)
        -- copying is done by rows
	-- out is resized if necessary */
#ifndef ANSI_C
#else

MAT *vm_move(const VEC *in,int i0,MAT *out,int i1,int j1,int m1,int n1)
#endif
{
  int dim0;
  int i;
  if (!in) 
    ev_err("copy.c",8,232,"vm_move",0);
  if (i0 < 0 || i1 < 0 || j1 < 0 || m1 < 0 || n1 < 0 || (i0 + m1 * n1) > in -> dim) 
    ev_err("copy.c",2,235,"vm_move",0);
  if (!out) 
    out = m_resize(out,i1 + m1,j1 + n1);
   else 
    out = m_resize(out,(((i1 + m1) > out -> m?(i1 + m1) : out -> m)),(((j1 + n1) > out -> n?(j1 + n1) : out -> n)));
  dim0 = m1 * n1;
  for (i = 0; i <= m1 - 1; i += 1) {
    memmove((&out -> me[i1 + i][j1]),(&in -> ve[i0 + i * n1]),n1 * sizeof(double ));
  }
  return out;
}
