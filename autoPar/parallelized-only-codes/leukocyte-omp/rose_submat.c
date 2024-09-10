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
/* 1.2 submat.c 11/25/87 */
#include	<stdio.h>
#include	"matrix.h"
#include <omp.h> 
static char rcsid[] = "$Id: submat.c,v 1.2 1994/01/13 05:28:12 des Exp $";
/* get_col -- gets a specified column of a matrix and retruns it as a vector */
#ifndef ANSI_C
#else

VEC *get_col(const MAT *mat,unsigned int col,VEC *vec)
#endif
{
  unsigned int i;
  if (mat == ((MAT *)((void *)0))) 
    ev_err("submat.c",8,48,"get_col",0);
  if (col >= mat -> n) 
    ev_err("submat.c",10,50,"get_col",0);
  if (vec == ((VEC *)((void *)0)) || vec -> dim < mat -> m) 
    vec = v_resize(vec,(mat -> m));
  
#pragma omp parallel for private (i) firstprivate (col)
  for (i = 0; i <= mat -> m - 1; i += 1) {
    vec -> ve[i] = mat -> me[i][col];
  }
  return vec;
}
/* get_row -- gets a specified row of a matrix and retruns it as a vector */
#ifndef ANSI_C
#else

VEC *get_row(const MAT *mat,unsigned int row,VEC *vec)
#endif
{
  unsigned int i;
  if (mat == ((MAT *)((void *)0))) 
    ev_err("submat.c",8,73,"get_row",0);
  if (row >= mat -> m) 
    ev_err("submat.c",10,75,"get_row",0);
  if (vec == ((VEC *)((void *)0)) || vec -> dim < mat -> n) 
    vec = v_resize(vec,(mat -> n));
  
#pragma omp parallel for private (i) firstprivate (row)
  for (i = 0; i <= mat -> n - 1; i += 1) {
    vec -> ve[i] = mat -> me[row][i];
  }
  return vec;
}
/* _set_col -- sets column of matrix to values given in vec (in situ)
	-- that is, mat(i0:lim,col) <- vec(i0:lim) */
#ifndef ANSI_C
#else

MAT *_set_col(MAT *mat,unsigned int col,const VEC *vec,unsigned int i0)
#endif
{
  unsigned int i;
  unsigned int lim;
  if (mat == ((MAT *)((void *)0)) || vec == ((VEC *)((void *)0))) 
    ev_err("submat.c",8,99,"_set_col",0);
  if (col >= mat -> n) 
    ev_err("submat.c",10,101,"_set_col",0);
  lim = (mat -> m > vec -> dim?vec -> dim : mat -> m);
  
#pragma omp parallel for private (i) firstprivate (col,lim)
  for (i = i0; i <= lim - 1; i += 1) {
    mat -> me[i][col] = vec -> ve[i];
  }
  return mat;
}
/* _set_row -- sets row of matrix to values given in vec (in situ) */
#ifndef ANSI_C
#else

MAT *_set_row(MAT *mat,unsigned int row,const VEC *vec,unsigned int j0)
#endif
{
  unsigned int j;
  unsigned int lim;
  if (mat == ((MAT *)((void *)0)) || vec == ((VEC *)((void *)0))) 
    ev_err("submat.c",8,122,"_set_row",0);
  if (row >= mat -> m) 
    ev_err("submat.c",10,124,"_set_row",0);
  lim = (mat -> n > vec -> dim?vec -> dim : mat -> n);
  
#pragma omp parallel for private (j) firstprivate (row,lim)
  for (j = j0; j <= lim - 1; j += 1) {
    mat -> me[row][j] = vec -> ve[j];
  }
  return mat;
}
/* sub_mat -- returns sub-matrix of old which is formed by the rectangle
   from (row1,col1) to (row2,col2)
   -- Note: storage is shared so that altering the "new"
   matrix will alter the "old" matrix */
#ifndef ANSI_C
#else

MAT *sub_mat(const MAT *old,unsigned int row1,unsigned int col1,unsigned int row2,unsigned int col2,MAT *new_output)
#endif
{
  unsigned int i;
  if (old == ((MAT *)((void *)0))) 
    ev_err("submat.c",8,147,"sub_mat",0);
  if (row1 > row2 || col1 > col2 || row2 >= old -> m || col2 >= old -> n) 
    ev_err("submat.c",10,149,"sub_mat",0);
  if (new_output == ((MAT *)((void *)0)) || new_output -> m < row2 - row1 + 1) {
    new_output = ((MAT *)(calloc((size_t )1,(size_t )(sizeof(MAT )))));
    new_output -> me = ((double **)(calloc((size_t )(row2 - row1 + 1),(size_t )(sizeof(double *)))));
    if (new_output == ((MAT *)((void *)0)) || new_output -> me == ((double **)((void *)0))) 
      ev_err("submat.c",3,155,"sub_mat",0);
     else if (mem_info_is_on()) {
      mem_bytes_list(0,0,(sizeof(MAT ) + (row2 - row1 + 1) * sizeof(double *)),0);
    }
  }
  new_output -> m = row2 - row1 + 1;
  new_output -> n = col2 - col1 + 1;
  new_output -> base = ((double *)((void *)0));
  
#pragma omp parallel for private (i) firstprivate (row1,col1)
  for (i = 0; i <= new_output -> m - 1; i += 1) {
    new_output -> me[i] = old -> me[i + row1] + col1;
  }
  return new_output;
}
/* sub_vec -- returns sub-vector which is formed by the elements i1 to i2
   -- as for sub_mat, storage is shared */
#ifndef ANSI_C
#else

VEC *sub_vec(const VEC *old,int i1,int i2,VEC *new_output)
#endif
{
  if (old == ((VEC *)((void *)0))) 
    ev_err("submat.c",8,186,"sub_vec",0);
  if (i1 > i2 || old -> dim < i2) 
    ev_err("submat.c",10,188,"sub_vec",0);
  if (new_output == ((VEC *)((void *)0))) 
    new_output = ((VEC *)(calloc((size_t )1,(size_t )(sizeof(VEC )))));
  if (new_output == ((VEC *)((void *)0))) 
    ev_err("submat.c",3,193,"sub_vec",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(3,0,(sizeof(VEC )),0);
  }
  new_output -> dim = (i2 - i1 + 1);
  new_output -> ve = &old -> ve[i1];
  return new_output;
}
