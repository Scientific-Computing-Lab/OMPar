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
/* memory.c 1.3 11/25/87 */
#include 	"matrix.h"
#include <omp.h> 
static char rcsid[] = "$Id: memory.c,v 1.13 1994/04/05 02:10:37 des Exp $";

int wrapped_m_free(void *p)
{
  return m_free((MAT *)p);
}

int wrapped_px_free(void *p)
{
  return px_free((PERM *)p);
}

int wrapped_bd_free(void *p)
{
  return bd_free((BAND *)p);
}

int wrapped_v_free(void *p)
{
  return v_free((VEC *)p);
}

int wrapped_iv_free(void *p)
{
  return iv_free((IVEC *)p);
}
/* m_get -- gets an mxn matrix (in MAT form) by dynamic memory allocation
	-- normally ALL matrices should be obtained this way
	-- if either m or n is negative this will raise an error
	-- note that 0 x n and m x 0 matrices can be created */
#ifndef ANSI_C
#else

MAT *m_get(int m,int n)
#endif
{
  MAT *matrix;
  int i;
  if (m < 0 || n < 0) 
    ev_err("memory.c",20,57,"m_get",0);
  if ((matrix = ((MAT *)(calloc((size_t )1,(size_t )(sizeof(MAT )))))) == ((MAT *)((void *)0))) 
    ev_err("memory.c",3,60,"m_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(0,0,(sizeof(MAT )),0);
    mem_numvar_list(0,1,0);
  }
  matrix -> m = m;
  matrix -> n = matrix -> max_n = n;
  matrix -> max_m = m;
  matrix -> max_size = (m * n);
#ifndef SEGMENTED
  if ((matrix -> base = ((double *)(calloc((size_t )(m * n),(size_t )(sizeof(double )))))) == ((double *)((void *)0))) {
    free(matrix);
    ev_err("memory.c",3,72,"m_get",0);
  }
   else if (mem_info_is_on()) {
    mem_bytes_list(0,0,((m * n) * sizeof(double )),0);
  }
#else
#endif
  if ((matrix -> me = ((double **)(calloc(m,sizeof(double *))))) == ((double **)((void *)0))) {
    free((matrix -> base));
    free(matrix);
    ev_err("memory.c",3,83,"m_get",0);
  }
   else if (mem_info_is_on()) {
    mem_bytes_list(0,0,(m * sizeof(double *)),0);
  }
#ifndef SEGMENTED
/* set up pointers */
  
#pragma omp parallel for private (i) firstprivate (m,n)
  for (i = 0; i <= m - 1; i += 1) {
    matrix -> me[i] = &matrix -> base[i * n];
  }
#else
#endif
  return matrix;
}
/* px_get -- gets a PERM of given 'size' by dynamic memory allocation
	-- Note: initialized to the identity permutation
	-- the permutation is on the set {0,1,2,...,size-1} */
#ifndef ANSI_C
#else

PERM *px_get(int size)
#endif
{
  PERM *permute;
  int i;
  if (size < 0) 
    ev_err("memory.c",20,120,"px_get",0);
  if ((permute = ((PERM *)(calloc((size_t )1,(size_t )(sizeof(PERM )))))) == ((PERM *)((void *)0))) 
    ev_err("memory.c",3,123,"px_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(2,0,(sizeof(PERM )),0);
    mem_numvar_list(2,1,0);
  }
  permute -> size = permute -> max_size = size;
  if ((permute -> pe = ((unsigned int *)(calloc((size_t )size,(size_t )(sizeof(unsigned int )))))) == ((unsigned int *)((void *)0))) 
    ev_err("memory.c",3,131,"px_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(2,0,(size * sizeof(unsigned int )),0);
  }
  
#pragma omp parallel for private (i) firstprivate (size)
  for (i = 0; i <= size - 1; i += 1) {
    permute -> pe[i] = i;
  }
  return permute;
}
/* v_get -- gets a VEC of dimension 'size'
   -- Note: initialized to zero */
#ifndef ANSI_C
#else

VEC *v_get(int size)
#endif
{
  VEC *vector;
  if (size < 0) 
    ev_err("memory.c",20,154,"v_get",0);
  if ((vector = ((VEC *)(calloc((size_t )1,(size_t )(sizeof(VEC )))))) == ((VEC *)((void *)0))) 
    ev_err("memory.c",3,157,"v_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(3,0,(sizeof(VEC )),0);
    mem_numvar_list(3,1,0);
  }
  vector -> dim = vector -> max_dim = size;
  if ((vector -> ve = ((double *)(calloc((size_t )size,(size_t )(sizeof(double )))))) == ((double *)((void *)0))) {
    free(vector);
    ev_err("memory.c",3,167,"v_get",0);
  }
   else if (mem_info_is_on()) {
    mem_bytes_list(3,0,(size * sizeof(double )),0);
  }
  return vector;
}
/* m_free -- returns MAT & asoociated memory back to memory heap */
#ifndef ANSI_C
#else

int m_free(MAT *mat)
#endif
{
#ifdef SEGMENTED
#endif
  if (mat == ((MAT *)((void *)0)) || ((int )(mat -> m)) < 0 || ((int )(mat -> n)) < 0) 
/* don't trust it */
    return - 1;
#ifndef SEGMENTED
  if (mat -> base != ((double *)((void *)0))) {
    if (mem_info_is_on()) {
      mem_bytes_list(0,((mat -> max_m * mat -> max_n) * sizeof(double )),0,0);
    }
    free(((char *)(mat -> base)));
  }
#else
#endif
  if (mat -> me != ((double **)((void *)0))) {
    if (mem_info_is_on()) {
      mem_bytes_list(0,((mat -> max_m) * sizeof(double *)),0,0);
    }
    free(((char *)(mat -> me)));
  }
  if (mem_info_is_on()) {
    mem_bytes_list(0,(sizeof(MAT )),0,0);
    mem_numvar_list(0,- 1,0);
  }
  free(((char *)mat));
  return 0;
}
/* px_free -- returns PERM & asoociated memory back to memory heap */
#ifndef ANSI_C
#else

int px_free(PERM *px)
#endif
{
  if (px == ((PERM *)((void *)0)) || ((int )(px -> size)) < 0) 
/* don't trust it */
    return - 1;
  if (px -> pe == ((unsigned int *)((void *)0))) {
    if (mem_info_is_on()) {
      mem_bytes_list(2,(sizeof(PERM )),0,0);
      mem_numvar_list(2,- 1,0);
    }
    free(((char *)px));
  }
   else {
    if (mem_info_is_on()) {
      mem_bytes_list(2,(sizeof(PERM ) + (px -> max_size) * sizeof(unsigned int )),0,0);
      mem_numvar_list(2,- 1,0);
    }
    free(((char *)(px -> pe)));
    free(((char *)px));
  }
  return 0;
}
/* v_free -- returns VEC & asoociated memory back to memory heap */
#ifndef ANSI_C
#else

int v_free(VEC *vec)
#endif
{
  if (vec == ((VEC *)((void *)0)) || ((int )(vec -> dim)) < 0) 
/* don't trust it */
    return - 1;
  if (vec -> ve == ((double *)((void *)0))) {
    if (mem_info_is_on()) {
      mem_bytes_list(3,(sizeof(VEC )),0,0);
      mem_numvar_list(3,- 1,0);
    }
    free(((char *)vec));
  }
   else {
    if (mem_info_is_on()) {
      mem_bytes_list(3,(sizeof(VEC ) + (vec -> max_dim) * sizeof(double )),0,0);
      mem_numvar_list(3,- 1,0);
    }
    free(((char *)(vec -> ve)));
    free(((char *)vec));
  }
  return 0;
}
/* m_resize -- returns the matrix A of size new_m x new_n; A is zeroed
   -- if A == NULL on entry then the effect is equivalent to m_get() */
#ifndef ANSI_C
#else

MAT *m_resize(MAT *A,int new_m,int new_n)
#endif
{
  int i;
  int new_max_m;
  int new_max_n;
  int new_size;
  int old_m;
  int old_n;
  if (new_m < 0 || new_n < 0) 
    ev_err("memory.c",20,309,"m_resize",0);
  if (!A) 
    return m_get(new_m,new_n);
/* nothing was changed */
  if (new_m == A -> m && new_n == A -> n) 
    return A;
  old_m = (A -> m);
  old_n = (A -> n);
  if (new_m > A -> max_m) {
/* re-allocate A->me */
    if (mem_info_is_on()) {
      mem_bytes_list(0,((A -> max_m) * sizeof(double *)),(new_m * sizeof(double *)),0);
    }
    A -> me = A -> me = ((double **)((A -> me?realloc(((char *)(A -> me)),(size_t )(new_m * sizeof(double *))) : calloc((size_t )new_m,(size_t )(sizeof(double *))))));
    if (!A -> me) 
      ev_err("memory.c",3,328,"m_resize",0);
  }
  new_max_m = ((new_m > A -> max_m?new_m : A -> max_m));
  new_max_n = ((new_n > A -> max_n?new_n : A -> max_n));
#ifndef SEGMENTED
  new_size = new_max_m * new_max_n;
  if (new_size > A -> max_size) {
/* re-allocate A->base */
    if (mem_info_is_on()) {
      mem_bytes_list(0,((A -> max_m * A -> max_n) * sizeof(double )),(new_size * sizeof(double )),0);
    }
    A -> base = A -> base = ((double *)((A -> base?realloc(((char *)(A -> base)),(size_t )(new_size * sizeof(double ))) : calloc((size_t )new_size,(size_t )(sizeof(double ))))));
    if (!A -> base) 
      ev_err("memory.c",3,344,"m_resize",0);
    A -> max_size = new_size;
  }
/* now set up A->me[i] */
  
#pragma omp parallel for private (i)
  for (i = 0; i <= new_m - 1; i += 1) {
    A -> me[i] = &A -> base[i * new_n];
  }
/* now shift data in matrix */
  if (old_n > new_n) {
    for (i = 1; i <= ((old_m > new_m?new_m : old_m)) - 1; i += 1) {
      memmove(((char *)(&A -> base[i * new_n])),((char *)(&A -> base[i * old_n])),sizeof(double ) * new_n);
    }
  }
   else if (old_n < new_n) {
    for (i = ((int )((old_m > new_m?new_m : old_m))) - 1; i >= 1; i += -1) {
/* copy & then zero extra space */
      memmove(((char *)(&A -> base[i * new_n])),((char *)(&A -> base[i * old_n])),sizeof(double ) * old_n);
      __zero__(&A -> base[i * new_n + old_n],new_n - old_n);
    }
    __zero__(&A -> base[old_n],new_n - old_n);
    A -> max_n = new_n;
  }
/* zero out the new rows.. */
  for (i = old_m; i <= new_m - 1; i += 1) {
    __zero__(&A -> base[i * new_n],new_n);
  }
#else
/* zero out the new rows.. */
#endif
  A -> max_m = new_max_m;
  A -> max_n = new_max_n;
  A -> max_size = A -> max_m * A -> max_n;
  A -> m = new_m;
  A -> n = new_n;
  return A;
}
/* px_resize -- returns the permutation px with size new_size
   -- px is set to the identity permutation */
#ifndef ANSI_C
#else

PERM *px_resize(PERM *px,int new_size)
#endif
{
  int i;
  if (new_size < 0) 
    ev_err("memory.c",20,449,"px_resize",0);
  if (!px) 
    return px_get(new_size);
/* nothing is changed */
  if (new_size == px -> size) 
    return px;
  if (new_size > px -> max_size) {
    if (mem_info_is_on()) {
      mem_bytes_list(2,((px -> max_size) * sizeof(unsigned int )),(new_size * sizeof(unsigned int )),0);
    }
    px -> pe = px -> pe = ((unsigned int *)((px -> pe?realloc(((char *)(px -> pe)),(size_t )(new_size * sizeof(unsigned int ))) : calloc((size_t )new_size,(size_t )(sizeof(unsigned int ))))));
    if (!px -> pe) 
      ev_err("memory.c",3,466,"px_resize",0);
    px -> max_size = new_size;
  }
  if (px -> size <= new_size) {
/* extend permutation */
    
#pragma omp parallel for private (i)
    for (i = (px -> size); i <= new_size - 1; i += 1) {
      px -> pe[i] = i;
    }
  }
   else {
    
#pragma omp parallel for private (i)
    for (i = 0; i <= new_size - 1; i += 1) {
      px -> pe[i] = i;
    }
  }
  px -> size = new_size;
  return px;
}
/* v_resize -- returns the vector x with dim new_dim
   -- x is set to the zero vector */
#ifndef ANSI_C
#else

VEC *v_resize(VEC *x,int new_dim)
#endif
{
  if (new_dim < 0) 
    ev_err("memory.c",20,494,"v_resize",0);
  if (!x) 
    return v_get(new_dim);
/* nothing is changed */
  if (new_dim == x -> dim) 
    return x;
  if (x -> max_dim == 0) 
/* assume that it's from sub_vec */
    return v_get(new_dim);
  if (new_dim > x -> max_dim) {
    if (mem_info_is_on()) {
      mem_bytes_list(3,((x -> max_dim) * sizeof(double )),(new_dim * sizeof(double )),0);
    }
    x -> ve = x -> ve = ((double *)((x -> ve?realloc(((char *)(x -> ve)),(size_t )(new_dim * sizeof(double ))) : calloc((size_t )new_dim,(size_t )(sizeof(double ))))));
    if (!x -> ve) 
      ev_err("memory.c",3,515,"v_resize",0);
    x -> max_dim = new_dim;
  }
  if (new_dim > x -> dim) 
    __zero__(&x -> ve[x -> dim],(new_dim - x -> dim));
  x -> dim = new_dim;
  return x;
}
/* Varying number of arguments */
/* other functions of this type are in sparse.c and zmemory.c */
#ifdef ANSI_C
/* To allocate memory to many arguments. 
   The function should be called:
   v_get_vars(dim,&x,&y,&z,...,NULL);
   where 
     int dim;
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     dim is the length of vectors x,y,z,...
     returned value is equal to the number of allocated variables
     Other gec_... functions are similar.
*/

int v_get_vars(int dim,... )
{
  va_list ap;
  int i = 0;
  VEC **par;
  __builtin_va_start(ap,dim);
  while(par = ((VEC **)(sizeof(VEC **)))){
/* NULL ends the list*/
     *par = v_get(dim);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int iv_get_vars(int dim,... )
{
  va_list ap;
  int i = 0;
  IVEC **par;
  __builtin_va_start(ap,dim);
  while(par = ((IVEC **)(sizeof(IVEC **)))){
/* NULL ends the list*/
     *par = iv_get(dim);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int m_get_vars(int m,int n,... )
{
  va_list ap;
  int i = 0;
  MAT **par;
  __builtin_va_start(ap,n);
  while(par = ((MAT **)(sizeof(MAT **)))){
/* NULL ends the list*/
     *par = m_get(m,n);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int px_get_vars(int dim,... )
{
  va_list ap;
  int i = 0;
  PERM **par;
  __builtin_va_start(ap,dim);
  while(par = ((PERM **)(sizeof(PERM **)))){
/* NULL ends the list*/
     *par = px_get(dim);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}
/* To resize memory for many arguments. 
   The function should be called:
   v_resize_vars(new_dim,&x,&y,&z,...,NULL);
   where 
     int new_dim;
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     rdim is the resized length of vectors x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
     Other *_resize_list() functions are similar.
*/

int v_resize_vars(int new_dim,... )
{
  va_list ap;
  int i = 0;
  VEC **par;
  __builtin_va_start(ap,new_dim);
  while(par = ((VEC **)(sizeof(VEC **)))){
/* NULL ends the list*/
     *par = v_resize( *par,new_dim);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int iv_resize_vars(int new_dim,... )
{
  va_list ap;
  int i = 0;
  IVEC **par;
  __builtin_va_start(ap,new_dim);
  while(par = ((IVEC **)(sizeof(IVEC **)))){
/* NULL ends the list*/
     *par = iv_resize( *par,new_dim);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int m_resize_vars(int m,int n,... )
{
  va_list ap;
  int i = 0;
  MAT **par;
  __builtin_va_start(ap,n);
  while(par = ((MAT **)(sizeof(MAT **)))){
/* NULL ends the list*/
     *par = m_resize( *par,m,n);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int px_resize_vars(int new_dim,... )
{
  va_list ap;
  int i = 0;
  PERM **par;
  __builtin_va_start(ap,new_dim);
  while(par = ((PERM **)(sizeof(PERM **)))){
/* NULL ends the list*/
     *par = px_resize( *par,new_dim);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}
/* To deallocate memory for many arguments. 
   The function should be called:
   v_free_vars(&x,&y,&z,...,NULL);
   where 
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     There must be at least one not NULL argument.
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
     Other *_free_list() functions are similar.
*/

int v_free_vars(VEC **pv,... )
{
  va_list ap;
  int i = 1;
  VEC **par;
  v_free( *pv);
   *pv = ((VEC *)((void *)0));
  __builtin_va_start(ap,pv);
  while(par = ((VEC **)(sizeof(VEC **)))){
/* NULL ends the list*/
    v_free( *par);
     *par = ((VEC *)((void *)0));
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int iv_free_vars(IVEC **ipv,... )
{
  va_list ap;
  int i = 1;
  IVEC **par;
  iv_free( *ipv);
   *ipv = ((IVEC *)((void *)0));
  __builtin_va_start(ap,ipv);
  while(par = ((IVEC **)(sizeof(IVEC **)))){
/* NULL ends the list*/
    iv_free( *par);
     *par = ((IVEC *)((void *)0));
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int px_free_vars(PERM **vpx,... )
{
  va_list ap;
  int i = 1;
  PERM **par;
  px_free( *vpx);
   *vpx = ((PERM *)((void *)0));
  __builtin_va_start(ap,vpx);
  while(par = ((PERM **)(sizeof(PERM **)))){
/* NULL ends the list*/
    px_free( *par);
     *par = ((PERM *)((void *)0));
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int m_free_vars(MAT **va,... )
{
  va_list ap;
  int i = 1;
  MAT **par;
  m_free( *va);
   *va = ((MAT *)((void *)0));
  __builtin_va_start(ap,va);
  while(par = ((MAT **)(sizeof(MAT **)))){
/* NULL ends the list*/
    m_free( *par);
     *par = ((MAT *)((void *)0));
    i++;
  }
  __builtin_va_end(ap);
  return i;
}
#elif VARARGS
/* old varargs is used */
/* To allocate memory to many arguments. 
   The function should be called:
   v_get_vars(dim,&x,&y,&z,...,VNULL);
   where 
     int dim;
     VEC *x, *y, *z,...;
     The last argument should be VNULL ! 
     dim is the length of vectors x,y,z,...
*/
/* NULL ends the list*/
/* NULL ends the list*/
/* NULL ends the list*/
/* NULL ends the list*/
/* To resize memory for many arguments. 
   The function should be called:
   v_resize_vars(new_dim,&x,&y,&z,...,NULL);
   where 
     int new_dim;
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     rdim is the resized length of vectors x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
     Other *_resize_list() functions are similar.
*/
/* NULL ends the list*/
/* NULL ends the list*/
/* NULL ends the list*/
/* NULL ends the list*/
/* To deallocate memory for many arguments. 
   The function should be called:
   v_free_vars(&x,&y,&z,...,NULL);
   where 
     VEC *x, *y, *z,...;
     The last argument should be NULL ! 
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
     Other *_free_list() functions are similar.
*/
/* NULL ends the list*/
/* NULL ends the list*/
/* NULL ends the list*/
/* NULL ends the list*/
#endif /* VARARGS */
