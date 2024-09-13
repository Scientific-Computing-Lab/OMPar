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
/* Memory allocation and de-allocation for complex matrices and vectors */
#include	<stdio.h>
#include	"zmatrix.h"
#include <omp.h> 
static char rcsid[] = "$Id: zmemory.c,v 1.2 1994/04/05 02:13:14 des Exp $";
/* zv_zero -- zeros all entries of a complex vector
   -- uses __zzero__() */
#ifndef ANSI_C
#else

ZVEC *zv_zero(ZVEC *x)
#endif
{
  if (!x) 
    ev_err("zmemory.c",8,46,"zv_zero",0);
  __zzero__(x -> ve,(x -> dim));
  return x;
}
/* zm_zero -- zeros all entries of a complex matrix
   -- uses __zzero__() */
#ifndef ANSI_C
#else

ZMAT *zm_zero(ZMAT *A)
#endif
{
  int i;
  if (!A) 
    ev_err("zmemory.c",8,64,"zm_zero",0);
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
    __zzero__(A -> me[i],(A -> n));
  }
  return A;
}
/* zm_get -- gets an mxn complex matrix (in ZMAT form) */
#ifndef ANSI_C
#else

ZMAT *zm_get(int m,int n)
#endif
{
  ZMAT *matrix;
  unsigned int i;
  if (m < 0 || n < 0) 
    ev_err("zmemory.c",20,83,"zm_get",0);
  if ((matrix = ((ZMAT *)(calloc((size_t )1,(size_t )(sizeof(ZMAT )))))) == ((ZMAT *)((void *)0))) 
    ev_err("zmemory.c",3,86,"zm_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(9,0,(sizeof(ZMAT )),0);
    mem_numvar_list(9,1,0);
  }
  matrix -> m = m;
  matrix -> n = matrix -> max_n = n;
  matrix -> max_m = m;
  matrix -> max_size = (m * n);
#ifndef SEGMENTED
  if ((matrix -> base = ((complex *)(calloc((size_t )(m * n),(size_t )(sizeof(complex )))))) == ((complex *)((void *)0))) {
    free(matrix);
    ev_err("zmemory.c",3,98,"zm_get",0);
  }
   else if (mem_info_is_on()) {
    mem_bytes_list(9,0,((m * n) * sizeof(complex )),0);
  }
#else
#endif
  if ((matrix -> me = ((complex **)(calloc(m,sizeof(complex *))))) == ((complex **)((void *)0))) {
    free((matrix -> base));
    free(matrix);
    ev_err("zmemory.c",3,109,"zm_get",0);
  }
   else if (mem_info_is_on()) {
    mem_bytes_list(9,0,(m * sizeof(complex *)),0);
  }
#ifndef SEGMENTED
/* set up pointers */
  
#pragma omp parallel for private (i) firstprivate (m,n)
  for (i = 0; i <= ((unsigned int )m) - 1; i += 1) {
    matrix -> me[i] = &matrix -> base[i * n];
  }
#else
#endif
  return matrix;
}
/* zv_get -- gets a ZVEC of dimension 'dim'
   -- Note: initialized to zero */
#ifndef ANSI_C
#else

ZVEC *zv_get(int size)
#endif
{
  ZVEC *vector;
  if (size < 0) 
    ev_err("zmemory.c",20,143,"zv_get",0);
  if ((vector = ((ZVEC *)(calloc((size_t )1,(size_t )(sizeof(ZVEC )))))) == ((ZVEC *)((void *)0))) 
    ev_err("zmemory.c",3,146,"zv_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(8,0,(sizeof(ZVEC )),0);
    mem_numvar_list(8,1,0);
  }
  vector -> dim = vector -> max_dim = size;
  if ((vector -> ve = ((complex *)(calloc((size_t )size,(size_t )(sizeof(complex )))))) == ((complex *)((void *)0))) {
    free(vector);
    ev_err("zmemory.c",3,155,"zv_get",0);
  }
   else if (mem_info_is_on()) {
    mem_bytes_list(8,0,(size * sizeof(complex )),0);
  }
  return vector;
}
/* zm_free -- returns ZMAT & asoociated memory back to memory heap */
#ifndef ANSI_C
#else

int zm_free(ZMAT *mat)
#endif
{
#ifdef SEGMENTED
#endif
  if (mat == ((ZMAT *)((void *)0)) || ((int )(mat -> m)) < 0 || ((int )(mat -> n)) < 0) 
/* don't trust it */
    return - 1;
#ifndef SEGMENTED
  if (mat -> base != ((complex *)((void *)0))) {
    if (mem_info_is_on()) {
      mem_bytes_list(9,((mat -> max_m * mat -> max_n) * sizeof(complex )),0,0);
    }
    free(((char *)(mat -> base)));
  }
#else
#endif
  if (mat -> me != ((complex **)((void *)0))) {
    if (mem_info_is_on()) {
      mem_bytes_list(9,((mat -> max_m) * sizeof(complex *)),0,0);
    }
    free(((char *)(mat -> me)));
  }
  if (mem_info_is_on()) {
    mem_bytes_list(9,(sizeof(ZMAT )),0,0);
    mem_numvar_list(9,- 1,0);
  }
  free(((char *)mat));
  return 0;
}

int wrapped_zm_free(void *p)
{
  return zm_free((ZMAT *)p);
}
/* zv_free -- returns ZVEC & asoociated memory back to memory heap */
#ifndef ANSI_C
#else

int zv_free(ZVEC *vec)
#endif
{
  if (vec == ((ZVEC *)((void *)0)) || ((int )(vec -> dim)) < 0) 
/* don't trust it */
    return - 1;
  if (vec -> ve == ((complex *)((void *)0))) {
    if (mem_info_is_on()) {
      mem_bytes_list(8,(sizeof(ZVEC )),0,0);
      mem_numvar_list(8,- 1,0);
    }
    free(((char *)vec));
  }
   else {
    if (mem_info_is_on()) {
      mem_bytes_list(8,((vec -> max_dim) * sizeof(complex ) + sizeof(ZVEC )),0,0);
      mem_numvar_list(8,- 1,0);
    }
    free(((char *)(vec -> ve)));
    free(((char *)vec));
  }
  return 0;
}

int wrapped_zv_free(void *p)
{
  return zv_free((ZVEC *)p);
}
/* zm_resize -- returns the matrix A of size new_m x new_n; A is zeroed
   -- if A == NULL on entry then the effect is equivalent to m_get() */
#ifndef ANSI_C
#else

ZMAT *zm_resize(ZMAT *A,int new_m,int new_n)
#endif
{
  unsigned int i;
  unsigned int new_max_m;
  unsigned int new_max_n;
  unsigned int new_size;
  unsigned int old_m;
  unsigned int old_n;
  if (new_m < 0 || new_n < 0) 
    ev_err("zmemory.c",20,263,"zm_resize",0);
  if (!A) 
    return zm_get(new_m,new_n);
  if (new_m == A -> m && new_n == A -> n) 
    return A;
  old_m = A -> m;
  old_n = A -> n;
  if (new_m > A -> max_m) {
/* re-allocate A->me */
    if (mem_info_is_on()) {
      mem_bytes_list(9,((A -> max_m) * sizeof(complex *)),(new_m * sizeof(complex *)),0);
    }
    A -> me = A -> me = ((complex **)((A -> me?realloc(((char *)(A -> me)),(size_t )(new_m * sizeof(complex *))) : calloc((size_t )new_m,(size_t )(sizeof(complex *))))));
    if (!A -> me) 
      ev_err("zmemory.c",3,281,"zm_resize",0);
  }
  new_max_m = (new_m > A -> max_m?new_m : A -> max_m);
  new_max_n = (new_n > A -> max_n?new_n : A -> max_n);
#ifndef SEGMENTED
  new_size = new_max_m * new_max_n;
  if (new_size > A -> max_size) {
/* re-allocate A->base */
    if (mem_info_is_on()) {
      mem_bytes_list(9,((A -> max_m * A -> max_n) * sizeof(complex )),(new_size * sizeof(complex )),0);
    }
    A -> base = A -> base = ((complex *)((A -> base?realloc(((char *)(A -> base)),(size_t )(new_size * sizeof(complex ))) : calloc((size_t )new_size,(size_t )(sizeof(complex ))))));
    if (!A -> base) 
      ev_err("zmemory.c",3,297,"zm_resize",0);
    A -> max_size = new_size;
  }
/* now set up A->me[i] */
  
#pragma omp parallel for private (i)
  for (i = 0; i <= ((unsigned int )new_m) - 1; i += 1) {
    A -> me[i] = &A -> base[i * new_n];
  }
/* now shift data in matrix */
  if (old_n > new_n) {
    for (i = 1; i <= ((old_m > ((unsigned int )new_m)?((unsigned int )new_m) : old_m)) - 1; i += 1) {
      memmove(((char *)(&A -> base[i * new_n])),((char *)(&A -> base[i * old_n])),sizeof(complex ) * new_n);
    }
  }
   else if (old_n < new_n) {
    for (i = ((old_m > new_m?new_m : old_m)) - 1; i >= ((unsigned int )0) + 1; i += -1) {
/* copy & then zero extra space */
      memmove(((char *)(&A -> base[i * new_n])),((char *)(&A -> base[i * old_n])),sizeof(complex ) * old_n);
      __zzero__(&A -> base[i * new_n + old_n],(new_n - old_n));
    }
    __zzero__(&A -> base[old_n],(new_n - old_n));
    A -> max_n = new_n;
  }
/* zero out the new rows.. */
  for (i = old_m; i <= ((unsigned int )new_m) - 1; i += 1) {
    __zzero__(&A -> base[i * new_n],new_n);
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
/* zv_resize -- returns the (complex) vector x with dim new_dim
   -- x is set to the zero vector */
#ifndef ANSI_C
#else

ZVEC *zv_resize(ZVEC *x,int new_dim)
#endif
{
  if (new_dim < 0) 
    ev_err("zmemory.c",20,400,"zv_resize",0);
  if (!x) 
    return zv_get(new_dim);
  if (new_dim == x -> dim) 
    return x;
/* assume that it's from sub_zvec */
  if (x -> max_dim == 0) 
    return zv_get(new_dim);
  if (new_dim > x -> max_dim) {
    if (mem_info_is_on()) {
      mem_bytes_list(8,((x -> max_dim) * sizeof(complex )),(new_dim * sizeof(complex )),0);
    }
    x -> ve = x -> ve = ((complex *)((x -> ve?realloc(((char *)(x -> ve)),(size_t )(new_dim * sizeof(complex ))) : calloc((size_t )new_dim,(size_t )(sizeof(complex ))))));
    if (!x -> ve) 
      ev_err("zmemory.c",3,420,"zv_resize",0);
    x -> max_dim = new_dim;
  }
  if (new_dim > x -> dim) 
    __zzero__(&x -> ve[x -> dim],(new_dim - x -> dim));
  x -> dim = new_dim;
  return x;
}
/* varying arguments */
#ifdef ANSI_C
#include <stdarg.h>
/* To allocate memory to many arguments. 
   The function should be called:
   zv_get_vars(dim,&x,&y,&z,...,NULL);
   where 
     int dim;
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     dim is the length of vectors x,y,z,...
     returned value is equal to the number of allocated variables
     Other gec_... functions are similar.
*/

int zv_get_vars(int dim,... )
{
  va_list ap;
  int i = 0;
  ZVEC **par;
  __builtin_va_start(ap,dim);
  while(par = ((ZVEC **)(sizeof(ZVEC **)))){
/* NULL ends the list*/
     *par = zv_get(dim);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int zm_get_vars(int m,int n,... )
{
  va_list ap;
  int i = 0;
  ZMAT **par;
  __builtin_va_start(ap,n);
  while(par = ((ZMAT **)(sizeof(ZMAT **)))){
/* NULL ends the list*/
     *par = zm_get(m,n);
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
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     rdim is the resized length of vectors x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
     Other *_resize_list() functions are similar.
*/

int zv_resize_vars(int new_dim,... )
{
  va_list ap;
  int i = 0;
  ZVEC **par;
  __builtin_va_start(ap,new_dim);
  while(par = ((ZVEC **)(sizeof(ZVEC **)))){
/* NULL ends the list*/
     *par = zv_resize( *par,new_dim);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int zm_resize_vars(int m,int n,... )
{
  va_list ap;
  int i = 0;
  ZMAT **par;
  __builtin_va_start(ap,n);
  while(par = ((ZMAT **)(sizeof(ZMAT **)))){
/* NULL ends the list*/
     *par = zm_resize( *par,m,n);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}
/* To deallocate memory for many arguments. 
   The function should be called:
   v_free_vars(&x,&y,&z,...,NULL);
   where 
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     There must be at least one not NULL argument.
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
     Other *_free_list() functions are similar.
*/

int zv_free_vars(ZVEC **pv,... )
{
  va_list ap;
  int i = 1;
  ZVEC **par;
  zv_free( *pv);
   *pv = ((ZVEC *)((void *)0));
  __builtin_va_start(ap,pv);
  while(par = ((ZVEC **)(sizeof(ZVEC **)))){
/* NULL ends the list*/
    zv_free( *par);
     *par = ((ZVEC *)((void *)0));
    i++;
  }
  __builtin_va_end(ap);
  return i;
}

int zm_free_vars(ZMAT **va,... )
{
  va_list ap;
  int i = 1;
  ZMAT **par;
  zm_free( *va);
   *va = ((ZMAT *)((void *)0));
  __builtin_va_start(ap,va);
  while(par = ((ZMAT **)(sizeof(ZMAT **)))){
/* NULL ends the list*/
    zm_free( *par);
     *par = ((ZMAT *)((void *)0));
    i++;
  }
  __builtin_va_end(ap);
  return i;
}
#elif VARARGS
#include <varargs.h>
/* To allocate memory to many arguments. 
   The function should be called:
   v_get_vars(dim,&x,&y,&z,...,NULL);
   where 
     int dim;
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     dim is the length of vectors x,y,z,...
     returned value is equal to the number of allocated variables
     Other gec_... functions are similar.
*/
/* NULL ends the list*/
/* NULL ends the list*/
/* To resize memory for many arguments. 
   The function should be called:
   v_resize_vars(new_dim,&x,&y,&z,...,NULL);
   where 
     int new_dim;
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     rdim is the resized length of vectors x,y,z,...
     returned value is equal to the number of allocated variables.
     If one of x,y,z,.. arguments is NULL then memory is allocated to this 
     argument. 
     Other *_resize_list() functions are similar.
*/
/* NULL ends the list*/
/* NULL ends the list*/
/* To deallocate memory for many arguments. 
   The function should be called:
   v_free_vars(&x,&y,&z,...,NULL);
   where 
     ZVEC *x, *y, *z,...;
     The last argument should be NULL ! 
     There must be at least one not NULL argument.
     returned value is equal to the number of allocated variables.
     Returned value of x,y,z,.. is VNULL.
     Other *_free_list() functions are similar.
*/
/* NULL ends the list*/
/* NULL ends the list*/
#endif
