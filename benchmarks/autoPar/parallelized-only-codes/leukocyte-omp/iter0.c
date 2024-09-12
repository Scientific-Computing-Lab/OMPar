/**************************************************************************
**
** Copyright (C) 1993 David E. Stewart & Zbigniew Leyk, all rights reserved.
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
/* iter0.c  14/09/93 */
/* ITERATIVE METHODS - service functions */
/* functions for creating and releasing ITER structures;
   for memory information;
   for getting some values from an ITER variable;
   for changing values in an ITER variable;
   see also iter.c
*/
#include        <stdio.h>
#include	<math.h>
#include        "iter.h"
static char rcsid[] = "$Id: iter0.c,v 1.3 1995/01/30 14:50:56 des Exp $";
/* standard functions */
/* standard information */
#ifndef ANSI_C
#else

void iter_std_info(const ITER *ip,double nres,VEC *res,VEC *Bres)
#endif
{
  if (nres >= 0.0) 
#ifndef MEX
    printf(" %d. residual = %g\n",ip -> steps,nres);
   else 
#else
#endif
#ifndef MEX
    printf(" %d. residual = %g (WARNING !!! should be >= 0) \n",ip -> steps,nres);
#else
#endif
}
/* standard stopping criterion */
#ifndef ANSI_C
#else

int iter_std_stop_crit(const ITER *ip,double nres,VEC *res,VEC *Bres)
#endif
{
/* standard stopping criterium */
  if (nres <= ip -> init_res * ip -> eps) 
    return 1;
  return 0;
}
/* iter_get - create a new structure pointing to ITER */
#ifndef ANSI_C
#else

ITER *iter_get(int lenb,int lenx)
#endif
{
  ITER *ip;
  if ((ip = ((ITER *)(calloc((size_t )1,(size_t )(sizeof(ITER )))))) == ((ITER *)((void *)0))) 
    ev_err("iter0.c",3,101,"iter_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(5,0,(sizeof(ITER )),0);
    mem_numvar_list(5,1,0);
  }
/* default values */
  ip -> shared_x = 0;
  ip -> shared_b = 0;
  ip -> k = 0;
  ip -> limit = 1000;
  ip -> eps = 1e-6;
  ip -> steps = 0;
  if (lenb > 0) 
    ip -> b = v_get(lenb);
   else 
    ip -> b = ((VEC *)((void *)0));
  if (lenx > 0) 
    ip -> x = v_get(lenx);
   else 
    ip -> x = ((VEC *)((void *)0));
  ip -> Ax = ((Fun_Ax )((void *)0));
  ip -> A_par = ((void *)0);
  ip -> ATx = ((Fun_Ax )((void *)0));
  ip -> AT_par = ((void *)0);
  ip -> Bx = ((Fun_Ax )((void *)0));
  ip -> B_par = ((void *)0);
  ip -> info = iter_std_info;
  ip -> stop_crit = iter_std_stop_crit;
  ip -> init_res = 0.0;
  return ip;
}
/* iter_free - release memory */
#ifndef ANSI_C
#else

int iter_free(ITER *ip)
#endif
{
  if (ip == ((ITER *)((void *)0))) 
    return - 1;
  if (mem_info_is_on()) {
    mem_bytes_list(5,(sizeof(ITER )),0,0);
    mem_numvar_list(5,- 1,0);
  }
  if (!ip -> shared_x && ip -> x != ((void *)0)) 
    v_free(ip -> x);
  if (!ip -> shared_b && ip -> b != ((void *)0)) 
    v_free(ip -> b);
  free(((char *)ip));
  return 0;
}

int wrapped_iter_free(void *p)
{
  return iter_free((ITER *)p);
}
#ifndef ANSI_C
#else

ITER *iter_resize(ITER *ip,int new_lenb,int new_lenx)
#endif
{
  VEC *old;
  if (ip == ((ITER *)((void *)0))) 
    ev_err("iter0.c",8,172,"iter_resize",0);
  old = ip -> x;
  ip -> x = v_resize(ip -> x,new_lenx);
  if (ip -> shared_x && old != ip -> x) 
    ev_err("iter0.c",4,177,"iter_resize",1);
  old = ip -> b;
  ip -> b = v_resize(ip -> b,new_lenb);
  if (ip -> shared_b && old != ip -> b) 
    ev_err("iter0.c",4,181,"iter_resize",1);
  return ip;
}
#ifndef MEX
/* print out ip structure - for diagnostic purposes mainly */
#ifndef ANSI_C
#else

void iter_dump(FILE *fp,ITER *ip)
#endif
{
  if (ip == ((void *)0)) {
    fprintf(fp," ITER structure: NULL\n");
    return ;
  }
  fprintf(fp,"\n ITER structure:\n");
  fprintf(fp," ip->shared_x = %s, ip->shared_b = %s\n",(ip -> shared_x?"TRUE" : "FALSE"),(ip -> shared_b?"TRUE" : "FALSE"));
  fprintf(fp," ip->k = %d, ip->limit = %d, ip->steps = %d, ip->eps = %g\n",ip -> k,ip -> limit,ip -> steps,ip -> eps);
  fprintf(fp," ip->x = 0x%p, ip->b = 0x%p\n",ip -> x,ip -> b);
  fprintf(fp," ip->Ax = 0x%p, ip->A_par = 0x%p\n",ip -> Ax,ip -> A_par);
  fprintf(fp," ip->ATx = 0x%p, ip->AT_par = 0x%p\n",ip -> ATx,ip -> AT_par);
  fprintf(fp," ip->Bx = 0x%p, ip->B_par = 0x%p\n",ip -> Bx,ip -> B_par);
  fprintf(fp," ip->info = 0x%p, ip->stop_crit = 0x%p, ip->init_res = %g\n",ip -> info,ip -> stop_crit,ip -> init_res);
  fprintf(fp,"\n");
}
#endif
/* copy the structure ip1 to ip2 preserving vectors x and b of ip2
   (vectors x and b in ip2 are the same before and after iter_copy2)
   if ip2 == NULL then a new structure is created with x and b being NULL
   and other members are taken from ip1
*/
#ifndef ANSI_C
#else

ITER *iter_copy2(ITER *ip1,ITER *ip2)
#endif
{
  VEC *x;
  VEC *b;
  int shx;
  int shb;
  if (ip1 == ((ITER *)((void *)0))) 
    ev_err("iter0.c",8,234,"iter_copy2",0);
  if (ip2 == ((ITER *)((void *)0))) {
    if ((ip2 = ((ITER *)(calloc((size_t )1,(size_t )(sizeof(ITER )))))) == ((ITER *)((void *)0))) 
      ev_err("iter0.c",3,238,"iter_copy2",0);
     else if (mem_info_is_on()) {
      mem_bytes_list(5,0,(sizeof(ITER )),0);
      mem_numvar_list(5,1,0);
    }
    ip2 -> x = ip2 -> b = ((void *)0);
    ip2 -> shared_x = ip2 -> shared_x = 0;
  }
  x = ip2 -> x;
  b = ip2 -> b;
  shb = ip2 -> shared_b;
  shx = ip2 -> shared_x;
  memmove(ip2,ip1,sizeof(ITER ));
  ip2 -> x = x;
  ip2 -> b = b;
  ip2 -> shared_x = shx;
  ip2 -> shared_b = shb;
  return ip2;
}
/* copy the structure ip1 to ip2 copying also the vectors x and b */
#ifndef ANSI_C
#else

ITER *iter_copy(const ITER *ip1,ITER *ip2)
#endif
{
  VEC *x;
  VEC *b;
  if (ip1 == ((ITER *)((void *)0))) 
    ev_err("iter0.c",8,272,"iter_copy",0);
  if (ip2 == ((ITER *)((void *)0))) {
    if ((ip2 = ((ITER *)(calloc((size_t )1,(size_t )(sizeof(ITER )))))) == ((ITER *)((void *)0))) 
      ev_err("iter0.c",3,276,"iter_copy2",0);
     else if (mem_info_is_on()) {
      mem_bytes_list(5,0,(sizeof(ITER )),0);
      mem_numvar_list(5,1,0);
    }
  }
  x = ip2 -> x;
  b = ip2 -> b;
  memmove(ip2,ip1,sizeof(ITER ));
  if (ip1 -> x) 
    ip2 -> x = _v_copy((ip1 -> x),x,0);
  if (ip1 -> b) 
    ip2 -> b = _v_copy((ip1 -> b),b,0);
  ip2 -> shared_x = ip2 -> shared_b = 0;
  return ip2;
}
/*** functions to generate sparse matrices with random entries ***/
/* iter_gen_sym -- generate symmetric positive definite
   n x n matrix, 
   nrow - number of nonzero entries in a row
   */
#ifndef ANSI_C
#else

SPMAT *iter_gen_sym(int n,int nrow)
#endif
{
  SPMAT *A;
  VEC *u;
  double s1;
  int i;
  int j;
  int k;
  int k_max;
  if (nrow <= 1) 
    nrow = 2;
/* nrow should be even */
  if (nrow & 1) 
    nrow -= 1;
  A = sp_get(n,n,nrow);
  u = v_get(A -> m);
  v_zero(u);
  for (i = 0; i <= A -> m - 1; i += 1) {
    k_max = (rand() >> 8) % (nrow / 2);
    for (k = 0; k <= k_max; k += 1) {
      j = (rand() >> 8) % A -> n;
      s1 = mrand();
      sp_set_val(A,i,j,s1);
      sp_set_val(A,j,i,s1);
      u -> ve[i] += fabs(s1);
      u -> ve[j] += fabs(s1);
    }
  }
/* ensure that A is positive definite */
  for (i = 0; i <= A -> m - 1; i += 1) {
    sp_set_val(A,i,i,u -> ve[i] + 1.0);
  }
  (v_free(u) , u = ((VEC *)((void *)0)));
  return A;
}
/* iter_gen_nonsym -- generate non-symmetric m x n sparse matrix, m >= n 
   nrow - number of entries in a row;
   diag - number which is put in diagonal entries and then permuted
   (if diag is zero then 1.0 is there)
*/
#ifndef ANSI_C
#else

SPMAT *iter_gen_nonsym(int m,int n,int nrow,double diag)
#endif
{
  SPMAT *A;
  PERM *px;
  int i;
  int j;
  int k;
  int k_max;
  double s1;
  if (nrow <= 1) 
    nrow = 2;
  if (diag == 0.0) 
    diag = 1.0;
  A = sp_get(m,n,nrow);
  px = px_get(n);
  for (i = 0; i <= A -> m - 1; i += 1) {
    k_max = (rand() >> 8) % (nrow - 1);
    for (k = 0; k <= k_max; k += 1) {
      j = (rand() >> 8) % A -> n;
      s1 = mrand();
      sp_set_val(A,i,j,-s1);
    }
  }
/* to make it likely that A is nonsingular, use pivot... */
  for (i = 0; i <= 2 * A -> n - 1; i += 1) {
    j = (rand() >> 8) % A -> n;
    k = (rand() >> 8) % A -> n;
    px_transp(px,j,k);
  }
  for (i = 0; i <= A -> n - 1; i += 1) {
    sp_set_val(A,i,px -> pe[i],diag);
  }
  (px_free(px) , px = ((PERM *)((void *)0)));
  return A;
}
#if ( 0 )
/* iter_gen_nonsym -- generate non-symmetric positive definite 
   n x n sparse matrix;
   nrow - number of entries in a row
*/
#ifndef ANSI_C
#else
#endif
/* ensure that A is positive definite */
#endif
