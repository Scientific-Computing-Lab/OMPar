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
  Sparse rows package
  See also: sparse.h, matrix.h
  */
#include	<stdio.h>
#include	<math.h>
#include        <stdlib.h>
#include	"sparse.h"
static char rcsid[] = "$Id: sprow.c,v 1.1 1994/01/13 05:35:36 des Exp $";
#define	MINROWLEN	10
#ifndef MEX
/* sprow_dump - prints relevant information about the sparse row r */
#ifndef ANSI_C
#else

void sprow_dump(FILE *fp,const SPROW *r)
#endif
{
  int j_idx;
  row_elt *elts;
  fprintf(fp,"SparseRow dump:\n");
  if (!r) {
    fprintf(fp,"*** NULL row ***\n");
    return ;
  }
  fprintf(fp,"row: len = %d, maxlen = %d, diag idx = %d\n",r -> len,r -> maxlen,r -> diag);
  fprintf(fp,"element list @ 0x%lx\n",(long )(r -> elt));
  if (!r -> elt) {
    fprintf(fp,"*** NULL element list ***\n");
    return ;
  }
  elts = r -> elt;
  for (j_idx = 0; j_idx <= r -> len - 1; (j_idx++ , elts++)) 
    fprintf(fp,"Col: %d, Val: %g, nxt_row = %d, nxt_idx = %d\n",elts -> col,elts -> val,elts -> nxt_row,elts -> nxt_idx);
  fprintf(fp,"\n");
}
#endif /* MEX */
/* sprow_idx -- get index into row for a given column in a given row
   -- return -1 on error
   -- return -(idx+2) where idx is index to insertion point */
#ifndef ANSI_C
#else

int sprow_idx(const SPROW *r,int col)
#endif
{
  int lo;
  int hi;
  int mid;
  int tmp;
  row_elt *r_elt;
/*******************************************
     if ( r == (SPROW *)NULL )
     return -1;
     if ( col < 0 )
     return -1;
     *******************************************/
  r_elt = r -> elt;
  if (r -> len <= 0) 
    return - 2;
/* try the hint */
/* if ( hint >= 0 && hint < r->len && r_elt[hint].col == col )
      return hint; */
/* otherwise use binary search... */
/* code from K&R Ch. 6, p. 125 */
  lo = 0;
  hi = r -> len - 1;
  mid = lo;
  while(lo <= hi){
    mid = (hi + lo) / 2;
    if ((tmp = r_elt[mid] . col - col) > 0) 
      hi = mid - 1;
     else if (tmp < 0) 
      lo = mid + 1;
     else 
/* tmp == 0 */
      return mid;
  }
  tmp = r_elt[mid] . col - col;
  if (tmp > 0) 
    return -(mid + 2);
   else 
/* insert at mid   */
/* tmp < 0 */
    return -(mid + 3);
/* insert at mid+1 */
}
/* sprow_get -- gets, initialises and returns a SPROW structure
   -- max. length is maxlen */
#ifndef ANSI_C
#else

SPROW *sprow_get(int maxlen)
#endif
{
  SPROW *r;
  if (maxlen < 0) 
    ev_err("sprow.c",20,139,"sprow_get",0);
  r = ((SPROW *)(calloc((size_t )1,(size_t )(sizeof(SPROW )))));
  if (!r) 
    ev_err("sprow.c",3,143,"sprow_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(6,0,(sizeof(SPROW )),0);
    mem_numvar_list(6,1,0);
  }
  r -> elt = ((row_elt *)(calloc((size_t )maxlen,(size_t )(sizeof(row_elt )))));
  if (!r -> elt) 
    ev_err("sprow.c",3,150,"sprow_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(6,0,(maxlen * sizeof(row_elt )),0);
  }
  r -> len = 0;
  r -> maxlen = maxlen;
  r -> diag = - 1;
  return r;
}
/* sprow_xpd -- expand row by means of realloc()
   -- type must be TYPE_SPMAT if r is a row of a SPMAT structure,
      otherwise it must be TYPE_SPROW
   -- returns r */
#ifndef ANSI_C
#else

SPROW *sprow_xpd(SPROW *r,int n,int type)
#endif
{
  int newlen;
  if (!r) {
    r = ((SPROW *)(calloc((size_t )1,(size_t )(sizeof(SPROW )))));
    if (!r) 
      ev_err("sprow.c",3,179,"sprow_xpd",0);
     else if (mem_info_is_on()) {
      if (type != 7 && type != 6) 
        ev_err("sprow.c",1,182,"sprow_xpd",1);
      mem_bytes_list(type,0,(sizeof(SPROW )),0);
      if (type == 6) 
        mem_numvar_list(type,1,0);
    }
  }
  if (!r -> elt) {
    r -> elt = ((row_elt *)(calloc((size_t )((unsigned int )n),(size_t )(sizeof(row_elt )))));
    if (!r -> elt) 
      ev_err("sprow.c",3,193,"sprow_xpd",0);
     else if (mem_info_is_on()) {
      mem_bytes_list(type,0,(n * sizeof(row_elt )),0);
    }
    r -> len = 0;
    r -> maxlen = n;
    return r;
  }
  if (n <= r -> len) 
    newlen = (2 * r -> len + 1 > 10?2 * r -> len + 1 : 10);
   else 
    newlen = n;
  if (newlen <= r -> maxlen) {
    memset(((char *)(&r -> elt[r -> len])),'\0',(newlen - r -> len) * sizeof(row_elt ));
    r -> len = newlen;
  }
   else {
    if (mem_info_is_on()) {
      mem_bytes_list(type,((r -> maxlen) * sizeof(row_elt )),(newlen * sizeof(row_elt )),0);
    }
    r -> elt = r -> elt = ((row_elt *)((r -> elt?realloc(((char *)(r -> elt)),(size_t )(newlen * sizeof(row_elt ))) : calloc((size_t )newlen,(size_t )(sizeof(row_elt ))))));
    if (!r -> elt) 
      ev_err("sprow.c",3,219,"sprow_xpd",0);
    r -> maxlen = newlen;
    r -> len = newlen;
  }
  return r;
}
/* sprow_resize -- resize a SPROW variable by means of realloc()
   -- n is a new size
   -- returns r */
#ifndef ANSI_C
#else

SPROW *sprow_resize(SPROW *r,int n,int type)
#endif
{
  if (n < 0) 
    ev_err("sprow.c",20,239,"sprow_resize",0);
  if (!r) 
    return sprow_get(n);
  if (n == r -> len) 
    return r;
  if (!r -> elt) {
    r -> elt = ((row_elt *)(calloc((size_t )((unsigned int )n),(size_t )(sizeof(row_elt )))));
    if (!r -> elt) 
      ev_err("sprow.c",3,251,"sprow_resize",0);
     else if (mem_info_is_on()) {
      mem_bytes_list(type,0,(n * sizeof(row_elt )),0);
    }
    r -> maxlen = r -> len = n;
    return r;
  }
  if (n <= r -> maxlen) 
    r -> len = n;
   else {
    if (mem_info_is_on()) {
      mem_bytes_list(type,((r -> maxlen) * sizeof(row_elt )),(n * sizeof(row_elt )),0);
    }
    r -> elt = r -> elt = ((row_elt *)((r -> elt?realloc(((char *)(r -> elt)),(size_t )(n * sizeof(row_elt ))) : calloc((size_t )n,(size_t )(sizeof(row_elt ))))));
    if (!r -> elt) 
      ev_err("sprow.c",3,269,"sprow_resize",0);
    r -> maxlen = r -> len = n;
  }
  return r;
}
/* release a row of a matrix */
#ifndef ANSI_C
#else

int sprow_free(SPROW *r)
#endif
{
  if (!r) 
    return - 1;
  if (mem_info_is_on()) {
    mem_bytes_list(6,(sizeof(SPROW )),0,0);
    mem_numvar_list(6,- 1,0);
  }
  if (r -> elt) {
    if (mem_info_is_on()) {
      mem_bytes_list(6,((r -> maxlen) * sizeof(row_elt )),0,0);
    }
    free(((char *)(r -> elt)));
  }
  free(((char *)r));
  return 0;
}

int wrapped_sprow_free(void *p)
{
  return sprow_free((SPROW *)p);
}
/* sprow_merge -- merges r1 and r2 into r_out
   -- cannot be done in-situ
   -- type must be TYPE_SPMAT or TYPE_SPROW depending on
      whether r_out is a row of a SPMAT structure
      or a SPROW variable
   -- returns r_out */
#ifndef ANSI_C
#else

SPROW *sprow_merge(const SPROW *r1,const SPROW *r2,SPROW *r_out,int type)
#endif
{
  int idx1;
  int idx2;
  int idx_out;
  int len1;
  int len2;
  int len_out;
  row_elt *elt1;
  row_elt *elt2;
  row_elt *elt_out;
  if (!r1 || !r2) 
    ev_err("sprow.c",8,324,"sprow_merge",0);
  if (!r_out) 
    r_out = sprow_get(10);
  if (r1 == r_out || r2 == r_out) 
    ev_err("sprow.c",12,328,"sprow_merge",0);
/* Initialise */
  len1 = r1 -> len;
  len2 = r2 -> len;
  len_out = r_out -> maxlen;
  idx1 = idx2 = idx_out = 0;
  elt1 = r1 -> elt;
  elt2 = r2 -> elt;
  elt_out = r_out -> elt;
  while(idx1 < len1 || idx2 < len2){
    if (idx_out >= len_out) {
/* r_out is too small */
      r_out -> len = idx_out;
      r_out = sprow_xpd(r_out,0,type);
      len_out = r_out -> len;
      elt_out = &r_out -> elt[idx_out];
    }
    if (idx2 >= len2 || idx1 < len1 && elt1 -> col <= elt2 -> col) {
      elt_out -> col = elt1 -> col;
      elt_out -> val = elt1 -> val;
      if (elt1 -> col == elt2 -> col && idx2 < len2) {
        elt2++;
        idx2++;
      }
      elt1++;
      idx1++;
    }
     else {
      elt_out -> col = elt2 -> col;
      elt_out -> val = elt2 -> val;
      elt2++;
      idx2++;
    }
    elt_out++;
    idx_out++;
  }
  r_out -> len = idx_out;
  return r_out;
}
/* sprow_copy -- copies r1 and r2 into r_out
   -- cannot be done in-situ
   -- type must be TYPE_SPMAT or TYPE_SPROW depending on
      whether r_out is a row of a SPMAT structure
      or a SPROW variable
   -- returns r_out */
#ifndef ANSI_C
#else

SPROW *sprow_copy(const SPROW *r1,const SPROW *r2,SPROW *r_out,int type)
#endif
{
  int idx1;
  int idx2;
  int idx_out;
  int len1;
  int len2;
  int len_out;
  row_elt *elt1;
  row_elt *elt2;
  row_elt *elt_out;
  if (!r1 || !r2) 
    ev_err("sprow.c",8,383,"sprow_copy",0);
  if (!r_out) 
    r_out = sprow_get(10);
  if (r1 == r_out || r2 == r_out) 
    ev_err("sprow.c",12,387,"sprow_copy",0);
/* Initialise */
  len1 = r1 -> len;
  len2 = r2 -> len;
  len_out = r_out -> maxlen;
  idx1 = idx2 = idx_out = 0;
  elt1 = r1 -> elt;
  elt2 = r2 -> elt;
  elt_out = r_out -> elt;
  while(idx1 < len1 || idx2 < len2){
    while(idx_out >= len_out){
/* r_out is too small */
      r_out -> len = idx_out;
      r_out = sprow_xpd(r_out,0,type);
      len_out = r_out -> maxlen;
      elt_out = &r_out -> elt[idx_out];
    }
    if (idx2 >= len2 || idx1 < len1 && elt1 -> col <= elt2 -> col) {
      elt_out -> col = elt1 -> col;
      elt_out -> val = elt1 -> val;
      if (elt1 -> col == elt2 -> col && idx2 < len2) {
        elt2++;
        idx2++;
      }
      elt1++;
      idx1++;
    }
     else {
      elt_out -> col = elt2 -> col;
      elt_out -> val = 0.0;
      elt2++;
      idx2++;
    }
    elt_out++;
    idx_out++;
  }
  r_out -> len = idx_out;
  return r_out;
}
/* sprow_mltadd -- sets r_out <- r1 + alpha.r2
   -- cannot be in situ
   -- only for columns j0, j0+1, ...
   -- type must be TYPE_SPMAT or TYPE_SPROW depending on
      whether r_out is a row of a SPMAT structure
      or a SPROW variable
   -- returns r_out */
#ifndef ANSI_C
#else

SPROW *sprow_mltadd(const SPROW *r1,const SPROW *r2,double alpha,int j0,SPROW *r_out,int type)
#endif
{
  int idx1;
  int idx2;
  int idx_out;
  int len1;
  int len2;
  int len_out;
  row_elt *elt1;
  row_elt *elt2;
  row_elt *elt_out;
  if (!r1 || !r2) 
    ev_err("sprow.c",8,445,"sprow_mltadd",0);
  if (r1 == r_out || r2 == r_out) 
    ev_err("sprow.c",12,447,"sprow_mltadd",0);
  if (j0 < 0) 
    ev_err("sprow.c",2,449,"sprow_mltadd",0);
  if (!r_out) 
    r_out = sprow_get(10);
/* Initialise */
  len1 = r1 -> len;
  len2 = r2 -> len;
  len_out = r_out -> maxlen;
/* idx1 = idx2 = idx_out = 0; */
  idx1 = sprow_idx(r1,j0);
  idx2 = sprow_idx(r2,j0);
  idx_out = sprow_idx(r_out,j0);
  idx1 = (idx1 < 0?-(idx1 + 2) : idx1);
  idx2 = (idx2 < 0?-(idx2 + 2) : idx2);
  idx_out = (idx_out < 0?-(idx_out + 2) : idx_out);
  elt1 = &r1 -> elt[idx1];
  elt2 = &r2 -> elt[idx2];
  elt_out = &r_out -> elt[idx_out];
  while(idx1 < len1 || idx2 < len2){
    if (idx_out >= len_out) {
/* r_out is too small */
      r_out -> len = idx_out;
      r_out = sprow_xpd(r_out,0,type);
      len_out = r_out -> maxlen;
      elt_out = &r_out -> elt[idx_out];
    }
    if (idx2 >= len2 || idx1 < len1 && elt1 -> col <= elt2 -> col) {
      elt_out -> col = elt1 -> col;
      elt_out -> val = elt1 -> val;
      if (idx2 < len2 && elt1 -> col == elt2 -> col) {
        elt_out -> val += alpha * elt2 -> val;
        elt2++;
        idx2++;
      }
      elt1++;
      idx1++;
    }
     else {
      elt_out -> col = elt2 -> col;
      elt_out -> val = alpha * elt2 -> val;
      elt2++;
      idx2++;
    }
    elt_out++;
    idx_out++;
  }
  r_out -> len = idx_out;
  return r_out;
}
/* sprow_add -- sets r_out <- r1 + r2
   -- cannot be in situ
   -- only for columns j0, j0+1, ...
   -- type must be TYPE_SPMAT or TYPE_SPROW depending on
      whether r_out is a row of a SPMAT structure
      or a SPROW variable
   -- returns r_out */
#ifndef ANSI_C
#else

SPROW *sprow_add(const SPROW *r1,const SPROW *r2,int j0,SPROW *r_out,int type)
#endif
{
  int idx1;
  int idx2;
  int idx_out;
  int len1;
  int len2;
  int len_out;
  row_elt *elt1;
  row_elt *elt2;
  row_elt *elt_out;
  if (!r1 || !r2) 
    ev_err("sprow.c",8,519,"sprow_add",0);
  if (r1 == r_out || r2 == r_out) 
    ev_err("sprow.c",12,521,"sprow_add",0);
  if (j0 < 0) 
    ev_err("sprow.c",2,523,"sprow_add",0);
  if (!r_out) 
    r_out = sprow_get(10);
/* Initialise */
  len1 = r1 -> len;
  len2 = r2 -> len;
  len_out = r_out -> maxlen;
/* idx1 = idx2 = idx_out = 0; */
  idx1 = sprow_idx(r1,j0);
  idx2 = sprow_idx(r2,j0);
  idx_out = sprow_idx(r_out,j0);
  idx1 = (idx1 < 0?-(idx1 + 2) : idx1);
  idx2 = (idx2 < 0?-(idx2 + 2) : idx2);
  idx_out = (idx_out < 0?-(idx_out + 2) : idx_out);
  elt1 = &r1 -> elt[idx1];
  elt2 = &r2 -> elt[idx2];
  elt_out = &r_out -> elt[idx_out];
  while(idx1 < len1 || idx2 < len2){
    if (idx_out >= len_out) {
/* r_out is too small */
      r_out -> len = idx_out;
      r_out = sprow_xpd(r_out,0,type);
      len_out = r_out -> maxlen;
      elt_out = &r_out -> elt[idx_out];
    }
    if (idx2 >= len2 || idx1 < len1 && elt1 -> col <= elt2 -> col) {
      elt_out -> col = elt1 -> col;
      elt_out -> val = elt1 -> val;
      if (idx2 < len2 && elt1 -> col == elt2 -> col) {
        elt_out -> val += elt2 -> val;
        elt2++;
        idx2++;
      }
      elt1++;
      idx1++;
    }
     else {
      elt_out -> col = elt2 -> col;
      elt_out -> val = elt2 -> val;
      elt2++;
      idx2++;
    }
    elt_out++;
    idx_out++;
  }
  r_out -> len = idx_out;
  return r_out;
}
/* sprow_sub -- sets r_out <- r1 - r2
   -- cannot be in situ
   -- only for columns j0, j0+1, ...
   -- type must be TYPE_SPMAT or TYPE_SPROW depending on
      whether r_out is a row of a SPMAT structure
      or a SPROW variable
   -- returns r_out */
#ifndef ANSI_C
#else

SPROW *sprow_sub(const SPROW *r1,const SPROW *r2,int j0,SPROW *r_out,int type)
#endif
{
  int idx1;
  int idx2;
  int idx_out;
  int len1;
  int len2;
  int len_out;
  row_elt *elt1;
  row_elt *elt2;
  row_elt *elt_out;
  if (!r1 || !r2) 
    ev_err("sprow.c",8,593,"sprow_sub",0);
  if (r1 == r_out || r2 == r_out) 
    ev_err("sprow.c",12,595,"sprow_sub",0);
  if (j0 < 0) 
    ev_err("sprow.c",2,597,"sprow_sub",0);
  if (!r_out) 
    r_out = sprow_get(10);
/* Initialise */
  len1 = r1 -> len;
  len2 = r2 -> len;
  len_out = r_out -> maxlen;
/* idx1 = idx2 = idx_out = 0; */
  idx1 = sprow_idx(r1,j0);
  idx2 = sprow_idx(r2,j0);
  idx_out = sprow_idx(r_out,j0);
  idx1 = (idx1 < 0?-(idx1 + 2) : idx1);
  idx2 = (idx2 < 0?-(idx2 + 2) : idx2);
  idx_out = (idx_out < 0?-(idx_out + 2) : idx_out);
  elt1 = &r1 -> elt[idx1];
  elt2 = &r2 -> elt[idx2];
  elt_out = &r_out -> elt[idx_out];
  while(idx1 < len1 || idx2 < len2){
    if (idx_out >= len_out) {
/* r_out is too small */
      r_out -> len = idx_out;
      r_out = sprow_xpd(r_out,0,type);
      len_out = r_out -> maxlen;
      elt_out = &r_out -> elt[idx_out];
    }
    if (idx2 >= len2 || idx1 < len1 && elt1 -> col <= elt2 -> col) {
      elt_out -> col = elt1 -> col;
      elt_out -> val = elt1 -> val;
      if (idx2 < len2 && elt1 -> col == elt2 -> col) {
        elt_out -> val -= elt2 -> val;
        elt2++;
        idx2++;
      }
      elt1++;
      idx1++;
    }
     else {
      elt_out -> col = elt2 -> col;
      elt_out -> val = -elt2 -> val;
      elt2++;
      idx2++;
    }
    elt_out++;
    idx_out++;
  }
  r_out -> len = idx_out;
  return r_out;
}
/* sprow_smlt -- sets r_out <- alpha*r1 
   -- can be in situ
   -- only for columns j0, j0+1, ...
   -- returns r_out */
#ifndef ANSI_C
#else

SPROW *sprow_smlt(const SPROW *r1,double alpha,int j0,SPROW *r_out,int type)
#endif
{
  int idx1;
  int idx_out;
  int len1;
  row_elt *elt1;
  row_elt *elt_out;
  if (!r1) 
    ev_err("sprow.c",8,665,"sprow_smlt",0);
  if (j0 < 0) 
    ev_err("sprow.c",2,667,"sprow_smlt",0);
  if (!r_out) 
    r_out = sprow_get(10);
/* Initialise */
  len1 = r1 -> len;
  idx1 = sprow_idx(r1,j0);
  idx_out = sprow_idx(r_out,j0);
  idx1 = (idx1 < 0?-(idx1 + 2) : idx1);
  idx_out = (idx_out < 0?-(idx_out + 2) : idx_out);
  elt1 = &r1 -> elt[idx1];
  r_out = sprow_resize(r_out,idx_out + len1 - idx1,type);
  elt_out = &r_out -> elt[idx_out];
  for (; idx1 <= len1 - 1; (((elt1++ , elt_out++) , idx1++) , idx_out++)) {
    elt_out -> col = elt1 -> col;
    elt_out -> val = alpha * elt1 -> val;
  }
  r_out -> len = idx_out;
  return r_out;
}
#ifndef MEX
/* sprow_foutput -- print a representation of r on stream fp */
#ifndef ANSI_C
#else

void sprow_foutput(FILE *fp,const SPROW *r)
#endif
{
  int i;
  int len;
  row_elt *e;
  if (!r) {
    fprintf(fp,"SparseRow: **** NULL ****\n");
    return ;
  }
  len = r -> len;
  fprintf(fp,"SparseRow: length: %d\n",len);
  for ((i = 0 , e = r -> elt); i <= len - 1; (i++ , e++)) 
    fprintf(fp,"Column %d: %g, next row: %d, next index %d\n",e -> col,e -> val,e -> nxt_row,e -> nxt_idx);
}
#endif
/* sprow_set_val -- sets the j-th column entry of the sparse row r
   -- Note: destroys the usual column & row access paths */
#ifndef ANSI_C
#else

double sprow_set_val(SPROW *r,int j,double val)
#endif
{
  int idx;
  int idx2;
  int new_len;
  if (!r) 
    ev_err("sprow.c",8,734,"sprow_set_val",0);
  idx = sprow_idx(r,j);
  if (idx >= 0) {
    r -> elt[idx] . val = val;
    return val;
  }
  if (
/* else */
idx < - 1) {
/* shift & insert new value */
    idx = -(idx + 2);
/* this is the intended insertion index */
    if (r -> len >= r -> maxlen) {
      r -> len = r -> maxlen;
      new_len = (2 * r -> maxlen + 1 > 5?2 * r -> maxlen + 1 : 5);
      if (mem_info_is_on()) {
        mem_bytes_list(6,((r -> maxlen) * sizeof(row_elt )),(new_len * sizeof(row_elt )),0);
      }
      r -> elt = r -> elt = ((row_elt *)((r -> elt?realloc(((char *)(r -> elt)),(size_t )(new_len * sizeof(row_elt ))) : calloc((size_t )new_len,(size_t )(sizeof(row_elt ))))));
/* can't allocate */
      if (!r -> elt) 
        ev_err("sprow.c",3,754,"sprow_set_val",0);
      r -> maxlen = 2 * r -> maxlen + 1;
    }
    for (idx2 = r -> len - 1; idx2 >= idx; idx2 += -1) {
      memmove(((char *)(&r -> elt[idx2 + 1])),((char *)(&r -> elt[idx2])),sizeof(row_elt ));
    }
/************************************************************
        if ( idx < r->len )
        MEM_COPY((char *)(&(r->elt[idx])),(char *)(&(r->elt[idx+1])),
        (r->len-idx)*sizeof(row_elt));
        ************************************************************/
    r -> len++;
    r -> elt[idx] . col = j;
    r -> elt[idx] . nxt_row = - 1;
    r -> elt[idx] . nxt_idx = - 1;
    return r -> elt[idx] . val = val;
  }
/* else -- idx == -1, error in index/matrix! */
  return 0.0;
}
