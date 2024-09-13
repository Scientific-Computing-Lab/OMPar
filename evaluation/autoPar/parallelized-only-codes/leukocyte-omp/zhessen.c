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
		File containing routines for determining Hessenberg
	factorisations.
	Complex version
*/
#include <omp.h> 
static char rcsid[] = "$Id: zhessen.c,v 1.2 1995/03/27 15:47:50 des Exp $";
#include	<stdio.h>
#include	"zmatrix.h"
#include        "zmatrix2.h"
/* zHfactor -- compute Hessenberg factorisation in compact form.
	-- factorisation performed in situ
	-- for details of the compact form see zQRfactor.c and zmatrix2.doc */

ZMAT *zHfactor(
//A, diag)
ZMAT *A,ZVEC *diag)
{
  static ZVEC *tmp1 = (ZVEC *)((void *)0);
  static ZVEC *w = (ZVEC *)((void *)0);
  double beta;
  int k;
  int limit;
  if (!A || !diag) 
    ev_err("zhessen.c",8,53,"zHfactor",0);
  if (diag -> dim < A -> m - 1) 
    ev_err("zhessen.c",1,55,"zHfactor",0);
  if (A -> m != A -> n) 
    ev_err("zhessen.c",9,57,"zHfactor",0);
  limit = (A -> m - 1);
  tmp1 = zv_resize(tmp1,(A -> m));
  w = zv_resize(w,(A -> n));
  mem_stat_reg_list((void **)(&tmp1),8,0,"zhessen.c",62);
  mem_stat_reg_list((void **)(&w),8,0,"zhessen.c",63);
  for (k = 0; k <= limit - 1; k += 1) {
    zget_col(A,k,tmp1);
    zhhvec(tmp1,k + 1,&beta,tmp1,&A -> me[k + 1][k]);
    diag -> ve[k] = tmp1 -> ve[k + 1];
/* printf("zHfactor: k = %d, beta = %g, tmp1 =\n",k,beta);
	    zv_output(tmp1); */
    _zhhtrcols(A,k + 1,k + 1,tmp1,beta,w);
    zhhtrrows(A,0,k + 1,tmp1,beta);
/* printf("# at stage k = %d, A =\n",k);	zm_output(A); */
  }
#ifdef	THREADSAFE
#endif
  return A;
}
/* zHQunpack -- unpack the compact representation of H and Q of a
	Hessenberg factorisation
	-- if either H or Q is NULL, then it is not unpacked
	-- it can be in situ with HQ == H
	-- returns HQ
*/

ZMAT *zHQunpack(
//HQ,diag,Q,H)
ZMAT *HQ,ZVEC *diag,ZMAT *Q,ZMAT *H)
{
  int i;
  int j;
  int limit;
  double beta;
  double r_ii;
  double tmp_val;
  static ZVEC *tmp1 = (ZVEC *)((void *)0);
  static ZVEC *tmp2 = (ZVEC *)((void *)0);
  if (HQ == ((ZMAT *)((void *)0)) || diag == ((ZVEC *)((void *)0))) 
    ev_err("zhessen.c",8,102,"zHQunpack",0);
  if (HQ == Q || H == Q) 
    ev_err("zhessen.c",12,104,"zHQunpack",0);
  limit = (HQ -> m - 1);
  if (diag -> dim < limit) 
    ev_err("zhessen.c",1,107,"zHQunpack",0);
  if (HQ -> m != HQ -> n) 
    ev_err("zhessen.c",9,109,"zHQunpack",0);
  if (Q != ((ZMAT *)((void *)0))) {
    Q = zm_resize(Q,(HQ -> m),(HQ -> m));
    tmp1 = zv_resize(tmp1,(H -> m));
    tmp2 = zv_resize(tmp2,(H -> m));
    mem_stat_reg_list((void **)(&tmp1),8,0,"zhessen.c",117);
    mem_stat_reg_list((void **)(&tmp2),8,0,"zhessen.c",118);
    for (i = 0; ((unsigned int )i) <= H -> m - 1; i += 1) {
/* tmp1 = i'th basis vector */
      
#pragma omp parallel for private (j)
      for (j = 0; ((unsigned int )j) <= H -> m - 1; j += 1) {
        tmp1 -> ve[j] . re = tmp1 -> ve[j] . im = 0.0;
      }
      tmp1 -> ve[i] . re = 1.0;
/* apply H/h transforms in reverse order */
      for (j = limit - 1; j >= 0; j += -1) {
        zget_col(HQ,j,tmp2);
        r_ii = zabs(tmp2 -> ve[j + 1]);
        tmp2 -> ve[j + 1] = diag -> ve[j];
        tmp_val = r_ii * zabs(diag -> ve[j]);
        beta = (tmp_val == 0.0?0.0 : 1.0 / tmp_val);
/* printf("zHQunpack: j = %d, beta = %g, tmp2 =\n",
			   j,beta);
		    zv_output(tmp2); */
        zhhtrvec(tmp2,beta,j + 1,tmp1,tmp1);
      }
/* insert into Q */
      zset_col(Q,i,tmp1);
    }
  }
  if (H != ((ZMAT *)((void *)0))) {
    H = _zm_copy(HQ,(zm_resize(H,(HQ -> m),(HQ -> n))),0,0);
    limit = (H -> m);
    
#pragma omp parallel for private (i,j) firstprivate (limit)
    for (i = 1; i <= limit - 1; i += 1) {
      
#pragma omp parallel for private (j)
      for (j = 0; j <= i - 1 - 1; j += 1) {
        H -> me[i][j] . re = H -> me[i][j] . im = 0.0;
      }
    }
  }
#ifdef	THREADSAFE
#endif
  return HQ;
}
