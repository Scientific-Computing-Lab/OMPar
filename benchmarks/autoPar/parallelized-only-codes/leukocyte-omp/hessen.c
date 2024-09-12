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
*/
#include <omp.h> 
static char rcsid[] = "$Id: hessen.c,v 1.2 1994/01/13 05:36:24 des Exp $";
#include	<stdio.h>
#include	"matrix.h"
#include        "matrix2.h"
/* Hfactor -- compute Hessenberg factorisation in compact form.
	-- factorisation performed in situ
	-- for details of the compact form see QRfactor.c and matrix2.doc */
#ifndef ANSI_C
#else

MAT *Hfactor(MAT *A,VEC *diag,VEC *beta)
#endif
{
  static VEC *hh = (VEC *)((void *)0);
  static VEC *w = (VEC *)((void *)0);
  int k;
  int limit;
  if (!A || !diag || !beta) 
    ev_err("hessen.c",8,56,"Hfactor",0);
  if (diag -> dim < A -> m - 1 || beta -> dim < A -> m - 1) 
    ev_err("hessen.c",1,58,"Hfactor",0);
  if (A -> m != A -> n) 
    ev_err("hessen.c",9,60,"Hfactor",0);
  limit = (A -> m - 1);
  hh = v_resize(hh,(A -> m));
  w = v_resize(w,(A -> n));
  mem_stat_reg_list((void **)(&hh),3,0,"hessen.c",65);
  mem_stat_reg_list((void **)(&w),3,0,"hessen.c",66);
  for (k = 0; k <= limit - 1; k += 1) {
/* compute the Householder vector hh */
    get_col(A,(unsigned int )k,hh);
/* printf("the %d'th column = ");	v_output(hh); */
    hhvec(hh,(k + 1),&beta -> ve[k],hh,&A -> me[k + 1][k]);
/* diag->ve[k] = hh->ve[k+1]; */
    diag -> ve[k] = hh -> ve[k + 1];
/* printf("H/h vector = ");	v_output(hh); */
/* printf("from the %d'th entry\n",k+1); */
/* printf("beta = %g\n",beta->ve[k]); */
/* apply Householder operation symmetrically to A */
    _hhtrcols(A,(k + 1),(k + 1),hh,beta -> ve[k],w);
    hhtrrows(A,0,(k + 1),hh,beta -> ve[k]);
/* printf("A = ");		m_output(A); */
  }
#ifdef THREADSAFE
#endif
  return A;
}
/* makeHQ -- construct the Hessenberg orthogonalising matrix Q;
	-- i.e. Hess M = Q.M.Q'	*/
#ifndef ANSI_C
#else

MAT *makeHQ(MAT *H,VEC *diag,VEC *beta,MAT *Qout)
#endif
{
  int i;
  int j;
  int limit;
  static VEC *tmp1 = (VEC *)((void *)0);
  static VEC *tmp2 = (VEC *)((void *)0);
  if (H == ((MAT *)((void *)0)) || diag == ((VEC *)((void *)0)) || beta == ((VEC *)((void *)0))) 
    ev_err("hessen.c",8,107,"makeHQ",0);
  limit = (H -> m - 1);
  if (diag -> dim < limit || beta -> dim < limit) 
    ev_err("hessen.c",1,110,"makeHQ",0);
  if (H -> m != H -> n) 
    ev_err("hessen.c",9,112,"makeHQ",0);
  Qout = m_resize(Qout,(H -> m),(H -> m));
  tmp1 = v_resize(tmp1,(H -> m));
  tmp2 = v_resize(tmp2,(H -> m));
  mem_stat_reg_list((void **)(&tmp1),3,0,"hessen.c",117);
  mem_stat_reg_list((void **)(&tmp2),3,0,"hessen.c",118);
  for (i = 0; ((unsigned int )i) <= H -> m - 1; i += 1) {
/* tmp1 = i'th basis vector */
    
#pragma omp parallel for private (j)
    for (j = 0; ((unsigned int )j) <= H -> m - 1; j += 1) {
/* tmp1->ve[j] = 0.0; */
      tmp1 -> ve[j] = 0.0;
    }
/* tmp1->ve[i] = 1.0; */
    tmp1 -> ve[i] = 1.0;
/* apply H/h transforms in reverse order */
    for (j = limit - 1; j >= 0; j += -1) {
      get_col(H,(unsigned int )j,tmp2);
/* tmp2->ve[j+1] = diag->ve[j]; */
      tmp2 -> ve[j + 1] = diag -> ve[j];
      hhtrvec(tmp2,beta -> ve[j],(j + 1),tmp1,tmp1);
    }
/* insert into Qout */
    _set_col(Qout,(unsigned int )i,tmp1,0);
  }
#ifdef THREADSAFE
#endif
  return Qout;
}
/* makeH -- construct actual Hessenberg matrix */
#ifndef ANSI_C
#else

MAT *makeH(const MAT *H,MAT *Hout)
#endif
{
  int i;
  int j;
  int limit;
  if (H == ((MAT *)((void *)0))) 
    ev_err("hessen.c",8,160,"makeH",0);
  if (H -> m != H -> n) 
    ev_err("hessen.c",9,162,"makeH",0);
  Hout = m_resize(Hout,(H -> m),(H -> m));
  Hout = _m_copy(H,Hout,0,0);
  limit = (H -> m);
  
#pragma omp parallel for private (i,j) firstprivate (limit)
  for (i = 1; i <= limit - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= i - 1 - 1; j += 1) {
/* Hout->me[i][j] = 0.0;*/
      Hout -> me[i][j] = 0.0;
    }
  }
  return Hout;
}
