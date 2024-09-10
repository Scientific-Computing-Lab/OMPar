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
/* matop.c 1.3 11/25/87 */
#include	<stdio.h>
#include	"matrix.h"
#include <omp.h> 
static char rcsid[] = "$Id: matop.c,v 1.4 1995/03/27 15:43:57 des Exp $";
/* m_add -- matrix addition -- may be in-situ */
#ifndef ANSI_C
#else

MAT *m_add(const MAT *mat1,const MAT *mat2,MAT *out)
#endif
{
  unsigned int m;
  unsigned int n;
  unsigned int i;
  if (mat1 == ((MAT *)((void *)0)) || mat2 == ((MAT *)((void *)0))) 
    ev_err("matop.c",8,47,"m_add",0);
  if (mat1 -> m != mat2 -> m || mat1 -> n != mat2 -> n) 
    ev_err("matop.c",1,49,"m_add",0);
  if (out == ((MAT *)((void *)0)) || out -> m != mat1 -> m || out -> n != mat1 -> n) 
    out = m_resize(out,(mat1 -> m),(mat1 -> n));
  m = mat1 -> m;
  n = mat1 -> n;
  for (i = 0; i <= m - 1; i += 1) {
    __add__(mat1 -> me[i],mat2 -> me[i],out -> me[i],(int )n);
/**************************************************
		for ( j=0; j<n; j++ )
			out->me[i][j] = mat1->me[i][j]+mat2->me[i][j];
		**************************************************/
  }
  return out;
}
/* m_sub -- matrix subtraction -- may be in-situ */
#ifndef ANSI_C
#else

MAT *m_sub(const MAT *mat1,const MAT *mat2,MAT *out)
#endif
{
  unsigned int m;
  unsigned int n;
  unsigned int i;
  if (mat1 == ((MAT *)((void *)0)) || mat2 == ((MAT *)((void *)0))) 
    ev_err("matop.c",8,76,"m_sub",0);
  if (mat1 -> m != mat2 -> m || mat1 -> n != mat2 -> n) 
    ev_err("matop.c",1,78,"m_sub",0);
  if (out == ((MAT *)((void *)0)) || out -> m != mat1 -> m || out -> n != mat1 -> n) 
    out = m_resize(out,(mat1 -> m),(mat1 -> n));
  m = mat1 -> m;
  n = mat1 -> n;
  for (i = 0; i <= m - 1; i += 1) {
    __sub__(mat1 -> me[i],mat2 -> me[i],out -> me[i],(int )n);
/**************************************************
		for ( j=0; j<n; j++ )
			out->me[i][j] = mat1->me[i][j]-mat2->me[i][j];
		**************************************************/
  }
  return out;
}
/* m_mlt -- matrix-matrix multiplication */
#ifndef ANSI_C
#else

MAT *m_mlt(const MAT *A,const MAT *B,MAT *OUT)
#endif
{
  unsigned int i;
/* j, */
  unsigned int k;
  unsigned int m;
  unsigned int n;
  unsigned int p;
  double **A_v;
  double **B_v;
/*, *B_row, *OUT_row, sum, tmp */
  if (A == ((MAT *)((void *)0)) || B == ((MAT *)((void *)0))) 
    ev_err("matop.c",8,106,"m_mlt",0);
  if (A -> n != B -> m) 
    ev_err("matop.c",1,108,"m_mlt",0);
  if (A == OUT || B == OUT) 
    ev_err("matop.c",12,110,"m_mlt",0);
  m = A -> m;
  n = A -> n;
  p = B -> n;
  A_v = A -> me;
  B_v = B -> me;
  if (OUT == ((MAT *)((void *)0)) || OUT -> m != A -> m || OUT -> n != B -> n) 
    OUT = m_resize(OUT,(A -> m),(B -> n));
/****************************************************************
	for ( i=0; i<m; i++ )
		for  ( j=0; j<p; j++ )
		{
			sum = 0.0;
			for ( k=0; k<n; k++ )
				sum += A_v[i][k]*B_v[k][j];
			OUT->me[i][j] = sum;
		}
****************************************************************/
  m_zero(OUT);
  for (i = 0; i <= m - 1; i += 1) {
    for (k = 0; k <= n - 1; k += 1) {
      if (A_v[i][k] != 0.0) 
        __mltadd__(OUT -> me[i],B_v[k],A_v[i][k],(int )p);
/**************************************************
		    B_row = B_v[k];	OUT_row = OUT->me[i];
		    for ( j=0; j<p; j++ )
			(*OUT_row++) += tmp*(*B_row++);
		    **************************************************/
    }
  }
  return OUT;
}
/* mmtr_mlt -- matrix-matrix transposed multiplication
	-- A.B^T is returned, and stored in OUT */
#ifndef ANSI_C
#else

MAT *mmtr_mlt(const MAT *A,const MAT *B,MAT *OUT)
#endif
{
  int i;
  int j;
  int limit;
/* Real	*A_row, *B_row, sum; */
  if (!A || !B) 
    ev_err("matop.c",8,156,"mmtr_mlt",0);
  if (A == OUT || B == OUT) 
    ev_err("matop.c",12,158,"mmtr_mlt",0);
  if (A -> n != B -> n) 
    ev_err("matop.c",1,160,"mmtr_mlt",0);
  if (!OUT || OUT -> m != A -> m || OUT -> n != B -> m) 
    OUT = m_resize(OUT,(A -> m),(B -> m));
  limit = (A -> n);
  for (i = 0; ((unsigned int )i) <= A -> m - 1; i += 1) {
    for (j = 0; ((unsigned int )j) <= B -> m - 1; j += 1) {
      OUT -> me[i][j] = __ip__(A -> me[i],B -> me[j],(int )limit);
/**************************************************
		    sum = 0.0;
		    A_row = A->me[i];
		    B_row = B->me[j];
		    for ( k = 0; k < limit; k++ )
			sum += (*A_row++)*(*B_row++);
		    OUT->me[i][j] = sum;
		    **************************************************/
    }
  }
  return OUT;
}
/* mtrm_mlt -- matrix transposed-matrix multiplication
	-- A^T.B is returned, result stored in OUT */
#ifndef ANSI_C
#else

MAT *mtrm_mlt(const MAT *A,const MAT *B,MAT *OUT)
#endif
{
  int i;
  int k;
  int limit;
/* Real	*B_row, *OUT_row, multiplier; */
  if (!A || !B) 
    ev_err("matop.c",8,195,"mmtr_mlt",0);
  if (A == OUT || B == OUT) 
    ev_err("matop.c",12,197,"mtrm_mlt",0);
  if (A -> m != B -> m) 
    ev_err("matop.c",1,199,"mmtr_mlt",0);
  if (!OUT || OUT -> m != A -> n || OUT -> n != B -> n) 
    OUT = m_resize(OUT,(A -> n),(B -> n));
  limit = (B -> n);
  m_zero(OUT);
  for (k = 0; ((unsigned int )k) <= A -> m - 1; k += 1) {
    for (i = 0; ((unsigned int )i) <= A -> n - 1; i += 1) {
      if (A -> me[k][i] != 0.0) 
        __mltadd__(OUT -> me[i],B -> me[k],A -> me[k][i],(int )limit);
/**************************************************
		    multiplier = A->me[k][i];
		    OUT_row = OUT->me[i];
		    B_row   = B->me[k];
		    for ( j = 0; j < limit; j++ )
			*(OUT_row++) += multiplier*(*B_row++);
		    **************************************************/
    }
  }
  return OUT;
}
/* mv_mlt -- matrix-vector multiplication 
		-- Note: b is treated as a column vector */
#ifndef ANSI_C
#else

VEC *mv_mlt(const MAT *A,const VEC *b,VEC *out)
#endif
{
  unsigned int i;
  unsigned int m;
  unsigned int n;
  double **A_v;
  double *b_v;
/*, *A_row */
/* register Real	sum; */
  if (A == ((MAT *)((void *)0)) || b == ((VEC *)((void *)0))) 
    ev_err("matop.c",8,237,"mv_mlt",0);
  if (A -> n != b -> dim) 
    ev_err("matop.c",1,239,"mv_mlt",0);
  if (b == out) 
    ev_err("matop.c",12,241,"mv_mlt",0);
  if (out == ((VEC *)((void *)0)) || out -> dim != A -> m) 
    out = v_resize(out,(A -> m));
  m = A -> m;
  n = A -> n;
  A_v = A -> me;
  b_v = b -> ve;
  for (i = 0; i <= m - 1; i += 1) {
/* for ( j=0; j<n; j++ )
			sum += A_v[i][j]*b_v[j]; */
    out -> ve[i] = __ip__(A_v[i],b_v,(int )n);
/**************************************************
		A_row = A_v[i];		b_v = b->ve;
		for ( j=0; j<n; j++ )
			sum += (*A_row++)*(*b_v++);
		out->ve[i] = sum;
		**************************************************/
  }
  return out;
}
/* sm_mlt -- scalar-matrix multiply -- may be in-situ */
#ifndef ANSI_C
#else

MAT *sm_mlt(double scalar,const MAT *matrix,MAT *out)
#endif
{
  unsigned int m;
  unsigned int n;
  unsigned int i;
  if (matrix == ((MAT *)((void *)0))) 
    ev_err("matop.c",8,275,"sm_mlt",0);
  if (out == ((MAT *)((void *)0)) || out -> m != matrix -> m || out -> n != matrix -> n) 
    out = m_resize(out,(matrix -> m),(matrix -> n));
  m = matrix -> m;
  n = matrix -> n;
  for (i = 0; i <= m - 1; i += 1) {
    __smlt__(matrix -> me[i],(double )scalar,out -> me[i],(int )n);
  }
/**************************************************
		for ( j=0; j<n; j++ )
			out->me[i][j] = scalar*matrix->me[i][j];
		**************************************************/
  return out;
}
/* vm_mlt -- vector-matrix multiplication 
		-- Note: b is treated as a row vector */
#ifndef ANSI_C
#else

VEC *vm_mlt(const MAT *A,const VEC *b,VEC *out)
#endif
{
  unsigned int j;
  unsigned int m;
  unsigned int n;
/* Real	sum,**A_v,*b_v; */
  if (A == ((MAT *)((void *)0)) || b == ((VEC *)((void *)0))) 
    ev_err("matop.c",8,302,"vm_mlt",0);
  if (A -> m != b -> dim) 
    ev_err("matop.c",1,304,"vm_mlt",0);
  if (b == out) 
    ev_err("matop.c",12,306,"vm_mlt",0);
  if (out == ((VEC *)((void *)0)) || out -> dim != A -> n) 
    out = v_resize(out,(A -> n));
  m = A -> m;
  n = A -> n;
  v_zero(out);
  for (j = 0; j <= m - 1; j += 1) {
    if (b -> ve[j] != 0.0) 
      __mltadd__(out -> ve,A -> me[j],b -> ve[j],(int )n);
  }
/**************************************************
	A_v = A->me;		b_v = b->ve;
	for ( j=0; j<n; j++ )
	{
		sum = 0.0;
		for ( i=0; i<m; i++ )
			sum += b_v[i]*A_v[i][j];
		out->ve[j] = sum;
	}
	**************************************************/
  return out;
}
/* m_transp -- transpose matrix */
#ifndef ANSI_C
#else

MAT *m_transp(const MAT *in,MAT *out)
#endif
{
  int i;
  int j;
  int in_situ;
  double tmp;
  if (in == ((MAT *)((void *)0))) 
    ev_err("matop.c",8,343,"m_transp",0);
  if (in == out && in -> n != in -> m) 
    ev_err("matop.c",11,345,"m_transp",0);
  in_situ = in == out;
  if (out == ((MAT *)((void *)0)) || out -> m != in -> n || out -> n != in -> m) 
    out = m_resize(out,(in -> n),(in -> m));
  if (!in_situ) {
    
#pragma omp parallel for private (i,j)
    for (i = 0; ((unsigned int )i) <= in -> m - 1; i += 1) {
      
#pragma omp parallel for private (j)
      for (j = 0; ((unsigned int )j) <= in -> n - 1; j += 1) {
        out -> me[j][i] = in -> me[i][j];
      }
    }
  }
   else {
    
#pragma omp parallel for private (tmp,i,j)
    for (i = 1; ((unsigned int )i) <= in -> m - 1; i += 1) {
      for (j = 0; j <= i - 1; j += 1) {
        tmp = in -> me[i][j];
        in -> me[i][j] = in -> me[j][i];
        in -> me[j][i] = tmp;
      }
    }
  }
  return out;
}
/* swap_rows -- swaps rows i and j of matrix A for cols lo through hi */
#ifndef ANSI_C
#else

MAT *swap_rows(MAT *A,int i,int j,int lo,int hi)
#endif
{
  int k;
  double **A_me;
  double tmp;
  if (!A) 
    ev_err("matop.c",8,378,"swap_rows",0);
  if (i < 0 || j < 0 || i >= A -> m || j >= A -> m) 
    ev_err("matop.c",1,380,"swap_rows",0);
  lo = (0 > lo?0 : lo);
  hi = ((hi > A -> n - 1?A -> n - 1 : hi));
  A_me = A -> me;
  
#pragma omp parallel for private (tmp,k) firstprivate (i,j,hi)
  for (k = lo; k <= hi; k += 1) {
    tmp = A_me[k][i];
    A_me[k][i] = A_me[k][j];
    A_me[k][j] = tmp;
  }
  return A;
}
/* swap_cols -- swap columns i and j of matrix A for cols lo through hi */
#ifndef ANSI_C
#else

MAT *swap_cols(MAT *A,int i,int j,int lo,int hi)
#endif
{
  int k;
  double **A_me;
  double tmp;
  if (!A) 
    ev_err("matop.c",8,407,"swap_cols",0);
  if (i < 0 || j < 0 || i >= A -> n || j >= A -> n) 
    ev_err("matop.c",1,409,"swap_cols",0);
  lo = (0 > lo?0 : lo);
  hi = ((hi > A -> m - 1?A -> m - 1 : hi));
  A_me = A -> me;
  
#pragma omp parallel for private (tmp,k) firstprivate (i,j,hi)
  for (k = lo; k <= hi; k += 1) {
    tmp = A_me[i][k];
    A_me[i][k] = A_me[j][k];
    A_me[j][k] = tmp;
  }
  return A;
}
/* ms_mltadd -- matrix-scalar multiply and add
	-- may be in situ
	-- returns out == A1 + s*A2 */
#ifndef ANSI_C
#else

MAT *ms_mltadd(const MAT *A1,const MAT *A2,double s,MAT *out)
#endif
{
/* register Real	*A1_e, *A2_e, *out_e; */
/* register int	j; */
  int i;
  int m;
  int n;
  if (!A1 || !A2) 
    ev_err("matop.c",8,439,"ms_mltadd",0);
  if (A1 -> m != A2 -> m || A1 -> n != A2 -> n) 
    ev_err("matop.c",1,441,"ms_mltadd",0);
  if (out != A1 && out != A2) 
    out = m_resize(out,(A1 -> m),(A1 -> n));
  if (s == 0.0) 
    return _m_copy(A1,out,0,0);
  if (s == 1.0) 
    return m_add(A1,A2,out);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      out = _m_copy(A1,out,0,0);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("matop.c",_err_num,451,"ms_mltadd",0);
    }
  }
  ;
  m = (A1 -> m);
  n = (A1 -> n);
  for (i = 0; i <= m - 1; i += 1) {
    __mltadd__(out -> me[i],A2 -> me[i],s,(int )n);
/**************************************************
		A1_e = A1->me[i];
		A2_e = A2->me[i];
		out_e = out->me[i];
		for ( j = 0; j < n; j++ )
		    out_e[j] = A1_e[j] + s*A2_e[j];
		**************************************************/
  }
  return out;
}
/* mv_mltadd -- matrix-vector multiply and add
	-- may not be in situ
	-- returns out == v1 + alpha*A*v2 */
#ifndef ANSI_C
#else

VEC *mv_mltadd(const VEC *v1,const VEC *v2,const MAT *A,double alpha,VEC *out)
#endif
{
/* register	int	j; */
  int i;
  int m;
  int n;
  double *v2_ve;
  double *out_ve;
  if (!v1 || !v2 || !A) 
    ev_err("matop.c",8,487,"mv_mltadd",0);
  if (out == v2) 
    ev_err("matop.c",12,489,"mv_mltadd",0);
  if (v1 -> dim != A -> m || v2 -> dim != A -> n) 
    ev_err("matop.c",1,491,"mv_mltadd",0);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      out = _v_copy(v1,out,0);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("matop.c",_err_num,493,"mv_mltadd",0);
    }
  }
  ;
  v2_ve = v2 -> ve;
  out_ve = out -> ve;
  m = (A -> m);
  n = (A -> n);
  if (alpha == 0.0) 
    return out;
  for (i = 0; i <= m - 1; i += 1) {
    out_ve[i] += alpha * __ip__(A -> me[i],v2_ve,(int )n);
/**************************************************
		A_e = A->me[i];
		sum = 0.0;
		for ( j = 0; j < n; j++ )
		    sum += A_e[j]*v2_ve[j];
		out_ve[i] = v1->ve[i] + alpha*sum;
		**************************************************/
  }
  return out;
}
/* vm_mltadd -- vector-matrix multiply and add
	-- may not be in situ
	-- returns out' == v1' + v2'*A */
#ifndef ANSI_C
#else

VEC *vm_mltadd(const VEC *v1,const VEC *v2,const MAT *A,double alpha,VEC *out)
#endif
{
/* i, */
  int j;
  int m;
  int n;
  double tmp;
/* *A_e, */
  double *out_ve;
  if (!v1 || !v2 || !A) 
    ev_err("matop.c",8,533,"vm_mltadd",0);
  if (v2 == out) 
    ev_err("matop.c",12,535,"vm_mltadd",0);
  if (v1 -> dim != A -> n || A -> m != v2 -> dim) 
    ev_err("matop.c",1,537,"vm_mltadd",0);
{
    jmp_buf _save;
    int _err_num;
    int _old_flag;
    _old_flag = set_err_flag(2);
    memmove(_save,restart,sizeof(jmp_buf ));
    if ((_err_num = _setjmp(restart)) == 0) {
      out = _v_copy(v1,out,0);
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
    }
     else {
      set_err_flag(_old_flag);
      memmove(restart,_save,sizeof(jmp_buf ));
      ev_err("matop.c",_err_num,539,"vm_mltadd",0);
    }
  }
  ;
  out_ve = out -> ve;
  m = (A -> m);
  n = (A -> n);
  for (j = 0; j <= m - 1; j += 1) {
    tmp = v2 -> ve[j] * alpha;
    if (tmp != 0.0) 
      __mltadd__(out_ve,A -> me[j],tmp,(int )n);
/**************************************************
		A_e = A->me[j];
		for ( i = 0; i < n; i++ )
		    out_ve[i] += A_e[i]*tmp;
		**************************************************/
  }
  return out;
}
