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
/* iotort.c  10/11/93 */
/* test of I/O functions */
#include <omp.h> 
static char rcsid[] = "$Id: $";
#include "sparse.h"
#include "zmatrix.h"
#define	errmesg(mesg)	printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)	printf("# Testing %s...\n",mesg);

void main()
{
  VEC *x;
  MAT *A;
  PERM *pivot;
  IVEC *ix;
  SPMAT *spA;
  ZVEC *zx;
  ZMAT *ZA;
  char yes;
  int i;
  FILE *fp;
  mem_info_on(1);
  if ((fp = fopen("iotort.dat","w")) == ((void *)0)) {
    printf(" !!! Cannot open file %s for writing\n\n","iotort.dat");
    exit(1);
  }
  x = v_get(10);
  A = m_get(3,3);
  zx = zv_get(10);
  ZA = zm_get(3,3);
  pivot = px_get(10);
  ix = iv_get(10);
  spA = sp_get(3,3,2);
  v_rand(x);
  m_rand(A);
  zv_rand(zx);
  zm_rand(ZA);
  px_ident(pivot);
  
#pragma omp parallel for private (i)
  for (i = 0; i <= 9; i += 1) {
    ix -> ive[i] = i + 1;
  }
  for (i = 0; i <= spA -> m - 1; i += 1) {
    sp_set_val(spA,i,i,1.0);
    if (i > 0) 
      sp_set_val(spA,i - 1,i,- 1.0);
  }
  printf("# Testing %s...\n"," VEC output");
  ;
  v_foutput(fp,x);
  printf("# Testing %s...\n"," MAT output");
  ;
  m_foutput(fp,A);
  printf("# Testing %s...\n"," ZVEC output");
  ;
  zv_foutput(fp,zx);
  printf("# Testing %s...\n"," ZMAT output");
  ;
  zm_foutput(fp,ZA);
  printf("# Testing %s...\n"," PERM output");
  ;
  px_foutput(fp,pivot);
  printf("# Testing %s...\n"," IVEC output");
  ;
  iv_foutput(fp,ix);
  printf("# Testing %s...\n"," SPMAT output");
  ;
  sp_foutput(fp,spA);
  fprintf(fp,"Y");
  fclose(fp);
  printf("\nENTER SOME VALUES:\n\n");
  if ((fp = fopen("iotort.dat","r")) == ((void *)0)) {
    printf(" !!! Cannot open file %s for reading\n\n","iotort.dat");
    exit(1);
  }
  printf("# Testing %s...\n"," VEC input/output");
  ;
  x = v_finput(fp,x);
  v_foutput(stdout,x);
  printf("# Testing %s...\n"," MAT input/output");
  ;
  A = m_finput(fp,A);
  m_foutput(stdout,A);
  printf("# Testing %s...\n"," ZVEC input/output");
  ;
  zx = zv_finput(fp,zx);
  zv_foutput(stdout,zx);
  printf("# Testing %s...\n"," ZMAT input/output");
  ;
  ZA = zm_finput(fp,ZA);
  zm_foutput(stdout,ZA);
  printf("# Testing %s...\n"," PERM input/output");
  ;
  pivot = px_finput(fp,pivot);
  px_foutput(stdout,pivot);
  printf("# Testing %s...\n"," IVEC input/output");
  ;
  ix = iv_finput(fp,ix);
  iv_foutput(stdout,ix);
  printf("# Testing %s...\n"," SPMAT input/output");
  ;
  (sp_free(spA) , spA = ((SPMAT *)((void *)0)));
  spA = sp_finput(fp);
  sp_foutput(stdout,spA);
  printf("# Testing %s...\n"," general input");
  ;
  ((isatty((fileno(fp)))?fprintf(stderr," finish the test?  ") : skipjunk(fp)) , fscanf(fp,"%c",&yes));
  if (yes == 'y' || yes == 'Y') 
    printf(" YES\n");
   else 
    printf(" NO\n");
  fclose(fp);
  mem_info_file(stdout,0);
}
