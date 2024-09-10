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
  Tests for mem_info.c functions
  */
#include <omp.h> 
static char rcsid[] = "$Id: $";
#include        <stdio.h>
#include        <math.h>
#include        "matrix2.h"
#include 	"sparse2.h"
#include  	"zmatrix2.h"
#define errmesg(mesg)   printf("Error: %s error: line %d\n",mesg,__LINE__)
#define notice(mesg)    printf("# Testing %s...\n",mesg)
/*  new types list */
extern MEM_CONNECT mem_connect[5];
/* the number of a new list */
#define FOO_LIST 1
/* numbers of types */
#define TYPE_FOO_1    1
#define TYPE_FOO_2    2
typedef struct {
int dim;
int fix_dim;
double (*a)[10];}FOO_1;
typedef struct {
int dim;
int fix_dim;
double (*a)[2];}FOO_2;

FOO_1 *foo_1_get(dim)
int dim;
{
  FOO_1 *f;
  if ((f = ((FOO_1 *)(malloc(sizeof(FOO_1 ))))) == ((void *)0)) 
    ev_err("memtort.c",3,75,"foo_1_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(1,0,(sizeof(FOO_1 )),1);
    mem_numvar_list(1,1,1);
  }
  f -> dim = dim;
  f -> fix_dim = 10;
  if ((f -> a = ((double (*)[10])(malloc(dim * sizeof(double [10]))))) == ((void *)0)) 
    ev_err("memtort.c",3,84,"foo_1_get",0);
   else if (mem_info_is_on()) 
    mem_bytes_list(1,0,(dim * sizeof(double [10])),1);
  return f;
}

FOO_2 *foo_2_get(dim)
int dim;
{
  FOO_2 *f;
  if ((f = ((FOO_2 *)(malloc(sizeof(FOO_2 ))))) == ((void *)0)) 
    ev_err("memtort.c",3,98,"foo_2_get",0);
   else if (mem_info_is_on()) {
    mem_bytes_list(2,0,(sizeof(FOO_2 )),1);
    mem_numvar_list(2,1,1);
  }
  f -> dim = dim;
  f -> fix_dim = 2;
  if ((f -> a = ((double (*)[2])(malloc(dim * sizeof(double [2]))))) == ((void *)0)) 
    ev_err("memtort.c",3,107,"foo_2_get",0);
   else if (mem_info_is_on()) 
    mem_bytes_list(2,0,(dim * sizeof(double [2])),1);
  return f;
}

int foo_1_free(f)
FOO_1 *f;
{
  if (f != ((void *)0)) {
    if (mem_info_is_on()) {
      mem_bytes_list(1,(sizeof(FOO_1 ) + (f -> dim) * sizeof(double [10])),0,1);
      mem_numvar_list(1,- 1,1);
    }
    free((f -> a));
    free(f);
  }
  return 0;
}

int foo_2_free(f)
FOO_2 *f;
{
  if (f != ((void *)0)) {
    if (mem_info_is_on()) {
      mem_bytes_list(2,(sizeof(FOO_2 ) + (f -> dim) * sizeof(double [2])),0,1);
      mem_numvar_list(2,- 1,1);
    }
    free((f -> a));
    free(f);
  }
  return 0;
}
char *foo_type_name[] = {("nothing"), ("FOO_1"), ("FOO_2")};
#define FOO_NUM_TYPES  (sizeof(foo_type_name)/sizeof(*foo_type_name))
int (*foo_free_func[3])() = {(((void *)0)), (foo_1_free), (foo_2_free)};
static MEM_ARRAY foo_info_sum[3];
/* px_rand -- generates sort-of random permutation */

PERM *px_rand(pi)
PERM *pi;
{
  int i;
  int j;
  int k;
  if (!pi) 
    ev_err("memtort.c",8,180,"px_rand",0);
  for (i = 0; ((unsigned int )i) <= ((unsigned int )3) * pi -> size - 1; i += 1) {
    j = ((rand() >> 8) % pi -> size);
    k = ((rand() >> 8) % pi -> size);
    px_transp(pi,j,k);
  }
  return pi;
}
#ifdef SPARSE

SPMAT *gen_non_symm(m,n)
int m;
int n;
{
  SPMAT *A;
  static PERM *px = (PERM *)((void *)0);
  int i;
  int j;
  int k;
  int k_max;
  double s1;
  A = sp_get(m,n,8);
  px = px_resize(px,n);
  mem_stat_reg_list((void **)(&px),2,0,"memtort.c",203);
  for (i = 0; i <= A -> m - 1; i += 1) {
    k_max = 1 + (rand() >> 8) % 10;
    for (k = 0; k <= k_max - 1; k += 1) {
      j = (rand() >> 8) % A -> n;
      s1 = (rand()) / ((double )((double )2147483647));
      sp_set_val(A,i,j,s1);
    }
  }
/* to make it likely that A is nonsingular, use pivot... */
  for (i = 0; i <= 2 * A -> n - 1; i += 1) {
    j = (rand() >> 8) % A -> n;
    k = (rand() >> 8) % A -> n;
    px_transp(px,j,k);
  }
  for (i = 0; i <= A -> n - 1; i += 1) {
    sp_set_val(A,i,px -> pe[i],1.0);
  }
  return A;
}
#endif

void stat_test1(par)
int par;
{
  static MAT *AT = (MAT *)((void *)0);
  static VEC *xt1 = (VEC *)((void *)0);
  static VEC *yt1 = (VEC *)((void *)0);
  static VEC *xt2 = (VEC *)((void *)0);
  static VEC *yt2 = (VEC *)((void *)0);
  static VEC *xt3 = (VEC *)((void *)0);
  static VEC *yt3 = (VEC *)((void *)0);
  static VEC *xt4 = (VEC *)((void *)0);
  static VEC *yt4 = (VEC *)((void *)0);
  AT = m_resize(AT,10,10);
  xt1 = v_resize(xt1,10);
  yt1 = v_resize(yt1,10);
  xt2 = v_resize(xt2,10);
  yt2 = v_resize(yt2,10);
  xt3 = v_resize(xt3,10);
  yt3 = v_resize(yt3,10);
  xt4 = v_resize(xt4,10);
  yt4 = v_resize(yt4,10);
  mem_stat_reg_list((void **)(&AT),0,0,"memtort.c",248);
#ifdef ANSI_C
  mem_stat_reg_vars(0,3,"memtort.c",251,&xt1,&xt2,&xt3,&xt4,&yt1,&yt2,&yt3,&yt4,(void *)0);
#else
#ifdef VARARGS
#else
#endif
#endif
  v_rand(xt1);
  m_rand(AT);
  mv_mlt(AT,xt1,yt1);
}

void stat_test2(par)
int par;
{
  static PERM *px = (PERM *)((void *)0);
  static IVEC *ixt = (IVEC *)((void *)0);
  static IVEC *iyt = (IVEC *)((void *)0);
  px = px_resize(px,10);
  ixt = iv_resize(ixt,10);
  iyt = iv_resize(iyt,10);
  mem_stat_reg_list((void **)(&px),2,0,"memtort.c",286);
  mem_stat_reg_list((void **)(&ixt),4,0,"memtort.c",287);
  mem_stat_reg_list((void **)(&iyt),4,0,"memtort.c",288);
  px_rand(px);
  px_inv(px,px);
}
#ifdef SPARSE

void stat_test3(par)
int par;
{
  static SPMAT *AT = (SPMAT *)((void *)0);
  static VEC *xt = (VEC *)((void *)0);
  static VEC *yt = (VEC *)((void *)0);
  static SPROW *r = (SPROW *)((void *)0);
  if (AT == ((SPMAT *)((void *)0))) 
    AT = gen_non_symm(100,100);
   else 
    AT = sp_resize(AT,100,100);
  xt = v_resize(xt,100);
  yt = v_resize(yt,100);
  if (r == ((void *)0)) 
    r = sprow_get(100);
  mem_stat_reg_list((void **)(&AT),7,0,"memtort.c",310);
  mem_stat_reg_list((void **)(&xt),3,0,"memtort.c",311);
  mem_stat_reg_list((void **)(&yt),3,0,"memtort.c",312);
  mem_stat_reg_list((void **)(&r),6,0,"memtort.c",313);
  v_rand(xt);
  sp_mv_mlt(AT,xt,yt);
}
#endif
#ifdef COMPLEX

void stat_test4(par)
int par;
{
  static ZMAT *AT = (ZMAT *)((void *)0);
  static ZVEC *xt = (ZVEC *)((void *)0);
  static ZVEC *yt = (ZVEC *)((void *)0);
  AT = zm_resize(AT,10,10);
  xt = zv_resize(xt,10);
  yt = zv_resize(yt,10);
  mem_stat_reg_list((void **)(&AT),9,0,"memtort.c",332);
  mem_stat_reg_list((void **)(&xt),8,0,"memtort.c",333);
  mem_stat_reg_list((void **)(&yt),8,0,"memtort.c",334);
  zv_rand(xt);
  zm_rand(AT);
  zmv_mlt(AT,xt,yt);
}
#endif

void main(argc,argv)
int argc;
char *argv[];
{
  VEC *x = (VEC *)((void *)0);
  VEC *y = (VEC *)((void *)0);
  VEC *z = (VEC *)((void *)0);
  PERM *pi1 = (PERM *)((void *)0);
  PERM *pi2 = (PERM *)((void *)0);
  PERM *pi3 = (PERM *)((void *)0);
  MAT *A = (MAT *)((void *)0);
  MAT *B = (MAT *)((void *)0);
  MAT *C = (MAT *)((void *)0);
#ifdef SPARSE
  SPMAT *sA;
  SPMAT *sB;
  SPROW *r;
#endif
  IVEC *ix = (IVEC *)((void *)0);
  IVEC *iy = (IVEC *)((void *)0);
  IVEC *iz = (IVEC *)((void *)0);
  int m;
  int n;
  int i;
  int j;
  int deg;
  int k;
  double s1;
  double s2;
#ifdef COMPLEX
  ZVEC *zx = (ZVEC *)((void *)0);
  ZVEC *zy = (ZVEC *)((void *)0);
  ZVEC *zz = (ZVEC *)((void *)0);
  ZMAT *zA = (ZMAT *)((void *)0);
  ZMAT *zB = (ZMAT *)((void *)0);
  ZMAT *zC = (ZMAT *)((void *)0);
  complex ONE;
#endif
/* variables for testing attaching new lists of types  */
  FOO_1 *foo_1;
  FOO_2 *foo_2;
  mem_info_on(1);
#if defined(ANSI_C) || defined(VARARGS)
  printf("# Testing %s...\n","vector initialize, copy & resize");
  n = v_get_vars(15,&x,&y,&z,(VEC **)((void *)0));
  if (n != 3) {
    printf("Error: %s error: line %d\n","v_get_vars",376);
    printf(" n = %d (should be 3)\n",n);
  }
  v_rand(x);
  v_rand(y);
  z = _v_copy(x,z,0);
  if (_v_norm2((v_sub(x,z,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","v_get_vars",384);
  _v_copy(x,y,0);
  n = v_resize_vars(10,&x,&y,&z,(void *)0);
  if (n != 3 || _v_norm2((v_sub(x,y,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","VEC copy/resize",388);
  n = v_resize_vars(20,&x,&y,&z,(void *)0);
  if (n != 3 || _v_norm2((v_sub(x,y,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","VEC resize",392);
  n = v_free_vars(&x,&y,&z,(void *)0);
  if (n != 3) 
    printf("Error: %s error: line %d\n","v_free_vars",396);
/* IVEC */
  printf("# Testing %s...\n","int vector initialise, copy & resize");
  n = iv_get_vars(15,&ix,&iy,&iz,(void *)0);
  if (n != 3) {
    printf("Error: %s error: line %d\n","iv_get_vars",403);
    printf(" n = %d (should be 3)\n",n);
  }
  
#pragma omp parallel for private (i)
  for (i = 0; ((unsigned int )i) <= ix -> dim - 1; i += 1) {
    ix -> ive[i] = 2 * i - 1;
    iy -> ive[i] = 3 * i + 2;
  }
  iz = iv_add(ix,iy,iz);
  for (i = 0; ((unsigned int )i) <= ix -> dim - 1; i += 1) {
    if (iz -> ive[i] != 5 * i + 1) 
      printf("Error: %s error: line %d\n","iv_get_vars",413);
  }
  n = iv_resize_vars(10,&ix,&iy,&iz,(void *)0);
  if (n != 3) 
    printf("Error: %s error: line %d\n","IVEC copy/resize",416);
  iv_add(ix,iy,iz);
  for (i = 0; ((unsigned int )i) <= ix -> dim - 1; i += 1) {
    if (iz -> ive[i] != 5 * i + 1) 
      printf("Error: %s error: line %d\n","IVEC copy/resize",421);
  }
  n = iv_resize_vars(20,&ix,&iy,&iz,(void *)0);
  if (n != 3) 
    printf("Error: %s error: line %d\n","IVEC resize",424);
  iv_add(ix,iy,iz);
  for (i = 0; i <= 9; i += 1) {
    if (iz -> ive[i] != 5 * i + 1) 
      printf("Error: %s error: line %d\n","IVEC copy/resize",429);
  }
  n = iv_free_vars(&ix,&iy,&iz,(void *)0);
  if (n != 3) 
    printf("Error: %s error: line %d\n","iv_free_vars",433);
/* MAT */
  printf("# Testing %s...\n","matrix initialise, copy & resize");
  n = m_get_vars(10,10,&A,&B,&C,(void *)0);
  if (n != 3) {
    printf("Error: %s error: line %d\n","m_get_vars",439);
    printf(" n = %d (should be 3)\n",n);
  }
  m_rand(A);
  m_rand(B);
  C = _m_copy(A,C,0,0);
  if (m_norm_inf((m_sub(A,C,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","MAT copy",447);
  _m_copy(A,B,0,0);
  n = m_resize_vars(5,5,&A,&B,&C,(void *)0);
  if (n != 3 || m_norm_inf((m_sub(A,B,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","MAT copy/resize",451);
  n = m_resize_vars(20,20,&A,&B,(void *)0);
  if (m_norm_inf((m_sub(A,B,C))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","MAT resize",455);
  k = m_free_vars(&A,&B,&C,(void *)0);
  if (k != 3) 
    printf("Error: %s error: line %d\n","MAT free",459);
/* PERM */
  printf("# Testing %s...\n","permutation initialise, inverting & permuting vectors");
  n = px_get_vars(15,&pi1,&pi2,&pi3,(void *)0);
  if (n != 3) {
    printf("Error: %s error: line %d\n","px_get_vars",465);
    printf(" n = %d (should be 3)\n",n);
  }
  v_get_vars(15,&x,&y,&z,(void *)0);
  px_rand(pi1);
  v_rand(x);
  px_vec(pi1,x,z);
  y = v_resize(y,(x -> dim));
  pxinv_vec(pi1,z,y);
  if (_v_norm2((v_sub(x,y,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","PERMute vector",477);
  pi2 = px_inv(pi1,pi2);
  pi3 = px_mlt(pi1,pi2,pi3);
  for (i = 0; ((unsigned int )i) <= pi3 -> size - 1; i += 1) {
    if (pi3 -> pe[i] != i) 
      printf("Error: %s error: line %d\n","PERM inverse/multiply",482);
  }
  px_resize_vars(20,&pi1,&pi2,&pi3,(void *)0);
  v_resize_vars(20,&x,&y,&z,(void *)0);
  px_rand(pi1);
  v_rand(x);
  px_vec(pi1,x,z);
  pxinv_vec(pi1,z,y);
  if (_v_norm2((v_sub(x,y,z)),((VEC *)((void *)0))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","PERMute vector",492);
  pi2 = px_inv(pi1,pi2);
  pi3 = px_mlt(pi1,pi2,pi3);
  for (i = 0; ((unsigned int )i) <= pi3 -> size - 1; i += 1) {
    if (pi3 -> pe[i] != i) 
      printf("Error: %s error: line %d\n","PERM inverse/multiply",497);
  }
  n = px_free_vars(&pi1,&pi2,&pi3,(void *)0);
  if (n != 3) 
    printf("Error: %s error: line %d\n","PERM px_free_vars",501);
#ifdef SPARSE   
/* set up two random sparse matrices */
  m = 120;
  n = 100;
  deg = 5;
  printf("# Testing %s...\n","allocating sparse matrices");
  k = sp_get_vars(m,n,deg,&sA,&sB,(void *)0);
  if (k != 2) {
    printf("Error: %s error: line %d\n","sp_get_vars",511);
    printf(" n = %d (should be 2)\n",k);
  }
  printf("# Testing %s...\n","setting and getting matrix entries");
  for (k = 0; k <= m * deg - 1; k += 1) {
    i = (rand() >> 8) % m;
    j = (rand() >> 8) % n;
    sp_set_val(sA,i,j,(rand()) / ((double )((double )2147483647)));
    i = (rand() >> 8) % m;
    j = (rand() >> 8) % n;
    sp_set_val(sB,i,j,(rand()) / ((double )((double )2147483647)));
  }
  for (k = 0; k <= 9; k += 1) {
    s1 = (rand()) / ((double )((double )2147483647));
    i = (rand() >> 8) % m;
    j = (rand() >> 8) % n;
    sp_set_val(sA,i,j,s1);
    s2 = sp_get_val(sA,i,j);
    if (fabs(s1 - s2) >= ((double )2.22044604925031308084726333618164062e-16L)) {
      printf(" s1 = %g, s2 = %g, |s1 - s2| = %g\n",s1,s2,(fabs(s1 - s2)));
      break; 
    }
  }
  if (k < 10) 
    printf("Error: %s error: line %d\n","sp_set_val()/sp_get_val()",539);
/* check column access paths */
  printf("# Testing %s...\n","resizing and access paths");
  k = sp_resize_vars(sA -> m + 10,sA -> n + 10,&sA,&sB,(void *)0);
  if (k != 2) {
    printf("Error: %s error: line %d\n","sp_get_vars",545);
    printf(" n = %d (should be 2)\n",k);
  }
  for (k = 0; k <= 19; k += 1) {
    i = sA -> m - 1 - (rand() >> 8) % 10;
    j = sA -> n - 1 - (rand() >> 8) % 10;
    s1 = (rand()) / ((double )((double )2147483647));
    sp_set_val(sA,i,j,s1);
    if (fabs(s1 - sp_get_val(sA,i,j)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
      break; 
  }
  if (k < 20) 
    printf("Error: %s error: line %d\n","sp_resize()",559);
  sp_col_access(sA);
  if (!chk_col_access(sA)) {
    printf("Error: %s error: line %d\n","sp_col_access()",563);
  }
  sp_diag_access(sA);
  for (i = 0; i <= sA -> m - 1; i += 1) {
    r = &sA -> row[i];
    if (r -> diag != sprow_idx(r,i)) 
      break; 
  }
  if (i < sA -> m) {
    printf("Error: %s error: line %d\n","sp_diag_access()",574);
  }
  k = sp_free_vars(&sA,&sB,(void *)0);
  if (k != 2) 
    printf("Error: %s error: line %d\n","sp_free_vars",579);
#endif  /* SPARSE */   
#ifdef COMPLEX
/* complex stuff */
  ONE = zmake(1.0,0.0);
  printf("# ONE = ");
  z_foutput(stdout,ONE);
  printf("# Check: MACHEPS = %g\n",(double )2.22044604925031308084726333618164062e-16L);
/* allocate, initialise, copy and resize operations */
/* ZVEC */
  printf("# Testing %s...\n","vector initialise, copy & resize");
  zv_get_vars(12,&zx,&zy,&zz,(void *)0);
  zv_rand(zx);
  zv_rand(zy);
  zz = _zv_copy(zx,zz,0);
  if (_zv_norm2((zv_sub(zx,zz,zz)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZVEC copy",598);
  _zv_copy(zx,zy,0);
  zv_resize_vars(10,&zx,&zy,(void *)0);
  if (_zv_norm2((zv_sub(zx,zy,zz)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZVEC copy/resize",603);
  zv_resize_vars(20,&zx,&zy,(void *)0);
  if (_zv_norm2((zv_sub(zx,zy,zz)),(VEC *)((void *)0)) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","VZEC resize",607);
  zv_free_vars(&zx,&zy,&zz,(void *)0);
/* ZMAT */
  printf("# Testing %s...\n","matrix initialise, copy & resize");
  zm_get_vars(8,5,&zA,&zB,&zC,(void *)0);
  zm_rand(zA);
  zm_rand(zB);
  zC = _zm_copy(zA,zC,0,0);
  if (zm_norm_inf((zm_sub(zA,zC,zC))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZMAT copy",619);
  _zm_copy(zA,zB,0,0);
  zm_resize_vars(3,5,&zA,&zB,&zC,(void *)0);
  if (zm_norm_inf((zm_sub(zA,zB,zC))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZMAT copy/resize",625);
  zm_resize_vars(20,20,&zA,&zB,&zC,(void *)0);
  if (zm_norm_inf((zm_sub(zA,zB,zC))) >= ((double )2.22044604925031308084726333618164062e-16L)) 
    printf("Error: %s error: line %d\n","ZMAT resize",629);
  zm_free_vars(&zA,&zB,&zC,(void *)0);
#endif /* COMPLEX */
#endif  /* if defined(ANSI_C) || defined(VARARGS) */
  printf("# test of mem_info_bytes and mem_info_numvar\n");
  printf("  TYPE VEC: %ld bytes allocated, %d variables allocated\n",(mem_info_bytes(3,0)),(mem_info_numvar(3,0)));
  printf("# Testing %s...\n","static memory test");
  mem_info_on(1);
  mem_stat_mark(1);
  for (i = 0; i <= 99; i += 1) {
    stat_test1(i);
  }
  mem_stat_free_list(1,0);
  mem_stat_mark(1);
  for (i = 0; i <= 99; i += 1) {
    stat_test1(i);
#ifdef COMPLEX
    stat_test4(i);
#endif
  }
  mem_stat_mark(2);
  for (i = 0; i <= 99; i += 1) {
    stat_test2(i);
  }
  mem_stat_mark(3);
#ifdef SPARSE
  for (i = 0; i <= 99; i += 1) {
    stat_test3(i);
  }
#endif
  mem_info_file(stdout,0);
  mem_dump_list(stdout,0);
  mem_stat_free_list(1,0);
  mem_stat_free_list(3,0);
  mem_stat_mark(4);
  for (i = 0; i <= 99; i += 1) {
    stat_test1(i);
#ifdef COMPLEX
    stat_test4(i);
#endif
  }
  mem_stat_dump(stdout,0);
  if (mem_stat_show_mark() != 4) {
    printf("Error: %s error: line %d\n","not 4 in mem_stat_show_mark()",681);
  }
  mem_stat_free_list(2,0);
  mem_stat_free_list(4,0);
  if (mem_stat_show_mark() != 0) {
    printf("Error: %s error: line %d\n","not 0 in mem_stat_show_mark()",688);
  }
/* add new list of types */
  mem_attach_list(1,(sizeof(foo_type_name) / sizeof(( *foo_type_name))),foo_type_name,foo_free_func,foo_info_sum);
  if (!mem_is_list_attached(1)) 
    printf("Error: %s error: line %d\n","list FOO_LIST is not attached",696);
  mem_dump_list(stdout,1);
  foo_1 = foo_1_get(6);
  foo_2 = foo_2_get(3);
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= foo_1 -> dim - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= foo_1 -> fix_dim - 1; j += 1) {
      foo_1 -> a[i][j] = (i + j);
    }
  }
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= foo_2 -> dim - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= foo_2 -> fix_dim - 1; j += 1) {
      foo_2 -> a[i][j] = (i + j);
    }
  }
  printf(" foo_1->a[%d][%d] = %g\n",5,9,foo_1 -> a[5][9]);
  printf(" foo_2->a[%d][%d] = %g\n",2,1,foo_2 -> a[2][1]);
  mem_stat_mark(5);
  mem_stat_reg_list((void **)(&foo_1),1,1,"memtort.c",711);
  mem_stat_reg_list((void **)(&foo_2),2,1,"memtort.c",712);
  mem_stat_dump(stdout,1);
  mem_info_file(stdout,1);
  mem_stat_free_list(5,1);
  mem_stat_dump(stdout,1);
  if (foo_1 != ((void *)0)) 
    printf("Error: %s error: line %d\n"," foo_1 is not released",718);
  if (foo_2 != ((void *)0)) 
    printf("Error: %s error: line %d\n"," foo_2 is not released",720);
  mem_dump_list(stdout,1);
  mem_info_file(stdout,1);
  mem_free_vars(1);
  if (mem_is_list_attached(1)) 
    printf("Error: %s error: line %d\n","list FOO_LIST is not detached",726);
  mem_info_file(stdout,0);
#if REAL == FLOAT
#elif REAL == DOUBLE
  printf("# DOUBLE PRECISION was used\n");
#endif
#define ANSI_OR_VAR
#ifndef ANSI_C
#ifndef VARARGS
#undef ANSI_OR_VAR
#endif
#endif
#ifdef ANSI_OR_VAR
  printf("# you should get: \n");
#if (REAL == FLOAT)
#elif (REAL == DOUBLE)
  printf("#   type VEC: 516 bytes allocated, 3 variables allocated\n");
#endif
  printf("#   and other types are zeros\n");
#endif /*#if defined(ANSI_C) || defined(VARAGS) */
  printf("# Finished memory torture test\n");
  dmalloc_shutdown();
  return ;
}
