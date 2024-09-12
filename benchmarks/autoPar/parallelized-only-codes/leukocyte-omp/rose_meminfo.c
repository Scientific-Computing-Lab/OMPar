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
/* meminfo.c  revised  22/11/93 */
/* 
  contains basic functions, types and arrays 
  to keep track of memory allocation/deallocation
*/
#include <stdio.h>
#include  "matrix.h"
#include  "meminfo.h"
#ifdef COMPLEX   
#include  "zmatrix.h"
#endif
#ifdef SPARSE
#include  "sparse.h"
#include  "iter.h"
#endif
static char rcsid[] = "$Id: meminfo.c,v 1.1 1994/01/13 05:31:39 des Exp $";
/* this array is defined further in this file */
extern MEM_CONNECT mem_connect[5];
/* names of types */
static char *mem_type_names[] = {("MAT"), ("BAND"), ("PERM"), ("VEC"), ("IVEC"), ("ITER"), ("SPROW"), ("SPMAT"), ("ZVEC"), ("ZMAT")
#ifdef SPARSE
#endif
#ifdef COMPLEX   
#endif
};
#define MEM_NUM_STD_TYPES  (sizeof(mem_type_names)/sizeof(mem_type_names[0]))
/* local array for keeping track of memory */
static MEM_ARRAY mem_info_sum[10];
/* for freeing various types */
static int (*mem_free_funcs[10])(void *) = {(wrapped_m_free), (wrapped_bd_free), (wrapped_px_free), (wrapped_v_free), (wrapped_iv_free), (wrapped_iter_free), (wrapped_sprow_free), (wrapped_sp_free), (wrapped_zv_free), (wrapped_zm_free)
#ifdef SPARSE
#endif
#ifdef COMPLEX
#endif
};
/* it is a global variable for passing 
   pointers to local arrays defined here */
MEM_CONNECT mem_connect[5] = {/* Need explicit braces: is this where we insert the class name? */ {(mem_type_names), (mem_free_funcs), ((sizeof(mem_type_names) / sizeof(mem_type_names[0]))), (mem_info_sum)}};
/* attach a new list of types */
#ifndef ANSI_C
/* number of a list and number of types there */
/* list of names of types */
/* list of releasing functions */
/* local table */
#else

int mem_attach_list(int list,int ntypes,char *type_names[],int (*free_funcs[])(void *),MEM_ARRAY info_sum[])
#endif
{
  if (list < 0 || list >= 5) 
    return - 1;
  if (type_names == ((void *)0) || free_funcs == ((void *)0) || info_sum == ((void *)0) || ntypes < 0) 
    return - 1;
/* if a list exists do not overwrite */
  if (mem_connect[list] . ntypes != 0) 
    ev_err("meminfo.c",21,128,"mem_attach_list",0);
  mem_connect[list] . ntypes = ntypes;
  mem_connect[list] . type_names = type_names;
  mem_connect[list] . free_funcs = free_funcs;
  mem_connect[list] . info_sum = info_sum;
  return 0;
}
/* release a list of types */
#ifndef ANSI_C
#else

int mem_free_vars(int list)
#endif
{
  if (list < 0 || list >= 5) 
    return - 1;
  mem_connect[list] . ntypes = 0;
  mem_connect[list] . type_names = ((void *)0);
  mem_connect[list] . free_funcs = ((void *)0);
  mem_connect[list] . info_sum = ((void *)0);
  return 0;
}
/* check if list is attached */
#ifndef ANSI_C
#else

int mem_is_list_attached(int list)
#endif
{
  if (list < 0 || list >= 5) 
    return 0;
  if (mem_connect[list] . type_names != ((void *)0) && mem_connect[list] . free_funcs != ((void *)0) && mem_connect[list] . info_sum != ((void *)0)) 
    return 1;
   else 
    return 0;
}
/* to print out the contents of mem_connect[list] */
#ifndef MEX
#ifndef ANSI_C
#else

void mem_dump_list(FILE *fp,int list)
#endif
{
  int i;
  MEM_CONNECT *mlist;
  if (list < 0 || list >= 5) 
    return ;
  mlist = &mem_connect[list];
  fprintf(fp," %15s[%d]:\n","CONTENTS OF mem_connect",list);
  fprintf(fp," %-7s   %-12s   %-9s   %s\n","name of","alloc.","# alloc.","address");
  fprintf(fp," %-7s   %-12s   %-9s   %s\n"," type","bytes","variables","of *_free()");
  for (i = 0; ((unsigned int )i) <= mlist -> ntypes - 1; i += 1) {
    fprintf(fp,"  %-7s   %-12ld   %-9d   %p\n",mlist -> type_names[i],mlist -> info_sum[i] . bytes,mlist -> info_sum[i] . numvar,mlist -> free_funcs[i]);
  }
  fprintf(fp,"\n");
}
#endif /* MEX */
/*=============================================================*/
/* local variables */
static int mem_switched_on = 0;
/* on/off */
/* switch on/off memory info */
#ifndef ANSI_C
#else

int mem_info_on(int sw)
#endif
{
  int old = mem_switched_on;
  mem_switched_on = sw;
  return old;
}
#ifdef ANSI_C

int mem_info_is_on()
#else
#endif
{
  return mem_switched_on;
}
/* information about allocated memory */
/* return the number of allocated bytes for type 'type' */
#ifndef ANSI_C
#else

long mem_info_bytes(int type,int list)
#endif
{
  if (list < 0 || list >= 5) 
    return 0l;
  if (!mem_switched_on || type < 0 || type >= mem_connect[list] . ntypes || mem_connect[list] . free_funcs[type] == ((void *)0)) 
    return 0l;
  return mem_connect[list] . info_sum[type] . bytes;
}
/* return the number of allocated variables for type 'type' */
#ifndef ANSI_C
#else

int mem_info_numvar(int type,int list)
#endif
{
  if (list < 0 || list >= 5) 
    return 0l;
  if (!mem_switched_on || type < 0 || type >= mem_connect[list] . ntypes || mem_connect[list] . free_funcs[type] == ((void *)0)) 
    return 0l;
  return mem_connect[list] . info_sum[type] . numvar;
}
#ifndef MEX
/* print out memory info to the file fp */
#ifndef ANSI_C
#else

void mem_info_file(FILE *fp,int list)
#endif
{
  unsigned int type;
  long t = 0l;
  long d;
  int n = 0;
  int nt = 0;
  MEM_CONNECT *mlist;
  if (!mem_switched_on) 
    return ;
  if (list < 0 || list >= 5) 
    return ;
  if (list == 0) 
    fprintf(fp," MEMORY INFORMATION (standard types):\n");
   else 
    fprintf(fp," MEMORY INFORMATION (list no. %d):\n",list);
  mlist = &mem_connect[list];
  for (type = 0; type <= mlist -> ntypes - 1; type += 1) {
    if (mlist -> type_names[type] == ((void *)0)) 
      continue; 
    d = mlist -> info_sum[type] . bytes;
    t += d;
    n = mlist -> info_sum[type] . numvar;
    nt += n;
    fprintf(fp," type %-7s %10ld alloc. byte%c  %6d alloc. variable%c\n",mlist -> type_names[type],d,(d != 1?'s' : ' '),n,(n != 1?'s' : ' '));
  }
  fprintf(fp," %-12s %10ld alloc. byte%c  %6d alloc. variable%c\n\n","total:",t,(t != 1?'s' : ' '),nt,(nt != 1?'s' : ' '));
}
#endif
/* function for memory information */
/* mem_bytes_list
   
   Arguments:
   type - the number of type;
   old_size - old size of allocated memory (in bytes);
   new_size - new size of allocated memory (in bytes);
   list - list of types
   */
#ifndef ANSI_C
#else

void mem_bytes_list(int type,int old_size,int new_size,int list)
#endif
{
  MEM_CONNECT *mlist;
  if (list < 0 || list >= 5) 
    return ;
  mlist = &mem_connect[list];
  if (type < 0 || type >= mlist -> ntypes || mlist -> free_funcs[type] == ((void *)0)) 
    return ;
  if (old_size < 0 || new_size < 0) 
    ev_err("meminfo.c",20,365,"mem_bytes_list",0);
  mlist -> info_sum[type] . bytes += (new_size - old_size);
/* check if the number of bytes is non-negative */
  if (old_size > 0) {
    if (mlist -> info_sum[type] . bytes < 0) {
#ifndef MEX
      fprintf(stderr,"\n WARNING !! memory info: allocated memory is less than 0\n");
      fprintf(stderr,"\t TYPE %s \n\n",mlist -> type_names[type]);
      if (!isatty((fileno(stdout)))) {
        fprintf(stdout,"\n WARNING !! memory info: allocated memory is less than 0\n");
        fprintf(stdout,"\t TYPE %s \n\n",mlist -> type_names[type]);
      }
#else
#endif
    }
  }
}
/* mem_numvar_list
   
   Arguments:
   type - the number of type;
   num - # of variables allocated (> 0) or deallocated ( < 0)
   list - list of types
   */
#ifndef ANSI_C
#else

void mem_numvar_list(int type,int num,int list)
#endif
{
  MEM_CONNECT *mlist;
  if (list < 0 || list >= 5) 
    return ;
  mlist = &mem_connect[list];
  if (type < 0 || type >= mlist -> ntypes || mlist -> free_funcs[type] == ((void *)0)) 
    return ;
  mlist -> info_sum[type] . numvar += num;
/* check if the number of variables is non-negative */
  if (num < 0) {
    if (mlist -> info_sum[type] . numvar < 0) {
#ifndef MEX
      fprintf(stderr,"\n WARNING !! memory info: allocated # of variables is less than 0\n");
      fprintf(stderr,"\t TYPE %s \n\n",mlist -> type_names[type]);
      if (!isatty((fileno(stdout)))) {
        fprintf(stdout,"\n WARNING !! memory info: allocated # of variables is less than 0\n");
        fprintf(stdout,"\t TYPE %s \n\n",mlist -> type_names[type]);
      }
#else
#endif
    }
  }
}
