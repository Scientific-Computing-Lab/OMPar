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
/*  mem_stat.c    6/09/93  */
/* Deallocation of static arrays */
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
static char rcsid[] = "$Id: memstat.c,v 1.1 1994/01/13 05:32:44 des Exp $";
/* global variable */
extern MEM_CONNECT mem_connect[5];
/* local type */
typedef struct {
void **var;
/* for &A, where A is a pointer */
int type;
/* type of A */
int mark;
/* what mark is chosen */
char *fname;
/* source file name where last registered */
int line;
/* line # of file where last registered */
}MEM_STAT_STRUCT;
/* local variables */
/* how many marks are used */
static int mem_stat_mark_many = 0;
/* current mark */
static int mem_stat_mark_curr = 0;
static MEM_STAT_STRUCT mem_stat_var[509];
/* array of indices (+1) to mem_stat_var */
static unsigned int mem_hash_idx[509];
/* points to the first unused element in mem_hash_idx */
static unsigned int mem_hash_idx_end = 0;
/* hashing function */
#ifndef ANSI_C
#else

static unsigned int mem_hash(void **ptr)
#endif
{
  unsigned long lp = (unsigned long )ptr;
  return (lp % 509);
}
/* look for a place in mem_stat_var */
#ifndef ANSI_C
#else

static int mem_lookup(void **var)
#endif
{
  int k;
  int j;
  k = (mem_hash(var));
  if (mem_stat_var[k] . var == var) {
    return - 1;
  }
   else if (mem_stat_var[k] . var == ((void *)0)) {
    return k;
  }
   else {
/* look for an empty place */
    j = k;
    while(mem_stat_var[j] . var != var && j < 509 && mem_stat_var[j] . var != ((void *)0))
      j++;
    if (mem_stat_var[j] . var == ((void *)0)) 
      return j;
     else if (mem_stat_var[j] . var == var) 
      return - 1;
     else {
/* if (j == MEM_HASHSIZE) */
      j = 0;
      while(mem_stat_var[j] . var != var && j < k && mem_stat_var[j] . var != ((void *)0))
        j++;
      if (mem_stat_var[j] . var == ((void *)0)) 
        return j;
       else if (mem_stat_var[j] . var == var) 
        return - 1;
       else {
/* if (j == k) */
        fprintf(stderr,"\n WARNING !!! static memory: mem_stat_var is too small\n");
        fprintf(stderr," Increase MEM_HASHSIZE in file: %s (currently = %d)\n\n","meminfo.h",509);
        if (!isatty((fileno(stdout)))) {
          fprintf(stdout,"\n WARNING !!! static memory: mem_stat_var is too small\n");
          fprintf(stdout," Increase MEM_HASHSIZE in file: %s (currently = %d)\n\n","meminfo.h",509);
        }
        ev_err("memstat.c",3,140,"mem_lookup",0);
      }
    }
  }
  return - 1;
}
/* register static variables;
   Input arguments:
     var - variable to be registered,
     type - type of this variable; 
     list - list of types
     fname - source file name where last registered
     line - line number of source file
   returned value < 0  --> error,
   returned value == 0 --> not registered,
   returned value >= 0 --> registered with this mark;
*/
#ifndef ANSI_C
#else

int mem_stat_reg_list(void **var,int type,int list,char *fname,int line)
#endif
{
  int n;
  if (list < 0 || list >= 5) 
    return - 1;
  if (mem_stat_mark_curr == 0) 
    return 0;
/* not registered */
  if (var == ((void *)0)) 
    return - 1;
/* error */
  if (type < 0 || type >= mem_connect[list] . ntypes || mem_connect[list] . free_funcs[type] == ((void *)0)) {
    ev_err("memstat.c",1,183,"mem_stat_reg_list",1);
    return - 1;
  }
  if ((n = mem_lookup(var)) >= 0) {
    mem_stat_var[n] . var = var;
    mem_stat_var[n] . mark = mem_stat_mark_curr;
    mem_stat_var[n] . type = type;
    mem_stat_var[n] . fname = fname;
    mem_stat_var[n] . line = line;
/* save n+1, not n */
    mem_hash_idx[mem_hash_idx_end++] = (n + 1);
  }
  return mem_stat_mark_curr;
}
/* set a mark;
   Input argument:
   mark - positive number denoting a mark;
   returned: 
             mark if mark > 0,
             0 if mark == 0,
	     -1 if mark is negative.
*/
#ifndef ANSI_C
#else

int mem_stat_mark(int mark)
#endif
{
  if (mark < 0) {
    mem_stat_mark_curr = 0;
    return - 1;
/* error */
  }
   else if (mark == 0) {
    mem_stat_mark_curr = 0;
    return 0;
  }
  mem_stat_mark_curr = mark;
  mem_stat_mark_many++;
  return mark;
}
/* deallocate static variables;
   Input argument:
   mark - a positive number denoting the mark;
   Returned:
     -1 if mark < 0 (error);
     0  if mark == 0;
*/
#ifndef ANSI_C
#else

int mem_stat_free_list(int mark,int list)
#endif
{
  unsigned int i;
  unsigned int j;
  int (*free_fn)(void *);
  if (list < 0 || list >= 5 || mem_connect[list] . free_funcs == ((void *)0)) 
    return - 1;
  if (mark < 0) {
    mem_stat_mark_curr = 0;
    return - 1;
  }
   else if (mark == 0) {
    mem_stat_mark_curr = 0;
    return 0;
  }
  if (mem_stat_mark_many <= 0) {
    ev_err("memstat.c",2,265,"mem_stat_free",1);
    return - 1;
  }
#ifdef DEBUG
#endif /* DEBUG */
/* deallocate the marked variables */
  for (i = 0; i <= mem_hash_idx_end - 1; i += 1) {
    j = mem_hash_idx[i];
    if (j == 0) 
      continue; 
     else {
      j--;
      if (mem_stat_var[j] . mark == mark) {
        free_fn = mem_connect[list] . free_funcs[mem_stat_var[j] . type];
#ifdef DEBUG
#endif /* DEBUG */
        if (free_fn != ((void *)0)) 
          ( *free_fn)(( *mem_stat_var[j] . var));
         else 
          ev_err("memstat.c",1,287,"mem_stat_free",1);
         *mem_stat_var[j] . var = ((void *)0);
        mem_stat_var[j] . var = ((void *)0);
        mem_stat_var[j] . mark = 0;
        mem_stat_var[j] . fname = ((void *)0);
        mem_stat_var[j] . line = 0;
        mem_hash_idx[i] = 0;
      }
    }
  }
  while(mem_hash_idx_end > 0 && mem_hash_idx[mem_hash_idx_end - 1] == 0)
    mem_hash_idx_end--;
  mem_stat_mark_curr = 0;
  mem_stat_mark_many--;
  return 0;
}
/* only for diagnostic purposes */
#ifndef ANSI_C
#else

void mem_stat_dump(FILE *fp,int list)
#endif
{
  unsigned int i;
  unsigned int j;
  unsigned int k = 1;
  if (list < 0 || list >= 5 || mem_connect[list] . free_funcs == ((void *)0)) 
    return ;
  fprintf(fp," Array mem_stat_var (list no. %d):\n",list);
  for (i = 0; i <= mem_hash_idx_end - 1; i += 1) {
    j = mem_hash_idx[i];
    if (j == 0) 
      continue; 
     else {
      j--;
      fprintf(fp," %d.  var = 0x%p, type = %s, mark = %d\n",k,mem_stat_var[j] . var,(mem_stat_var[j] . type < mem_connect[list] . ntypes && mem_connect[list] . free_funcs[mem_stat_var[j] . type] != ((void *)0)?mem_connect[list] . type_names[(int )mem_stat_var[j] . type] : "???"),mem_stat_var[j] . mark);
      k++;
    }
  }
  fprintf(fp,"\n");
}
/* query function about the current mark */
#ifdef ANSI_C

int mem_stat_show_mark()
#else
#endif
{
  return mem_stat_mark_curr;
}
/* Varying number of arguments */
#ifdef ANSI_C
/* To allocate memory to many arguments. 
   The function should be called:
   mem_stat_vars(list,type,&v1,&v2,&v3,...,VNULL);
   where 
     int list,type;
     void **v1, **v2, **v3,...;
     The last argument should be VNULL ! 
     type is the type of variables v1,v2,v3,...
     (of course they must be of the same type)
*/

int mem_stat_reg_vars(int list,int type,char *fname,int line,... )
{
  va_list ap;
  int i = 0;
  void **par;
/* va_start(ap, type); */
  __builtin_va_start(ap,line);
/* Changed for Linux 7th Oct, 2003 */
  while(par = ((void **)(sizeof(void **)))){
/* NULL ends the list*/
    mem_stat_reg_list(par,type,list,fname,line);
    i++;
  }
  __builtin_va_end(ap);
  return i;
}
#elif VARARGS
/* old varargs is used */
/* To allocate memory to many arguments. 
   The function should be called:
   mem_stat_vars(list,type,&v1,&v2,&v3,...,VNULL);
   where 
     int list,type;
     void **v1, **v2, **v3,...;
     The last argument should be VNULL ! 
     type is the type of variables v1,v2,v3,...
     (of course they must be of the same type)
*/
/* NULL ends the list*/
#endif
