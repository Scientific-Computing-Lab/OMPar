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
/*
  File with basic error-handling operations
*/
static char rcsid[] = "$Id: err.c,v 1.6 1995/01/30 14:49:14 des Exp $";
#include	<stdio.h>
#include	<setjmp.h>
#include	<ctype.h>
#include	<unistd.h>
#include        "err.h"
#ifdef SYSV
/* AT&T System V */
#include	<sys/signal.h>
#else
/* something else -- assume BSD or ANSI C */
#include	<signal.h>
#endif
#define		FALSE	0
#define		TRUE	1
#define	EF_EXIT		0
#define	EF_ABORT	1
#define	EF_JUMP		2
#define	EF_SILENT	3
/* The only error caught in this file! */
#define	E_SIGNAL	16
static char *err_mesg[] = {("unknown error"), ("sizes of objects don't match"), ("index out of bounds"), ("can't allocate memory"), ("singular matrix"), ("matrix not positive definite"), ("incorrect format input"), ("bad input file/device"), ("NULL objects passed"), ("matrix not square"), ("object out of range"), ("can't do operation in situ for non-square matrix"), ("can't do operation in situ"), ("excessive number of iterations"), ("convergence criterion failed"), ("bad starting value"), ("floating exception"), ("internal inconsistency (data structure)"), ("unexpected end-of-file"), ("shared vectors (cannot release them)"), ("negative argument"), ("cannot overwrite object"), ("breakdown in iterative method")
/* 0 */
/* 1 */
/* 2 */
/* 3 */
/* 4 */
/* 5 */
/* 6 */
/* 7 */
/* 8 */
/* 9 */
/* 10 */
/* 11 */
/* 12 */
/* 13 */
/* 14 */
/* 15 */
/* 16 */
/* 17 */
/* 18 */
/* 19 */
/* 20 */
/* 21 */
/* 22 */
};
#define	MAXERR	(sizeof(err_mesg)/sizeof(char *))
static char *warn_mesg[] = {("unknown warning"), ("wrong type number (use macro TYPE_*)"), ("no corresponding mem_stat_mark"), ("computed norm of a residual is less than 0"), ("resizing a shared vector")
/* 0 */
/* 1 */
/* 2 */
/* 3 */
/* 4 */
};
#define MAXWARN  (sizeof(warn_mesg)/sizeof(char *))
#define	MAX_ERRS	100
jmp_buf restart;
/* array of pointers to lists of errors */
typedef struct {
char **listp;
/* pointer to a list of errors */
unsigned int len;
/* length of the list */
unsigned int warn;
/* =FALSE - errors, =TRUE - warnings */
}Err_list;
static Err_list err_list[10] = {/* Need explicit braces: is this where we insert the class name? */ {(err_mesg), ((sizeof(err_mesg) / sizeof(char *))), (0)}, 
/* basic errors list */
/* Need explicit braces: is this where we insert the class name? */ {(warn_mesg), ((sizeof(warn_mesg) / sizeof(char *))), (1)}
/* basic warnings list */
};
static int err_list_end = 2;
/* number of elements in err_list */
/* attach a new list of errors pointed by err_ptr
   or change a previous one;
   list_len is the number of elements in the list;
   list_num is the list number;
   warn == FALSE - errors (stop the program),
   warn == TRUE - warnings (continue the program);
   Note: lists numbered 0 and 1 are attached automatically,
   you do not need to do it
   */
#ifndef ANSI_C
#else

int err_list_attach(int list_num,int list_len,char **err_ptr,int warn)
#endif
{
  if (list_num < 0 || list_len <= 0 || err_ptr == ((char **)((void *)0))) 
    return - 1;
  if (list_num >= 10) {
    fprintf(stderr,"\n file \"%s\": %s %s\n","err.c","increase the value of ERR_LIST_MAX_LEN","in matrix.h and zmatdef.h");
    if (!isatty((fileno(stdout)))) 
      fprintf(stderr,"\n file \"%s\": %s %s\n","err.c","increase the value of ERR_LIST_MAX_LEN","in matrix.h and zmatdef.h");
    printf("Exiting program\n");
    exit(0);
  }
  if (err_list[list_num] . listp != ((char **)((void *)0)) && err_list[list_num] . listp != err_ptr) 
    free(((char *)err_list[list_num] . listp));
  err_list[list_num] . listp = err_ptr;
  err_list[list_num] . len = list_len;
  err_list[list_num] . warn = warn;
  err_list_end = list_num + 1;
  return list_num;
}
/* release the error list numbered list_num */
#ifndef ANSI_C
#else

int err_list_free(int list_num)
#endif
{
  if (list_num < 0 || list_num >= err_list_end) 
    return - 1;
  if (err_list[list_num] . listp != ((char **)((void *)0))) {
    err_list[list_num] . listp = ((char **)((void *)0));
    err_list[list_num] . len = 0;
    err_list[list_num] . warn = 0;
  }
  return 0;
}
/* check if list_num is attached;
   return FALSE if not;
   return TRUE if yes
   */
#ifndef ANSI_C
#else

int err_is_list_attached(int list_num)
#endif
{
  if (list_num < 0 || list_num >= err_list_end) 
    return 0;
  if (err_list[list_num] . listp != ((char **)((void *)0))) 
    return 1;
  return 0;
}
/* other local variables */
static int err_flag = 0;
static int num_errs = 0;
static int cnt_errs = 1;
/* set_err_flag -- sets err_flag -- returns old err_flag */
#ifndef ANSI_C
#else

int set_err_flag(int flag)
#endif
{
  int tmp;
  tmp = err_flag;
  err_flag = flag;
  return tmp;
}
/* count_errs -- sets cnt_errs (TRUE/FALSE) & returns old value */
#ifndef ANSI_C
#else

int count_errs(int flag)
#endif
{
  int tmp;
  tmp = cnt_errs;
  cnt_errs = flag;
  return tmp;
}
/* ev_err -- reports error (err_num) in file "file" at line "line_num" and
   returns to user error handler;
   list_num is an error list number (0 is the basic list 
   pointed by err_mesg, 1 is the basic list of warnings)
 */
#ifndef ANSI_C
#else

int ev_err(const char *file,int err_num,int line_num,const char *fn_name,int list_num)
#endif
{
  int num;
  if (err_num < 0) 
    err_num = 0;
  if (list_num < 0 || list_num >= err_list_end || err_list[list_num] . listp == ((char **)((void *)0))) {
    fprintf(stderr,"\n Not (properly) attached list of errors: list_num = %d\n",list_num);
    fprintf(stderr," Call \"err_list_attach\" in your program\n");
    if (!isatty((fileno(stdout)))) {
      fprintf(stderr,"\n Not (properly) attached list of errors: list_num = %d\n",list_num);
      fprintf(stderr," Call \"err_list_attach\" in your program\n");
    }
    printf("\nExiting program\n");
    exit(0);
  }
  num = err_num;
  if (num >= err_list[list_num] . len) 
    num = 0;
  if (cnt_errs && ++num_errs >= 100) 
/* too many errors */
{
    fprintf(stderr,"\n\"%s\", line %d: %s in function %s()\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
    if (!isatty((fileno(stdout)))) 
      fprintf(stdout,"\n\"%s\", line %d: %s in function %s()\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
    printf("Sorry, too many errors: %d\n",num_errs);
    printf("Exiting program\n");
    exit(0);
  }
  if (err_list[list_num] . warn) 
    switch(err_flag){
      case 3:
      break; 
      default:
      fprintf(stderr,"\n\"%s\", line %d: %s in function %s()\n\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
      if (!isatty((fileno(stdout)))) 
        fprintf(stdout,"\n\"%s\", line %d: %s in function %s()\n\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
      break; 
    }
   else 
    switch(err_flag){
      case 3:
      longjmp(restart,(err_num == 0?- 1 : err_num));
      break; 
      case 1:
      fprintf(stderr,"\n\"%s\", line %d: %s in function %s()\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
      if (!isatty((fileno(stdout)))) 
        fprintf(stdout,"\n\"%s\", line %d: %s in function %s()\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
      abort();
      break; 
      case 2:
      fprintf(stderr,"\n\"%s\", line %d: %s in function %s()\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
      if (!isatty((fileno(stdout)))) 
        fprintf(stdout,"\n\"%s\", line %d: %s in function %s()\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
      longjmp(restart,(err_num == 0?- 1 : err_num));
      break; 
      default:
      fprintf(stderr,"\n\"%s\", line %d: %s in function %s()\n\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
      if (!isatty((fileno(stdout)))) 
        fprintf(stdout,"\n\"%s\", line %d: %s in function %s()\n\n",file,line_num,err_list[list_num] . listp[num],((( *fn_name) & ~0x7f) == 0?fn_name : "???"));
      break; 
    }
/* ensure exit if fall through */
  if (!err_list[list_num] . warn) 
    exit(0);
  return 0;
}
/* float_error -- catches floating arithmetic signals */
#ifndef ANSI_C
#else

static void float_error(int num)
#endif
{
  signal(8,float_error);
/* fprintf(stderr,"SIGFPE: signal #%d\n",num); */
/* fprintf(stderr,"errno = %d\n",errno); */
  ev_err("???.c",16,0,"???",0);
}
/* catch_signal -- sets up float_error() to catch SIGFPE's */

void catch_FPE()
{
  signal(8,float_error);
}
