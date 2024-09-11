#include "nfautil.h"
#include "regex.h"
#define DEREF(arr,i) ((*(arr))[(i)])
/* constructor for SimpleReBuilder */
#include <omp.h> 

void simpleReBuilder(SimpleReBuilder **builder,int len)
{
  ( *builder) -> re = ((char *)(malloc((len + 3))));
  ( *builder) -> size = len + 3;
}

void _simpleReBuilder(SimpleReBuilder *builder)
{
  free((builder -> re));
/* the rest is on the stack */
}

void regex_error(int i)
{
  fprintf(stderr,"Improper regex at character %d",i);
  exit(1);
}

char *stringify(char *nonull,int j)
{
  char *proper = (char *)(malloc((j + 2)));
  memcpy(proper,nonull,j);
  proper[j + 1] = '\0';
  return proper;
}

void insertIntoComplexRe(char **complexRe,int where,int *len,const char *toInsert)
{
  char *buf;
  int insertLen = (strlen(toInsert));
  int i = where;
/* enough space for complexRe+the new range */
   *len =  *len + (insertLen + 1);
   *complexRe = ((char *)(realloc(( *complexRe),( *len))));
/* buffer the rest */
  buf = ((char *)(malloc(( *len))));
  int k;
  for (k = i + 2; k <=  *len - (insertLen + 1) - 1; k += 1) {
    buf[k - (i + 2)] = ( *complexRe)[k];
  }
/* insert the string */
  for (k = 0; k <= insertLen - 1; k += 1) {
    ( *complexRe)[i++] = toInsert[k];
  }
/* put the buffer back in */
  for (k = i; k <=  *len - 1; k += 1) {
    ( *complexRe)[k] = buf[k - i];
  }
  free(buf);
}

void handle_escape(SimpleReBuilder *builder,char **complexRe,int *len,int *bi,int *ci)
{
  int i =  *ci;
  int j =  *bi;
  if (i + 1 >  *len) 
    regex_error(i);
  i++;
//go to escaped character and ignore '/'
  switch(( *complexRe)[i]){
    case 't':
    insertIntoComplexRe(complexRe,--i,len,"\t");
    break; 
    case 'n':
    insertIntoComplexRe(complexRe,--i,len,"\n");
    break; 
    case 'd':
    insertIntoComplexRe(complexRe,--i,len,"[0-9]");
    break; 
    case 'w':
    insertIntoComplexRe(complexRe,--i,len,"([a-z]|[A-Z]|_)");
    break; 
    case 's':
    insertIntoComplexRe(complexRe,--i,len,"( |\t|\n)");
    break; 
/* ... see www.cs.tut.fi/~jkorpela/perl/regexp.html */
// default is just ignoring the backslash and taking the 
// LITERAL character after no matter what
    default:
    builder -> re[j++] = ( *complexRe)[i++];
    break; 
  }
   *ci = i - 1;
//-1 because we incremented at the end
   *bi = j - 1;
//-1 because we incremented at the end
}

void putRange(SimpleReBuilder *builder,char start,char end,int *bi)
{
  int i =  *bi;
  int amount = (end - start + 1) * 2 + 1;
  builder -> size += amount;
  builder -> re = ((char *)(realloc((builder -> re),(builder -> size))));
  builder -> re[i++] = 0x05;
  builder -> re[i++] = start;
  char k;
  for (k = (start + 1); k <= end; k += 1) {
    builder -> re[i++] = 0x04;
    builder -> re[i++] = k;
  }
  builder -> re[i] = 0x06;
   *bi = i;
}

void handle_range(SimpleReBuilder *builder,char *complexRe,int len,int *bi,int *ci)
{
  int i =  *ci;
  if (complexRe[i + 4] != ']' || complexRe[i + 2] != '-' || complexRe[i + 1] > complexRe[i + 3] || complexRe[i + 1] <= 0x20) {
    fprintf(stderr,"Invalid range at character %d\n",i);
    exit(1);
  }
  putRange(builder,complexRe[i + 1],complexRe[i + 3],bi);
   *ci = i + 4;
}

SimpleReBuilder *simplifyRe(char **complexRe,SimpleReBuilder *builder)
{
  int len = (strlen(( *complexRe)));
  simpleReBuilder(&builder,len);
  int i;
  int j;
  for ((i = 0 , j = 0); i <= len - 1; (i++ , j++)) {
    switch(( *complexRe)[i]){
      case '\\':
      handle_escape(builder,complexRe,&len,&j,&i);
      break; 
      case '.':
      builder -> re[j] = 0x15;
//nak is ANY
      break; 
      case '+':
      builder -> re[j] = 0x01;
//0x01 is +
      break; 
      case '?':
      builder -> re[j] = 0x02;
//0x02 is ?
      break; 
      case '*':
      builder -> re[j] = 0x03;
//0x03 is *
      break; 
      case '|':
      builder -> re[j] = 0x04;
//0x04 is |
      break; 
      case '(':
      builder -> re[j] = 0x05;
//0x05 is (
      break; 
      case ')':
      builder -> re[j] = 0x06;
//0x06 is )
      break; 
      case '[':
      handle_range(builder, *complexRe,len,&j,&i);
      break; 
      default:
      builder -> re[j] = ( *complexRe)[i];
      break; 
    }
  }
  builder -> re[j] = '\0';
  return builder;
}

char *stringifyRegex(const char *oldRegex)
{
  int len = (strlen(oldRegex));
  char *newRegex = (char *)(malloc((len + 1)));
  int i;
  
#pragma omp parallel for private (i) firstprivate (len)
  for (i = 0; i <= len - 1; i += 1) {
    switch(oldRegex[i]){
      case 0x15:
      newRegex[i] = '.';
      break; 
      case 0x1b:
      newRegex[i] = '`';
      break; 
      case 0x04:
      newRegex[i] = '|';
      break; 
      case 0x02:
      newRegex[i] = '?';
      break; 
      case 0x03:
      newRegex[i] = '*';
      break; 
      case 0x01:
      newRegex[i] = '+';
      break; 
      case 0x05:
      newRegex[i] = '(';
      break; 
      case 0x06:
      newRegex[i] = ')';
      break; 
      default:
      newRegex[i] = oldRegex[i];
      break; 
    }
  }
  newRegex[i] = '\0';
  return newRegex;
}
