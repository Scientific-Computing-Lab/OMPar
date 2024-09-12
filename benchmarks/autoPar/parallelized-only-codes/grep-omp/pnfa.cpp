#include "putil.cpp" 
#include <omp.h> 
inline void paddstate(::List *,::State *,::List *);
inline void pstep(::List *,int ,::List *);

inline int pstrlen(char *str)
{
  int len = 0;
  while(( *str) != 0){
    len++;
    str += 1;
  }
  return len;
}
/*
 * Convert infix regexp re to postfix notation.
 * Insert ESC (or 0x1b) as explicit concatenation operator.
 * Cheesy parser, return static buffer.
 */

inline char *pre2post(char *re,char *dst)
{
  int nalt;
  int natom;
  struct __anonymous_0x55d26f59a5a0 {
  int nalt;
  int natom;}paren[100];
  struct __anonymous_0x55d26f59a5a0 *p;
  p = paren;
  nalt = 0;
  natom = 0;
  int len = pstrlen(re);
  if (len >= 8000 / 2) 
    return 0L;
  for (; ( *re); re++) {
    switch(( *re)){
      case 0x05:
// (
      if (natom > 1) {
        --natom;
         *(dst++) = 0x1b;
      }
      if (p >= paren + 100) 
        return 0L;
      p -> nalt = nalt;
      p -> natom = natom;
      p++;
      nalt = 0;
      natom = 0;
      break; 
      case 0x04:
// |
      if (natom == 0) 
        return 0L;
      while(--natom > 0)
         *(dst++) = 0x1b;
      nalt++;
      break; 
      case 0x06:
// )
      if (p == paren) 
        return 0L;
      if (natom == 0) 
        return 0L;
      while(--natom > 0)
         *(dst++) = 0x1b;
      for (; nalt >= 1; nalt += -1) {
         *(dst++) = 0x04;
      }
      --p;
      nalt = p -> nalt;
      natom = p -> natom;
      natom++;
      break; 
      case 0x03:;
// *
      case 0x01:;
// +
      case 0x02:
// ?
      if (natom == 0) 
        return 0L;
       *(dst++) =  *re;
      break; 
      default:
      if (natom > 1) {
        --natom;
         *(dst++) = 0x1b;
      }
       *(dst++) =  *re;
      natom++;
      break; 
    }
  }
  if (p != paren) 
    return 0L;
  while(--natom > 0)
     *(dst++) = 0x1b;
  for (; nalt >= 1; nalt += -1) {
     *(dst++) = 0x04;
  }
   *dst = 0;
  return dst;
}
/* Compute initial state list */

inline ::List *pstartlist(::State *start,::List *l)
{
  l -> n = 0;
  ::List addStartState;
  paddstate(l,start,&addStartState);
  return l;
}
/* Check whether state list contains a match. */

inline int ispmatch(::List *l)
{
  int i;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= l -> n - 1; i += 1) {
    if (l -> s[i] -> c == Match) 
      return 1;
  }
  return 0;
}
/* Add s to l, following unlabeled arrows. */

inline void paddstate(::List *l,::State *s,::List *addStateList)
{
  addStateList -> n = 0;
  addStateList -> s[addStateList -> n++] = s;
/* follow unlabeled arrows */
  while(!(addStateList -> n == 0)){
    s = addStateList -> s[--addStateList -> n];
    ;
// lastlist check is present to ensure that if
// multiple states point to this state, then only
//one instance of the state is added to the list
    if (s == 0L) 
      ;
     else if (s -> c == Split) {
      addStateList -> s[addStateList -> n++] = s -> out;
      addStateList -> s[addStateList -> n++] = s -> out1;
    }
     else {
      l -> s[l -> n++] = s;
    }
  }
}
/*
 * pstep the NFA from the states in clist
 * past the character c,
 * to create next NFA state set nlist.
 */

inline void pstep(::List *clist,int c,::List *nlist)
{
  int i;
  ::State *s;
  nlist -> n = 0;
  for (i = 0; i <= clist -> n - 1; i += 1) {
    s = clist -> s[i];
    if (s -> c == c || s -> c == Any) {
      ::List addStartState;
      paddstate(nlist,s -> out,&addStartState);
    }
  }
}
/* Run NFA to determine whether it matches s. */

inline int pmatch(::State *start,char *s,::List *dl1,::List *dl2)
{
  int c;
  ::List *clist;
  ::List *nlist;
  ::List *t;
  clist = pstartlist(start,dl1);
  nlist = dl2;
  for (; ( *s); s++) {
    c = ( *s) & 0xFF;
    pstep(clist,c,nlist);
    t = clist;
    clist = nlist;
    nlist = t;
// swap clist, nlist 
  }
  return ispmatch(clist);
}
/* Check for a string match at all possible start positions */

inline int panypmatch(::State *start,char *s,::List *dl1,::List *dl2)
{
  int c;
  ::List *clist;
  ::List *nlist;
  ::List *t;
  clist = pstartlist(start,dl1);
  nlist = dl2;
  for (; ( *s); s++) {
    c = ( *s) & 0xFF;
    pstep(clist,c,nlist);
    t = clist;
    clist = nlist;
    nlist = t;
// swap clist, nlist 
  }
  return ispmatch(clist);
}
