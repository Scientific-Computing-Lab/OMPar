#include "pnfa.h"
inline ::State *pstate(int ,::State *,::State *,::State *,int *);
inline ::Frag pfrag(::State *,::Ptrlist *);
inline ::Ptrlist *plist1(::State **);
inline void ppatch(::Ptrlist *,::State *);
inline ::Ptrlist *pappend(::Ptrlist *,::Ptrlist *);
inline ::State *ppost2nfa(char *,::State *,int *,::State *);
/* Allocate and initialize State */

inline ::State *pstate(int c,::State *out,::State *out1,::State *lstate,int *pnstate)
{
  ::State *s = lstate +  *pnstate;
// assign a state
  s -> id =  *pnstate;
  ( *pnstate)++;
  s -> lastlist = 0;
  s -> c = c;
  s -> out = out;
  s -> out1 = out1;
// device pointer of itself
// serves no real purpose other than to help transfer the NFA over
  s -> dev = 0L;
  s -> free = 0;
  return s;
}
/* Initialize frag struct. */

inline ::Frag pfrag(::State *start,::Ptrlist *out)
{
  ::Frag n = {start, out};
  return n;
}
/* Create singleton list containing just outp. */

inline ::Ptrlist *plist1(::State **outp)
{
  ::Ptrlist *l;
  l = ((::Ptrlist *)outp);
  l -> next = 0L;
  return l;
}
/* Patch the list of states at out to point to start. */

inline void ppatch(::Ptrlist *l,::State *s)
{
  ::Ptrlist *next;
  for (; l; l = next) {
    next = l -> next;
    l -> s = s;
  }
}
/* Join the two lists l1 and l2, returning the combination. */

inline ::Ptrlist *pappend(::Ptrlist *l1,::Ptrlist *l2)
{
  ::Ptrlist *oldl1;
  oldl1 = l1;
  while((l1 -> next))
    l1 = l1 -> next;
  l1 -> next = l2;
  return oldl1;
}
/*
 * Convert postfix regular expression to NFA.
 * Return start state.
 */

inline ::State *ppost2nfa(char *postfix,::State *lstate,int *pnstate,::State *pmatchstate)
{
  char *p;
  ::Frag stack[1000];
  ::Frag *stackp;
  ::Frag e1;
  ::Frag e2;
  ::Frag e;
  ::State *s;
// fprintf(stderr, "postfix: %s\n", postfix);
  if (postfix == 0L) 
    return 0L;
#define push(s) *stackp++ = s
#define pop() *--stackp
  stackp = stack;
  for (p = postfix; ( *p); p++) {
    switch(( *p)){
      case 0x15:
/* any (.) */
      s = pstate(Any,0L,0L,lstate,pnstate);
       *(stackp++) = pfrag(s,(plist1(&s -> out)));
      break; 
      default:
      s = pstate(( *p),0L,0L,lstate,pnstate);
       *(stackp++) = pfrag(s,(plist1(&s -> out)));
      break; 
      case 0x1b:
/* catenate */
      e2 =  *(--stackp);
      e1 =  *(--stackp);
      ppatch(e1 . out,e2 . start);
       *(stackp++) = pfrag(e1 . start,e2 . out);
      break; 
      case 0x04:
/* alternate (|)*/
      e2 =  *(--stackp);
      e1 =  *(--stackp);
      s = pstate(Split,e1 . start,e2 . start,lstate,pnstate);
       *(stackp++) = pfrag(s,(pappend(e1 . out,e2 . out)));
      break; 
      case 0x02:
/* zero or one (?)*/
      e =  *(--stackp);
      s = pstate(Split,e . start,0L,lstate,pnstate);
       *(stackp++) = pfrag(s,(pappend(e . out,(plist1(&s -> out1)))));
      break; 
      case 0x03:
/* zero or more (*)*/
      e =  *(--stackp);
      s = pstate(Split,e . start,0L,lstate,pnstate);
      ppatch(e . out,s);
       *(stackp++) = pfrag(s,(plist1(&s -> out1)));
      break; 
      case 0x01:
/* one or more (+)*/
      e =  *(--stackp);
      s = pstate(Split,e . start,0L,lstate,pnstate);
      ppatch(e . out,s);
       *(stackp++) = pfrag(e . start,(plist1(&s -> out1)));
      break; 
    }
  }
  e =  *(--stackp);
  if (stackp != stack) 
    return 0L;
//ppatch(e.out, &pmatchstate);
  ppatch(e . out,pmatchstate);
  return e . start;
#undef pop
#undef push
}
