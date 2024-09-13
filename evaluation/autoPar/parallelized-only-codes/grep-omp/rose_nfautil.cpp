#include "nfautil.h"
/*
 * Visualize the NFA in stdout
 */
#include <omp.h> 
int visited[5000];
int count[5000];
int visited_index = 0;
int nstate;
::State matchstate = {(Match)};
/* matching state */
::List l1;
::List l2;
static int listid;
void addstate(::List *,::State *);
void step(::List *,int ,::List *);
/* Compute initial state list */

::List *startlist(::State *start,::List *l)
{
  l -> n = 0;
  listid++;
  addstate(l,start);
  return l;
}
/* Check whether state list contains a match. */

int ismatch(::List *l)
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

void addstate(::List *l,::State *s)
{
// lastlist check is present to ensure that if
// multiple states point to this state, then only
// one instance of the state is added to the list
  if (s == 0L || s -> lastlist == listid) 
    return ;
  s -> lastlist = listid;
  if (s -> c == Split) {
/* follow unlabeled arrows */
    addstate(l,s -> out);
    addstate(l,s -> out1);
    return ;
  }
  l -> s[l -> n++] = s;
}
/*
 * Step the NFA from the states in clist
 * past the character c,
 * to create next NFA state set nlist.
 */

void step(::List *clist,int c,::List *nlist)
{
  int i;
  ::State *s;
  listid++;
  nlist -> n = 0;
  for (i = 0; i <= clist -> n - 1; i += 1) {
    s = clist -> s[i];
    if (s -> c == c || s -> c == Any) 
      addstate(nlist,s -> out);
  }
}
/* Run NFA to determine whether it matches s. */

int match(::State *start,char *s)
{
  int c;
  ::List *clist;
  ::List *nlist;
  ::List *t;
  clist = startlist(start,&l1);
  nlist = &l2;
  for (; ( *s); s++) {
    c = ( *s) & 0xFF;
    step(clist,c,nlist);
    t = clist;
    clist = nlist;
    nlist = t;
// swap clist, nlist 
// check for a match in the middle of the string
    if ((ismatch(clist))) 
      return 1;
  }
  return ismatch(clist);
}
/* Check for a string match at all possible start positions */

int anyMatch(::State *start,char *s)
{
  int isMatch = match(start,s);
  int index = 0;
  int len = (strlen(s));
  while(!isMatch && index <= len){
    isMatch = match(start,s + index);
    index++;
  }
  return isMatch;
}
/* Allocate and initialize State */

::State *state(int c,::State *out,::State *out1)
{
  ::State *s;
  s = ((::State *)(malloc(sizeof(( *s)))));
  s -> id = ++nstate;
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
/* Initialize Frag struct. */

::Frag frag(::State *start,::Ptrlist *out)
{
  ::Frag n = {start, out};
  return n;
}
/* Create singleton list containing just outp. */

::Ptrlist *list1(::State **outp)
{
  ::Ptrlist *l;
  l = ((::Ptrlist *)outp);
  l -> next = 0L;
  return l;
}
/* Patch the list of states at out to point to start. */

void patch(::Ptrlist *l,::State *s)
{
  ::Ptrlist *next;
  for (; l; l = next) {
    next = l -> next;
    l -> s = s;
  }
}
/* Join the two lists l1 and l2, returning the combination. */

::Ptrlist *append(::Ptrlist *l1,::Ptrlist *l2)
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

::State *post2nfa(char *postfix)
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
      s = state(Any,0L,0L);
       *(stackp++) = frag(s,(list1(&s -> out)));
      break; 
      default:
      s = state(( *p),0L,0L);
       *(stackp++) = frag(s,(list1(&s -> out)));
      break; 
      case 0x1b:
/* catenate */
      e2 =  *(--stackp);
      e1 =  *(--stackp);
      patch(e1 . out,e2 . start);
       *(stackp++) = frag(e1 . start,e2 . out);
      break; 
      case 0x04:
/* alternate (|)*/
      e2 =  *(--stackp);
      e1 =  *(--stackp);
      s = state(Split,e1 . start,e2 . start);
       *(stackp++) = frag(s,(append(e1 . out,e2 . out)));
      break; 
      case 0x02:
/* zero or one (?)*/
      e =  *(--stackp);
      s = state(Split,e . start,0L);
       *(stackp++) = frag(s,(append(e . out,(list1(&s -> out1)))));
      break; 
      case 0x03:
/* zero or more (*)*/
      e =  *(--stackp);
      s = state(Split,e . start,0L);
      patch(e . out,s);
       *(stackp++) = frag(s,(list1(&s -> out1)));
      break; 
      case 0x01:
/* one or more (+)*/
      e =  *(--stackp);
      s = state(Split,e . start,0L);
      patch(e . out,s);
       *(stackp++) = frag(e . start,(list1(&s -> out1)));
      break; 
    }
  }
  e =  *(--stackp);
  if (stackp != stack) 
    return 0L;
  patch(e . out,&matchstate);
  return e . start;
#undef pop
#undef push
}
/*
 * Convert infix regexp re to postfix notation.
 * Insert ESC (or 0x1b) as explicit concatenation operator.
 * Cheesy parser, return static buffer.
 */

char *re2post(char *re)
{
  int nalt;
  int natom;
  static char buf[8000];
  char *dst;
  struct __anonymous_0x5616db57c530 {
  int nalt;
  int natom;}paren[100];
  struct __anonymous_0x5616db57c530 *p;
  p = paren;
  dst = buf;
  nalt = 0;
  natom = 0;
  if (strlen(re) >= sizeof(buf) / 2) 
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
  return buf;
}

void readFile(char *fileName,char ***lines,int *lineIndex)
{
  FILE *fp = fopen(fileName,"r");
  char *source = 0L;
  if (fp != 0L) {
/* Go to the end of the file. */
    if (fseek(fp,0L,2) == 0) {
/* Get the size of the file. */
      long bufsize = ftell(fp);
/* Error */
      if (bufsize == (- 1)) {
      }
/* Allocate our buffer to that size. */
      source = ((char *)(malloc(sizeof(char ) * (bufsize + 1))));
/* Go back to the start of the file. */
/* Error */
      if (fseek(fp,0L,0) == 0) {
      }
/* Read the entire file into memory. */
      size_t newLen = fread(source,sizeof(char ),bufsize,fp);
      if (newLen == 0) {
        fputs("Error reading file",stderr);
      }
       else {
        source[newLen] = '\0';
/* Just to be safe. */
      }
    }
    fclose(fp);
  }
   *lines = ((char **)(malloc(sizeof(char *) * 1)));
   *( *lines) = source;
   *lineIndex = 1;
}

void usage(const char *progname)
{
  printf("Usage: %s [options] [pattern] \n",progname);
  printf("Program Options:\n");
  printf("  -v  Visualize the NFA then exit\n");
  printf("  -p  View postfix expression then exit\n");
  printf("  -s  View simplified expression then exit\n");
  printf("  -t  Print timing data\n");
  printf("  -f <FILE> --file Input file to be matched\n");
  printf("  -r <FILE> --regex Input file with regexs\n");
  printf("  -? This message\n");
  printf("[pattern] required only if -r or --regex is not used\n");
}

void parseCmdLine(int argc,char **argv,int *visualize,int *postfix,int *time,int *simplified,char **fileName,char **regexFile)
{
  if (argc < 3) {
    usage(argv[0]);
    exit(0);
  }
  int opt;
  static struct option long_options[] = {/* Need explicit braces: is this where we insert the class name? */ {("help"), (0), (0), ('?')}, /* Need explicit braces: is this where we insert the class name? */ {("postfix"), (0), (0), ('p')}, /* Need explicit braces: is this where we insert the class name? */ {("simplified"), (0), (0), ('s')}, /* Need explicit braces: is this where we insert the class name? */ {("visualize"), (0), (0), ('v')}, /* Need explicit braces: is this where we insert the class name? */ {("file"), (1), (0), ('f')}, /* Need explicit braces: is this where we insert the class name? */ {("regex"), (1), (0), ('r')}, /* Need explicit braces: is this where we insert the class name? */ {("time"), (0), (0), ('t')}, /* Need explicit braces: is this where we insert the class name? */ {(0), (0), (0), (0)}};
   *visualize = 0;
   *postfix = 0;
   *time = 0;
   *simplified = 0;
  while((opt = getopt_long_only(argc,argv,"tvpsf:r:?",long_options,0L)) != - 1){
    switch(opt){
      case 'v':
       *visualize = 1;
      break; 
      case 'p':
       *postfix = 1;
      break; 
      case 'f':
       *fileName = optarg;
      break; 
      case 'r':
       *regexFile = optarg;
      break; 
      case 't':
       *time = 1;
      break; 
      case 's':
       *simplified = 1;
      break; 
      default:
      usage(argv[0]);
      exit(0);
    }
  }
}

int hasSeen(::State *start,int *index)
{
  int i;
  for (i = 0; i <= 4999; i += 1) {
    if (visited[i] == start -> id) {
       *index = i;
      return 0;
    }
  }
  return 1;
}

void visualize_nfa_help(::State *start)
{
  int index;
  if (start == 0L) {
    return ;
  }
  if (hasSeen(start,&index) == 0) {
    if (count[index] > 0) {
      return ;
    }
  }
  count[start -> id]++;
  visited[start -> id] = start -> id;
  char data[10];
  if (start -> c == Match) {
    strcpy(data,"Match");
  }
   else if (start -> c == Split) {
    strcpy(data,"Split");
  }
   else if (start -> c == Any) {
    strcpy(data,"Any");
  }
   else {
    sprintf(data,"Char %c",start -> c);
  }
  int outId;
  int outId1;
  outId = (start -> out == 0L?- 1 : start -> out -> id);
  outId1 = (start -> out1 == 0L?- 1 : start -> out1 -> id);
  printf("{ \"id\": \"%d\", \"data\":\"%s\", \"out\":\"%d\", \"out1\":\"%d\" \n},",start -> id,data,outId,outId1);
  visualize_nfa_help(start -> out);
  visualize_nfa_help(start -> out1);
}

void visualize_nfa(::State *start)
{
  memset(visited,0,5000 * sizeof(int ));
  memset(count,0,5000 * sizeof(int ));
  printf("[");
  visualize_nfa_help(start);
  printf("]\n");
}

double gettime()
{
  struct timeval tv;
  gettimeofday(&tv,0L);
  return tv . tv_sec + tv . tv_usec / 1000000.0;
}
