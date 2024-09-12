/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team, UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer and the GNU Lesser General Public License.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.4 $
 ***********************************************************************EHEADER*/
/******************************************************************************
 *
 * Memory management utilities
 *
 *****************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include "utilities.h"
#ifdef HYPRE_USE_PTHREADS
#include "threading.h"
#ifdef HYPRE_USE_UMALLOC
#include "umalloc_local.h"
#define _umalloc_(size) (threadid == hypre_NumThreads) ? \
                        (char *) malloc(size) : \
                        (char *) _umalloc(_uparam[threadid].myheap, size)
#define _ucalloc_(count, size) (threadid == hypre_NumThreads) ? \
                               (char *) calloc(count, size) : \
                               (char *) _ucalloc(_uparam[threadid].myheap,\
                                                 count, size)
#define _urealloc_(ptr, size) (threadid == hypre_NumThreads) ? \
                              (char *) realloc(ptr, size) : \
                              (char *) _urealloc(ptr, size)
#define _ufree_(ptr)          (threadid == hypre_NumThreads) ? \
                              free(ptr) : _ufree(ptr)
#endif
#else
#ifdef HYPRE_USE_UMALLOC
#undef HYPRE_USE_UMALLOC
#endif
#endif
/******************************************************************************
 *
 * Standard routines
 *
 *****************************************************************************/
/*--------------------------------------------------------------------------
 * hypre_OutOfMemory
 *--------------------------------------------------------------------------*/

int hypre_OutOfMemory(int size)
{
  printf("Out of memory trying to allocate %d bytes\n",size);
  fflush(stdout);
  hypre_error_handler("hypre_memory.cpp",78,2);
  return 0;
}
/*--------------------------------------------------------------------------
 * hypre_MAlloc
 *--------------------------------------------------------------------------*/

char *hypre_MAlloc(int size)
{
  char *ptr;
  if (size > 0) {
#ifdef HYPRE_USE_UMALLOC
#else
    ptr = ((char *)(malloc(size)));
#endif
#if 1
    if (ptr == 0L) {
      hypre_OutOfMemory(size);
    }
#endif
  }
   else {
    ptr = 0L;
  }
  return ptr;
}
/*--------------------------------------------------------------------------
 * hypre_CAlloc
 *--------------------------------------------------------------------------*/

char *hypre_CAlloc(int count,int elt_size)
{
  char *ptr;
  int size = count * elt_size;
  if (size > 0) {
#ifdef HYPRE_USE_UMALLOC
#else
    ptr = ((char *)(calloc(count,elt_size)));
#endif
#if 1
    if (ptr == 0L) {
      hypre_OutOfMemory(size);
    }
#endif
  }
   else {
    ptr = 0L;
  }
  return ptr;
}
/*--------------------------------------------------------------------------
 * hypre_ReAlloc
 *--------------------------------------------------------------------------*/

char *hypre_ReAlloc(char *ptr,int size)
{
#ifdef HYPRE_USE_UMALLOC
#else
  if (ptr == 0L) {
    ptr = ((char *)(malloc(size)));
  }
   else {
    ptr = ((char *)(realloc(ptr,size)));
  }
#endif
#if 1
  if (ptr == 0L && size > 0) {
    hypre_OutOfMemory(size);
  }
#endif
  return ptr;
}
/*--------------------------------------------------------------------------
 * hypre_Free
 *--------------------------------------------------------------------------*/

void hypre_Free(char *ptr)
{
  if (ptr) {
#ifdef HYPRE_USE_UMALLOC
#else
    free(ptr);
#endif
  }
}
/*--------------------------------------------------------------------------
 * These Shared routines are for one thread to allocate memory for data
 * will be visible to all threads.  The file-scope pointer
 * global_alloc_ptr is used in these routines.
 *--------------------------------------------------------------------------*/
#ifdef HYPRE_USE_PTHREADS
/*--------------------------------------------------------------------------
 * hypre_SharedMAlloc
 *--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------
 * hypre_SharedCAlloc
 *--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------
 * hypre_SharedReAlloc
 *--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------
 * hypre_SharedFree
 *--------------------------------------------------------------------------*/
/*--------------------------------------------------------------------------
 * hypre_IncrementSharedDataPtr
 *--------------------------------------------------------------------------*/
#endif
