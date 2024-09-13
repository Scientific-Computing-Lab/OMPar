/*BHEADER**********************************************************************
 * Copyright (c) 2006   The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by the HYPRE team. UCRL-CODE-222953.
 * All rights reserved.
 *
 * This file is part of HYPRE (see http://www.llnl.gov/CASC/hypre/).
 * Please see the COPYRIGHT_and_LICENSE file for the copyright notice, 
 * disclaimer, contact information and the GNU Lesser General Public License.
 *
 * HYPRE is free software; you can redistribute it and/or modify it under the 
 * terms of the GNU General Public License (as published by the Free Software
 * Foundation) version 2.1 dated February 1999.
 *
 * HYPRE is distributed in the hope that it will be useful, but WITHOUT ANY 
 * WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS 
 * FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Revision: 2.8 $
 ***********************************************************************EHEADER*/
/******************************************************************************
 *
 * Member functions for hypre_Vector class.
 *
 *****************************************************************************/
#include "headers.h"
#include <assert.h>
/*--------------------------------------------------------------------------
 * hypre_SeqVectorCreate
 *--------------------------------------------------------------------------*/
#include <omp.h> 

extern "C" hypre_Vector *hypre_SeqVectorCreate(int size)
{
  hypre_Vector *vector;
  vector = ((hypre_Vector *)(hypre_CAlloc(((unsigned int )1),((unsigned int )(sizeof(hypre_Vector ))))));
  vector -> data = 0L;
  vector -> size = size;
  vector -> num_vectors = 1;
  vector -> multivec_storage_method = 0;
/* set defaults */
  vector -> owns_data = 1;
  return vector;
}
/*--------------------------------------------------------------------------
 * hypre_SeqMultiVectorCreate
 *--------------------------------------------------------------------------*/

extern "C" hypre_Vector *hypre_SeqMultiVectorCreate(int size,int num_vectors)
{
  hypre_Vector *vector = hypre_SeqVectorCreate(size);
  vector -> num_vectors = num_vectors;
  return vector;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorDestroy
 *--------------------------------------------------------------------------*/

extern "C" int hypre_SeqVectorDestroy(hypre_Vector *vector)
{
  int ierr = 0;
  if (vector) {
    if ((vector -> owns_data)) {
      (hypre_Free((char *)(vector -> data)) , vector -> data = 0L);
    }
    (hypre_Free((char *)vector) , vector = 0L);
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorInitialize
 *--------------------------------------------------------------------------*/

extern "C" int hypre_SeqVectorInitialize(hypre_Vector *vector)
{
  int size = vector -> size;
  int ierr = 0;
  int num_vectors = vector -> num_vectors;
  int multivec_storage_method = vector -> multivec_storage_method;
  if (!(vector -> data)) 
    vector -> data = ((double *)(hypre_CAlloc(((unsigned int )(num_vectors * size)),((unsigned int )(sizeof(double ))))));
  if (multivec_storage_method == 0) {
    vector -> vecstride = size;
    vector -> idxstride = 1;
  }
   else if (multivec_storage_method == 1) {
    vector -> vecstride = 1;
    vector -> idxstride = num_vectors;
  }
   else 
    ++ierr;
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetDataOwner
 *--------------------------------------------------------------------------*/

extern "C" int hypre_SeqVectorSetDataOwner(hypre_Vector *vector,int owns_data)
{
  int ierr = 0;
  vector -> owns_data = owns_data;
  return ierr;
}
/*--------------------------------------------------------------------------
 * ReadVector
 *--------------------------------------------------------------------------*/

extern "C" hypre_Vector *hypre_SeqVectorRead(char *file_name)
{
  hypre_Vector *vector;
  FILE *fp;
  double *data;
  int size;
  int j;
/*----------------------------------------------------------
    * Read in the data
    *----------------------------------------------------------*/
  fp = fopen(file_name,"r");
  fscanf(fp,"%d",&size);
  vector = hypre_SeqVectorCreate(size);
  hypre_SeqVectorInitialize(vector);
  data = vector -> data;
  for (j = 0; j <= size - 1; j += 1) {
    fscanf(fp,"%le",&data[j]);
  }
  fclose(fp);
/* multivector code not written yet >>> */
  if (!(vector -> num_vectors == 1)) {
    fprintf(stderr,"hypre_assert failed: %s\n","hypre_VectorNumVectors(vector) == 1");
    hypre_error_handler("vector.cpp",177,1);
  }
  ;
  return vector;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorPrint
 *--------------------------------------------------------------------------*/

extern "C" int hypre_SeqVectorPrint(hypre_Vector *vector,char *file_name)
{
  FILE *fp;
  double *data;
  int size;
  int num_vectors;
  int vecstride;
  int idxstride;
  int i;
  int j;
  int ierr = 0;
  num_vectors = vector -> num_vectors;
  vecstride = vector -> vecstride;
  idxstride = vector -> idxstride;
/*----------------------------------------------------------
    * Print in the data
    *----------------------------------------------------------*/
  data = vector -> data;
  size = vector -> size;
  fp = fopen(file_name,"w");
  if (vector -> num_vectors == 1) {
    fprintf(fp,"%d\n",size);
  }
   else {
    fprintf(fp,"%d vectors of size %d\n",num_vectors,size);
  }
  if (num_vectors > 1) {
    for (j = 0; j <= num_vectors - 1; j += 1) {
      fprintf(fp,"vector %d\n",j);
      for (i = 0; i <= size - 1; i += 1) {
        fprintf(fp,"%.14e\n",data[j * vecstride + i * idxstride]);
      }
    }
  }
   else {
    for (i = 0; i <= size - 1; i += 1) {
      fprintf(fp,"%.14e\n",data[i]);
    }
  }
  fclose(fp);
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorSetConstantValues
 *--------------------------------------------------------------------------*/

extern "C" int hypre_SeqVectorSetConstantValues(hypre_Vector *v,double value)
{
  double *vector_data = v -> data;
  int size = v -> size;
  int i;
  int ierr = 0;
  size *= v -> num_vectors;
  
#pragma omp parallel for private (i) firstprivate (value,size)
  for (i = 0; i <= size - 1; i += 1) {
    vector_data[i] = value;
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorCopy
 * copies data from x to y
 * y should have already been initialized at the same size as x
 *--------------------------------------------------------------------------*/

extern "C" int hypre_SeqVectorCopy(hypre_Vector *x,hypre_Vector *y)
{
  double *x_data = x -> data;
  double *y_data = y -> data;
  int size = x -> size;
  int i;
  int ierr = 0;
  size *= x -> num_vectors;
  
#pragma omp parallel for private (i) firstprivate (size)
  for (i = 0; i <= size - 1; i += 1) {
    y_data[i] = x_data[i];
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneDeep
 * Returns a complete copy of x - a deep copy, with its own copy of the data.
 *--------------------------------------------------------------------------*/

extern "C" hypre_Vector *hypre_SeqVectorCloneDeep(hypre_Vector *x)
{
  int size = x -> size;
  int num_vectors = x -> num_vectors;
  hypre_Vector *y = hypre_SeqMultiVectorCreate(size,num_vectors);
  y -> multivec_storage_method = x -> multivec_storage_method;
  y -> vecstride = x -> vecstride;
  y -> idxstride = x -> idxstride;
  hypre_SeqVectorInitialize(y);
  hypre_SeqVectorCopy(x,y);
  return y;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorCloneShallow
 * Returns a complete copy of x - a shallow copy, pointing the data of x
 *--------------------------------------------------------------------------*/

extern "C" hypre_Vector *hypre_SeqVectorCloneShallow(hypre_Vector *x)
{
  int size = x -> size;
  int num_vectors = x -> num_vectors;
  hypre_Vector *y = hypre_SeqMultiVectorCreate(size,num_vectors);
  y -> multivec_storage_method = x -> multivec_storage_method;
  y -> vecstride = x -> vecstride;
  y -> idxstride = x -> idxstride;
  y -> data = x -> data;
  hypre_SeqVectorSetDataOwner(y,0);
  hypre_SeqVectorInitialize(y);
  return y;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorScale
 *--------------------------------------------------------------------------*/

extern "C" int hypre_SeqVectorScale(double alpha,hypre_Vector *y)
{
  double *y_data = y -> data;
  int size = y -> size;
  int i;
  int ierr = 0;
  size *= y -> num_vectors;
  
#pragma omp parallel for private (i) firstprivate (alpha,size)
  for (i = 0; i <= size - 1; i += 1) {
    y_data[i] *= alpha;
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorAxpy
 *--------------------------------------------------------------------------*/

extern "C" int hypre_SeqVectorAxpy(double alpha,hypre_Vector *x,hypre_Vector *y)
{
  double *x_data = x -> data;
  double *y_data = y -> data;
  int size = x -> size;
  int i;
  int ierr = 0;
  size *= x -> num_vectors;
  
#pragma omp parallel for private (i) firstprivate (alpha,size)
  for (i = 0; i <= size - 1; i += 1) {
    y_data[i] += alpha * x_data[i];
  }
  return ierr;
}
/*--------------------------------------------------------------------------
 * hypre_SeqVectorInnerProd
 *--------------------------------------------------------------------------*/

extern "C" double hypre_SeqVectorInnerProd(hypre_Vector *x,hypre_Vector *y)
{
  double *x_data = x -> data;
  double *y_data = y -> data;
  int size = x -> size;
  int i;
  double result = 0.0;
  size *= x -> num_vectors;
  
#pragma omp parallel for private (i) reduction (+:result) firstprivate (size)
  for (i = 0; i <= size - 1; i += 1) {
    result += y_data[i] * x_data[i];
  }
  return result;
}
/*--------------------------------------------------------------------------
 * hypre_VectorSumElts:
 * Returns the sum of all vector elements.
 *--------------------------------------------------------------------------*/

extern "C" double hypre_VectorSumElts(hypre_Vector *vector)
{
  double sum = 0;
  double *data = vector -> data;
  int size = vector -> size;
  int i;
  
#pragma omp parallel for private (i) reduction (+:sum) firstprivate (size)
  for (i = 0; i <= size - 1; i += 1) {
    sum += data[i];
  }
  return sum;
}
