/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include <cstdlib>
#include <cstdio>
///////////////////////////////////////////////////////////////////////////////
// Using Viterbi algorithm to search for a Hidden Markov Model for the most
// probable state path given the observation sequence.
///////////////////////////////////////////////////////////////////////////////
#include <omp.h> 

int ViterbiCPU(float &viterbiProb,int *viterbiPath,int *obs,const int &nObs,float *initProb,float *mtState,const int &nState,float *mtEmit)
{
  float *maxProbNew = (float *)(malloc(sizeof(float ) * nState));
  float *maxProbOld = (float *)(malloc(sizeof(float ) * nState));
  int **path = (int **)(malloc(sizeof(int *) * (nObs - 1)));
  for (int i = 0; i <= nObs - 1 - 1; i += 1) {
    path[i] = ((int *)(malloc(sizeof(int ) * nState)));
  }
// initial probability
  
#pragma omp parallel for
  for (int i = 0; i <= nState - 1; i += 1) {
    maxProbOld[i] = initProb[i];
  }
// main iteration of Viterbi algorithm
  for (int t = 1; t <= nObs - 1; t += 1) 
// for every input observation
{
    
#pragma omp parallel for private (preState)
    for (int iState = 0; iState <= nState - 1; iState += 1) {
// find the most probable previous state leading to iState
      float maxProb = 0.0;
      int maxState = - 1;
      for (int preState = 0; preState <= nState - 1; preState += 1) {
        float p = maxProbOld[preState] + mtState[iState * nState + preState];
        if (p > maxProb) {
          maxProb = p;
          maxState = preState;
        }
      }
      maxProbNew[iState] = maxProb + mtEmit[obs[t] * nState + iState];
      path[t - 1][iState] = maxState;
    }
    
#pragma omp parallel for
    for (int iState = 0; iState <= nState - 1; iState += 1) {
      maxProbOld[iState] = maxProbNew[iState];
    }
  }
// find the final most probable state
  float maxProb = 0.0;
  int maxState = - 1;
  for (int i = 0; i <= nState - 1; i += 1) {
    if (maxProbNew[i] > maxProb) {
      maxProb = maxProbNew[i];
      maxState = i;
    }
  }
  viterbiProb = maxProb;
// backtrace to find the Viterbi path
  viterbiPath[nObs - 1] = maxState;
  for (int t = nObs - 2; t >= 0; t += -1) {
    viterbiPath[t] = path[t][viterbiPath[t + 1]];
  }
  free(maxProbNew);
  free(maxProbOld);
  for (int i = 0; i <= nObs - 1 - 1; i += 1) {
    free(path[i]);
  }
  free(path);
  return 1;
}
