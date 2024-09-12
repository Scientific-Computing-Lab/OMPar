/*
   Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
   */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#define MAXDISTANCE    (200)
/**
 * Returns the lesser of the two unsigned integers a and b
 */
#include <omp.h> 

unsigned int minimum(unsigned int a,unsigned int b)
{
  return b < a?b : a;
}
/**
 * Reference CPU implementation of FloydWarshall PathFinding
 * for performance comparison
 * @param pathDistanceMatrix Distance between nodes of a graph
 * @param intermediate node between two nodes of a graph
 * @param number of nodes in the graph
 */

void floydWarshallCPUReference(unsigned int *pathDistanceMatrix,unsigned int *pathMatrix,unsigned int numNodes)
{
  unsigned int distanceYtoX;
  unsigned int distanceYtoK;
  unsigned int distanceKtoX;
  unsigned int indirectDistance;
/*
   * pathDistanceMatrix is the adjacency matrix(square) with
   * the dimension equal to the number of nodes in the graph.
   */
  unsigned int width = numNodes;
  unsigned int yXwidth;
/*
   * for each intermediate node k in the graph find the shortest distance between
   * the nodes i and j and update as
   *
   * ShortestPath(i,j,k) = min(ShortestPath(i,j,k-1), ShortestPath(i,k,k-1) + ShortestPath(k,j,k-1))
   */
  for (unsigned int k = 0; k <= numNodes - 1; k += 1) {
    for (unsigned int y = 0; y <= numNodes - 1; y += 1) {
      yXwidth = y * numNodes;
      for (unsigned int x = 0; x <= numNodes - 1; x += 1) {
        distanceYtoX = pathDistanceMatrix[yXwidth + x];
        distanceYtoK = pathDistanceMatrix[yXwidth + k];
        distanceKtoX = pathDistanceMatrix[k * width + x];
        indirectDistance = distanceYtoK + distanceKtoX;
        if (indirectDistance < distanceYtoX) {
          pathDistanceMatrix[yXwidth + x] = indirectDistance;
          pathMatrix[yXwidth + x] = k;
        }
      }
    }
  }
}
/*!
 * The floyd Warshall algorithm is a multipass algorithm
 * that calculates the shortest path between each pair of
 * nodes represented by pathDistanceBuffer.
 *
 * In each pass a node k is introduced and the pathDistanceBuffer
 * which has the shortest distance between each pair of nodes
 * considering the (k-1) nodes (that are introduced in the previous
 * passes) is updated such that
 *
 * ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) + ShortestPath(k,y,k-1))
 * where x and y are the pair of nodes between which the shortest distance
 * is being calculated.
 *
 * pathBuffer stores the intermediate nodes through which the shortest
 * path goes for each pair of nodes.
 *
 * numNodes is the number of nodes in the graph.
 *
 * for more detailed explaination of the algorithm kindly refer to the document
 * provided with the sample
 */

int main(int argc,char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of nodes> <iterations> <block size>\n",argv[0]);
    return 1;
  }
// There are three required command-line arguments
  unsigned int numNodes = (atoi(argv[1]));
  unsigned int numIterations = (atoi(argv[2]));
  unsigned int blockSize = (atoi(argv[3]));
// numNodes is a multiple of blockSize
  if (numNodes % blockSize != 0) {
    numNodes = (numNodes / blockSize + 1) * blockSize;
  }
// allocate and init memory used by host
  unsigned int *pathMatrix = 0L;
  unsigned int *pathDistanceMatrix = 0L;
  unsigned int *verificationPathDistanceMatrix = 0L;
  unsigned int *verificationPathMatrix = 0L;
  unsigned int matrixSize;
  unsigned int matrixSizeBytes;
  matrixSize = numNodes * numNodes;
  matrixSizeBytes = ((numNodes * numNodes) * sizeof(unsigned int ));
  pathDistanceMatrix = ((unsigned int *)(malloc(matrixSizeBytes)));
  ((bool )(pathDistanceMatrix != 0L))?((void )0) : __assert_fail("pathDistanceMatrix != NULL","main.cpp",140,__PRETTY_FUNCTION__);
  pathMatrix = ((unsigned int *)(malloc(matrixSizeBytes)));
  ((bool )(pathMatrix != 0L))?((void )0) : __assert_fail("pathMatrix != NULL","main.cpp",143,__PRETTY_FUNCTION__);
// input must be initialized; otherwise host and device results may be different
  srand(2);
  for (unsigned int i = 0; i <= numNodes - 1; i += 1) {
    for (unsigned int j = 0; j <= numNodes - 1; j += 1) {
      int index = (i * numNodes + j);
      pathDistanceMatrix[index] = (rand() % (200 + 1));
    }
  }
  for (unsigned int i = 0; i <= numNodes - 1; i += 1) {
    unsigned int iXWidth = i * numNodes;
    pathDistanceMatrix[iXWidth + i] = 0;
  }
/*
   * pathMatrix is the intermediate node from which the path passes
   * pathMatrix(i,j) = k means the shortest path from i to j
   * passes through an intermediate node k
   * Initialized such that pathMatrix(i,j) = i
   */
  for (unsigned int i = 0; i <= numNodes - 1; i += 1) {
    for (unsigned int j = 0; j <= i - 1; j += 1) {
      pathMatrix[i * numNodes + j] = i;
      pathMatrix[j * numNodes + i] = j;
    }
    pathMatrix[i * numNodes + i] = i;
  }
  verificationPathDistanceMatrix = ((unsigned int *)(malloc(matrixSizeBytes)));
  ((bool )(verificationPathDistanceMatrix != 0L))?((void )0) : __assert_fail("verificationPathDistanceMatrix != NULL","main.cpp",176,__PRETTY_FUNCTION__);
  verificationPathMatrix = ((unsigned int *)(malloc(matrixSizeBytes)));
  ((bool )(verificationPathMatrix != 0L))?((void )0) : __assert_fail("verificationPathMatrix != NULL","main.cpp",179,__PRETTY_FUNCTION__);
  memcpy(verificationPathDistanceMatrix,pathDistanceMatrix,matrixSizeBytes);
  memcpy(verificationPathMatrix,pathMatrix,matrixSizeBytes);
  unsigned int numPasses = numNodes;
{
    float total_time = 0.f;
    for (unsigned int n = 0; n <= numIterations - 1; n += 1) {
/*
       * The floyd Warshall algorithm is a multipass algorithm
       * that calculates the shortest path between each pair of
       * nodes represented by pathDistanceBuffer.
       *
       * In each pass a node k is introduced and the pathDistanceBuffer
       * which has the shortest distance between each pair of nodes
       * considering the (k-1) nodes (that are introduced in the previous
       * passes) is updated such that
       *
       * ShortestPath(x,y,k) = min(ShortestPath(x,y,k-1), ShortestPath(x,k,k-1) + ShortestPath(k,y,k-1))
       * where x and y are the pair of nodes between which the shortest distance
       * is being calculated.
       *
       * pathBuffer stores the intermediate nodes through which the shortest
       * path goes for each pair of nodes.
       */
      auto start = std::chrono::_V2::steady_clock::now();
      for (unsigned int k = 0; k <= numPasses - 1; k += 1) {
        
#pragma omp parallel for private (x)
        for (unsigned int y = 0; y <= numNodes - 1; y += 1) {
          
#pragma omp parallel for
          for (unsigned int x = 0; x <= numNodes - 1; x += 1) {
            unsigned int distanceYtoX = pathDistanceMatrix[y * numNodes + x];
            unsigned int distanceYtoK = pathDistanceMatrix[y * numNodes + k];
            unsigned int distanceKtoX = pathDistanceMatrix[k * numNodes + x];
            unsigned int indirectDistance = distanceYtoK + distanceKtoX;
            if (indirectDistance < distanceYtoX) {
              pathDistanceMatrix[y * numNodes + x] = indirectDistance;
              pathMatrix[y * numNodes + x] = k;
            }
          }
        }
      }
      auto end = std::chrono::_V2::steady_clock::now();
      auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
      total_time += time;
    }
    printf("Average kernel execution time %f (s)\n",(total_time * 1e-9f / numIterations));
  }
// verify
  floydWarshallCPUReference(verificationPathDistanceMatrix,verificationPathMatrix,numNodes);
  if (memcmp(pathDistanceMatrix,verificationPathDistanceMatrix,(numNodes * numNodes) * sizeof(unsigned int )) == 0) {
    printf("PASS\n");
  }
   else {
    printf("FAIL\n");
    if (numNodes <= 8) {
      for (unsigned int i = 0; i <= numNodes - 1; i += 1) {
        for (unsigned int j = 0; j <= numNodes - 1; j += 1) {
          printf("host: %u ",verificationPathDistanceMatrix[i * numNodes + j]);
        }
        printf("\n");
      }
      for (unsigned int i = 0; i <= numNodes - 1; i += 1) {
        for (unsigned int j = 0; j <= numNodes - 1; j += 1) {
          printf("device: %u ",pathDistanceMatrix[i * numNodes + j]);
        }
        printf("\n");
      }
    }
  }
  free(pathDistanceMatrix);
  free(pathMatrix);
  free(verificationPathDistanceMatrix);
  free(verificationPathMatrix);
  return 0;
}
