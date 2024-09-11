/***************************************************************************
 *
 *            (C) Copyright 2007 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 256
#define KERNEL_Q_THREADS_PER_BLOCK 256
#define KERNEL_Q_K_ELEMS_PER_GRID 1024

#include <string.h>  // memcpy

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float PhiMag;
};

void computeQ_GPU(
  int numK,
  int numX,
  float *x,
  float *y,
  float *z,
  kValues *kVals,
  kValues *ck,
  float *Qr,
  float *Qi)
{
  int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
  if (numK % KERNEL_Q_K_ELEMS_PER_GRID) QGrids++;

  int QBlocks = numX / KERNEL_Q_THREADS_PER_BLOCK;
  if (numX % KERNEL_Q_THREADS_PER_BLOCK) QBlocks++;

/* Values in the k-space coordinate system are stored in device memory
 * on the GPU */
  for (int QGrid = 0; QGrid < QGrids; QGrid++) {
    // Put the tile of K values into constant mem
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);

    memcpy(ck, kValsTile, numElems * sizeof(kValues));

    int kGlobalIndex = QGridBase;

    for (int xIndex = 0; xIndex < numX; xIndex++) {
      // Read block's X values from global mem to shared mem
      float sX = x[xIndex];
      float sY = y[xIndex];
      float sZ = z[xIndex];
      float sQr = Qr[xIndex];
      float sQi = Qi[xIndex];

      // Loop over all elements of K in constant mem to compute a partial value
      // for X.
      int kIndex = 0;
      if (numK % 2) {
        float expArg = PIx2 * (ck[0].Kx * sX + ck[0].Ky * sY + ck[0].Kz * sZ);
        sQr += ck[0].PhiMag * cosf(expArg);
        sQi += ck[0].PhiMag * sinf(expArg);
        kIndex++;
        kGlobalIndex++;
      }

      for (; (kIndex < KERNEL_Q_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
           kIndex += 2, kGlobalIndex += 2) {
        float expArg = PIx2 * (ck[kIndex].Kx * sX +
            		   ck[kIndex].Ky * sY +
            		   ck[kIndex].Kz * sZ);
        sQr += ck[kIndex].PhiMag * cosf(expArg);
        sQi += ck[kIndex].PhiMag * sinf(expArg);

        int kIndex1 = kIndex + 1;
        float expArg1 = PIx2 * (ck[kIndex1].Kx * sX +
            		    ck[kIndex1].Ky * sY +
            		    ck[kIndex1].Kz * sZ);
        sQr += ck[kIndex1].PhiMag * cosf(expArg1);
        sQi += ck[kIndex1].PhiMag * sinf(expArg1);
      }

      Qr[xIndex] = sQr;
      Qi[xIndex] = sQi;
    }
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
	 float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
}

