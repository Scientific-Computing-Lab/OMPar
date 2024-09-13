#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
// 2D block size
#define BSIZE 16
// Tile size in the x direction
#define XTILE 20
#include <omp.h> 
typedef float Real;

void stencil3d(const Real *d_psi,Real *d_npsi,const Real *d_sigmaX,const Real *d_sigmaY,const Real *d_sigmaZ,int bdimx,int bdimy,int bdimz,int nx,int ny,int nz)
{
{
    Real sm_psi[4][16][16];
{
      #define V0(y,z) sm_psi[pii][y][z]
      #define V1(y,z) sm_psi[cii][y][z]
      #define V2(y,z) sm_psi[nii][y][z]
      #define sigmaX(x,y,z,dir) d_sigmaX[ z + nz * ( y + ny * ( x + nx * dir ) ) ]
      #define sigmaY(x,y,z,dir) d_sigmaY[ z + nz * ( y + ny * ( x + nx * dir ) ) ]
      #define sigmaZ(x,y,z,dir) d_sigmaZ[ z + nz * ( y + ny * ( x + nx * dir ) ) ]
      #define psi(x,y,z) d_psi[ z + nz * ( (y) + ny * (x) ) ]
      #define npsi(x,y,z) d_npsi[ z + nz * ( (y) + ny * (x) ) ]
      const int tjj = omp_get_thread_num() / 16;
      const int tkk = omp_get_thread_num() % 16;
      const int blockIdx_x = omp_get_team_num() % bdimx;
      const int blockIdx_y = omp_get_team_num() / bdimx % bdimy;
      const int blockIdx_z = omp_get_team_num() / (bdimx * bdimy);
      const int gridDim_x = bdimx;
      const int gridDim_y = bdimy;
      const int gridDim_z = bdimz;
// shift for each tile by updating device pointers
      d_psi = &d_psi[(16 - 2) * blockIdx_z + nz * ((16 - 2) * blockIdx_y + ny * (20 * blockIdx_x))];
      d_npsi = &d_npsi[(16 - 2) * blockIdx_z + nz * ((16 - 2) * blockIdx_y + ny * (20 * blockIdx_x))];
      d_sigmaX = &d_sigmaX[(16 - 2) * blockIdx_z + nz * ((16 - 2) * blockIdx_y + ny * (20 * blockIdx_x + nx * 0))];
      d_sigmaY = &d_sigmaY[(16 - 2) * blockIdx_z + nz * ((16 - 2) * blockIdx_y + ny * (20 * blockIdx_x + nx * 0))];
      d_sigmaZ = &d_sigmaZ[(16 - 2) * blockIdx_z + nz * ((16 - 2) * blockIdx_y + ny * (20 * blockIdx_x + nx * 0))];
      int nLast_x = 20 + 1;
      int nLast_y = 16 - 1;
      int nLast_z = 16 - 1;
      if (blockIdx_x == gridDim_x - 1) 
        nLast_x = nx - 2 - 20 * blockIdx_x + 1;
      if (blockIdx_y == gridDim_y - 1) 
        nLast_y = ny - 2 - (16 - 2) * blockIdx_y + 1;
      if (blockIdx_z == gridDim_z - 1) 
        nLast_z = nz - 2 - (16 - 2) * blockIdx_z + 1;
// previous, current, next, and temp indices
      int pii;
      int cii;
      int nii;
      int tii;
      Real xcharge;
      Real ycharge;
      Real zcharge;
      Real dV = 0;
      if (tjj <= nLast_y && tkk <= nLast_z) {
        pii = 0;
        cii = 1;
        nii = 2;
        sm_psi[cii][tjj][tkk] = d_psi[tkk + nz * (tjj + ny * 0)];
        sm_psi[nii][tjj][tkk] = d_psi[tkk + nz * (tjj + ny * 1)];
      }
//initial
      if (tkk > 0 && tkk < nLast_z && tjj > 0 && tjj < nLast_y) {
        Real xd = -sm_psi[cii][tjj][tkk] + sm_psi[nii][tjj][tkk];
        Real yd = ((-sm_psi[cii][- 1 + tjj][tkk] + sm_psi[cii][1 + tjj][tkk] - sm_psi[nii][- 1 + tjj][tkk] + sm_psi[nii][1 + tjj][tkk]) / 4.);
        Real zd = ((-sm_psi[cii][tjj][- 1 + tkk] + sm_psi[cii][tjj][1 + tkk] - sm_psi[nii][tjj][- 1 + tkk] + sm_psi[nii][tjj][1 + tkk]) / 4.);
        dV -= d_sigmaX[tkk + nz * (tjj + ny * (1 + nx * 0))] * xd + d_sigmaX[tkk + nz * (tjj + ny * (1 + nx * 1))] * yd + d_sigmaX[tkk + nz * (tjj + ny * (1 + nx * 2))] * zd;
      }
      if (tjj <= nLast_y && tkk <= nLast_z) {
        tii = pii;
        pii = cii;
        cii = nii;
        nii = tii;
      }
      for (int ii = 1; ii <= nLast_x - 1; ii += 1) {
        if (tjj <= nLast_y && tkk <= nLast_z) 
          sm_psi[nii][tjj][tkk] = d_psi[tkk + nz * (tjj + ny * (ii + 1))];
// y face current
        if (tkk > 0 && tkk < nLast_z && tjj < nLast_y) {
          Real xd = ((-sm_psi[pii][tjj][tkk] - sm_psi[pii][1 + tjj][tkk] + sm_psi[nii][tjj][tkk] + sm_psi[nii][1 + tjj][tkk]) / 4.);
          Real yd = -sm_psi[cii][tjj][tkk] + sm_psi[cii][1 + tjj][tkk];
          Real zd = ((-sm_psi[cii][tjj][- 1 + tkk] + sm_psi[cii][tjj][1 + tkk] - sm_psi[cii][1 + tjj][- 1 + tkk] + sm_psi[cii][1 + tjj][1 + tkk]) / 4.);
          ycharge = d_sigmaY[tkk + nz * (tjj + 1 + ny * (ii + nx * 0))] * xd + d_sigmaY[tkk + nz * (tjj + 1 + ny * (ii + nx * 1))] * yd + d_sigmaY[tkk + nz * (tjj + 1 + ny * (ii + nx * 2))] * zd;
          dV += ycharge;
          sm_psi[3][tjj][tkk] = ycharge;
        }
        if (tkk > 0 && tkk < nLast_z && tjj > 0 && tjj < nLast_y) 
          dV -= sm_psi[3][tjj - 1][tkk];
//bring from left
// z face current
        if (tkk < nLast_z && tjj > 0 && tjj < nLast_y) {
          Real xd = ((-sm_psi[pii][tjj][tkk] - sm_psi[pii][tjj][1 + tkk] + sm_psi[nii][tjj][tkk] + sm_psi[nii][tjj][1 + tkk]) / 4.);
          Real yd = ((-sm_psi[cii][- 1 + tjj][tkk] - sm_psi[cii][- 1 + tjj][1 + tkk] + sm_psi[cii][1 + tjj][tkk] + sm_psi[cii][1 + tjj][1 + tkk]) / 4.);
          Real zd = -sm_psi[cii][tjj][tkk] + sm_psi[cii][tjj][1 + tkk];
          zcharge = d_sigmaZ[tkk + 1 + nz * (tjj + ny * (ii + nx * 0))] * xd + d_sigmaZ[tkk + 1 + nz * (tjj + ny * (ii + nx * 1))] * yd + d_sigmaZ[tkk + 1 + nz * (tjj + ny * (ii + nx * 2))] * zd;
          dV += zcharge;
          sm_psi[3][tjj][tkk] = zcharge;
        }
        if (tkk > 0 && tkk < nLast_z && tjj > 0 && tjj < nLast_y) 
          dV -= sm_psi[3][tjj][tkk - 1];
// x face current
        if (tkk > 0 && tkk < nLast_z && tjj > 0 && tjj < nLast_y) {
          Real xd = -sm_psi[cii][tjj][tkk] + sm_psi[nii][tjj][tkk];
          Real yd = ((-sm_psi[cii][- 1 + tjj][tkk] + sm_psi[cii][1 + tjj][tkk] - sm_psi[nii][- 1 + tjj][tkk] + sm_psi[nii][1 + tjj][tkk]) / 4.);
          Real zd = ((-sm_psi[cii][tjj][- 1 + tkk] + sm_psi[cii][tjj][1 + tkk] - sm_psi[nii][tjj][- 1 + tkk] + sm_psi[nii][tjj][1 + tkk]) / 4.);
          xcharge = d_sigmaX[tkk + nz * (tjj + ny * (ii + 1 + nx * 0))] * xd + d_sigmaX[tkk + nz * (tjj + ny * (ii + 1 + nx * 1))] * yd + d_sigmaX[tkk + nz * (tjj + ny * (ii + 1 + nx * 2))] * zd;
          dV += xcharge;
          d_npsi[tkk + nz * (tjj + ny * ii)] = dV;
//store dV
          dV = -xcharge;
//pass to the next cell in x-dir
        }
        if (tjj <= nLast_y && tkk <= nLast_z) {
          tii = pii;
          pii = cii;
          cii = nii;
          nii = tii;
        }
      }
    }
  }
}

int main(int argc,char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <grid dimension> <repeat>\n",argv[0]);
    return 1;
  }
  const int size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int nx = size;
  const int ny = size;
  const int nz = size;
  const int vol = nx * ny * nz;
  printf("Grid dimension: nx=%d ny=%d nz=%d\n",nx,ny,nz);
// allocate and initialize Vm
  Real *h_Vm = (Real *)(malloc(sizeof(Real ) * vol));
/* (previously processed: ignoring self-referential macro declaration) macro name = h_Vm */ 
  for (int ii = 0; ii <= nx - 1; ii += 1) {
    
#pragma omp parallel for private (kk)
    for (int jj = 0; jj <= ny - 1; jj += 1) {
      
#pragma omp parallel for
      for (int kk = 0; kk <= nz - 1; kk += 1) {
        h_Vm[kk + nz * (jj + ny * ii)] = ((ii * (ny * nz) + jj * nz + kk) % 19);
      }
    }
  }
// allocate and initialize sigma
  Real *h_sigma = (Real *)(malloc(sizeof(Real ) * vol * 9));
  
#pragma omp parallel for
  for (int i = 0; i <= vol * 9 - 1; i += 1) {
    h_sigma[i] = (i % 19);
  }
// reset dVm
  Real *h_dVm = (Real *)(malloc(sizeof(Real ) * vol));
  memset(h_dVm,0,sizeof(Real ) * vol);
//determine block sizes
  int bdimz = (nz - 2) / (16 - 2) + (((nz - 2) % (16 - 2) == 0?0 : 1));
  int bdimy = (ny - 2) / (16 - 2) + (((ny - 2) % (16 - 2) == 0?0 : 1));
  int bdimx = (nx - 2) / 20 + (((nx - 2) % 20 == 0?0 : 1));
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      stencil3d(h_Vm,h_dVm,h_sigma,(h_sigma + 3 * vol),(h_sigma + 6 * vol),bdimx,bdimy,bdimz,nx,ny,nz);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
  }
#ifdef DUMP
#endif
  free(h_sigma);
  free(h_Vm);
  free(h_dVm);
  return 0;
}
