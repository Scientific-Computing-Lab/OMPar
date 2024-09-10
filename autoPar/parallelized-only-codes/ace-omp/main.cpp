#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
//define the data set size (cubic volume)
#define DATAXSIZE 100
#define DATAYSIZE 100
#define DATAZSIZE 100
#define SQ(x) ((x)*(x))
#include <omp.h> 
typedef double nRarray[100][100];
#ifdef VERIFY
#include <string.h>
#include "reference.h"
#endif

double dFphi(double phi,double u,double lambda)
{
  return -phi * (1.0 - phi * phi) + lambda * u * (1.0 - phi * phi) * (1.0 - phi * phi);
}

double GradientX(double phi[][100][100],double dx,double dy,double dz,int x,int y,int z)
{
  return (phi[x + 1][y][z] - phi[x - 1][y][z]) / (2.0 * dx);
}

double GradientY(double phi[][100][100],double dx,double dy,double dz,int x,int y,int z)
{
  return (phi[x][y + 1][z] - phi[x][y - 1][z]) / (2.0 * dy);
}

double GradientZ(double phi[][100][100],double dx,double dy,double dz,int x,int y,int z)
{
  return (phi[x][y][z + 1] - phi[x][y][z - 1]) / (2.0 * dz);
}

double Divergence(double phix[][100][100],double phiy[][100][100],double phiz[][100][100],double dx,double dy,double dz,int x,int y,int z)
{
  return GradientX(phix,dx,dy,dz,x,y,z) + GradientY(phiy,dx,dy,dz,x,y,z) + GradientZ(phiz,dx,dy,dz,x,y,z);
}

double Laplacian(double phi[][100][100],double dx,double dy,double dz,int x,int y,int z)
{
  double phixx = (phi[x + 1][y][z] + phi[x - 1][y][z] - 2.0 * phi[x][y][z]) / (dx * dx);
  double phiyy = (phi[x][y + 1][z] + phi[x][y - 1][z] - 2.0 * phi[x][y][z]) / (dy * dy);
  double phizz = (phi[x][y][z + 1] + phi[x][y][z - 1] - 2.0 * phi[x][y][z]) / (dz * dz);
  return phixx + phiyy + phizz;
}

double An(double phix,double phiy,double phiz,double epsilon)
{
  if (phix != 0.0 || phiy != 0.0 || phiz != 0.0) {
    return (1.0 - 3.0 * epsilon) * (1.0 + 4.0 * epsilon / (1.0 - 3.0 * epsilon) * ((phix * phix * (phix * phix) + phiy * phiy * (phiy * phiy) + phiz * phiz * (phiz * phiz)) / ((phix * phix + phiy * phiy + phiz * phiz) * (phix * phix + phiy * phiy + phiz * phiz))));
  }
   else {
    return 1.0 - 5.0 / 3.0 * epsilon;
  }
}

double Wn(double phix,double phiy,double phiz,double epsilon,double W0)
{
  return W0 * An(phix,phiy,phiz,epsilon);
}

double taun(double phix,double phiy,double phiz,double epsilon,double tau0)
{
  return tau0 * (An(phix,phiy,phiz,epsilon) * An(phix,phiy,phiz,epsilon));
}

double dFunc(double l,double m,double n)
{
  if (l != 0.0 || m != 0.0 || n != 0.0) {
    return (l * l * l * (m * m + n * n) - l * (m * m * (m * m) + n * n * (n * n))) / ((l * l + m * m + n * n) * (l * l + m * m + n * n));
  }
   else {
    return 0.0;
  }
}

void calculateForce(double phi[][100][100],double Fx[][100][100],double Fy[][100][100],double Fz[][100][100],double dx,double dy,double dz,double epsilon,double W0,double tau0)
{
  for (int ix = 0; ix <= 99; ix += 1) {
    for (int iy = 0; iy <= 99; iy += 1) {
      for (int iz = 0; iz <= 99; iz += 1) {
        if (ix < 100 - 1 && iy < 100 - 1 && iz < 100 - 1 && ix > 0 && iy > 0 && iz > 0) {
          double phix = GradientX(phi,dx,dy,dz,ix,iy,iz);
          double phiy = GradientY(phi,dx,dy,dz,ix,iy,iz);
          double phiz = GradientZ(phi,dx,dy,dz,ix,iy,iz);
          double sqGphi = phix * phix + phiy * phiy + phiz * phiz;
          double c = 16.0 * W0 * epsilon;
          double w = Wn(phix,phiy,phiz,epsilon,W0);
          double w2 = w * w;
          Fx[ix][iy][iz] = w2 * phix + sqGphi * w * c * dFunc(phix,phiy,phiz);
          Fy[ix][iy][iz] = w2 * phiy + sqGphi * w * c * dFunc(phiy,phiz,phix);
          Fz[ix][iy][iz] = w2 * phiz + sqGphi * w * c * dFunc(phiz,phix,phiy);
        }
         else {
          Fx[ix][iy][iz] = 0.0;
          Fy[ix][iy][iz] = 0.0;
          Fz[ix][iy][iz] = 0.0;
        }
      }
    }
  }
}
// device function to set the 3D volume

void allenCahn(double phinew[][100][100],double phiold[][100][100],double uold[][100][100],double Fx[][100][100],double Fy[][100][100],double Fz[][100][100],double epsilon,double W0,double tau0,double lambda,double dt,double dx,double dy,double dz)
{
  for (int ix = 1; ix <= 98; ix += 1) {
    for (int iy = 1; iy <= 98; iy += 1) {
      for (int iz = 1; iz <= 98; iz += 1) {
        double phix = GradientX(phiold,dx,dy,dz,ix,iy,iz);
        double phiy = GradientY(phiold,dx,dy,dz,ix,iy,iz);
        double phiz = GradientZ(phiold,dx,dy,dz,ix,iy,iz);
        phinew[ix][iy][iz] = phiold[ix][iy][iz] + dt / taun(phix,phiy,phiz,epsilon,tau0) * (Divergence(Fx,Fy,Fz,dx,dy,dz,ix,iy,iz) - dFphi(phiold[ix][iy][iz],uold[ix][iy][iz],lambda));
      }
    }
  }
}

void boundaryConditionsPhi(double phinew[][100][100])
{
  
#pragma omp parallel for private (iy,iz)
  for (int ix = 0; ix <= 99; ix += 1) {
    
#pragma omp parallel for private (iz)
    for (int iy = 0; iy <= 99; iy += 1) {
      
#pragma omp parallel for
      for (int iz = 0; iz <= 99; iz += 1) {
        if (ix == 0) {
          phinew[ix][iy][iz] = - 1.0;
        }
         else if (ix == 100 - 1) {
          phinew[ix][iy][iz] = - 1.0;
        }
         else if (iy == 0) {
          phinew[ix][iy][iz] = - 1.0;
        }
         else if (iy == 100 - 1) {
          phinew[ix][iy][iz] = - 1.0;
        }
         else if (iz == 0) {
          phinew[ix][iy][iz] = - 1.0;
        }
         else if (iz == 100 - 1) {
          phinew[ix][iy][iz] = - 1.0;
        }
      }
    }
  }
}

void thermalEquation(double unew[][100][100],double uold[][100][100],double phinew[][100][100],double phiold[][100][100],double D,double dt,double dx,double dy,double dz)
{
  for (int ix = 1; ix <= 98; ix += 1) {
    for (int iy = 1; iy <= 98; iy += 1) {
      for (int iz = 1; iz <= 98; iz += 1) {
        unew[ix][iy][iz] = uold[ix][iy][iz] + 0.5 * (phinew[ix][iy][iz] - phiold[ix][iy][iz]) + dt * D * Laplacian(uold,dx,dy,dz,ix,iy,iz);
      }
    }
  }
}

void boundaryConditionsU(double unew[][100][100],double delta)
{
  
#pragma omp parallel for private (iy,iz)
  for (int ix = 0; ix <= 99; ix += 1) {
    
#pragma omp parallel for private (iz)
    for (int iy = 0; iy <= 99; iy += 1) {
      
#pragma omp parallel for firstprivate (delta)
      for (int iz = 0; iz <= 99; iz += 1) {
        if (ix == 0) {
          unew[ix][iy][iz] = -delta;
        }
         else if (ix == 100 - 1) {
          unew[ix][iy][iz] = -delta;
        }
         else if (iy == 0) {
          unew[ix][iy][iz] = -delta;
        }
         else if (iy == 100 - 1) {
          unew[ix][iy][iz] = -delta;
        }
         else if (iz == 0) {
          unew[ix][iy][iz] = -delta;
        }
         else if (iz == 100 - 1) {
          unew[ix][iy][iz] = -delta;
        }
      }
    }
  }
}

void swapGrid(double cnew[][100][100],double cold[][100][100])
{
  
#pragma omp parallel for private (iy,iz)
  for (int ix = 0; ix <= 99; ix += 1) {
    
#pragma omp parallel for private (iz)
    for (int iy = 0; iy <= 99; iy += 1) {
      
#pragma omp parallel for
      for (int iz = 0; iz <= 99; iz += 1) {
        double tmp = cnew[ix][iy][iz];
        cnew[ix][iy][iz] = cold[ix][iy][iz];
        cold[ix][iy][iz] = tmp;
      }
    }
  }
}

void initializationPhi(double phi[][100][100],double r0)
{
  
#pragma omp parallel for private (iy,iz)
  for (int ix = 0; ix <= 99; ix += 1) {
    
#pragma omp parallel for private (iz)
    for (int iy = 0; iy <= 99; iy += 1) {
      
#pragma omp parallel for firstprivate (r0)
      for (int iz = 0; iz <= 99; iz += 1) {
        double r = sqrt((ix - 0.5 * 100) * (ix - 0.5 * 100) + (iy - 0.5 * 100) * (iy - 0.5 * 100) + (iz - 0.5 * 100) * (iz - 0.5 * 100));
        if (r < r0) {
          phi[ix][iy][iz] = 1.0;
        }
         else {
          phi[ix][iy][iz] = - 1.0;
        }
      }
    }
  }
}

void initializationU(double u[][100][100],double r0,double delta)
{
  for (int ix = 0; ix <= 99; ix += 1) {
    for (int iy = 0; iy <= 99; iy += 1) {
      for (int iz = 0; iz <= 99; iz += 1) {
        double r = sqrt((ix - 0.5 * 100) * (ix - 0.5 * 100) + (iy - 0.5 * 100) * (iy - 0.5 * 100) + (iz - 0.5 * 100) * (iz - 0.5 * 100));
        if (r < r0) {
          u[ix][iy][iz] = 0.0;
        }
         else {
          u[ix][iy][iz] = -delta * (1.0 - exp(-(r - r0)));
        }
      }
    }
  }
}

int main(int argc,char *argv[])
{
  const int num_steps = atoi(argv[1]);
//6000;
  const double dx = 0.4;
  const double dy = 0.4;
  const double dz = 0.4;
  const double dt = 0.01;
  const double delta = 0.8;
  const double r0 = 5.0;
  const double epsilon = 0.07;
  const double W0 = 1.0;
  const double beta0 = 0.0;
  const double D = 2.0;
  const double d0 = 0.5;
  const double a1 = 1.25 / sqrt(2.0);
  const double a2 = 0.64;
  const double lambda = W0 * a1 / d0;
  const double tau0 = W0 * W0 * W0 * a1 * a2 / (d0 * D) + W0 * W0 * beta0 / d0;
// overall data set sizes
  const int nx = 100;
  const int ny = 100;
  const int nz = 100;
  const int vol = nx * ny * nz;
  const size_t vol_in_bytes = sizeof(double ) * vol;
// storage for result stored on host
  nRarray *phi_host = (nRarray *)(malloc(vol_in_bytes));
  nRarray *u_host = (nRarray *)(malloc(vol_in_bytes));
  initializationPhi(phi_host,r0);
  initializationU(u_host,r0,delta);
#ifdef VERIFY
#endif 
  auto offload_start = std::chrono::_V2::steady_clock::now();
// storage for result computed on device
  double *d_phiold = (double *)phi_host;
  double *d_uold = (double *)u_host;
  double *d_phinew = (double *)(malloc(vol_in_bytes));
  double *d_unew = (double *)(malloc(vol_in_bytes));
  double *d_Fx = (double *)(malloc(vol_in_bytes));
  double *d_Fy = (double *)(malloc(vol_in_bytes));
  double *d_Fz = (double *)(malloc(vol_in_bytes));
{
    int t = 0;
    auto start = std::chrono::_V2::steady_clock::now();
    while(t <= num_steps){
      calculateForce((nRarray *)d_phiold,(nRarray *)d_Fx,(nRarray *)d_Fy,(nRarray *)d_Fz,dx,dy,dz,epsilon,W0,tau0);
      allenCahn((nRarray *)d_phinew,(nRarray *)d_phiold,(nRarray *)d_uold,(nRarray *)d_Fx,(nRarray *)d_Fy,(nRarray *)d_Fz,epsilon,W0,tau0,lambda,dt,dx,dy,dz);
      boundaryConditionsPhi((nRarray *)d_phinew);
      thermalEquation((nRarray *)d_unew,(nRarray *)d_uold,(nRarray *)d_phinew,(nRarray *)d_phiold,D,dt,dx,dy,dz);
      boundaryConditionsU((nRarray *)d_unew,delta);
      swapGrid((nRarray *)d_phinew,(nRarray *)d_phiold);
      swapGrid((nRarray *)d_unew,(nRarray *)d_uold);
      t++;
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Total kernel execution time: %.3f (ms)\n",(time * 1e-6f));
  }
  auto offload_end = std::chrono::_V2::steady_clock::now();
  auto offload_time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((offload_end-offload_start)) . count();
  printf("Offload time: %.3f (ms)\n",(offload_time * 1e-6f));
#ifdef VERIFY
#endif
  free(phi_host);
  free(u_host);
  free(d_phinew);
  free(d_unew);
  free(d_Fx);
  free(d_Fy);
  free(d_Fz);
  return 0;
}
