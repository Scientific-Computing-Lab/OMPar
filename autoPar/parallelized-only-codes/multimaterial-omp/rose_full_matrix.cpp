#include <math.h>
#include <stdio.h>
#include <omp.h> 

struct full_data 
{
  int sizex;
  int sizey;
  int Nmats;
  double *rho;
  double *rho_mat_ave;
  double *p;
  double *Vf;
  double *t;
  double *V;
  double *x;
  double *y;
  double *n;
  double *rho_ave;
}
;

void full_matrix_cell_centric(struct full_data cc)
{
  int sizex = cc . sizex;
  int sizey = cc . sizey;
  int Nmats = cc . Nmats;
  double *Vf = cc . Vf;
  double *V = cc . V;
  double *rho = cc . rho;
  double *rho_ave = cc . rho_ave;
  double *p = cc . p;
  double *t = cc . t;
  double *x = cc . x;
  double *y = cc . y;
  double *n = cc . n;
  double *rho_mat_ave = cc . rho_mat_ave;
#if defined(NACC)
#endif
{
// Cell-centric algorithms
// Computational loop 1 - average density in cell
//double t1 = omp_get_wtime();
#if defined(OMP)
#elif defined(NACC)
#endif
    
#pragma omp parallel for private (i,mat)
    for (int j = 0; j <= sizey - 1; j += 1) {
#if defined(NACC)
#endif
      
#pragma omp parallel for private (mat) firstprivate (Nmats)
      for (int i = 0; i <= sizex - 1; i += 1) {
        double ave = 0.0;
//#pragma omp simd reduction(+:ave)
        
#pragma omp parallel for reduction (+:ave)
        for (int mat = 0; mat <= Nmats - 1; mat += 1) {
// Optimisation:
          if (Vf[(i + sizex * j) * Nmats + mat] > 0.0) 
            ave += rho[(i + sizex * j) * Nmats + mat] * Vf[(i + sizex * j) * Nmats + mat];
        }
        rho_ave[i + sizex * j] = ave / V[i + sizex * j];
      }
    }
#ifdef DEBUG
#endif
// Computational loop 2 - Pressure for each cell and each material
//t1 = omp_get_wtime();
#if defined(OMP)
#elif defined(NACC)
#endif
    for (int j = 0; j <= sizey - 1; j += 1) {
#if defined(NACC)
#endif
      
#pragma omp parallel for private (mat_nom_3) firstprivate (Nmats)
      for (int i = 0; i <= sizex - 1; i += 1) {
#if defined(NACC)
#endif
//#pragma omp simd
        
#pragma omp parallel for
        for (int mat = 0; mat <= Nmats - 1; mat += 1) {
          if (Vf[(i + sizex * j) * Nmats + mat] > 0.0) {
            double nm = n[mat];
            p[(i + sizex * j) * Nmats + mat] = nm * rho[(i + sizex * j) * Nmats + mat] * t[(i + sizex * j) * Nmats + mat] / Vf[(i + sizex * j) * Nmats + mat];
          }
           else {
            p[(i + sizex * j) * Nmats + mat] = 0.0;
          }
        }
      }
    }
#ifdef DEBUG
#endif
// Computational loop 3 - Average density of each material over neighborhood of each cell
//t1 = omp_get_wtime();
#if defined(OMP)
#elif defined(NACC)
#endif
    for (int j = 1; j <= sizey - 1 - 1; j += 1) {
#if defined(NACC)
#endif
      for (int i = 1; i <= sizex - 1 - 1; i += 1) {
// o: outer
        double xo = x[i + sizex * j];
        double yo = y[i + sizex * j];
// There are at most 9 neighbours in 2D case.
        double dsqr[9];
        
#pragma omp parallel for private (ni)
        for (int nj = - 1; nj <= 1; nj += 1) {
          
#pragma omp parallel for firstprivate (xo,yo)
          for (int ni = - 1; ni <= 1; ni += 1) {
            dsqr[(nj + 1) * 3 + (ni + 1)] = 0.0;
// i: inner
            double xi = x[i + ni + sizex * (j + nj)];
            double yi = y[i + ni + sizex * (j + nj)];
            dsqr[(nj + 1) * 3 + (ni + 1)] += (xo - xi) * (xo - xi);
            dsqr[(nj + 1) * 3 + (ni + 1)] += (yo - yi) * (yo - yi);
          }
        }
        
#pragma omp parallel for private (nj_nom_7,ni_nom_8)
        for (int mat = 0; mat <= Nmats - 1; mat += 1) {
          if (Vf[(i + sizex * j) * Nmats + mat] > 0.0) {
            double rho_sum = 0.0;
            int Nn = 0;
            
#pragma omp parallel for private (ni_nom_8) reduction (+:rho_sum,Nn)
            for (int nj = - 1; nj <= 1; nj += 1) {
              if (j + nj < 0 || j + nj >= sizey) 
// TODO: better way?
                continue; 
              
#pragma omp parallel for reduction (+:rho_sum,Nn)
              for (int ni = - 1; ni <= 1; ni += 1) {
                if (i + ni < 0 || i + ni >= sizex) 
// TODO: better way?
                  continue; 
                if (Vf[(i + ni + sizex * (j + nj)) * Nmats + mat] > 0.0) {
                  rho_sum += rho[(i + ni + sizex * (j + nj)) * Nmats + mat] / dsqr[(nj + 1) * 3 + (ni + 1)];
                  Nn += 1;
                }
              }
            }
            rho_mat_ave[(i + sizex * j) * Nmats + mat] = rho_sum / Nn;
          }
           else {
            rho_mat_ave[(i + sizex * j) * Nmats + mat] = 0.0;
          }
        }
      }
    }
#ifdef DEBUG
#endif
  }
}

void full_matrix_material_centric(struct full_data cc,struct full_data mc)
{
  int sizex = mc . sizex;
  int sizey = mc . sizey;
  int Nmats = mc . Nmats;
  int ncells = sizex * sizey;
  double *Vf = mc . Vf;
  double *V = mc . V;
  double *rho = mc . rho;
  double *rho_ave = mc . rho_ave;
  double *p = mc . p;
  double *t = mc . t;
  double *x = mc . x;
  double *y = mc . y;
  double *n = mc . n;
  double *rho_mat_ave = mc . rho_mat_ave;
#if defined(NACC)
#endif
{
// Material-centric algorithms
// Computational loop 1 - average density in cell
//double t1 = omp_get_wtime();
#if defined(OMP)
#elif defined(NACC)
#endif
    
#pragma omp parallel for private (i)
    for (int j = 0; j <= sizey - 1; j += 1) {
#if defined(NACC)
#endif
//#pragma omp simd
      
#pragma omp parallel for
      for (int i = 0; i <= sizex - 1; i += 1) {
        rho_ave[i + sizex * j] = 0.0;
      }
    }
    for (int mat = 0; mat <= Nmats - 1; mat += 1) {
#if defined(OMP)
#elif defined(NACC)
#endif
      
#pragma omp parallel for private (i_nom_10)
      for (int j = 0; j <= sizey - 1; j += 1) {
#if defined(NACC)
#endif
//#pragma omp simd
        
#pragma omp parallel for firstprivate (ncells)
        for (int i = 0; i <= sizex - 1; i += 1) {
// Optimisation:
          if (Vf[ncells * mat + i + sizex * j] > 0.0) 
            rho_ave[i + sizex * j] += rho[ncells * mat + i + sizex * j] * Vf[ncells * mat + i + sizex * j];
        }
      }
    }
#if defined(OMP)
#elif defined(NACC)
#endif
    
#pragma omp parallel for private (i_nom_12) firstprivate (sizex)
    for (int j = 0; j <= sizey - 1; j += 1) {
#if defined(NACC)
#endif
//#pragma omp simd
      
#pragma omp parallel for
      for (int i = 0; i <= sizex - 1; i += 1) {
        rho_ave[i + sizex * j] /= V[i + sizex * j];
      }
    }
#ifdef DEBUG
#endif
// Computational loop 2 - Pressure for each cell and each material
//t1 = omp_get_wtime();
#if defined(OMP)
#elif defined(NACC)
#endif
    
#pragma omp parallel for private (j_nom_14,i_nom_15)
    for (int mat = 0; mat <= Nmats - 1; mat += 1) {
#if defined(NACC)
#endif
      
#pragma omp parallel for private (i_nom_15) firstprivate (sizex)
      for (int j = 0; j <= sizey - 1; j += 1) {
#if defined(NACC)
#endif
//#pragma omp simd
        
#pragma omp parallel for firstprivate (ncells)
        for (int i = 0; i <= sizex - 1; i += 1) {
          double nm = n[mat];
          if (Vf[ncells * mat + i + sizex * j] > 0.0) {
            p[ncells * mat + i + sizex * j] = nm * rho[ncells * mat + i + sizex * j] * t[ncells * mat + i + sizex * j] / Vf[ncells * mat + i + sizex * j];
          }
           else {
            p[ncells * mat + i + sizex * j] = 0.0;
          }
        }
      }
    }
#ifdef DEBUG
#endif
// Computational loop 3 - Average density of each material over neighborhood of each cell
//t1 = omp_get_wtime();
#if defined(OMP)
#elif defined(NACC)
#endif
    
#pragma omp parallel for private (j_nom_17,i_nom_18,nj,ni) firstprivate (sizey,Nmats)
    for (int mat = 0; mat <= Nmats - 1; mat += 1) {
#if defined(NACC)
#endif
      
#pragma omp parallel for private (i_nom_18,nj,ni) firstprivate (sizex)
      for (int j = 1; j <= sizey - 1 - 1; j += 1) {
#if defined(NACC)
#endif
        
#pragma omp parallel for private (nj,ni) firstprivate (ncells)
        for (int i = 1; i <= sizex - 1 - 1; i += 1) {
          if (Vf[ncells * mat + i + sizex * j] > 0.0) {
// o: outer
            double xo = x[i + sizex * j];
            double yo = y[i + sizex * j];
            double rho_sum = 0.0;
            int Nn = 0;
            
#pragma omp parallel for private (ni) reduction (+:rho_sum,Nn)
            for (int nj = - 1; nj <= 1; nj += 1) {
              if (j + nj < 0 || j + nj >= sizey) 
// TODO: better way?
                continue; 
              
#pragma omp parallel for reduction (+:rho_sum,Nn)
              for (int ni = - 1; ni <= 1; ni += 1) {
                if (i + ni < 0 || i + ni >= sizex) 
// TODO: better way?
                  continue; 
                if (Vf[ncells * mat + (i + ni) + sizex * (j + nj)] > 0.0) {
                  double dsqr = 0.0;
// i: inner
                  double xi = x[i + ni + sizex * (j + nj)];
                  double yi = y[i + ni + sizex * (j + nj)];
                  dsqr += (xo - xi) * (xo - xi);
                  dsqr += (yo - yi) * (yo - yi);
                  rho_sum += rho[ncells * mat + (i + ni) + sizex * (j + nj)] / dsqr;
                  Nn += 1;
                }
              }
            }
            rho_mat_ave[ncells * mat + i + sizex * j] = rho_sum / Nn;
          }
           else {
            rho_mat_ave[ncells * mat + i + sizex * j] = 0.0;
          }
        }
      }
    }
#ifdef DEBUG
#endif
  }
}

bool full_matrix_check_results(struct full_data cc,struct full_data mc)
{
  int sizex = cc . sizex;
  int sizey = cc . sizey;
  int Nmats = cc . Nmats;
  int ncells = sizex * sizey;
#ifdef DEBUG
#endif
  for (int j = 0; j <= sizey - 1; j += 1) {
    for (int i = 0; i <= sizex - 1; i += 1) {
      if (fabs(cc . rho_ave[i + sizex * j] - mc . rho_ave[i + sizex * j]) > 0.0001) {
        printf("1. cell-centric and material-centric values are not equal! (%f, %f, %d, %d)\n",cc . rho_ave[i + sizex * j],mc . rho_ave[i + sizex * j],i,j);
        return false;
      }
      for (int mat = 0; mat <= Nmats - 1; mat += 1) {
        if (fabs(cc . p[(i + sizex * j) * Nmats + mat] - mc . p[ncells * mat + i + sizex * j]) > 0.0001) {
          printf("2. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",cc . p[(i + sizex * j) * Nmats + mat],mc . p[ncells * mat + i + sizex * j],i,j,mat);
          return false;
        }
        if (fabs(cc . rho_mat_ave[(i + sizex * j) * Nmats + mat] - mc . rho_mat_ave[ncells * mat + i + sizex * j]) > 0.0001) {
          printf("3. cell-centric and material-centric values are not equal! (%f, %f, %d, %d, %d)\n",cc . rho_mat_ave[(i + sizex * j) * Nmats + mat],mc . rho_mat_ave[ncells * mat + i + sizex * j],i,j,mat);
          return false;
        }
      }
    }
  }
#ifdef DEBUG
#endif
  return true;
}
