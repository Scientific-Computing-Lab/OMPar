#include <chrono>
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

struct compact_data 
{
  int sizex;
  int sizey;
  int Nmats;
  double *rho_compact;
  double *rho_compact_list;
  double *rho_mat_ave_compact;
  double *rho_mat_ave_compact_list;
  double *p_compact;
  double *p_compact_list;
  double *Vf_compact_list;
  double *t_compact;
  double *t_compact_list;
  double *V;
  double *x;
  double *y;
  double *n;
  double *rho_ave_compact;
  int *imaterial;
  int *matids;
  int *nextfrac;
  int *mmc_index;
  int *mmc_i;
  int *mmc_j;
  int mm_len;
  int mmc_cells;
}
;

void compact_cell_centric(struct full_data cc,struct compact_data ccc,int argc,char **argv)
{
  int sizex = cc . sizex;
  int sizey = cc . sizey;
  int Nmats = cc . Nmats;
  int mmc_cells = ccc . mmc_cells;
  int mm_len = ccc . mm_len;
  int *imaterial = ccc . imaterial;
  int *matids = ccc . matids;
  int *nextfrac = ccc . nextfrac;
  int *mmc_index = ccc . mmc_index;
  int *mmc_i = ccc . mmc_i;
  int *mmc_j = ccc . mmc_j;
  double *x = ccc . x;
  double *y = ccc . y;
  double *rho_compact = ccc . rho_compact;
  double *rho_compact_list = ccc . rho_compact_list;
  double *rho_mat_ave_compact = ccc . rho_mat_ave_compact;
  double *rho_mat_ave_compact_list = ccc . rho_mat_ave_compact_list;
  double *p_compact = ccc . p_compact;
  double *p_compact_list = ccc . p_compact_list;
  double *t_compact = ccc . t_compact;
  double *t_compact_list = ccc . t_compact_list;
  double *V = ccc . V;
  double *Vf_compact_list = ccc . Vf_compact_list;
  double *n = ccc . n;
  double *rho_ave_compact = ccc . rho_ave_compact;
{
    const int thx = 32;
    const int thy = 4;
// Cell-centric algorithms
// Computational loop 1 - average density in cell
    auto t0 = std::chrono::_V2::system_clock::now();
//ccc_loop1 <<< dim3(blocks), dim3(threads) >>> (d_imaterial, d_nextfrac, d_rho_compact, d_rho_compact_list, d_Vf_compact_list, d_V, d_rho_ave_compact, sizex, sizey, d_mmc_index);
    
#pragma omp parallel for private (i)
    for (int j = 0; j <= sizey - 1; j += 1) {
      
#pragma omp parallel for
      for (int i = 0; i <= sizex - 1; i += 1) {
    #ifdef FUSED
// condition is 'ix >= 0', this is the equivalent of
// 'until ix < 0' from the paper
    #ifdef LINKED
    #else
    #endif
    #endif
// We use a distinct output array for averages.
// In case of a pure cell, the average density equals to the total.
        rho_ave_compact[i + sizex * j] = rho_compact[i + sizex * j] / V[i + sizex * j];
    #ifdef FUSED
    #endif
      }
    }
#ifndef FUSED
// ccc_loop1_2 <<< dim3((mmc_cells-1)/(thx*thy)+1), dim3((thx*thy)) >>> (d_rho_compact_list, d_Vf_compact_list, d_V, d_rho_ave_compact, d_mmc_index, mmc_cells, d_mmc_i, d_mmc_j, sizex, sizey);
    for (int c = 0; c <= mmc_cells - 1; c += 1) {
      double ave = 0.0;
      
#pragma omp parallel for reduction (+:ave)
      for (int m = mmc_index[c]; m <= mmc_index[c + 1] - 1; m += 1) {
        ave += rho_compact_list[m] * Vf_compact_list[m];
      }
      rho_ave_compact[mmc_i[c] + sizex * mmc_j[c]] = ave / V[mmc_i[c] + sizex * mmc_j[c]];
    }
#endif
    struct std::chrono::duration< double  , class std::ratio< 1 , 1L >  > t1 = (std::chrono::_V2::system_clock::now()-t0);
    printf("Compact matrix, cell centric, alg 1: %g msec\n",t1 . count() * 1000);
// Computational loop 2 - Pressure for each cell and each material
    t0 = std::chrono::_V2::system_clock::now();
// ccc_loop2 <<< dim3(blocks), dim3(threads) >>> (d_imaterial, d_matids,d_nextfrac, d_rho_compact, d_rho_compact_list, d_t_compact, d_t_compact_list, d_Vf_compact_list, d_n, d_p_compact, d_p_compact_list, sizex, sizey, d_mmc_index);
    
#pragma omp parallel for private (i_nom_2) firstprivate (sizex)
    for (int j = 0; j <= sizey - 1; j += 1) {
      
#pragma omp parallel for
      for (int i = 0; i <= sizex - 1; i += 1) {
        int ix = imaterial[i + sizex * j];
        if (ix <= 0) {
#ifdef FUSED
// NOTE: I think the paper describes this algorithm (Alg. 9) wrong.
// The solution below is what I believe to good.
// condition is 'ix >= 0', this is the equivalent of
// 'until ix < 0' from the paper
#ifdef LINKED
#else
#endif
#endif
        }
         else {
// NOTE: HACK: we index materials from zero, but zero can be a list index
          int mat = ix - 1;
// NOTE: There is no division by Vf here, because the fractional volume is 1.0 in the pure cell case.
          p_compact[i + sizex * j] = n[mat] * rho_compact[i + sizex * j] * t_compact[i + sizex * j];
          ;
        }
      }
    }
#ifndef FUSED
//ccc_loop2_2 <<< dim3((mm_len-1)/(thx*thy)+1), dim3((thx*thy)) >>> (d_matids, d_rho_compact_list, d_t_compact_list, d_Vf_compact_list, d_n, d_p_compact_list, d_mmc_index, mm_len);
    
#pragma omp parallel for firstprivate (mm_len)
    for (int idx = 0; idx <= mm_len - 1; idx += 1) {
      double nm = n[matids[idx]];
      p_compact_list[idx] = nm * rho_compact_list[idx] * t_compact_list[idx] / Vf_compact_list[idx];
    }
#endif
    struct std::chrono::duration< double  , class std::ratio< 1 , 1L >  > t2 = (std::chrono::_V2::system_clock::now()-t0);
    printf("Compact matrix, cell centric, alg 2: %g msec\n",t2 . count() * 1000);
// Computational loop 3 - Average density of each material over neighborhood of each cell
    t0 = std::chrono::_V2::system_clock::now();
//ccc_loop3 <<< dim3(blocks), dim3(threads) >>> (d_imaterial,d_nextfrac, d_matids, d_rho_compact, d_rho_compact_list, d_rho_mat_ave_compact, d_rho_mat_ave_compact_list, d_x, d_y, sizex, sizey, d_mmc_index);  
// if (i >= sizex-1 || j >= sizey-1 || i < 1 || j < 1) return;
    for (int j = 1; j <= sizey - 1 - 1; j += 1) {
      for (int i = 1; i <= sizex - 1 - 1; i += 1) {
        double xo = x[i + sizex * j];
        double yo = y[i + sizex * j];
// There are at most 9 neighbours in 2D case.
        double dsqr[9];
// for all neighbours
        
#pragma omp parallel for private (ni)
        for (int nj = - 1; nj <= 1; nj += 1) {
          
#pragma omp parallel for firstprivate (xo,yo)
          for (int ni = - 1; ni <= 1; ni += 1) {
            dsqr[(nj + 1) * 3 + (ni + 1)] = 0.0;
            double xi = x[i + ni + sizex * (j + nj)];
            double yi = y[i + ni + sizex * (j + nj)];
            dsqr[(nj + 1) * 3 + (ni + 1)] += (xo - xi) * (xo - xi);
            dsqr[(nj + 1) * 3 + (ni + 1)] += (yo - yi) * (yo - yi);
          }
        }
        int ix = imaterial[i + sizex * j];
        if (ix <= 0) {
#ifdef LINKED
#else
          
#pragma omp parallel for private (nj_nom_5,ni_nom_6)
          for (int ix = mmc_index[-imaterial[i_nom_4 + sizex * j_nom_3]]; ix <= mmc_index[-imaterial[i + sizex * j] + 1] - 1; ix += 1) {
#endif
            int mat = matids[ix];
            double rho_sum = 0.0;
            int Nn = 0;
// for all neighbours
            
#pragma omp parallel for private (ni_nom_6)
            for (int nj = - 1; nj <= 1; nj += 1) {
              for (int ni = - 1; ni <= 1; ni += 1) {
                int ci = i + ni;
                int cj = j + nj;
                int jx = imaterial[ci + sizex * cj];
                if (jx <= 0) {
#ifdef LINKED
#else
                  for (int jx = mmc_index[-imaterial[ci + sizex * cj]]; jx <= mmc_index[-imaterial[ci + sizex * cj] + 1] - 1; jx += 1) {
#endif
                    if (matids[jx] == mat) {
                      rho_sum += rho_compact_list[jx] / dsqr[(nj + 1) * 3 + (ni + 1)];
                      Nn += 1;
// The loop has an extra condition: "and not found".
// This makes sense, if the material is found, there won't be any more of the same.
                      break; 
                    }
                  }
                }
                 else {
// NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
// In contrast, Algorithm 10 loads matids[jx] which I think is wrong.
// NOTE: HACK: we index materials from zero, but zero can be a list index
                  int mat_neighbour = jx - 1;
                  if (mat == mat_neighbour) {
                    rho_sum += rho_compact[ci + sizex * cj] / dsqr[(nj + 1) * 3 + (ni + 1)];
                    Nn += 1;
                  }
                }
// end if (jx <= 0)
// end for (int ni)
              }
            }
// end for (int nj)
            rho_mat_ave_compact_list[ix] = rho_sum / Nn;
// end for (ix = -ix)
          }
        }
         else 
// end if (ix <= 0)
{
// NOTE: In this case, the cell is a pure cell, its material index is in ix.
// In contrast, Algorithm 10 loads matids[ix] which I think is wrong.
// NOTE: HACK: we index materials from zero, but zero can be a list index
          int mat = ix - 1;
          double rho_sum = 0.0;
          int Nn = 0;
// for all neighbours
          
#pragma omp parallel for private (ni_nom_8)
          for (int nj = - 1; nj <= 1; nj += 1) {
            if (j + nj < 0 || j + nj >= sizey) 
// TODO: better way?
              continue; 
            for (int ni = - 1; ni <= 1; ni += 1) {
              if (i + ni < 0 || i + ni >= sizex) 
// TODO: better way?
                continue; 
              int ci = i + ni;
              int cj = j + nj;
              int jx = imaterial[ci + sizex * cj];
              if (jx <= 0) {
// condition is 'jx >= 0', this is the equivalent of
// 'until jx < 0' from the paper
#ifdef LINKED
#else
                for (int jx = mmc_index[-imaterial[ci + sizex * cj]]; jx <= mmc_index[-imaterial[ci + sizex * cj] + 1] - 1; jx += 1) {
#endif
                  if (matids[jx] == mat) {
                    rho_sum += rho_compact_list[jx] / dsqr[(nj + 1) * 3 + (ni + 1)];
                    Nn += 1;
// The loop has an extra condition: "and not found".
// This makes sense, if the material is found, there won't be any more of the same.
                    break; 
                  }
                }
              }
               else {
// NOTE: In this case, the neighbour is a pure cell, its material index is in jx.
// In contrast, Algorithm 10 loads matids[jx] which I think is wrong.
// NOTE: HACK: we index materials from zero, but zero can be a list index
                int mat_neighbour = jx - 1;
                if (mat == mat_neighbour) {
                  rho_sum += rho_compact[ci + sizex * cj] / dsqr[(nj + 1) * 3 + (ni + 1)];
                  Nn += 1;
                }
              }
// end if (jx <= 0)
            }
// end for (int ni)
          }
// end for (int nj)
          rho_mat_ave_compact[i + sizex * j] = rho_sum / Nn;
// end else
        }
      }
    }
    struct std::chrono::duration< double  , class std::ratio< 1 , 1L >  > t3 = (std::chrono::_V2::system_clock::now()-t0);
    printf("Compact matrix, cell centric, alg 3: %g msec\n",t3 . count() * 1000);
// omp target region
  }
}

bool compact_check_results(struct full_data cc,struct compact_data ccc)
{
  int sizex = cc . sizex;
  int sizey = cc . sizey;
  int Nmats = cc . Nmats;
  int mmc_cells = ccc . mmc_cells;
  int mm_len = ccc . mm_len;
  printf("Checking results of compact representation... ");
  for (int j = 0; j <= sizey - 1; j += 1) {
    for (int i = 0; i <= sizex - 1; i += 1) {
      if (fabs(cc . rho_ave[i + sizex * j] - ccc . rho_ave_compact[i + sizex * j]) > 0.0001) {
        printf("1. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d)\n",cc . rho_ave[i + sizex * j],ccc . rho_ave_compact[i + sizex * j],i,j);
        return false;
      }
      int ix = ccc . imaterial[i + sizex * j];
      if (ix <= 0) {
#ifdef LINKED
#else
        for (int ix = ccc . mmc_index[-ccc . imaterial[i + sizex * j]]; ix <= ccc . mmc_index[-ccc . imaterial[i + sizex * j] + 1] - 1; ix += 1) {
#endif
          int mat = ccc . matids[ix];
          if (fabs(cc . p[(i + sizex * j) * Nmats + mat] - ccc . p_compact_list[ix]) > 0.0001) {
            printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",cc . p[(i + sizex * j) * Nmats + mat],ccc . p_compact_list[ix],i,j,mat);
            return false;
          }
          if (fabs(cc . rho_mat_ave[(i + sizex * j) * Nmats + mat] - ccc . rho_mat_ave_compact_list[ix]) > 0.0001) {
            printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",cc . rho_mat_ave[(i + sizex * j) * Nmats + mat],ccc . rho_mat_ave_compact_list[ix],i,j,mat);
            return false;
          }
        }
      }
       else {
// NOTE: HACK: we index materials from zero, but zero can be a list index
        int mat = ix - 1;
        if (fabs(cc . p[(i + sizex * j) * Nmats + mat] - ccc . p_compact[i + sizex * j]) > 0.0001) {
          printf("2. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",cc . p[(i + sizex * j) * Nmats + mat],ccc . p_compact[i + sizex * j],i,j,mat);
          return false;
        }
        if (fabs(cc . rho_mat_ave[(i + sizex * j) * Nmats + mat] - ccc . rho_mat_ave_compact[i + sizex * j]) > 0.0001) {
          printf("3. full matrix and compact cell-centric values are not equal! (%f, %f, %d, %d, %d)\n",cc . rho_mat_ave[(i + sizex * j) * Nmats + mat],ccc . rho_mat_ave_compact[i + sizex * j],i,j,mat);
          return false;
        }
      }
    }
  }
  printf("All tests passed!\n");
  return true;
}
