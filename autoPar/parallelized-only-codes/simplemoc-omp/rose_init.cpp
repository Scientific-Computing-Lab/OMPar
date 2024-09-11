#include "SimpleMOC-kernel_header.h"
// Gets I from user and sets defaults
#include <omp.h> 

Input *set_default_input()
{
  Input *I = (Input *)(malloc(sizeof(Input )));
  I -> source_2D_regions = 5000;
  I -> coarse_axial_intervals = 27;
  I -> fine_axial_intervals = 5;
  I -> decomp_assemblies_ax = 20;
// Number of subdomains per assembly axially
#ifdef VERIFY
#else
  I -> segments = 50000000;
#endif
  I -> egroups = 128;
  I -> repeat = 1;
  return I;
}

Source *initialize_sources(Input *I)
{
  I -> nbytes = 0;
// Source Data Structure Allocation
  Source *sources = (Source *)(malloc((I -> source_3D_regions) * sizeof(Source )));
  I -> nbytes += (I -> source_3D_regions) * sizeof(Source );
// Allocate Fine Source Data
  float *data = (float *)(malloc((I -> source_3D_regions * I -> fine_axial_intervals * I -> egroups) * sizeof(float )));
  I -> nbytes += (I -> source_3D_regions * I -> fine_axial_intervals * I -> egroups) * sizeof(float );
  
#pragma omp parallel for
  for (int i = 0; i <= I -> source_3D_regions - 1; i += 1) {
    sources[i] . fine_source = &data[i * I -> fine_axial_intervals * I -> egroups];
  }
// Allocate Fine Flux Data
  data = ((float *)(malloc((I -> source_3D_regions * I -> fine_axial_intervals * I -> egroups) * sizeof(float ))));
  I -> nbytes += (I -> source_3D_regions * I -> fine_axial_intervals * I -> egroups) * sizeof(float );
  
#pragma omp parallel for
  for (int i = 0; i <= I -> source_3D_regions - 1; i += 1) {
    sources[i] . fine_flux = &data[i * I -> fine_axial_intervals * I -> egroups];
  }
// Allocate SigT
  data = ((float *)(malloc((I -> source_3D_regions * I -> egroups) * sizeof(float ))));
  I -> nbytes += (I -> source_3D_regions * I -> egroups) * sizeof(float );
  
#pragma omp parallel for
  for (int i = 0; i <= I -> source_3D_regions - 1; i += 1) {
    sources[i] . sigT = &data[i * I -> egroups];
  }
// Initialize fine source and flux to random numbers
  for (int i = 0; i <= I -> source_3D_regions - 1; i += 1) {
    for (int j = 0; j <= I -> fine_axial_intervals - 1; j += 1) {
      for (int k = 0; k <= I -> egroups - 1; k += 1) {
        sources[i] . fine_source[j * I -> egroups + k] = (rand()) / ((float )2147483647);
        sources[i] . fine_flux[j * I -> egroups + k] = (rand()) / ((float )2147483647);
      }
    }
  }
// Initialize SigT Values
  for (int i = 0; i <= I -> source_3D_regions - 1; i += 1) {
    for (int j = 0; j <= I -> egroups - 1; j += 1) {
      sources[i] . sigT[j] = (rand()) / ((float )2147483647);
    }
  }
  return sources;
}

SIMD_Vectors allocate_simd_vectors(Input *I)
{
  SIMD_Vectors A;
  float *ptr = (float *)(malloc((I -> egroups * 14) * sizeof(float )));
  A . q0 = ptr;
  ptr += I -> egroups;
  A . q1 = ptr;
  ptr += I -> egroups;
  A . q2 = ptr;
  ptr += I -> egroups;
  A . sigT = ptr;
  ptr += I -> egroups;
  A . tau = ptr;
  ptr += I -> egroups;
  A . sigT2 = ptr;
  ptr += I -> egroups;
  A . expVal = ptr;
  ptr += I -> egroups;
  A . reuse = ptr;
  ptr += I -> egroups;
  A . flux_integral = ptr;
  ptr += I -> egroups;
  A . tally = ptr;
  ptr += I -> egroups;
  A . t1 = ptr;
  ptr += I -> egroups;
  A . t2 = ptr;
  ptr += I -> egroups;
  A . t3 = ptr;
  ptr += I -> egroups;
  A . t4 = ptr;
  return A;
}
// Timer function. Depends on if compiled with MPI, openmp, or vanilla

double get_time()
{
#ifdef OPENMP
#endif
  time_t time;
  time = clock();
  return ((double )time) / ((double )((__clock_t )1000000));
}

Source *copy_sources(Input *I,Source *S)
{
  Source *sources = (Source *)(malloc(sizeof(Source )));
  sources -> fine_source = 0L;
  sources -> fine_flux = 0L;
  sources -> sigT = 0L;
// Allocate Fine Source Data
  posix_memalign((void **)(&sources -> fine_source),1024,(I -> source_3D_regions * I -> fine_axial_intervals * I -> egroups) * sizeof(float ));
// Allocate Fine Flux Data
  posix_memalign((void **)(&sources -> fine_flux),1024,(I -> source_3D_regions * I -> fine_axial_intervals * I -> egroups) * sizeof(float ));
// Allocate SigT
  posix_memalign((void **)(&sources -> sigT),1024,(I -> source_3D_regions * I -> egroups) * sizeof(float ));
// Initialize fine source and flux 
  for (int i = 0; i <= I -> source_3D_regions - 1; i += 1) {
    for (int j = 0; j <= I -> fine_axial_intervals - 1; j += 1) {
      for (int k = 0; k <= I -> egroups - 1; k += 1) {
        sources -> fine_source[i * I -> egroups * I -> fine_axial_intervals + j * I -> egroups + k] = S[i] . fine_source[j * I -> egroups + k];
        sources -> fine_flux[i * I -> egroups * I -> fine_axial_intervals + j * I -> egroups + k] = S[i] . fine_flux[j * I -> egroups + k];
      }
    }
  }
// Initialize 1-D SigT Values
  for (int i = 0; i <= I -> source_3D_regions - 1; i += 1) {
    for (int j = 0; j <= I -> egroups - 1; j += 1) {
      sources -> sigT[i * I -> egroups + j] = S[i] . sigT[j];
    }
  }
  return sources;
}
