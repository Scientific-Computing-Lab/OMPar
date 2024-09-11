// https://www.particleincell.com/2016/cuda-pic/
// https://www.particleincell.com/wp-content/uploads/2016/02/sheath-gpu.cu
/* 1D sheath PIC simulation with OpenMP offloading */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <omp.h>
/*constants*/
#define EPS_0 8.85418782e-12 // F/m, vacuum permittivity
/* (previously processed: ignoring self-referential macro declaration) macro name = K */ 
#define ME 9.10938215e-31    // kg, electron mass
#define QE 1.602176565e-19   // C, elementary charge
#define AMU 1.660538921e-27  // kg, atomic mass unit
#define EV_TO_K 11604.52     // 1eV in Kelvin, QE/K
/*simulation parameters, these could come from an input file*/
#define PLASMA_DEN 1e16      // plasma density to load
#define NUM_IONS 500000      // number of ions
#define NUM_ELECTRONS 500000 // number of electrons
#define DX 1e-4              // cell spacing
#define NC 100               // number of cells
#define NUM_TS 1000          // number of time steps
#define DT 1e-11             // time step size
#define ELECTRON_TEMP 3.0    // electron temperature in eV
#define ION_TEMP 1.0         // ion temperature in eV
/*domain parameters, set here so can access from GPU*/
#define X0 0           /*mesh origin*/
#define XL NC* DX      /*domain length*/
#define XMAX (X0 + XL) /*domain max position*/
#include <omp.h> 
const int THREADS_PER_BLOCK = 256;
/* Data structure to hold domain information*/

struct Domain 
{
  const int ni = 100 + 1;
/*number of nodes*/
  const double x0 = 0;
  const double dx = 1e-4;
  const double xl = 100 * 1e-4;
  const double xmax = 0 + 100 * 1e-4;
/*data structures*/
  double *phi;
/*potential*/
  double *ef;
/*electric field on the cpu*/
  double *rho;
/*charge density*/
  float *ndi;
/*ion density on the CPU*/
  float *nde;
/*electron density on the CPU*/
}
;
/* Data structure for particle storage **/

struct Particle 
{
  double x;
/*position*/
  double v;
/*velocity*/
  bool alive;
/*flag to avoid removing particles*/
}
;
/* Data structure to hold species information*/

struct Species 
{
  double mass;
/*particle mass in kg*/
  double charge;
/*particle charge in Coulomb*/
  double spwt;
/*species specific weight*/
  int np;
/*number of particles*/
  int np_alloc;
/*size of the allocated data array*/
  struct Particle *part;
/*array holding particles on the CPU*/
}
;
/** FUNCTION PROTOTYPES **/
double rnd();
double SampleVel(double v_th);
void ScatterSpecies(struct Species *species,struct Particle *particles,float *den,double &time);
void ComputeRho(struct Species *ions,struct Species *electrons);
bool SolvePotential(double *phi,double *rho);
bool SolvePotentialDirect(double *phi,double *rho);
void ComputeEF(double *phi,double *ef);
void PushSpecies(struct Species *species,struct Particle *particles,double *ef);
void RewindSpecies(struct Species *species,struct Particle *particles,double *ef);
void AddParticle(struct Species *species,double x,double v);
double XtoL(double pos);
double gather(double lc,const double *field);
void scatter(double lc,float value,float *field);
void WriteResults(int ts);
/* GLOBAL VARIABLES */
struct Domain domain;
FILE *file_res;
/* --------- main -------------*/

int main(int argc,char *argv[])
{
  int p;
  int ts;
// time step
  double sp_time = 0.0;
// total time of the scatter-particle kernel
  domain . phi = (new double [domain . ni]);
// potential
  domain . rho = (new double [domain . ni]);
// charge density
  domain . ef = (new double [domain . ni]);
// electric field
  domain . nde = (new float [domain . ni]);
// number density of electrons
  domain . ndi = (new float [domain . ni]);
// number density of ions
/*save pointers so we can write phi instead of domain.phi*/
  double *phi = domain . phi;
  double *rho = domain . rho;
  double *ef = domain . ef;
  float *nde = domain . nde;
  float *ndi = domain . ndi;
/*clear data*/
  memset(phi,0,sizeof(double ) * domain . ni);
/*variables to hold species data ions*/
  struct Species ions;
  struct Species electrons;
/*set material data*/
  ions . mass = 16 * 1.660538921e-27;
  ions . charge = 1.602176565e-19;
  ions . spwt = 1e16 * domain . xl / 500000;
  ions . np = 0;
  ions . np_alloc = 500000;
  ions . part = (new Particle [500000]);
  electrons . mass = 9.10938215e-31;
// electrons
  electrons . charge = - 1.602176565e-19;
  electrons . spwt = 1e16 * domain . xl / 500000;
  electrons . np = 0;
  electrons . np_alloc = 500000;
  electrons . part = (new Particle [500000]);
  struct Particle *ions_part = ions . part;
  struct Particle *electrons_part = electrons . part;
/*randomize RNG*/
  srand(123);
/*load uniformly spaced ions */
  double delta_ions = domain . xl / 500000;
  double v_thi = sqrt(2 * 1.38065e-23 * 1.0 * 11604.52 / ions . mass);
  for (p = 0; p <= 499999; p += 1) {
    double x = domain . x0 + p * delta_ions;
    double v = SampleVel(v_thi);
    AddParticle(&ions,x,v);
  }
/*load uniformly spaced electrons */
  double delta_electrons = domain . xl / 500000;
  double v_the = sqrt(2 * 1.38065e-23 * 3.0 * 11604.52 / electrons . mass);
  for (p = 0; p <= 499999; p += 1) {
    double x = domain . x0 + p * delta_electrons;
    double v = SampleVel(v_the);
    AddParticle(&electrons,x,v);
  }
{
/*compute number density*/
    ScatterSpecies(&ions,ions_part,ndi,sp_time);
    ScatterSpecies(&electrons,electrons_part,nde,sp_time);
/*compute charge density and solve potential*/
    ComputeRho(&ions,&electrons);
    SolvePotential(phi,rho);
    ComputeEF(phi,ef);
    RewindSpecies(&ions,ions_part,ef);
    RewindSpecies(&electrons,electrons_part,ef);
/*OUTPUT*/
    file_res = fopen("result.dat","w");
    fprintf(file_res,"VARIABLES = x nde ndi rho phi ef\n");
    WriteResults(0);
    auto start = std::chrono::_V2::steady_clock::now();
/* MAIN LOOP*/
    for (ts = 1; ts <= 1000; ts += 1) {
/*compute number density*/
      ScatterSpecies(&ions,ions_part,ndi,sp_time);
      ScatterSpecies(&electrons,electrons_part,nde,sp_time);
      ComputeRho(&ions,&electrons);
      SolvePotential(phi,rho);
      ComputeEF(phi,ef);
/*move particles*/
      PushSpecies(&electrons,electrons_part,ef);
      PushSpecies(&ions,ions_part,ef);
/*write diagnostics*/
      if (ts % 25 == 0) {
/*max phi*/
        double max_phi = std::abs(phi[0]);
        for (int i = 0; i <= domain . ni - 1; i += 1) {
          if (std::abs(phi[i]) > max_phi) 
            max_phi = std::abs(phi[i]);
        }
        printf("TS:%i\tnp_i:%d\tnp_e:%d\tdphi:%.3g\n",ts,ions . np,electrons . np,max_phi - phi[0]);
      }
/*save data*/
      if (ts % 1000 == 0) 
        WriteResults(ts);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    fclose(file_res);
/*free up memory*/
    delete phi;
    delete rho;
    delete ef;
    delete nde;
    delete ndi;
/*free particles*/
    delete ions . part;
    delete electrons . part;
    (printf("Total kernel execution time (scatter particles) : %.3g (s)\n",sp_time * 1e-9f) , printf("Total time for %d time steps: %.3g (s)\n",1000,(time * 1e-9f)));
    printf("Time per time step: %.3g (ms)\n",(time * 1e-6f / 1000));
/* openmp */
  }
  return 0;
}
/***** HELPER FUNCTIONS *********************************************************/
/* random number generator
   for now using built-in but this is not adequate for real simulations*/

double rnd()
{
  return (rand()) / ((double )2147483647);
}
/* samples random velocity from Maxwellian distribution using Birdsall's method*/

double SampleVel(double v_th)
{
  const int M = 12;
  double sum = 0;
  for (int i = 0; i <= M - 1; i += 1) {
    sum += rnd();
  }
  return sqrt(0.5) * v_th * (sum - M / 2.0) / sqrt(M / 12.0);
}
/*scatter particles of species to the mesh*/

void ScatterSpecies(struct Species *species,struct Particle *particles,float *den,double &time)
{
/*initialize densities to zero*/
  int nodes = domain . ni;
  
#pragma omp parallel for firstprivate (nodes)
  for (int p = 0; p <= nodes - 1; p += 1) {
    den[p] = 0;
  }
  int size = species -> np_alloc;
  auto start = std::chrono::_V2::steady_clock::now();
/*scatter particles to the mesh*/
  for (long p = 0; p <= ((long )size) - 1; p += 1) {
    if (particles[p] . alive) {
      double lc = XtoL(particles[p] . x);
      scatter(lc,1.f,den);
    }
  }
  auto end = std::chrono::_V2::steady_clock::now();
  time += (std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count());
/*copy density back to CPU*/
/*divide by cell volume*/
  for (int i = 0; i <= domain . ni - 1; i += 1) {
    den[i] *= species -> spwt / domain . dx;
  }
/*only half cell at boundaries*/
  den[0] *= 2.0;
  den[domain . ni - 1] *= 2.0;
}
/*adds new particle to the species, returns pointer to the newly added data*/

void AddParticle(struct Species *species,double x,double v)
{
/*abort the simulation if we ran out of space to store this particle*/
  if (species -> np > species -> np_alloc - 1) {
    printf("Too many particles!\n");
    exit(- 1);
  }
/*store position and velocity of this particle*/
  species -> part[species -> np] . x = x;
  species -> part[species -> np] . v = v;
  species -> part[species -> np] . alive = true;
/*increment particle counter*/
  species -> np++;
}
/*computes charge density by adding ion and electron data*/

void ComputeRho(struct Species *ions,struct Species *electrons)
{
  double *rho = domain . rho;
  
#pragma omp parallel for
  for (int i = 0; i <= domain . ni - 1; i += 1) {
    rho[i] = ions -> charge * domain . ndi[i] + electrons -> charge * domain . nde[i];
  }
}
/*Thomas algorithm for a tri-diagonal matrix*/

bool SolvePotentialDirect(double *x,double *rho)
{
/*set coefficients, this should be pre-computed*/
  int ni = domain . ni;
  double dx2 = domain . dx * domain . dx;
  int i;
  double *a = new double [ni];
  double *b = new double [ni];
  double *c = new double [ni];
/*central difference on internal nodes*/
  
#pragma omp parallel for private (i)
  for (i = 1; i <= ni - 1 - 1; i += 1) {
    a[i] = 1;
    b[i] = (- 2);
    c[i] = 1;
  }
/*dirichlet b.c. on boundaries*/
  a[0] = 0;
  b[0] = 1;
  c[0] = 0;
  a[ni - 1] = 0;
  b[ni - 1] = 1;
  c[ni - 1] = 0;
/*multiply RHS*/
  
#pragma omp parallel for private (i) firstprivate (dx2)
  for (i = 1; i <= domain . ni - 1 - 1; i += 1) {
    x[i] = -rho[i] * dx2 / 8.85418782e-12;
  }
  x[0] = 0;
  x[ni - 1] = 0;
/* Modify the coefficients. */
  c[0] /= b[0];
/* Division by zero risk. */
  x[0] /= b[0];
/* Division by zero would imply a singular matrix. */
  for (i = 1; i <= ni - 1; i += 1) {
    double id = b[i] - c[i - 1] * a[i];
/* Division by zero risk. */
    c[i] /= id;
/* Last value calculated is redundant. */
    x[i] = (x[i] - x[i - 1] * a[i]) / id;
  }
/* Now back substitute. */
  for (i = ni - 2; i >= 0; i += -1) {
    x[i] = x[i] - c[i] * x[i + 1];
  }
  return true;
}
/* solves potential using the Gauss Seidel Method, returns true if converged*/

bool SolvePotential(double *phi,double *rho)
{
  double L2;
  double dx2 = domain . dx * domain . dx;
/*precompute*/
/*initialize boundaries*/
  phi[0] = phi[domain . ni - 1] = 0;
/*solve potential, identical to lesson 2*/
  for (int solver_it = 0; solver_it <= 39999; solver_it += 1) {
/*Gauss Seidel method, phi[i-1]-2*phi[i]+phi[i+1] = -dx^2*rho[i]/eps_0*/
    
#pragma omp parallel for firstprivate (dx2)
    for (int i = 1; i <= domain . ni - 1 - 1; i += 1) {
/*SOR*/
      double g = 0.5 * (phi[i - 1] + phi[i + 1] + dx2 * rho[i] / 8.85418782e-12);
      phi[i] = phi[i] + 1.4 * (g - phi[i]);
    }
/*check for convergence*/
    if (solver_it % 25 == 0) {
      double sum = 0;
      
#pragma omp parallel for reduction (+:sum) firstprivate (dx2)
      for (int i = 1; i <= domain . ni - 1 - 1; i += 1) {
        double R = -rho[i] / 8.85418782e-12 - (phi[i - 1] - 2 * phi[i] + phi[i + 1]) / dx2;
        sum += R * R;
      }
      L2 = sqrt(sum) / domain . ni;
      if (L2 < 1e-4) {
        return true;
      }
    }
  }
  printf("Gauss-Seidel solver failed to converge, L2=%.3g!\n",L2);
  return false;
}
/* computes electric field by differentiating potential*/

void ComputeEF(double *phi,double *ef)
{
  for (int i = 1; i <= domain . ni - 1 - 1; i += 1) {
    ef[i] = -(phi[i + 1] - phi[i - 1]) / (2 * domain . dx);
  }
// central difference
/*one sided difference at boundaries*/
  ef[0] = -(phi[1] - phi[0]) / domain . dx;
  ef[domain . ni - 1] = -(phi[domain . ni - 1] - phi[domain . ni - 2]) / domain . dx;
/*copy to the gpu*/
}
/* moves particles of a single species, returns wall charge*/

void PushSpecies(struct Species *species,struct Particle *particles,double *ef)
{
/*precompute q/m*/
  double qm = species -> charge / species -> mass;
  int size = species -> np_alloc;
/*loop over particles*/
  for (long p = 0; p <= ((long )size) - 1; p += 1) {
    if (particles[p] . alive) {
/*grab pointer to this particle*/
      struct Particle *part = &particles[p];
/*compute particle node position*/
      double lc = XtoL(part -> x);
/*gather electric field onto particle position*/
      double part_ef = gather(lc,ef);
/*advance velocity*/
      part -> v += 1e-11 * qm * part_ef;
/*advance position*/
      part -> x += 1e-11 * part -> v;
/*remove particles leaving the domain*/
      if (part -> x < 0 || part -> x >= 0 + 100 * 1e-4) 
        part -> alive = false;
    }
  }
}
/* rewinds particle velocities by -0.5DT*/

void RewindSpecies(struct Species *species,struct Particle *particles,double *ef)
{
/*precompute q/m*/
  double qm = species -> charge / species -> mass;
  int size = species -> np_alloc;
/*loop over particles*/
  
#pragma omp parallel for firstprivate (size)
  for (long p = 0; p <= ((long )size) - 1; p += 1) {
    if (particles[p] . alive) {
/*grab pointer to this particle*/
      struct Particle *part = &particles[p];
/*compute particle node position*/
      double lc = XtoL(part -> x);
/*gather electric field onto particle position*/
      double part_ef = gather(lc,ef);
/*advance velocity*/
      part -> v -= 0.5 * 1e-11 * qm * part_ef;
    }
  }
}
/* converts physical coordinate to logical*/

double XtoL(double pos)
{
  double li = (pos - 0) / 1e-4;
  return li;
}
/* atomic scatter of scalar value onto a field at logical coordinate lc*/

void scatter(double lc,float value,float *field)
{
  int i = (int )lc;
  float di = (lc - i);
  field[i] += value * (1 - di);
  field[i + 1] += value * di;
}
/* gathers field value at logical coordinate lc*/

double gather(double lc,const double *field)
{
  int i = (int )lc;
  double di = lc - i;
/*gather field value onto particle position*/
  double val = field[i] * (1 - di) + field[i + 1] * di;
  return val;
}
/* writes new zone to the results file*/

void WriteResults(int ts)
{
  fprintf(file_res,"ZONE I=%d T=ZONE_%06d\n",domain . ni,ts);
  for (int i = 0; i <= domain . ni - 1; i += 1) {
    fprintf(file_res,"%g %g %g %g %g %g\n",i * domain . dx,domain . nde[i],domain . ndi[i],domain . rho[i],domain . phi[i],domain . ef[i]);
  }
  fflush(file_res);
}
