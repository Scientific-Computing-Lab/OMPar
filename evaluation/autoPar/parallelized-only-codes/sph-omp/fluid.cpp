#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include "sph.h"
////////////////////////////////////////////////////////////////////////////
// B spline smoothing kernel
////////////////////////////////////////////////////////////////////////////
#include <omp.h> 

double W(struct double3 p_pos,struct double3 q_pos,double h)
{
  double r = sqrt((p_pos . x - q_pos . x) * (p_pos . x - q_pos . x) + (p_pos . y - q_pos . y) * (p_pos . y - q_pos . y) + (p_pos . z - q_pos . z) * (p_pos . z - q_pos . z));
  double C = 1.0 / (3.14159265358979323846 * h * h * h);
  double u = r / h;
  double val = 0.0;
  if (u >= 2.0) 
    return val;
   else if (u < 1.0) 
    val = 1.0 - 3.0 / 2.0 * u * u + 3.0 / 4.0 * u * u * u;
   else if (u >= 1.0 && u < 2.0) 
    val = 1.0 / 4.0 * pow(2.0 - u,3.0);
  val *= C;
  return val;
}
// Gradient of B spline kernel

double del_W(struct double3 p_pos,struct double3 q_pos,double h)
{
  double r = sqrt((p_pos . x - q_pos . x) * (p_pos . x - q_pos . x) + (p_pos . y - q_pos . y) * (p_pos . y - q_pos . y) + (p_pos . z - q_pos . z) * (p_pos . z - q_pos . z));
  double C = 1.0 / (3.14159265358979323846 * h * h * h);
  double u = r / h;
  double val = 0.0;
  if (u >= 2.0) 
    return val;
   else if (u < 1.0) 
    val = - 1.0 / (h * h) * (3.0 - 9.0 / 4.0 * u);
   else if (u >= 1.0 && u < 2.0) 
    val = - 3.0 / (4.0 * h * r) * pow(2.0 - u,2.0);
  val *= C;
  return val;
}
////////////////////////////////////////////////////////////////////////////
// Boundary particle force
// http://iopscience.iop.org/0034-4885/68/8/R01/pdf/0034-4885_68_8_R01.pdf
////////////////////////////////////////////////////////////////////////////

double boundaryGamma(struct double3 p_pos,struct double3 k_pos,struct double3 k_n,double h,double speed_sound)
{
// Radial distance between p,q
  double r = sqrt((p_pos . x - k_pos . x) * (p_pos . x - k_pos . x) + (p_pos . y - k_pos . y) * (p_pos . y - k_pos . y) + (p_pos . z - k_pos . z) * (p_pos . z - k_pos . z));
// Distance to p normal to surface particle
  double y = sqrt((p_pos . x - k_pos . x) * (p_pos . x - k_pos . x) * (k_n . x * k_n . x) + (p_pos . y - k_pos . y) * (p_pos . y - k_pos . y) * (k_n . y * k_n . y) + (p_pos . z - k_pos . z) * (p_pos . z - k_pos . z) * (k_n . z * k_n . z));
// Tangential distance
  double x = r - y;
  double u = y / h;
  double xi = (1 - x / h)?(x < h) : 0.0;
  double C = xi * 2.0 * 0.02 * speed_sound * speed_sound / y;
  double val = 0.0;
  if (u > 0.0 && u < 2.0 / 3.0) 
    val = 2.0 / 3.0;
   else if (u < 1.0 && u > 2.0 / 3.0) 
    val = 2 * u - 3.0 / 2.0 * u * u;
   else if (u < 2.0 && u > 1.0) 
    val = 0.5 * (2.0 - u) * (2.0 - u);
   else 
    val = 0.0;
  val *= C;
  return val;
}
////////////////////////////////////////////////////////////////////////////
// Particle attribute computations
////////////////////////////////////////////////////////////////////////////

double computeDensity(struct double3 p_pos,struct double3 p_v,struct double3 q_pos,struct double3 q_v,const struct param *params)
{
  double v_x = p_v . x - q_v . x;
  double v_y = p_v . y - q_v . y;
  double v_z = p_v . z - q_v . z;
  double density = params -> mass_particle * del_W(p_pos,q_pos,params -> smoothing_radius);
  double density_x = density * v_x * (p_pos . x - q_pos . x);
  double density_y = density * v_y * (p_pos . y - q_pos . y);
  double density_z = density * v_z * (p_pos . z - q_pos . z);
  density = (density_x + density_y + density_z) * params -> time_step;
  return density;
}

double computePressure(double p_density,const struct param *params)
{
  double gam = 7.0;
  double B = params -> rest_density * params -> speed_sound * params -> speed_sound / gam;
  double pressure = B * (pow(p_density / params -> rest_density,gam) - 1.0);
  return pressure;
}

struct double3 computeBoundaryAcceleration(struct double3 p_pos,struct double3 k_pos,struct double3 k_n,double h,double speed_sound)
{
  struct double3 p_a;
  double bGamma = boundaryGamma(p_pos,k_pos,k_n,h,speed_sound);
  p_a . x = bGamma * k_n . x;
  p_a . y = bGamma * k_n . y;
  p_a . z = bGamma * k_n . z;
  return p_a;
}

struct double3 computeAcceleration(struct double3 p_pos,struct double3 p_v,double p_density,double p_pressure,struct double3 q_pos,struct double3 q_v,double q_density,double q_pressure,const struct param *params)
{
  struct double3 a;
  double accel;
  double h = params -> smoothing_radius;
  double alpha = params -> alpha;
  double speed_sound = params -> speed_sound;
  double mass_particle = params -> mass_particle;
  double surface_tension = params -> surface_tension;
// Pressure force
  accel = (p_pressure / (p_density * p_density) + q_pressure / (q_density * q_density)) * mass_particle * del_W(p_pos,q_pos,h);
  a . x = -accel * (p_pos . x - q_pos . x);
  a . y = -accel * (p_pos . y - q_pos . y);
  a . z = -accel * (p_pos . z - q_pos . z);
// Viscosity force
  double VdotR = (p_v . x - q_v . x) * (p_pos . x - q_pos . x) + (p_v . y - q_v . y) * (p_pos . y - q_pos . y) + (p_v . z - q_v . z) * (p_pos . z - q_pos . z);
  if (VdotR < 0.0) {
    double nu = 2.0 * alpha * h * speed_sound / (p_density + q_density);
    double r2 = (p_pos . x - q_pos . x) * (p_pos . x - q_pos . x) + (p_pos . y - q_pos . y) * (p_pos . y - q_pos . y) + (p_pos . z - q_pos . z) * (p_pos . z - q_pos . z);
    double eps = h / 10.0;
    double stress = nu * VdotR / (r2 + eps * h * h);
    accel = mass_particle * stress * del_W(p_pos,q_pos,h);
    a . x += accel * (p_pos . x - q_pos . x);
    a . y += accel * (p_pos . y - q_pos . y);
    a . z += accel * (p_pos . z - q_pos . z);
  }
//Surface tension
// BT 07 http://cg.informatik.uni-freiburg.de/publications/2011_GRAPP_airBubbles.pdf
  accel = surface_tension * W(p_pos,q_pos,h);
  a . x += accel * (p_pos . x - q_pos . x);
  a . y += accel * (p_pos . y - q_pos . y);
  a . z += accel * (p_pos . z - q_pos . z);
  return a;
}
// Seed simulation with Euler step v(t-dt/2) needed by leap frog integrator
// Should calculate all accelerations but assuming just g simplifies acc port

void eulerStart(struct fluid_particle *fluid_particles,struct boundary_particle *boundary_particles,struct param *params)
{
// Set V (t0 - dt/2)
  double dt_half = params -> time_step / 2.0;
  for (int i = 0; i <= params -> number_fluid_particles - 1; i += 1) {
// Velocity at t + dt/2
    struct double3 v = fluid_particles[i] . v;
    struct double3 v_half;
    v_half . x = v . x;
    v_half . y = v . y;
    v_half . z = v . z - params -> g * dt_half;
    fluid_particles[i] . v_half = v_half;
  }
}
// Initialize particles

void initParticles(struct fluid_particle **fluid_particles,struct boundary_particle **boundary_particles,struct AABB *water,struct AABB *boundary,struct param *params)
{
// Allocate fluid particles array
   *fluid_particles = ((struct fluid_particle *)(malloc((params -> number_fluid_particles) * sizeof(struct fluid_particle ))));
// Allocate boundary particles array
   *boundary_particles = ((struct boundary_particle *)(malloc((params -> number_boundary_particles) * sizeof(struct boundary_particle ))));
  double spacing = params -> spacing_particle;
// Initialize particle values
  for (int i = 0; i <= params -> number_fluid_particles - 1; i += 1) {
    ( *fluid_particles)[i] . a . x = 0.0;
    ( *fluid_particles)[i] . a . y = 0.0;
    ( *fluid_particles)[i] . a . z = 0.0;
    ( *fluid_particles)[i] . v . x = 0.0;
    ( *fluid_particles)[i] . v . y = 0.0;
    ( *fluid_particles)[i] . v . z = 0.0;
    ( *fluid_particles)[i] . density = params -> rest_density;
  }
// Place particles inside bounding volume
  double x;
  double y;
  double z;
  int i = 0;
  for (z = water -> min_z; z <= water -> max_z; z += spacing) {
    for (y = water -> min_y; y <= water -> max_y; y += spacing) {
      for (x = water -> min_x; x <= water -> max_x; x += spacing) {
        if (i < params -> number_fluid_particles) {
          ( *fluid_particles)[i] . pos . x = x;
          ( *fluid_particles)[i] . pos . y = y;
          ( *fluid_particles)[i] . pos . z = z;
          i++;
        }
      }
    }
  }
  params -> number_fluid_particles = i;
// Construct bounding box
  constructBoundaryBox( *boundary_particles,boundary,params);
}

void initParams(struct AABB *water_volume,struct AABB *boundary_volume,struct param *params)
{
// Boundary box
  boundary_volume -> min_x = 0.0;
  boundary_volume -> max_x = 1.1;
  boundary_volume -> min_y = 0.0;
  boundary_volume -> max_y = 1.1;
  boundary_volume -> min_z = 0.0;
  boundary_volume -> max_z = 1.1;
// water volume
  water_volume -> min_x = 0.1;
  water_volume -> max_x = 0.5;
  water_volume -> min_y = 0.1;
  water_volume -> max_y = 0.5;
  water_volume -> min_z = 0.08;
  water_volume -> max_z = 0.8;
// Simulation parameters
  params -> number_fluid_particles = 2048;
  params -> rest_density = 1000.0;
  params -> g = 9.8;
  params -> alpha = 0.02;
  params -> surface_tension = 0.01;
  params -> number_steps = 500;
// reduce from 5000
  params -> time_step = 0.00035;
// Mass of each particle
  double volume = (water_volume -> max_x - water_volume -> min_x) * (water_volume -> max_y - water_volume -> min_y) * (water_volume -> max_z - water_volume -> min_z);
  params -> mass_particle = params -> rest_density * (volume / (params -> number_fluid_particles));
// Cube calculated spacing
  params -> spacing_particle = pow(volume / (params -> number_fluid_particles),1.0 / 3.0);
// Smoothing radius, h
  params -> smoothing_radius = params -> spacing_particle;
// Boundary particles
  int num_x = (ceil((boundary_volume -> max_x - boundary_volume -> min_x) / params -> spacing_particle));
  int num_y = (ceil((boundary_volume -> max_y - boundary_volume -> min_y) / params -> spacing_particle));
  int num_z = (ceil((boundary_volume -> max_z - boundary_volume -> min_z) / params -> spacing_particle));
  int num_boundary_particles = 2 * num_x * num_z + 2 * num_y * num_z + 2 * num_y * num_z;
  params -> number_boundary_particles = num_boundary_particles;
// Total number of particles
  params -> number_particles = params -> number_boundary_particles + params -> number_fluid_particles;
// Number of steps before frame needs to be written for 30 fps
  params -> steps_per_frame = ((int )(1.0 / (params -> time_step * 30.0)));
// Calculate speed of sound for simulation
  double max_height = water_volume -> max_y;
  double max_velocity = sqrt(2.0 * params -> g * max_height);
  params -> speed_sound = max_velocity / sqrt(0.01);
// Minimum stepsize from Courant-Friedrichs-Lewy condition
  double recomend_step = 0.4 * params -> smoothing_radius / (params -> speed_sound * (1 + 0.6 * params -> alpha));
  printf("Using time step: %f, Minimum recomended %f\n",params -> time_step,recomend_step);
}

void finalizeParticles(struct fluid_particle *fluid_particles,struct boundary_particle *boundary_particles)
{
  free(fluid_particles);
  free(boundary_particles);
}

int main(int argc,char *argv[])
{
  struct param params;
  struct AABB water_volume;
  struct AABB boundary_volume;
  struct fluid_particle *fluid_particles = 0L;
  struct boundary_particle *boundary_particles = 0L;
  initParams(&water_volume,&boundary_volume,&params);
  initParticles(&fluid_particles,&boundary_particles,&water_volume,&boundary_volume,&params);
  eulerStart(fluid_particles,boundary_particles,&params);
  int num_fluid_particles = params . number_fluid_particles;
  int num_boundary_particles = params . number_boundary_particles;
  int num_particles = num_fluid_particles < num_boundary_particles?num_fluid_particles : num_boundary_particles;
{
    auto start = std::chrono::_V2::steady_clock::now();
// Main simulation loop
    for (int n = 0; n <= params . number_steps - 1; n += 1) {
//updatePressures <<< dim3(grid1D_FP), dim3(block1D) >>> (d_fluid_particles, d_params);
      for (int i = 0; i <= num_fluid_particles - 1; i += 1) {
        struct double3 p_pos = fluid_particles[i] . pos;
        struct double3 p_v = fluid_particles[i] . v;
        double density = fluid_particles[i] . density;
        for (int j = 0; j <= num_fluid_particles - 1; j += 1) {
          struct double3 q_pos = fluid_particles[j] . pos;
          struct double3 q_v = fluid_particles[j] . v;
          density += computeDensity(p_pos,p_v,q_pos,q_v,(&params));
        }
        fluid_particles[i] . density = density;
        fluid_particles[i] . pressure = computePressure(density,(&params));
      }
//updateAccelerationsFP <<< dim3(grid1D_FP), dim3(block1D) >>> (d_fluid_particles, d_params);
      for (int i = 0; i <= num_fluid_particles - 1; i += 1) {
        double ax = 0.0;
        double ay = 0.0;
        double az = - 9.8;
        struct double3 p_pos = fluid_particles[i] . pos;
        struct double3 p_v = fluid_particles[i] . v;
        double p_density = fluid_particles[i] . density;
        double p_pressure = fluid_particles[i] . pressure;
        
#pragma omp parallel for reduction (+:ax,ay,az)
        for (int j = 0; j <= num_fluid_particles - 1; j += 1) {
          if (i != j) {
            struct double3 q_pos = fluid_particles[j] . pos;
            struct double3 q_v = fluid_particles[j] . v;
            double q_density = fluid_particles[j] . density;
            double q_pressure = fluid_particles[j] . pressure;
            struct double3 tmp_a = computeAcceleration(p_pos,p_v,p_density,p_pressure,q_pos,q_v,q_density,q_pressure,(&params));
            ax += tmp_a . x;
            ay += tmp_a . y;
            az += tmp_a . z;
          }
        }
        fluid_particles[i] . a . x = ax;
        fluid_particles[i] . a . y = ay;
        fluid_particles[i] . a . z = az;
      }
//updateAccelerationsBP ()<<< dim3(grid1D_BP), dim3(block1D) >>> (d_fluid_particles, d_boundary_particles, d_params);
      for (int i = 0; i <= num_particles - 1; i += 1) {
        double ax = fluid_particles[i] . a . x;
        double ay = fluid_particles[i] . a . y;
        double az = fluid_particles[i] . a . z;
        struct double3 p_pos = fluid_particles[i] . pos;
        
#pragma omp parallel for reduction (+:ax,ay,az)
        for (int j = 0; j <= num_boundary_particles - 1; j += 1) {
          struct double3 k_pos = boundary_particles[j] . pos;
          struct double3 k_n = boundary_particles[j] . n;
          struct double3 tmp_a = computeBoundaryAcceleration(p_pos,k_pos,k_n,params . smoothing_radius,params . speed_sound);
          ax += tmp_a . x;
          ay += tmp_a . y;
          az += tmp_a . z;
        }
        fluid_particles[i] . a . x = ax;
        fluid_particles[i] . a . y = ay;
        fluid_particles[i] . a . z = az;
      }
//updatePositions <<< dim3(grid1D_FP), dim3(block1D) >>> (d_fluid_particles, d_params);
      for (int i = 0; i <= num_fluid_particles - 1; i += 1) {
        double dt = params . time_step;
// Velocity at t + dt/2
        struct double3 v_half = fluid_particles[i] . v_half;
        struct double3 v = fluid_particles[i] . v;
        struct double3 pos = fluid_particles[i] . pos;
        struct double3 a = fluid_particles[i] . a;
        v_half . x = v_half . x + dt * a . x;
        v_half . y = v_half . y + dt * a . y;
        v_half . z = v_half . z + dt * a . z;
// Velocity at t + dt, must estimate for foce calc
        v . x = v_half . x + a . x * (dt / 2.0);
        v . y = v_half . y + a . y * (dt / 2.0);
        v . z = v_half . z + a . z * (dt / 2.0);
// Position at time t + dt
        pos . x = pos . x + dt * v_half . x;
        pos . y = pos . y + dt * v_half . y;
        pos . z = pos . z + dt * v_half . z;
        fluid_particles[i] . v_half = v_half;
        fluid_particles[i] . v = v;
        fluid_particles[i] . pos = pos;
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average execution time of sph kernels: %f (ms)\n",(time * 1e-6f / params . number_steps));
  }
  writeFile(fluid_particles,&params);
  finalizeParticles(fluid_particles,boundary_particles);
  return 0;
}
