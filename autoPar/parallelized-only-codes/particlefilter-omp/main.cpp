#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#define BLOCK_X 16
#define BLOCK_Y 16
#define PI 3.1415926535897932f
#define A 1103515245
#define C 12345
#define M INT_MAX
#define SCALE_FACTOR 300.0f
#ifndef BLOCK_SIZE 
#define BLOCK_SIZE 256
#endif
/**
  @var M value for Linear Congruential Generator (LCG); use GCC's value
  */
//long M = INT_MAX;
/**
  @var A value for LCG
  */
//int A = 1103515245;
/**
  @var C value for LCG
  */
//int C = 12345;
#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif
// returns a long int representing the time
#include <omp.h> 

long long get_time()
{
  struct timeval tv;
  gettimeofday(&tv,0L);
  return (tv . tv_sec * 1000000 + tv . tv_usec);
}
// Returns the number of seconds elapsed between the two specified times

float elapsed_time(long long start_time,long long end_time)
{
  return ((float )(end_time - start_time)) / (1000 * 1000);
}
/**
 * Generates a uniformly distributed random number using the provided seed and GCC's settings for the Linear Congruential Generator (LCG)
 * @see http://en.wikipedia.org/wiki/Linear_congruential_generator
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a uniformly distributed number [0, 1)
 */

float randu(int *seed,int index)
{
  int num = 1103515245 * seed[index] + 12345;
  seed[index] = num % 0x7fffffff;
  return std::fabs(seed[index] / ((float )0x7fffffff));
}
/**
 * Generates a normally distributed random number using the Box-Muller transformation
 * @note This function is thread-safe
 * @param seed The seed array
 * @param index The specific index of the seed to be advanced
 * @return a float representing random number generated using the Box-Muller algorithm
 * @see http://en.wikipedia.org/wiki/Normal_distribution, section computing value for normal random distribution
 */

float randn(int *seed,int index)
{
/*Box-Muller algorithm*/
  float u = randu(seed,index);
  float v = randu(seed,index);
  float cosine = std::cos(2 * 3.1415926535897932f * v);
  float rt = (- 2) * std::log(u);
  return std::sqrt(rt) * cosine;
}
/**
 * Takes in a float and returns an integer that approximates to that float
 * @return if the mantissa < .5 => return value < input value; else return value > input value
 */

float roundFloat(float value)
{
  int newValue = (int )value;
  if ((value - newValue) < .5) 
    return newValue;
   else 
    return (newValue++);
}
/**
 * Set values of the 3D array to a newValue if that value is equal to the testValue
 * @param testValue The value to be replaced
 * @param newValue The value to replace testValue with
 * @param array3D The image vector
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 */

void setIf(int testValue,int newValue,unsigned char *array3D,int *dimX,int *dimY,int *dimZ)
{
  int x;
  int y;
  int z;
  for (x = 0; x <=  *dimX - 1; x += 1) {
    for (y = 0; y <=  *dimY - 1; y += 1) {
      for (z = 0; z <=  *dimZ - 1; z += 1) {
        if (array3D[x *  *dimY *  *dimZ + y *  *dimZ + z] == testValue) 
          array3D[x *  *dimY *  *dimZ + y *  *dimZ + z] = newValue;
      }
    }
  }
}
/**
 * Sets values of 3D matrix using randomly generated numbers from a normal distribution
 * @param array3D The video to be modified
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param seed The seed array
 */

void addNoise(unsigned char *array3D,int *dimX,int *dimY,int *dimZ,int *seed)
{
  int x;
  int y;
  int z;
  for (x = 0; x <=  *dimX - 1; x += 1) {
    for (y = 0; y <=  *dimY - 1; y += 1) {
      for (z = 0; z <=  *dimZ - 1; z += 1) {
        array3D[x *  *dimY *  *dimZ + y *  *dimZ + z] = (array3D[x *  *dimY *  *dimZ + y *  *dimZ + z] + ((unsigned char )(5 * randn(seed,0))));
      }
    }
  }
}
/**
 * Fills a radius x radius matrix representing the disk
 * @param disk The pointer to the disk to be made
 * @param radius  The radius of the disk to be made
 */

void strelDisk(int *disk,int radius)
{
  int diameter = radius * 2 - 1;
  int x;
  int y;
  for (x = 0; x <= diameter - 1; x += 1) {
    for (y = 0; y <= diameter - 1; y += 1) {
      float distance = (sqrt(std::pow((float )(x - radius + 1),2) + std::pow((float )(y - radius + 1),2)));
      if (distance < radius) 
        disk[x * diameter + y] = 1;
       else 
        disk[x * diameter + y] = 0;
    }
  }
}
/**
 * Dilates the provided video
 * @param matrix The video to be dilated
 * @param posX The x location of the pixel to be dilated
 * @param posY The y location of the pixel to be dilated
 * @param poxZ The z location of the pixel to be dilated
 * @param dimX The x dimension of the frame
 * @param dimY The y dimension of the frame
 * @param dimZ The number of frames
 * @param error The error radius
 */

void dilate_matrix(unsigned char *matrix,int posX,int posY,int posZ,int dimX,int dimY,int dimZ,int error)
{
  int startX = posX - error;
  while(startX < 0)
    startX++;
  int startY = posY - error;
  while(startY < 0)
    startY++;
  int endX = posX + error;
  while(endX > dimX)
    endX--;
  int endY = posY + error;
  while(endY > dimY)
    endY--;
  int x;
  int y;
  for (x = startX; x <= endX - 1; x += 1) {
    for (y = startY; y <= endY - 1; y += 1) {
      float distance = (sqrt(std::pow((float )(x - posX),2) + std::pow((float )(y - posY),2)));
      if (distance < error) 
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
  }
}
/**
 * Dilates the target matrix using the radius as a guide
 * @param matrix The reference matrix
 * @param dimX The x dimension of the video
 * @param dimY The y dimension of the video
 * @param dimZ The z dimension of the video
 * @param error The error radius to be dilated
 * @param newMatrix The target matrix
 */

void imdilate_disk(unsigned char *matrix,int dimX,int dimY,int dimZ,int error,unsigned char *newMatrix)
{
  int x;
  int y;
  int z;
  for (z = 0; z <= dimZ - 1; z += 1) {
    for (x = 0; x <= dimX - 1; x += 1) {
      for (y = 0; y <= dimY - 1; y += 1) {
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1) {
          dilate_matrix(newMatrix,x,y,z,dimX,dimY,dimZ,error);
        }
      }
    }
  }
}
/**
 * Fills a 2D array describing the offsets of the disk object
 * @param se The disk object
 * @param numOnes The number of ones in the disk
 * @param neighbors The array that will contain the offsets
 * @param radius The radius used for dilation
 */

void getneighbors(int *se,int numOnes,int *neighbors,int radius)
{
  int x;
  int y;
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (x = 0; x <= diameter - 1; x += 1) {
    for (y = 0; y <= diameter - 1; y += 1) {
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = ((int )(y - center));
        neighbors[neighY * 2 + 1] = ((int )(x - center));
        neighY++;
      }
    }
  }
}
/**
 * The synthetic video sequence we will work with here is composed of a
 * single moving object, circular in shape (fixed radius)
 * The motion here is a linear motion
 * the foreground intensity and the backgrounf intensity is known
 * the image is corrupted with zero mean Gaussian noise
 * @param I The video itself
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames of the video
 * @param seed The seed array used for number generation
 */

void videoSequence(unsigned char *I,int IszX,int IszY,int Nfr,int *seed)
{
  int k;
  int max_size = IszX * IszY * Nfr;
/*get object centers*/
  int x0 = (int )(roundFloat((IszY / 2.0)));
  int y0 = (int )(roundFloat((IszX / 2.0)));
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;
/*move point*/
  int xk;
  int yk;
  int pos;
  for (k = 1; k <= Nfr - 1; k += 1) {
    xk = abs(x0 + (k - 1));
    yk = abs(y0 - 2 * (k - 1));
    pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size) 
      pos = 0;
    I[pos] = 1;
  }
/*dilate matrix*/
  unsigned char *newMatrix = (unsigned char *)(calloc((IszX * IszY * Nfr),sizeof(unsigned char )));
  imdilate_disk(I,IszX,IszY,Nfr,5,newMatrix);
  int x;
  int y;
  for (x = 0; x <= IszX - 1; x += 1) {
    
#pragma omp parallel for private (k,y)
    for (y = 0; y <= IszY - 1; y += 1) {
      
#pragma omp parallel for private (k)
      for (k = 0; k <= Nfr - 1; k += 1) {
        I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
      }
    }
  }
  free(newMatrix);
/*define background, add noise*/
  setIf(0,100,I,&IszX,&IszY,&Nfr);
  setIf(1,228,I,&IszX,&IszY,&Nfr);
/*add noise*/
  addNoise(I,&IszX,&IszY,&Nfr,seed);
}
/**
 * Finds the first element in the CDF that is greater than or equal to the provided value and returns that index
 * @note This function uses sequential search
 * @param CDF The CDF
 * @param lengthCDF The length of CDF
 * @param value The value to be found
 * @return The index of value in the CDF; if value is never found, returns the last index
 */

int findIndex(float *CDF,int lengthCDF,float value)
{
  int index = - 1;
  int x;
  for (x = 0; x <= lengthCDF - 1; x += 1) {
    if (CDF[x] >= value) {
      index = x;
      break; 
    }
  }
  if (index == - 1) {
    return lengthCDF - 1;
  }
  return index;
}
/**
 * The implementation of the particle filter using OpenMP for many frames
 * @see http://openmp.org/wp/
 * @note This function is designed to work with a video of several frames. In addition, it references a provided MATLAB function which takes the video, the objxy matrix and the x and y arrays as arguments and returns the likelihoods
 * @param I The video to be run
 * @param IszX The x dimension of the video
 * @param IszY The y dimension of the video
 * @param Nfr The number of frames
 * @param seed The seed array used for random number generation
 * @param Nparticles The number of particles to be used
 */

int particleFilter(unsigned char *I,int IszX,int IszY,int Nfr,int *seed,int Nparticles)
{
  int max_size = IszX * IszY * Nfr;
//original particle centroid
  float xe = roundFloat((IszY / 2.0));
  float ye = roundFloat((IszX / 2.0));
//expected object locations, compared to center
  int radius = 5;
  int diameter = radius * 2 - 1;
  int *disk = (int *)(calloc((diameter * diameter),sizeof(int )));
  strelDisk(disk,radius);
  int countOnes = 0;
  int x;
  int y;
  
#pragma omp parallel for private (x,y) reduction (+:countOnes) firstprivate (diameter)
  for (x = 0; x <= diameter - 1; x += 1) {
    
#pragma omp parallel for private (y) reduction (+:countOnes)
    for (y = 0; y <= diameter - 1; y += 1) {
      if (disk[x * diameter + y] == 1) 
        countOnes++;
    }
  }
  int *objxy = (int *)(calloc((countOnes * 2),sizeof(int )));
  getneighbors(disk,countOnes,objxy,radius);
//initial weights are all equal (1/Nparticles)
  float *weights = (float *)(calloc(Nparticles,sizeof(float )));
  
#pragma omp parallel for private (x)
  for (x = 0; x <= Nparticles - 1; x += 1) {
    weights[x] = 1 / ((float )Nparticles);
  }
/****************************************************************
   **************   B E G I N   A L L O C A T E *******************
   ****************************************************************/
  float *likelihood = (float *)(calloc((Nparticles + 1),sizeof(float )));
  float *partial_sums = (float *)(calloc((Nparticles + 1),sizeof(float )));
  float *arrayX = (float *)(calloc(Nparticles,sizeof(float )));
  float *arrayY = (float *)(calloc(Nparticles,sizeof(float )));
  float *xj = (float *)(calloc(Nparticles,sizeof(float )));
  float *yj = (float *)(calloc(Nparticles,sizeof(float )));
  float *CDF = (float *)(calloc(Nparticles,sizeof(float )));
//GPU copies of arrays
  int *ind = (int *)(calloc((countOnes * Nparticles),sizeof(int )));
  float *u = (float *)(calloc(Nparticles,sizeof(float )));
//Donnie - this loop is different because in this kernel, arrayX and arrayY
//  are set equal to xj before every iteration, so effectively, arrayX and
//  arrayY will be set to xe and ye before the first iteration.
  
#pragma omp parallel for private (x) firstprivate (xe,ye)
  for (x = 0; x <= Nparticles - 1; x += 1) {
    xj[x] = xe;
    yj[x] = ye;
  }
  long long offload_start = get_time();
  int k;
  int num_blocks = (Nparticles + 256 - 1) / 256;
#ifdef DEBUG
#endif
{
    long long start = get_time();
    for (k = 1; k <= Nfr - 1; k += 1) {
/****************** L I K E L I H O O D ************************************/
{
        float weights_local[256];
{
          int block_id = omp_get_team_num();
          int thread_id = omp_get_thread_num();
          int block_dim = omp_get_num_threads();
          int i = block_id * block_dim + thread_id;
          int y;
          int indX;
          int indY;
          float u;
          float v;
          if (i < Nparticles) {
            arrayX[i] = xj[i];
            arrayY[i] = yj[i];
            weights[i] = 1.0f / ((float )Nparticles);
            seed[i] = (1103515245 * seed[i] + 12345) % 0x7fffffff;
            u = fabsf(seed[i] / ((float )0x7fffffff));
            seed[i] = (1103515245 * seed[i] + 12345) % 0x7fffffff;
            v = fabsf(seed[i] / ((float )0x7fffffff));
            arrayX[i] += 1.0f + 5.0f * (sqrtf(- 2.0f * logf(u)) * cosf(2.0f * 3.1415926535897932f * v));
            seed[i] = (1103515245 * seed[i] + 12345) % 0x7fffffff;
            u = fabsf(seed[i] / ((float )0x7fffffff));
            seed[i] = (1103515245 * seed[i] + 12345) % 0x7fffffff;
            v = fabsf(seed[i] / ((float )0x7fffffff));
            arrayY[i] += - 2.0f + 2.0f * (sqrtf(- 2.0f * logf(u)) * cosf(2.0f * 3.1415926535897932f * v));
          }
          if (i < Nparticles) {
            for (y = 0; y <= countOnes - 1; y += 1) {
              int iX = arrayX[i];
              int iY = arrayY[i];
              int rnd_iX = arrayX[i] - iX < .5f?iX : iX++;
              int rnd_iY = arrayY[i] - iY < .5f?iY : iY++;
              indX = rnd_iX + objxy[y * 2 + 1];
              indY = rnd_iY + objxy[y * 2];
              ind[i * countOnes + y] = abs(indX * IszY * Nfr + indY * Nfr + k);
              if (ind[i * countOnes + y] >= max_size) 
                ind[i * countOnes + y] = 0;
            }
            float likelihoodSum = 0.0f;
            
#pragma omp parallel for reduction (+:likelihoodSum)
            for (int x = 0; x <= countOnes - 1; x += 1) {
              likelihoodSum += ((I[ind[i * countOnes + x]] - 100) * (I[ind[i * countOnes + x]] - 100) - (I[ind[i * countOnes + x]] - 228) * (I[ind[i * countOnes + x]] - 228)) / 50.0f;
            }
            likelihood[i] = likelihoodSum / countOnes - 300.0f;
            weights[i] = weights[i] * expf(likelihood[i]);
          }
          weights_local[thread_id] = (i < Nparticles?weights[i] : 0.f);
          for (unsigned int s = (block_dim / 2); s >= ((unsigned int )0) + 1; s >>= 1) {
            if (thread_id < s) {
              weights_local[thread_id] += weights_local[thread_id + s];
            }
          }
          if (thread_id == 0) {
            partial_sums[block_id] = weights_local[0];
          }
        }
      }
{
        float sum = 0;
        int num_blocks = (Nparticles + 256 - 1) / 256;
        
#pragma omp parallel for reduction (+:sum) firstprivate (num_blocks)
        for (int x = 0; x <= num_blocks - 1; x += 1) {
          sum += partial_sums[x];
        }
        partial_sums[0] = sum;
      }
#ifdef DEBUG
// this shows the sum of all partial_sum results
#endif
{
        float u1;
        float sumWeights;
{
          int local_id = omp_get_thread_num();
          int i = omp_get_team_num() * omp_get_num_threads() + local_id;
          if (0 == local_id) 
            sumWeights = partial_sums[0];
          if (i < Nparticles) {
            weights[i] = weights[i] / sumWeights;
          }
          if (i == 0) {
            CDF[0] = weights[0];
            for (int x = 1; x <= Nparticles - 1; x += 1) {
              CDF[x] = weights[x] + CDF[x - 1];
            }
            seed[i] = (1103515245 * seed[i] + 12345) % 0x7fffffff;
            float p = fabsf(seed[i] / ((float )0x7fffffff));
            seed[i] = (1103515245 * seed[i] + 12345) % 0x7fffffff;
            float q = fabsf(seed[i] / ((float )0x7fffffff));
            u[0] = 1.0f / ((float )Nparticles) * (sqrtf(- 2.0f * logf(p)) * cosf(2.0f * 3.1415926535897932f * q));
// do this to allow all threads in all blocks to use the same u1
          }
          if (0 == local_id) 
            u1 = u[0];
          if (i < Nparticles) {
            u[i] = u1 + i / ((float )Nparticles);
          }
        }
      }
#ifdef DEBUG
// estimate the object location by expected values
#endif
      
#pragma omp parallel for
      for (int i = 0; i <= Nparticles - 1; i += 1) {
        int index = - 1;
        int x;
        for (x = 0; x <= Nparticles - 1; x += 1) {
          if (CDF[x] >= u[i]) {
            index = x;
            break; 
          }
        }
        if (index == - 1) {
          index = Nparticles - 1;
        }
        xj[i] = arrayX[index];
        yj[i] = arrayY[index];
      }
//end loop
    }
    long long end = get_time();
    printf("Average execution time of kernels: %f (s)\n",(elapsed_time(start,end) / (Nfr - 1)));
// #pragma 
  }
  long long offload_end = get_time();
  printf("Device offloading time: %lf (s)\n",(elapsed_time(offload_start,offload_end)));
  xe = 0;
  ye = 0;
// estimate the object location by expected values
  
#pragma omp parallel for private (x) reduction (+:xe,ye) firstprivate (Nparticles)
  for (x = 0; x <= Nparticles - 1; x += 1) {
    xe += arrayX[x] * weights[x];
    ye += arrayY[x] * weights[x];
  }
  float distance = (sqrt(std::pow((float )(xe - ((int )(roundFloat((IszY / 2.0))))),2) + std::pow((float )(ye - ((int )(roundFloat((IszX / 2.0))))),2)));
//Output results
  FILE *fid;
  fid = fopen("output.txt","w+");
  if (fid == 0L) {
    printf("The file was not opened for writing\n");
    return - 1;
  }
  fprintf(fid,"XE: %lf\n",xe);
  fprintf(fid,"YE: %lf\n",ye);
  fprintf(fid,"distance: %lf\n",distance);
  fclose(fid);
//free regular memory
  free(likelihood);
  free(partial_sums);
  free(arrayX);
  free(arrayY);
  free(xj);
  free(yj);
  free(CDF);
  free(ind);
  free(u);
  return 0;
}

int main(int argc,char *argv[])
{
  const char *usage = "./main -x <dimX> -y <dimY> -z <Nfr> -np <Nparticles>";
//check number of arguments
  if (argc != 9) {
    printf("%s\n",usage);
    return 0;
  }
//check args deliminators
  if ((strcmp(argv[1],"-x")) || (strcmp(argv[3],"-y")) || (strcmp(argv[5],"-z")) || (strcmp(argv[7],"-np"))) {
    printf("%s\n",usage);
    return 0;
  }
  int IszX;
  int IszY;
  int Nfr;
  int Nparticles;
//converting a string to a integer
  if (sscanf(argv[2],"%d",&IszX) == - 1) {
    printf("ERROR: dimX input is incorrect");
    return 0;
  }
  if (IszX <= 0) {
    printf("dimX must be > 0\n");
    return 0;
  }
//converting a string to a integer
  if (sscanf(argv[4],"%d",&IszY) == - 1) {
    printf("ERROR: dimY input is incorrect");
    return 0;
  }
  if (IszY <= 0) {
    printf("dimY must be > 0\n");
    return 0;
  }
//converting a string to a integer
  if (sscanf(argv[6],"%d",&Nfr) == - 1) {
    printf("ERROR: Number of frames input is incorrect");
    return 0;
  }
  if (Nfr <= 0) {
    printf("number of frames must be > 0\n");
    return 0;
  }
//converting a string to a integer
  if (sscanf(argv[8],"%d",&Nparticles) == - 1) {
    printf("ERROR: Number of particles input is incorrect");
    return 0;
  }
  if (Nparticles <= 0) {
    printf("Number of particles must be > 0\n");
    return 0;
  }
#ifdef DEBUG
#endif
//establish seed
  int *seed = (int *)(calloc(Nparticles,sizeof(int )));
  int i;
  
#pragma omp parallel for private (i)
  for (i = 0; i <= Nparticles - 1; i += 1) {
    seed[i] = i + 1;
  }
//calloc matrix
  unsigned char *I = (unsigned char *)(calloc((IszX * IszY * Nfr),sizeof(unsigned char )));
  long long start = get_time();
//call video sequence
  videoSequence(I,IszX,IszY,Nfr,seed);
  long long endVideoSequence = get_time();
  printf("VIDEO SEQUENCE TOOK %f (s)\n",(elapsed_time(start,endVideoSequence)));
//call particle filter
  particleFilter(I,IszX,IszY,Nfr,seed,Nparticles);
  long long endParticleFilter = get_time();
  printf("PARTICLE FILTER TOOK %f (s)\n",(elapsed_time(endVideoSequence,endParticleFilter)));
  printf("ENTIRE PROGRAM TOOK %f (s)\n",(elapsed_time(start,endParticleFilter)));
  free(seed);
  free(I);
  return 0;
}
