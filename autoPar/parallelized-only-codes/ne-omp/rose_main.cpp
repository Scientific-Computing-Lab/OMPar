#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include <omp.h> 
typedef struct {
float x;
float y;
float z;}float3;
typedef struct {
float x;
float y;
float z;
float w;}float4;

inline float3 operator*(const float3 &a,const float b)
{
  return {a . x * b, a . y * b, a . z * b};
}

inline float3 operator-(const float3 &a,const float3 &b)
{
  return {a . x - b . x, a . y - b . y, a . z - b . z};
}

inline float dot(const float3 &a,const float3 &b)
{
  return a . x * b . x + a . y * b . y + a . z * b . z;
}

inline float3 normalize(const float3 &v)
{
  float invLen = 1.f / sqrtf((dot(v,v)));
  return v*invLen;
}

inline float3 cross(const float3 &a,const float3 &b)
{
  return {a . y * b . z - a . z * b . y, a . z * b . x - a . x * b . z, a . x * b . y - a . y * b . x};
}

inline float length(const float3 &v)
{
  return sqrtf((dot(v,v)));
}

inline float4 normalEstimate(const float3 *points,int idx,int width,int height)
{
  float3 query_pt = points[idx];
  if (std::isnan(query_pt . z)) 
    return {(0.f), (0.f), (0.f), (0.f)};
  int xIdx = idx % width;
  int yIdx = idx / width;
// are we at a border? are our neighbor valid points?
  bool west_valid = xIdx > 1 && !std::isnan(points[idx - 1] . z) && fabsf(points[idx - 1] . z - query_pt . z) < 200.f;
  bool east_valid = xIdx < width - 1 && !std::isnan(points[idx + 1] . z) && fabsf(points[idx + 1] . z - query_pt . z) < 200.f;
  bool north_valid = yIdx > 1 && !std::isnan(points[idx - width] . z) && fabsf(points[idx - width] . z - query_pt . z) < 200.f;
  bool south_valid = yIdx < height - 1 && !std::isnan(points[idx + width] . z) && fabsf(points[idx + width] . z - query_pt . z) < 200.f;
  float3 horiz;
  float3 vert;
  if ((west_valid & east_valid)) 
    horiz = points[idx + 1]-points[idx - 1];
  if ((west_valid & (!east_valid))) 
    horiz = points[idx]-points[idx - 1];
  if (((!west_valid) & east_valid)) 
    horiz = points[idx + 1]-points[idx];
  if (((!west_valid) & (!east_valid))) 
    return {(0.f), (0.f), (0.f), (1.f)};
  if ((south_valid & north_valid)) 
    vert = points[idx - width]-points[idx + width];
  if ((south_valid & (!north_valid))) 
    vert = points[idx]-points[idx + width];
  if (((!south_valid) & north_valid)) 
    vert = points[idx - width]-points[idx];
  if (((!south_valid) & (!north_valid))) 
    return {(0.f), (0.f), (0.f), (1.f)};
  float3 normal = cross(horiz,vert);
  float curvature = length(normal);
  curvature = (fabsf(horiz . z) > 0.04f || fabsf(vert . z) > 0.04f || !west_valid || !east_valid || !north_valid || !south_valid);
  float3 mc = normalize(normal);
  if (dot(query_pt,mc) > 0.f) 
    mc = mc*- 1.f;
  return {mc . x, mc . y, mc . z, curvature};
}

int main(int argc,char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n",argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]);
  const int height = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  const int numPts = width * height;
  const int size = (numPts * sizeof(float3 ));
  const int normal_size = (numPts * sizeof(float4 ));
  float3 *points = (float3 *)(malloc(size));
  float4 *normal_points = (float4 *)(malloc(normal_size));
  srand(123);
  for (int i = 0; i <= numPts - 1; i += 1) {
    points[i] . x = (rand() % width);
    points[i] . y = (rand() % height);
    points[i] . z = (rand() % 256);
  }
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (int i = 0; i <= repeat - 1; i += 1) {
      for (int idx = 0; idx <= numPts - 1; idx += 1) {
        normal_points[idx] = normalEstimate(points,idx,width,height);
      }
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average kernel execution time: %f (s)\n",(time * 1e-9f / repeat));
  }
  float sx;
  float sy;
  float sz;
  float sw;
  sx = sy = sz = sw = 0.f;
  
#pragma omp parallel for reduction (+:sx,sy,sz,sw) firstprivate (numPts)
  for (int i = 0; i <= numPts - 1; i += 1) {
    sx += normal_points[i] . x;
    sy += normal_points[i] . y;
    sz += normal_points[i] . z;
    sw += normal_points[i] . w;
  }
  printf("Checksum: x=%f y=%f z=%f w=%f\n",sx,sy,sz,sw);
  free(normal_points);
  free(points);
  return 0;
}
