/*************************************************************************/
/**   File:         rmse.c												**/
/**   Description:  calculate root mean squared error of particular     **/
/**                 clustering.											**/
/**   Author:  Sang-Ha Lee												**/
/**            University of Virginia.									**/
/**																		**/
/**   Note: euclid_dist_2() and find_nearest_point() adopted from       **/
/**			Minebench code.												**/
/**                                                                     **/
/*************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
//#include <omp.h>
#include "kmeans.h"
#include <omp.h> 
extern double wtime();
/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */

inline float euclid_dist_2(float *pt1,float *pt2,int numdims)
{
  int i;
  float ans = 0.0;
  
#pragma omp parallel for private (i) reduction (+:ans) firstprivate (numdims)
  for (i = 0; i <= numdims - 1; i += 1) {
    ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);
  }
  return ans;
}
/*----< find_nearest_point() >-----------------------------------------------*/

inline int find_nearest_point(float *pt,
/* [nfeatures] */
int nfeatures,float **pts,
/* [npts][nfeatures] */
int npts)
{
  int index;
  int i;
  float max_dist = 3.40282346638528859811704183484516925e+38F;
/* find the cluster center id with min distance to pt */
  for (i = 0; i <= npts - 1; i += 1) {
    float dist;
    dist = euclid_dist_2(pt,pts[i],nfeatures);
/* no need square root */
    if (dist < max_dist) {
      max_dist = dist;
      index = i;
    }
  }
  return index;
}
/*----< rms_err(): calculates RMSE of clustering >-------------------------------------*/

float rms_err(float **feature,
/* [npoints][nfeatures] */
int nfeatures,int npoints,float **cluster_centres,
/* [nclusters][nfeatures] */
int nclusters)
{
  int i;
  int nearest_cluster_index;
/* cluster center id with min distance to pt */
  float sum_euclid = 0.0;
/* sum of Euclidean distance squares */
  float ret;
/* return value */
/* calculate and sum the sqaure of euclidean distance*/
  for (i = 0; i <= npoints - 1; i += 1) {
    nearest_cluster_index = find_nearest_point(feature[i],nfeatures,cluster_centres,nclusters);
    sum_euclid += euclid_dist_2(feature[i],cluster_centres[nearest_cluster_index],nfeatures);
  }
/* divide by n, then take sqrt */
  ret = (sqrt((sum_euclid / npoints)));
  return ret;
}
