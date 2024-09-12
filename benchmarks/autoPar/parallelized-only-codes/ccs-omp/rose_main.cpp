/*
 Condition-dependent Correlation Subgroups (CCS) 
 Description: Biclustering has been emerged as a powerful tool for 
 identification of a group of co-expressed genes under a subset 
 of experimental conditions (measurements) present in a gene 
 expression dataset.  In this program we implemented CCS biclustering. 
 Developer: Dr. Anindya Bhattacharya and Dr. Yan Cui, UTHSC, Memphis, TN, USA
 Email: anindyamail123@gmail.com; ycui2@uthsc.edu 
 Note: The minimum number of genes and the samples per bicluster is 10. 
 User can alter the minimum size by changing the values for 'mingene' 
 and 'minsample' defined in "ccs.h" file for minimum number of genes and samples
 respectively. 
*/
#include <chrono>
#include <omp.h>
#include "ccs.h"
#include "matrixsize.c"
#include "readgene.c"
#include "pair_cor.c"
#include "bicluster_pair_score.c"
#include "merge_bicluster.c"
#include "print_bicluster.c"
// number of samples in the input datamatrix. 
// Fixed here to make static shared memory on a device
#define MAXSAMPLE 200 
#include <omp.h> 

struct pair_r compute(float *genekj,float *geneij,const char *sample,int wid,int k,int i,int D,const float *gene)
{
  int j;
  float sx = 0.f;
  float sxx = 0.f;
  float sy = 0.f;
  float sxy = 0.f;
  float syy = 0.f;
  float sx_n = 0.f;
  float sxx_n = 0.f;
  float sy_n = 0.f;
  float sxy_n = 0.f;
  float syy_n = 0.f;
  struct pair_r rval = {(0.f), (0.f)};
  for (j = 0; j <= D - 1; j += 1) {
    genekj[j] = gene[k * (D + 1) + j];
    if (sample[j] == '1') 
      sx += genekj[j];
     else 
      sx_n += genekj[j];
  }
  sx /= wid;
  sx_n /= (D - wid);
  
#pragma omp parallel for private (j) reduction (+:sxx,sxx_n) firstprivate (sx,sx_n)
  for (j = 0; j <= D - 1; j += 1) {
    if (sample[j] == '1') 
      sxx += (sx - genekj[j]) * (sx - genekj[j]);
     else 
      sxx_n += (sx_n - genekj[j]) * (sx_n - genekj[j]);
  }
  sxx = sqrtf(sxx);
  sxx_n = sqrtf(sxx_n);
  for (j = 0; j <= D - 1; j += 1) {
    geneij[j] = gene[i * (D + 1) + j];
    if (sample[j] == '1') 
      sy += geneij[j];
     else 
      sy_n += geneij[j];
  }
  sy /= wid;
  sy_n /= (D - wid);
  
#pragma omp parallel for private (j) reduction (+:sxy,syy,sxy_n,syy_n) firstprivate (D,sx,sy,sx_n,sy_n)
  for (j = 0; j <= D - 1; j += 1) {
    if (sample[j] == '1') {
      sxy += (sx - genekj[j]) * (sy - geneij[j]);
      syy += (sy - geneij[j]) * (sy - geneij[j]);
    }
     else {
      sxy_n += (sx_n - genekj[j]) * (sy_n - geneij[j]);
      syy_n += (sy_n - geneij[j]) * (sy_n - geneij[j]);
    }
  }
  syy = sqrtf(syy);
  syy_n = sqrtf(syy_n);
  rval . r = fabsf(sxy / (sxx * syy));
  rval . n_r = fabsf(sxy_n / (sxx_n * syy_n));
  return rval;
}

void compute_bicluster(const float *gene,const int n,const int maxbcn,const int D,const float thr,char *maxbc_sample,char *maxbc_data,float *maxbc_score,int *maxbc_datacount,int *maxbc_samplecount,char *tmpbc_sample,char *tmpbc_data)
{
{
    float s_genekj[200];
    float s_geneij[200];
    char s_vect[600];
{
      int k = omp_get_team_num();
      if (k < maxbcn) {
        float jcc;
        float mean_k;
        float mean_i;
        int i;
        int j;
        int l;
        int vl;
        int wid;
        int wid_0;
        int wid_1;
        int wid_2;
        int l_i;
        int t_tot;
        int t_dif;
        int dif;
        int tot;
        struct pair_r rval;
        int tmpbc_datacount;
        int tmpbc_samplecount;
        float genekj;
        float geneij;
        maxbc_score[k] = 1.f;
        maxbc_datacount[k] = 0;
//calculate mean expression for gene k
        mean_k = gene[k * (D + 1) + D];
        for (i = k + 1; i <= n - 1; i += 1) 
//pair k,i
{
//calculate mean expression for gene i
          mean_i = gene[i * (D + 1) + D];
          wid_0 = 0;
          wid_1 = 0;
          wid_2 = 0;
          for (j = 0; j <= D - 1; j += 1) {
            genekj = gene[k * (D + 1) + j];
            geneij = gene[i * (D + 1) + j];
            if (genekj - mean_k >= 0 && geneij - mean_i >= 0) 
//i and k upregulated : positive correlation
{
              s_vect[0 * 3 + j] = '1';
              s_vect[1 * 3 + j] = '0';
              s_vect[2 * 3 + j] = '0';
              wid_0++;
            }
             else if (genekj - mean_k < 0 && geneij - mean_i < 0) 
// i and k down regulated : positive correlation
{
              s_vect[0 * 3 + j] = '0';
              s_vect[1 * 3 + j] = '1';
              s_vect[2 * 3 + j] = '0';
              wid_1++;
            }
             else if ((genekj - mean_k) * (geneij - mean_i) < 0) 
//betwenn i and k one is up regulated and the other one is down regulated : negative correlation
{
              s_vect[0 * 3 + j] = '0';
              s_vect[1 * 3 + j] = '0';
              s_vect[2 * 3 + j] = '1';
              wid_2++;
            }
          }
          for (vl = 0; vl <= 2; vl += 1) {
            dif = 0;
            tot = 0;
            if (vl == 0) 
              wid = wid_0;
             else if (vl == 1) 
              wid = wid_1;
            if (vl == 2) 
              wid = wid_2;
            if (wid > 10) {
//minimum samples required to form a bicluster module. Default minimum set to 10 in ccs.h   
              rval = compute(s_genekj,s_geneij,(s_vect + vl * 200),wid,k,i,D,gene);
            }
             else {
              continue; 
            }
            if (rval . r > thr) {
              tot++;
              if (rval . n_r > thr) 
                dif++;
              
#pragma omp parallel for private (j)
              for (j = 0; j <= D - 1; j += 1) {
                tmpbc_sample[k * D + j] = s_vect[vl * 200 + j];
              }
              
#pragma omp parallel for private (j)
              for (j = 0; j <= n - 1; j += 1) {
                tmpbc_data[k * n + j] = '0';
              }
              tmpbc_data[k * n + k] = '1';
              tmpbc_data[k * n + i] = '1';
              tmpbc_datacount = 2;
              tmpbc_samplecount = wid;
              for (l = 0; l <= n - 1; l += 1) {
//bicluster augmentation
                if (l != i && l != k) {
                  t_tot = 0;
                  t_dif = 0;
                  for (l_i = 0; l_i <= n - 1; l_i += 1) {
                    if (tmpbc_data[k * n + l_i] == '1') {
                      rval = compute(s_genekj,s_geneij,(s_vect + vl * 200),wid,l,l_i,D,gene);
                      if (rval . r > thr) 
                        t_tot += 1;
                       else {
                        t_tot = 0;
                        break; 
                      }
                      if (rval . n_r > thr) 
                        t_dif += 1;
                    }
                  }
                  if (t_tot > 0) {
                    tmpbc_data[k * n + l] = '1';
                    tmpbc_datacount += 1;
                    tot += t_tot;
                    dif += t_dif;
                  }
                }
              }
// end of augmentation
// Compute Jaccard score
              if (tot > 0) 
                jcc = ((float )dif) / tot;
               else 
                jcc = 1.f;
/*   Select bicluster candidate as the largest (maxbc[k].datacount<tmpbc.datacount) 
                   of all condition dependent (jaccard score <0.01) bicluster for k. Minimum number of gene 
                   for a bicluster is set at 10. See the mingene at ccs.h                                */
              if (jcc < 0.01f && maxbc_datacount[k] < tmpbc_datacount && tmpbc_datacount > 10) {
                maxbc_score[k] = jcc;
                
#pragma omp parallel for private (j)
                for (j = 0; j <= n - 1; j += 1) {
                  maxbc_data[k * n + j] = tmpbc_data[k * n + j];
                }
                
#pragma omp parallel for private (j)
                for (j = 0; j <= D - 1; j += 1) {
                  maxbc_sample[k * D + j] = tmpbc_sample[k * D + j];
                }
                maxbc_datacount[k] = tmpbc_datacount;
                maxbc_samplecount[k] = tmpbc_samplecount;
              }
            }
//end of r>thr condition
          }
//end of loop for vl  
        }
// end of i loop
      }
    }
  }
}

int main(int argc,char *argv[])
{
  FILE *in;
  FILE *out;
  struct gn *gene;
  char **Hd;
  char *infile;
  char *outfile;
  int c;
  int errflag;
  int maxbcn = 1000;
  int print_type = 0;
  int repeat = 0;
  int i;
  int n;
  int D;
  extern char *optarg;
  float thr;
  struct bicl *bicluster;
  float overlap = 100.f;
  infile = outfile = 0L;
  in = out = 0L;
  errflag = n = D = 0;
  thr = 0.f;
  while((c = getopt(argc,argv,"ht:m:r:i:p:o:g:?")) != - 1){
    switch(c){
      case 'h':
// help
      printUsage();
      exit(0);
      case 't':
// threshold value
      thr = (atof(optarg));
      break; 
      case 'm':
// maximum number of bicluster search
      maxbcn = atoi(optarg);
      break; 
      case 'r':
// kernel repeat times
      repeat = atoi(optarg);
      break; 
      case 'g':
// output file format
      overlap = (atof(optarg));
      break; 
      case 'p':
// output file format
      print_type = atoi(optarg);
      break; 
      case 'i':
// the input expression file
      infile = optarg;
      break; 
      case 'o':
// the output file
      outfile = optarg;
      break; 
      case ':':
/* -f or -o without operand */
      printf("Option -%c requires an operand\n",optopt);
      errflag++;
      break; 
      case '?':
      fprintf(stderr,"Unrecognized option: -%c\n",optopt);
      errflag++;
    }
  }
  if (thr == 0) {
    fprintf(stderr,"***** WARNING: Threshold Theta (corr coeff) value assumed to be ZERO (0)\n");
  }
  if (outfile == 0L) {
    fprintf(stderr,"***** WARNING: Output file assumed to be STDOUT\n");
    out = stdout;
  }
   else if ((out = fopen(outfile,"w")) == 0L) 
//write open bicluster file
{
    fprintf(stderr,"***** ERROR: Unable to open Output file %s\n",outfile);
    errflag++;
  }
  if (thr < 0 || thr > 1) {
    fprintf(stderr,"***** ERROR: Threshold Theta (corr coeff) must be between 0.0-1.0\n");
  }
  if (infile == 0L) {
    fprintf(stderr,"***** ERROR: Input file not defined\n");
    if (out) 
      fclose(out);
    errflag++;
  }
   else if ((in = fopen(infile,"r")) == 0L) 
//open gene file
{
    fprintf(stderr,"***** ERROR: Unable to open Input %s\n",infile);
    if (out) 
      fclose(out);
    errflag++;
  }
  if (errflag) {
    printUsage();
    exit(1);
  }
  getmatrixsize(in,&n,&D);
  printf("Number of rows=%d\tNumber of columns=%d\n",n,D);
  if (maxbcn > n) 
    maxbcn = n;
  gene = ((struct gn *)(calloc(n,sizeof(struct gn ))));
  Hd = ((char **)(calloc((D + 1),sizeof(char *))));
  for (i = 0; i <= n - 1; i += 1) {
    gene[i] . x = ((float *)(calloc((D + 1),sizeof(float ))));
  }
  bicluster = ((struct bicl *)(calloc(maxbcn,sizeof(struct bicl ))));
  for (i = 0; i <= maxbcn - 1; i += 1) {
    bicluster[i] . sample = ((char *)(calloc(D,sizeof(char ))));
    bicluster[i] . data = ((char *)(calloc(n,sizeof(char ))));
  }
// initialize the gene data
  readgene(infile,gene,Hd,n,D);
  auto start = std::chrono::_V2::steady_clock::now();
  float *d_gene = (float *)(malloc(sizeof(float ) * n * (D + 1)));
  for (i = 0; i <= n - 1; i += 1) {
    memcpy((d_gene + i * (D + 1)),gene[i] . x,sizeof(float ) * (D + 1));
  }
  float *d_bc_score = (float *)(malloc(sizeof(float ) * maxbcn));
  int *d_bc_datacount = (int *)(malloc(sizeof(int ) * maxbcn));
  int *d_bc_samplecount = (int *)(malloc(sizeof(int ) * maxbcn));
  char *d_bc_sample = (char *)(malloc(sizeof(char ) * D * maxbcn));
  char *d_bc_sample_tmp = (char *)(malloc(sizeof(char ) * D * maxbcn));
  char *d_bc_data = (char *)(malloc(sizeof(char ) * n * maxbcn));
  char *d_bc_data_tmp = (char *)(malloc(sizeof(char ) * n * maxbcn));
{
    auto kstart = std::chrono::_V2::steady_clock::now();
    for (i = 0; i <= repeat - 1; i += 1) {
      compute_bicluster(d_gene,n,maxbcn,D,thr,d_bc_sample,d_bc_data,d_bc_score,d_bc_datacount,d_bc_samplecount,d_bc_sample_tmp,d_bc_data_tmp);
    }
    auto kend = std::chrono::_V2::steady_clock::now();
    auto ktime = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((kend-kstart)) . count();
    printf("Average kernel execution time %f (s)\n",(ktime * 1e-9f / repeat));
  }
  for (i = 0; i <= maxbcn - 1; i += 1) {
    memcpy(bicluster[i] . sample,(d_bc_sample_tmp + D * i),sizeof(char ) * D);
    memcpy(bicluster[i] . data,(d_bc_data_tmp + n * i),sizeof(char ) * n);
  }
  
#pragma omp parallel for private (i)
  for (i = 0; i <= maxbcn - 1; i += 1) {
    bicluster[i] . score = d_bc_score[i];
    bicluster[i] . datacount = d_bc_datacount[i];
    bicluster[i] . samplecount = d_bc_samplecount[i];
  }
  printbicluster(out,gene,Hd,n,D,maxbcn,thr,bicluster,print_type,overlap);
  for (i = 0; i <= n - 1; i += 1) {
    free(gene[i] . x);
    free(gene[i] . id);
  }
  free(gene);
  for (i = 0; i <= D + 1 - 1; i += 1) {
    free(Hd[i]);
  }
  free(Hd);
  for (i = 0; i <= maxbcn - 1; i += 1) {
    free(bicluster[i] . sample);
    free(bicluster[i] . data);
  }
  free(d_gene);
  free(d_bc_score);
  free(d_bc_datacount);
  free(d_bc_samplecount);
  free(d_bc_sample);
  free(d_bc_sample_tmp);
  free(d_bc_data);
  free(d_bc_data_tmp);
  free(bicluster);
  auto end = std::chrono::_V2::steady_clock::now();
  auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
  printf("Elapsed time = %f (s)\n",(time * 1e-9f));
  if (print_type == 0) 
    fprintf(out,"\n\nElapsed time = %f s\n",(time * 1e-9f));
  if (out) 
    fclose(out);
  return 0;
}
