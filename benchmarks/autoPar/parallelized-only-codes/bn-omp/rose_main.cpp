#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include "kernels.cpp"
#include <omp.h> 
const int HIGHEST = 3;
const int ITER = 100;
const int WORKLOAD = 1;
int sizepernode;
// global var
float preScore = - 99999999999.f;
float score = 0.0;
float maxScore[3] = {(- 999999999.f)};
bool orders[45][45];
bool preOrders[45][45];
bool preGraph[45][45];
bool bestGraph[3][45][45];
bool graph[45][45];
float *localscore;
float *scores;
float *LG;
int *parents;
void initial();
// initial orders and data
int genOrders();
// swap
int ConCore();
// discard new order or not
// get every possible set of parents for a node
void incr(int *bit,int n);
// binary code increases 1 each time
void incrS(int *bit,int n);
// STATE_N code increases 1 each time
// get every possible combination of state for a parent set
bool getState(int parN,int *state,int time);
float logGamma(int N);
// log and gamma
float findBestGraph(float *D_localscore,int *D_resP,float *D_Score,bool *D_parent);
void genScore();
void sortGraph();
void swap(int a,int b);
void Pre_logGamma();
int findindex(int *arr,int size);
int C(int n,int a);

int main(int argc,char **argv)
{
  if (argc != 3) {
    printf("Usage: ./%s <path to output file> <repeat>\n",argv[0]);
    return 1;
  }
// save output in a file
  FILE *fpout = fopen(argv[1],"w");
  if (fpout == 0L) {
    printf("Error: failed to open %s. Exit..\n",argv[1]);
    return - 1;
  }
  const int repeat = atoi(argv[2]);
  int i;
  int j;
  int c = 0;
  int tmp;
  int a;
  int b;
  float tmpd;
  printf("NODE_N=%d\nInitialization...\n",NODE_N);
  srand(2);
  initial();
  Pre_logGamma();
  scores = ((float *)(malloc((sizepernode / (256 * WORKLOAD) + 1) * sizeof(float ))));
  parents = ((int *)(malloc(((sizepernode / (256 * WORKLOAD) + 1) * 4) * sizeof(int ))));
//float *D_Score = (float*) malloc ((sizepernode / (256 * WORKLOAD) + 1) * sizeof(float));
  float *D_Score = scores;
  bool *D_parent = (bool *)(malloc(NODE_N * sizeof(bool )));
  int *D_resP = (int *)(malloc(((sizepernode / (256 * WORKLOAD) + 1) * 4) * sizeof(int )));
{
    auto start = std::chrono::_V2::steady_clock::now();
    for (i = 0; i <= repeat - 1; i += 1) {
      genScoreKernel(sizepernode,localscore,data,LG);
    }
    auto end = std::chrono::_V2::steady_clock::now();
    auto time = std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
    printf("Average execution time of genScoreKernel: %f (s)\n",(time * 1e-9f / repeat));
    #ifdef DEBUG
    #endif
    long findBestGraph_time = 0;
    i = 0;
    while(i != ITER){
      i++;
      score = 0;
      
#pragma omp parallel for private (j,a) firstprivate (NODE_N)
      for (a = 0; a <= NODE_N - 1; a += 1) {
        
#pragma omp parallel for private (j)
        for (j = 0; j <= NODE_N - 1; j += 1) {
          orders[a][j] = preOrders[a][j];
        }
      }
      tmp = rand() % 6;
      for (j = 0; j <= tmp - 1; j += 1) {
        genOrders();
      }
      start = std::chrono::_V2::steady_clock::now();
      score = findBestGraph(localscore,D_resP,D_Score,D_parent);
      end = std::chrono::_V2::steady_clock::now();
      findBestGraph_time += std::chrono::duration_cast< std::chrono::nanoseconds  , int64_t  , std::nano  > ((end-start)) . count();
      ConCore();
// store the top HIGHEST highest orders
      if (c < HIGHEST) {
        tmp = 1;
        
#pragma omp parallel for private (j)
        for (j = 0; j <= c - 1; j += 1) {
          if (maxScore[j] == preScore) {
            tmp = 0;
          }
        }
        if (tmp != 0) {
          maxScore[c] = preScore;
          
#pragma omp parallel for private (a,b)
          for (a = 0; a <= NODE_N - 1; a += 1) {
            
#pragma omp parallel for private (b)
            for (b = 0; b <= NODE_N - 1; b += 1) {
              bestGraph[c][a][b] = preGraph[a][b];
            }
          }
          c++;
        }
      }
       else if (c == HIGHEST) {
        sortGraph();
        c++;
      }
       else {
        tmp = 1;
        for (j = 0; j <= HIGHEST - 1; j += 1) {
          if (maxScore[j] == preScore) {
            tmp = 0;
            break; 
          }
        }
        if (tmp != 0 && preScore > maxScore[HIGHEST - 1]) {
          maxScore[HIGHEST - 1] = preScore;
          
#pragma omp parallel for private (a,b)
          for (a = 0; a <= NODE_N - 1; a += 1) {
            
#pragma omp parallel for private (b)
            for (b = 0; b <= NODE_N - 1; b += 1) {
              bestGraph[HIGHEST - 1][a][b] = preGraph[a][b];
            }
          }
          b = HIGHEST - 1;
          for (a = HIGHEST - 2; a >= 0; a += -1) {
            if (maxScore[b] > maxScore[a]) {
              swap(a,b);
              tmpd = maxScore[a];
              maxScore[a] = maxScore[b];
              maxScore[b] = tmpd;
              b = a;
            }
          }
        }
      }
    }
// endwhile
    printf("Find best graph time %lf (s)\n",findBestGraph_time * 1e-9);
  }
  free(LG);
  free(localscore);
  free(scores);
  free(parents);
  free(D_parent);
  free(D_resP);
  for (j = 0; j <= HIGHEST - 1; j += 1) {
    fprintf(fpout,"score:%f\n",maxScore[j]);
    fprintf(fpout,"Best Graph:\n");
    for (int a = 0; a <= NODE_N - 1; a += 1) {
      for (int b = 0; b <= NODE_N - 1; b += 1) {
        fprintf(fpout,"%d ",bestGraph[j][a][b]);
      }
      fprintf(fpout,"\n");
    }
    fprintf(fpout,"--------------------------------------------------------------------\n");
  }
  return 0;
}

float findBestGraph(float *D_localscore,int *D_resP,float *D_Score,bool *D_parent)
{
  float bestls = - 99999999.f;
  int bestparent[5];
  int bestpN;
  int total;
  int node;
  int index;
  int pre[45] = {(0)};
  int parent[45] = {(0)};
  int posN = 0;
  int i;
  int j;
  int parN;
  int tmp;
  int k;
  int l;
  float ls = - 99999999999.f;
  float score = 0.f;
  int blocknum;
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= NODE_N - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= NODE_N - 1; j += 1) {
      graph[i][j] = 0;
    }
  }
  for (node = 0; node <= NODE_N - 1; node += 1) {
    bestls = - 99999999.f;
    posN = 0;
    for (i = 0; i <= NODE_N - 1; i += 1) {
      if (orders[node][i] == 1) {
        pre[posN++] = i;
      }
    }
    if (posN >= 0) {
      total = C(posN,4) + C(posN,3) + C(posN,2) + posN + 1;
      blocknum = total / (256 * WORKLOAD) + 1;
      const int sizePerNode = sizepernode;
      
#pragma omp parallel for
      for (int t = 0; t <= blocknum * 4 - 1; t += 1) {
        D_resP[t] = 0;
      }
      
#pragma omp parallel for
      for (int t = 0; t <= blocknum - 1; t += 1) {
        D_Score[t] = - 999999.f;
      }
      memcpy(D_parent,orders[node],NODE_N * sizeof(bool ));
      computeKernel(WORKLOAD,sizePerNode,D_localscore,D_parent,node,total,D_Score,D_resP,blocknum);
      memcpy(parents,D_resP,(blocknum * 4) * sizeof(int ));
#ifdef DEBUG
#endif
      for (i = 0; i <= blocknum - 1; i += 1) {
        if (D_Score[i] > bestls) {
          bestls = D_Score[i];
          parN = 0;
          for (tmp = 0; tmp <= 3; tmp += 1) {
            if (parents[i * 4 + tmp] < 0) 
              break; 
            bestparent[tmp] = parents[i * 4 + tmp];
            parN++;
          }
          bestpN = parN;
        }
      }
    }
     else {
      if (posN >= 4) {
        for (i = 0; i <= posN - 1; i += 1) {
          for (j = i + 1; j <= posN - 1; j += 1) {
            for (k = j + 1; k <= posN - 1; k += 1) {
              for (l = k + 1; l <= posN - 1; l += 1) {
                parN = 4;
                if (pre[i] > node) 
                  parent[1] = pre[i];
                 else 
                  parent[1] = pre[i] + 1;
                if (pre[j] > node) 
                  parent[2] = pre[j];
                 else 
                  parent[2] = pre[j] + 1;
                if (pre[k] > node) 
                  parent[3] = pre[k];
                 else 
                  parent[3] = pre[k] + 1;
                if (pre[l] > node) 
                  parent[4] = pre[l];
                 else 
                  parent[4] = pre[l] + 1;
                index = findindex(parent,parN);
                index += sizepernode * node;
                ls = localscore[index];
                if (ls > bestls) {
                  bestls = ls;
                  bestpN = parN;
                  
#pragma omp parallel for private (tmp) firstprivate (parN)
                  for (tmp = 0; tmp <= parN - 1; tmp += 1) {
                    bestparent[tmp] = parent[tmp + 1];
                  }
                }
              }
            }
          }
        }
      }
      if (posN >= 3) {
        for (i = 0; i <= posN - 1; i += 1) {
          for (j = i + 1; j <= posN - 1; j += 1) {
            for (k = j + 1; k <= posN - 1; k += 1) {
              parN = 3;
              if (pre[i] > node) 
                parent[1] = pre[i];
               else 
                parent[1] = pre[i] + 1;
              if (pre[j] > node) 
                parent[2] = pre[j];
               else 
                parent[2] = pre[j] + 1;
              if (pre[k] > node) 
                parent[3] = pre[k];
               else 
                parent[3] = pre[k] + 1;
              index = findindex(parent,parN);
              index += sizepernode * node;
              ls = localscore[index];
              if (ls > bestls) {
                bestls = ls;
                bestpN = parN;
                
#pragma omp parallel for private (tmp) firstprivate (parN)
                for (tmp = 0; tmp <= parN - 1; tmp += 1) {
                  bestparent[tmp] = parent[tmp + 1];
                }
              }
            }
          }
        }
      }
      if (posN >= 2) {
        for (i = 0; i <= posN - 1; i += 1) {
          for (j = i + 1; j <= posN - 1; j += 1) {
            parN = 2;
            if (pre[i] > node) 
              parent[1] = pre[i];
             else 
              parent[1] = pre[i] + 1;
            if (pre[j] > node) 
              parent[2] = pre[j];
             else 
              parent[2] = pre[j] + 1;
            index = findindex(parent,parN);
            index += sizepernode * node;
            ls = localscore[index];
            if (ls > bestls) {
              bestls = ls;
              bestpN = parN;
              
#pragma omp parallel for private (tmp) firstprivate (parN)
              for (tmp = 0; tmp <= parN - 1; tmp += 1) {
                bestparent[tmp] = parent[tmp + 1];
              }
            }
          }
        }
      }
      if (posN >= 1) {
        for (i = 0; i <= posN - 1; i += 1) {
          parN = 1;
          if (pre[i] > node) 
            parent[1] = pre[i];
           else 
            parent[1] = pre[i] + 1;
          index = findindex(parent,parN);
          index += sizepernode * node;
          ls = localscore[index];
          if (ls > bestls) {
            bestls = ls;
            bestpN = parN;
            
#pragma omp parallel for private (tmp) firstprivate (parN)
            for (tmp = 0; tmp <= parN - 1; tmp += 1) {
              bestparent[tmp] = parent[tmp + 1];
            }
          }
        }
      }
      parN = 0;
      index = sizepernode * node;
      ls = localscore[index];
      if (ls > bestls) {
        bestls = ls;
        bestpN = 0;
      }
    }
    if (bestls > - 99999999.f) {
      for (i = 0; i <= bestpN - 1; i += 1) {
        if (bestparent[i] < node) 
          graph[node][bestparent[i] - 1] = 1;
         else 
          graph[node][bestparent[i]] = 1;
      }
      score += bestls;
    }
  }
  return score;
}

void sortGraph()
{
  float max = - 99999999999999.f;
  int maxi;
  int i;
  int j;
  float tmp;
  for (j = 0; j <= HIGHEST - 1 - 1; j += 1) {
    max = maxScore[j];
    maxi = j;
    for (i = j + 1; i <= HIGHEST - 1; i += 1) {
      if (maxScore[i] > max) {
        max = maxScore[i];
        maxi = i;
      }
    }
    swap(j,maxi);
    tmp = maxScore[j];
    maxScore[j] = max;
    maxScore[maxi] = tmp;
  }
}

void swap(int a,int b)
{
  int i;
  int j;
  bool tmp;
  
#pragma omp parallel for private (tmp,i,j) firstprivate (NODE_N)
  for (i = 0; i <= NODE_N - 1; i += 1) {
    
#pragma omp parallel for private (tmp,j) firstprivate (a,b)
    for (j = 0; j <= NODE_N - 1; j += 1) {
      tmp = bestGraph[a][i][j];
      bestGraph[a][i][j] = bestGraph[b][i][j];
      bestGraph[b][i][j] = tmp;
    }
  }
}

void initial()
{
  int i;
  int j;
  int tmp;
  int a;
  int b;
  int r;
  bool tmpd;
  tmp = 1;
  for (i = 1; i <= 4; i += 1) {
    tmp += C(NODE_N - 1,i);
  }
  sizepernode = tmp;
  tmp *= NODE_N;
  localscore = ((float *)(malloc(tmp * sizeof(float ))));
  
#pragma omp parallel for private (i) firstprivate (tmp)
  for (i = 0; i <= tmp - 1; i += 1) {
    localscore[i] = 0;
  }
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= NODE_N - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= NODE_N - 1; j += 1) {
      orders[i][j] = 0;
    }
  }
  
#pragma omp parallel for private (i,j)
  for (i = 0; i <= NODE_N - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= i - 1; j += 1) {
      orders[i][j] = 1;
    }
  }
  r = rand() % 10000;
  for (i = 0; i <= r - 1; i += 1) {
    a = rand() % NODE_N;
    b = rand() % NODE_N;
    
#pragma omp parallel for private (tmpd,j)
    for (j = 0; j <= NODE_N - 1; j += 1) {
      tmpd = orders[j][a];
      orders[j][a] = orders[j][b];
      orders[j][b] = tmpd;
    }
    
#pragma omp parallel for private (tmpd,j) firstprivate (a,b)
    for (j = 0; j <= NODE_N - 1; j += 1) {
      tmpd = orders[a][j];
      orders[a][j] = orders[b][j];
      orders[b][j] = tmpd;
    }
  }
  
#pragma omp parallel for private (i,j) firstprivate (NODE_N)
  for (i = 0; i <= NODE_N - 1; i += 1) {
    
#pragma omp parallel for private (j)
    for (j = 0; j <= NODE_N - 1; j += 1) {
      preOrders[i][j] = orders[i][j];
    }
  }
}
// generate ramdom order

int genOrders()
{
  int a;
  int b;
  int j;
  bool tmp;
  a = rand() % NODE_N;
  b = rand() % NODE_N;
  
#pragma omp parallel for private (tmp,j)
  for (j = 0; j <= NODE_N - 1; j += 1) {
    tmp = orders[a][j];
    orders[a][j] = orders[b][j];
    orders[b][j] = tmp;
  }
  
#pragma omp parallel for private (tmp,j) firstprivate (NODE_N,a,b)
  for (j = 0; j <= NODE_N - 1; j += 1) {
    tmp = orders[j][a];
    orders[j][a] = orders[j][b];
    orders[j][b] = tmp;
  }
  return 1;
}
// decide leave or discard an order

int ConCore()
{
  int i;
  int j;
  float tmp;
  tmp = (log((rand() % 100000) / 100000.0));
  if (tmp < score - preScore) {
    
#pragma omp parallel for private (i,j) firstprivate (NODE_N)
    for (i = 0; i <= NODE_N - 1; i += 1) {
      
#pragma omp parallel for private (j)
      for (j = 0; j <= NODE_N - 1; j += 1) {
        preOrders[i][j] = orders[i][j];
        preGraph[i][j] = graph[i][j];
      }
    }
    preScore = score;
    return 1;
  }
  return 0;
}

void genScore()
{
}

void Pre_logGamma()
{
  LG = ((float *)(malloc((DATA_N + 2) * sizeof(float ))));
  LG[1] = (log(1.0));
  float i;
  for (i = 2; i <= (DATA_N + 1); i += 1) {
    LG[(int )i] = LG[((int )i) - 1] + std::log((float )i);
  }
}

void incr(int *bit,int n)
{
  bit[n]++;
  if (bit[n] >= 2) {
    bit[n] = 0;
    incr(bit,n + 1);
  }
  return ;
}

void incrS(int *bit,int n)
{
  bit[n]++;
  if (bit[n] >= STATE_N) {
    bit[n] = 0;
    incr(bit,n + 1);
  }
  return ;
}

bool getState(int parN,int *state,int time)
{
  int j = 1;
  j = (std::pow(STATE_N,(float )parN) - 1);
  if (time > j) 
    return false;
  if (time >= 1) 
    incrS(state,0);
  return true;
}

int findindex(int *arr,int size)
{
// reminder: arr[0] has to be 0 && size ==
// array size-1 && index start from 0
  int i;
  int j;
  int index = 0;
  for (i = 1; i <= size - 1; i += 1) {
    index += C(NODE_N - 1,i);
  }
  for (i = 1; i <= size - 1; i += 1) {
    for (j = arr[i - 1] + 1; j <= arr[i] - 1; j += 1) {
      index += C(NODE_N - 1 - j,size - i);
    }
  }
  index += arr[size] - arr[size - 1];
  return index;
}

int C(int n,int a)
{
  int i;
  int res = 1;
  int atmp = a;
  for (i = 0; i <= atmp - 1; i += 1) {
    res *= n;
    n--;
  }
  for (i = 0; i <= atmp - 1; i += 1) {
    res /= a;
    a--;
  }
  return res;
}
