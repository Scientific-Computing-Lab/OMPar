#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "linear.h"

double gettime()
{
  struct timeval t;
  gettimeofday(&t,0L);
  return t . tv_sec + t . tv_usec * 1e-6;
}
clock_t start;
clock_t end;
extern int cpu_offset;
/* Read file */

static void create_dataset(linear_param_t *params,data_t *dataset)
{
  FILE *ptr_file = fopen((params -> filename),"r");
  if (ptr_file == 0L) {
    perror("Failed to load dataset file");
    exit(1);
  }
  char *token;
  char buf[1024];
  for (size_t i = 0; i < params -> size && fgets(buf,1024,ptr_file) != 0L; i++) {
    token = strtok(buf,"\t");
    dataset[i] . x = (atof(token));
    token = strtok(0L,"\t");
    dataset[i] . y = (atof(token));
  }
  fclose(ptr_file);
}

static void temperature_regression(results_t *results,int repeat)
{
  linear_param_t params;
  params . repeat = repeat;
  params . filename = "assets/temperature.txt";
  params . size = 96453;
  params . wg_size = 63;
  params . wg_count = (96453 / 63);
  data_t *dataset = (data_t *)(malloc(sizeof(data_t ) * params . size));
  create_dataset(&params,dataset);
  results -> parallelized . ktime = 0;
  parallelized_regression(&params,dataset,&results -> parallelized);
  iterative_regression(&params,dataset,&results -> iterative);
  free(dataset);
}

static void print_results(results_t *results)
{
  printf("\t%s\n\t--------\n\t| Equation: y = %.3fx + %.3f\n\n","Parallelized",results -> parallelized . a1,results -> parallelized . a0);
  ;
  printf("\t%s\n\t--------\n\t| Equation: y = %.3fx + %.3f\n\n","Iterative",results -> iterative . a1,results -> iterative . a0);
  ;
}

static void write_results(results_t *results,const char *restricts)
{
  FILE *file = fopen("assets/_results.txt",restricts);
  fprintf(file,"%.3f   %.3f   %.3f   %d\n",results -> parallelized . a0,results -> parallelized . a1,results -> parallelized . ktime,results -> parallelized . rsquared);
  ;
  fprintf(file,"%.3f   %.3f   %.3f   %d\n",results -> iterative . a0,results -> iterative . a1,results -> iterative . ktime,results -> iterative . rsquared);
  ;
  fclose(file);
}

int main(int argc,char *argv[])
{
  results_t results = {/* Need explicit braces: is this where we insert the class name? */ {(0)}};
  if (argc != 3) {
    printf("Usage: linear <repeat> <cpu offset>\n");
    printf("Device execution only when cpu offset is 0\n");
    printf("Host execution only when cpu offset is 100\n");
    exit(0);
  }
  int repeat = atoi(argv[1]);
  cpu_offset = atoi(argv[2]);
  printf("CPU offset: %d\n",cpu_offset);
  double starttime = gettime();
  temperature_regression(&results,repeat);
  double endtime = gettime();
  printf("Total execution time: %lf ms\n",1000.0 * (endtime - starttime));
  printf("Average kernel execution time: %lf us\n",results . parallelized . ktime * 1e-3 / repeat);
  write_results(&results,"a");
  printf("\n> TEMPERATURE REGRESSION (%d)\n\n",96453);
  print_results(&results);
  return 0;
}
