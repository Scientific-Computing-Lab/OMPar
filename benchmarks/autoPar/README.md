# AutoPar Pararallelization

Following are the details about autoPar relevant data, and directories.

`autoPar/parallelized-only-codes`: This directory holds the HeCBench benchmarks that have been successfully parallelized using autoPar.

`autoPar/autoPar-scale-test_benchmarks.txt`: This file lists the autoPar parallelized benchmarks that have successfully passed both compilation and execution tests.

## Experimental Configuration

[AutoParBench](https://github.com/llnl/autoparbench) tool is employed to remove parallel directives from the baseline HeCBench code. Specifically, the `pragmaRemover` function is utilized to process and remove `omp pragmas`.

```bash
pragmaRemover.out input.cpp
```

AutoPar is a tool developed using the ROSE compiler framework. The experiments were conducted with AutoPar version `ROSE 0.11.67.0.1`. Installation instructions for the ROSE compiler can be found here,  <https://github.com/rose-compiler/rose/wiki>.

To perform the automatic parallelization using autoPar the following command is used. This command specifies an include directory (-I) for CUDA files and compiles the input C++ file (-c).

```bash
autoPar -I"$source_dir/$dir-cuda" -c "$input_cpp_file"
```

## Scale Test

To perform scaling, follow these steps:

1. Run the `Makefile` for Benchmarks: Execute the Makefile for the benchmarks listed in `autoPar/autoPar-scale-test_benchmarks.txt`.

2. Compile the Code: Use the following command to compile:

    ```bash
    make DEVICE=CPU
    ```

3. Perform the Scale Test: Set OMP_NUM_THREADS to values in the range of 1, 2, 4, and 8. For execution run,

    ```bash
    make run
    ```
