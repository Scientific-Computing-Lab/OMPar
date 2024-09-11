# Automatic Parallelization using Intel Compiler

Intel Compiler (ICC) 19.1.3.304 20200925 is used for performing automatic OpenMP parallelizaiton.

## Compile and Run Test

1. As an initial step, run the `PragmaRemover` as outlined in the `autoPar/autoPar_README` file to eliminate the parallel directives from the baseline HeCBench codes inorder make it sequential verison.

2. Update the `Makefile` in each HeCBench code with the following compiler options:

    - Add `-qopenmp -parallel` to enable automatic parallelization with OpenMP.
    - Include `-qopt-report:3` to generate a detailed parallelization report, which lists both the lines that were auto-parallelized and those that were not, along with reasons. The parallelization report files are represented with an extension `.optrpt`.

    ```bash
    CC = icpc
    CFLAGS +=-qopenmp -parallel -qopt-report:3
    ```

3. To compile and run each codes use,

    ```bash
    make DEVICE=CPU
    make run
    ```

Note that the scale test were performed by setting the OMP_NUM_THREADS=1,2,4 and 8.
