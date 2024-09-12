# Dataset Creation - HeCBench

In order to generate serialized OpenMP data with corresponding labels, we utilized HeCBench, a repository containing a wide variety of numerical code benchmarks that implement multiple parallel schemes, including OpenMP.

 ## Process Overview
  1. Removing OpenMP Pragmas:
We used the remove_pragma.py script to iterate through the benchmark codes, detect OpenMP pragmas, and remove them.
The resulting output, with the OpenMP pragmas removed, serves as the input for different methods and evaluations.

  2. Dataset Creation:
To create the dataset for HeCBench evaluation, follow the steps below.


### Steps to Reproduce

Clone the HeCBench repository and run the dataset creation script:

```bash
git clone https://github.com/zjin-lcf/HeCBench
python create_dataset.py
```

This will create a dataset that can be used for evaluating different methods and models on the HeCBench benchmarks.
