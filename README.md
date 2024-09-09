# OMPAR

OMPAR is a compiler-oriented tool designed to identify and generate parallelization opportunities for serial code. It consists of the following pipeline:

  1. [OMPify](https://github.com/Scientific-Computing-Lab-NRCN/OMPify): Detects opportunities for parallelization in code.
  2. [MonoCoder](https://github.com/Scientific-Computing-Lab-NRCN/MonoCoder): Generates the appropriate OpenMP pragmas when a for loop is identified as beneficial for parallelization.
Note: The weights for OMPify are not included in the repository and will be provided upon request.

## Building OMPAR
To build OMPAR, ensure that CUDA 12.1 is supported on your system. Follow these steps:

Clone the repository:
```bash
git clone https://github.com/Scientific-Computing-Lab/OMPar
cd OMPar
```

Create and activate the Conda environment:
```bash
conda create -n ompar_env python=3.11 -f environment.yml
conda activate ompar_env
```

Build the parser:
```bash
cd parser
./build.sh
```

## Usage
Here’s an example of how to use OMPAR:

```python
code = """for(int i = 0; i <= 1000; i++){
                partial_Sum += i;
          }"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ompar = OMPAR(model_path=main_args.model_weights, device=device, args=main_args)

pragma = ompar.auto_comp(code)
```

To run additional use cases, execute the following command:

```bash
python run_ompar.py --model_weights /home/k.tal/OMPify
```
