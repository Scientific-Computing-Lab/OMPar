# OMPAR

OMPAR is a compiler-oriented tool designed to identify and generate parallelization opportunities for serial code. It consists of the following pipeline:

  1. [OMPify](https://github.com/Scientific-Computing-Lab-NRCN/OMPify): Detects opportunities for parallelization in code.
  2. [MonoCoder](https://github.com/Scientific-Computing-Lab-NRCN/MonoCoder): Generates the appropriate OpenMP pragmas when a for loop is identified as beneficial for parallelization.
Note: The weights for OMPify are not included in the repository and will be provided upon request.

![OMPAR Workflow](./OMPar.jpg)

*Figure 1: OMPar workflow using a simple pi code example, comparing it with other compilers. Source-to-source automatic compilers (such as AutoPar) generate the necessary pragma, while HPC compilers (such as ICPC) generate a binary parallel output. In contrast, OMPar relies on two LLMs: one for classifying parallelization needs (OMPify) and one for generating the full pragma (MonoCoder-OMP). Both were trained on a large corpus of codes. The evaluation checks if the code compiles, performs with increasing threads, and verifies outputs.*

## Building OMPar
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

To use OMPar, you need to download the Ompify model weights from [here](https://drive.google.com/drive/folders/1tnJf9YvjpDLktVi23TkW-rpjqfdZoybf?usp=sharing).

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
python run_ompar.py --model_weights /path/to/OMPify
```


## OMPar Evaluation 

The following table shows the performance of OMPar on HeCBench test set of 770 loops. 
OMPar accurately predicted the pragma in 74% of the test loop.

| Test setup                                      | TP  | FP  | TN  | FN  | Precision | Recall | Accuracy |
|-------------------------------------------------|-----|-----|-----|-----|-----------|--------|----------|
| OMPar accuracy with ground-truth label          | 311 | 127 | 262 | 70  | 71%       | 81%    | **74%**      |
| AutoPar accuracy with ground-truth label        | 63  | 25  | 365 | 317 | 71%       | 17%    | 56%      |
| ICPC accuracy with ground-truth label           | 95  | 11  | 285 | 379 | 90%       | 25%    | 62%      |
| OMPar accuracy with compile and run check       | 407 | 31  | 262 | 70  | 92%       | 85%    | **86%**      |
| AutoPar accuracy with compile and run check     | 24  | 25  | 365 | 356 | 49%       | 6%     | 50%      |
| ICPC accuracy with compile and run check        | 68  | 5   | 312 | 385 | 93%       | 15%    | 49%      |

The results for OMPar, AutoPar, and ICPC can be reproduced using the information provided in the `evaluation` folder. Specifically, the `data` subfolder contains the code to gather the HeCBench serial codes, while the `icpc`, `autoPar`, and `ompar` subfolders describe the methods for running these codes.
