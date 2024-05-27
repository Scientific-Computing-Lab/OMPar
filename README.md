# OMPAR

OMPAR is a compiler-oriented tool composed of the following pipeline: it uses (OMPify)[https://github.com/Scientific-Computing-Lab-NRCN/OMPify] to detect parallelization opportunities. When a for loop that would benefit from parallelization is found, (MonoCoder)[https://github.com/Scientific-Computing-Lab-NRCN/MonoCoder] is applied to generate the OpenMP pragma.

## Usage
Here is an example of how to use OMPAR:

```python
code = """for(int i = 0; i <= 1000; i++){
                partial_Sum += i;
            }"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ompar = OMPAR(model_path=main_args.model_weights, device=device, args=main_args)

pragma = ompar.auto_comp(code)
```


**Note:** The weights of OMPify are not included in the repository and will be provided on demand.

