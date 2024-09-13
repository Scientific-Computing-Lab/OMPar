# Automatic Parallelization using OMPar

For any HeCBench loop, we used OMPar to predict the need for parallelization, and if required, it generates the pragma in the following way:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ompar = OMPAR(model_path=main_args.model_weights, device=device, args=main_args)

pragma = ompar.auto_comp(for_loop)
```
