import json
import os 
import glob
import random
from utils.remove_pragma import remove_omp
from utils.data_path import get_kernels



def load_code(file_path):
    with open(file_path, 'r', errors='ignore') as file:
        return file.read()


def extract_samples(kernels):
    kernel_names = list(set([kernel['kernel_name'] for kernel in kernels]))
    samples = []

    for kernel in kernels:
        code = load_code(kernel['path'])
          
        if kernel['parallel_api'] == 'omp':
            samples.append({'kernel_name': kernel['kernel_name'], 'parallel_api': 'serial', 'code': remove_omp(code)})

    return samples


if __name__=='__main__':
    dataset_path = '/path/to/HeCBench/src'
    kernels = get_kernels(dataset_path, detailed=True)
    samples = extract_codes(kernels)

    for sample in samples:
        with open('dataset,jsonl', 'w') as f:
            f.write(json.dumps(kernel) + '\n')
