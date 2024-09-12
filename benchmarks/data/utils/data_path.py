import os 
import glob


main_mapping = {'bonds': 'bondsKernelsGpu', 'black': 'blackScholesAnalyticEngineKernelsCpu',
        'crc64': 'CRC64', 'gaussian': 'gaussianElim', 'hotspot3D': '3D', 'hmm': 'HiddenMarkovModel',
        'leukocyte': 'detect_main', 'nn': 'nearestNeighbor', 'multimaterial': 'multimat', 
        'su3': 'su3_nn_bench', 'mt': 'MT'}

def find_main(kernel_path, kernel):
    files = glob.glob(os.path.join(kernel_path, 'main.c*'), recursive=False)
    
    if not files:
        files = glob.glob(os.path.join(kernel_path, '**', 'main.c*'), recursive=True)
    if not files:
        kernel_name = kernel.split('-')[0]

        if kernel_name in main_mapping:
            files = glob.glob(os.path.join(kernel_path, f'{main_mapping[kernel_name]}.*'), recursive=True)
        else:
            files = glob.glob(os.path.join(kernel_path, '**', f'{kernel_name}.*'), recursive=True)
    if not files:
        files = glob.glob(os.path.join(kernel_path, '*.cpp'), recursive=True) + \
                    glob.glob(os.path.join(kernel_path, '*.cu'), recursive=True)

    return files


def get_kernels(dataset_path, detailed=False):
    kernels = []
    
    for kernel in os.listdir(dataset_path):
        kernel_path = os.path.join(dataset_path, kernel)
        files = find_main(kernel_path, kernel)
        
        api = kernel.split('-')[-1]
        if len(files) > 1:
            if api in ['hip', 'cuda']:
                files = [path for path in files if path.endswith('.cu')]
            if api in ['omp', 'sycl']:
                files = [path for path in files if not path.endswith('.h') and not path.endswith('.hpp')]
        
        if len(files) > 1:
            files = [path for path in files if 'src' in path[len(kernel_path):].split('/')]

        if len(files) == 1:

            if not detailed:
                kernels.append(files[0])
            else:
                file_path = files[0]
                kernel_name, api = kernel.rsplit('-', 1)

                kernels.append({'kernel_name': kernel_name,
                                'parallel_api': api,
                                'path': file_path
                                })

        else:
            pass
            # print(f'Error {kernel}')

    return kernels


    
           
