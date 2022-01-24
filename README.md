# DASHA: Distributed Nonconvex Optimization with Communication Compression, Optimal Oracle Complexity and Without Client Synchronization
This repository provides the code to reproduce the experiments of the submission for The Thirty-ninth International Conference on Machine Learning (ICML 2022)

## Quick Start
### 1. Install [Singularity](https://sylabs.io/guides/3.5/user-guide/introduction.html) (optional)
If you don't want to install Singularity, make sure that you have all dependecies from Singularity.def (python3, numpy, pytorch, etc.)

a. Pull an image 
````
singularity pull library://k3nfalt/default/python_ml:sha256.efcd1fc038228cb7eb0f6f1942dfbaa439cd95d6463015b83ceb2dbaad9e1e98
````
b. Open a shell console of the image
````
singularity shell --nv ~/python_ml_sha256.efcd1fc038228cb7eb0f6f1942dfbaa439cd95d6463015b83ceb2dbaad9e1e98.sif
````
### 2. Prepare scripts for experiments
````
PYTHONPATH=./code python3 ./code/distributed_optimization_library/experiments/zero_marina/config_libsvm_zero_marina.py 
--dumps_path SOME_PATH --dataset_path PATH_TO_FOLDER_WITH_DATASET --dataset mushrooms 
--experiments_name EXPERIMENT_NAME --num_nodes_list 5 
--step_size_range -10 4 --number_of_seeds 1 --number_of_iterations 21000 
--algorithm_names zero_marina marina --function nonconvex  
--compressors rand_k  --number_of_coordinates 10 --quality_check_rate 10
````
### 3. Execute scripts
````
sh SOME_PATH/EXPERIMENT_NAME/singularity_*.sh
````
### 4. Plot results
````
PYTHONPATH=./code python3 ./code/distributed_optimization_library/experiments/plots/zero_marina/plot_marina_mushrooms_gradient.py 
--dumps_paths SOME_PATH/EXPERIMENT_NAME
--output_path SOME_PATH_FOR_PLOTS
````

One can find all other scripts [here](https://github.com/mysteryresearcher/dasha/blob/ac7d0dce798898fb6255e7c0ab181def8ac88f48/code/distributed_optimization_library/experiments/plots/zero_marina/script.txt#L1) that generate experiments from the paper.
