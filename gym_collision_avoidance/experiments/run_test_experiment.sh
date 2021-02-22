#!/bin/bash
set -e
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $DIR/utils.sh

# Train tf 
print_header "Running Experiment 1 (Self-play)"

# # Comment for using GPU
 export CUDA_VISIBLE_DEVICES=0
#-1

# Experiment
cd $DIR

#python3 src/run_experiment_1.py --output_name "exp1_ETH_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 10 --dataset_name "ETH"

#python3 src/run_experiment_1.py --output_name "exp1_UNIV_CADRL" --experiment_num 1 --algorithm_name "CADRL" --experiment_iteration 3 --timeout 10 --dataset_name "UNIV"

#python3 src/run_experiment_1.py --output_name "exp1_ZARA2_STGCNN" --experiment_num 1 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 100 --dataset_name "ZARA2"

#python3 src/run_experiment_1.py --output_name "exp1_UNIV_SOCIALFORCE" --experiment_num 1 --algorithm_name "SOCIALFORCE" --experiment_iteration 10 --timeout 60 --dataset_name "UNIV"



#python3 src/run_experiment_1.py --output_name "exp3_0.3_CADRL_SOCIALFORCE" --experiment_num 3 --algorithm_name ["CADRL","SOCIALFORCE"] --experiment_iteration 2 --timeout 40 --population_density 0.3

#python3 src/run_experiment_1.py --output_name "exp1_ETH_STGCNN" --experiment_num 1 --algorithm_name "STGCNN" --experiment_iteration 3 --timeout 10 --dataset_name "ETH"

#python3 src/run_experiment_1.py --output_name "exp2_0.2_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 2 --timeout 10 --population_density 0.2



#python3 src/run_experiment_1.py --output_name "exp2_0.3_CADRL" --experiment_num 2 --algorithm_name "CADRL" --experiment_iteration 2 --timeout 100 --population_density 0.3

#python3 src/run_experiment_1.py --output_name "exp2_0.3_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 2 --timeout 100 --population_density 0.3

#python3 src/run_experiment_1.py --output_name "exp2_0.3_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 2 --timeout 100 --population_density 0.3

#python3 src/run_experiment_1.py --output_name "exp2_0.3_SOCIALFORCE" --experiment_num 2 --algorithm_name "SOCIALFORCE" --experiment_iteration 2 --timeout 100 --population_density 0.3

#python3 src/run_experiment_1.py --output_name "exp2_0.2_LINEAR" --experiment_num 2 --algorithm_name "LINEAR" --experiment_iteration 2 --timeout 100 --population_density 0.2


####MOTION

#python3 src/run_experiment_1.py --output_name "exp2_0.3_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 2 --timeout 50 --population_density 0.3

#python3 src/run_experiment_1.py --output_name "exp2_0.3_SPEC" --experiment_num 2 --algorithm_name "SPEC" --experiment_iteration 2 --timeout 10 --population_density 0.3

#python3 src/run_experiment_1.py --output_name "exp3_0.3_CVM_SLSTM" --experiment_num 3 --algorithm_name ["CVM","SLSTM"] --experiment_iteration 2 --timeout 10 --population_density 0.3


#python3 src/run_experiment_1.py --output_name "exp3_0.3_SOCIALGAN_STGCNN" --experiment_num 4 --algorithm_name ["SOCIALGAN","STGCNN"] --experiment_iteration 3 --timeout 60 --population_density 0.3

#SOCIALGAN STGCNN

python3 src/run_experiment_1.py --output_name "exp2_0.3_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 2 --timeout 20 --population_density 0.3

#TESTING 
#SPEC SLSTM CVM
