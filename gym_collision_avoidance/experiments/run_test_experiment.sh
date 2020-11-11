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


python3 src/run_experiment_1.py --output_name "exp3_0.6_SOCIALFORCE_RVO" --experiment_num 3 --algorithm_name ["SOCIALFORCE","RVO"] --experiment_iteration 10 --timeout 60 --population_density 0.6


#python3 src/run_experiment_1.py --output_name "exp3_0.2_LINEAR_LINEAR" --experiment_num 4 --algorithm_name ["LINEAR","LINEAR"] --experiment_iteration 2 --timeout 15 --population_density 0.2



#python3 src/run_experiment_1.py --output_name "exp3_0.2_RVO_LINEAR" --experiment_num 3 --algorithm_name ["RVO","LINEAR"] --experiment_iteration 3 --timeout 15 --population_density 0.2
