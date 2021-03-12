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


python3 src/run_experiment_1.py --output_name "exp4_0.3_CADRL_SOCIALFORCE" --experiment_num 4 --algorithm_name ["CADRL","SOCIALFORCE"] --experiment_iteration 3 --timeout 30 --population_density 0.3

