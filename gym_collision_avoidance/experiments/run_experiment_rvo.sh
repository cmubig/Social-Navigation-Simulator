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
#python3 src/run_experiment_1.py --output_name "exp1_ETH_RVO" --experiment_num 1 --algorithm_name "RVO" --experiment_iteration 3 --timeout 10 --dataset_name "ETH"

#python3 src/run_experiment_1.py --output_name "exp1_UNIV_RVO" --experiment_num 1 --algorithm_name "RVO" --experiment_iteration 3 --timeout 10 --dataset_name "UNIV"

#python3 src/run_experiment_1.py --output_name "exp2_0.1_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 3 --timeout 10 --population_density 0.1

##############################################################################
python3 src/run_experiment_1.py --output_name "exp1_ETH_RVO" --experiment_num 1 --algorithm_name "RVO" --experiment_iteration 20 --timeout 120 --dataset_name "ETH"

python3 src/run_experiment_1.py --output_name "exp1_HOTEL_RVO" --experiment_num 1 --algorithm_name "RVO" --experiment_iteration 20 --timeout 120 --dataset_name "HOTEL"

python3 src/run_experiment_1.py --output_name "exp1_UNIV_RVO" --experiment_num 1 --algorithm_name "RVO" --experiment_iteration 5  --timeout 120 --dataset_name "UNIV"

python3 src/run_experiment_1.py --output_name "exp1_ZARA1_RVO" --experiment_num 1 --algorithm_name "RVO" --experiment_iteration 20 --timeout 120 --dataset_name "ZARA1"

python3 src/run_experiment_1.py --output_name "exp1_ZARA2_RVO" --experiment_num 1 --algorithm_name "RVO" --experiment_iteration 20 --timeout 120 --dataset_name "ZARA2"

#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_RVO" --experiment_num 2 --algorithm_name "RVO" --experiment_iteration 10 --timeout 60 --population_density 1.0

########################################################################
