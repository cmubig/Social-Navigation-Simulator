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
#python3 src/run_experiment_1.py --output_name "exp1_ETH_CVM" --experiment_num 1 --algorithm_name "CVM" --experiment_iteration 3 --timeout 10 --dataset_name "ETH"

#python3 src/run_experiment_1.py --output_name "exp1_UNIV_CVM" --experiment_num 1 --algorithm_name "CVM" --experiment_iteration 3 --timeout 10 --dataset_name "UNIV"

#python3 src/run_experiment_1.py --output_name "exp2_0.1_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 3 --timeout 10 --population_density 0.1

##############################################################################
python3 src/run_experiment_1.py --output_name "exp1_ETH_CVM" --experiment_num 1 --algorithm_name "CVM" --experiment_iteration 20 --timeout 15 --dataset_name "ETH"

python3 src/run_experiment_1.py --output_name "exp1_HOTEL_CVM" --experiment_num 1 --algorithm_name "CVM" --experiment_iteration 20 --timeout 15 --dataset_name "HOTEL"

python3 src/run_experiment_1.py --output_name "exp1_UNIV_CVM" --experiment_num 1 --algorithm_name "CVM" --experiment_iteration 5  --timeout 15 --dataset_name "UNIV"

python3 src/run_experiment_1.py --output_name "exp1_ZARA1_CVM" --experiment_num 1 --algorithm_name "CVM" --experiment_iteration 20 --timeout 15 --dataset_name "ZARA1"

python3 src/run_experiment_1.py --output_name "exp1_ZARA2_CVM" --experiment_num 1 --algorithm_name "CVM" --experiment_iteration 20 --timeout 15 --dataset_name "ZARA2"

#########################################################################
python3 src/run_experiment_1.py --output_name "exp2_0.1_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_CVM" --experiment_num 2 --algorithm_name "CVM" --experiment_iteration 10 --timeout 15 --population_density 1.0

########################################################################
