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
############################################################################## SOCIALGAN EXP 1
python3 src/run_experiment_1.py --output_name "exp1_ETH_SOCIALGAN" --experiment_num 1 --algorithm_name "SOCIALGAN" --experiment_iteration 20 --timeout 15 --dataset_name "ETH"

python3 src/run_experiment_1.py --output_name "exp1_HOTEL_SOCIALGAN" --experiment_num 1 --algorithm_name "SOCIALGAN" --experiment_iteration 20 --timeout 15 --dataset_name "HOTEL"

python3 src/run_experiment_1.py --output_name "exp1_UNIV_SOCIALGAN" --experiment_num 1 --algorithm_name "SOCIALGAN" --experiment_iteration 5  --timeout 15 --dataset_name "UNIV"

python3 src/run_experiment_1.py --output_name "exp1_ZARA1_SOCIALGAN" --experiment_num 1 --algorithm_name "SOCIALGAN" --experiment_iteration 20 --timeout 15 --dataset_name "ZARA1"

python3 src/run_experiment_1.py --output_name "exp1_ZARA2_SOCIALGAN" --experiment_num 1 --algorithm_name "SOCIALGAN" --experiment_iteration 20 --timeout 15 --dataset_name "ZARA2"

#########################################################################  SOCIALGAN EXP 2           
python3 src/run_experiment_1.py --output_name "exp2_0.1_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_SOCIALGAN" --experiment_num 2 --algorithm_name "SOCIALGAN" --experiment_iteration 10 --timeout 15 --population_density 1.0


############################################################################## STGCNN EXP 1
python3 src/run_experiment_1.py --output_name "exp1_ETH_STGCNN" --experiment_num 1 --algorithm_name "STGCNN" --experiment_iteration 20 --timeout 15 --dataset_name "ETH"

python3 src/run_experiment_1.py --output_name "exp1_HOTEL_STGCNN" --experiment_num 1 --algorithm_name "STGCNN" --experiment_iteration 20 --timeout 15 --dataset_name "HOTEL"

python3 src/run_experiment_1.py --output_name "exp1_UNIV_STGCNN" --experiment_num 1 --algorithm_name "STGCNN" --experiment_iteration 5  --timeout 15 --dataset_name "UNIV"

python3 src/run_experiment_1.py --output_name "exp1_ZARA1_STGCNN" --experiment_num 1 --algorithm_name "STGCNN" --experiment_iteration 20 --timeout 15 --dataset_name "ZARA1"

python3 src/run_experiment_1.py --output_name "exp1_ZARA2_STGCNN" --experiment_num 1 --algorithm_name "STGCNN" --experiment_iteration 20 --timeout 15 --dataset_name "ZARA2"

#########################################################################  STGCNN EXP 2           
python3 src/run_experiment_1.py --output_name "exp2_0.1_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.1

python3 src/run_experiment_1.py --output_name "exp2_0.2_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.2

python3 src/run_experiment_1.py --output_name "exp2_0.3_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.3

python3 src/run_experiment_1.py --output_name "exp2_0.4_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.4

python3 src/run_experiment_1.py --output_name "exp2_0.5_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.5

python3 src/run_experiment_1.py --output_name "exp2_0.6_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.6

python3 src/run_experiment_1.py --output_name "exp2_0.7_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.7

python3 src/run_experiment_1.py --output_name "exp2_0.8_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.8

python3 src/run_experiment_1.py --output_name "exp2_0.9_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 0.9

python3 src/run_experiment_1.py --output_name "exp2_1.0_STGCNN" --experiment_num 2 --algorithm_name "STGCNN" --experiment_iteration 10 --timeout 15 --population_density 1.0

